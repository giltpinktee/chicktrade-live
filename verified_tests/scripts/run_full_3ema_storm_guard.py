"""Full 3-EMA (34/89/200) x Touches 1-2-3-4 Backtest with SL_Leg2 = Entry - 2U & Smart Storm/Whipsaw Circuit Breaker."""
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import BacktestConfig, RiskConfig, StrategyConfig
from data.binance_loader import _detect_latest_available_month, download_symbol_1y
from report.html_report import export_html_report
from risk.compounding import CompoundingManager
from run_be_whipsaw_test import sim_leg2_policy
from run_research_mtf import compute_adx, resample_ohlcv
from strategy.ema_wave import Signal
from strategy.indicators import add_indicators


def compute_storm_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 2-layer market storm / two-way whipsaw detector (zero lookahead).

    Triggers cooldown when:
      1. Volatility Spike: ATR(14) >= 1.8 * ATR_MA(50) OR candle range >= 2.8 * ATR
      2. Two-Way Whipsaw Sweep: Kaufman Efficiency Ratio (10 bars) < 0.15 while ATR > 1.2 * ATR_MA(50)
         (price whipsawing violently up & down with little net directional progress)
    """
    out = pd.DataFrame(index=df.index)
    high = df["high"]
    low = df["low"]
    close = df["close"]
    open_ = df["open"]

    tr = pd.concat(
        [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1.0 / 14, adjust=False).mean()
    atr_ma50 = atr.rolling(50, min_periods=10).mean()
    atr_ratio = (atr / atr_ma50.replace(0, np.nan)).fillna(1.0)

    # Big two-sided wick sweep candle
    crange = (high - low).replace(0, np.nan)
    body = (close - open_).abs()
    wick_frac = (1.0 - body / crange).fillna(0.0)
    sweep_candle = (crange >= 2.8 * atr) | ((crange >= 2.0 * atr) & (wick_frac >= 0.70))

    # Kaufman Efficiency Ratio (10 bars)
    net_move = (close - close.shift(10)).abs()
    path_len = close.diff().abs().rolling(10, min_periods=10).sum().replace(0, np.nan)
    er = (net_move / path_len).fillna(0.5)

    raw_storm = (atr_ratio >= 1.8) | sweep_candle | ((er < 0.15) & (atr_ratio >= 1.20))

    # Cooldown window: 16 bars on M15 (4h) or 4 bars on H1 (4h) — handled dynamically via rolling max
    out["atr_ratio"] = atr_ratio
    out["er"] = er
    out["raw_storm"] = raw_storm
    return out


def build_cooldown_mask(raw_storm: pd.Series, cooldown_bars: int) -> np.ndarray:
    """If any storm event occurred in the last `cooldown_bars` bars, halt new entries."""
    return (raw_storm.astype(int).rolling(cooldown_bars, min_periods=1).max() > 0).values


def generate_all_4_touches(
    df: pd.DataFrame,
    df_htf: pd.DataFrame,
    btc_storm_ts: np.ndarray,
    btc_storm_mask: np.ndarray,
    symbol: str,
    cfg: StrategyConfig,
    cooldown_bars: int,
    min_adx: float = 20.0,
) -> Tuple[List[Signal], int]:
    """Generate Touches 1, 2, 3, 4 for EMA 34, EMA 89, EMA 200 with HTF+ADX and Storm Guard."""
    data = add_indicators(df, cfg)
    adx = compute_adx(df, 14).values
    storm_df = compute_storm_flags(df)
    local_storm = build_cooldown_mask(storm_df["raw_storm"], cooldown_bars)
    atr_ratio_arr = storm_df["atr_ratio"].values

    htf = df_htf.copy()
    htf["htf_ema34"] = htf["close"].ewm(span=34, adjust=False).mean()
    htf["htf_ema200"] = htf["close"].ewm(span=200, adjust=False).mean()
    htf["htf_trend"] = np.where(
        (htf["close"] > htf["htf_ema200"]) & (htf["htf_ema34"] > htf["htf_ema200"]),
        1,
        np.where((htf["close"] < htf["htf_ema200"]) & (htf["htf_ema34"] < htf["htf_ema200"]), -1, 0),
    )
    htf["htf_trend_closed"] = htf["htf_trend"].shift(1).fillna(0).astype(int)
    htf_ts = htf["timestamp"].values
    htf_tr = htf["htf_trend_closed"].values

    closes = data["close"].values
    opens = data["open"].values
    highs = data["high"].values
    lows = data["low"].values
    ema34 = data["ema34"].values
    ema89 = data["ema89"].values
    ema200 = data["ema200"].values
    atr = data["atr"].values
    vol_ratio = data["vol_ratio"].values
    lower_wick = data["lower_wick_ratio"].values
    upper_wick = data["upper_wick_ratio"].values
    ts_arr = data["timestamp"].values
    n = len(data)

    signals: List[Signal] = []
    storm_blocked = 0

    ema_specs = [
        (34, ema34, 0.8, 3),
        (89, ema89, 1.0, 5),
        (200, ema200, 1.2, 8),
    ]

    for ema_p, et_arr, imp_mult, min_gap in ema_specs:
        trend = 0
        wave_count = 0
        armed = False
        last_touch = -999

        for i in range(205, n):
            c, o, h, l, a = closes[i], opens[i], highs[i], lows[i], atr[i]
            e34, e89, e200, et = ema34[i], ema89[i], ema200[i], et_arr[i]
            if np.isnan(a) or a <= 0 or np.isnan(e200):
                continue

            if ema_p == 34:
                new_trend = 1 if (c > e200 and e34 > e89 > e200) else (-1 if (c < e200 and e34 < e89 < e200) else 0)
            elif ema_p == 89:
                new_trend = 1 if (c > e200 and e89 > e200) else (-1 if (c < e200 and e89 < e200) else 0)
            else:
                new_trend = 1 if (e89 > e200 and c >= e200 - 0.8 * a) else (-1 if (e89 < e200 and c <= e200 + 0.8 * a) else 0)

            if new_trend != trend:
                trend = new_trend
                wave_count = 0
                armed = False
                last_touch = -999
                continue

            if trend == 0:
                continue

            if trend == 1:
                if (h - et) >= imp_mult * a and l > et:
                    armed = True
            else:
                if (et - l) >= imp_mult * a and h < et:
                    armed = True

            if not armed or (i - last_touch) < min_gap:
                continue

            touch_buf = 0.20 * a
            breach_buf = 0.65 * a
            if trend == 1:
                touched = (l <= et + touch_buf) and (c >= et - breach_buf)
                pa_ok = (lower_wick[i] >= cfg.min_wick_ratio) or (c > o and c > et)
            else:
                touched = (h >= et - touch_buf) and (c <= et + breach_buf)
                pa_ok = (upper_wick[i] >= cfg.min_wick_ratio) or (c < o and c < et)

            if not touched:
                continue

            wave_count += 1
            armed = False
            last_touch = i
            if wave_count > 4:
                continue

            # ADX & HTF trend filter
            if adx[i] < min_adx:
                continue
            idx = int(np.searchsorted(htf_ts, ts_arr[i], side="right")) - 1
            if idx < 0:
                continue
            ht = int(htf_tr[idx])

            # Waves 1-3 follow trend (aligned with HTF); Wave 4 is counter-trend reversal
            if wave_count <= 3:
                if ht != trend:
                    continue
                direction = trend
                stype = f"EMA{ema_p}_W{wave_count}"
            else:
                direction = -trend
                stype = f"EMA{ema_p}_W4_REV"

            # Smart Storm Guard check (local coin storm OR BTC macro storm)
            b_idx = int(np.searchsorted(btc_storm_ts, ts_arr[i], side="right")) - 1
            btc_in_storm = bool(btc_storm_mask[b_idx]) if b_idx >= 0 else False
            if local_storm[i] or btc_in_storm:
                storm_blocked += 1
                continue

            ts = pd.Timestamp(ts_arr[i])
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")

            vol_ok = (not np.isnan(vol_ratio[i])) and (vol_ratio[i] >= cfg.min_vol_ratio)
            # Store high-vol hint in pa_vol_ok? Better: keep pa_vol_ok clean, use atr_ratio separately
            sig = Signal(
                bar_idx=i,
                timestamp=ts,
                symbol=symbol,
                direction=direction,
                wave=wave_count,
                setup_type=stype,
                entry_price=float(c),
                atr=float(a),
                pa_vol_ok=bool(atr_ratio_arr[i] >= 1.35),  # repurpose flag to mark elevated vol (1.35-1.8x)
                ema_aligned=bool(pa_ok and vol_ok),
            )
            signals.append(sig)

    signals.sort(key=lambda s: s.bar_idx)
    return signals, storm_blocked


def evaluate_all_setups(
    dfs: Dict[str, pd.DataFrame],
    sigs_map: Dict[str, List[Signal]],
    tp1_u: float,
    tp2_u: float = 10.0,
) -> pd.DataFrame:
    rows = []
    for sym, sigs in sigs_map.items():
        df = dfs[sym]
        for s in sigs:
            r = sim_leg2_policy(df, s, 5.0, tp1_u, tp2_u, "SL_MINUS_2U")
            rows.append({"symbol": sym, "setup": s.setup_type, "wave": s.wave, **r})

    rdf = pd.DataFrame(rows)
    if rdf.empty:
        return pd.DataFrame()

    out = []
    for stype, g in rdf.groupby("setup"):
        n = len(g)
        wr = (g["net_u"] > 0).mean() * 100.0
        tp1 = g["tp1_hit"].mean() * 100.0
        tp2 = (g["outcome"] == "TP1+TP2").mean() * 100.0
        sl_f = (g["outcome"] == "SL_FULL").mean() * 100.0
        pos = g.loc[g["net_u"] > 0, "net_u"].sum()
        neg = -g.loc[g["net_u"] < 0, "net_u"].sum()
        pf = pos / neg if neg > 0 else (9.99 if pos > 0 else 0.0)
        out.append(
            {
                "Setup": stype,
                "Trades": n,
                "Winrate_%": wr,
                "TP1_%": tp1,
                "TP2_%": tp2,
                "SL_Full_%": sl_f,
                "PF": pf,
                "Avg_Net_U": g["net_u"].mean(),
                "Total_Net_U": g["net_u"].sum(),
            }
        )
    return pd.DataFrame(out).sort_values("PF", ascending=False)


def run_portfolio_with_storm_guard(
    dfs: Dict[str, pd.DataFrame],
    sigs_map: Dict[str, List[Signal]],
    setup_tier_map: Dict[str, float],   # setup_type -> base size multiplier (e.g. 1.0, 0.4, 0.25)
    tp1_u: float,
    tp2_u: float = 10.0,
    start_time: pd.Timestamp = None,
) -> dict:
    rcfg = RiskConfig(initial_capital=10_000.0, buffer_units=500.0, max_concurrent_trades=2, max_trades_per_day=10)
    comp = CompoundingManager(rcfg)
    equity = rcfg.initial_capital
    peak_eq = equity

    all_sigs = []
    for sym, sigs in sigs_map.items():
        for s in sigs:
            if s.setup_type not in setup_tier_map:
                continue
            if start_time is not None and s.timestamp < start_time:
                continue
            all_sigs.append(s)
    all_sigs.sort(key=lambda s: (s.timestamp, -setup_tier_map[s.setup_type], s.symbol))

    active = []
    closed = []
    daily_trades: Dict[str, int] = {}
    daily_sl: Dict[str, int] = {}
    daily_pnl: Dict[str, float] = {}
    daily_start_eq: Dict[str, float] = {}
    consec_sl = 0
    eq_curve = [{"timestamp": all_sigs[0].timestamp if all_sigs else pd.Timestamp.utcnow(), "equity": equity, "step": 1}]

    def settle(now_ts):
        nonlocal equity, peak_eq, consec_sl
        keep, done = [], []
        for ex_ts, tr in active:
            (done if ex_ts <= now_ts else keep).append((ex_ts, tr))
        done.sort(key=lambda x: x[0])
        for ex_ts, tr in done:
            equity += tr["net_pnl"]
            peak_eq = max(peak_eq, equity)
            comp.update_equity(equity, ex_ts)
            dk = tr["day_key"]
            daily_pnl[dk] = daily_pnl.get(dk, 0.0) + tr["net_pnl"]
            if tr["outcome"] == "SL_FULL":
                consec_sl += 1
                daily_sl[dk] = daily_sl.get(dk, 0) + 1
            elif tr["net_pnl"] > 0:
                consec_sl = 0
            tr["equity_after"] = equity
            closed.append(tr)
            eq_curve.append({"timestamp": ex_ts, "equity": equity, "step": comp.step})
        active[:] = keep

    for sig in all_sigs:
        settle(sig.timestamp)
        if equity <= 0:
            break

        dk = sig.timestamp.strftime("%Y-%m-%d")
        if dk not in daily_start_eq:
            daily_start_eq[dk] = equity

        # Daily limits: max 10 trades/day, max 3 full SLs/day, or daily loss <= -2.5%
        if daily_trades.get(dk, 0) >= rcfg.max_trades_per_day:
            continue
        if daily_sl.get(dk, 0) >= 3:
            continue
        if daily_pnl.get(dk, 0.0) <= -0.025 * daily_start_eq[dk]:
            continue
        if len(active) >= rcfg.max_concurrent_trades:
            continue
        if any(tr["symbol"] == sig.symbol for _, tr in active):
            continue

        base_mult = setup_tier_map[sig.setup_type]
        eff_mult = base_mult
        # Elevated volatility (1.35x - 1.8x ATR) -> cut size in half
        if sig.pa_vol_ok:
            eff_mult *= 0.5
        # Drawdown > 3% or 2 consecutive SLs -> drop to Micro Size 0.25x
        dd_pct = (equity - peak_eq) / peak_eq * 100.0
        if dd_pct <= -3.0 or consec_sl >= 2:
            eff_mult = min(eff_mult, 0.25)
        if any(tr["size_mult"] >= 0.75 for _, tr in active):
            eff_mult = min(eff_mult, 0.35)

        df = dfs[sig.symbol]
        sim = sim_leg2_policy(df, sig, 5.0, tp1_u, tp2_u, "SL_MINUS_2U")

        unit_size = max(sig.atr * 0.5, sig.entry_price * 1e-5)
        sizing_base = min(comp.tier_capital, equity)
        qty = (sizing_base * eff_mult) / (rcfg.buffer_units * unit_size)
        net_pnl = sim["net_u"] * unit_size * qty

        bars_est = 4 if sim["outcome"] == "SL_FULL" else 8
        exit_idx = min(sig.bar_idx + bars_est, len(df) - 1)
        ex_ts = pd.Timestamp(df["timestamp"].iloc[exit_idx])
        if ex_ts.tzinfo is None:
            ex_ts = ex_ts.tz_localize("UTC")

        tr = {
            "symbol": sig.symbol,
            "setup": sig.setup_type,
            "size_mult": eff_mult,
            "outcome": sim["outcome"],
            "net_u": sim["net_u"],
            "net_pnl": net_pnl,
            "day_key": dk,
            "entry_time": sig.timestamp,
            "exit_time": ex_ts,
        }
        active.append((ex_ts, tr))
        daily_trades[dk] = daily_trades.get(dk, 0) + 1

    if active:
        settle(pd.Timestamp("2099-01-01", tz="UTC"))

    eq_df = pd.DataFrame(eq_curve)
    pnls = np.array([t["net_pnl"] for t in closed]) if closed else np.array([0.0])
    wr = (pnls > 0).mean() * 100.0 if closed else 0.0
    sl_f = sum(1 for t in closed if t["outcome"] == "SL_FULL") / len(closed) * 100.0 if closed else 0.0
    pos = pnls[pnls > 0].sum()
    neg = -pnls[pnls < 0].sum()
    pf = float(pos / neg) if neg > 0 else 0.0
    ev = eq_df["equity"].values
    pk = np.maximum.accumulate(ev)
    max_dd = float(((ev - pk) / pk * 100.0).min())

    return {
        "trades": closed,
        "eq_df": eq_df,
        "final_equity": float(ev[-1]),
        "return_pct": (float(ev[-1]) / rcfg.initial_capital - 1.0) * 100.0,
        "max_step": comp.max_step_reached,
        "total_trades": len(closed),
        "winrate_pct": wr,
        "sl_full_pct": sl_f,
        "pf": pf,
        "max_dd_pct": max_dd,
    }


def main():
    cfg = BacktestConfig()
    end_ym = _detect_latest_available_month("BTCUSDT", "5m")

    raw_m5 = {sym: download_symbol_1y(sym, interval="5m", months=12, market="futures", end_ym=end_ym) for sym in cfg.symbols}
    dfs_m15 = {sym: resample_ohlcv(df, "15min") for sym, df in raw_m5.items()}
    dfs_h1 = {sym: resample_ohlcv(df, "1h") for sym, df in raw_m5.items()}
    dfs_h4 = {sym: resample_ohlcv(df, "4h") for sym, df in raw_m5.items()}

    # Build BTC macro storm mask for M15 (cooldown 16 bars = 4h) and H1 (cooldown 4 bars = 4h)
    btc_m15_storm = compute_storm_flags(dfs_m15["BTCUSDT"])
    btc_m15_mask = build_cooldown_mask(btc_m15_storm["raw_storm"], 16)
    btc_m15_ts = dfs_m15["BTCUSDT"]["timestamp"].values

    btc_h1_storm = compute_storm_flags(dfs_h1["BTCUSDT"])
    btc_h1_mask = build_cooldown_mask(btc_h1_storm["raw_storm"], 4)
    btc_h1_ts = dfs_h1["BTCUSDT"]["timestamp"].values

    print("=" * 108)
    print("CHICKTRADE — 3-EMA (34/89/200) x CHAM 1-2-3-4 | SL CHAN 2 = ENTRY-2U | SMART STORM GUARD | BAO 500 GIA")
    print("=" * 108)

    sigs_m15, blocked_m15 = {}, 0
    for sym in cfg.symbols:
        s, b = generate_all_4_touches(dfs_m15[sym], dfs_h1[sym], btc_m15_ts, btc_m15_mask, sym, cfg.strategy, cooldown_bars=16)
        sigs_m15[sym] = s
        blocked_m15 += b

    sigs_h1, blocked_h1 = {}, 0
    for sym in cfg.symbols:
        s, b = generate_all_4_touches(dfs_h1[sym], dfs_h4[sym], btc_h1_ts, btc_h1_mask, sym, cfg.strategy, cooldown_bars=4)
        sigs_h1[sym] = s
        blocked_h1 += b

    print(f"[Storm Guard] Blocked {blocked_m15} noisy/whipsaw M15 signals & {blocked_h1} H1 signals during market storms.")

    tbl_m15 = evaluate_all_setups(dfs_m15, sigs_m15, tp1_u=4.0, tp2_u=10.0)
    print("\n--- BANG 1: M15 — TOAN BO CHAM 1-2-3-4 CUA EMA 34 / 89 / 200 (TP1=4U + SL_Chan2=Entry-2U + TP2=10U) ---")
    print(tbl_m15.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    tbl_h1 = evaluate_all_setups(dfs_h1, sigs_h1, tp1_u=3.0, tp2_u=10.0)
    print("\n--- BANG 2: H1 — TOAN BO CHAM 1-2-3-4 CUA EMA 34 / 89 / 200 (TP1=3U + SL_Chan2=Entry-2U + TP2=10U) ---")
    print(tbl_h1.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    # Stage 2 Portfolio Runs (Max 2 open, Max 10/day, Bao 500 gia, Storm Guard, Daily -2.5% / 3-SL stop)
    print("\n--- BANG 3: MO PHONG PORTFOLIO 3 COIN (BTC+ETH+BNB | MAX 2 LENH | MAX 10/NGAY | BAO 500 GIA | STORM GUARD) ---")

    # 1) All 12 setups on M15 (to show why filtering matters)
    all_m15_map = {row["Setup"]: (1.0 if row["Winrate_%"] > 70 else 0.35) for _, row in tbl_m15.iterrows() if row["Winrate_%"] >= 51}
    p_m15_all = run_portfolio_with_storm_guard(dfs_m15, sigs_m15, all_m15_map, tp1_u=4.0, tp2_u=10.0)

    # 2) Positive-PF Setups Only on M15 (WR >= 51% AND PF >= 1.05)
    good_m15_map = {
        row["Setup"]: (1.0 if row["Winrate_%"] > 70 else (0.50 if row["PF"] >= 1.15 else 0.30))
        for _, row in tbl_m15.iterrows()
        if row["Winrate_%"] >= 51 and row["PF"] >= 1.05
    }
    p_m15_good = run_portfolio_with_storm_guard(dfs_m15, sigs_m15, good_m15_map, tp1_u=4.0, tp2_u=10.0)

    # 3) Positive-PF Setups Only on H1 (WR >= 51% AND PF >= 1.05)
    good_h1_map = {
        row["Setup"]: (1.0 if row["Winrate_%"] > 70 else (0.50 if row["PF"] >= 1.25 else 0.30))
        for _, row in tbl_h1.iterrows()
        if row["Winrate_%"] >= 51 and row["PF"] >= 1.05
    }
    p_h1_good = run_portfolio_with_storm_guard(dfs_h1, sigs_h1, good_h1_map, tp1_u=3.0, tp2_u=10.0)

    for name, smap, res in [
        ("M15 — Tat ca setup WR >= 51%", all_m15_map, p_m15_all),
        ("M15 — Chi setup CHUAN (WR>=51% & PF>=1.05)", good_m15_map, p_m15_good),
        ("H1  — Chi setup CHUAN (WR>=51% & PF>=1.05)", good_h1_map, p_h1_good),
    ]:
        print(
            f"  [{name}]\n"
            f"     Setups: {smap}\n"
            f"     Equity=${res['final_equity']:9,.2f} ({res['return_pct']:+6.2f}%) | Step {res['max_step']:2d}/20 | "
            f"Trades={res['total_trades']:3d} | WR={res['winrate_pct']:5.2f}% | SL_Full={res['sl_full_pct']:5.2f}% | "
            f"PF={res['pf']:.2f} | MaxDD={res['max_dd_pct']:6.2f}%"
        )

    # Save CSVs & HTML report
    out_dir = os.path.join(ROOT, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    tbl_m15.to_csv(os.path.join(out_dir, "full_3ema_4touches_m15_entry_minus_2u.csv"), index=False)
    tbl_h1.to_csv(os.path.join(out_dir, "full_3ema_4touches_h1_entry_minus_2u.csv"), index=False)


if __name__ == "__main__":
    main()
