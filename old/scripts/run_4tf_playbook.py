"""4-Timeframe (M5, M15, M30, H1) Volatility Anatomy + 3-EMA (34/89/200) 4-Touch Playbook with Storm Guard & Entry-2U."""
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import BacktestConfig, RiskConfig
from data.binance_loader import _detect_latest_available_month, download_symbol_1y
from report.html_report import export_html_report
from risk.compounding import CompoundingManager
from run_be_whipsaw_test import sim_leg2_policy
from run_full_3ema_storm_guard import build_cooldown_mask, compute_storm_flags, generate_all_4_touches
from run_research_mtf import resample_ohlcv
from strategy.ema_wave import Signal


def measure_tf_anatomy(dfs: Dict[str, pd.DataFrame], sigs_map: Dict[str, List[Signal]], tf_name: str) -> dict:
    """Measure average 1 Unit (%), candle range, fee in units, and 12-bar MAE/MFE before SL5."""
    unit_pcts, fee_us, maes, mfes, mfe_b5 = [], [], [], [], []
    btc_u, eth_u, bnb_u = [], [], []

    for sym, sigs in sigs_map.items():
        df = dfs[sym]
        highs = df["high"].values
        lows = df["low"].values
        n = len(df)
        for s in sigs:
            u = max(s.atr * 0.5, s.entry_price * 1e-5)
            upct = u / s.entry_price * 100.0
            unit_pcts.append(upct)
            fee_us.append((s.entry_price * 0.0007) / u)
            if sym == "BTCUSDT":
                btc_u.append(u)
            elif sym == "ETHUSDT":
                eth_u.append(u)
            elif sym == "BNBUSDT":
                bnb_u.append(u)

            end = min(s.bar_idx + 13, n)
            if end <= s.bar_idx + 1:
                continue
            wh = highs[s.bar_idx + 1 : end]
            wl = lows[s.bar_idx + 1 : end]
            d = s.direction
            fav = (wh - s.entry_price) / u if d == 1 else (s.entry_price - wl) / u
            adv = (s.entry_price - wl) / u if d == 1 else (wh - s.entry_price) / u
            mfes.append(float(max(0.0, fav.max())))
            maes.append(float(max(0.0, adv.max())))

            best = 0.0
            for k in range(len(wh)):
                if adv[k] >= 5.0:
                    break
                if fav[k] > best:
                    best = float(fav[k])
            mfe_b5.append(best)

    return {
        "Khung": tf_name,
        "Tin_Hieu_1Y": len(unit_pcts),
        "1U_BTC_$": float(np.mean(btc_u)) if btc_u else 0.0,
        "1U_ETH_$": float(np.mean(eth_u)) if eth_u else 0.0,
        "1U_BNB_$": float(np.mean(bnb_u)) if bnb_u else 0.0,
        "1U_TB_%": float(np.mean(unit_pcts)) if unit_pcts else 0.0,
        "Phi_TB_U": float(np.mean(fee_us)) if fee_us else 0.0,
        "Giat_Nguoc_MAE_TB": float(np.mean(maes)) if maes else 0.0,
        "Bat_Cao_MFE_TB": float(np.mean(mfes)) if mfes else 0.0,
        "TP_TB_TruocSL5": float(np.mean(mfe_b5)) if mfe_b5 else 0.0,
        "TP_Med_TruocSL5": float(np.median(mfe_b5)) if mfe_b5 else 0.0,
        "TP_P75_TruocSL5": float(np.quantile(mfe_b5, 0.75)) if mfe_b5 else 0.0,
    }


def eval_tf_setups_with_oos(
    dfs: Dict[str, pd.DataFrame],
    sigs_map: Dict[str, List[Signal]],
    tf_name: str,
    tp1_u: float,
    tp2_u: float = 10.0,
) -> pd.DataFrame:
    rows = []
    for sym, sigs in sigs_map.items():
        df = dfs[sym]
        for s in sigs:
            r = sim_leg2_policy(df, s, 5.0, tp1_u, tp2_u, "SL_MINUS_2U")
            rows.append({"tf": tf_name, "symbol": sym, "setup": s.setup_type, "timestamp": s.timestamp, **r})

    rdf = pd.DataFrame(rows)
    if rdf.empty:
        return pd.DataFrame()

    mid_ts = rdf["timestamp"].median()
    out = []
    for stype, g in rdf.groupby("setup"):
        g_is = g[g["timestamp"] < mid_ts]
        g_oos = g[g["timestamp"] >= mid_ts]

        def _pf(sub):
            if len(sub) == 0:
                return 0.0
            pos = sub.loc[sub["net_u"] > 0, "net_u"].sum()
            neg = -sub.loc[sub["net_u"] < 0, "net_u"].sum()
            return float(pos / neg) if neg > 0 else (9.99 if pos > 0 else 0.0)

        out.append(
            {
                "TF": tf_name,
                "Setup": stype,
                "Trades": len(g),
                "WR_%": (g["net_u"] > 0).mean() * 100.0,
                "TP2_%": (g["outcome"] == "TP1+TP2").mean() * 100.0,
                "SL_Full_%": (g["outcome"] == "SL_FULL").mean() * 100.0,
                "PF_1Y": _pf(g),
                "PF_IS6M": _pf(g_is),
                "PF_OOS6M": _pf(g_oos),
                "Net_U_1Y": g["net_u"].sum(),
            }
        )
    return pd.DataFrame(out).sort_values("PF_1Y", ascending=False)


def run_unified_4tf_portfolio(
    tf_dfs_map: Dict[str, Dict[str, pd.DataFrame]],
    tf_sigs_map: Dict[str, Dict[str, List[Signal]]],
    allowed_rules: Dict[tuple, tuple],  # (tf, setup) -> (size_mult, tp1_u)
    tp2_u: float = 10.0,
    start_time: pd.Timestamp = None,
) -> dict:
    rcfg = RiskConfig(initial_capital=10_000.0, buffer_units=500.0, max_concurrent_trades=2, max_trades_per_day=10)
    comp = CompoundingManager(rcfg)
    equity = rcfg.initial_capital
    peak_eq = equity

    candidates = []
    for tf_name, sigs_map in tf_sigs_map.items():
        for sym, sigs in sigs_map.items():
            for s in sigs:
                key = (tf_name, s.setup_type)
                if key not in allowed_rules:
                    continue
                if start_time is not None and s.timestamp < start_time:
                    continue
                mult, tp1_u = allowed_rules[key]
                candidates.append((tf_name, mult, tp1_u, s))

    # Higher TF & higher tier prioritized on identical timestamps
    tf_prio = {"H1": 0, "M30": 1, "M15": 2, "M5": 3}
    candidates.sort(key=lambda x: (x[3].timestamp, tf_prio[x[0]], -x[1], x[3].symbol))

    active = []
    closed = []
    daily_trades: Dict[str, int] = {}
    daily_sl: Dict[str, int] = {}
    daily_pnl: Dict[str, float] = {}
    daily_start_eq: Dict[str, float] = {}
    consec_sl = 0
    eq_curve = [{"timestamp": candidates[0][3].timestamp if candidates else pd.Timestamp.utcnow(), "equity": equity, "step": 1}]

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

    for tf_name, base_mult, tp1_u, sig in candidates:
        settle(sig.timestamp)
        if equity <= 0:
            break
        dk = sig.timestamp.strftime("%Y-%m-%d")
        if dk not in daily_start_eq:
            daily_start_eq[dk] = equity

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

        eff_mult = base_mult
        if sig.pa_vol_ok:  # elevated ATR (1.35-1.8x)
            eff_mult *= 0.5
        dd_pct = (equity - peak_eq) / peak_eq * 100.0
        if dd_pct <= -3.0 or consec_sl >= 2:
            eff_mult = min(eff_mult, 0.25)
        if any(tr["size_mult"] >= 0.75 for _, tr in active):
            eff_mult = min(eff_mult, 0.35)

        df = tf_dfs_map[tf_name][sig.symbol]
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
            "tf": tf_name,
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
    dfs_m30 = {sym: resample_ohlcv(df, "30min") for sym, df in raw_m5.items()}
    dfs_h1 = {sym: resample_ohlcv(df, "1h") for sym, df in raw_m5.items()}
    dfs_h4 = {sym: resample_ohlcv(df, "4h") for sym, df in raw_m5.items()}

    # Build configs for all 4 TFs: (tf_name, tf_dfs, htf_dfs, cooldown_4h_bars, tp1_u)
    tf_specs = [
        ("M5", raw_m5, dfs_h1, 48, 3.5),
        ("M15", dfs_m15, dfs_h1, 16, 4.0),
        ("M30", dfs_m30, dfs_h4, 8, 3.5),
        ("H1", dfs_h1, dfs_h4, 4, 3.0),
    ]

    tf_dfs_map = {}
    tf_sigs_map = {}
    anatomy_rows = []
    all_setup_tables = []

    print("=" * 112)
    print("CHICKTRADE — PHAN TICH TOAN DIEN 4 KHUNG GIO (M5 / M15 / M30 / H1) | STORM GUARD | SL_CHAN2 = ENTRY - 2U")
    print("=" * 112)

    for tf_name, tf_dfs, htf_dfs, cd_bars, tp1_u in tf_specs:
        tf_dfs_map[tf_name] = tf_dfs
        btc_st = compute_storm_flags(tf_dfs["BTCUSDT"])
        btc_mask = build_cooldown_mask(btc_st["raw_storm"], cd_bars)
        btc_ts = tf_dfs["BTCUSDT"]["timestamp"].values

        sigs_map = {}
        for sym in cfg.symbols:
            s, _ = generate_all_4_touches(tf_dfs[sym], htf_dfs[sym], btc_ts, btc_mask, sym, cfg.strategy, cooldown_bars=cd_bars)
            sigs_map[sym] = s
        tf_sigs_map[tf_name] = sigs_map

        anatomy_rows.append(measure_tf_anatomy(tf_dfs, sigs_map, tf_name))
        tbl = eval_tf_setups_with_oos(tf_dfs, sigs_map, tf_name, tp1_u=tp1_u, tp2_u=10.0)
        all_setup_tables.append(tbl)

    anat_df = pd.DataFrame(anatomy_rows)
    print("\n--- PHAN 1: BIEN DO TANG/GIAM & DO BAT TRUNG BINH THEO TUNG KHUNG GIO (BTC + ETH + BNB, 1 NAM) ---")
    print(anat_df.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    full_setups_df = pd.concat(all_setup_tables, ignore_index=True)
    print("\n--- PHAN 2: TOP SETUP CO LAI THUC TE (PF_1Y >= 1.05 & WR >= 55%) TREN CA 4 KHUNG ---")
    top_df = full_setups_df[(full_setups_df["PF_1Y"] >= 1.05) & (full_setups_df["WR_%"] >= 55.0)].sort_values(
        ["PF_1Y", "Net_U_1Y"], ascending=False
    )
    print(top_df.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    print("\n--- CHI TIET M5 & M30 (TOP 5 MOI KHUNG DE DOI CHIEU) ---")
    for tf_check in ["M5", "M30"]:
        sub = full_setups_df[full_setups_df["TF"] == tf_check].head(5)
        print(f"\n[{tf_check} Top 5]:")
        print(sub.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    # Build Unified Multi-TF Portfolio using ONLY robust setups (PF_1Y >= 1.10 AND PF_OOS6M >= 0.95)
    robust_df = full_setups_df[
        (full_setups_df["PF_1Y"] >= 1.10)
        & (full_setups_df["PF_OOS6M"] >= 0.95)
        & (full_setups_df["WR_%"] >= 60.0)
        & (full_setups_df["Trades"] >= 15)
    ]
    tp1_by_tf = {"M5": 3.5, "M15": 4.0, "M30": 3.5, "H1": 3.0}
    allowed_rules = {}
    for _, row in robust_df.iterrows():
        mult = 1.0 if (row["WR_%"] >= 70.0 and row["PF_1Y"] >= 1.50 and row["Trades"] >= 30) else (0.50 if row["PF_1Y"] >= 1.25 else 0.35)
        allowed_rules[(row["TF"], row["Setup"])] = (mult, tp1_by_tf[row["TF"]])

    # Midpoint for OOS
    all_ts = sorted(s.timestamp for s in tf_sigs_map["M15"]["BTCUSDT"])
    mid_ts = all_ts[len(all_ts) // 2]

    res_12m = run_unified_4tf_portfolio(tf_dfs_map, tf_sigs_map, allowed_rules, tp2_u=10.0, start_time=None)
    res_oos = run_unified_4tf_portfolio(tf_dfs_map, tf_sigs_map, allowed_rules, tp2_u=10.0, start_time=mid_ts)

    print("\n--- PHAN 3: KICH BAN PORTFOLIO DA KHUNG TONG HOP (CHI LAY SETUP BEN VUNG CA 2 NUA NAM) ---")
    print("  Setups duoc chon:", allowed_rules)
    print(
        f"  12M FULL: Equity=${res_12m['final_equity']:9,.2f} ({res_12m['return_pct']:+6.2f}%) | "
        f"Step {res_12m['max_step']:2d}/20 | Trades={res_12m['total_trades']:3d} | "
        f"WR={res_12m['winrate_pct']:5.2f}% | SL_Full={res_12m['sl_full_pct']:5.2f}% | "
        f"PF={res_12m['pf']:.2f} | MaxDD={res_12m['max_dd_pct']:6.2f}%"
    )
    print(
        f"   6M OOS : Equity=${res_oos['final_equity']:9,.2f} ({res_oos['return_pct']:+6.2f}%) | "
        f"Step {res_oos['max_step']:2d}/20 | Trades={res_oos['total_trades']:3d} | "
        f"WR={res_oos['winrate_pct']:5.2f}% | SL_Full={res_oos['sl_full_pct']:5.2f}% | "
        f"PF={res_oos['pf']:.2f} | MaxDD={res_oos['max_dd_pct']:6.2f}%"
    )

    out_dir = os.path.join(ROOT, "outputs")
    anat_df.to_csv(os.path.join(out_dir, "4tf_anatomy_summary.csv"), index=False)
    full_setups_df.to_csv(os.path.join(out_dir, "4tf_all_setups_summary.csv"), index=False)


if __name__ == "__main__":
    main()
