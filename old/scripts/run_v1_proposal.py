"""Test & Verify High-Winrate Tiered-Volume Proposal (Full 1.0x / 0.4x / 0.25x + Partial TP + BE+Fee + Anti-DD)."""
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
from run_3ema_complete import generate_3ema_signals
from run_partial_tp_test import simulate_partial_tp
from run_research_mtf import resample_ohlcv
from strategy.ema_wave import Signal


def add_rsi_and_vol_regime(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    delta = out["close"].diff()
    gain = delta.clip(lower=0.0).ewm(alpha=1.0 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0.0)).ewm(alpha=1.0 / 14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    out["rsi"] = (100.0 - 100.0 / (1.0 + rs)).fillna(50.0)

    tr = pd.concat(
        [
            out["high"] - out["low"],
            (out["high"] - out["close"].shift(1)).abs(),
            (out["low"] - out["close"].shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    out["atr"] = tr.ewm(alpha=1.0 / 14, adjust=False).mean()
    out["atr_ma50"] = out["atr"].rolling(50, min_periods=1).mean()
    out["high_vol_regime"] = out["atr"] > (1.5 * out["atr_ma50"])
    return out


def simulate_tiered_portfolio(
    dfs_m15: Dict[str, pd.DataFrame],
    dfs_h1: Dict[str, pd.DataFrame],
    sigs_m15: Dict[str, List[Signal]],
    sigs_h1: Dict[str, List[Signal]],
    use_rsi_filter: bool = True,
    m15_tp1: float = 3.5,
    h1_tp1: float = 3.0,
    tp2: float = 10.0,
    start_time: pd.Timestamp = None,
) -> dict:
    """Run unified multi-TF portfolio with 3-tier sizing (1.0x / 0.4x / 0.25x) + drawdown brake + 3-SL daily stop."""
    rcfg = RiskConfig(initial_capital=10_000.0, buffer_units=500.0, max_concurrent_trades=2, max_trades_per_day=10)
    comp = CompoundingManager(rcfg)
    equity = rcfg.initial_capital
    peak_equity = equity

    # Precompute RSI & volatility regime
    m15_feat = {s: add_rsi_and_vol_regime(df) for s, df in dfs_m15.items()}
    h1_feat = {s: add_rsi_and_vol_regime(df) for s, df in dfs_h1.items()}

    # Merge candidate signals: H1 EMA34_W2 (Tier A: 1.0x) + M15 EMA34_W1 (Tier B: 0.40x)
    candidates = []
    for sym in dfs_h1:
        feat = h1_feat[sym]
        for s in sigs_h1[sym]:
            if s.setup_type != "EMA34_W2":
                continue
            if start_time is not None and s.timestamp < start_time:
                continue
            rsi = float(feat["rsi"].iloc[s.bar_idx])
            hv = bool(feat["high_vol_regime"].iloc[s.bar_idx])
            if use_rsi_filter:
                if s.direction == 1 and not (35.0 <= rsi <= 62.0):
                    continue
                if s.direction == -1 and not (38.0 <= rsi <= 65.0):
                    continue
            candidates.append(("H1", "TIER_A", 1.0, h1_tp1, hv, s))

    for sym in dfs_m15:
        feat = m15_feat[sym]
        for s in sigs_m15[sym]:
            if s.setup_type != "EMA34_W1":
                continue
            if start_time is not None and s.timestamp < start_time:
                continue
            rsi = float(feat["rsi"].iloc[s.bar_idx])
            hv = bool(feat["high_vol_regime"].iloc[s.bar_idx])
            if use_rsi_filter:
                if s.direction == 1 and not (38.0 <= rsi <= 60.0):
                    continue
                if s.direction == -1 and not (40.0 <= rsi <= 62.0):
                    continue
            candidates.append(("M15", "TIER_B", 0.40, m15_tp1, hv, s))

    # Sort chronologically (prioritize Tier A H1 over Tier B M15 on same timestamp)
    candidates.sort(key=lambda x: (x[5].timestamp, 0 if x[1] == "TIER_A" else 1, x[5].symbol))

    active = []  # (exit_time, trade_dict)
    closed = []
    daily_trades: Dict[str, int] = {}
    daily_sl_streak: Dict[str, int] = {}
    consec_sl = 0
    eq_curve = [{"timestamp": candidates[0][5].timestamp if candidates else pd.Timestamp.utcnow(), "equity": equity, "step": 1}]

    def settle(now_ts):
        nonlocal equity, peak_equity, consec_sl
        keep, done = [], []
        for ex_ts, tr in active:
            (done if ex_ts <= now_ts else keep).append((ex_ts, tr))
        done.sort(key=lambda x: x[0])
        for ex_ts, tr in done:
            equity += tr["net_pnl"]
            peak_equity = max(peak_equity, equity)
            comp.update_equity(equity, ex_ts)
            if tr["outcome"] == "SL_FULL":
                consec_sl += 1
                dk = tr["day_key"]
                daily_sl_streak[dk] = daily_sl_streak.get(dk, 0) + 1
            elif tr["net_pnl"] > 0:
                consec_sl = 0
            tr["equity_after"] = equity
            closed.append(tr)
            eq_curve.append({"timestamp": ex_ts, "equity": equity, "step": comp.step})
        active[:] = keep

    for tf_name, tier, base_mult, tp1_u, is_high_vol, sig in candidates:
        settle(sig.timestamp)
        if equity <= 0:
            break

        day_key = sig.timestamp.strftime("%Y-%m-%d")
        # Daily circuit breaker: stop after 3 full SLs in a day OR 10 trades/day
        if daily_trades.get(day_key, 0) >= rcfg.max_trades_per_day:
            continue
        if daily_sl_streak.get(day_key, 0) >= 3:
            continue
        if len(active) >= rcfg.max_concurrent_trades:
            continue
        if any(tr["symbol"] == sig.symbol for _, tr in active):
            continue

        # Dynamic risk reduction when volatility spikes, in drawdown > 3%, or after 2 consecutive SLs
        dd_pct = (equity - peak_equity) / peak_equity * 100.0
        eff_mult = base_mult
        if is_high_vol:
            eff_mult *= 0.5   # High volatility -> cut size in half (e.g., 0.40 -> 0.20, 1.0 -> 0.50)
        if dd_pct <= -3.0 or consec_sl >= 2:
            eff_mult = min(eff_mult, 0.25)  # Defensive Micro Size (0.25x => SL 5U = only 0.25% risk!)
        # If Tier A (1.0x) is already open, cap 2nd concurrent trade to 0.35x
        if any(tr["size_mult"] >= 0.75 for _, tr in active):
            eff_mult = min(eff_mult, 0.35)

        df = dfs_h1[sig.symbol] if tf_name == "H1" else dfs_m15[sig.symbol]
        sim = simulate_partial_tp(df, sig, sl_units=5.0, tp1_units=tp1_u, tp2_units=tp2, tp1_frac=0.5)

        unit_size = max(sig.atr * 0.5, sig.entry_price * 1e-5)
        sizing_base = min(comp.tier_capital, equity)
        qty = (sizing_base * eff_mult) / (rcfg.buffer_units * unit_size)
        net_pnl = sim["net_units"] * unit_size * qty
        fee_usd = sim["fee_units"] * unit_size * qty

        # Approximate exit timestamp (+6 bars for TP1+BE/TP1+TP2 or +12 bars max)
        bars_est = 4 if sim["outcome"] == "SL_FULL" else 8
        exit_idx = min(sig.bar_idx + bars_est, len(df) - 1)
        ex_ts = pd.Timestamp(df["timestamp"].iloc[exit_idx])
        if ex_ts.tzinfo is None:
            ex_ts = ex_ts.tz_localize("UTC")

        tr = {
            "tf": tf_name,
            "tier": tier,
            "symbol": sig.symbol,
            "setup": sig.setup_type,
            "size_mult": eff_mult,
            "outcome": sim["outcome"],
            "net_units": sim["net_units"],
            "net_pnl": net_pnl,
            "fee_usd": fee_usd,
            "day_key": day_key,
            "entry_time": sig.timestamp,
            "exit_time": ex_ts,
        }
        active.append((ex_ts, tr))
        daily_trades[day_key] = daily_trades.get(day_key, 0) + 1

    if active:
        settle(pd.Timestamp("2099-01-01", tz="UTC"))

    eq_df = pd.DataFrame(eq_curve)
    pnls = np.array([t["net_pnl"] for t in closed]) if closed else np.array([0.0])
    wins = int((pnls > 0).sum())
    wr = wins / len(closed) * 100.0 if closed else 0.0
    pos = pnls[pnls > 0].sum()
    neg = -pnls[pnls < 0].sum()
    pf = float(pos / neg) if neg > 0 else 0.0
    eq_vals = eq_df["equity"].values
    pk = np.maximum.accumulate(eq_vals)
    max_dd = float(((eq_vals - pk) / pk * 100.0).min())

    return {
        "trades": closed,
        "eq_df": eq_df,
        "final_equity": float(eq_vals[-1]),
        "return_pct": (float(eq_vals[-1]) / rcfg.initial_capital - 1.0) * 100.0,
        "max_step": comp.max_step_reached,
        "total_trades": len(closed),
        "winrate_pct": wr,
        "sl_full_pct": sum(1 for t in closed if t["outcome"] == "SL_FULL") / len(closed) * 100.0 if closed else 0.0,
        "pf": pf,
        "max_dd_pct": max_dd,
        "fees": float(sum(t["fee_usd"] for t in closed)),
    }


def main():
    cfg = BacktestConfig()
    end_ym = _detect_latest_available_month("BTCUSDT", "5m")

    raw_m5 = {sym: download_symbol_1y(sym, interval="5m", months=12, market="futures", end_ym=end_ym) for sym in cfg.symbols}
    dfs_m15 = {sym: resample_ohlcv(df, "15min") for sym, df in raw_m5.items()}
    dfs_h1 = {sym: resample_ohlcv(df, "1h") for sym, df in raw_m5.items()}
    dfs_h4 = {sym: resample_ohlcv(df, "4h") for sym, df in raw_m5.items()}

    sigs_m15 = {sym: generate_3ema_signals(dfs_m15[sym], dfs_h1[sym], sym, cfg.strategy, min_adx=20.0, max_wave=2) for sym in cfg.symbols}
    sigs_h1 = {sym: generate_3ema_signals(dfs_h1[sym], dfs_h4[sym], sym, cfg.strategy, min_adx=20.0, max_wave=2) for sym in cfg.symbols}

    all_ts = sorted(s.timestamp for sigs in sigs_m15.values() for s in sigs)
    mid_ts = all_ts[len(all_ts) // 2]

    print("=" * 100)
    print("CHICKTRADE V1 PROPOSAL — 3-TIER ADAPTIVE VOLUME (1.0x / 0.40x / 0.25x) + PARTIAL TP + BE_FEE")
    print("=" * 100)

    configs = [
        ("V1-A: No RSI | TP1=3U/3.5U + BE_Fee + TP2=10U", False, 3.5, 3.0, 10.0),
        ("V1-B: +RSI Filter | TP1=3U/3.5U + BE_Fee + TP2=10U", True, 3.5, 3.0, 10.0),
        ("V1-C: +RSI Filter | TP1=3U/4U + BE_Fee + TP2=10U", True, 4.0, 3.0, 10.0),
        ("V1-D: No RSI | TP1=3U/4U + BE_Fee + TP2=10U", False, 4.0, 3.0, 10.0),
    ]

    for label, use_rsi, m15_tp1, h1_tp1, tp2 in configs:
        r_full = simulate_tiered_portfolio(dfs_m15, dfs_h1, sigs_m15, sigs_h1, use_rsi, m15_tp1, h1_tp1, tp2, None)
        r_oos = simulate_tiered_portfolio(dfs_m15, dfs_h1, sigs_m15, sigs_h1, use_rsi, m15_tp1, h1_tp1, tp2, mid_ts)
        print(f"\n[{label}]")
        print(
            f"  12M FULL: Equity=${r_full['final_equity']:9,.2f} ({r_full['return_pct']:+6.2f}%) | "
            f"Step {r_full['max_step']:2d}/20 | Trades={r_full['total_trades']:3d} | "
            f"WR={r_full['winrate_pct']:5.2f}% | SL_Full={r_full['sl_full_pct']:5.2f}% | "
            f"PF={r_full['pf']:.2f} | MaxDD={r_full['max_dd_pct']:6.2f}% | Fees=${r_full['fees']:,.0f}"
        )
        print(
            f"   6M OOS : Equity=${r_oos['final_equity']:9,.2f} ({r_oos['return_pct']:+6.2f}%) | "
            f"Step {r_oos['max_step']:2d}/20 | Trades={r_oos['total_trades']:3d} | "
            f"WR={r_oos['winrate_pct']:5.2f}% | SL_Full={r_oos['sl_full_pct']:5.2f}% | "
            f"PF={r_oos['pf']:.2f} | MaxDD={r_oos['max_dd_pct']:6.2f}% | Fees=${r_oos['fees']:,.0f}"
        )


if __name__ == "__main__":
    main()
