"""6-Coin Parallel Portfolio Backtest (BTC, ETH, BNB, SOL, XRP, LINK) | Max 3 Concurrent | Bao 250 Gia | Storm Guard."""
import os
import sys
from typing import Dict

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
from run_buffer_comparison import VERIFIED_RULES
from run_full_3ema_storm_guard import build_cooldown_mask, compute_storm_flags, generate_all_4_touches
from run_research_mtf import resample_ohlcv

SIX_COINS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "LINKUSDT"]


def run_6coin_portfolio(
    tf_dfs_map: Dict[str, Dict[str, pd.DataFrame]],
    tf_sigs_map: Dict[str, Dict[str, list]],
    buffer_units: float = 250.0,
    max_concurrent: int = 3,
    max_trades_per_day: int = 10,
    start_time: pd.Timestamp = None,
) -> dict:
    rcfg = RiskConfig(
        initial_capital=10_000.0,
        buffer_units=buffer_units,
        max_concurrent_trades=max_concurrent,
        max_trades_per_day=max_trades_per_day,
    )
    comp = CompoundingManager(rcfg)
    equity = rcfg.initial_capital
    peak_eq = equity

    candidates = []
    for tf_name, sigs_map in tf_sigs_map.items():
        for sym, sigs in sigs_map.items():
            for s in sigs:
                key = (tf_name, s.setup_type)
                if key not in VERIFIED_RULES:
                    continue
                if start_time is not None and s.timestamp < start_time:
                    continue
                mult, tp1_u = VERIFIED_RULES[key]
                candidates.append((tf_name, mult, tp1_u, s))

    tf_prio = {"H1": 0, "M30": 1, "M15": 2}
    candidates.sort(key=lambda x: (x[3].timestamp, tf_prio[x[0]], -x[1], x[3].symbol))

    active, closed = [], []
    daily_trades, daily_sl, daily_pnl, daily_start_eq = {}, {}, {}, {}
    consec_sl = 0
    eq_curve = [{"timestamp": candidates[0][3].timestamp, "equity": equity, "step": 1}]

    daily_loss_cap = 0.05 if buffer_units <= 250 else 0.04
    dd_brake_thresh = -6.0 if buffer_units <= 250 else -4.5

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
        if daily_pnl.get(dk, 0.0) <= -daily_loss_cap * daily_start_eq[dk]:
            continue
        if len(active) >= rcfg.max_concurrent_trades:
            continue
        if any(tr["symbol"] == sig.symbol for _, tr in active):
            continue

        eff_mult = base_mult
        if sig.pa_vol_ok:
            eff_mult *= 0.5
        dd_pct = (equity - peak_eq) / peak_eq * 100.0
        if dd_pct <= dd_brake_thresh or consec_sl >= 2:
            eff_mult = min(eff_mult, 0.25)

        # Portfolio exposure cap across 3 concurrent positions: total open size_mult <= 1.50x
        open_mult = sum(tr["size_mult"] for _, tr in active)
        remaining_cap = max(0.0, 1.50 - open_mult)
        eff_mult = min(eff_mult, remaining_cap)
        if eff_mult < 0.20:
            continue

        df = tf_dfs_map[tf_name][sig.symbol]
        sim = sim_leg2_policy(df, sig, 5.0, tp1_u, 10.0, "SL_MINUS_2U")
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
        "full_count": sum(1 for t in closed if t["size_mult"] >= 0.75),
        "half_count": sum(1 for t in closed if t["size_mult"] < 0.75),
    }


def main():
    cfg = BacktestConfig()
    end_ym = _detect_latest_available_month("BTCUSDT", "5m")

    print("=" * 110)
    print("CHICKTRADE — 6-COIN PARALLEL BACKTEST (BTC, ETH, BNB, SOL, XRP, LINK) | MAX 3 OPEN | BAO 250 GIA")
    print("=" * 110)
    print(f"[1/3] Loading 12 months M5 futures data ending {end_ym[0]}-{end_ym[1]:02d}...")

    raw_m5 = {}
    for sym in SIX_COINS:
        df = download_symbol_1y(sym, interval="5m", months=12, market="futures", end_ym=end_ym)
        raw_m5[sym] = df
        print(f"  - {sym:8s}: {len(df):6d} bars")

    dfs_m15 = {s: resample_ohlcv(d, "15min") for s, d in raw_m5.items()}
    dfs_m30 = {s: resample_ohlcv(d, "30min") for s, d in raw_m5.items()}
    dfs_h1 = {s: resample_ohlcv(d, "1h") for s, d in raw_m5.items()}
    dfs_h4 = {s: resample_ohlcv(d, "4h") for s, d in raw_m5.items()}
    tf_dfs_map = {"M15": dfs_m15, "M30": dfs_m30, "H1": dfs_h1}

    tf_sigs_map = {}
    for tf_name, tf_dfs, htf_dfs, cd in [("M15", dfs_m15, dfs_h1, 16), ("M30", dfs_m30, dfs_h4, 8), ("H1", dfs_h1, dfs_h4, 4)]:
        bst = compute_storm_flags(tf_dfs["BTCUSDT"])
        bmask = build_cooldown_mask(bst["raw_storm"], cd)
        bts = tf_dfs["BTCUSDT"]["timestamp"].values
        tf_sigs_map[tf_name] = {
            sym: generate_all_4_touches(tf_dfs[sym], htf_dfs[sym], bts, bmask, sym, cfg.strategy, cooldown_bars=cd)[0]
            for sym in SIX_COINS
        }

    # Per-coin independent stats across VERIFIED_RULES
    print("\n[2/3] Per-Coin Breakdown (12 Months, Verified M15+M30+H1 Setups, SL_Leg2 = Entry - 2U, Storm Guard):")
    coin_rows = []
    for sym in SIX_COINS:
        sim_res = []
        for tf_name, sigs_map in tf_sigs_map.items():
            df = tf_dfs_map[tf_name][sym]
            for s in sigs_map[sym]:
                key = (tf_name, s.setup_type)
                if key not in VERIFIED_RULES:
                    continue
                _, tp1_u = VERIFIED_RULES[key]
                r = sim_leg2_policy(df, s, 5.0, tp1_u, 10.0, "SL_MINUS_2U")
                sim_res.append(r)
        rdf = pd.DataFrame(sim_res)
        pos = rdf.loc[rdf["net_u"] > 0, "net_u"].sum()
        neg = -rdf.loc[rdf["net_u"] < 0, "net_u"].sum()
        pf = pos / neg if neg > 0 else 0.0
        coin_rows.append(
            {
                "Coin": sym,
                "Signals_1Y": len(rdf),
                "WR_%": (rdf["net_u"] > 0).mean() * 100.0,
                "TP2_%": (rdf["outcome"] == "TP1+TP2").mean() * 100.0,
                "SL_Full_%": (rdf["outcome"] == "SL_FULL").mean() * 100.0,
                "PF": pf,
                "Net_U_1Y": rdf["net_u"].sum(),
            }
        )
    coin_df = pd.DataFrame(coin_rows).sort_values("PF", ascending=False)
    print(coin_df.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    all_ts = sorted(s.timestamp for s in tf_sigs_map["M15"]["BTCUSDT"])
    mid_ts = all_ts[len(all_ts) // 2]

    print("\n[3/3] 6-Coin Parallel Portfolio Simulation (Max 3 Concurrent Trades, Max 10/Day, Storm Guard):")
    portfolio_runs = {}
    summary_rows = []
    for buf in [250.0, 300.0]:
        r12 = run_6coin_portfolio(tf_dfs_map, tf_sigs_map, buffer_units=buf, max_concurrent=3, start_time=None)
        r6 = run_6coin_portfolio(tf_dfs_map, tf_sigs_map, buffer_units=buf, max_concurrent=3, start_time=mid_ts)
        summary_rows.append(
            {
                "Bao_Gia": f"{int(buf)} U",
                "12M_Equity_$": r12["final_equity"],
                "12M_Return_%": r12["return_pct"],
                "12M_Step": f"{r12['max_step']}/20",
                "12M_Trades": r12["total_trades"],
                "12M_WR_%": r12["winrate_pct"],
                "12M_PF": r12["pf"],
                "12M_MaxDD_%": r12["max_dd_pct"],
                "OOS6M_Return_%": r6["return_pct"],
                "OOS6M_WR_%": r6["winrate_pct"],
                "OOS6M_MaxDD_%": r6["max_dd_pct"],
            }
        )
        portfolio_runs[f"6Coin_Buf{int(buf)}_12M"] = {
            "label": f"6 Coin — Bao {int(buf)} giá (12 Tháng, Max 3 lệnh)",
            "trades": [],
            "equity_curve": r12["eq_df"],
            "metrics": {
                "final_equity": r12["final_equity"],
                "return_pct": r12["return_pct"],
                "max_step": r12["max_step"],
                "total_trades": r12["total_trades"],
                "winrate_pct": r12["winrate_pct"],
                "profit_factor": r12["pf"],
                "max_dd_pct": r12["max_dd_pct"],
                "total_fees": 0.0,
                "full_count": r12["full_count"],
                "half_count": r12["half_count"],
            },
        }
        portfolio_runs[f"6Coin_Buf{int(buf)}_OOS6M"] = {
            "label": f"6 Coin — Bao {int(buf)} giá (OOS 6 Tháng cuối)",
            "trades": [],
            "equity_curve": r6["eq_df"],
            "metrics": {
                "final_equity": r6["final_equity"],
                "return_pct": r6["return_pct"],
                "max_step": r6["max_step"],
                "total_trades": r6["total_trades"],
                "winrate_pct": r6["winrate_pct"],
                "profit_factor": r6["pf"],
                "max_dd_pct": r6["max_dd_pct"],
                "total_fees": 0.0,
                "full_count": r6["full_count"],
                "half_count": r6["half_count"],
            },
        }

    sum_df = pd.DataFrame(summary_rows)
    print(sum_df.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    out_dir = os.path.join(ROOT, "outputs")
    report_path = export_html_report(
        out_dir,
        {"6Coin_Per_Coin_Stats": coin_df, "6Coin_Portfolio_Summary": sum_df},
        portfolio_runs,
    )
    print(f"\nUpdated HTML report saved to: {report_path}")


if __name__ == "__main__":
    main()
