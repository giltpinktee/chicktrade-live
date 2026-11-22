"""Per-Coin DNA Discovery & Strict Out-of-Sample Validation across 6 Coins (BTC, ETH, BNB, SOL, XRP, LINK)."""
import os
import sys
from typing import Dict, List, Tuple

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

SIX_COINS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "LINKUSDT"]
TP1_BY_TF = {"M15": 4.0, "M30": 3.5, "H1": 3.0}


def run_personalized_portfolio(
    tf_dfs_map: Dict[str, Dict[str, pd.DataFrame]],
    tf_sigs_map: Dict[str, Dict[str, list]],
    coin_rules: Dict[Tuple[str, str, str], Tuple[float, float]],  # (symbol, tf, setup) -> (mult, tp1_u)
    buffer_units: float = 250.0,
    max_concurrent: int = 3,
    start_time: pd.Timestamp = None,
) -> dict:
    rcfg = RiskConfig(initial_capital=10_000.0, buffer_units=buffer_units, max_concurrent_trades=max_concurrent, max_trades_per_day=10)
    comp = CompoundingManager(rcfg)
    equity = rcfg.initial_capital
    peak_eq = equity

    candidates = []
    for tf_name, sigs_map in tf_sigs_map.items():
        for sym, sigs in sigs_map.items():
            for s in sigs:
                key = (sym, tf_name, s.setup_type)
                if key not in coin_rules:
                    continue
                if start_time is not None and s.timestamp < start_time:
                    continue
                mult, tp1_u = coin_rules[key]
                candidates.append((tf_name, mult, tp1_u, s))

    if not candidates:
        return {"final_equity": equity, "return_pct": 0.0, "max_step": 1, "total_trades": 0, "winrate_pct": 0.0, "pf": 0.0, "max_dd_pct": 0.0, "eq_df": pd.DataFrame()}

    tf_prio = {"H1": 0, "M30": 1, "M15": 2}
    candidates.sort(key=lambda x: (x[3].timestamp, tf_prio[x[0]], -x[1], x[3].symbol))

    active, closed = [], []
    daily_trades, daily_sl, daily_pnl, daily_start_eq = {}, {}, {}, {}
    consec_sl = 0
    eq_curve = [{"timestamp": candidates[0][3].timestamp, "equity": equity, "step": 1}]

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
        if daily_pnl.get(dk, 0.0) <= -0.05 * daily_start_eq[dk]:
            continue
        if len(active) >= rcfg.max_concurrent_trades:
            continue
        if any(tr["symbol"] == sig.symbol for _, tr in active):
            continue

        eff_mult = base_mult
        if sig.pa_vol_ok:
            eff_mult *= 0.5
        dd_pct = (equity - peak_eq) / peak_eq * 100.0
        if dd_pct <= -6.0 or consec_sl >= 2:
            eff_mult = min(eff_mult, 0.25)

        open_mult = sum(tr["size_mult"] for _, tr in active)
        eff_mult = min(eff_mult, max(0.0, 1.50 - open_mult))
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
        "pf": pf,
        "max_dd_pct": max_dd,
        "full_count": sum(1 for t in closed if t["size_mult"] >= 0.75),
        "half_count": sum(1 for t in closed if t["size_mult"] < 0.75),
    }


def main():
    cfg = BacktestConfig()
    end_ym = _detect_latest_available_month("BTCUSDT", "5m")

    raw_m5 = {sym: download_symbol_1y(sym, "5m", 12, "futures", end_ym) for sym in SIX_COINS}
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

    all_ts = sorted(s.timestamp for s in tf_sigs_map["M15"]["BTCUSDT"])
    mid_ts = all_ts[len(all_ts) // 2]

    print("=" * 112)
    print("DO DNA RIENG TUNG COIN (6 COINS: BTC, ETH, BNB, SOL, XRP, LINK) — KIEM CHUNG CA 2 NUA NAM (IS 6T -> OOS 6T)")
    print("=" * 112)

    all_coin_rows = []
    robust_rules_both_halves = {}   # Robust in BOTH IS & OOS (for true DNA profile)
    walkforward_rules_is_only = {}  # Selected strictly from IS 6M ONLY -> tested blindly on OOS 6M!

    for sym in SIX_COINS:
        sym_rows = []
        for tf_name in ["M15", "M30", "H1"]:
            df = tf_dfs_map[tf_name][sym]
            tp1_u = TP1_BY_TF[tf_name]
            sigs = tf_sigs_map[tf_name][sym]
            by_setup: Dict[str, list] = {}
            for s in sigs:
                r = sim_leg2_policy(df, s, 5.0, tp1_u, 10.0, "SL_MINUS_2U")
                r["timestamp"] = s.timestamp
                by_setup.setdefault(s.setup_type, []).append(r)

            for stype, rlist in by_setup.items():
                g = pd.DataFrame(rlist)
                if len(g) < 12:
                    continue
                g_is = g[g["timestamp"] < mid_ts]
                g_oos = g[g["timestamp"] >= mid_ts]

                def _pf(sub):
                    if len(sub) == 0:
                        return 0.0
                    p = sub.loc[sub["net_u"] > 0, "net_u"].sum()
                    ng = -sub.loc[sub["net_u"] < 0, "net_u"].sum()
                    return float(p / ng) if ng > 0 else (9.99 if p > 0 else 0.0)

                pf_1y = _pf(g)
                pf_is = _pf(g_is)
                pf_oos = _pf(g_oos)
                wr_1y = (g["net_u"] > 0).mean() * 100.0
                wr_is = (g_is["net_u"] > 0).mean() * 100.0 if len(g_is) else 0.0
                wr_oos = (g_oos["net_u"] > 0).mean() * 100.0 if len(g_oos) else 0.0

                row = {
                    "Coin": sym,
                    "TF": tf_name,
                    "Setup": stype,
                    "Trades_1Y": len(g),
                    "WR_1Y_%": wr_1y,
                    "PF_1Y": pf_1y,
                    "PF_IS6M": pf_is,
                    "PF_OOS6M": pf_oos,
                    "Net_U_1Y": g["net_u"].sum(),
                }
                sym_rows.append(row)
                all_coin_rows.append(row)

                # Walk-forward selection: ONLY look at IS 6M!
                if len(g_is) >= 8 and pf_is >= 1.25 and wr_is >= 62.0:
                    mult = 1.0 if (wr_is >= 72.0 and pf_is >= 1.60) else 0.50
                    walkforward_rules_is_only[(sym, tf_name, stype)] = (mult, tp1_u)

                # Confirmed robust DNA: profitable in BOTH halves (IS >= 1.10 AND OOS >= 1.10)
                if pf_is >= 1.10 and pf_oos >= 1.10 and wr_1y >= 62.0:
                    mult = 1.0 if (wr_1y >= 72.0 and pf_1y >= 1.60) else 0.50
                    robust_rules_both_halves[(sym, tf_name, stype)] = (mult, tp1_u)

        sdf = pd.DataFrame(sym_rows)
        robust_sub = sdf[(sdf["PF_IS6M"] >= 1.10) & (sdf["PF_OOS6M"] >= 1.10) & (sdf["WR_1Y_%"] >= 62.0)].sort_values("PF_1Y", ascending=False)
        print(f"\n>>> [{sym}] — CAC SETUP BEN VUNG CA 2 NUA NAM (IS >= 1.10 & OOS >= 1.10):")
        if robust_sub.empty:
            print("    (Khong co setup nao du ben ca 2 nua nam — Top 3 theo PF_1Y de tham khao:)")
            print(sdf.sort_values("PF_1Y", ascending=False).head(3).to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
        else:
            print(robust_sub.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    print("\n" + "=" * 112)
    print("KIEM CHUNG PORTFOLIO 6 COIN DANH RIENG THEO DNA TUNG COIN (BAO 250 GIA, MAX 3 LENH, STORM GUARD)")
    print("=" * 112)

    # 1) Strict Blind Walk-Forward OOS (rules picked ONLY from first 6M -> tested on unseen last 6M)
    r_wf_oos = run_personalized_portfolio(tf_dfs_map, tf_sigs_map, walkforward_rules_is_only, buffer_units=250.0, max_concurrent=3, start_time=mid_ts)
    # 2) Robust DNA Portfolio (12M & OOS 6M)
    r_dna_12m = run_personalized_portfolio(tf_dfs_map, tf_sigs_map, robust_rules_both_halves, buffer_units=250.0, max_concurrent=3, start_time=None)
    r_dna_oos = run_personalized_portfolio(tf_dfs_map, tf_sigs_map, robust_rules_both_halves, buffer_units=250.0, max_concurrent=3, start_time=mid_ts)

    print(
        f"  [1. BLIND WALK-FORWARD OOS 6M (Chon luat CHI TU 6T dau -> test mu 6T cuoi, {len(walkforward_rules_is_only)} rules)]\n"
        f"      Equity=${r_wf_oos['final_equity']:9,.2f} ({r_wf_oos['return_pct']:+6.2f}%) | Step {r_wf_oos['max_step']:2d}/20 | "
        f"Trades={r_wf_oos['total_trades']:3d} | WR={r_wf_oos['winrate_pct']:5.2f}% | PF={r_wf_oos['pf']:.2f} | MaxDD={r_wf_oos['max_dd_pct']:6.2f}%"
    )
    print(
        f"  [2. ROBUST PER-COIN DNA — 12M FULL ({len(robust_rules_both_halves)} rules ben ca 2 nua nam)]\n"
        f"      Equity=${r_dna_12m['final_equity']:9,.2f} ({r_dna_12m['return_pct']:+6.2f}%) | Step {r_dna_12m['max_step']:2d}/20 | "
        f"Trades={r_dna_12m['total_trades']:3d} | WR={r_dna_12m['winrate_pct']:5.2f}% | PF={r_dna_12m['pf']:.2f} | MaxDD={r_dna_12m['max_dd_pct']:6.2f}%"
    )
    print(
        f"  [3. ROBUST PER-COIN DNA — 6M OOS]\n"
        f"      Equity=${r_dna_oos['final_equity']:9,.2f} ({r_dna_oos['return_pct']:+6.2f}%) | Step {r_dna_oos['max_step']:2d}/20 | "
        f"Trades={r_dna_oos['total_trades']:3d} | WR={r_dna_oos['winrate_pct']:5.2f}% | PF={r_dna_oos['pf']:.2f} | MaxDD={r_dna_oos['max_dd_pct']:6.2f}%"
    )

    out_dir = os.path.join(ROOT, "outputs")
    pd.DataFrame(all_coin_rows).to_csv(os.path.join(out_dir, "per_coin_dna_6coins.csv"), index=False)


if __name__ == "__main__":
    main()
