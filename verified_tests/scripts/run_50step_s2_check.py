"""50-Step S2_LOCKED_HYSTERESIS_10PCT Expansion Check under U3_HTF_MACRO_ATR (2023-11-01 -> 2026-10-31).

Compares 3 compounding ceiling configurations under S2_LOCKED_HYSTERESIS_10PCT (+8%/step, -10% hysteresis):
  1. S2_20_STEPS        : max_step=20 ($46.6k cap), uncapped notional
  2. S2_50_STEPS_LIQ_CAP: max_step=50 ($469.0k cap) + live liquidity cap ($300k BTC/ETH, $150k Alts per trade)
  3. S2_50_STEPS_PURE   : max_step=50 ($469.0k cap), uncapped notional

Evaluated on:
  - v3.0 (58 setups): Full 36M (11/2023->10/2026) & Blind Y3 (11/2025->10/2026, reset $10k)
  - Test C Reverse Blind (59 setups): Blind 2Y (11/2023->10/2025, reset $10k)
for both Bao 400G and Auto-Turbo 250G.

Output:
  - e:\CHICKTRADE\verified_tests\results\step50_s2_check.csv
"""
import os
import sys
from typing import Dict, Optional

import numpy as np
import pandas as pd

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
VERIFIED_DIR = os.path.dirname(SCRIPTS_DIR)
ROOT = os.path.dirname(VERIFIED_DIR)
for p in (ROOT, SCRIPTS_DIR, os.path.join(ROOT, "old", "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from config import BacktestConfig
from run_full_216_exit_sweetspot import sim_exit_arch
from run_full_3ema_storm_guard import build_cooldown_mask, compute_storm_flags, generate_all_4_touches
from run_invariant_1u_benchmark import discover_dual_split_rules_arrays, precompute_invariant_u_arrays
from run_master_6coin_html import SIX_COINS
from run_pct_vs_atr_3year_study import filter_sigs, load_3year_cached_m5
from run_research_mtf import resample_ohlcv
from run_step_sizing_ab_test import load_v30_rules_map

RESULTS_DIR = os.path.join(VERIFIED_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def sim_s2_steps(
    tf_dfs_map: dict,
    tf_sigs_map: dict,
    rules_map: dict,
    u3_arrays: Dict[str, Dict[str, np.ndarray]],
    max_step: int,
    apply_liq_cap: bool,
    window_start_ts: pd.Timestamp,
    window_end_ts: pd.Timestamp,
    base_buffer: float = 400.0,
    auto_turbo_buffer: Optional[float] = None,
    hard_floor_eq: float = 11_000.0,
    core_only_sl_trigger: bool = True,
) -> dict:
    """Run S2_LOCKED_HYSTERESIS_10PCT with configurable max_step and optional per-trade notional cap."""
    step = 0
    max_step_reached = 0
    tier_capital = 10_000.0
    equity = 10_000.0
    peak_eq = equity
    peak_ts = window_start_ts
    max_rec_days = 0.0
    open_cap = 4.25
    in_turbo = False
    cap_hits = 0

    cands = []
    for tfl, sm in tf_sigs_map.items():
        for sym, sigs in sm.items():
            for s in sigs:
                k = (sym, tfl, s.setup_type)
                if k in rules_map:
                    cands.append((tfl, rules_map[k], s))

    tf_prio = {"1h": 0, "30m": 1, "15m": 2, "5m": 3}
    cands.sort(key=lambda x: (x[2].timestamp, tf_prio[x[0]], -x[1]["vol"], x[2].symbol))

    active, closed = [], []
    daily_t, daily_pnl, daily_eq = {}, {}, {}
    eq_curve = [equity]

    port_consec_sl = 0
    port_shadow_paused = False
    port_paused_until = pd.Timestamp("1970-01-01", tz="UTC")

    def settle(now_ts: pd.Timestamp):
        nonlocal equity, peak_eq, peak_ts, max_rec_days, step, max_step_reached, tier_capital
        nonlocal port_consec_sl, port_shadow_paused, port_paused_until, in_turbo
        keep, done = [], []
        for ex_ts, tr in active:
            (done if ex_ts <= now_ts else keep).append((ex_ts, tr))
        done.sort(key=lambda x: x[0])
        for ex_ts, tr in done:
            equity += tr["pnl"]
            if equity >= peak_eq:
                max_rec_days = max(max_rec_days, (ex_ts - peak_ts).total_seconds() / 86400.0)
                peak_eq = equity
                peak_ts = ex_ts

            while step < max_step and equity >= 10_000.0 * (1.08 ** (step + 1)):
                step += 1
                tier_capital = 10_000.0 * (1.08 ** step)
            while step > 0 and equity < tier_capital * 0.90:
                step -= 1
                tier_capital = 10_000.0 * (1.08 ** step)
            max_step_reached = max(max_step_reached, step)

            dk = tr["dk"]
            daily_pnl[dk] = daily_pnl.get(dk, 0.0) + tr["pnl"]

            if auto_turbo_buffer is not None:
                if (not in_turbo) and equity >= 12_500.0:
                    in_turbo = True
                elif in_turbo and equity < 11_500.0:
                    in_turbo = False

            if tr["out"] == "SL":
                if (not core_only_sl_trigger) or tr["base_vol"] > 0.45:
                    port_consec_sl += 1
                    if port_consec_sl >= 2:
                        port_consec_sl = 0
                        port_shadow_paused = True
                        port_paused_until = ex_ts + pd.Timedelta(hours=72)
            elif tr["pnl"] > 0:
                port_consec_sl = 0

            closed.append(tr)
            eq_curve.append(equity)
        active[:] = keep

    for tfl, rule, sig in cands:
        settle(sig.timestamp)
        if equity <= 0:
            break
        dk = sig.timestamp.strftime("%Y-%m-%d")
        if dk not in daily_eq:
            daily_eq[dk] = equity

        df = tf_dfs_map[tfl][sig.symbol]
        u = float(u3_arrays[tfl][sig.symbol][sig.bar_idx])

        if port_shadow_paused:
            if sig.timestamp >= port_paused_until:
                port_shadow_paused = False
            else:
                _, _, _, _, nu_sh = sim_exit_arch(
                    df["high"].values, df["low"].values, df["close"].values,
                    sig.bar_idx, sig.entry_price, sig.direction, u,
                    rule["sl_u"], rule["tp1_u"], rule["tp2_u"], rule["arch"], rule["l2_u"],
                )
                if nu_sh > 0:
                    port_shadow_paused = False
                continue

        if daily_t.get(dk, 0) >= 10 or daily_pnl.get(dk, 0.0) <= -0.05 * daily_eq[dk]:
            continue
        if len(active) >= 5 or any(tr["sym"] == sig.symbol for _, tr in active):
            continue

        eff_m = rule["vol"]
        if sig.pa_vol_ok:
            eff_m *= 0.5

        cur_dd_pct = (equity - peak_eq) / peak_eq * 100.0
        if cur_dd_pct <= -6.0:
            eff_m = min(eff_m, 0.25)

        eff_m = min(eff_m, max(0.0, open_cap - sum(tr["m"] for _, tr in active)))
        if eff_m < 0.20:
            continue

        sb = tier_capital
        cur_buf = base_buffer
        if auto_turbo_buffer is not None and in_turbo:
            cur_buf = auto_turbo_buffer
            open_sl_risk = sum(tr["worst_sl_usd"] for _, tr in active)
            new_sl_risk = (sb * eff_m) * (rule["sl_u"] / cur_buf) * 1.05
            if equity - (open_sl_risk + new_sl_risk) < hard_floor_eq:
                cur_buf = base_buffer

        out, _, _, _, nu = sim_exit_arch(
            df["high"].values, df["low"].values, df["close"].values,
            sig.bar_idx, sig.entry_price, sig.direction, u,
            rule["sl_u"], rule["tp1_u"], rule["tp2_u"], rule["arch"], rule["l2_u"],
        )
        qty = (sb * eff_m) / (cur_buf * u)
        if apply_liq_cap:
            liq_cap = 300_000.0 if sig.symbol in ("BTCUSDT", "ETHUSDT") else 150_000.0
            if qty * sig.entry_price > liq_cap:
                qty = liq_cap / sig.entry_price
                cap_hits += 1

        pnl = nu * u * qty
        worst_sl_usd = qty * (rule["sl_u"] * u) * 1.05

        ex_i = min(sig.bar_idx + (4 if out == "SL" else 8), len(df) - 1)
        ex_ts = pd.Timestamp(df["timestamp"].iloc[ex_i])
        if ex_ts.tzinfo is None:
            ex_ts = ex_ts.tz_localize("UTC")

        active.append((ex_ts, {
            "sym": sig.symbol,
            "dir": sig.direction,
            "base_vol": rule["vol"],
            "m": eff_m,
            "out": out,
            "pnl": float(pnl),
            "dk": dk,
            "worst_sl_usd": worst_sl_usd,
        }))
        daily_t[dk] = daily_t.get(dk, 0) + 1

    if active:
        settle(pd.Timestamp("2099-01-01", tz="UTC"))

    if equity < peak_eq:
        max_rec_days = max(max_rec_days, (window_end_ts - peak_ts).total_seconds() / 86400.0)

    ev = np.array(eq_curve)
    pk = np.maximum.accumulate(ev)
    mdd = float(((ev - pk) / pk * 100.0).min())
    pnls = np.array([t["pnl"] for t in closed])
    wr = float((pnls > 0).mean() * 100.0) if len(pnls) else 0.0
    pf = float(pnls[pnls > 0].sum() / max(-pnls[pnls < 0].sum(), 0.01)) if len(pnls) else 0.0
    ret = (float(ev[-1]) / 10_000.0 - 1.0) * 100.0
    return {
        "trades": len(closed),
        "wr": wr,
        "pf": pf,
        "final_eq": float(ev[-1]),
        "ret": ret,
        "mdd": mdd,
        "calmar": ret / max(abs(mdd), 0.1),
        "max_step": max_step_reached,
        "final_step": step,
        "cap_hits": cap_hits,
        "max_rec_days": max_rec_days,
    }


def main():
    print("Loading 36-month cached 5m futures data (2023-11-01 -> 2026-10-31)...")
    raw_m5 = load_3year_cached_m5()
    cfg = BacktestConfig()
    dfs_m15 = {s: resample_ohlcv(d, "15min") for s, d in raw_m5.items()}
    dfs_m30 = {s: resample_ohlcv(d, "30min") for s, d in raw_m5.items()}
    dfs_h1 = {s: resample_ohlcv(d, "1h") for s, d in raw_m5.items()}
    dfs_h4 = {s: resample_ohlcv(d, "4h") for s, d in raw_m5.items()}
    tf_dfs_map = {"5m": raw_m5, "15m": dfs_m15, "30m": dfs_m30, "1h": dfs_h1}

    u_arrays, _ = precompute_invariant_u_arrays(tf_dfs_map, dfs_h4)
    u3_arrays = u_arrays["U3_HTF_MACRO_ATR"]

    print("Generating 4-touch signals across 36M...")
    tf_sigs_map = {}
    for tfl, tdf, hdf, cd in [("5m", raw_m5, dfs_h1, 48), ("15m", dfs_m15, dfs_h1, 16), ("30m", dfs_m30, dfs_h4, 8), ("1h", dfs_h1, dfs_h4, 4)]:
        bst = compute_storm_flags(tdf["BTCUSDT"])
        bmask = build_cooldown_mask(bst["raw_storm"], cd)
        bts = tdf["BTCUSDT"]["timestamp"].values
        tf_sigs_map[tfl] = {
            sym: generate_all_4_touches(tdf[sym], hdf[sym], bts, bmask, sym, cfg.strategy, cooldown_bars=cd)[0]
            for sym in SIX_COINS
        }

    ts_2y_start = pd.Timestamp("2023-11-01 00:00:00", tz="UTC")
    ts_2y_end = pd.Timestamp("2025-10-31 23:59:59", tz="UTC")
    ts_y3_start = pd.Timestamp("2025-11-01 00:00:00", tz="UTC")
    ts_3y_end = pd.Timestamp("2026-10-31 23:59:59", tz="UTC")
    ts_y3_mid = pd.Timestamp("2026-04-30 23:59:59", tz="UTC")

    sigs_2y = filter_sigs(tf_sigs_map, ts_2y_start, ts_2y_end)
    sigs_y3 = filter_sigs(tf_sigs_map, ts_y3_start, ts_3y_end)
    sigs_36m = filter_sigs(tf_sigs_map, ts_2y_start, ts_3y_end)

    v30_rules = load_v30_rules_map()
    testc_rules = discover_dual_split_rules_arrays(tf_dfs_map, sigs_y3, "U3_HTF_MACRO_ATR", u3_arrays, ts_y3_mid, min_trades=8)

    configs = [
        ("S2_20_STEPS", 20, False),
        ("S2_50_STEPS_LIQ_CAP", 50, True),
        ("S2_50_STEPS_PURE", 50, False),
    ]
    experiments = [
        ("1_v30_58Setups_Full36M", v30_rules, sigs_36m, ts_2y_start, ts_3y_end),
        ("2_v30_58Setups_BlindY3", v30_rules, sigs_y3, ts_y3_start, ts_3y_end),
        ("3_TestC_59Setups_Blind2Y", testc_rules, sigs_2y, ts_2y_start, ts_2y_end),
    ]

    rows = []
    for ds_name, rmap, wsigs, w_start, w_end in experiments:
        for cfg_name, max_st, liq_cap in configs:
            for buf_name, turbo_buf in [("Bao_400G", None), ("Auto_Turbo_250G", 250.0)]:
                r = sim_s2_steps(
                    tf_dfs_map, wsigs, rmap, u3_arrays, max_st, liq_cap,
                    w_start, w_end, base_buffer=400.0, auto_turbo_buffer=turbo_buf,
                )
                rows.append({
                    "Dataset_Window": ds_name,
                    "Buffer_Mode": buf_name,
                    "Config": cfg_name,
                    "Trades": r["trades"],
                    "WR_%": round(r["wr"], 2),
                    "PF": round(r["pf"], 2),
                    "Return_%": round(r["ret"], 2),
                    "Final_Eq_$": round(r["final_eq"], 2),
                    "Max_Step": r["max_step"],
                    "Final_Step": r["final_step"],
                    "Cap_Hits": r["cap_hits"],
                    "MaxDD_%": round(r["mdd"], 2),
                    "Calmar": round(r["calmar"], 2),
                    "Max_Recovery_Days": round(r["max_rec_days"], 1),
                })

    df_out = pd.DataFrame(rows)
    out_csv = os.path.join(RESULTS_DIR, "step50_s2_check.csv")
    df_out.to_csv(out_csv, index=False)

    for ds_name in df_out["Dataset_Window"].unique():
        print("\n" + "=" * 136)
        print(f"50-STEP S2 CHECK: {ds_name}")
        print("=" * 136)
        print(df_out[df_out["Dataset_Window"] == ds_name].to_string(index=False))

    print(f"\nSaved results to: {out_csv}")


if __name__ == "__main__":
    main()
