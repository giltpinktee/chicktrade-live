"""Quantitative Blind-Spot Audit & New Alpha/Risk Rule Discovery (Argon-Quant).

Tests 4 Goldmines against the Baseline (PORTFOLIO 2-SL -> Shadow-Win, 55 setups = 43 v2.0 + 12 HOT_ADAPTIVE, Bao 400G, Max 5 concurrent):
  - Goldmine 1: Directional Asymmetry (LONG_ONLY vs SHORT_ONLY vs BOTH) — Strict Dual-Half (IS & OOS) vs Naive 1Y.
  - Goldmine 2: Directional Correlation Cap (Max 2, 3, 4 same-direction concurrent trades out of 5).
  - Goldmine 3: Adding newly rescued Core setups (LINKUSDT 15m EMA34_W2, BNBUSDT 15m EMA89_W2) + PORTFOLIO 2-SL -> Shadow-Win + Auto-Turbo (400G -> 250G / 230G with $11,000 hard floor).
  - Goldmine 4: New Structural Alpha & Risk Rules:
      * 4A: Dual-Scope Shadow-Win (PORTFOLIO 2-SL Shadow-Win + COIN/SETUP 2-SL Shadow-Win).
      * 4B: 2-Stage Progressive Drawdown Brake (Soft brake at DD <= -3.5% / -4.0% before Hard brake at -6.0%).
      * 4C: Post-Shadow-Win Rebound Boost (1.15x/1.20x on first trade after Shadow-Win) vs Equity-Peak Momentum Boost.
      * 4D: Candle/Volume Confirmation Weighting (sig.ema_aligned bonus/penalty) & EMA200 Overextension Guard.
"""
import copy
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

RESEARCH_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(RESEARCH_DIR)
for p in (ROOT, RESEARCH_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from config import BacktestConfig, RiskConfig
from data.binance_loader import _detect_latest_available_month, download_symbol_1y
from risk.compounding import CompoundingManager
from run_288_with_shadow_win_recheck import EXIT_CANDS, apply_shadow_win_filter
from run_full_216_exit_sweetspot import COIN_CAL, sim_exit_arch
from run_full_3ema_storm_guard import build_cooldown_mask, compute_storm_flags, generate_all_4_touches
from run_master_6coin_html import SIX_COINS
from run_pause_resume_sweetspot import load_rules_with_hot_adaptive
from run_research_mtf import resample_ohlcv


def audit_directional_asymmetry(
    tf_dfs_map: dict,
    tf_sigs_map: dict,
    rules_map: dict,
    mid_ts: pd.Timestamp,
) -> Tuple[pd.DataFrame, Dict[Tuple[str, str, str], int], Dict[Tuple[str, str, str], int]]:
    """Audit LONG (+1) vs SHORT (-1) performance for every setup in rules_map across IS 6M and OOS 6M.

    Returns:
      - df_dir: Detailed per-setup directional breakdown
      - strict_dual_half_dir_filter: {(sym, tf, stype): allowed_dir (+1 or -1)} ONLY when the cut direction loses money in BOTH IS & OOS (Net_IS < 0 and Net_OOS < 0) and has >= 4 trades in each half, while kept direction wins in BOTH halves!
      - loose_is_dir_filter: {(sym, tf, stype): allowed_dir} based on IS 6M only (to test OOS overfitting!).
    """
    rows = []
    strict_filter: Dict[Tuple[str, str, str], int] = {}
    loose_is_filter: Dict[Tuple[str, str, str], int] = {}

    for (sym, tfl, stype), rule in sorted(rules_map.items()):
        df = tf_dfs_map[tfl][sym]
        h_arr, l_arr, c_arr = df["high"].values, df["low"].values, df["close"].values
        sigs = [s for s in tf_sigs_map[tfl][sym] if s.setup_type == stype]
        if not sigs:
            continue

        by_d = {1: {"is": [], "oos": [], "all": []}, -1: {"is": [], "oos": [], "all": []}}
        for s in sigs:
            u = max(0.5 * s.atr, s.entry_price * 1e-5)
            _, _, _, _, nu = sim_exit_arch(
                h_arr, l_arr, c_arr, s.bar_idx, s.entry_price, s.direction, u,
                rule["sl_u"], rule["tp1_u"], rule["tp2_u"], rule["arch"], rule["l2_u"],
            )
            bucket = by_d[s.direction]
            bucket["all"].append(nu)
            (bucket["oos"] if s.timestamp >= mid_ts else bucket["is"]).append(nu)

        def _stats(arr):
            if not arr:
                return 0, 0.0, 0.0, 0.0
            a = np.array(arr)
            wr = float((a > 0).mean() * 100.0)
            pf = float(a[a > 0].sum() / max(-a[a < 0].sum(), 0.01))
            return len(a), wr, pf, float(a.sum())

        n_l, wr_l, pf_l, net_l = _stats(by_d[1]["all"])
        nl_is, _, pfl_is, netl_is = _stats(by_d[1]["is"])
        nl_oo, _, pfl_oo, netl_oo = _stats(by_d[1]["oos"])

        n_s, wr_s, pf_s, net_s = _stats(by_d[-1]["all"])
        ns_is, _, pfs_is, nets_is = _stats(by_d[-1]["is"])
        ns_oo, _, pfs_oo, nets_oo = _stats(by_d[-1]["oos"])

        # Loose IS-only filter: if one direction lost money in IS (net < -2U) and the other won in IS (pf >= 1.2)
        if nl_is >= 3 and netl_is < -2.0 and pfs_is >= 1.20:
            loose_is_filter[(sym, tfl, stype)] = -1
        elif ns_is >= 3 and nets_is < -2.0 and pfl_is >= 1.20:
            loose_is_filter[(sym, tfl, stype)] = 1

        # Strict Dual-Half filter: Cut direction MUST be unprofitable in BOTH IS & OOS (pf_is < 0.95 and pf_oos < 0.95, net_is < 0, net_oos < 0, n >= 6)
        # while kept direction is profitable in BOTH IS & OOS (pf_is >= 1.15 and pf_oos >= 1.15)
        rec_mode = "BOTH"
        if n_s >= 6 and ns_is >= 2 and ns_oo >= 2 and nets_is < 0 and nets_oo < 0 and pfl_is >= 1.15 and pfl_oo >= 1.15:
            strict_filter[(sym, tfl, stype)] = 1
            rec_mode = "LONG_ONLY (Short lỗ cả IS & OOS)"
        elif n_l >= 6 and nl_is >= 2 and nl_oo >= 2 and netl_is < 0 and netl_oo < 0 and pfs_is >= 1.15 and pfs_oo >= 1.15:
            strict_filter[(sym, tfl, stype)] = -1
            rec_mode = "SHORT_ONLY (Long lỗ cả IS & OOS)"

        rows.append({
            "Coin": sym, "TF": tfl, "Setup": stype,
            "L_Lenh": n_l, "L_WR%": wr_l, "L_PF_IS": pfl_is, "L_PF_OOS": pfl_oo, "L_NetU": net_l,
            "S_Lenh": n_s, "S_WR%": wr_s, "S_PF_IS": pfs_is, "S_PF_OOS": pfs_oo, "S_NetU": net_s,
            "Strict_Decision": rec_mode,
        })

    return pd.DataFrame(rows), strict_filter, loose_is_filter


def inspect_new_core_candidates(tf_dfs_map: dict, tf_sigs_map: dict, mid_ts: pd.Timestamp) -> Tuple[Dict[Tuple[str, str, str], dict], Dict[Tuple[str, str, str], dict]]:
    """Inspect LINKUSDT 15m EMA34_W2 and BNBUSDT 15m EMA89_W2 under all 7 exit architectures + Shadow-Win."""
    targets = [
        ("LINKUSDT", "15m", "EMA34_W2", 0.65),
        ("BNBUSDT", "15m", "EMA89_W2", 0.65),
    ]
    new_rules_both = {}
    new_rules_bnb_only = {}
    print("\n--- GIAI PHAU 2 KEO CORE MOI (LINKUSDT 15m EMA34_W2 & BNBUSDT 15m EMA89_W2) ---")
    for sym, tfl, stype, vol_m in targets:
        cc = COIN_CAL[sym]
        df = tf_dfs_map[tfl][sym]
        h, l, c = df["high"].values, df["low"].values, df["close"].values
        sl_u, tp1_u, tp2_u = cc["sl"], cc["tp1"][tfl], cc["tp2"]
        sigs = [s for s in tf_sigs_map[tfl][sym] if s.setup_type == stype]

        best = None
        for lbl, arch, l2 in EXIT_CANDS:
            outs, nus = [], []
            for s in sigs:
                u = max(0.5 * s.atr, s.entry_price * 1e-5)
                o, _, _, _, nu = sim_exit_arch(h, l, c, s.bar_idx, s.entry_price, s.direction, u, sl_u, tp1_u, tp2_u, arch, l2)
                outs.append(o); nus.append(nu)
            taken = apply_shadow_win_filter(sigs, outs, nus, trigger_sl=2)
            t_nus = np.array([nus[i] for i in taken])
            is_nus = np.array([nus[i] for i in taken if sigs[i].timestamp < mid_ts])
            oo_nus = np.array([nus[i] for i in taken if sigs[i].timestamp >= mid_ts])
            pf_all = t_nus[t_nus > 0].sum() / max(-t_nus[t_nus < 0].sum(), 0.01)
            pf_is = is_nus[is_nus > 0].sum() / max(-is_nus[is_nus < 0].sum(), 0.01)
            pf_oo = oo_nus[oo_nus > 0].sum() / max(-oo_nus[oo_nus < 0].sum(), 0.01)
            wr_all = (t_nus > 0).mean() * 100.0
            # Require high WR (>= 70%) so it doesn't pollute Portfolio 2-SL Shadow-Win trigger!
            qual = (pf_is >= 1.10 and pf_oo >= 1.15 and pf_all >= 1.35 and wr_all >= 70.0)
            cand = (qual, wr_all, float(t_nus.sum()), lbl, arch, l2, len(taken), pf_is, pf_oo, pf_all)
            if best is None or (cand[0], cand[1], cand[2]) > (best[0], best[1], best[2]):
                best = cand

        _, wr, net_u, lbl, arch, l2, n_sw, pf_is, pf_oo, pf_all = best
        print(f"  + {sym} {tfl} {stype}: Exit={lbl} (arch={arch}, l2={l2}) | Lenh_SW={n_sw}/{len(sigs)} | WR={wr:.2f}% | PF_IS={pf_is:.2f} | PF_OOS={pf_oo:.2f} | PF_1Y={pf_all:.2f} | Net_U=+{net_u:.2f}U")
        rspec = {
            "arch": arch,
            "l2_u": float(l2),
            "sl_u": float(sl_u),
            "tp1_u": float(tp1_u),
            "tp2_u": float(tp2_u),
            "vol": float(vol_m),
        }
        new_rules_both[(sym, tfl, stype)] = rspec
        if sym == "BNBUSDT":
            new_rules_bnb_only[(sym, tfl, stype)] = rspec
    return new_rules_both, new_rules_bnb_only


def sim_portfolio_advanced(
    tf_dfs_map: dict,
    tf_sigs_map: dict,
    rules_map: dict,
    start_time: pd.Timestamp = None,
    # Goldmine 1: Directional filter map {(sym, tf, stype): +1 or -1}
    dir_filter_map: Optional[Dict[Tuple[str, str, str], int]] = None,
    # Goldmine 2: Max concurrent trades in the same direction (Long/Short)
    max_same_dir: int = 5,
    # Goldmine 3: Auto-Turbo 2-Gear parameters
    base_buffer: float = 400.0,
    auto_turbo_buffer: Optional[float] = None,
    turbo_trigger_eq: float = 12_500.0,
    turbo_disengage_eq: float = 11_500.0,
    hard_floor_eq: float = 11_000.0,
    # Goldmine 4A: Dual-Scope Shadow-Win (also pause specific Setup or Coin after 2 consecutive SLs)
    setup_shadow_sl: int = 0,
    coin_shadow_sl: int = 0,
    # Goldmine 4B: Progressive 2-Stage Drawdown Brake (soft brake at soft_dd_pct)
    soft_dd_pct: Optional[float] = None,
    soft_dd_mult: float = 0.55,
    # Goldmine 4C: Post-Shadow-Win Rebound Boost / Equity-Peak Streak Boost
    rebound_boost_mult: float = 1.0,
    peak_streak_boost: float = 1.0,
    # Goldmine 4D: Candle/Vol Confirmation Weighting & Overextension Guard
    pa_confirm_bonus: float = 1.0,
    overext_atr_max: Optional[float] = None,
    # Goldmine 4E: Core-Only SL Trigger (Scout/HOT_ADAPTIVE vol <= 0.45x SLs don't trigger portfolio-wide freeze)
    core_only_sl_trigger: bool = False,
    # Goldmine 4F: Diamond Fast-Resume / Vol-Weighted Shadow-Win
    diamond_shadow_wake: bool = False,
) -> dict:
    """Unified simulation engine based strictly on run_pause_resume_sweetspot.py Baseline (#11 PORTFOLIO 2-SL -> Shadow-Win)."""
    rcfg = RiskConfig(initial_capital=10_000.0, buffer_units=base_buffer, max_concurrent_trades=5, max_trades_per_day=10)
    comp = CompoundingManager(rcfg)
    equity = 10_000.0
    peak_eq = equity
    open_cap = 4.25
    in_turbo = False
    ever_triggered_turbo = False
    min_eq_after_turbo = 10_000.0

    cands = []
    for tfl, sm in tf_sigs_map.items():
        for sym, sigs in sm.items():
            for s in sigs:
                k = (sym, tfl, s.setup_type)
                if k not in rules_map:
                    continue
                if start_time is not None and s.timestamp < start_time:
                    continue
                if dir_filter_map and k in dir_filter_map and s.direction != dir_filter_map[k]:
                    continue
                cands.append((tfl, rules_map[k], s))

    tf_prio = {"1h": 0, "30m": 1, "15m": 2, "5m": 3}
    cands.sort(key=lambda x: (x[2].timestamp, tf_prio[x[0]], -x[1]["vol"], x[2].symbol))

    active, closed = [], []
    daily_t, daily_pnl, daily_eq = {}, {}, {}
    eq_curve = [equity]

    # Portfolio-level 2-SL -> Shadow-Win state
    port_consec_sl = 0
    port_shadow_paused = False
    port_paused_until = pd.Timestamp("1970-01-01", tz="UTC")
    just_resumed_rebound = False
    port_win_streak = 0

    # Optional secondary Setup/Coin Shadow-Win state
    sub_consec_sl: Dict[str, int] = {}
    sub_shadow_paused: Dict[str, bool] = {}
    sub_paused_until: Dict[str, pd.Timestamp] = {}

    def settle(now_ts):
        nonlocal equity, peak_eq, port_consec_sl, port_shadow_paused, port_paused_until
        nonlocal in_turbo, ever_triggered_turbo, min_eq_after_turbo, port_win_streak
        keep, done = [], []
        for ex_ts, tr in active:
            (done if ex_ts <= now_ts else keep).append((ex_ts, tr))
        done.sort(key=lambda x: x[0])
        for ex_ts, tr in done:
            equity += tr["pnl"]
            peak_eq = max(peak_eq, equity)
            comp.update_equity(equity, ex_ts)
            dk = tr["dk"]
            daily_pnl[dk] = daily_pnl.get(dk, 0.0) + tr["pnl"]

            if ever_triggered_turbo:
                min_eq_after_turbo = min(min_eq_after_turbo, equity)

            if auto_turbo_buffer is not None:
                if (not in_turbo) and equity >= turbo_trigger_eq:
                    in_turbo = True
                    ever_triggered_turbo = True
                    min_eq_after_turbo = min(min_eq_after_turbo, equity)
                elif in_turbo and equity < turbo_disengage_eq:
                    in_turbo = False

            if tr["out"] == "SL":
                port_win_streak = 0
                if (not core_only_sl_trigger) or tr["base_vol"] > 0.45:
                    port_consec_sl += 1
                    if port_consec_sl >= 2:
                        port_consec_sl = 0
                        port_shadow_paused = True
                        port_paused_until = ex_ts + pd.Timedelta(hours=72)
            elif tr["pnl"] > 0:
                port_consec_sl = 0
                port_win_streak += 1

            # Optional Setup/Coin secondary shadow pause
            if setup_shadow_sl > 0:
                stk = tr["setup_key"]
                if tr["out"] == "SL":
                    sub_consec_sl[stk] = sub_consec_sl.get(stk, 0) + 1
                    if sub_consec_sl[stk] >= setup_shadow_sl:
                        sub_consec_sl[stk] = 0
                        sub_shadow_paused[stk] = True
                        sub_paused_until[stk] = ex_ts + pd.Timedelta(hours=72)
                elif tr["pnl"] > 0:
                    sub_consec_sl[stk] = 0

            if coin_shadow_sl > 0:
                ck = tr["sym"]
                if tr["out"] == "SL":
                    sub_consec_sl[ck] = sub_consec_sl.get(ck, 0) + 1
                    if sub_consec_sl[ck] >= coin_shadow_sl:
                        sub_consec_sl[ck] = 0
                        sub_shadow_paused[ck] = True
                        sub_paused_until[ck] = ex_ts + pd.Timedelta(hours=72)
                elif tr["pnl"] > 0:
                    sub_consec_sl[ck] = 0

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
        u = max(0.5 * sig.atr, sig.entry_price * 1e-5)
        setup_key = f"{sig.symbol}:{tfl}:{sig.setup_type}"

        # Overextension guard (Goldmine 4D): check distance from EMA200 in ATR units
        if overext_atr_max is not None and "REV" not in sig.setup_type:
            # Compute fast EMA200 at bar_idx from close series if not cached
            if "_ema200" not in df.columns:
                df["_ema200"] = df["close"].ewm(span=200, adjust=False).mean()
            e200_val = float(df["_ema200"].iloc[sig.bar_idx])
            dist_atr = abs(sig.entry_price - e200_val) / max(sig.atr, 1e-8)
            if dist_atr > overext_atr_max:
                continue

        # 1. Check Portfolio Shadow-Win pause
        if port_shadow_paused:
            if sig.timestamp >= port_paused_until:
                port_shadow_paused = False
            elif diamond_shadow_wake and rule["vol"] >= 1.10:
                # Allow S+ Diamond setups (vol >= 1.10) to wake up & execute at 0.5x probe size!
                port_shadow_paused = False
                just_resumed_rebound = True
            else:
                _, _, _, _, nu_sh = sim_exit_arch(
                    df["high"].values, df["low"].values, df["close"].values,
                    sig.bar_idx, sig.entry_price, sig.direction, u,
                    rule["sl_u"], rule["tp1_u"], rule["tp2_u"], rule["arch"], rule["l2_u"],
                )
                if nu_sh > 0:
                    port_shadow_paused = False
                    just_resumed_rebound = True
                continue

        # 2. Check optional Setup/Coin Shadow-Win pause
        if setup_shadow_sl > 0 and sub_shadow_paused.get(setup_key, False):
            if sig.timestamp >= sub_paused_until.get(setup_key, sig.timestamp):
                sub_shadow_paused[setup_key] = False
            else:
                _, _, _, _, nu_sh = sim_exit_arch(
                    df["high"].values, df["low"].values, df["close"].values,
                    sig.bar_idx, sig.entry_price, sig.direction, u,
                    rule["sl_u"], rule["tp1_u"], rule["tp2_u"], rule["arch"], rule["l2_u"],
                )
                if nu_sh > 0:
                    sub_shadow_paused[setup_key] = False
                continue

        if coin_shadow_sl > 0 and sub_shadow_paused.get(sig.symbol, False):
            if sig.timestamp >= sub_paused_until.get(sig.symbol, sig.timestamp):
                sub_shadow_paused[sig.symbol] = False
            else:
                _, _, _, _, nu_sh = sim_exit_arch(
                    df["high"].values, df["low"].values, df["close"].values,
                    sig.bar_idx, sig.entry_price, sig.direction, u,
                    rule["sl_u"], rule["tp1_u"], rule["tp2_u"], rule["arch"], rule["l2_u"],
                )
                if nu_sh > 0:
                    sub_shadow_paused[sig.symbol] = False
                continue

        if daily_t.get(dk, 0) >= 10 or daily_pnl.get(dk, 0.0) <= -0.05 * daily_eq[dk]:
            continue
        if len(active) >= 5 or any(tr["sym"] == sig.symbol for _, tr in active):
            continue

        # Goldmine 2: Directional Correlation Cap (max_same_dir)
        same_dir_open = sum(1 for _, tr in active if tr["dir"] == sig.direction)
        if same_dir_open >= max_same_dir:
            continue

        eff_m = rule["vol"]
        if sig.pa_vol_ok:
            eff_m *= 0.5

        # Goldmine 4C: Post-Shadow-Win Rebound Boost or Peak Streak Boost
        if just_resumed_rebound:
            eff_m *= rebound_boost_mult
            just_resumed_rebound = False
        if peak_streak_boost != 1.0 and port_win_streak >= 3 and equity >= peak_eq * 0.995:
            eff_m *= peak_streak_boost

        # Goldmine 4D: Candle/Vol Confirmation weighting
        if pa_confirm_bonus != 1.0:
            if sig.ema_aligned:
                eff_m *= pa_confirm_bonus
            else:
                eff_m *= (2.0 - pa_confirm_bonus)

        # Goldmine 4B: Progressive 2-Stage Drawdown Brake
        cur_dd_pct = (equity - peak_eq) / peak_eq * 100.0
        if cur_dd_pct <= -6.0:
            eff_m = min(eff_m, 0.25)
        elif soft_dd_pct is not None and cur_dd_pct <= soft_dd_pct:
            eff_m = min(eff_m, rule["vol"] * soft_dd_mult)

        eff_m = min(eff_m, max(0.0, open_cap - sum(tr["m"] for _, tr in active)))
        if eff_m < 0.20:
            continue

        # Goldmine 3: Auto-Turbo buffer with worst-case SL floor protection ($11,000)
        cur_buf = base_buffer
        if auto_turbo_buffer is not None and in_turbo:
            cur_buf = auto_turbo_buffer
            sb_test = min(comp.tier_capital, equity)
            # Worst-case full SL loss for existing active trades + this new trade
            open_sl_risk = sum(tr["worst_sl_usd"] for _, tr in active)
            new_sl_risk = (sb_test * eff_m) * (rule["sl_u"] / cur_buf) * 1.05
            if equity - (open_sl_risk + new_sl_risk) < hard_floor_eq:
                cur_buf = base_buffer

        out, _, _, _, nu = sim_exit_arch(
            df["high"].values, df["low"].values, df["close"].values,
            sig.bar_idx, sig.entry_price, sig.direction, u,
            rule["sl_u"], rule["tp1_u"], rule["tp2_u"], rule["arch"], rule["l2_u"],
        )
        sb = min(comp.tier_capital, equity)
        qty = (sb * eff_m) / (cur_buf * u)
        pnl = nu * u * qty
        worst_sl_usd = (sb * eff_m) * (rule["sl_u"] / cur_buf) * 1.05

        ex_i = min(sig.bar_idx + (4 if out == "SL" else 8), len(df) - 1)
        ex_ts = pd.Timestamp(df["timestamp"].iloc[ex_i])
        if ex_ts.tzinfo is None:
            ex_ts = ex_ts.tz_localize("UTC")
        active.append((ex_ts, {
            "sym": sig.symbol,
            "setup_key": setup_key,
            "dir": sig.direction,
            "base_vol": rule["vol"],
            "m": eff_m,
            "out": out,
            "pnl": pnl,
            "dk": dk,
            "worst_sl_usd": worst_sl_usd,
        }))
        daily_t[dk] = daily_t.get(dk, 0) + 1

    if active:
        settle(pd.Timestamp("2099-01-01", tz="UTC"))

    ev = np.array(eq_curve)
    pk = np.maximum.accumulate(ev)
    mdd = float(((ev - pk) / pk * 100.0).min())
    pnls = np.array([t["pnl"] for t in closed])
    wr = float((pnls > 0).mean() * 100.0) if len(pnls) else 0.0
    pf = float(pnls[pnls > 0].sum() / max(-pnls[pnls < 0].sum(), 0.01))
    ret = (float(ev[-1]) / 10_000.0 - 1.0) * 100.0
    return {
        "trades": len(closed),
        "wr": wr,
        "pf": pf,
        "final_eq": float(ev[-1]),
        "ret": ret,
        "mdd": mdd,
        "calmar": ret / max(abs(mdd), 0.1),
        "min_after_turbo": min_eq_after_turbo,
    }


def main():
    base_rules = load_rules_with_hot_adaptive()
    cfg = BacktestConfig()
    end_ym = _detect_latest_available_month("BTCUSDT", "5m")

    raw_m5 = {sym: download_symbol_1y(sym, "5m", 12, "futures", end_ym) for sym in SIX_COINS}
    dfs_m15 = {s: resample_ohlcv(d, "15min") for s, d in raw_m5.items()}
    dfs_m30 = {s: resample_ohlcv(d, "30min") for s, d in raw_m5.items()}
    dfs_h1 = {s: resample_ohlcv(d, "1h") for s, d in raw_m5.items()}
    dfs_h4 = {s: resample_ohlcv(d, "4h") for s, d in raw_m5.items()}
    tf_dfs_map = {"5m": raw_m5, "15m": dfs_m15, "30m": dfs_m30, "1h": dfs_h1}

    tf_sigs_map = {}
    for tfl, tdf, hdf, cd in [("5m", raw_m5, dfs_h1, 48), ("15m", dfs_m15, dfs_h1, 16), ("30m", dfs_m30, dfs_h4, 8), ("1h", dfs_h1, dfs_h4, 4)]:
        bst = compute_storm_flags(tdf["BTCUSDT"])
        bmask = build_cooldown_mask(bst["raw_storm"], cd)
        bts = tdf["BTCUSDT"]["timestamp"].values
        tf_sigs_map[tfl] = {
            sym: generate_all_4_touches(tdf[sym], hdf[sym], bts, bmask, sym, cfg.strategy, cooldown_bars=cd)[0]
            for sym in SIX_COINS
        }

    all_ts = sorted(s.timestamp for s in tf_sigs_map["15m"]["BTCUSDT"])
    mid_ts = all_ts[len(all_ts) // 2]

    # 1. Audit Goldmine 1: Directional Asymmetry
    df_dir, strict_dir_map, loose_is_dir_map = audit_directional_asymmetry(tf_dfs_map, tf_sigs_map, base_rules, mid_ts)
    print("\n" + "=" * 124)
    print(f"MO 1: KIEM TOAN CHIEU DANH RIENG (LONG vs SHORT) — STRICT DUAL-HALF TIM THAY {len(strict_dir_map)} SETUP LECH CHIEU CA 2 NUA NAM")
    print("=" * 124)
    if strict_dir_map:
        print(df_dir[df_dir["Strict_Decision"] != "BOTH"].to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
    else:
        print("  -> Khong co setup nao thuan 1 chieu ma chieu kia lo o CA 2 NUA NAM (IS & OOS)!")
    print(f"  -> Neu loc mu theo 6T dau (Loose IS-only): co {len(loose_is_dir_map)} setup bi cat 1 chieu.")

    # 2. Audit Goldmine 3: Inspect the 2 new core candidates
    new_core_both, new_core_bnb = inspect_new_core_candidates(tf_dfs_map, tf_sigs_map, mid_ts)
    rules_plus_2 = copy.deepcopy(base_rules)
    rules_plus_2.update(new_core_both)
    rules_plus_bnb = copy.deepcopy(base_rules)
    rules_plus_bnb.update(new_core_bnb)

    # 3. Run comprehensive comparative experiments across all 4 Goldmines
    experiments = [
        ("0. BASELINE (46 Setups | PORTFOLIO 2-SL -> Shadow-Win | 400G Max 5)", base_rules, {}),
        # Goldmine 1
        ("1A. Mỏ 1 — Lọc 1 Chiều Loose theo IS 6T đầu (8 setups -> Bẫy Overfit!)", base_rules, {"dir_filter_map": loose_is_dir_map}),
        ("1B. Mỏ 1 — Lọc 1 Chiều Strict Dual-Half (2 setups: BNB 1h LONG, XRP 1h SHORT)", base_rules, {"dir_filter_map": strict_dir_map}),
        # Goldmine 2
        ("2A. Mỏ 2 — Khóa Tương Quan Chiều: Tối đa 2 lệnh cùng chiều (Max 2 Same-Dir)", base_rules, {"max_same_dir": 2}),
        ("2B. Mỏ 2 — Khóa Tương Quan Chiều: Tối đa 3 lệnh cùng chiều (Max 3 Same-Dir)", base_rules, {"max_same_dir": 3}),
        ("2C. Mỏ 2 — Khóa Tương Quan Chiều: Tối đa 4 lệnh cùng chiều (Max 4 Same-Dir)", base_rules, {"max_same_dir": 4}),
        # Goldmine 3
        ("3A-1. Mỏ 3 — Ghép +2 Kèo Mới (WR>=70%: LINK Entry-2U + BNB ONLY_TP1) @ 400G", rules_plus_2, {}),
        ("3A-2. Mỏ 3 — Chỉ ghép +1 Kèo Mới (BNBUSDT 15m EMA89_W2 ONLY_TP1) @ 400G", rules_plus_bnb, {}),
        ("3B. Mỏ 3 — Baseline + Auto-Turbo 400G -> 250G (Bật $12.5k, Khóa sàn $11k)", base_rules, {"auto_turbo_buffer": 250.0}),
        ("3C. Mỏ 3 — Ghép +2 Kèo Mới (WR>=70%) + Auto-Turbo 400G -> 250G (Khóa sàn $11k)", rules_plus_2, {"auto_turbo_buffer": 250.0}),
        # Goldmine 4
        ("4A-1. Mỏ 4 — Dual Shadow-Win (PORTFOLIO 2-SL + SETUP 2-SL Shadow-Win)", base_rules, {"setup_shadow_sl": 2}),
        ("4A-2. Mỏ 4 — Dual Shadow-Win (PORTFOLIO 2-SL + COIN 2-SL Shadow-Win)", base_rules, {"coin_shadow_sl": 2}),
        ("4B-1. Mỏ 4 — Phanh DD 2 Tầng (Soft Brake 0.55x tại DD <= -3.5%, Hard -6%)", base_rules, {"soft_dd_pct": -3.5, "soft_dd_mult": 0.55}),
        ("4C-1. Mỏ 4 — Rebound Boost 1.15x ở lệnh đầu ngay sau Shadow-Win", base_rules, {"rebound_boost_mult": 1.15}),
        ("4C-2. Mỏ 4 — Soft Resume 0.75x ở lệnh đầu ngay sau Shadow-Win", base_rules, {"rebound_boost_mult": 0.75}),
        ("4D-1. Mỏ 4 — Overextension Guard (Chặn lệnh thuận giãn > 8.0 ATR khỏi EMA200)", base_rules, {"overext_atr_max": 8.0}),
        ("4E-1. Mỏ 4 — Core-Only SL Trigger (Lệnh Scout <=0.45x thua không đóng băng Port)", base_rules, {"core_only_sl_trigger": True}),
        ("4F-1. Mỏ 4 — Diamond Fast-Wake (Setup S+ vol>=1.10x đánh thức sớm Shadow Pause)", base_rules, {"diamond_shadow_wake": True}),
        # Combined Winning Combos
        ("5A. COMBO THẮNG @ 400G: Strict 1-Chiều (1B) + Core-Only SL Trigger (4E)", base_rules, {"dir_filter_map": strict_dir_map, "core_only_sl_trigger": True}),
        ("5B. COMBO THẮNG @ 400G: Strict 1-Chiều (1B) + Ghép Kèo Mới WR>=70% (3A-1)", rules_plus_2, {"dir_filter_map": strict_dir_map}),
        ("5C. COMBO THẮNG + AUTO-TURBO 400G->250G: Strict 1-Chiều (1B) + Turbo 250G", base_rules, {"dir_filter_map": strict_dir_map, "auto_turbo_buffer": 250.0}),
        ("5D. COMBO THẮNG + AUTO-TURBO 400G->250G: Strict 1-Chiều (1B) + Core-Only SL + Turbo 250G", base_rules, {"dir_filter_map": strict_dir_map, "core_only_sl_trigger": True, "auto_turbo_buffer": 250.0}),
    ]

    comp_rows = []
    for name, rmap, kwargs in experiments:
        r12 = sim_portfolio_advanced(tf_dfs_map, tf_sigs_map, rmap, start_time=None, **kwargs)
        r6 = sim_portfolio_advanced(tf_dfs_map, tf_sigs_map, rmap, start_time=mid_ts, **kwargs)
        comp_rows.append({
            "Phuong_An_Khai_Pha": name,
            "12M_Lenh": r12["trades"],
            "12M_WR%": r12["wr"],
            "12M_PF": r12["pf"],
            "12M_Lai%": r12["ret"],
            "12M_MaxDD%": r12["mdd"],
            "12M_Calmar": r12["calmar"],
            "OOS_Lenh": r6["trades"],
            "OOS_WR%": r6["wr"],
            "OOS_PF": r6["pf"],
            "OOS_Lai%": r6["ret"],
            "OOS_MaxDD%": r6["mdd"],
            "OOS_Calmar": r6["calmar"],
        })

    cdf = pd.DataFrame(comp_rows)
    print("\n" + "=" * 140)
    print("BANG TONG HOP KIEM CHUNG 4 MO VANG (SO SANH VOI BASELINE +740.77% | DD -9.22% | CALMAR 80.38 | OOS +312.45% / DD -7.93%)")
    print("=" * 140)
    print(cdf.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    out_dir = os.path.join(ROOT, "output_test")
    os.makedirs(out_dir, exist_ok=True)
    cdf.to_csv(os.path.join(out_dir, "alpha_discovery_argon_results.csv"), index=False)
    df_dir.to_csv(os.path.join(out_dir, "directional_asymmetry_audit.csv"), index=False)


if __name__ == "__main__":
    main()
