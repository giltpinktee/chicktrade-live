"""3-Year Per-Coin Percentage 1U Study (2023-11-01 -> 2026-10-31).

Addresses the cross-coin volatility heterogeneity issue in M1_PCT_TIGHT (where BTC and
altcoins like LINK/SOL shared a single flat % per timeframe). Computes empirical median
1U_pct[coin][tf] = median(0.5 * ATR(14) / close) strictly on each experiment's Train
window (zero lookahead!) and tests 0.9x, 1.0x, 1.2x scales against M1_PCT_TIGHT and M0_PURE_ATR.

Test A: Frozen 75 setups (v2.3) + Rule 1B, evaluated on 2Y Blind (11/2023->10/2025) & 1Y Recent (11/2025->10/2026).
        (Per-coin 1U_pct measured on 75-setup Train year 11/2025->10/2026 so 2Y Blind is zero-lookahead.)
Test B: True Multi-Year Walk-Forward (Train 2Y 11/2023->10/2025 -> Blind Test Year 3 11/2025->10/2026 + 36M).
        (Per-coin 1U_pct measured strictly on Train 2Y 11/2023->10/2025 so Year 3 is 100% blind.)

Outputs:
  - e:\CHICKTRADE\verified_tests\results\per_coin_pct_3year_results.csv
"""
import os
import sys
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
VERIFIED_DIR = os.path.dirname(SCRIPTS_DIR)
ROOT = os.path.dirname(VERIFIED_DIR)
for p in (ROOT, SCRIPTS_DIR, os.path.join(ROOT, "old", "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from config import BacktestConfig, RiskConfig
from risk.compounding import CompoundingManager
from run_288_with_shadow_win_recheck import EXIT_CANDS
from run_full_216_exit_sweetspot import COIN_CAL, sim_exit_arch
from run_full_3ema_storm_guard import build_cooldown_mask, compute_storm_flags, generate_all_4_touches
from run_master_6coin_html import SIX_COINS
from run_pct_vs_atr_3year_study import PCT_MAPS, filter_sigs, load_3year_cached_m5, load_frozen_75
from run_research_mtf import resample_ohlcv

RESULTS_DIR = os.path.join(VERIFIED_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def measure_per_coin_u_pct(
    tf_dfs_map: dict,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> Dict[str, Dict[str, float]]:
    """Measure median (0.5 * ATR(14) / close) strictly inside [start_ts, end_ts] for each (coin, tf)."""
    table: Dict[str, Dict[str, float]] = {sym: {} for sym in SIX_COINS}
    for tfl in ("5m", "15m", "30m", "1h"):
        for sym in SIX_COINS:
            df = tf_dfs_map[tfl][sym]
            high = df["high"]
            low = df["low"]
            close = df["close"]
            tr = pd.concat(
                [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
                axis=1,
            ).max(axis=1)
            atr14 = tr.ewm(alpha=1.0 / 14, adjust=False).mean()
            u_pct_series = (0.5 * atr14) / close
            mask = (df["timestamp"] >= start_ts) & (df["timestamp"] <= end_ts)
            med_u_pct = float(u_pct_series.loc[mask].dropna().median())
            table[sym][tfl] = med_u_pct
    return table


def compute_u_custom(
    mode_name: str,
    sym: str,
    tfl: str,
    price: float,
    atr: float,
    per_coin_table: Optional[Dict[str, Dict[str, float]]] = None,
) -> float:
    if mode_name == "M0_PURE_ATR":
        return max(0.5 * atr, price * 1e-5)
    if mode_name == "M1_PCT_TIGHT":
        return max(price * PCT_MAPS["M1_PCT_TIGHT"][tfl], price * 1e-5)
    # PER_COIN_PCT_<scale>x
    scale = float(mode_name.split("_")[-1].replace("x", ""))
    return max(price * per_coin_table[sym][tfl] * scale, price * 1e-5)


def sim_portfolio_custom_u(
    tf_dfs_map: dict,
    tf_sigs_map: dict,
    rules_map: dict,
    mode_name: str,
    per_coin_table: Optional[Dict[str, Dict[str, float]]] = None,
    dir_filter_map: Optional[Dict[Tuple[str, str, str], int]] = None,
    base_buffer: float = 400.0,
    auto_turbo_buffer: Optional[float] = None,
    hard_floor_eq: float = 11_000.0,
    core_only_sl_trigger: bool = True,
) -> dict:
    rcfg = RiskConfig(initial_capital=10_000.0, buffer_units=base_buffer, max_concurrent_trades=5, max_trades_per_day=10)
    comp = CompoundingManager(rcfg)
    equity = 10_000.0
    peak_eq = equity
    open_cap = 4.25
    in_turbo = False

    cands = []
    for tfl, sm in tf_sigs_map.items():
        for sym, sigs in sm.items():
            for s in sigs:
                k = (sym, tfl, s.setup_type)
                if k not in rules_map:
                    continue
                if dir_filter_map and k in dir_filter_map and s.direction != dir_filter_map[k]:
                    continue
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
        nonlocal equity, peak_eq, port_consec_sl, port_shadow_paused, port_paused_until, in_turbo
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
        u = compute_u_custom(mode_name, sig.symbol, tfl, sig.entry_price, sig.atr, per_coin_table)

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

        cur_buf = base_buffer
        if auto_turbo_buffer is not None and in_turbo:
            cur_buf = auto_turbo_buffer
            sb_test = min(comp.tier_capital, equity)
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
    }


def discover_dual_split_rules_custom(
    tf_dfs_map: dict,
    train_sigs: dict,
    mode_name: str,
    per_coin_table: Optional[Dict[str, Dict[str, float]]],
    split_ts: pd.Timestamp,
    min_trades: int = 10,
) -> dict:
    """Scan all 216 setups (15m/30m/1h) on train_sigs, picking best exit_arch/l2_u passing both halves."""
    discovered = {}
    for sym in SIX_COINS:
        cc = COIN_CAL[sym]
        sl_u = float(cc["sl"])
        tp2_u = float(cc["tp2"])
        for tfl in ("15m", "30m", "1h"):
            tp1_u = float(cc["tp1"][tfl])
            df = tf_dfs_map[tfl][sym]
            h_arr, l_arr, c_arr = df["high"].values, df["low"].values, df["close"].values
            sigs_by_setup = {}
            for s in train_sigs[tfl][sym]:
                sigs_by_setup.setdefault(s.setup_type, []).append(s)

            for stype, sigs in sorted(sigs_by_setup.items()):
                if len(sigs) < min_trades:
                    continue
                best_qual = None
                for _, arch, l2_u in EXIT_CANDS:
                    if arch == "PARTIAL_MOVE_SL" and l2_u < 0 and tp1_u <= abs(l2_u) + 0.6:
                        continue
                    h1_u, h2_u, all_u = [], [], []
                    for s in sigs:
                        u = compute_u_custom(mode_name, sym, tfl, s.entry_price, s.atr, per_coin_table)
                        _, _, _, _, nu = sim_exit_arch(
                            h_arr, l_arr, c_arr, s.bar_idx, s.entry_price, s.direction, u,
                            sl_u, tp1_u, tp2_u, arch, l2_u,
                        )
                        all_u.append(nu)
                        (h1_u if s.timestamp <= split_ts else h2_u).append(nu)

                    def _pf(arr):
                        if not arr:
                            return 0.0
                        a = np.array(arr, dtype=float)
                        gp, gl = float(a[a > 0].sum()), float(-a[a < 0].sum())
                        return gp / gl if gl > 0.01 else (99.99 if gp > 0 else 0.0)

                    pf1 = _pf(h1_u)
                    pf2 = _pf(h2_u)
                    pfall = _pf(all_u)
                    net_all = float(sum(all_u))
                    if pf1 >= 1.05 and pf2 >= 1.05 and pfall >= 1.20 and net_all > 0:
                        cand = (net_all, pfall, arch, float(l2_u))
                        if best_qual is None or cand[0] > best_qual[0]:
                            best_qual = cand

                if best_qual is not None:
                    _, pfall, arch, l2_u = best_qual
                    vol_m = 0.85 if pfall >= 1.50 else 0.55
                    discovered[(sym, tfl, stype)] = {
                        "arch": arch,
                        "l2_u": l2_u,
                        "sl_u": sl_u,
                        "tp1_u": tp1_u,
                        "tp2_u": tp2_u,
                        "vol": vol_m,
                    }
    return discovered


def print_per_coin_table(title: str, tbl: Dict[str, Dict[str, float]]):
    print("\n" + "=" * 125)
    print(title)
    print("=" * 125)
    rows = []
    for sym in SIX_COINS:
        cc = COIN_CAL[sym]
        sl_u = cc["sl"]
        tp2_u = cc["tp2"]
        for tfl in ("15m", "30m", "1h"):
            u_pct = tbl[sym][tfl] * 100.0
            tp1_u = cc["tp1"][tfl]
            rows.append({
                "Coin": sym,
                "TF": tfl,
                "1U_%": round(u_pct, 4),
                "SL_U": sl_u,
                "SL_%": round(sl_u * u_pct, 2),
                "TP1_U": tp1_u,
                "TP1_%": round(tp1_u * u_pct, 2),
                "TP2_U": tp2_u,
                "TP2_%": round(tp2_u * u_pct, 2),
            })
    print(pd.DataFrame(rows).to_string(index=False))


def main():
    print("Loading 36-month cached 5m futures data (2023-11-01 -> 2026-10-31)...")
    raw_m5 = load_3year_cached_m5()
    cfg = BacktestConfig()
    dfs_m15 = {s: resample_ohlcv(d, "15min") for s, d in raw_m5.items()}
    dfs_m30 = {s: resample_ohlcv(d, "30min") for s, d in raw_m5.items()}
    dfs_h1 = {s: resample_ohlcv(d, "1h") for s, d in raw_m5.items()}
    dfs_h4 = {s: resample_ohlcv(d, "4h") for s, d in raw_m5.items()}
    tf_dfs_map = {"5m": raw_m5, "15m": dfs_m15, "30m": dfs_m30, "1h": dfs_h1}

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
    ts_y1_end = pd.Timestamp("2024-10-31 23:59:59", tz="UTC")
    ts_2y_end = pd.Timestamp("2025-10-31 23:59:59", tz="UTC")
    ts_y3_start = pd.Timestamp("2025-11-01 00:00:00", tz="UTC")
    ts_3y_end = pd.Timestamp("2026-10-31 23:59:59", tz="UTC")

    # Measure empirical per-coin 1U_pct strictly on respective Train windows (zero lookahead!)
    pct_tbl_train_2y = measure_per_coin_u_pct(tf_dfs_map, ts_2y_start, ts_2y_end)   # For Test B (Blind Y3)
    pct_tbl_train_y3 = measure_per_coin_u_pct(tf_dfs_map, ts_y3_start, ts_3y_end)   # For Test A (75 setups trained on Y3)

    print_per_coin_table("PER-COIN 1U_% MEASURED ON TRAIN 2Y (11/2023 -> 10/2025) — USED IN TEST B (ZERO Y3 LOOKAHEAD)", pct_tbl_train_2y)
    print_per_coin_table("PER-COIN 1U_% MEASURED ON Y3 (11/2025 -> 10/2026) — USED IN TEST A (75 SETUPS TRAIN YEAR)", pct_tbl_train_y3)

    sigs_2y = filter_sigs(tf_sigs_map, ts_2y_start, ts_2y_end)
    sigs_y3 = filter_sigs(tf_sigs_map, ts_y3_start, ts_3y_end)
    sigs_36m = filter_sigs(tf_sigs_map, ts_2y_start, ts_3y_end)

    rules_75, dir_filter = load_frozen_75()
    modes = ["PER_COIN_PCT_0.9x", "PER_COIN_PCT_1.0x", "PER_COIN_PCT_1.2x", "M1_PCT_TIGHT", "M0_PURE_ATR"]
    all_rows = []

    # =========================================================================
    # TEST A: 75 Setups v2.3 on 2Y Blind (11/2023->10/2025) & 1Y Recent (11/2025->10/2026)
    # =========================================================================
    for mode_name in modes:
        tbl_for_a = pct_tbl_train_y3 if mode_name.startswith("PER_COIN_PCT") else None
        for w_name, wsigs in [("2Y_Blind (11/2023->10/2025)", sigs_2y), ("1Y_Recent (11/2025->10/2026)", sigs_y3)]:
            for buf_name, turbo_buf in [("Bao_400G", None), ("Auto_Turbo_250G", 250.0)]:
                r = sim_portfolio_custom_u(
                    tf_dfs_map, wsigs, rules_75, mode_name, tbl_for_a,
                    dir_filter_map=dir_filter, base_buffer=400.0, auto_turbo_buffer=turbo_buf,
                )
                all_rows.append({
                    "Test": "TestA_75_Setups_v2.3",
                    "U_Mode": mode_name,
                    "Setups_Count": len(rules_75),
                    "Window": w_name,
                    "Buffer_Mode": buf_name,
                    "Trades": r["trades"],
                    "WR_%": round(r["wr"], 2),
                    "PF": round(r["pf"], 2),
                    "Return_%": round(r["ret"], 2),
                    "MaxDD_%": round(r["mdd"], 2),
                    "Calmar": round(r["calmar"], 2),
                    "Final_Eq_$": round(r["final_eq"], 2),
                })

    # =========================================================================
    # TEST B: True Multi-Year Walk-Forward (Train 2Y 11/2023->10/2025 -> Blind Test Y3 11/2025->10/2026 + 36M)
    # =========================================================================
    for mode_name in modes:
        tbl_for_b = pct_tbl_train_2y if mode_name.startswith("PER_COIN_PCT") else None
        wf_rules = discover_dual_split_rules_custom(tf_dfs_map, sigs_2y, mode_name, tbl_for_b, ts_y1_end, min_trades=10)
        n085 = sum(1 for v in wf_rules.values() if v["vol"] >= 0.80)
        n055 = len(wf_rules) - n085
        print(f"  Test B Walk-Forward [{mode_name}]: {len(wf_rules)} setups ({n085} @ 0.85x, {n055} @ 0.55x)")
        for w_name, wsigs in [
            ("1_Train_2Y (11/2023->10/2025)", sigs_2y),
            ("2_Blind_Y3 (11/2025->10/2026)", sigs_y3),
            ("3_Full_36M (11/2023->10/2026)", sigs_36m),
        ]:
            for buf_name, turbo_buf in [("Bao_400G", None), ("Auto_Turbo_250G", 250.0)]:
                r = sim_portfolio_custom_u(
                    tf_dfs_map, wsigs, wf_rules, mode_name, tbl_for_b,
                    dir_filter_map=None, base_buffer=400.0, auto_turbo_buffer=turbo_buf,
                )
                all_rows.append({
                    "Test": "TestB_WalkForward_MultiYear",
                    "U_Mode": mode_name,
                    "Setups_Count": len(wf_rules),
                    "Window": w_name,
                    "Buffer_Mode": buf_name,
                    "Trades": r["trades"],
                    "WR_%": round(r["wr"], 2),
                    "PF": round(r["pf"], 2),
                    "Return_%": round(r["ret"], 2),
                    "MaxDD_%": round(r["mdd"], 2),
                    "Calmar": round(r["calmar"], 2),
                    "Final_Eq_$": round(r["final_eq"], 2),
                })

    # =========================================================================
    # TEST C: Reverse Blind Check (Train 1Y 11/2025->10/2026 -> Blind Test Prior 2Y 11/2023->10/2025 + 36M)
    # =========================================================================
    ts_y3_mid = pd.Timestamp("2026-04-30 23:59:59", tz="UTC")
    for mode_name in modes:
        tbl_for_c = pct_tbl_train_y3 if mode_name.startswith("PER_COIN_PCT") else None
        wf_rules_c = discover_dual_split_rules_custom(tf_dfs_map, sigs_y3, mode_name, tbl_for_c, ts_y3_mid, min_trades=8)
        n085 = sum(1 for v in wf_rules_c.values() if v["vol"] >= 0.80)
        n055 = len(wf_rules_c) - n085
        print(f"  Test C Reverse-Blind [{mode_name}]: {len(wf_rules_c)} setups ({n085} @ 0.85x, {n055} @ 0.55x)")
        for w_name, wsigs in [
            ("1_Train_Y3 (11/2025->10/2026)", sigs_y3),
            ("2_Blind_2Y (11/2023->10/2025)", sigs_2y),
            ("3_Full_36M (11/2023->10/2026)", sigs_36m),
        ]:
            for buf_name, turbo_buf in [("Bao_400G", None), ("Auto_Turbo_250G", 250.0)]:
                r = sim_portfolio_custom_u(
                    tf_dfs_map, wsigs, wf_rules_c, mode_name, tbl_for_c,
                    dir_filter_map=None, base_buffer=400.0, auto_turbo_buffer=turbo_buf,
                )
                all_rows.append({
                    "Test": "TestC_TrainY3_Blind2Y",
                    "U_Mode": mode_name,
                    "Setups_Count": len(wf_rules_c),
                    "Window": w_name,
                    "Buffer_Mode": buf_name,
                    "Trades": r["trades"],
                    "WR_%": round(r["wr"], 2),
                    "PF": round(r["pf"], 2),
                    "Return_%": round(r["ret"], 2),
                    "MaxDD_%": round(r["mdd"], 2),
                    "Calmar": round(r["calmar"], 2),
                    "Final_Eq_$": round(r["final_eq"], 2),
                })

    df_out = pd.DataFrame(all_rows)
    out_csv = os.path.join(RESULTS_DIR, "per_coin_pct_3year_results.csv")
    df_out.to_csv(out_csv, index=False)

    print("\n" + "=" * 130)
    print("TEST A: 75 SETUPS v2.3 — PER_COIN_PCT (0.9x / 1.0x / 1.2x) vs M1_PCT_TIGHT vs M0_PURE_ATR")
    print("=" * 130)
    print(df_out[df_out["Test"] == "TestA_75_Setups_v2.3"].to_string(index=False))

    print("\n" + "=" * 130)
    print("TEST B: WALK-FORWARD (TRAIN 2Y 11/2023->10/2025 -> BLIND Y3 11/2025->10/2026 + 36M)")
    print("=" * 130)
    print(df_out[df_out["Test"] == "TestB_WalkForward_MultiYear"].to_string(index=False))

    print("\n" + "=" * 130)
    print("TEST C: REVERSE BLIND (TRAIN 1Y 11/2025->10/2026 -> BLIND 2Y 11/2023->10/2025 + 36M)")
    print("=" * 130)
    print(df_out[df_out["Test"] == "TestC_TrainY3_Blind2Y"].to_string(index=False))
    print(f"\nSaved results to: {out_csv}")


if __name__ == "__main__":
    main()
