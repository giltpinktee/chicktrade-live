"""3-Year Quantitative Study (2023-11-01 -> 2026-10-31): Percentage 1U vs Hybrid (ATR + % Floor) vs Pure ATR.

Reads cached 24m (2023-11 -> 2025-10) + 12m (2025-11 -> 2026-10) 5m OHLCV from data/cache/.
Does NOT modify core/, bot/, or any existing verified_tests scripts.

Experiment 1:
  Tests 6 `u` sizing formulas (M0_PURE_ATR, M1_PCT_TIGHT, M2_PCT_MID, M3_PCT_WIDE,
  M4_HYBRID_FLOOR_MID, M5_HYBRID_FLOOR_WIDE) on the exact frozen 75 setups (v2.3) + Rule 1B
  over both 2Y Blind (2023-11 -> 2025-10) and 1Y Recent (2025-11 -> 2026-10), for Bao 400G & Auto-Turbo 250G.

Experiment 2 (True Multi-Year Walk-Forward):
  Using the 3 best `u` modes from Experiment 1 (M0_PURE_ATR, best PCT, best HYBRID):
  1. Scans all 216 setups (15m/30m/1h x 6 coins x 3 EMAs x 4 touches) strictly on Train 2Y (11/2023 -> 10/2025).
  2. Selects True Multi-Year Edge setups passing BOTH Year 1 and Year 2:
       PF_Y1 >= 1.05, PF_Y2 >= 1.05, PF_2Y >= 1.20, N_2Y >= 10
     with Volume = 0.85x if PF_2Y >= 1.50 else 0.55x.
  3. Evaluates the discovered portfolio (Bao 400G & Auto-Turbo 250G) on:
       - Train 2Y (11/2023 -> 10/2025)
       - Blind Test Year 3 (11/2025 -> 10/2026, reset $10,000)
       - Continuous 36M (11/2023 -> 10/2026, start $10,000)

Output:
  - e:\CHICKTRADE\verified_tests\results\pct_vs_atr_3year_results.csv
"""
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

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
from run_alpha_discovery_argon import sim_portfolio_advanced
from run_full_216_exit_sweetspot import COIN_CAL, sim_exit_arch
from run_full_3ema_storm_guard import build_cooldown_mask, compute_storm_flags, generate_all_4_touches
from run_master_6coin_html import SIX_COINS
from run_research_mtf import resample_ohlcv

CACHE_DIR = os.path.join(ROOT, "data", "cache")
RULES_PATH = os.path.join(ROOT, "core", "core_strategy_rules.json")
RESULTS_DIR = os.path.join(VERIFIED_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

U_MODES = [
    "M0_PURE_ATR",
    "M1_PCT_TIGHT",
    "M2_PCT_MID",
    "M3_PCT_WIDE",
    "M4_HYBRID_FLOOR_MID",
    "M5_HYBRID_FLOOR_WIDE",
]

PCT_MAPS = {
    "M1_PCT_TIGHT": {"5m": 0.0010, "15m": 0.0015, "30m": 0.0025, "1h": 0.0040},
    "M2_PCT_MID": {"5m": 0.0012, "15m": 0.0020, "30m": 0.0030, "1h": 0.0045},
    "M3_PCT_WIDE": {"5m": 0.0015, "15m": 0.0025, "30m": 0.0035, "1h": 0.0050},
    "M4_HYBRID_FLOOR_MID": {"5m": 0.0010, "15m": 0.0015, "30m": 0.0022, "1h": 0.0035},
    "M5_HYBRID_FLOOR_WIDE": {"5m": 0.0012, "15m": 0.0020, "30m": 0.0030, "1h": 0.0045},
}


def compute_u(u_mode: str, tfl: str, price: float, atr: float) -> float:
    if u_mode == "M0_PURE_ATR":
        return max(0.5 * atr, price * 1e-5)
    pct_val = price * PCT_MAPS[u_mode][tfl]
    if u_mode.startswith("M4_") or u_mode.startswith("M5_"):
        return max(0.5 * atr, pct_val)
    return max(pct_val, price * 1e-5)


def load_3year_cached_m5() -> Dict[str, pd.DataFrame]:
    """Load and stitch cached 24M (2023-11..2025-10) + 12M (2025-11..2026-10) 5m futures data."""
    raw_m5: Dict[str, pd.DataFrame] = {}
    for sym in SIX_COINS:
        p24 = os.path.join(CACHE_DIR, f"{sym}_futures_5m_202510_24m.csv.gz")
        p12 = os.path.join(CACHE_DIR, f"{sym}_futures_5m_202610_12m.csv.gz")
        df24 = pd.read_csv(p24, parse_dates=["timestamp"])
        df12 = pd.read_csv(p12, parse_dates=["timestamp"])
        for d in (df24, df12):
            if d["timestamp"].dt.tz is None:
                d["timestamp"] = d["timestamp"].dt.tz_localize("UTC")
        df = pd.concat([df24, df12], ignore_index=True)
        df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        t0, t1 = df["timestamp"].min(), df["timestamp"].max()
        assert str(t0).startswith("2023-11-01"), f"{sym} start mismatch: {t0}"
        assert str(t1).startswith("2026-10-31"), f"{sym} end mismatch: {t1}"
        raw_m5[sym] = df
        print(f"  {sym:8s}: {len(df):,} bars | {t0} -> {t1}")
    return raw_m5


def load_frozen_75() -> Tuple[dict, Dict[Tuple[str, str, str], int]]:
    with open(RULES_PATH, encoding="utf-8") as f:
        cfg = json.load(f)
    rules_75 = {}
    dir_filter = {}
    for r in cfg["rules"]:
        k = (r["coin"], r["tf"], r["setup_code"])
        rules_75[k] = {
            "arch": r["exit_arch"],
            "l2_u": float(r["leg2_sl_u"]),
            "sl_u": float(r["sl_u"]),
            "tp1_u": float(r["tp1_u"]),
            "tp2_u": float(r["tp2_u"]),
            "vol": float(r["adaptive_volume"]["init_vol_mult"]),
        }
        ad = r.get("allowed_direction", "BOTH")
        if ad == "LONG_ONLY":
            dir_filter[k] = 1
        elif ad == "SHORT_ONLY":
            dir_filter[k] = -1
    return rules_75, dir_filter


def filter_sigs(tf_sigs_map: dict, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> dict:
    return {
        tfl: {sym: [s for s in sigs if start_ts <= s.timestamp <= end_ts] for sym, sigs in sm.items()}
        for tfl, sm in tf_sigs_map.items()
    }


def sim_portfolio_u_mode(
    tf_dfs_map: dict,
    tf_sigs_map: dict,
    rules_map: dict,
    u_mode: str,
    dir_filter_map: Optional[Dict[Tuple[str, str, str], int]] = None,
    base_buffer: float = 400.0,
    auto_turbo_buffer: Optional[float] = None,
    hard_floor_eq: float = 11_000.0,
    core_only_sl_trigger: bool = True,
) -> dict:
    """Unified 6-coin portfolio simulator matching sim_portfolio_advanced with parameterized `u_mode`."""
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
        nonlocal equity, peak_eq, port_consec_sl, port_shadow_paused, port_paused_until
        nonlocal in_turbo, ever_triggered_turbo, min_eq_after_turbo
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
                if (not in_turbo) and equity >= 12_500.0:
                    in_turbo = True
                    ever_triggered_turbo = True
                    min_eq_after_turbo = min(min_eq_after_turbo, equity)
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
        u = compute_u(u_mode, tfl, sig.entry_price, sig.atr)

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


def discover_multi_year_rules(
    tf_dfs_map: dict,
    train_2y_sigs: dict,
    u_mode: str,
    y1_end: pd.Timestamp,
) -> dict:
    """Scan all 216 setups (6 coins x 15m/30m/1h x 3 EMAs x 4 touches) strictly on Train 2Y (11/2023 -> 10/2025).

    Filter criteria:
      - PF_Y1 >= 1.05
      - PF_Y2 >= 1.05
      - PF_2Y >= 1.20
      - N_2Y >= 10
    Volume:
      - 0.85x if PF_2Y >= 1.50 else 0.55x
    """
    discovered = {}
    for sym in SIX_COINS:
        cc = COIN_CAL[sym]
        sl_u = float(cc["sl"])
        tp2_u = float(cc["tp2"])
        for tfl in ("15m", "30m", "1h"):
            tp1_u = float(cc["tp1"][tfl])
            df = tf_dfs_map[tfl][sym]
            h_arr, l_arr, c_arr = df["high"].values, df["low"].values, df["close"].values
            sigs_by_setup: Dict[str, list] = {}
            for s in train_2y_sigs[tfl][sym]:
                sigs_by_setup.setdefault(s.setup_type, []).append(s)

            for stype, sigs in sorted(sigs_by_setup.items()):
                if len(sigs) < 10:
                    continue
                best_qual = None
                for _, arch, l2_u in EXIT_CANDS:
                    if arch == "PARTIAL_MOVE_SL" and l2_u < 0 and tp1_u <= abs(l2_u) + 0.6:
                        continue
                    y1_u, y2_u, all_u = [], [], []
                    for s in sigs:
                        u = compute_u(u_mode, tfl, s.entry_price, s.atr)
                        _, _, _, _, nu = sim_exit_arch(
                            h_arr, l_arr, c_arr, s.bar_idx, s.entry_price, s.direction, u,
                            sl_u, tp1_u, tp2_u, arch, l2_u,
                        )
                        all_u.append(nu)
                        (y1_u if s.timestamp <= y1_end else y2_u).append(nu)

                    def _pf(arr):
                        if not arr:
                            return 0.0
                        a = np.array(arr, dtype=float)
                        gp, gl = float(a[a > 0].sum()), float(-a[a < 0].sum())
                        return gp / gl if gl > 0.01 else (99.99 if gp > 0 else 0.0)

                    pf1 = _pf(y1_u)
                    pf2 = _pf(y2_u)
                    pf2y = _pf(all_u)
                    net2y = float(sum(all_u))
                    if pf1 >= 1.05 and pf2 >= 1.05 and pf2y >= 1.20 and net2y > 0:
                        cand = (net2y, pf2y, arch, float(l2_u))
                        if best_qual is None or cand[0] > best_qual[0]:
                            best_qual = cand

                if best_qual is not None:
                    _, pf2y, arch, l2_u = best_qual
                    vol_m = 0.85 if pf2y >= 1.50 else 0.55
                    discovered[(sym, tfl, stype)] = {
                        "arch": arch,
                        "l2_u": l2_u,
                        "sl_u": sl_u,
                        "tp1_u": tp1_u,
                        "tp2_u": tp2_u,
                        "vol": vol_m,
                    }
    return discovered


def main():
    print("Loading 36-month cached 5m futures data (2023-11-01 -> 2026-10-31)...")
    raw_m5 = load_3year_cached_m5()
    cfg = BacktestConfig()
    dfs_m15 = {s: resample_ohlcv(d, "15min") for s, d in raw_m5.items()}
    dfs_m30 = {s: resample_ohlcv(d, "30min") for s, d in raw_m5.items()}
    dfs_h1 = {s: resample_ohlcv(d, "1h") for s, d in raw_m5.items()}
    dfs_h4 = {s: resample_ohlcv(d, "4h") for s, d in raw_m5.items()}
    tf_dfs_map = {"5m": raw_m5, "15m": dfs_m15, "30m": dfs_m30, "1h": dfs_h1}

    print("Generating 4-touch signals across full 36 months...")
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

    sigs_2y = filter_sigs(tf_sigs_map, ts_2y_start, ts_2y_end)
    sigs_1y_recent = filter_sigs(tf_sigs_map, ts_y3_start, ts_3y_end)
    sigs_3y = filter_sigs(tf_sigs_map, ts_2y_start, ts_3y_end)

    rules_75, dir_filter = load_frozen_75()

    # Parity check: M0_PURE_ATR vs sim_portfolio_advanced on 2Y Blind
    ref_m0 = sim_portfolio_advanced(
        tf_dfs_map, sigs_2y, rules_75,
        start_time=None, dir_filter_map=dir_filter, base_buffer=400.0, core_only_sl_trigger=True,
    )
    chk_m0 = sim_portfolio_u_mode(
        tf_dfs_map, sigs_2y, rules_75, "M0_PURE_ATR",
        dir_filter_map=dir_filter, base_buffer=400.0, core_only_sl_trigger=True,
    )
    assert abs(ref_m0["final_eq"] - chk_m0["final_eq"]) < 0.05, f"Parity mismatch: {ref_m0['final_eq']} vs {chk_m0['final_eq']}"

    all_rows = []

    # =========================================================================
    # EXPERIMENT 1: 75 Setups (v2.3) across M0..M5 on 2Y Blind & 1Y Recent
    # =========================================================================
    exp1_windows = [
        ("2Y_Blind (11/2023->10/2025)", sigs_2y),
        ("1Y_Recent (11/2025->10/2026)", sigs_1y_recent),
    ]
    for u_mode in U_MODES:
        for w_name, wsigs in exp1_windows:
            for mode_name, turbo_buf in [("Bao_400G", None), ("Auto_Turbo_250G", 250.0)]:
                r = sim_portfolio_u_mode(
                    tf_dfs_map, wsigs, rules_75, u_mode,
                    dir_filter_map=dir_filter, base_buffer=400.0, auto_turbo_buffer=turbo_buf,
                    hard_floor_eq=11_000.0, core_only_sl_trigger=True,
                )
                all_rows.append({
                    "Experiment": "EXP1_75_Setups_v2.3",
                    "U_Mode": u_mode,
                    "Setups_Count": len(rules_75),
                    "Window": w_name,
                    "Buffer_Mode": mode_name,
                    "Trades": r["trades"],
                    "WR_%": round(r["wr"], 2),
                    "PF": round(r["pf"], 2),
                    "Return_%": round(r["ret"], 2),
                    "MaxDD_%": round(r["mdd"], 2),
                    "Calmar": round(r["calmar"], 2),
                    "Final_Eq_$": round(r["final_eq"], 2),
                })

    df_exp1 = pd.DataFrame([r for r in all_rows if r["Experiment"] == "EXP1_75_Setups_v2.3"])
    print("\n" + "=" * 130)
    print("EXPERIMENT 1: 75 SETUPS v2.3 — M0 (PURE ATR) vs M1..M3 (PURE % PRICE) vs M4..M5 (HYBRID ATR + % FLOOR)")
    print("=" * 130)
    print(df_exp1.to_string(index=False))

    # Pick best PCT (M1..M3) and best HYBRID (M4..M5) by 2Y_Blind Bao_400G Final_Eq_$ (then PF, then 1Y_Recent)
    pct_cands = df_exp1[(df_exp1["U_Mode"].isin(["M1_PCT_TIGHT", "M2_PCT_MID", "M3_PCT_WIDE"])) & (df_exp1["Window"].str.startswith("2Y_Blind")) & (df_exp1["Buffer_Mode"] == "Bao_400G")]
    best_pct_mode = pct_cands.sort_values(["Final_Eq_$", "PF"], ascending=False).iloc[0]["U_Mode"]

    hyb_cands = df_exp1[(df_exp1["U_Mode"].isin(["M4_HYBRID_FLOOR_MID", "M5_HYBRID_FLOOR_WIDE"])) & (df_exp1["Window"].str.startswith("2Y_Blind")) & (df_exp1["Buffer_Mode"] == "Bao_400G")]
    best_hyb_mode = hyb_cands.sort_values(["Final_Eq_$", "PF"], ascending=False).iloc[0]["U_Mode"]

    print(f"\nSelected 3 modes for Experiment 2 Walk-Forward: ['M0_PURE_ATR', '{best_pct_mode}', '{best_hyb_mode}']")

    # =========================================================================
    # EXPERIMENT 2: True Multi-Year Walk-Forward (Train 2Y -> Blind Test Year 3 + Full 36M)
    # =========================================================================
    wf_modes = ["M0_PURE_ATR", best_pct_mode, best_hyb_mode]
    exp2_windows = [
        ("1_Train_2Y (11/2023->10/2025)", sigs_2y),
        ("2_Blind_Y3 (11/2025->10/2026)", sigs_1y_recent),
        ("3_Full_36M (11/2023->10/2026)", sigs_3y),
    ]
    for u_mode in wf_modes:
        wf_rules = discover_multi_year_rules(tf_dfs_map, sigs_2y, u_mode, ts_y1_end)
        n_085 = sum(1 for v in wf_rules.values() if v["vol"] >= 0.80)
        n_055 = len(wf_rules) - n_085
        print(f"  Walk-Forward [{u_mode}]: discovered {len(wf_rules)} True Multi-Year Edge setups ({n_085} @ 0.85x, {n_055} @ 0.55x)")
        for w_name, wsigs in exp2_windows:
            for mode_name, turbo_buf in [("Bao_400G", None), ("Auto_Turbo_250G", 250.0)]:
                r = sim_portfolio_u_mode(
                    tf_dfs_map, wsigs, wf_rules, u_mode,
                    dir_filter_map=None, base_buffer=400.0, auto_turbo_buffer=turbo_buf,
                    hard_floor_eq=11_000.0, core_only_sl_trigger=True,
                )
                all_rows.append({
                    "Experiment": "EXP2_WalkForward_MultiYear",
                    "U_Mode": u_mode,
                    "Setups_Count": len(wf_rules),
                    "Window": w_name,
                    "Buffer_Mode": mode_name,
                    "Trades": r["trades"],
                    "WR_%": round(r["wr"], 2),
                    "PF": round(r["pf"], 2),
                    "Return_%": round(r["ret"], 2),
                    "MaxDD_%": round(r["mdd"], 2),
                    "Calmar": round(r["calmar"], 2),
                    "Final_Eq_$": round(r["final_eq"], 2),
                })

    df_all = pd.DataFrame(all_rows)
    out_csv = os.path.join(RESULTS_DIR, "pct_vs_atr_3year_results.csv")
    df_all.to_csv(out_csv, index=False)

    df_exp2 = df_all[df_all["Experiment"] == "EXP2_WalkForward_MultiYear"]
    print("\n" + "=" * 130)
    print("EXPERIMENT 2: TRUE MULTI-YEAR WALK-FORWARD (TRAIN 2Y 11/2023->10/2025 -> BLIND TEST Y3 11/2025->10/2026 + CONTINUOUS 36M)")
    print("=" * 130)
    print(df_exp2.to_string(index=False))
    print(f"\nSaved full results to: {out_csv}")


if __name__ == "__main__":
    main()
