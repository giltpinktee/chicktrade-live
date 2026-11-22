"""Full Re-Audit of All 288 Setups ("Gom Nhặt Chặt Bị" — Rare Snipers & Missed Sweet Spots).

Evaluates all currently skipped 15m/30m/1h setups across 6 coins (keeping 5m filtered out
for fee/slippage protection, except LINKUSDT 5m EMA89_W2 already proven in v2.1) under the
EXACT v2.1 Portfolio Engine (PORTFOLIO 2-SL Core-Only -> Shadow-Win, Rule 1B, Bao 400G & Auto-Turbo).
"""
import copy
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(SCRIPTS_DIR))
for p in (ROOT, SCRIPTS_DIR, os.path.join(ROOT, "old", "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from config import BacktestConfig, RiskConfig
from data.binance_loader import _detect_latest_available_month, download_symbol_1y
from run_288_with_shadow_win_recheck import EXIT_CANDS
from run_alpha_discovery_argon import sim_portfolio_advanced as run_portfolio_sim
from run_full_216_exit_sweetspot import COIN_CAL, sim_exit_arch
from run_full_3ema_storm_guard import build_cooldown_mask, compute_storm_flags, generate_all_4_touches
from run_master_6coin_html import SIX_COINS
from run_research_mtf import resample_ohlcv

RESULTS_DIR = os.path.join(ROOT, "verified_tests", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_v21_rules() -> Tuple[dict, dict]:
    """Load the exact 46 active rules + Rule 1B direction filter from core_strategy_rules.json."""
    with open(os.path.join(ROOT, "core", "core_strategy_rules.json"), encoding="utf-8") as f:
        cfg = json.load(f)

    rules_map = {}
    dir_filter = {}
    for r in cfg["rules"]:
        k = (r["coin"], r["tf"], r["setup_code"])
        rules_map[k] = {
            "arch": r["exit_arch"],
            "sl_u": float(r["sl_u"]),
            "tp1_u": float(r["tp1_u"]),
            "tp2_u": float(r["tp2_u"]),
            "l2_u": float(r["leg2_sl_u"]),
            "vol": float(r["adaptive_volume"]["init_vol_mult"]),
            "status": r["status"],
        }
        ad = r.get("allowed_direction", "BOTH")
        if ad == "LONG_ONLY":
            dir_filter[k] = 1
        elif ad == "SHORT_ONLY":
            dir_filter[k] = -1
    return rules_map, dir_filter


def eval_best_exit_for_setup(df: pd.DataFrame, sigs: list, sym: str, tfl: str, mid_ts: pd.Timestamp) -> dict:
    """Evaluate all 7 exit architectures with real Binance Futures fees+slippage; pick best Net U."""
    h_arr, l_arr, c_arr = df["high"].values, df["low"].values, df["close"].values
    sl_u = COIN_CAL[sym]["sl"]
    tp1_u = COIN_CAL[sym]["tp1"][tfl]
    tp2_u = COIN_CAL[sym]["tp2"]

    best = None
    for label, arch, l2_u in EXIT_CANDS:
        # Fee-leak guard: if PARTIAL_MOVE_SL, ensure TP1_u > |l2_u| + 0.6U
        if arch == "PARTIAL_MOVE_SL" and l2_u is not None and l2_u < 0 and (tp1_u <= abs(l2_u) + 0.6):
            continue
        is_u, oos_u, all_u = [], [], []
        for s in sigs:
            u = max(0.5 * s.atr, s.entry_price * 1e-5)
            _, _, _, _, nu = sim_exit_arch(
                h_arr, l_arr, c_arr, s.bar_idx, s.entry_price, s.direction, u,
                sl_u, tp1_u, tp2_u, arch, l2_u,
            )
            all_u.append(nu)
            (oos_u if s.timestamp >= mid_ts else is_u).append(nu)

        a = np.array(all_u)
        n = len(a)
        if n == 0:
            continue
        wr = float((a > 0).mean() * 100.0)
        net_u = float(a.sum())
        gp, gl = float(a[a > 0].sum()), float(-a[a < 0].sum())
        pf_1y = gp / gl if gl > 0.01 else (99.99 if gp > 0 else 0.0)

        def _half_stats(arr):
            if not arr:
                return 0, 99.99, 0.0
            ha = np.array(arr)
            hgp, hgl = float(ha[ha > 0].sum()), float(-ha[ha < 0].sum())
            hpf = hgp / hgl if hgl > 0.01 else (99.99 if hgp > 0 else 0.0)
            return len(ha), hpf, float(ha.sum())

        n_is, pf_is, net_is = _half_stats(is_u)
        n_oos, pf_oos, net_oos = _half_stats(oos_u)

        cand = {
            "label": label, "arch": arch, "sl_u": sl_u, "tp1_u": tp1_u, "tp2_u": tp2_u,
            "l2_u": l2_u if l2_u is not None else -sl_u,
            "n": n, "n_is": n_is, "n_oos": n_oos,
            "wr": wr, "pf_1y": pf_1y, "pf_is": pf_is, "pf_oos": pf_oos,
            "net_u": net_u, "net_is": net_is, "net_oos": net_oos,
        }
        if best is None or cand["net_u"] > best["net_u"]:
            best = cand
    return best


def main():
    print("=" * 90)
    print("FULL 288-SETUP RE-AUDIT: GOM NHAT CHAT BI (RARE SNIPERS + MISSED SWEET SPOTS)")
    print("=" * 90)

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
    print(f"Mid OOS split: {mid_ts}")

    v21_rules, dir_filter = load_v21_rules()
    print(f"Loaded v2.1 baseline: {len(v21_rules)} setups | Rule 1B filters: {len(dir_filter)}")

    # Audit every inactive setup on 15m / 30m / 1h
    cand_rows = []
    cand_rule_objs = {}

    for tfl in ["15m", "30m", "1h"]:
        for sym in SIX_COINS:
            df = tf_dfs_map[tfl][sym]
            by_setup = {}
            for s in tf_sigs_map[tfl][sym]:
                by_setup.setdefault(s.setup_type, []).append(s)

            for stype, sigs in sorted(by_setup.items()):
                k = (sym, tfl, stype)
                if k in v21_rules:
                    continue
                best = eval_best_exit_for_setup(df, sigs, sym, tfl, mid_ts)
                if best is None or best["net_u"] <= 0:
                    continue

                # Classify into clean buckets
                n = best["n"]
                wr = best["wr"]
                pf1 = best["pf_1y"]
                pf_is = best["pf_is"]
                pf_oos = best["pf_oos"]
                net_is = best["net_is"]
                net_oos = best["net_oos"]
                n_is = best["n_is"]
                n_oos = best["n_oos"]

                # Bucket logic:
                # 1) SNIPER_RARE: 2 <= N <= 7, WR >= 60%, PF_1Y >= 1.35, no half deeply negative (net >= -2U)
                # 2) MISSED_DUAL_STRICT: N >= 8, WR >= 60%, PF_IS >= 1.15, PF_OOS >= 1.15, PF_1Y >= 1.40
                # 3) MISSED_DUAL_SOLID: N >= 8, PF_IS >= 1.05, PF_OOS >= 1.05, PF_1Y >= 1.25 (not in STRICT)
                # 4) ONE_HALF_ONLY: N >= 8, PF_1Y >= 1.25, but 1 half < 1.05
                # 5) WEAK_MARGINAL: Net_U > 0 but PF_1Y < 1.25 or N == 1
                if 2 <= n <= 7 and wr >= 60.0 and pf1 >= 1.35 and (n_is == 0 or net_is >= -1.0) and (n_oos == 0 or net_oos >= -1.0):
                    bucket = "1_SNIPER_RARE"
                elif n >= 8 and wr >= 60.0 and pf_is >= 1.15 and pf_oos >= 1.15 and pf1 >= 1.40:
                    bucket = "2A_MISSED_DUAL_STRICT"
                elif n >= 8 and pf_is >= 1.05 and pf_oos >= 1.05 and pf1 >= 1.25:
                    bucket = "2B_MISSED_DUAL_SOLID"
                elif n >= 8 and pf1 >= 1.25:
                    bucket = "3_ONE_HALF_BROKEN"
                else:
                    bucket = "4_MARGINAL_NOISE"

                cand_rows.append({
                    "Bucket": bucket, "Coin": sym, "TF": tfl, "Setup": stype,
                    "N": n, "N_IS": n_is, "N_OOS": n_oos,
                    "WR%": round(wr, 1), "PF_IS": round(pf_is, 2), "PF_OOS": round(pf_oos, 2),
                    "PF_1Y": round(pf1, 2), "Net_U": round(best["net_u"], 2),
                    "Net_IS": round(net_is, 2), "Net_OOS": round(net_oos, 2),
                    "Best_Exit": best["label"], "Arch": best["arch"],
                    "SL_U": best["sl_u"], "TP1_U": best["tp1_u"], "TP2_U": best["tp2_u"], "L2_U": best["l2_u"],
                })
                cand_rule_objs[k] = {
                    "arch": best["arch"], "sl_u": best["sl_u"], "tp1_u": best["tp1_u"],
                    "tp2_u": best["tp2_u"], "l2_u": best["l2_u"], "bucket": bucket,
                    "wr": wr, "pf_1y": pf1, "net_u": best["net_u"], "n": n,
                }

    df_cands = pd.DataFrame(cand_rows).sort_values(["Bucket", "Net_U"], ascending=[True, False])
    df_cands.to_csv(os.path.join(RESULTS_DIR, "gom_nhat_candidates.csv"), index=False)

    print("\n--- CANDIDATE BUCKETS (15m / 30m / 1h, Net_U > 0, NOT in v2.1) ---")
    print(df_cands.groupby("Bucket").agg(
        Count=("Net_U", "count"),
        Total_Trades=("N", "sum"),
        Total_Net_U=("Net_U", "sum"),
        Total_OOS_U=("Net_OOS", "sum"),
        Avg_WR=("WR%", "mean"),
    ).to_string())

    for b in ["1_SNIPER_RARE", "2A_MISSED_DUAL_STRICT", "2B_MISSED_DUAL_SOLID"]:
        sub = df_cands[df_cands["Bucket"] == b]
        print(f"\n=== {b} ({len(sub)} setups, +{sub['Net_U'].sum():.1f}U) ===")
        print(sub[["Coin", "TF", "Setup", "N", "WR%", "PF_IS", "PF_OOS", "PF_1Y", "Net_U", "Net_OOS", "Best_Exit"]].to_string(index=False))

    # Build Portfolio Test Scenarios
    def make_rules(include_buckets: List[str], vol_map: Dict[str, float]) -> dict:
        rm = copy.deepcopy(v21_rules)
        for k, obj in cand_rule_objs.items():
            if obj["bucket"] in include_buckets:
                rm[k] = {
                    "arch": obj["arch"], "sl_u": obj["sl_u"], "tp1_u": obj["tp1_u"],
                    "tp2_u": obj["tp2_u"], "l2_u": obj["l2_u"],
                    "vol": vol_map[obj["bucket"]],
                    "status": obj["bucket"],
                }
        return rm

    scenarios = [
        ("0_BASELINE_v2.1 (46 setups)", v21_rules),
        ("1_v2.1 + SNIPER_RARE @ 0.40x", make_rules(["1_SNIPER_RARE"], {"1_SNIPER_RARE": 0.40})),
        ("2_v2.1 + SNIPER_RARE @ 0.65x", make_rules(["1_SNIPER_RARE"], {"1_SNIPER_RARE": 0.65})),
        ("3_v2.1 + SNIPER_RARE @ 0.85x", make_rules(["1_SNIPER_RARE"], {"1_SNIPER_RARE": 0.85})),
        ("4_v2.1 + 2A_DUAL_STRICT @ 0.65x", make_rules(["2A_MISSED_DUAL_STRICT"], {"2A_MISSED_DUAL_STRICT": 0.65})),
        ("5_v2.1 + 2A_DUAL_STRICT @ 0.85x", make_rules(["2A_MISSED_DUAL_STRICT"], {"2A_MISSED_DUAL_STRICT": 0.85})),
        ("6_v2.1 + SNIPER(0.45x) + 2A_STRICT(0.75x)", make_rules(
            ["1_SNIPER_RARE", "2A_MISSED_DUAL_STRICT"],
            {"1_SNIPER_RARE": 0.45, "2A_MISSED_DUAL_STRICT": 0.75},
        )),
        ("7_v2.1 + SNIPER(0.65x) + 2A_STRICT(0.85x)", make_rules(
            ["1_SNIPER_RARE", "2A_MISSED_DUAL_STRICT"],
            {"1_SNIPER_RARE": 0.65, "2A_MISSED_DUAL_STRICT": 0.85},
        )),
        ("8_v2.1 + SNIPER(0.45x) + 2A(0.75x) + 2B(0.45x)", make_rules(
            ["1_SNIPER_RARE", "2A_MISSED_DUAL_STRICT", "2B_MISSED_DUAL_SOLID"],
            {"1_SNIPER_RARE": 0.45, "2A_MISSED_DUAL_STRICT": 0.75, "2B_MISSED_DUAL_SOLID": 0.45},
        )),
        ("9_v2.1 + ALL_1_2_3 (incl. 1-half broken @ 0.40x)", make_rules(
            ["1_SNIPER_RARE", "2A_MISSED_DUAL_STRICT", "2B_MISSED_DUAL_SOLID", "3_ONE_HALF_BROKEN"],
            {"1_SNIPER_RARE": 0.45, "2A_MISSED_DUAL_STRICT": 0.75, "2B_MISSED_DUAL_SOLID": 0.45, "3_ONE_HALF_BROKEN": 0.40},
        )),
    ]

    print("\n" + "=" * 110)
    print("EXACT PORTFOLIO BACKTEST (PORTFOLIO 2-SL Core-Only -> Shadow-Win + Rule 1B)")
    print("=" * 110)

    res_rows = []
    for mode_name, turbo_buf in [("BASE_400G", None), ("TURBO_400G_250G", 250.0)]:
        print(f"\n>>> MODE: {mode_name} <<<")
        for sc_name, rm in scenarios:
            r12 = run_portfolio_sim(
                tf_dfs_map, tf_sigs_map, rm,
                start_time=None, base_buffer=400.0,
                dir_filter_map=dir_filter,
                auto_turbo_buffer=turbo_buf,
                core_only_sl_trigger=True,
            )
            roos = run_portfolio_sim(
                tf_dfs_map, tf_sigs_map, rm,
                start_time=mid_ts, base_buffer=400.0,
                dir_filter_map=dir_filter,
                auto_turbo_buffer=turbo_buf,
                core_only_sl_trigger=True,
            )
            row = {
                "Mode": mode_name,
                "Scenario": sc_name,
                "Setups": len(rm),
                "Trades_12M": r12["trades"],
                "WR_12M%": round(r12["wr"], 2),
                "Ret_12M%": round(r12["ret"], 2),
                "Eq_12M$": round(r12["final_eq"], 0),
                "MaxDD_12M%": round(r12["mdd"], 2),
                "Calmar_12M": round(r12["calmar"], 2),
                "Ret_OOS%": round(roos["ret"], 2),
                "MaxDD_OOS%": round(roos["mdd"], 2),
                "Calmar_OOS": round(roos["calmar"], 2),
            }
            res_rows.append(row)
            print(
                f"  {sc_name:48s} | N={len(rm):2d} | Tr={r12['trades']:3d} | "
                f"12M: +{r12['ret']:7.1f}% (DD {r12['mdd']:6.2f}%, Cal {r12['calmar']:6.1f}) | "
                f"OOS 6M: +{roos['ret']:6.1f}% (DD {roos['mdd']:6.2f}%, Cal {roos['calmar']:5.1f})"
            )

    df_res = pd.DataFrame(res_rows)
    df_res.to_csv(os.path.join(RESULTS_DIR, "gom_nhat_full_audit.csv"), index=False)
    print("\nSaved results to verified_tests/results/gom_nhat_full_audit.csv & gom_nhat_candidates.csv")


if __name__ == "__main__":
    main()
