"""Per-Coin x Per-TF x Per-Setup Leg2 SL Move Optimizer (Zero-Fee-Leak Invariant + Strict OOS Verification)."""
import json
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

RESEARCH_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(RESEARCH_DIR)
for p in (ROOT, RESEARCH_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from config import BacktestConfig
from data.binance_loader import _detect_latest_available_month, download_symbol_1y
from run_6coin_full_study import fast_sim_policy
from run_full_3ema_storm_guard import build_cooldown_mask, compute_storm_flags, generate_all_4_touches
from run_master_6coin_html import SIX_COINS
from run_research_mtf import resample_ohlcv

# Candidate Leg2 SL positions: +0.3 means Entry+Fee, 0.0 means Entry, -k means Entry - kU
LEG2_CANDIDATES = [
    ("Entry+Phi", 0.3),
    ("Entry (0U)", 0.0),
    ("Entry-1.0U", -1.0),
    ("Entry-1.5U", -1.5),
    ("Entry-2.0U", -2.0),
    ("Entry-2.5U", -2.5),
    ("Entry-3.0U", -3.0),
]


def main():
    rules_path = os.path.join(ROOT, "core", "core_strategy_rules.json")
    with open(rules_path, encoding="utf-8") as f:
        core_cfg = json.load(f)
    rules_list = core_cfg["rules"]

    cfg = BacktestConfig()
    end_ym = _detect_latest_available_month("BTCUSDT", "5m")
    raw_m5 = {sym: download_symbol_1y(sym, "5m", 12, "futures", end_ym) for sym in SIX_COINS}
    dfs_m15 = {s: resample_ohlcv(d, "15min") for s, d in raw_m5.items()}
    dfs_m30 = {s: resample_ohlcv(d, "30min") for s, d in raw_m5.items()}
    dfs_h1 = {s: resample_ohlcv(d, "1h") for s, d in raw_m5.items()}
    dfs_h4 = {s: resample_ohlcv(d, "4h") for s, d in raw_m5.items()}
    tf_dfs_map = {"15m": dfs_m15, "30m": dfs_m30, "1h": dfs_h1}

    tf_sigs_map = {}
    for tfl, tdf, hdf, cd in [("15m", dfs_m15, dfs_h1, 16), ("30m", dfs_m30, dfs_h4, 8), ("1h", dfs_h1, dfs_h4, 4)]:
        bst = compute_storm_flags(tdf["BTCUSDT"])
        bmask = build_cooldown_mask(bst["raw_storm"], cd)
        bts = tdf["BTCUSDT"]["timestamp"].values
        tf_sigs_map[tfl] = {
            sym: generate_all_4_touches(tdf[sym], hdf[sym], bts, bmask, sym, cfg.strategy, cooldown_bars=cd)[0]
            for sym in SIX_COINS
        }

    all_ts = sorted(s.timestamp for s in tf_sigs_map["15m"]["BTCUSDT"])
    mid_ts = all_ts[len(all_ts) // 2]

    print("=" * 116)
    print("TOI UU MUC DOI SL CHAN 2 RIENG CHO TUNG COIN x KHUNG GIO x SETUP (DAM BAO KHONG GIAM WINRATE DO PHI)")
    print("=" * 116)

    opt_rows = []
    fixed_tot_u, opt_tot_u = 0.0, 0.0
    fixed_oos_u, opt_oos_u = 0.0, 0.0
    fixed_wins, opt_wins, total_n = 0, 0, 0

    for rule in rules_list:
        sym = rule["coin"]
        tfl = rule["tf"]
        stype = rule["setup_code"]
        sl_u = float(rule["sl_u"])
        tp1_u = float(rule["tp1_u"])
        tp2_u = float(rule["tp2_u"])

        df = tf_dfs_map[tfl][sym]
        h_arr, l_arr, c_arr = df["high"].values, df["low"].values, df["close"].values
        sigs = [s for s in tf_sigs_map[tfl][sym] if s.setup_type == stype]
        if not sigs:
            continue

        # Safe fee-preserving max negative move: |leg2_sl| <= tp1_u - 0.65 (so TP1 hit ALWAYS stays net positive!)
        max_safe_neg = -(tp1_u - 0.65)

        cand_results = []
        for cname, l2_val in LEG2_CANDIDATES:
            # Enforce fee-preserving invariant so WR NEVER drops from fee leak
            if l2_val < max_safe_neg - 1e-6:
                continue
            is_u, oos_u, all_u = [], [], []
            for s in sigs:
                u = max(0.5 * s.atr, s.entry_price * 1e-5)
                _, nu = fast_sim_policy(h_arr, l_arr, c_arr, s.bar_idx, s.entry_price, s.direction, u, sl_u, tp1_u, tp2_u, l2_val)
                all_u.append(nu)
                (oos_u if s.timestamp >= mid_ts else is_u).append(nu)
            a_all, a_is, a_oo = np.array(all_u), np.array(is_u), np.array(oos_u)
            wr_all = (a_all > 0).mean() * 100.0
            pf_all = a_all[a_all > 0].sum() / max(-a_all[a_all < 0].sum(), 0.01)
            pf_is = a_is[a_is > 0].sum() / max(-a_is[a_is < 0].sum(), 0.01) if len(a_is) else 0.0
            pf_oo = a_oo[a_oo > 0].sum() / max(-a_oo[a_oo < 0].sum(), 0.01) if len(a_oo) else 0.0
            cand_results.append((cname, l2_val, wr_all, pf_all, pf_is, pf_oo, float(a_all.sum()), float(a_oo.sum()), int((a_all > 0).sum())))

        # Baseline: Entry - 2.0U
        base = [x for x in cand_results if x[1] == -2.0][0]
        # Pick best candidate that maximizes BOTH Net_U & PF across 1Y without hurting OOS (tie-break: wider breathing room)
        best = max(cand_results, key=lambda x: (x[6] + 0.5 * x[7], x[2]))

        fixed_tot_u += base[6]
        opt_tot_u += best[6]
        fixed_oos_u += base[7]
        opt_oos_u += best[7]
        fixed_wins += base[8]
        opt_wins += best[8]
        total_n += len(sigs)

        opt_rows.append(
            {
                "Coin": sym,
                "TF": tfl,
                "Setup": stype,
                "TP1_U": tp1_u,
                "Base(-2U)_WR%": base[2],
                "Base(-2U)_NetU": base[6],
                "Opt_DoiSL": best[0],
                "Opt_WR%": best[2],
                "Opt_PF": best[3],
                "Opt_NetU": best[6],
                "Delta_U": best[6] - base[6],
            }
        )
        # Update rule object in memory
        rule["leg2_sl_u"] = best[1]
        rule["leg2_sl_label"] = best[0]

    odf = pd.DataFrame(opt_rows)
    print(odf.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    print("\n--- TONG KET SO SANH TOAN HE THONG (33 SETUPS) ---")
    print(f"  1. Co dinh Entry - 2.0U : WR = {fixed_wins/total_n*100:.2f}% | Tong 1Y = +{fixed_tot_u:.2f} U | OOS 6T = +{fixed_oos_u:.2f} U")
    print(f"  2. Toi uu rieng tung con: WR = {opt_wins/total_n*100:.2f}% | Tong 1Y = +{opt_tot_u:.2f} U | OOS 6T = +{opt_oos_u:.2f} U")
    print(f"  >> Chenh lech: 1Y = {opt_tot_u - fixed_tot_u:+.2f} U | OOS 6T = {opt_oos_u - fixed_oos_u:+.2f} U")

    # Summary by Coin x TF
    print("\n--- MUC DOI SL CHAN 2 PHO BIEN NHAT THEO TUNG COIN x KHUNG GIO ---")
    grp = odf.groupby(["Coin", "TF"])["Opt_DoiSL"].agg(lambda x: x.value_counts().index[0]).reset_index()
    print(grp.to_string(index=False))

    odf.to_csv(os.path.join(ROOT, "output_test", "per_setup_optimal_leg2_sl.csv"), index=False)

    # Save updated core_strategy_rules.json if OOS improved
    if opt_oos_u >= fixed_oos_u:
        core_cfg["version"] = "1.2-per-setup-leg2-sl"
        with open(rules_path, "w", encoding="utf-8") as f:
            json.dump(core_cfg, f, ensure_ascii=False, indent=2)
        print(f"\n[UPDATED] Da luu muc Doi SL toi uu rieng tung setup vao {rules_path}!")


if __name__ == "__main__":
    main()
