"""Compare Exit Policies for Lower-WR Core Setups (WR < 74%):
  1. Single 1R (100% @ TP = SL)
  2. Single Quick TP1 (100% @ TP1 = 3.5-4.0U)
  3. Partial Short (50% @ TP1 + Opt Leg2 SL + 50% @ 1R)
  4. Partial Far (50% @ TP1 + Opt Leg2 SL + 50% @ TP2 = 8-12U, current v1.4)
"""
import json
import os
import sys

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


def sim_single_tp(h_arr, l_arr, c_arr, bar_idx, entry, d, u, sl_u, tp_u, max_bars=200):
    entry_fee, tp_fee, sl_fee, slip = 0.0002, 0.0002, 0.0005, 0.0001
    sl_p = entry - d * sl_u * u
    tp_p = entry + d * tp_u * u
    end_i = min(bar_idx + max_bars, len(h_arr) - 1)
    for j in range(bar_idx + 1, end_i + 1):
        h, l = h_arr[j], l_arr[j]
        if (l <= sl_p) if d == 1 else (h >= sl_p):
            fill = sl_p * (1.0 - d * slip)
            gross = d * (fill - entry) / u
            fee = (entry * entry_fee + fill * sl_fee) / u
            return "SL", gross - fee
        if (h >= tp_p) if d == 1 else (l <= tp_p):
            fee = (entry * entry_fee + tp_p * tp_fee) / u
            return "TP", tp_u - fee
    fill = c_arr[end_i] * (1.0 - d * slip)
    gross = d * (fill - entry) / u
    fee = (entry * entry_fee + fill * sl_fee) / u
    return "EOD", gross - fee


def main():
    with open(os.path.join(ROOT, "core", "core_strategy_rules.json"), encoding="utf-8") as f:
        core_cfg = json.load(f)

    # Filter setups with WR < 74% (the lower-WR tier among our 33 setups)
    lower_rules = [r for r in core_cfg["rules"] if float(r["prior_1y"]["wr_pct"]) < 74.0]

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

    rows = []
    totals = {"1R_Single": [], "TP1_Single": [], "Partial_1R": [], "Partial_Far": []}

    for r in lower_rules:
        sym, tfl, stype = r["coin"], r["tf"], r["setup_code"]
        sl_u, tp1_u, tp2_u, l2_u = float(r["sl_u"]), float(r["tp1_u"]), float(r["tp2_u"]), float(r["leg2_sl_u"])
        df = tf_dfs_map[tfl][sym]
        h_arr, l_arr, c_arr = df["high"].values, df["low"].values, df["close"].values
        sigs = [s for s in tf_sigs_map[tfl][sym] if s.setup_type == stype]

        u_1r, u_tp1, u_p1r, u_pfar = [], [], [], []
        for s in sigs:
            u = max(0.5 * s.atr, s.entry_price * 1e-5)
            _, n1 = sim_single_tp(h_arr, l_arr, c_arr, s.bar_idx, s.entry_price, s.direction, u, sl_u, sl_u)
            _, n2 = sim_single_tp(h_arr, l_arr, c_arr, s.bar_idx, s.entry_price, s.direction, u, sl_u, tp1_u)
            _, n3 = fast_sim_policy(h_arr, l_arr, c_arr, s.bar_idx, s.entry_price, s.direction, u, sl_u, tp1_u, sl_u, l2_u)
            _, n4 = fast_sim_policy(h_arr, l_arr, c_arr, s.bar_idx, s.entry_price, s.direction, u, sl_u, tp1_u, tp2_u, l2_u)
            u_1r.append(n1); u_tp1.append(n2); u_p1r.append(n3); u_pfar.append(n4)
            totals["1R_Single"].append(n1)
            totals["TP1_Single"].append(n2)
            totals["Partial_1R"].append(n3)
            totals["Partial_Far"].append(n4)

        best_mode = max(
            [("1R_Single", sum(u_1r)), ("TP1_Single", sum(u_tp1)), ("Partial_1R", sum(u_p1r)), ("Partial_Far", sum(u_pfar))],
            key=lambda x: x[1],
        )[0]
        rows.append(
            {
                "Coin": sym,
                "TF": tfl,
                "Setup": stype,
                "Lenh": len(sigs),
                "1R_1Cuc_U": sum(u_1r),
                "1R_WR%": np.mean([x > 0 for x in u_1r]) * 100,
                "TP1_1Cuc_U": sum(u_tp1),
                "Part_1R_U": sum(u_p1r),
                "Part_Xa_U": sum(u_pfar),
                "PartXa_WR%": np.mean([x > 0 for x in u_pfar]) * 100,
                "Toi_Uu": best_mode,
            }
        )

    print("=" * 116)
    print("SO SANH 4 CACH CHOT LOI CHO NHOM SETUP WINRATE THAP HON (<74%): AN 1R vs CHOT TP1 vs CHIA 2 CHAN")
    print("=" * 116)
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    print("\n--- TONG HOP TOAN BO NHOM WR < 74% ---")
    for k, vals in totals.items():
        arr = np.array(vals)
        wr = (arr > 0).mean() * 100.0
        pf = arr[arr > 0].sum() / max(-arr[arr < 0].sum(), 0.01)
        print(f"  {k:12s}: Tong_Net_U = {arr.sum():+8.2f} U | WR = {wr:5.2f}% | PF = {pf:.2f}")


if __name__ == "__main__":
    main()
