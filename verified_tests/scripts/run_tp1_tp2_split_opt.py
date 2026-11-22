"""Optimize TP1/TP2 Position Split Ratio (0/100, 30/70, 50/50, 70/30, 100/0) across all 33 Core+Scout Setups."""
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
from run_full_3ema_storm_guard import build_cooldown_mask, compute_storm_flags, generate_all_4_touches
from run_master_6coin_html import SIX_COINS
from run_research_mtf import resample_ohlcv


def sim_split_ratio(h_arr, l_arr, c_arr, bar_idx, entry, d, u, sl_u, tp1_u, tp2_u, l2_u, tp1_frac: float, max_bars=200):
    """Simulate arbitrary TP1/TP2 split ratio (tp1_frac in [0.0 .. 1.0])."""
    entry_fee, tp_fee, sl_fee, slip = 0.0002, 0.0002, 0.0005, 0.0001
    sl_orig = entry - d * sl_u * u
    tp1_p = entry + d * tp1_u * u
    tp2_p = entry + d * tp2_u * u
    leg2_sl = entry + d * entry * (entry_fee + sl_fee + slip) if l2_u > 0 else entry + d * l2_u * u

    end_i = min(bar_idx + max_bars, len(h_arr) - 1)

    # Special case: 0% TP1 (Full 100% hold straight to TP2 with original SL)
    if tp1_frac <= 0.0:
        for j in range(bar_idx + 1, end_i + 1):
            h, l = h_arr[j], l_arr[j]
            if (l <= sl_orig) if d == 1 else (h >= sl_orig):
                fill = sl_orig * (1.0 - d * slip)
                gross = d * (fill - entry) / u
                fee = (entry * entry_fee + fill * sl_fee) / u
                return "SL", gross - fee
            if (h >= tp2_p) if d == 1 else (l <= tp2_p):
                fee = (entry * entry_fee + tp2_p * tp_fee) / u
                return "TP2", tp2_u - fee
        fill = c_arr[end_i] * (1.0 - d * slip)
        return "EOD", d * (fill - entry) / u - (entry * entry_fee + fill * sl_fee) / u

    # Special case: 100% TP1 (Full 100% exit at TP1)
    if tp1_frac >= 1.0:
        for j in range(bar_idx + 1, end_i + 1):
            h, l = h_arr[j], l_arr[j]
            if (l <= sl_orig) if d == 1 else (h >= sl_orig):
                fill = sl_orig * (1.0 - d * slip)
                gross = d * (fill - entry) / u
                fee = (entry * entry_fee + fill * sl_fee) / u
                return "SL", gross - fee
            if (h >= tp1_p) if d == 1 else (l <= tp1_p):
                fee = (entry * entry_fee + tp1_p * tp_fee) / u
                return "TP1", tp1_u - fee
        fill = c_arr[end_i] * (1.0 - d * slip)
        return "EOD", d * (fill - entry) / u - (entry * entry_fee + fill * sl_fee) / u

    # General 2-leg split (tp1_frac at TP1 + (1 - tp1_frac) at TP2/Leg2_SL)
    tp1_hit = False
    fee1_u = (entry * entry_fee + tp1_p * tp_fee) / u
    leg1_net = 0.0
    rem = 1.0 - tp1_frac

    for j in range(bar_idx + 1, end_i + 1):
        h, l = h_arr[j], l_arr[j]
        if not tp1_hit:
            if (l <= sl_orig) if d == 1 else (h >= sl_orig):
                fill = sl_orig * (1.0 - d * slip)
                gross = d * (fill - entry) / u
                fee = (entry * entry_fee + fill * sl_fee) / u
                return "SL", gross - fee
            if (h >= tp1_p) if d == 1 else (l <= tp1_p):
                tp1_hit = True
                leg1_net = tp1_frac * (tp1_u - fee1_u)
                if (h >= tp2_p) if d == 1 else (l <= tp2_p):
                    fee2 = (entry * entry_fee + tp2_p * tp_fee) / u
                    return "TP2", leg1_net + rem * (tp2_u - fee2)
        else:
            if (l <= leg2_sl) if d == 1 else (h >= leg2_sl):
                fill = leg2_sl * (1.0 - d * slip)
                gross2 = d * (fill - entry) / u
                fee2 = (entry * entry_fee + fill * sl_fee) / u
                return "BE", leg1_net + rem * (gross2 - fee2)
            if (h >= tp2_p) if d == 1 else (l <= tp2_p):
                fee2 = (entry * entry_fee + tp2_p * tp_fee) / u
                return "TP2", leg1_net + rem * (tp2_u - fee2)

    fill = c_arr[end_i] * (1.0 - d * slip)
    gross = d * (fill - entry) / u
    fee = (entry * entry_fee + fill * sl_fee) / u
    return ("EOD", leg1_net + rem * (gross - fee)) if tp1_hit else ("EOD", gross - fee)


def main():
    with open(os.path.join(ROOT, "core", "core_strategy_rules.json"), encoding="utf-8") as f:
        core_cfg = json.load(f)
    rules = core_cfg["rules"]

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

    splits = [
        ("0% TP1 / 100% TP2 (Khong TP1)", 0.0),
        ("30% TP1 / 70% TP2", 0.30),
        ("40% TP1 / 60% TP2", 0.40),
        ("50% TP1 / 50% TP2 (Hien tai)", 0.50),
        ("60% TP1 / 40% TP2", 0.60),
        ("70% TP1 / 30% TP2", 0.70),
        ("100% TP1 / 0% TP2 (Khong TP2)", 1.00),
    ]

    summary = []
    for label, frac in splits:
        nets = []
        consec_losses = []
        cur_loss = 0
        for r in rules:
            sym, tfl, stype = r["coin"], r["tf"], r["setup_code"]
            sl_u, tp1_u, tp2_u, l2_u = float(r["sl_u"]), float(r["tp1_u"]), float(r["tp2_u"]), float(r["leg2_sl_u"])
            df = tf_dfs_map[tfl][sym]
            h_arr, l_arr, c_arr = df["high"].values, df["low"].values, df["close"].values
            for s in tf_sigs_map[tfl][sym]:
                if s.setup_type != stype:
                    continue
                u = max(0.5 * s.atr, s.entry_price * 1e-5)
                _, nu = sim_split_ratio(h_arr, l_arr, c_arr, s.bar_idx, s.entry_price, s.direction, u, sl_u, tp1_u, tp2_u, l2_u, frac)
                nets.append(nu)
                if nu < 0:
                    cur_loss += 1
                    consec_losses.append(cur_loss)
                else:
                    cur_loss = 0
        arr = np.array(nets)
        wr = (arr > 0).mean() * 100.0
        pos = arr[arr > 0].sum()
        neg = -arr[arr < 0].sum()
        pf = pos / max(neg, 0.01)
        avg_w = arr[arr > 0].mean()
        avg_l = -arr[arr < 0].mean()
        summary.append(
            {
                "Ty_Le_TP1_TP2": label,
                "WR_%": wr,
                "PF": pf,
                "AvgWin_U": avg_w,
                "AvgLoss_U": avg_l,
                "Lenh_Go_1SL": avg_l / max(avg_w, 0.01),
                "Max_Chuoi_SL": max(consec_losses) if consec_losses else 0,
                "Tong_Lai_1Y_U": arr.sum(),
            }
        )

    print("=" * 114)
    print("TOI UU TY LE CHIA LENH TP1 / TP2 (TOAN BO 33 SETUPS — 824 LENH / 1 NAM, DA TRU DU PHI BINANCE)")
    print("=" * 114)
    print(pd.DataFrame(summary).to_string(index=False, float_format=lambda x: f"{x:,.2f}"))


if __name__ == "__main__":
    main()
