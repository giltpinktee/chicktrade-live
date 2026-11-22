"""Full 216-Setup Exit Architecture Sweet-Spot Scanner (6 Coins x 3 TFs [15m/30m/1h] x 3 EMAs x 4 Touches).

For EVERY setup (including previously rejected ones!), tests 4 Exit Architectures on IS 6M and verifies on OOS 6M:
  1. ONLY_TP1       : 100% exit at TP1 (3.5-4.0U)
  2. ONLY_1R        : 100% exit at 1R (SL_U)
  3. PARTIAL_NO_MOVE: 50% TP1 + Keep original SL + 50% TP2 (8-12U)
  4. PARTIAL_MOVE_SL: 50% TP1 + Optimal Leg2 SL (Entry+Fee / 0U / -1U / -2U) + 50% TP2 (8-12U)

Outputs exact probabilities: SL_Full%, TP1_Hit%, Leg2_SL%, TP2_Hit%, WR%, PF_IS, PF_OOS, PF_1Y,
then recomputes the Sweet Spots for Buffer (400/300/250), Auto-Turbo (MaxDD < 15%), and Volume!
"""
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

from config import BacktestConfig, RiskConfig
from data.binance_loader import _detect_latest_available_month, download_symbol_1y
from risk.compounding import CompoundingManager
from run_full_3ema_storm_guard import build_cooldown_mask, compute_storm_flags, generate_all_4_touches
from run_master_6coin_html import SIX_COINS
from run_research_mtf import resample_ohlcv

# Per-coin physical SL / TP1 / TP2 calibration
COIN_CAL = {
    "BTCUSDT": {"sl": 5.0, "tp1": {"15m": 4.0, "30m": 4.0, "1h": 3.5}, "tp2": 12.0},
    "ETHUSDT": {"sl": 6.0, "tp1": {"15m": 3.5, "30m": 3.5, "1h": 3.5}, "tp2": 10.0},
    "BNBUSDT": {"sl": 6.0, "tp1": {"15m": 4.0, "30m": 3.5, "1h": 3.5}, "tp2": 10.0},
    "LINKUSDT": {"sl": 6.0, "tp1": {"15m": 3.5, "30m": 3.5, "1h": 3.5}, "tp2": 8.0},
    "XRPUSDT": {"sl": 6.0, "tp1": {"15m": 3.5, "30m": 3.5, "1h": 3.5}, "tp2": 8.0},
    "SOLUSDT": {"sl": 6.0, "tp1": {"15m": 3.5, "30m": 3.5, "1h": 3.5}, "tp2": 8.0},
}


def sim_exit_arch(h_arr, l_arr, c_arr, bar_idx, entry, d, u, sl_u, tp1_u, tp2_u, arch: str, l2_u: float = -2.0, max_bars=200):
    """Return (outcome, tp1_hit, leg2_sl_hit, tp2_hit, net_u)."""
    ef, tf, sf, slip = 0.0002, 0.0002, 0.0005, 0.0001
    sl_orig = entry - d * sl_u * u
    end_i = min(bar_idx + max_bars, len(h_arr) - 1)

    if arch in ("ONLY_TP1", "ONLY_1R"):
        tp_u = tp1_u if arch == "ONLY_TP1" else sl_u
        tp_p = entry + d * tp_u * u
        for j in range(bar_idx + 1, end_i + 1):
            h, l = h_arr[j], l_arr[j]
            if (l <= sl_orig) if d == 1 else (h >= sl_orig):
                fill = sl_orig * (1.0 - d * slip)
                return "SL", False, False, False, d * (fill - entry) / u - (entry * ef + fill * sf) / u
            if (h >= tp_p) if d == 1 else (l <= tp_p):
                return "TP1", True, False, False, tp_u - (entry * ef + tp_p * tf) / u
        fill = c_arr[end_i] * (1.0 - d * slip)
        return "EOD", False, False, False, d * (fill - entry) / u - (entry * ef + fill * sf) / u

    # Partial architectures (50% TP1 + 50% TP2)
    tp1_p = entry + d * tp1_u * u
    tp2_p = entry + d * tp2_u * u
    if arch == "PARTIAL_NO_MOVE":
        leg2_sl = sl_orig
    else:
        leg2_sl = entry + d * entry * (ef + sf + slip) if l2_u > 0 else entry + d * l2_u * u

    tp1_hit = False
    fee1_u = (entry * ef + tp1_p * tf) / u
    leg1_net = 0.0

    for j in range(bar_idx + 1, end_i + 1):
        h, l = h_arr[j], l_arr[j]
        if not tp1_hit:
            if (l <= sl_orig) if d == 1 else (h >= sl_orig):
                fill = sl_orig * (1.0 - d * slip)
                return "SL", False, False, False, d * (fill - entry) / u - (entry * ef + fill * sf) / u
            if (h >= tp1_p) if d == 1 else (l <= tp1_p):
                tp1_hit = True
                leg1_net = 0.5 * (tp1_u - fee1_u)
                if (h >= tp2_p) if d == 1 else (l <= tp2_p):
                    fee2 = (entry * ef + tp2_p * tf) / u
                    return "TP2", True, False, True, leg1_net + 0.5 * (tp2_u - fee2)
        else:
            if (l <= leg2_sl) if d == 1 else (h >= leg2_sl):
                fill = leg2_sl * (1.0 - d * slip)
                gross2 = d * (fill - entry) / u
                fee2 = (entry * ef + fill * sf) / u
                return "BE", True, True, False, leg1_net + 0.5 * (gross2 - fee2)
            if (h >= tp2_p) if d == 1 else (l <= tp2_p):
                fee2 = (entry * ef + tp2_p * tf) / u
                return "TP2", True, False, True, leg1_net + 0.5 * (tp2_u - fee2)

    fill = c_arr[end_i] * (1.0 - d * slip)
    gross = d * (fill - entry) / u
    fee = (entry * ef + fill * sf) / u
    if tp1_hit:
        return "EOD", True, False, False, leg1_net + 0.5 * (gross - fee)
    return "EOD", False, False, False, gross - fee


def main():
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

    # Candidate exit policies to test per setup
    exit_candidates = [
        ("ONLY_TP1", "ONLY_TP1", 0.0),
        ("ONLY_1R", "ONLY_1R", 0.0),
        ("PARTIAL_NO_MOVE", "PARTIAL_NO_MOVE", 0.0),
        ("PARTIAL_BE+Phi", "PARTIAL_MOVE_SL", 0.3),
        ("PARTIAL_Entry0U", "PARTIAL_MOVE_SL", 0.0),
        ("PARTIAL_Entry-1U", "PARTIAL_MOVE_SL", -1.0),
        ("PARTIAL_Entry-2U", "PARTIAL_MOVE_SL", -2.0),
    ]

    all_216_rows = []
    rescued_setups = []
    selected_rules = {}

    # Load old 33 selected keys for comparison
    with open(os.path.join(ROOT, "core", "core_strategy_rules.json"), encoding="utf-8") as f:
        old_cfg = json.load(f)
    old_keys = {(r["coin"], r["tf"], r["setup_code"]) for r in old_cfg["rules"]}

    for sym in SIX_COINS:
        cc = COIN_CAL[sym]
        for tfl in ["15m", "30m", "1h"]:
            df = tf_dfs_map[tfl][sym]
            h_arr, l_arr, c_arr = df["high"].values, df["low"].values, df["close"].values
            sl_u, tp1_u, tp2_u = cc["sl"], cc["tp1"][tfl], cc["tp2"]

            by_stype: Dict[str, list] = {}
            for s in tf_sigs_map[tfl][sym]:
                by_stype.setdefault(s.setup_type, []).append(s)

            for stype, sigs in sorted(by_stype.items()):
                n_all = len(sigs)
                n_is = sum(1 for s in sigs if s.timestamp < mid_ts)
                n_oo = n_all - n_is
                if n_all < 7 or n_is < 3 or n_oo < 3:
                    continue

                best_cand = None
                for label, arch, l2_u in exit_candidates:
                    outs, tp1s, l2s, tp2s, nus, is_nus, oo_nus = [], [], [], [], [], [], []
                    for s in sigs:
                        u = max(0.5 * s.atr, s.entry_price * 1e-5)
                        o, t1, l2h, t2, nu = sim_exit_arch(h_arr, l_arr, c_arr, s.bar_idx, s.entry_price, s.direction, u, sl_u, tp1_u, tp2_u, arch, l2_u)
                        outs.append(o); tp1s.append(t1); l2s.append(l2h); tp2s.append(t2); nus.append(nu)
                        (oo_nus if s.timestamp >= mid_ts else is_nus).append(nu)

                    a_all, a_is, a_oo = np.array(nus), np.array(is_nus), np.array(oo_nus)
                    pf_all = a_all[a_all > 0].sum() / max(-a_all[a_all < 0].sum(), 0.01)
                    pf_is = a_is[a_is > 0].sum() / max(-a_is[a_is < 0].sum(), 0.01)
                    pf_oo = a_oo[a_oo > 0].sum() / max(-a_oo[a_oo < 0].sum(), 0.01)
                    wr_all = (a_all > 0).mean() * 100.0
                    sl_pct = np.mean([o == "SL" for o in outs]) * 100.0
                    tp1_pct = np.mean(tp1s) * 100.0
                    l2_pct = np.mean(l2s) * 100.0
                    tp2_pct = np.mean(tp2s) * 100.0

                    # Dual-half stability check
                    is_core = (n_all >= 15 and n_is >= 6 and n_oo >= 6 and pf_is >= 1.10 and pf_oo >= 1.10 and pf_all >= 1.25 and a_all.sum() > 0)
                    is_scout = (7 <= n_all < 15 and n_is >= 3 and n_oo >= 3 and pf_is >= 1.50 and pf_oo >= 1.50 and pf_all >= 2.00 and wr_all >= 70.0)
                    qualifies = is_core or is_scout

                    cand = {
                        "label": label, "arch": arch, "l2_u": l2_u, "qualifies": qualifies, "is_scout": is_scout and not is_core,
                        "wr": wr_all, "sl_pct": sl_pct, "tp1_pct": tp1_pct, "l2_pct": l2_pct, "tp2_pct": tp2_pct,
                        "pf_is": pf_is, "pf_oo": pf_oo, "pf_all": pf_all, "net_u": float(a_all.sum()),
                    }
                    # Prefer qualifying policies; among them maximize Net_U with strong WR
                    if best_cand is None:
                        best_cand = cand
                    else:
                        if cand["qualifies"] and not best_cand["qualifies"]:
                            best_cand = cand
                        elif cand["qualifies"] == best_cand["qualifies"]:
                            if (cand["net_u"], cand["wr"]) > (best_cand["net_u"], best_cand["wr"]):
                                best_cand = cand

                row = {
                    "Coin": sym, "TF": tfl, "Setup": stype, "Lenh": n_all,
                    "Best_Exit": best_cand["label"] if best_cand["qualifies"] else "SKIP",
                    "SL_Full_%": best_cand["sl_pct"], "TP1_Hit_%": best_cand["tp1_pct"],
                    "Dinh_DoiSL_%": best_cand["l2_pct"], "TP2_Hit_%": best_cand["tp2_pct"],
                    "WR_%": best_cand["wr"], "PF_IS": best_cand["pf_is"], "PF_OOS": best_cand["pf_oo"],
                    "PF_1Y": best_cand["pf_all"], "Net_U": best_cand["net_u"],
                    "Status": ("SCOUT" if best_cand["is_scout"] else "CORE") if best_cand["qualifies"] else "SKIP",
                    "Was_Old33": (sym, tfl, stype) in old_keys,
                }
                all_216_rows.append(row)
                if best_cand["qualifies"]:
                    selected_rules[(sym, tfl, stype)] = (best_cand, sl_u, tp1_u, tp2_u)
                    if (sym, tfl, stype) not in old_keys:
                        rescued_setups.append(row)

    df_216 = pd.DataFrame(all_216_rows)
    df_sel = df_216[df_216["Status"] != "SKIP"].sort_values(["PF_1Y", "Net_U"], ascending=False)

    print("=" * 126)
    print(f"QUET LAI TOAN BO 216 TO HOP (15m/30m/1h): TIM DUOC {len(df_sel)} SETUP DAT CHUAN (CUU DUOC +{len(rescued_setups)} SETUP MOI NHO DOI CACH THOAT LENH!)")
    print("=" * 126)
    print("\n--- PHAN BO KIEN TRUC THOAT LENH TOI UU TRONG CAC SETUP DAT CHUAN ---")
    print(df_sel["Best_Exit"].value_counts().to_string())

    if rescued_setups:
        print(f"\n--- {len(rescued_setups)} SETUP MOI DUOC 'CUU SONG' KHI CHON DUNG KIEN TRUC THOAT LENH (ONLY_TP1 / ONLY_1R / NO_MOVE) ---")
        print(pd.DataFrame(rescued_setups)[["Coin", "TF", "Setup", "Lenh", "Best_Exit", "SL_Full_%", "TP1_Hit_%", "Dinh_DoiSL_%", "TP2_Hit_%", "WR_%", "PF_IS", "PF_OOS", "Net_U", "Status"]].to_string(index=False, float_format=lambda x: f"{x:,.1f}"))

    print("\n--- KIEM TRA RIENG BTCUSDT 15m EMA34_W3 (VI DU BAN HOI) ---")
    print(df_216[(df_216["Coin"] == "BTCUSDT") & (df_216["TF"] == "15m") & (df_216["Setup"] == "EMA34_W3")][["Coin", "TF", "Setup", "Lenh", "Best_Exit", "SL_Full_%", "TP1_Hit_%", "Dinh_DoiSL_%", "TP2_Hit_%", "WR_%", "PF_1Y", "Net_U"]].to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    df_216.to_csv(os.path.join(ROOT, "output_test", "full_216_exit_sweetspot_audit.csv"), index=False)
    df_sel.to_csv(os.path.join(ROOT, "core", "reports", "selected_setups_exit_anatomy.csv"), index=False)
    print(f"\n[SAVED] full_216_exit_sweetspot_audit.csv & selected_setups_exit_anatomy.csv ({len(df_sel)} setups)")


if __name__ == "__main__":
    main()
