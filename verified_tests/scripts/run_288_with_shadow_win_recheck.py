"""Re-scan ALL 288 Setups (6 Coins x 5m/15m/30m/1h x EMA 34/89/200 x Touches 1-4)
WITH 2-SL -> Shadow-Win Pause/Resume filter to discover any newly rescued sweet spots!
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
from run_full_216_exit_sweetspot import COIN_CAL, sim_exit_arch
from run_full_3ema_storm_guard import build_cooldown_mask, compute_storm_flags, generate_all_4_touches
from run_master_6coin_html import SIX_COINS
from run_research_mtf import resample_ohlcv

EXIT_CANDS = [
    ("ONLY_TP1", "ONLY_TP1", 0.0),
    ("ONLY_1R", "ONLY_1R", 0.0),
    ("PARTIAL_NO_MOVE", "PARTIAL_NO_MOVE", 0.0),
    ("PARTIAL_BE+Phi", "PARTIAL_MOVE_SL", 0.3),
    ("PARTIAL_Entry0U", "PARTIAL_MOVE_SL", 0.0),
    ("PARTIAL_Entry-1U", "PARTIAL_MOVE_SL", -1.0),
    ("PARTIAL_Entry-2U", "PARTIAL_MOVE_SL", -2.0),
]


def apply_shadow_win_filter(sigs, outcomes, net_us, trigger_sl=2):
    """Apply 2-SL -> Shadow-Win filter sequentially (zero lookahead). Returns filtered indices."""
    taken_idx = []
    consec_sl = 0
    in_shadow = False
    for idx, (o, nu) in enumerate(zip(outcomes, net_us)):
        if in_shadow:
            if nu > 0:
                in_shadow = False
                consec_sl = 0
            continue
        taken_idx.append(idx)
        if o == "SL":
            consec_sl += 1
            if consec_sl >= trigger_sl:
                in_shadow = True
                consec_sl = 0
        elif nu > 0:
            consec_sl = 0
    return taken_idx


def main():
    with open(os.path.join(ROOT, "core", "core_strategy_rules.json"), encoding="utf-8") as f:
        old_cfg = json.load(f)
    old_keys = {(r["coin"], r["tf"], r["setup_code"]) for r in old_cfg["rules"]}

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

    newly_rescued_5m = []
    newly_rescued_mtf = []
    summary_by_tf_ema_touch = []

    for sym in SIX_COINS:
        cc = COIN_CAL[sym]
        for tfl in ["5m", "15m", "30m", "1h"]:
            df = tf_dfs_map[tfl][sym]
            h, l, c = df["high"].values, df["low"].values, df["close"].values
            sl_u = cc["sl"]
            tp1_u = 4.0 if tfl == "5m" else cc["tp1"][tfl]
            tp2_u = cc["tp2"]

            by_s = {}
            for s in tf_sigs_map[tfl][sym]:
                by_s.setdefault(s.setup_type, []).append(s)

            for stype, sigs in sorted(by_s.items()):
                if len(sigs) < 10:
                    continue
                best = None
                for lbl, arch, l2 in EXIT_CANDS:
                    outs, nus = [], []
                    for s in sigs:
                        u = max(0.5 * s.atr, s.entry_price * 1e-5)
                        o, _, _, _, nu = sim_exit_arch(h, l, c, s.bar_idx, s.entry_price, s.direction, u, sl_u, tp1_u, tp2_u, arch, l2)
                        outs.append(o); nus.append(nu)
                    taken = apply_shadow_win_filter(sigs, outs, nus, trigger_sl=2)
                    if len(taken) < 10:
                        continue
                    t_nus = np.array([nus[i] for i in taken])
                    is_nus = np.array([nus[i] for i in taken if sigs[i].timestamp < mid_ts])
                    oo_nus = np.array([nus[i] for i in taken if sigs[i].timestamp >= mid_ts])
                    if len(is_nus) < 5 or len(oo_nus) < 5:
                        continue
                    pf_all = t_nus[t_nus > 0].sum() / max(-t_nus[t_nus < 0].sum(), 0.01)
                    pf_is = is_nus[is_nus > 0].sum() / max(-is_nus[is_nus < 0].sum(), 0.01)
                    pf_oo = oo_nus[oo_nus > 0].sum() / max(-oo_nus[oo_nus < 0].sum(), 0.01)
                    wr_all = (t_nus > 0).mean() * 100.0
                    qual = (pf_is >= 1.15 and pf_oo >= 1.15 and pf_all >= 1.30 and wr_all >= 60.0)
                    cand = (qual, float(t_nus.sum()), lbl, len(taken), wr_all, pf_is, pf_oo, pf_all)
                    if best is None or (cand[0], cand[1]) > (best[0], best[1]):
                        best = cand

                if best is None:
                    continue
                ema_n = stype.split("_")[0]
                tch = int(stype.split("_W")[1][0])
                summary_by_tf_ema_touch.append({"TF": tfl, "EMA": ema_n, "Cham": tch, "Qual": best[0], "WR": best[4], "PF": best[7], "NetU": best[1]})

                if best[0] and (sym, tfl, stype) not in old_keys:
                    rec = {
                        "Coin": sym, "TF": tfl, "Setup": stype, "Best_Exit": best[2],
                        "Lenh_Sau_SW": best[3], "WR_%": best[4], "PF_IS": best[5], "PF_OOS": best[6], "PF_1Y": best[7], "Net_U": best[1],
                    }
                    if tfl == "5m":
                        newly_rescued_5m.append(rec)
                    else:
                        newly_rescued_mtf.append(rec)

    print("=" * 116)
    print("CHECK LAI 288 TO HOP (5m / 15m / 30m / 1h x EMA 34/89/200 x CHAM 1-4) VOI BO LOC 2-SL -> SHADOW-WIN")
    print("=" * 116)
    print(f"\n>>> [KHUNG 5m] Setup MOI dat chuan ben ca 2 nua nam nho Shadow-Win: {len(newly_rescued_5m)}")
    if newly_rescued_5m:
        print(pd.DataFrame(newly_rescued_5m).sort_values("PF_1Y", ascending=False).to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    print(f"\n>>> [KHUNG 15m / 30m / 1h] Setup MOI (ngoai 43 setup v2.0) duoc cuu song nho Shadow-Win: {len(newly_rescued_mtf)}")
    if newly_rescued_mtf:
        print(pd.DataFrame(newly_rescued_mtf).sort_values("PF_1Y", ascending=False).to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    sdf = pd.DataFrame(summary_by_tf_ema_touch)
    print("\n--- TONG HOP SO SETUP DAT CHUAN THEO KHUNG GIO x DUONG EMA ---")
    piv = sdf.groupby(["TF", "EMA"]).agg(Tong_To_Hop=("Qual", "count"), So_Dat_Chuan=("Qual", "sum"), WR_TB=("WR", "mean"), PF_TB=("PF", "mean")).reset_index()
    print(piv.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    print("\n--- TONG HOP SO SETUP DAT CHUAN THEO CHAM 1-2-3-4 (TREN 15m / 30m / 1h) ---")
    mtf_s = sdf[sdf["TF"] != "5m"]
    piv_t = mtf_s.groupby("Cham").agg(Tong_To_Hop=("Qual", "count"), So_Dat_Chuan=("Qual", "sum"), WR_TB=("WR", "mean"), PF_TB=("PF", "mean")).reset_index()
    print(piv_t.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))


if __name__ == "__main__":
    main()
