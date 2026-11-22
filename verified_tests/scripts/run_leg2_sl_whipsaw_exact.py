"""Exact Mathematical Optimization of Leg2 SL Move per Setup (33 Core+Scout Setups):
- Strict Winrate Preservation (ZERO fee-leak WR drop!)
- Measures exact Whipsaw Rate (% of TP1 winners that reverse to hit Leg2 SL before TP2)
- Measures exact TP2 Hit Rate (%) & Net Units after fees
- Selects optimal Leg2 SL per setup maximizing TP2 Hit Rate & Net_U at Maximum Winrate.
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

    rows = []
    tot_n = 0
    tot_wins_opt = 0
    tot_tp2_opt = 0
    tot_tp2_be = 0
    tot_net_opt = 0.0
    tot_net_be = 0.0

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

        cands = []
        for cname, l2_val in LEG2_CANDIDATES:
            outs, nets = [], []
            for s in sigs:
                u = max(0.5 * s.atr, s.entry_price * 1e-5)
                o, nu = fast_sim_policy(h_arr, l_arr, c_arr, s.bar_idx, s.entry_price, s.direction, u, sl_u, tp1_u, tp2_u, l2_val)
                outs.append(o)
                nets.append(nu)
            arr = np.array(nets)
            n = len(arr)
            wins = int((arr > 0).sum())
            wr = wins / n * 100.0
            tp2_cnt = sum(1 for o in outs if o == "TP2")
            be_cnt = sum(1 for o in outs if o == "BE")
            tp1_total = tp2_cnt + be_cnt + sum(1 for o in outs if o == "EOD" and nets[outs.index(o)] > 0)
            whipsaw_pct = (be_cnt / max(tp2_cnt + be_cnt, 1)) * 100.0
            tp2_pct = tp2_cnt / n * 100.0
            pf = arr[arr > 0].sum() / max(-arr[arr < 0].sum(), 0.01)
            cands.append(
                {
                    "name": cname,
                    "val": l2_val,
                    "wr": wr,
                    "wins": wins,
                    "tp2_cnt": tp2_cnt,
                    "tp2_pct": tp2_pct,
                    "whipsaw_pct": whipsaw_pct,
                    "pf": pf,
                    "net_u": float(arr.sum()),
                }
            )

        # Reference: Entry+Phi
        ref_be = cands[0]
        # Hard Constraint #1: MUST preserve Maximum Winrate (within 0.01% of highest WR among all candidates)
        max_wr = max(c["wr"] for c in cands)
        wr_safe = [c for c in cands if c["wr"] >= max_wr - 0.01]
        # Objective #2: Maximize Net_U, tie-break by highest TP2% and lowest Whipsaw%
        best = max(wr_safe, key=lambda c: (round(c["net_u"], 2), c["tp2_pct"], -c["whipsaw_pct"]))

        tot_n += len(sigs)
        tot_wins_opt += best["wins"]
        tot_tp2_opt += best["tp2_cnt"]
        tot_tp2_be += ref_be["tp2_cnt"]
        tot_net_opt += best["net_u"]
        tot_net_be += ref_be["net_u"]

        rule["leg2_sl_u"] = best["val"]
        rule["leg2_sl_label"] = best["name"]
        rule["leg2_metrics"] = {
            "whipsaw_pct_at_opt": round(best["whipsaw_pct"], 2),
            "whipsaw_pct_at_be_fee": round(ref_be["whipsaw_pct"], 2),
            "tp2_hit_pct": round(best["tp2_pct"], 2),
            "wr_preserved_pct": round(best["wr"], 2),
        }

        rows.append(
            {
                "Coin": sym,
                "TF": tfl,
                "Setup": stype,
                "Lenh": len(sigs),
                "TP1": f"{tp1_u:.1f}U",
                "TP2": f"{tp2_u:.0f}U",
                "Quet_Neu_BEPhi_%": ref_be["whipsaw_pct"],
                "Doi_SL_Toi_Uu": best["name"],
                "Quet_Sau_Toi_Uu_%": best["whipsaw_pct"],
                "Cham_TP2_%": best["tp2_pct"],
                "WR_Bao_Toan_%": best["wr"],
                "PF": best["pf"],
                "Net_U": best["net_u"],
            }
        )

    odf = pd.DataFrame(rows).sort_values(["Coin", "TF", "WR_Bao_Toan_%"], ascending=[True, True, False])
    print("=" * 126)
    print("MA TRAN TOI UU DOI SL CHAN 2 RIENG TUNG SETUP (33 SETUPS | BAO TOAN 100% WINRATE TOI DA + GIAM QUET SL CHAN 2)")
    print("=" * 126)
    print(odf.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    print("\n--- TONG KET TOAN HOC (SO SANH DOI SL VE ENTRY+PHI vs TOI UU RIENG TUNG SETUP) ---")
    print(f"  - Neu doi cung Entry+Phi : Cham TP2 = {tot_tp2_be/tot_n*100:.2f}% ({tot_tp2_be}/{tot_n}) | Tong lai = +{tot_net_be:.2f} U")
    print(f"  - Toi uu rieng tung setup: Cham TP2 = {tot_tp2_opt/tot_n*100:.2f}% ({tot_tp2_opt}/{tot_n}) | WR = {tot_wins_opt/tot_n*100:.2f}% (KHONG GIAM 1% WR!) | Tong lai = +{tot_net_opt:.2f} U ({tot_net_opt - tot_net_be:+.2f} U)")

    out_csv = os.path.join(ROOT, "core", "reports", "leg2_sl_whipsaw_matrix.csv")
    odf.to_csv(out_csv, index=False)

    core_cfg["version"] = "1.4-pareto-400u-exact-leg2-sl"
    with open(rules_path, "w", encoding="utf-8") as f:
        json.dump(core_cfg, f, ensure_ascii=False, indent=2)
    print(f"\n[SAVED] Updated {rules_path} (v1.4) & {out_csv}")


if __name__ == "__main__":
    main()
