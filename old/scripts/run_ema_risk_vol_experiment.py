"""Quantitative Experiment:
1. Measure Risk Profile per EMA (EMA34 vs EMA89 vs EMA200) across 1-year data.
2. Compute 2-Layer Volume Formula: Vol = WR_Tier_Mult * EMA_Risk_Mult * Stability_Mult.
3. Empirically test whether adding Lower-WR / High-Risk setups (WR 58-65% or 1-half-unstable with PF_1Y >= 1.10)
   at 0.00x (SKIP) vs 0.15x vs 0.25x vs 0.35x improves or hurts 12M & OOS 6M Return, WR, PF, and MaxDD!
"""
import os
import sys
from typing import Dict

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
from run_master_6coin_html import (
    SIX_COINS,
    simulate_6coin_portfolio,
)
from run_research_mtf import resample_ohlcv


def wr_tier_mult(wr_pct: float) -> float:
    if wr_pct >= 80.0:
        return 1.20
    if wr_pct >= 75.0:
        return 1.00
    if wr_pct >= 70.0:
        return 0.80
    if wr_pct >= 65.0:
        return 0.60
    if wr_pct >= 60.0:
        return 0.40
    return 0.0


def main():
    audit_csv = os.path.join(ROOT, "output_test", "master_all_288_setups_audit.csv")
    df_all = pd.read_csv(audit_csv)
    df_mtf = df_all[df_all["TF"].isin(["15m", "30m", "1h"]) & (df_all["Trades_1Y"] >= 7)].copy()

    # 1. Measure empirical risk profile per EMA (EMA34 vs EMA89 vs EMA200) on 15m/30m/1h
    print("=" * 108)
    print("PHAN 1: DANH GIA RUI RO RIENG TUNG DUONG EMA (34 / 89 / 200) TREN 1 NAM DU LIEU (15m / 30m / 1h)")
    print("=" * 108)
    ema_stats = []
    for ema in ["EMA34", "EMA89", "EMA200"]:
        sub = df_mtf[df_mtf["EMA"] == ema]
        sel = sub[sub["Selected"]]
        dual_pass_rate = ((sub["PF_IS6M"] >= 1.08) & (sub["PF_OOS6M"] >= 1.08)).mean() * 100.0
        avg_sl_all = sub["SL_Full_%"].mean()
        avg_sl_sel = sel["SL_Full_%"].mean() if len(sel) else 0.0
        avg_wr_all = sub["WR_1Y_%"].mean()
        avg_pf_all = sub["PF_1Y"].mean()
        avg_pf_sel = sel["PF_1Y"].mean() if len(sel) else 0.0
        ema_stats.append(
            {
                "EMA": ema,
                "To_Hop_MTF": len(sub),
                "So_Setup_Dat": len(sel),
                "Ty_Le_Ben_2_Nua_%": dual_pass_rate,
                "SL_Full_TB_All_%": avg_sl_all,
                "SL_Full_TB_Dat_%": avg_sl_sel,
                "WR_TB_All_%": avg_wr_all,
                "PF_TB_All": avg_pf_all,
                "PF_TB_Dat": avg_pf_sel,
            }
        )
    ema_df = pd.DataFrame(ema_stats)
    print(ema_df.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    # Derive data-driven EMA Risk Multipliers:
    # EMA with higher dual-half pass rate & lower SL_Full gets 1.00x; riskier EMAs get slight discount
    ema_risk_mult = {}
    max_pass = ema_df["Ty_Le_Ben_2_Nua_%"].max()
    for _, r in ema_df.iterrows():
        # Blend stability ratio + inverse SL rate, bounded [0.80, 1.00]
        rel = 0.80 + 0.20 * (r["Ty_Le_Ben_2_Nua_%"] / max(max_pass, 1.0))
        ema_risk_mult[r["EMA"]] = round(min(1.0, max(0.80, rel)), 2)
    print("\n>>> He so dieu chinh rui ro theo tung duong EMA (tu so lieu 1 nam):", ema_risk_mult)

    # 2. Identify "Lower-WR / High-Risk but Positive 1Y" candidate setups (not in 33 Selected)
    # e.g. WR >= 58%, PF_1Y >= 1.10, Trades_1Y >= 15 (includes BTC 15m EMA34 W1!)
    risky_pool = df_mtf[
        (~df_mtf["Selected"])
        & (df_mtf["Trades_1Y"] >= 15)
        & (df_mtf["WR_1Y_%"] >= 58.0)
        & (df_mtf["PF_1Y"] >= 1.10)
    ].copy()
    print(f"\n>>> Tim thay {len(risky_pool)} setup 'Rui ro cao / Gay 1 nua nam nhung tong 1Y van duong (WR>=58%, PF_1Y>=1.10)':")
    print(
        risky_pool[["Coin", "TF", "Setup_Code", "Trades_1Y", "WR_IS_%", "WR_OOS_%", "WR_1Y_%", "SL_Full_%", "PF_IS6M", "PF_OOS6M", "PF_1Y", "Net_U_1Y"]]
        .sort_values("Net_U_1Y", ascending=False)
        .head(12)
        .to_string(index=False, float_format=lambda x: f"{x:,.2f}")
    )

    # 3. Load cached 1Y market data for portfolio simulation
    cfg = BacktestConfig()
    end_ym = _detect_latest_available_month("BTCUSDT", "5m")
    raw_m5 = {sym: download_symbol_1y(sym, "5m", 12, "futures", end_ym) for sym in SIX_COINS}
    dfs_m15 = {s: resample_ohlcv(d, "15min") for s, d in raw_m5.items()}
    dfs_m30 = {s: resample_ohlcv(d, "30min") for s, d in raw_m5.items()}
    dfs_h1 = {s: resample_ohlcv(d, "1h") for s, d in raw_m5.items()}
    dfs_h4 = {s: resample_ohlcv(d, "4h") for s, d in raw_m5.items()}
    tf_dfs_map = {"M15": dfs_m15, "M30": dfs_m30, "H1": dfs_h1}

    tf_sigs_map = {}
    for tf_name, tdf, hdf, cd in [("M15", dfs_m15, dfs_h1, 16), ("M30", dfs_m30, dfs_h4, 8), ("H1", dfs_h1, dfs_h4, 4)]:
        bst = compute_storm_flags(tdf["BTCUSDT"])
        bmask = build_cooldown_mask(bst["raw_storm"], cd)
        bts = tdf["BTCUSDT"]["timestamp"].values
        tf_sigs_map[tf_name] = {
            sym: generate_all_4_touches(tdf[sym], hdf[sym], bts, bmask, sym, cfg.strategy, cooldown_bars=cd)[0]
            for sym in SIX_COINS
        }

    all_ts = sorted(s.timestamp for s in tf_sigs_map["M15"]["BTCUSDT"])
    mid_ts = all_ts[len(all_ts) // 2]

    # Build base 33-setup rules using 2-Layer Formula: Vol = WR_Tier_Mult * EMA_Risk_Mult (Scouts capped at 0.45x)
    def _parse_u(val) -> float:
        return float(str(val).split("U")[0])

    def build_rule_dict(risky_vol: float) -> dict:
        rules = {}
        # 33 Core + Scout setups
        for _, r in df_mtf[df_mtf["Selected"]].iterrows():
            sym, tfc, stype, ema = r["Coin"], r["TF_Code"], r["Setup_Code"], r["EMA"]
            is_scout = int(r["Trades_1Y"]) < 15
            base_v = wr_tier_mult(float(r["WR_1Y_%"])) * ema_risk_mult[ema]
            mult = round(min(base_v, 0.45) if is_scout else base_v, 2)
            rules[(sym, tfc, stype)] = {
                "mult": mult,
                "sl_u": _parse_u(r["SL_U"]),
                "tp1_u": _parse_u(r["TP1_U"]),
                "tp2_u": _parse_u(r["TP2_U"]),
                "leg2_mode": "SL_MINUS_2U",
                "tier_label": "Scout" if is_scout else "Core",
                "wr_1y": float(r["WR_1Y_%"]),
                "pf_1y": float(r["PF_1Y"]),
            }
        # Optionally add risky_pool setups at `risky_vol` * ema_risk_mult
        if risky_vol > 0:
            for _, r in risky_pool.iterrows():
                sym, tfc, stype, ema = r["Coin"], r["TF_Code"], r["Setup_Code"], r["EMA"]
                mult = round(risky_vol * ema_risk_mult[ema], 2)
                if mult < 0.12:
                    continue
                rules[(sym, tfc, stype)] = {
                    "mult": mult,
                    "sl_u": _parse_u(r["SL_U"]),
                    "tp1_u": _parse_u(r["TP1_U"]),
                    "tp2_u": _parse_u(r["TP2_U"]),
                    "leg2_mode": "SL_MINUS_2U",
                    "tier_label": f"Risky_{risky_vol:.2f}x",
                    "wr_1y": float(r["WR_1Y_%"]),
                    "pf_1y": float(r["PF_1Y"]),
                }
        return rules

    print("\n" + "=" * 116)
    print("PHAN 2: THUC NGHIEM — CO NEN DANH THEM NHOM RUI RO CAO / WINRATE THAP (0.00x vs 0.15x vs 0.25x vs 0.35x)?")
    print("(Test tren Bao 400 gia [Max 5 lenh] & Bao 300 gia [Max 3 lenh] | 12T Full & 6T OOS)")
    print("=" * 116)

    exp_rows = []
    for buf, mc in [(400.0, 5), (300.0, 3)]:
        cap = round(mc * 0.85, 2)
        for label, rv in [
            ("A. BO HAN (0.00x — Chi 33 setup chuan)", 0.0),
            ("B. Them nhom Rui ro cao @ 0.15x", 0.15),
            ("C. Them nhom Rui ro cao @ 0.25x", 0.25),
            ("D. Them nhom Rui ro cao @ 0.35x", 0.35),
        ]:
            rules = build_rule_dict(rv)
            r12 = simulate_6coin_portfolio(tf_dfs_map, tf_sigs_map, rules, buffer_units=buf, max_concurrent=mc, max_trades_per_day=10, open_mult_cap=cap, start_time=None)
            r6 = simulate_6coin_portfolio(tf_dfs_map, tf_sigs_map, rules, buffer_units=buf, max_concurrent=mc, max_trades_per_day=10, open_mult_cap=cap, start_time=mid_ts)
            calmar_12 = r12["return_pct"] / max(abs(r12["max_dd_pct"]), 0.1)
            exp_rows.append(
                {
                    "Bao": f"{int(buf)}G(M{mc})",
                    "Kich_Ban": label,
                    "Setups": len(rules),
                    "12M_Lenh": r12["total_trades"],
                    "12M_WR%": r12["winrate_pct"],
                    "12M_PF": r12["pf"],
                    "12M_Lai%": r12["return_pct"],
                    "12M_MaxDD%": r12["max_dd_pct"],
                    "Calmar": calmar_12,
                    "OOS_WR%": r6["winrate_pct"],
                    "OOS_Lai%": r6["return_pct"],
                    "OOS_MaxDD%": r6["max_dd_pct"],
                }
            )

    res_df = pd.DataFrame(exp_rows)
    print(res_df.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
    res_df.to_csv(os.path.join(ROOT, "output_test", "ema_risk_and_low_wr_experiment.csv"), index=False)


if __name__ == "__main__":
    main()
