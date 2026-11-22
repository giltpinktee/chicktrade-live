"""Build official CHICKTRADE core package:
1. Generate core/core_strategy_rules.json with Bayesian priors & mathematical Kelly volume tiers.
2. Generate core/reports/adaptive_volume_matrix.csv & inject/build core HTML summary.
3. Organize legacy research outputs cleanly into output_test/.
"""
import json
import os
import shutil
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.adaptive_sizing import BayesianKellySizer, SetupPrior


def main():
    outputs_dir = os.path.join(ROOT, "outputs")
    output_test_dir = os.path.join(ROOT, "output_test")
    core_reports_dir = os.path.join(ROOT, "core", "reports")
    os.makedirs(output_test_dir, exist_ok=True)
    os.makedirs(core_reports_dir, exist_ok=True)

    # 1. Load the 25 verified core setups + rare high-conviction scout setups (7-14 trades, WR>=75%, PF_IS>=1.5 & PF_OOS>=1.5)
    src_csv = os.path.join(outputs_dir, "master_selected_core_setups.csv")
    if not os.path.exists(src_csv):
        src_csv = os.path.join(output_test_dir, "master_selected_core_setups.csv")
    df_core = pd.read_csv(src_csv)
    df_core["Is_Scout"] = False

    audit_csv = os.path.join(output_test_dir, "master_all_288_setups_audit.csv")
    if os.path.exists(audit_csv):
        df_all = pd.read_csv(audit_csv)
        scouts = df_all[
            (~df_all["Selected"])
            & (df_all["TF"].isin(["15m", "30m", "1h"]))
            & (df_all["Trades_1Y"] >= 7)
            & (df_all["Trades_1Y"] < 15)
            & (df_all["Trades_IS"] >= 3)
            & (df_all["Trades_OOS"] >= 3)
            & (df_all["WR_1Y_%"] >= 75.0)
            & (df_all["PF_1Y"] >= 2.0)
            & (df_all["PF_IS6M"] >= 1.5)
            & (df_all["PF_OOS6M"] >= 1.5)
        ].copy()
        scouts["Is_Scout"] = True
        df = pd.concat([df_core, scouts[df_core.columns]], ignore_index=True)
    else:
        df = df_core

    sizer = BayesianKellySizer()
    rules_json = []
    matrix_rows = []

    for _, row in df.iterrows():
        n_1y = int(row["Trades_1Y"])
        wr_1y = float(row["WR_1Y_%"]) / 100.0
        wins_1y = int(round(n_1y * wr_1y))
        pf_1y = float(row["PF_1Y"])
        sl_u = float(str(row["SL_U"]).replace("U", ""))
        sl_full_pct = float(row["SL_Full_%"])

        # Derive exact implied AvgWin_U & AvgLoss_U from PF, WR, and SL_U
        avg_loss_u = sl_u + 0.25  # SL + fee/slippage
        losses_1y = max(n_1y - wins_1y, 1)
        avg_win_u = (pf_1y * losses_1y * avg_loss_u) / max(wins_1y, 1)

        prior = SetupPrior(
            coin=row["Coin"],
            tf=row["TF"],
            setup_code=row["Setup_Code"],
            prior_trades=n_1y,
            prior_wins=wins_1y,
            avg_win_u=avg_win_u,
            avg_loss_u=avg_loss_u,
            sl_full_pct=sl_full_pct,
        )
        dec = sizer.evaluate(prior)

        # Also simulate what happens if live adds +5 wins vs +3 SLs (to show adaptive range)
        dec_hot = sizer.evaluate(prior, [avg_win_u] * 5)
        dec_cold = sizer.evaluate(prior, [-avg_loss_u] * 3)

        is_scout = bool(row.get("Is_Scout", False))
        # Cap initial volume of rare scout setups (7-14 trades) to 0.45x until live Bayesian confirms
        init_vol = min(dec.vol_mult, 0.45) if is_scout else dec.vol_mult
        tier_str = "Scout (Rare 0.45x)" if is_scout else dec.tier_label

        rule_obj = {
            "coin": row["Coin"],
            "tf": row["TF"],
            "ema": row["EMA"],
            "cham": row["Cham"],
            "setup_code": row["Setup_Code"],
            "is_scout": is_scout,
            "sl_u": sl_u,
            "tp1_u": float(str(row["TP1_U"]).split("U")[0]),
            "leg2_sl_u": -2.0,
            "tp2_u": float(str(row["TP2_U"]).split("U")[0]),
            "prior_1y": {
                "trades": n_1y,
                "wins": wins_1y,
                "wr_pct": round(wr_1y * 100.0, 2),
                "sl_full_pct": round(sl_full_pct, 2),
                "tp2_hit_pct": round(float(row["TP2_Hit_%"]), 2),
                "pf_1y": round(pf_1y, 2),
                "avg_win_u": round(avg_win_u, 2),
                "avg_loss_u": round(avg_loss_u, 2),
            },
            "adaptive_volume": {
                "kelly_full_pct": round(dec.kelly_full_pct, 2),
                "quarter_kelly_pct": round(dec.quarter_kelly_pct, 2),
                "init_vol_mult": init_vol,
                "hot_streak_max_mult": dec_hot.vol_mult,
                "cold_streak_3sl_mult": dec_cold.vol_mult,
                "tier": tier_str,
            },
        }
        rules_json.append(rule_obj)
        matrix_rows.append(
            {
                "Coin": row["Coin"],
                "TF": row["TF"],
                "Setup": row["Setup_Code"],
                "Trades_1Y": n_1y,
                "WR_1Y_%": round(wr_1y * 100.0, 2),
                "SL_Full_%": round(sl_full_pct, 2),
                "TP2_Hit_%": round(float(row["TP2_Hit_%"]), 2),
                "PF_1Y": round(pf_1y, 2),
                "Kelly_Full_%": round(dec.kelly_full_pct, 1),
                "Vol_Init": f"{init_vol:.2f}x",
                "Vol_Hot(+5W)": f"{dec_hot.vol_mult:.2f}x",
                "Vol_Cold(-3SL)": f"{dec_cold.vol_mult:.2f}x",
                "Tier": tier_str,
            }
        )

    # Save official JSON & CSV in core/
    with open(os.path.join(ROOT, "core", "core_strategy_rules.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "version": "1.1-swarm-consensus",
                "default_buffer_units": 300,
                "supported_buffers": [250, 300, 400, 500],
                "buffer_max_concurrent": {"250": 2, "300": 3, "400": 5, "500": 7},
                "max_concurrent_trades": 3,
                "max_open_mult_cap": 2.40,
                "max_trades_per_day": 10,
                "compounding_steps": 20,
                "step_gain_pct": 0.08,
                "storm_guard": {
                    "cooldown_hours": 4,
                    "atr_spike_mult": 1.8,
                    "btc_er_min": 0.15,
                    "btc_er_atr_mult": 1.2,
                    "elevated_atr_mult": 1.35,
                    "dd_brake_pct": -6.0,
                    "consec_sl_brake": 2,
                    "daily_sl_stop": 3,
                    "daily_loss_stop_pct": -5.0,
                },
                "rules": rules_json,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    mdf = pd.DataFrame(matrix_rows).sort_values(["Kelly_Full_%", "WR_1Y_%"], ascending=False)
    mdf.to_csv(os.path.join(core_reports_dir, "adaptive_volume_matrix.csv"), index=False)

    # 2. Copy official Master HTML report into core/reports/
    master_html_src = os.path.join(outputs_dir, "chicktrade_6coin_master_report.html")
    master_html_dst = os.path.join(core_reports_dir, "chicktrade_6coin_master_report.html")
    if os.path.exists(master_html_src):
        shutil.copy2(master_html_src, master_html_dst)

    # 3. Move all test outputs from outputs/ -> output_test/
    if os.path.exists(outputs_dir):
        for item in os.listdir(outputs_dir):
            s = os.path.join(outputs_dir, item)
            d = os.path.join(output_test_dir, item)
            if os.path.exists(d):
                if os.path.isdir(d):
                    shutil.rmtree(d)
                else:
                    os.remove(d)
            shutil.move(s, d)
        os.rmdir(outputs_dir)

    print("=== ADAPTIVE BAYESIAN-KELLY VOLUME MATRIX (25 CORE SETUPS) ===")
    print(mdf.to_string(index=False))
    print("\nOrganized workspace cleanly: core/ (production rules & master report) + output_test/ (all backtest artifacts).")


if __name__ == "__main__":
    main()
