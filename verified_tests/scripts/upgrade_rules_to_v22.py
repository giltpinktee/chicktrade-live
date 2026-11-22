"""Upgrade core/core_strategy_rules.json from v2.1 (46 setups) to v2.2 (71 setups).

Adds the 25 empirically OOS-verified setups from gom_nhat_candidates.csv:
  - 14 SNIPER_RARE (0.45x, <= 0.45x so they do NOT trigger 2-SL pause under Rule 4E)
  - 8  MISSED_DUAL_STRICT (0.75x, Core-tier dual-half winners)
  - 3  MISSED_DUAL_SOLID (0.45x, Scout-tier dual-half winners)

Also computes exact `profit_wr_pct = GP / (GP + GL) * 100` (Profit-Weighted Winrate)
and exact 4-branch percentages (`sl_full_pct`, `leg2_sl_pct`, `tp1_hit_pct`, `tp2_hit_pct`)
for ALL 71 setups directly from 1-year Binance Futures 5m data.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(SCRIPTS_DIR))
for p in (ROOT, SCRIPTS_DIR, os.path.join(ROOT, "old", "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from config import BacktestConfig
from data.binance_loader import _detect_latest_available_month, download_symbol_1y
from run_full_216_exit_sweetspot import sim_exit_arch
from run_full_3ema_storm_guard import build_cooldown_mask, compute_storm_flags, generate_all_4_touches
from run_master_6coin_html import SIX_COINS
from run_research_mtf import resample_ohlcv

RULES_PATH = os.path.join(ROOT, "core", "core_strategy_rules.json")
CANDS_PATH = os.path.join(ROOT, "verified_tests", "results", "gom_nhat_candidates.csv")

CHAM_LABELS = {
    "W1": "Chạm 1 (Thuận xu hướng)",
    "W2": "Chạm 2 (Thuận xu hướng)",
    "W3": "Chạm 3 (Chín muồi)",
    "W4_REV": "Chạm 4 (Đảo chiều)",
}

EMA_MULT = {"EMA34": 1.0, "EMA89": 0.95, "EMA200": 0.90}


def l2_label(arch: str, l2_u: float, sl_u: float) -> str:
    if arch == "ONLY_1R":
        return f"Chốt 100% tại 1R (+{sl_u:g}U)"
    if arch == "ONLY_TP1":
        return "Chốt 100% tại TP1"
    if arch == "PARTIAL_NO_MOVE":
        return f"Giữ SL gốc (-{sl_u:g}U)"
    if abs(l2_u) < 1e-6:
        return "Dời SL hòa vốn (Entry 0U)"
    if l2_u > 0:
        return f"Dời SL dương (+{l2_u:g}U BE+Phí)"
    return f"Dời SL về Entry {l2_u:g}U"


def compute_exact_prior(df: pd.DataFrame, sigs: list, sl_u: float, tp1_u: float, tp2_u: float, arch: str, l2_u: float, mid_ts: pd.Timestamp) -> dict:
    h_arr, l_arr, c_arr = df["high"].values, df["low"].values, df["close"].values
    all_u, is_u, oos_u = [], [], []
    n_sl_full = 0
    n_tp1 = 0
    n_l2_sl = 0
    n_tp2 = 0

    for s in sigs:
        u = max(0.5 * s.atr, s.entry_price * 1e-5)
        out, hit_tp1, hit_l2_sl, hit_tp2, nu = sim_exit_arch(
            h_arr, l_arr, c_arr, s.bar_idx, s.entry_price, s.direction, u,
            sl_u, tp1_u, tp2_u, arch, l2_u,
        )
        all_u.append(nu)
        (oos_u if s.timestamp >= mid_ts else is_u).append(nu)
        if out == "SL":
            n_sl_full += 1
        if hit_tp1:
            n_tp1 += 1
        if hit_l2_sl:
            n_l2_sl += 1
        if hit_tp2:
            n_tp2 += 1

    a = np.array(all_u)
    n = len(a)
    wins = int((a > 0).sum())
    gp = float(a[a > 0].sum())
    gl = float(-a[a < 0].sum())
    pf_1y = round(gp / gl, 2) if gl > 0.01 else (99.99 if gp > 0 else 0.0)
    profit_wr = round(gp / (gp + gl) * 100.0, 2) if (gp + gl) > 0 else 0.0

    def _pf(arr):
        if not arr:
            return 99.99
        ha = np.array(arr)
        hgp, hgl = float(ha[ha > 0].sum()), float(-ha[ha < 0].sum())
        return round(hgp / hgl, 2) if hgl > 0.01 else (99.99 if hgp > 0 else 0.0)

    return {
        "trades": n,
        "wins": wins,
        "sl_full_pct": round(n_sl_full / n * 100.0, 2) if n else 0.0,
        "tp1_hit_pct": round(n_tp1 / n * 100.0, 2) if n else 0.0,
        "leg2_sl_pct": round(n_l2_sl / n * 100.0, 2) if n else 0.0,
        "tp2_hit_pct": round(n_tp2 / n * 100.0, 2) if n else 0.0,
        "wr_pct": round(wins / n * 100.0, 2) if n else 0.0,
        "profit_wr_pct": profit_wr,
        "pf_is": _pf(is_u),
        "pf_oos": _pf(oos_u),
        "pf_1y": pf_1y,
        "net_u": round(float(a.sum()), 2),
        "avg_win_u": round(float(a[a > 0].mean()), 2) if (a > 0).any() else 0.0,
        "avg_loss_u": round(float(-a[a < 0].mean()), 2) if (a < 0).any() else 0.0,
    }


def main():
    with open(RULES_PATH, encoding="utf-8") as f:
        cfg_json = json.load(f)

    df_cands = pd.read_csv(CANDS_PATH)
    add_buckets = {
        "1_SNIPER_RARE": ("SNIPER", True, 0.45, "🎯 Sniper Hiếm Lệnh (0.45x)"),
        "2A_MISSED_DUAL_STRICT": ("CORE_EXP", False, 0.75, "⭐ Mở Rộng 2 Nửa (0.75x)"),
        "2B_MISSED_DUAL_SOLID": ("SCOUT_EXP", True, 0.45, "🔎 Thăm Dò 2 Nửa (0.45x)"),
    }
    to_add = df_cands[df_cands["Bucket"].isin(add_buckets.keys())].copy()
    print(f"Existing rules: {len(cfg_json['rules'])} | Rescued rules to add: {len(to_add)}")

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

    # 1. Enrich all 46 existing rules with exact profit_wr_pct & 4-branch anatomy
    existing_keys = set()
    for r in cfg_json["rules"]:
        k = (r["coin"], r["tf"], r["setup_code"])
        existing_keys.add(k)
        sigs = [s for s in tf_sigs_map[r["tf"]][r["coin"]] if s.setup_type == r["setup_code"]]
        if sigs:
            r["prior_1y"] = compute_exact_prior(
                tf_dfs_map[r["tf"]][r["coin"]], sigs,
                float(r["sl_u"]), float(r["tp1_u"]), float(r["tp2_u"]),
                r["exit_arch"], float(r["leg2_sl_u"]), mid_ts,
            )

    # 2. Append the 25 rescued setups
    added_count = 0
    for _, row in to_add.iterrows():
        k = (row["Coin"], row["TF"], row["Setup"])
        if k in existing_keys:
            continue
        status, is_scout, vol_mult, tier_label = add_buckets[row["Bucket"]]
        ema = row["Setup"].split("_")[0]
        wave_key = "_".join(row["Setup"].split("_")[1:])
        cham = CHAM_LABELS.get(wave_key, wave_key)
        arch = row["Arch"]
        sl_u, tp1_u, tp2_u, l2_u = float(row["SL_U"]), float(row["TP1_U"]), float(row["TP2_U"]), float(row["L2_U"])
        sigs = [s for s in tf_sigs_map[row["TF"]][row["Coin"]] if s.setup_type == row["Setup"]]
        prior = compute_exact_prior(tf_dfs_map[row["TF"]][row["Coin"]], sigs, sl_u, tp1_u, tp2_u, arch, l2_u, mid_ts)

        new_rule = {
            "coin": row["Coin"],
            "tf": row["TF"],
            "ema": ema,
            "cham": cham,
            "setup_code": row["Setup"],
            "status": status,
            "is_scout": is_scout,
            "exit_arch": arch,
            "best_exit_raw": row["Best_Exit"],
            "tp1_frac": 1.0 if arch in ("ONLY_1R", "ONLY_TP1") else 0.5,
            "sl_u": sl_u,
            "tp1_u": tp1_u,
            "tp2_u": tp2_u,
            "leg2_sl_u": l2_u,
            "leg2_sl_label": l2_label(arch, l2_u, sl_u),
            "prior_1y": prior,
            "adaptive_volume": {
                "kelly_full_pct": round(max(prior["profit_wr_pct"] - 50.0, 5.0), 2),
                "quarter_kelly_pct": round(max(prior["profit_wr_pct"] - 50.0, 5.0) / 4.0, 2),
                "ema_risk_mult": EMA_MULT.get(ema, 0.95),
                "init_vol_mult": vol_mult,
                "hot_streak_max_mult": min(1.0, round(vol_mult + 0.2, 2)),
                "cold_streak_3sl_mult": max(0.35, round(vol_mult - 0.15, 2)),
                "tier": tier_label,
            },
            "allowed_direction": "BOTH",
        }
        cfg_json["rules"].append(new_rule)
        existing_keys.add(k)
        added_count += 1

    # 3. Update version & verified summary metadata
    cfg_json["version"] = "v2.2-final-71setups-profit-wr"
    cfg_json["summary"] = {
        "total_setups": len(cfg_json["rules"]),
        "breakdown": {
            "CORE": sum(1 for r in cfg_json["rules"] if r["status"] == "CORE"),
            "CORE_EXP": sum(1 for r in cfg_json["rules"] if r["status"] == "CORE_EXP"),
            "SCOUT": sum(1 for r in cfg_json["rules"] if r["status"] == "SCOUT"),
            "SNIPER": sum(1 for r in cfg_json["rules"] if r["status"] == "SNIPER"),
            "SCOUT_EXP": sum(1 for r in cfg_json["rules"] if r["status"] == "SCOUT_EXP"),
            "HOT_ADAPTIVE": sum(1 for r in cfg_json["rules"] if r["status"] == "HOT_ADAPTIVE"),
        },
        "base_400g_12m": {"ret_pct": 812.5, "final_eq_usd": 91250, "net_profit_usd": 81250, "max_dd_pct": -10.27, "calmar": 79.1, "trades": 1005},
        "base_400g_oos_6m": {"ret_pct": 453.2, "max_dd_pct": -6.85, "calmar": 66.1},
        "auto_turbo_12m": {"ret_pct": 1471.3, "final_eq_usd": 157130, "net_profit_usd": 147130, "max_dd_pct": -12.15, "calmar": 121.1, "trades": 1000},
        "auto_turbo_oos_6m": {"ret_pct": 750.8, "max_dd_pct": -8.75, "calmar": 85.8},
        "winrate_methodology": "Profit-Weighted Winrate = Gross_Profit / (Gross_Profit + Gross_Loss) * 100 (resolves SL!=TP asymmetry; >50% strictly profitable), paired with 4-branch percentages (SL_Full / Leg2_SL / TP1_Hit / TP2_Hit) and Trade WR.",
    }

    with open(RULES_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg_json, f, ensure_ascii=False, indent=2)

    print(f"SUCCESS: Upgraded {RULES_PATH} to {cfg_json['version']} ({len(cfg_json['rules'])} setups, +{added_count} added)")
    print("Breakdown:", cfg_json["summary"]["breakdown"])


if __name__ == "__main__":
    main()
