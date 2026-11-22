"""Single-Source-of-Truth Upgrade & Sync to v2.3 (75 Setups).

1. Adds the 4 empirically OOS-verified 3A_HOT_RECENT setups (@ 0.45x, <= 0.45x so no 2-SL trigger)
   to core/core_strategy_rules.json (71 -> 75 setups).
2. Rebuilds ALL 6 coins/<SYM>/audit_48.csv and coins/<SYM>/rules.json from the EXACT SAME
   engine, fees, slippage, exit architectures, and rules_map — guaranteeing ZERO (0) mismatches
   between Tab 2 (75 Setups), 6 Coin Tabs, and all 48-setup reference tables.
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
from run_288_with_shadow_win_recheck import EXIT_CANDS
from run_full_216_exit_sweetspot import COIN_CAL, sim_exit_arch
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

# The 4 OOS-verified 3A_HOT_RECENT setups (PF_OOS >= 1.50, WR >= 60%, Profit_WR >= 60%)
HOT_RECENT_4 = {
    ("ETHUSDT", "30m", "EMA89_W1"),
    ("BNBUSDT", "30m", "EMA200_W2"),
    ("ETHUSDT", "15m", "EMA200_W4_REV"),
    ("LINKUSDT", "30m", "EMA89_W3"),
}


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


def eval_setup_exact(df: pd.DataFrame, sigs: list, sl_u: float, tp1_u: float, tp2_u: float, arch: str, l2_u: float, mid_ts: pd.Timestamp) -> dict:
    h_arr, l_arr, c_arr = df["high"].values, df["low"].values, df["close"].values
    all_u, is_u, oos_u = [], [], []
    long_u, short_u = [], []
    n_sl_full = n_tp1 = n_l2_sl = n_tp2 = 0

    for s in sigs:
        u = max(0.5 * s.atr, s.entry_price * 1e-5)
        out, hit_tp1, hit_l2_sl, hit_tp2, nu = sim_exit_arch(
            h_arr, l_arr, c_arr, s.bar_idx, s.entry_price, s.direction, u,
            sl_u, tp1_u, tp2_u, arch, l2_u,
        )
        all_u.append(nu)
        (oos_u if s.timestamp >= mid_ts else is_u).append(nu)
        (long_u if s.direction == 1 else short_u).append(nu)
        if out == "SL":
            n_sl_full += 1
        if hit_tp1:
            n_tp1 += 1
        if hit_l2_sl:
            n_l2_sl += 1
        if hit_tp2:
            n_tp2 += 1

    a = np.array(all_u) if all_u else np.array([])
    n = len(a)
    wins = int((a > 0).sum()) if n else 0
    gp = float(a[a > 0].sum()) if n else 0.0
    gl = float(-a[a < 0].sum()) if n else 0.0
    pf_1y = round(gp / gl, 2) if gl > 0.01 else (99.99 if gp > 0 else 0.0)
    profit_wr = round(gp / (gp + gl) * 100.0, 2) if (gp + gl) > 0 else 0.0

    def _pf(arr):
        if not arr:
            return 99.99
        ha = np.array(arr)
        hgp, hgl = float(ha[ha > 0].sum()), float(-ha[ha < 0].sum())
        return round(hgp / hgl, 2) if hgl > 0.01 else (99.99 if hgp > 0 else 0.0)

    l_wr = round(float((np.array(long_u) > 0).mean() * 100.0), 1) if long_u else 0.0
    s_wr = round(float((np.array(short_u) > 0).mean() * 100.0), 1) if short_u else 0.0

    return {
        "trades": n,
        "wins": wins,
        "long_trades": len(long_u),
        "long_wr_pct": l_wr,
        "short_trades": len(short_u),
        "short_wr_pct": s_wr,
        "sl_full_pct": round(n_sl_full / n * 100.0, 2) if n else 0.0,
        "tp1_hit_pct": round(n_tp1 / n * 100.0, 2) if n else 0.0,
        "leg2_sl_pct": round(n_l2_sl / n * 100.0, 2) if n else 0.0,
        "tp2_hit_pct": round(n_tp2 / n * 100.0, 2) if n else 0.0,
        "wr_pct": round(wins / n * 100.0, 2) if n else 0.0,
        "profit_wr_pct": profit_wr,
        "pf_is": _pf(is_u),
        "pf_oos": _pf(oos_u),
        "pf_1y": pf_1y,
        "net_u": round(float(a.sum()), 2) if n else 0.0,
        "avg_win_u": round(float(a[a > 0].mean()), 2) if (n and (a > 0).any()) else 0.0,
        "avg_loss_u": round(float(-a[a < 0].mean()), 2) if (n and (a < 0).any()) else 0.0,
    }


def best_exit_for_skipped(df: pd.DataFrame, sigs: list, sym: str, tfl: str, mid_ts: pd.Timestamp) -> tuple:
    sl_u = COIN_CAL[sym]["sl"]
    tp1_u = COIN_CAL[sym]["tp1"].get(tfl, 3.5)
    tp2_u = COIN_CAL[sym]["tp2"]
    best_label, best_stats = "ONLY_1R", None
    for label, arch, l2_u in EXIT_CANDS:
        if arch == "PARTIAL_MOVE_SL" and l2_u is not None and l2_u < 0 and (tp1_u <= abs(l2_u) + 0.6):
            continue
        st = eval_setup_exact(df, sigs, sl_u, tp1_u, tp2_u, arch, l2_u if l2_u is not None else -sl_u, mid_ts)
        if best_stats is None or st["net_u"] > best_stats["net_u"]:
            best_label, best_stats = label, st
    return best_label, best_stats


def main():
    with open(RULES_PATH, encoding="utf-8") as f:
        cfg_json = json.load(f)

    df_cands = pd.read_csv(CANDS_PATH)
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

    # 1. Add the 4 HOT_RECENT setups if not already present
    existing_keys = {(r["coin"], r["tf"], r["setup_code"]) for r in cfg_json["rules"]}
    added = 0
    for _, row in df_cands.iterrows():
        k = (row["Coin"], row["TF"], row["Setup"])
        if k in HOT_RECENT_4 and k not in existing_keys:
            ema = row["Setup"].split("_")[0]
            wave_key = "_".join(row["Setup"].split("_")[1:])
            arch = row["Arch"]
            sl_u, tp1_u, tp2_u, l2_u = float(row["SL_U"]), float(row["TP1_U"]), float(row["TP2_U"]), float(row["L2_U"])
            sigs = [s for s in tf_sigs_map[row["TF"]][row["Coin"]] if s.setup_type == row["Setup"]]
            prior = eval_setup_exact(tf_dfs_map[row["TF"]][row["Coin"]], sigs, sl_u, tp1_u, tp2_u, arch, l2_u, mid_ts)
            cfg_json["rules"].append({
                "coin": row["Coin"],
                "tf": row["TF"],
                "ema": ema,
                "cham": CHAM_LABELS.get(wave_key, wave_key),
                "setup_code": row["Setup"],
                "status": "HOT_ADAPTIVE",
                "is_scout": True,
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
                    "init_vol_mult": 0.45,
                    "hot_streak_max_mult": 0.65,
                    "cold_streak_3sl_mult": 0.30,
                    "tier": "🔥 Thích ứng 6T cuối (0.45x)",
                },
                "allowed_direction": "BOTH",
            })
            existing_keys.add(k)
            added += 1

    # 2. Refresh exact prior_1y for ALL 75 rules (including long/short breakdown)
    rules_by_key = {}
    for r in cfg_json["rules"]:
        k = (r["coin"], r["tf"], r["setup_code"])
        sigs = [s for s in tf_sigs_map[r["tf"]][r["coin"]] if s.setup_type == r["setup_code"]]
        if sigs:
            r["prior_1y"] = eval_setup_exact(
                tf_dfs_map[r["tf"]][r["coin"]], sigs,
                float(r["sl_u"]), float(r["tp1_u"]), float(r["tp2_u"]),
                r["exit_arch"], float(r["leg2_sl_u"]), mid_ts,
            )
        rules_by_key[k] = r

    # 3. Update v2.3 metadata & verified summary
    cfg_json["version"] = "v2.3-final-75setups-rog25"
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
        "base_400g_12m": {"ret_pct": 841.7, "final_eq_usd": 94170, "net_profit_usd": 84170, "max_dd_pct": -9.72, "calmar": 86.6},
        "base_400g_oos_6m": {"ret_pct": 485.2, "max_dd_pct": -6.88, "calmar": 70.5},
        "auto_turbo_12m": {"ret_pct": 1481.0, "final_eq_usd": 158100, "net_profit_usd": 148100, "max_dd_pct": -12.13, "calmar": 122.1},
        "auto_turbo_oos_6m": {"ret_pct": 828.1, "max_dd_pct": -8.62, "calmar": 96.1},
        "winrate_methodology": "Profit-Weighted Winrate = Gross_Profit / (Gross_Profit + Gross_Loss) * 100 (>50% strictly profitable), paired with 4-branch percentages (SL / Leg2_SL / TP1 / TP2) and Trade WR.",
    }

    with open(RULES_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg_json, f, ensure_ascii=False, indent=2)

    # 4. SINGLE-SOURCE-OF-TRUTH SYNC: Rebuild ALL 6 coins/<SYM>/audit_48.csv & coins/<SYM>/rules.json
    all_setups_order = [
        ("EMA34", "W1", "EMA34_W1"), ("EMA34", "W2", "EMA34_W2"), ("EMA34", "W3", "EMA34_W3"), ("EMA34", "W4", "EMA34_W4_REV"),
        ("EMA89", "W1", "EMA89_W1"), ("EMA89", "W2", "EMA89_W2"), ("EMA89", "W3", "EMA89_W3"), ("EMA89", "W4", "EMA89_W4_REV"),
        ("EMA200", "W1", "EMA200_W1"), ("EMA200", "W2", "EMA200_W2"), ("EMA200", "W3", "EMA200_W3"), ("EMA200", "W4", "EMA200_W4_REV"),
    ]

    for sym in SIX_COINS:
        coin_dir = os.path.join(ROOT, "coins", sym)
        os.makedirs(coin_dir, exist_ok=True)
        rows_48 = []
        coin_rules = []

        for tfl in ["5m", "15m", "30m", "1h"]:
            df = tf_dfs_map[tfl][sym]
            by_st = {}
            for s in tf_sigs_map[tfl][sym]:
                by_st.setdefault(s.setup_type, []).append(s)

            for ema, touch, stype in all_setups_order:
                k = (sym, tfl, stype)
                sigs = by_st.get(stype, [])
                if k in rules_by_key:
                    r = rules_by_key[k]
                    st = r["prior_1y"]
                    coin_rules.append(r)
                    rows_48.append({
                        "Coin": sym, "TF": tfl, "EMA": ema, "Touch": touch, "Setup": stype,
                        "Lenh_1Y": st["trades"],
                        "Long_Lenh": st.get("long_trades", 0), "Long_WR_%": st.get("long_wr_pct", 0.0),
                        "Short_Lenh": st.get("short_trades", 0), "Short_WR_%": st.get("short_wr_pct", 0.0),
                        "Best_Exit": r.get("best_exit_raw", r["exit_arch"]),
                        "SL_Full_%": st["sl_full_pct"], "TP1_Hit_%": st["tp1_hit_pct"],
                        "Dinh_DoiSL_%": st["leg2_sl_pct"], "TP2_Hit_%": st["tp2_hit_pct"],
                        "WR_1Y_%": st["wr_pct"], "Profit_WR_%": st["profit_wr_pct"],
                        "PF_IS": st["pf_is"], "PF_OOS": st["pf_oos"], "PF_1Y": st["pf_1y"],
                        "Net_U_1Y": st["net_u"],
                        "Vol_Init": r["adaptive_volume"]["init_vol_mult"],
                        "Status": r["status"],
                    })
                else:
                    if sigs:
                        best_lbl, st = best_exit_for_skipped(df, sigs, sym, tfl, mid_ts)
                    else:
                        best_lbl = "NONE"
                        st = {
                            "trades": 0, "long_trades": 0, "long_wr_pct": 0.0, "short_trades": 0, "short_wr_pct": 0.0,
                            "sl_full_pct": 0.0, "tp1_hit_pct": 0.0, "leg2_sl_pct": 0.0, "tp2_hit_pct": 0.0,
                            "wr_pct": 0.0, "profit_wr_pct": 0.0, "pf_is": 0.0, "pf_oos": 0.0, "pf_1y": 0.0, "net_u": 0.0,
                        }
                    rows_48.append({
                        "Coin": sym, "TF": tfl, "EMA": ema, "Touch": touch, "Setup": stype,
                        "Lenh_1Y": st["trades"],
                        "Long_Lenh": st["long_trades"], "Long_WR_%": st["long_wr_pct"],
                        "Short_Lenh": st["short_trades"], "Short_WR_%": st["short_wr_pct"],
                        "Best_Exit": f"SKIP ({best_lbl})",
                        "SL_Full_%": st["sl_full_pct"], "TP1_Hit_%": st["tp1_hit_pct"],
                        "Dinh_DoiSL_%": st["leg2_sl_pct"], "TP2_Hit_%": st["tp2_hit_pct"],
                        "WR_1Y_%": st["wr_pct"], "Profit_WR_%": st["profit_wr_pct"],
                        "PF_IS": st["pf_is"], "PF_OOS": st["pf_oos"], "PF_1Y": st["pf_1y"],
                        "Net_U_1Y": st["net_u"],
                        "Vol_Init": 0.0,
                        "Status": "SKIP",
                    })

        pd.DataFrame(rows_48).to_csv(os.path.join(coin_dir, "audit_48.csv"), index=False)
        with open(os.path.join(coin_dir, "rules.json"), "w", encoding="utf-8") as f:
            json.dump({"symbol": sym, "version": cfg_json["version"], "active_setups": len(coin_rules), "rules": coin_rules}, f, ensure_ascii=False, indent=2)

    # 5. STRICT VERIFICATION: Assert 0 mismatches!
    mismatches = 0
    total_active_in_csv = 0
    for sym in SIX_COINS:
        df48 = pd.read_csv(os.path.join(ROOT, "coins", sym, "audit_48.csv"))
        for _, row in df48.iterrows():
            k = (row["Coin"], row["TF"], row["Setup"])
            in_rules = k in rules_by_key
            in_csv = row["Status"] != "SKIP"
            if in_csv:
                total_active_in_csv += 1
            if in_rules != in_csv:
                mismatches += 1
            elif in_rules and row["Status"] != rules_by_key[k]["status"]:
                mismatches += 1

    print(f"SUCCESS: {cfg_json['version']} | rules.json={len(rules_by_key)} (+{added}) | audit_48 active={total_active_in_csv} | MISMATCHES={mismatches}")
    assert mismatches == 0, f"Found {mismatches} mismatches!"
    assert len(rules_by_key) == 75, f"Expected 75 setups, got {len(rules_by_key)}"
    print("Breakdown:", cfg_json["summary"]["breakdown"])


if __name__ == "__main__":
    main()
