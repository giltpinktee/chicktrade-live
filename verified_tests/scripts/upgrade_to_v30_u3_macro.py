"""Single-Source-of-Truth Upgrade & Sync to v3.0 — U3_HTF_MACRO_ATR (58 Setups).

1. Discovers the 58 setups via Test B U3_HTF_MACRO_ATR (Train 2Y 11/2023->10/2025,
   split at 1Y boundary, min_trades=10 -> 38 @ 0.85x CORE + 20 @ 0.55x CORE_EXP).
2. Upgrades e:\CHICKTRADE\core\core_strategy_rules.json to `v3.0-u3-htf-macro-4h`
   with explicit `unit_config`, full 36M + 2Y-IS / Y3-OOS `prior_1y` stats, and
   portfolio summary verified for Bao 400G & Auto-Turbo 250G.
3. Rebuilds ALL 6 `coins/<SYM>/audit_48.csv` and `coins/<SYM>/rules.json` from the
   EXACT SAME U3_HTF_MACRO_ATR source and asserts `MISMATCHES == 0`.

Does NOT modify bot/.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
VERIFIED_DIR = os.path.dirname(SCRIPTS_DIR)
ROOT = os.path.dirname(VERIFIED_DIR)
for p in (ROOT, SCRIPTS_DIR, os.path.join(ROOT, "old", "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from config import BacktestConfig
from run_288_with_shadow_win_recheck import EXIT_CANDS
from run_full_216_exit_sweetspot import COIN_CAL, sim_exit_arch
from run_full_3ema_storm_guard import build_cooldown_mask, compute_storm_flags, generate_all_4_touches
from run_invariant_1u_benchmark import (
    GAMMA_4H,
    MIN_U_PCT,
    discover_dual_split_rules_arrays,
    precompute_invariant_u_arrays,
    sim_portfolio_u_arrays,
)
from run_master_6coin_html import SIX_COINS
from run_pct_vs_atr_3year_study import filter_sigs, load_3year_cached_m5
from run_research_mtf import resample_ohlcv

RULES_PATH = os.path.join(ROOT, "core", "core_strategy_rules.json")

CHAM_LABELS = {
    "W1": "Chạm 1 (Thuận xu hướng)",
    "W2": "Chạm 2 (Thuận xu hướng)",
    "W3": "Chạm 3 (Chín muồi)",
    "W4_REV": "Chạm 4 (Đảo chiều)",
}
EMA_MULT = {"EMA34": 1.0, "EMA89": 0.95, "EMA200": 0.90}

ALL_SETUPS_ORDER = [
    ("EMA34", "W1", "EMA34_W1"), ("EMA34", "W2", "EMA34_W2"), ("EMA34", "W3", "EMA34_W3"), ("EMA34", "W4", "EMA34_W4_REV"),
    ("EMA89", "W1", "EMA89_W1"), ("EMA89", "W2", "EMA89_W2"), ("EMA89", "W3", "EMA89_W3"), ("EMA89", "W4", "EMA89_W4_REV"),
    ("EMA200", "W1", "EMA200_W1"), ("EMA200", "W2", "EMA200_W2"), ("EMA200", "W3", "EMA200_W3"), ("EMA200", "W4", "EMA200_W4_REV"),
]


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


def best_exit_raw_label(arch: str, l2_u: float) -> str:
    for lbl, a, lu in EXIT_CANDS:
        if a == arch and abs(float(lu) - float(l2_u)) < 1e-6:
            return lbl
    return arch


def eval_setup_u3(
    df: pd.DataFrame,
    sigs: list,
    u_arr: np.ndarray,
    sl_u: float,
    tp1_u: float,
    tp2_u: float,
    arch: str,
    l2_u: float,
    ts_oos_start: pd.Timestamp,
) -> dict:
    """Evaluate a setup over 36M signals using U3_HTF_MACRO_ATR (pf_is=2Y Train, pf_oos=Blind Y3)."""
    h_arr, l_arr, c_arr = df["high"].values, df["low"].values, df["close"].values
    all_u, is_u, oos_u = [], [], []
    long_u, short_u = [], []
    n_sl_full = n_tp1 = n_l2_sl = n_tp2 = 0

    for s in sigs:
        u = float(u_arr[s.bar_idx])
        out, hit_tp1, hit_l2_sl, hit_tp2, nu = sim_exit_arch(
            h_arr, l_arr, c_arr, s.bar_idx, s.entry_price, s.direction, u,
            sl_u, tp1_u, tp2_u, arch, l2_u,
        )
        all_u.append(nu)
        (oos_u if s.timestamp >= ts_oos_start else is_u).append(nu)
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
    pf_36m = round(gp / gl, 2) if gl > 0.01 else (99.99 if gp > 0 else 0.0)
    profit_wr = round(gp / (gp + gl) * 100.0, 2) if (gp + gl) > 0 else 0.0

    def _pf(arr):
        if not arr:
            return 0.0
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
        "pf_1y": pf_36m,
        "net_u": round(float(a.sum()), 2) if n else 0.0,
        "avg_win_u": round(float(a[a > 0].mean()), 2) if (n and (a > 0).any()) else 0.0,
        "avg_loss_u": round(float(-a[a < 0].mean()), 2) if (n and (a < 0).any()) else 0.0,
    }


def best_exit_for_skipped_u3(
    df: pd.DataFrame,
    sigs: list,
    u_arr: np.ndarray,
    sym: str,
    tfl: str,
    ts_oos_start: pd.Timestamp,
) -> tuple:
    sl_u = float(COIN_CAL[sym]["sl"])
    tp1_u = float(COIN_CAL[sym]["tp1"].get(tfl, 3.5))
    tp2_u = float(COIN_CAL[sym]["tp2"])
    best_label, best_stats = "ONLY_1R", None
    for label, arch, l2_u in EXIT_CANDS:
        if arch == "PARTIAL_MOVE_SL" and l2_u is not None and l2_u < 0 and (tp1_u <= abs(l2_u) + 0.6):
            continue
        st = eval_setup_u3(df, sigs, u_arr, sl_u, tp1_u, tp2_u, arch, float(l2_u), ts_oos_start)
        if best_stats is None or st["net_u"] > best_stats["net_u"]:
            best_label, best_stats = label, st
    return best_label, best_stats


def _summarize_run(r: dict) -> dict:
    return {
        "ret_pct": round(float(r["ret"]), 2),
        "final_eq_usd": round(float(r["final_eq"]), 2),
        "net_profit_usd": round(float(r["final_eq"]) - 10_000.0, 2),
        "max_dd_pct": round(float(r["mdd"]), 2),
        "calmar": round(float(r["calmar"]), 2),
        "pf": round(float(r["pf"]), 2),
        "wr_pct": round(float(r["wr"]), 2),
        "trades": int(r["trades"]),
    }


def main():
    with open(RULES_PATH, encoding="utf-8") as f:
        cfg_json = json.load(f)

    print("Loading 36-month cached 5m futures data (2023-11-01 -> 2026-10-31)...")
    raw_m5 = load_3year_cached_m5()
    cfg = BacktestConfig()
    dfs_m15 = {s: resample_ohlcv(d, "15min") for s, d in raw_m5.items()}
    dfs_m30 = {s: resample_ohlcv(d, "30min") for s, d in raw_m5.items()}
    dfs_h1 = {s: resample_ohlcv(d, "1h") for s, d in raw_m5.items()}
    dfs_h4 = {s: resample_ohlcv(d, "4h") for s, d in raw_m5.items()}
    tf_dfs_map = {"5m": raw_m5, "15m": dfs_m15, "30m": dfs_m30, "1h": dfs_h1}

    u_arrays, _ = precompute_invariant_u_arrays(tf_dfs_map, dfs_h4)
    u3_arrays = u_arrays["U3_HTF_MACRO_ATR"]

    print("Generating 4-touch signals across 36M...")
    tf_sigs_map = {}
    for tfl, tdf, hdf, cd in [("5m", raw_m5, dfs_h1, 48), ("15m", dfs_m15, dfs_h1, 16), ("30m", dfs_m30, dfs_h4, 8), ("1h", dfs_h1, dfs_h4, 4)]:
        bst = compute_storm_flags(tdf["BTCUSDT"])
        bmask = build_cooldown_mask(bst["raw_storm"], cd)
        bts = tdf["BTCUSDT"]["timestamp"].values
        tf_sigs_map[tfl] = {
            sym: generate_all_4_touches(tdf[sym], hdf[sym], bts, bmask, sym, cfg.strategy, cooldown_bars=cd)[0]
            for sym in SIX_COINS
        }

    ts_2y_start = pd.Timestamp("2023-11-01 00:00:00", tz="UTC")
    ts_y1_end = pd.Timestamp("2024-10-31 23:59:59", tz="UTC")
    ts_2y_end = pd.Timestamp("2025-10-31 23:59:59", tz="UTC")
    ts_y3_start = pd.Timestamp("2025-11-01 00:00:00", tz="UTC")
    ts_3y_end = pd.Timestamp("2026-10-31 23:59:59", tz="UTC")

    sigs_2y = filter_sigs(tf_sigs_map, ts_2y_start, ts_2y_end)
    sigs_y3 = filter_sigs(tf_sigs_map, ts_y3_start, ts_3y_end)
    sigs_36m = filter_sigs(tf_sigs_map, ts_2y_start, ts_3y_end)

    print("Discovering v3.0 58 setups via Test B U3_HTF_MACRO_ATR (Train 2Y 11/2023->10/2025)...")
    wf_rules = discover_dual_split_rules_arrays(
        tf_dfs_map, sigs_2y, "U3_HTF_MACRO_ATR", u3_arrays, ts_y1_end, min_trades=10,
    )
    assert len(wf_rules) == 58, f"Expected 58 Test B U3 setups, got {len(wf_rules)}"

    # Build the 58 rule entries with exact 36M + 2Y-IS / Y3-OOS prior_1y stats
    new_rules = []
    rules_by_key = {}
    for (sym, tfl, stype), rm in sorted(wf_rules.items()):
        ema = stype.split("_")[0]
        wave_key = "_".join(stype.split("_")[1:])
        sigs_setup = [s for s in sigs_36m[tfl][sym] if s.setup_type == stype]
        prior = eval_setup_u3(
            tf_dfs_map[tfl][sym], sigs_setup, u3_arrays[tfl][sym],
            rm["sl_u"], rm["tp1_u"], rm["tp2_u"], rm["arch"], rm["l2_u"], ts_y3_start,
        )
        vol_m = float(rm["vol"])
        is_core = vol_m >= 0.80
        status = "CORE" if is_core else "CORE_EXP"
        tier_lbl = "A+ (Core 0.85x)" if is_core else "B (Mở rộng 0.55x)"
        rule_obj = {
            "coin": sym,
            "tf": tfl,
            "ema": ema,
            "cham": CHAM_LABELS.get(wave_key, wave_key),
            "setup_code": stype,
            "status": status,
            "is_scout": False,
            "exit_arch": rm["arch"],
            "best_exit_raw": best_exit_raw_label(rm["arch"], rm["l2_u"]),
            "tp1_frac": 1.0 if rm["arch"] in ("ONLY_1R", "ONLY_TP1") else 0.5,
            "sl_u": rm["sl_u"],
            "tp1_u": rm["tp1_u"],
            "tp2_u": rm["tp2_u"],
            "leg2_sl_u": rm["l2_u"],
            "leg2_sl_label": l2_label(rm["arch"], rm["l2_u"], rm["sl_u"]),
            "prior_1y": prior,
            "adaptive_volume": {
                "kelly_full_pct": round(max(prior["profit_wr_pct"] - 50.0, 5.0), 2),
                "quarter_kelly_pct": round(max(prior["profit_wr_pct"] - 50.0, 5.0) / 4.0, 2),
                "ema_risk_mult": EMA_MULT.get(ema, 0.95),
                "init_vol_mult": vol_m,
                "hot_streak_max_mult": vol_m,
                "cold_streak_3sl_mult": round(vol_m * 0.5, 2),
                "tier": tier_lbl,
            },
            "allowed_direction": "BOTH",
        }
        new_rules.append(rule_obj)
        rules_by_key[(sym, tfl, stype)] = rule_obj

    # Verify portfolio runs dynamically
    res_b_2y = sim_portfolio_u_arrays(tf_dfs_map, sigs_2y, wf_rules, "U3_HTF_MACRO_ATR", u3_arrays, None, 400.0, None)
    res_b_y3 = sim_portfolio_u_arrays(tf_dfs_map, sigs_y3, wf_rules, "U3_HTF_MACRO_ATR", u3_arrays, None, 400.0, None)
    res_b_36m = sim_portfolio_u_arrays(tf_dfs_map, sigs_36m, wf_rules, "U3_HTF_MACRO_ATR", u3_arrays, None, 400.0, None)
    res_t_2y = sim_portfolio_u_arrays(tf_dfs_map, sigs_2y, wf_rules, "U3_HTF_MACRO_ATR", u3_arrays, None, 400.0, 250.0)
    res_t_y3 = sim_portfolio_u_arrays(tf_dfs_map, sigs_y3, wf_rules, "U3_HTF_MACRO_ATR", u3_arrays, None, 400.0, 250.0)
    res_t_36m = sim_portfolio_u_arrays(tf_dfs_map, sigs_36m, wf_rules, "U3_HTF_MACRO_ATR", u3_arrays, None, 400.0, 250.0)

    # Update core_strategy_rules.json
    cfg_json["version"] = "v3.0-u3-htf-macro-4h"
    cfg_json["unit_config"] = {
        "mode": "U3_HTF_MACRO_ATR",
        "htf": "4h",
        "atr_period": 14,
        "ema_smooth": 20,
        "gamma_tf": dict(GAMMA_4H),
        "min_u_pct": MIN_U_PCT,
    }
    cfg_json["rules"] = new_rules
    cfg_json["summary"] = {
        "total_setups": len(new_rules),
        "breakdown": {
            "CORE": sum(1 for r in new_rules if r["status"] == "CORE"),
            "CORE_EXP": sum(1 for r in new_rules if r["status"] == "CORE_EXP"),
            "SCOUT": 0,
            "SNIPER": 0,
            "SCOUT_EXP": 0,
            "HOT_ADAPTIVE": 0,
        },
        "base_400g_36m": _summarize_run(res_b_36m),
        "base_400g_train_2y": _summarize_run(res_b_2y),
        "base_400g_blind_y3": _summarize_run(res_b_y3),
        "auto_turbo_36m": _summarize_run(res_t_36m),
        "auto_turbo_train_2y": _summarize_run(res_t_2y),
        "auto_turbo_blind_y3": _summarize_run(res_t_y3),
        # Compatibility aliases mapped to verified v3.0 36M & Blind Y3 results
        "base_400g_12m": _summarize_run(res_b_y3),
        "base_400g_oos_6m": _summarize_run(res_b_y3),
        "auto_turbo_12m": _summarize_run(res_t_y3),
        "auto_turbo_oos_6m": _summarize_run(res_t_y3),
        "winrate_methodology": (
            "1U = U3_HTF_MACRO_ATR (Close * max(0.0008, EMA20_4H(0.5*ATR14_4H/Close_4H) * gamma_tf)), "
            "evaluated across 36M (11/2023->10/2026) with pf_is = Train 2Y (11/2023->10/2025) & "
            "pf_oos = Blind Y3 (11/2025->10/2026). Profit-Weighted WR = GP / (GP + GL) * 100."
        ),
    }

    with open(RULES_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg_json, f, ensure_ascii=False, indent=2)

    # Rebuild ALL 6 coins/<SYM>/audit_48.csv & coins/<SYM>/rules.json from the exact same U3 source
    for sym in SIX_COINS:
        coin_dir = os.path.join(ROOT, "coins", sym)
        os.makedirs(coin_dir, exist_ok=True)
        rows_48 = []
        coin_rules = []

        for tfl in ["5m", "15m", "30m", "1h"]:
            df = tf_dfs_map[tfl][sym]
            u_arr = u3_arrays[tfl][sym]
            by_st = {}
            for s in sigs_36m[tfl][sym]:
                by_st.setdefault(s.setup_type, []).append(s)

            for ema, touch, stype in ALL_SETUPS_ORDER:
                k = (sym, tfl, stype)
                sigs = by_st.get(stype, [])
                if k in rules_by_key:
                    r = rules_by_key[k]
                    st = r["prior_1y"]
                    coin_rules.append(r)
                    rows_48.append({
                        "Coin": sym, "TF": tfl, "EMA": ema, "Touch": touch, "Setup": stype,
                        "Lenh_1Y": st["trades"],
                        "Long_Lenh": st["long_trades"], "Long_WR_%": st["long_wr_pct"],
                        "Short_Lenh": st["short_trades"], "Short_WR_%": st["short_wr_pct"],
                        "Best_Exit": r["best_exit_raw"],
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
                        best_lbl, st = best_exit_for_skipped_u3(df, sigs, u_arr, sym, tfl, ts_y3_start)
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
            json.dump(
                {"symbol": sym, "version": cfg_json["version"], "active_setups": len(coin_rules), "rules": coin_rules},
                f, ensure_ascii=False, indent=2,
            )

    # Strict Single-Source-of-Truth verification
    mismatches = 0
    total_active_in_csv = 0
    for sym in SIX_COINS:
        df48 = pd.read_csv(os.path.join(ROOT, "coins", sym, "audit_48.csv"))
        with open(os.path.join(ROOT, "coins", sym, "rules.json"), encoding="utf-8") as f:
            cj = json.load(f)
        if len(cj["rules"]) != sum(1 for r in new_rules if r["coin"] == sym):
            mismatches += 1
        for _, row in df48.iterrows():
            k = (row["Coin"], row["TF"], row["Setup"])
            in_rules = k in rules_by_key
            in_csv = row["Status"] != "SKIP"
            if in_csv:
                total_active_in_csv += 1
            if in_rules != in_csv:
                mismatches += 1
            elif in_rules:
                r = rules_by_key[k]
                if row["Status"] != r["status"] or abs(float(row["Vol_Init"]) - float(r["adaptive_volume"]["init_vol_mult"])) > 1e-6:
                    mismatches += 1

    print(f"SUCCESS: {cfg_json['version']} | rules.json={len(rules_by_key)} | audit_48 active={total_active_in_csv} | MISMATCHES={mismatches}")
    assert mismatches == 0, f"Found {mismatches} mismatches!"
    assert len(rules_by_key) == 58 and total_active_in_csv == 58
    print("Breakdown:", cfg_json["summary"]["breakdown"])
    print("Bao 400G 36M:", cfg_json["summary"]["base_400g_36m"])
    print("Bao 400G Blind Y3:", cfg_json["summary"]["base_400g_blind_y3"])
    print("Auto-Turbo 36M:", cfg_json["summary"]["auto_turbo_36m"])
    print("Auto-Turbo Blind Y3:", cfg_json["summary"]["auto_turbo_blind_y3"])


if __name__ == "__main__":
    main()
