"""Empirical Sweet-Spot Study: Pause Scope (SETUP vs COIN vs PORTFOLIO) x Trigger (2 vs 3 SL) x Resume (12h/24h/48h/Shadow-Win).
Evaluated on the 6-Coin Portfolio (43 Setups v2.0 + HOT_ADAPTIVE setups) at Bao 400G (Max 5 concurrent) over 12M & OOS 6M!
"""
import json
import os
import sys
from typing import Dict, Tuple

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
from run_full_216_exit_sweetspot import sim_exit_arch
from run_full_3ema_storm_guard import build_cooldown_mask, compute_storm_flags, generate_all_4_touches
from run_master_6coin_html import SIX_COINS
from run_research_mtf import resample_ohlcv


def load_rules_with_hot_adaptive() -> dict:
    with open(os.path.join(ROOT, "core", "core_strategy_rules.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    rules = {}
    for r in cfg["rules"]:
        rules[(r["coin"], r["tf"], r["setup_code"])] = {
            "arch": r["exit_arch"],
            "l2_u": float(r["leg2_sl_u"]),
            "sl_u": float(r["sl_u"]),
            "tp1_u": float(r["tp1_u"]),
            "tp2_u": float(r["tp2_u"]),
            "vol": float(r["adaptive_volume"]["init_vol_mult"]),
        }
    # Add HOT_ADAPTIVE setups from per-coin audit_48.csv (PF_1Y >= 1.35 & PF_OOS >= 1.80 & Lenh >= 15)
    hot_added = 0
    for sym in SIX_COINS:
        acsv = os.path.join(ROOT, "coins", sym, "audit_48.csv")
        if not os.path.exists(acsv):
            continue
        adf = pd.read_csv(acsv)
        hots = adf[(adf["Status"] == "SKIP") & (adf["TF"] != "5m") & (adf["Lenh_1Y"] >= 15) & (adf["PF_1Y"] >= 1.35) & (adf["PF_OOS"] >= 1.80)]
        for _, r in hots.iterrows():
            k = (sym, r["TF"], r["Setup"])
            if k not in rules:
                rules[k] = {
                    "arch": "PARTIAL_MOVE_SL",
                    "l2_u": -2.0,
                    "sl_u": 5.0 if sym == "BTCUSDT" else 6.0,
                    "tp1_u": 4.0 if sym == "BTCUSDT" else 3.5,
                    "tp2_u": 12.0 if sym == "BTCUSDT" else 10.0,
                    "vol": 0.40,
                }
                hot_added += 1
    print(f"Loaded {len(rules)} total setups (43 v2.0 Core+Scout + {hot_added} HOT_ADAPTIVE @ 0.40x).")
    return rules


def sim_with_pause_policy(
    tf_dfs_map: dict,
    tf_sigs_map: dict,
    rules_map: dict,
    scope: str,           # "NONE", "SETUP", "COIN", "PORTFOLIO"
    trigger_sl: int,      # 2 or 3 consecutive SLs
    resume_mode: str,     # "12h", "24h", "48h", "SHADOW_WIN"
    start_time: pd.Timestamp = None,
) -> dict:
    rcfg = RiskConfig(initial_capital=10_000.0, buffer_units=400.0, max_concurrent_trades=5, max_trades_per_day=10)
    comp = CompoundingManager(rcfg)
    equity = 10_000.0
    peak_eq = equity
    open_cap = 4.25

    cands = []
    for tfl, sm in tf_sigs_map.items():
        for sym, sigs in sm.items():
            for s in sigs:
                k = (sym, tfl, s.setup_type)
                if k not in rules_map:
                    continue
                if start_time is not None and s.timestamp < start_time:
                    continue
                cands.append((tfl, rules_map[k], s))

    tf_prio = {"1h": 0, "30m": 1, "15m": 2, "5m": 3}
    cands.sort(key=lambda x: (x[2].timestamp, tf_prio[x[0]], -x[1]["vol"], x[2].symbol))

    active, closed = [], []
    daily_t, daily_pnl, daily_eq = {}, {}, {}
    eq_curve = [equity]

    # Pause tracking dictionaries keyed by scope_key
    consec_sl_by_key: Dict[str, int] = {}
    paused_until: Dict[str, pd.Timestamp] = {}
    shadow_paused: Dict[str, bool] = {}
    pause_events = 0
    skipped_by_pause = 0

    def scope_key(sym: str, tfl: str, stype: str) -> str:
        if scope == "SETUP":
            return f"{sym}:{tfl}:{stype}"
        if scope == "COIN":
            return sym
        return "PORTFOLIO"

    hours_map = {"12h": 12, "24h": 24, "48h": 48, "SHADOW_WIN": 72}
    cd_hours = hours_map.get(resume_mode, 24)

    def settle(now_ts):
        nonlocal equity, peak_eq, pause_events
        keep, done = [], []
        for ex_ts, tr in active:
            (done if ex_ts <= now_ts else keep).append((ex_ts, tr))
        done.sort(key=lambda x: x[0])
        for ex_ts, tr in done:
            equity += tr["pnl"]
            peak_eq = max(peak_eq, equity)
            comp.update_equity(equity, ex_ts)
            dk = tr["dk"]
            daily_pnl[dk] = daily_pnl.get(dk, 0.0) + tr["pnl"]

            if scope != "NONE":
                sk = tr["sk"]
                if tr["out"] == "SL":
                    consec_sl_by_key[sk] = consec_sl_by_key.get(sk, 0) + 1
                    if consec_sl_by_key[sk] >= trigger_sl:
                        pause_events += 1
                        consec_sl_by_key[sk] = 0
                        if resume_mode == "SHADOW_WIN":
                            shadow_paused[sk] = True
                            paused_until[sk] = ex_ts + pd.Timedelta(hours=72)
                        else:
                            paused_until[sk] = ex_ts + pd.Timedelta(hours=cd_hours)
                elif tr["pnl"] > 0:
                    consec_sl_by_key[sk] = 0

            closed.append(tr)
            eq_curve.append(equity)
        active[:] = keep

    for tfl, rule, sig in cands:
        settle(sig.timestamp)
        if equity <= 0:
            break
        dk = sig.timestamp.strftime("%Y-%m-%d")
        if dk not in daily_eq:
            daily_eq[dk] = equity

        sk = scope_key(sig.symbol, tfl, sig.setup_type)
        df = tf_dfs_map[tfl][sig.symbol]
        u = max(0.5 * sig.atr, sig.entry_price * 1e-5)

        # Check if paused under current policy
        is_paused = False
        if scope != "NONE":
            if resume_mode == "SHADOW_WIN":
                if shadow_paused.get(sk, False):
                    if sig.timestamp >= paused_until.get(sk, sig.timestamp):
                        shadow_paused[sk] = False
                    else:
                        is_paused = True
            else:
                if sig.timestamp < paused_until.get(sk, pd.Timestamp("1970-01-01", tz="UTC")):
                    is_paused = True

        if is_paused:
            skipped_by_pause += 1
            # In SHADOW_WIN mode, evaluate signal silently: if it wins, resume immediately!
            if resume_mode == "SHADOW_WIN":
                _, _, _, _, nu_sh = sim_exit_arch(
                    df["high"].values, df["low"].values, df["close"].values,
                    sig.bar_idx, sig.entry_price, sig.direction, u,
                    rule["sl_u"], rule["tp1_u"], rule["tp2_u"], rule["arch"], rule["l2_u"],
                )
                if nu_sh > 0:
                    shadow_paused[sk] = False
            continue

        if daily_t.get(dk, 0) >= 10 or daily_pnl.get(dk, 0.0) <= -0.05 * daily_eq[dk]:
            continue
        if len(active) >= 5 or any(tr["sym"] == sig.symbol for _, tr in active):
            continue

        eff_m = rule["vol"]
        if sig.pa_vol_ok:
            eff_m *= 0.5
        if (equity - peak_eq) / peak_eq * 100.0 <= -6.0:
            eff_m = min(eff_m, 0.25)
        eff_m = min(eff_m, max(0.0, open_cap - sum(tr["m"] for _, tr in active)))
        if eff_m < 0.20:
            continue

        out, _, _, _, nu = sim_exit_arch(
            df["high"].values, df["low"].values, df["close"].values,
            sig.bar_idx, sig.entry_price, sig.direction, u,
            rule["sl_u"], rule["tp1_u"], rule["tp2_u"], rule["arch"], rule["l2_u"],
        )
        sb = min(comp.tier_capital, equity)
        qty = (sb * eff_m) / (400.0 * u)
        pnl = nu * u * qty

        ex_i = min(sig.bar_idx + (4 if out == "SL" else 8), len(df) - 1)
        ex_ts = pd.Timestamp(df["timestamp"].iloc[ex_i])
        if ex_ts.tzinfo is None:
            ex_ts = ex_ts.tz_localize("UTC")
        active.append((ex_ts, {"sym": sig.symbol, "sk": sk, "m": eff_m, "out": out, "pnl": pnl, "dk": dk}))
        daily_t[dk] = daily_t.get(dk, 0) + 1

    if active:
        settle(pd.Timestamp("2099-01-01", tz="UTC"))

    ev = np.array(eq_curve)
    pk = np.maximum.accumulate(ev)
    mdd = float(((ev - pk) / pk * 100.0).min())
    pnls = np.array([t["pnl"] for t in closed])
    wr = (pnls > 0).mean() * 100.0 if len(pnls) else 0.0
    pf = float(pnls[pnls > 0].sum() / max(-pnls[pnls < 0].sum(), 0.01))
    ret = (float(ev[-1]) / 10_000.0 - 1.0) * 100.0
    return {
        "trades": len(closed), "pauses": pause_events, "skipped": skipped_by_pause,
        "wr": wr, "pf": pf, "ret": ret, "mdd": mdd, "calmar": ret / max(abs(mdd), 0.1),
    }


def main():
    rules_map = load_rules_with_hot_adaptive()
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

    policies = [
        ("0. Khong Pause (Baseline)", "NONE", 99, "24h"),
        ("1. SETUP Pause: 2 SL -> nghi 24h", "SETUP", 2, "24h"),
        ("2. SETUP Pause: 2 SL -> nghi 48h", "SETUP", 2, "48h"),
        ("3. SETUP Pause: 2 SL -> Shadow-Win", "SETUP", 2, "SHADOW_WIN"),
        ("4. COIN Pause: 2 SL -> nghi 12h", "COIN", 2, "12h"),
        ("5. COIN Pause: 2 SL -> nghi 24h (hom sau)", "COIN", 2, "24h"),
        ("6. COIN Pause: 2 SL -> nghi 48h", "COIN", 2, "48h"),
        ("7. COIN Pause: 2 SL -> Shadow-Win", "COIN", 2, "SHADOW_WIN"),
        ("8. COIN Pause: 3 SL -> nghi 24h", "COIN", 3, "24h"),
        ("9. PORTFOLIO Pause: 3 SL -> nghi 12h", "PORTFOLIO", 3, "12h"),
        ("10. PORTFOLIO Pause: 3 SL -> nghi 24h", "PORTFOLIO", 3, "24h"),
        ("11. PORTFOLIO Pause: 2 SL -> Shadow-Win", "PORTFOLIO", 2, "SHADOW_WIN"),
    ]

    rows = []
    for label, sc, trig, rmode in policies:
        r12 = sim_with_pause_policy(tf_dfs_map, tf_sigs_map, rules_map, sc, trig, rmode, None)
        r6 = sim_with_pause_policy(tf_dfs_map, tf_sigs_map, rules_map, sc, trig, rmode, mid_ts)
        rows.append(
            {
                "Chinh_Sach_Pause_Resume": label,
                "So_Lan_Pause": r12["pauses"],
                "Bo_Qua_Lenh": r12["skipped"],
                "12M_Lenh": r12["trades"],
                "12M_WR%": r12["wr"],
                "12M_PF": r12["pf"],
                "12M_Lai%": r12["ret"],
                "12M_MaxDD%": r12["mdd"],
                "Calmar": r12["calmar"],
                "OOS_Lai%": r6["ret"],
                "OOS_MaxDD%": r6["mdd"],
            }
        )

    res_df = pd.DataFrame(rows).sort_values("Calmar", ascending=False)
    print("=" * 124)
    print("THUC NGHIEM DIEM NGOT CO CHE PAUSE & RESUME (SETUP vs COIN vs PORTFOLIO | 2-3 SL | 12h/24h/48h/SHADOW-WIN)")
    print("=" * 124)
    print(res_df.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
    res_df.to_csv(os.path.join(ROOT, "output_test", "pause_resume_sweetspot.csv"), index=False)


if __name__ == "__main__":
    main()
