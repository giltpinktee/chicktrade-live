"""CHICKTRADE — 2023-2026 (36M) Gold-Standard S4_REAL_FUNDING Sweet-Spot & 288-Setup Discovery.

Exact scope requested by user:
  - Period: 2023-11-01 -> 2026-10-31 (36 months, with full 2021-2023 pre-warmup so EMA200 & wave counts are 100% converged!)
  - Universe: 6 Coins x 4 TFs (5m, 15m, 30m, 1h) x 3 EMAs (34, 89, 200) x Touches 1-4 (W1, W2, W3, W4_REV) = 288 Setups
  - Gold-Standard S4_REAL_FUNDING Simulator:
      * Real exit bar & timestamp (natural hold up to 30 days, NO 4/8-bar shortcut)
      * Conservative SL-before-TP ordering in same candle
      * Exact LIMIT IOC +/- 0.25U fill check on next bar open/high/low
      * Taker entry 0.05% + Maker TP 0.02% / Taker SL 0.05% + 1 bps slippage
      * Exact historical 8-hour Binance Funding Rate integrated at every 8h timestamp
  - Outputs:
      * e:/CHICKTRADE/verified_tests/results/s4_2023_2026_sweetspot_288.json
      * e:/CHICKTRADE/core/reports/chicktrade_2023_2026_s4_sweetspot_report.html
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from verified_tests.scripts.run_gold_standard_6year_audit import (
    SIX_COINS,
    GAMMA_TF,
    TF_COOLDOWN,
    TF_PRIORITY,
    TF_MS,
    MAX_HOLD_BARS,
    TAKER_FEE,
    MAKER_FEE,
    SLIP_BPS,
    MIN_U_PCT,
    load_coin_6y,
    compute_atr14,
    extract_ema_signals_6y,
    sim_trade_scenario,
    simulate_portfolio,
)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_JSON = os.path.join(ROOT, "verified_tests", "results", "s4_2023_2026_sweetspot_288.json")
TFS = ("5m", "15m", "30m", "1h")

EXIT_GRID = [
    (5.0, 3.5, 10.0, "PARTIAL_BE"),
    (5.0, 3.5, 12.0, "PARTIAL_M1U"),
    (5.0, 4.0, 12.0, "PARTIAL_NO_MOVE"),
    (6.0, 3.5, 10.0, "PARTIAL_BE"),
    (6.0, 4.0, 12.0, "PARTIAL_M1U"),
    (6.0, 4.0, 15.0, "PARTIAL_M1U"),
    (6.0, 4.0, 12.0, "PARTIAL_NO_MOVE"),
    (7.0, 4.0, 15.0, "PARTIAL_NO_MOVE"),
    (5.0, 3.5, 3.5, "ONLY_TP1"),
    (6.0, 4.0, 4.0, "ONLY_TP1"),
    (5.0, 5.0, 5.0, "ONLY_1R"),
]


def calc_stats(trades: List[dict], missed: int = 0) -> dict:
    n = len(trades)
    if n == 0:
        return {"n": 0, "missed": missed, "wr": 0.0, "pwr": 0.0, "pf": 0.0, "net_u": 0.0, "avg_u": 0.0, "avg_h": 0.0, "fund_u": 0.0}
    gp = sum(t["pnl_u"] for t in trades if t["pnl_u"] > 0)
    gl = sum(abs(t["pnl_u"]) for t in trades if t["pnl_u"] <= 0)
    wins = sum(1 for t in trades if t["pnl_u"] > 0)
    net = gp - gl
    avg_h = float(np.mean([(t["close_ts"] - t["open_ts"]) / 3_600_000.0 for t in trades]))
    fund_u = sum(t.get("funding_u", 0.0) for t in trades)
    return {
        "n": n,
        "missed": missed,
        "wr": round(wins / n * 100.0, 1),
        "pwr": round(gp / max(gp + gl, 1e-9) * 100.0, 1),
        "pf": round(gp / max(gl, 1e-9), 2) if gl > 1e-9 else (9.99 if gp > 0 else 0.0),
        "net_u": round(net, 2),
        "avg_u": round(net / n, 3),
        "avg_h": round(avg_h, 1),
        "fund_u": round(fund_u, 2),
    }


def main():
    print("Loading 6-year 5m data + exact 8h Binance Funding (using 2021-2023 as full EMA200 warmup)...")
    market: Dict[str, Dict[str, pd.DataFrame]] = {}
    funding_map: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    with ThreadPoolExecutor(max_workers=6) as pool:
        futs = {sym: pool.submit(load_coin_6y, sym) for sym in SIX_COINS}
        for sym, fut in futs.items():
            m_d, f_ts, f_rt = fut.result()
            market[sym] = m_d
            funding_map[sym] = (f_ts, f_rt)

    print("Extracting converged EMA 34/89/200 Touches 1-4 (W1, W2, W3, W4_REV) across M5/M15/M30/1H...")
    sig_map = extract_ema_signals_6y(market, tfs=TFS, max_wave=4)

    t_start = int(datetime(2023, 11, 1, tzinfo=timezone.utc).timestamp() * 1000)
    t_end = int(datetime(2026, 11, 1, tzinfo=timezone.utc).timestamp() * 1000)
    expected_setups = [f"EMA{e}_W{w}" for e in (34, 89, 200) for w in (1, 2, 3)] + [f"EMA{e}_W4_REV" for e in (34, 89, 200)]

    per_coin = {}
    family_acc: Dict[Tuple[str, str], List[dict]] = {}
    selected_trades_s4: List[dict] = []
    selected_trades_s5: List[dict] = []
    selected_missed_s4 = 0
    selected_missed_s5 = 0

    for sym in SIX_COINS:
        print(f"Sweeping S4_REAL_FUNDING Sweet Spots for {sym} (2023-11 -> 2026-10)...")
        df4 = market[sym]["4h"]
        df4_sub = df4[(df4["open_time_ms"] >= t_start) & (df4["open_time_ms"] < t_end)]
        macro4 = ((0.5 * compute_atr14(df4)) / df4["close"]).ewm(span=20, adjust=False).mean()
        macro4_sub = macro4.loc[df4_sub.index]
        u_avg = {tf: round(float(max(MIN_U_PCT, macro4_sub.mean() * GAMMA_TF[tf]) * 100.0), 3) for tf in TFS}
        fr_ts, fr_rate = funding_map[sym]

        coin_rows = []
        tf_summary = {}

        for tf in TFS:
            df = market[sym][tf]
            op, hi, lo, cl, ts = df["open"].values, df["high"].values, df["low"].values, df["close"].values, df["open_time_ms"].values
            mb = MAX_HOLD_BARS[tf]
            tf_best_trades = []

            for st_name in expected_setups:
                sigs = [s for s in sig_map.get((sym, tf, st_name), []) if t_start <= s["open_ts"] < t_end]
                if not sigs:
                    coin_rows.append({
                        "sym": sym, "tf": tf, "setup": st_name, "status": "DROP", "vol_mult": 0.0,
                        "sl_u": 5.0, "tp1_u": 3.5, "tp2_u": 10.0, "mode": "PARTIAL_BE",
                        "score": -99.0, "s4": calc_stats([]), "s5": calc_stats([]), "is_must": False,
                    })
                    continue

                best_sc = -1e12
                best_cfg = EXIT_GRID[0]
                best_tr = []
                best_miss = 0
                best_st = None

                for sl_u, tp1_u, tp2_u, mode in EXIT_GRID:
                    busy_idx = -1
                    tl = []
                    miss = 0
                    for s in sigs:
                        if s["idx"] <= busy_idx:
                            continue
                        res = sim_trade_scenario(op, hi, lo, cl, ts, fr_ts, fr_rate, s, sl_u, tp1_u, tp2_u, mode, mb, "S4_REAL_FUNDING")
                        if res is None:
                            continue
                        if res.get("ioc_missed"):
                            miss += 1
                            continue
                        busy_idx = s["idx"] + res["bars_held"]
                        tl.append(res)
                    st = calc_stats(tl, miss)
                    sc = st["net_u"] + (st["pwr"] - 50.0) * 3.0 + (st["pf"] - 1.0) * 15.0
                    if sc > best_sc:
                        best_sc = sc
                        best_cfg = (sl_u, tp1_u, tp2_u, mode)
                        best_tr = tl
                        best_miss = miss
                        best_st = st

                sl_u, tp1_u, tp2_u, mode = best_cfg
                # Stress-check exact same sweet spot under S5_PESSIMISTIC
                busy_idx = -1
                s5_tr = []
                s5_miss = 0
                for s in sigs:
                    if s["idx"] <= busy_idx:
                        continue
                    res = sim_trade_scenario(op, hi, lo, cl, ts, fr_ts, fr_rate, s, sl_u, tp1_u, tp2_u, mode, mb, "S5_PESSIMISTIC")
                    if res is None:
                        continue
                    if res.get("ioc_missed"):
                        s5_miss += 1
                        continue
                    busy_idx = s["idx"] + res["bars_held"]
                    s5_tr.append(res)
                s5_st = calc_stats(s5_tr, s5_miss)

                family_acc.setdefault((tf, st_name), []).extend(best_tr)
                tf_best_trades.extend(best_tr)

                # Strict S4_REAL_FUNDING Economic Grading (must also survive S5_PESSIMISTIC > 0!)
                if best_st["n"] >= 25 and best_st["pwr"] >= 56.0 and best_st["pf"] >= 1.28 and best_st["net_u"] >= 25.0 and best_st["avg_u"] >= 0.40 and s5_st["net_u"] > 0:
                    status, vm = "CORE", 1.0
                elif best_st["n"] >= 15 and best_st["pwr"] >= 53.5 and best_st["pf"] >= 1.15 and best_st["net_u"] >= 12.0 and best_st["avg_u"] >= 0.25 and s5_st["net_u"] > 0:
                    status, vm = "SCOUT", 0.5
                else:
                    status, vm = "DROP", 0.0

                score = (30.0 if status == "CORE" else (12.0 if status == "SCOUT" else -50.0)) + best_st["net_u"] * 0.45 + best_st["pwr"] * 0.6 + min(best_st["pf"], 3.5) * 5.0
                coin_rows.append({
                    "sym": sym, "tf": tf, "setup": st_name, "status": status, "vol_mult": vm,
                    "sl_u": sl_u, "tp1_u": tp1_u, "tp2_u": tp2_u, "mode": mode,
                    "score": round(score, 2), "s4": best_st, "s5": s5_st, "is_must": False,
                })
                if status in ("CORE", "SCOUT"):
                    selected_missed_s4 += best_miss
                    selected_missed_s5 += s5_miss
                    for tr in best_tr:
                        selected_trades_s4.append({**tr, "vol_mult": vm, "score": score})
                    for tr in s5_tr:
                        selected_trades_s5.append({**tr, "vol_mult": vm, "score": score})

            tf_summary[tf] = {
                "u_avg_pct": u_avg[tf],
                "n_core": sum(1 for r in coin_rows if r["tf"] == tf and r["status"] == "CORE"),
                "n_scout": sum(1 for r in coin_rows if r["tf"] == tf and r["status"] == "SCOUT"),
                "stats": calc_stats(tf_best_trades),
            }

        act = sorted([r for r in coin_rows if r["status"] != "DROP"], key=lambda x: -x["score"])
        must_keys = {(r["tf"], r["setup"]) for r in act[:5]}
        for r in coin_rows:
            r["is_must"] = (r["tf"], r["setup"]) in must_keys
        coin_rows.sort(key=lambda x: (-1 if x["status"] == "CORE" else (0 if x["status"] == "SCOUT" else 1), -x["score"]))

        per_coin[sym] = {
            "u_avg": u_avg,
            "n_core": sum(1 for r in coin_rows if r["status"] == "CORE"),
            "n_scout": sum(1 for r in coin_rows if r["status"] == "SCOUT"),
            "tf_summary": tf_summary,
            "setups": coin_rows,
        }

    p_s4_400 = simulate_portfolio(selected_trades_s4, selected_missed_s4, 400, 250, False, 3, True)
    p_s4_turbo = simulate_portfolio(selected_trades_s4, selected_missed_s4, 400, 250, True, 3, True)
    p_s5_400 = simulate_portfolio(selected_trades_s5, selected_missed_s5, 400, 250, False, 3, True)
    p_s5_turbo = simulate_portfolio(selected_trades_s5, selected_missed_s5, 400, 250, True, 3, True)

    tot_core = sum(c["n_core"] for c in per_coin.values())
    tot_scout = sum(c["n_scout"] for c in per_coin.values())
    print(f"\n=== 2023-2026 S4_REAL_FUNDING SWEET-SPOT PROPOSAL ({tot_core} CORE + {tot_scout} SCOUT = {tot_core+tot_scout} / 288) ===")
    print(f"  [S4 REAL FUNDING] Bao 400G : ${p_s4_400['final_eq']:>10,.2f} ({p_s4_400['roi_pct']:+7.1f}%) | DD={p_s4_400['max_dd_pct']:6.2f}% | PF={p_s4_400['pf']} | PWR={p_s4_400['pwr']}% | N={p_s4_400['trades']} | Fund=${p_s4_400['funding_usd']:,.0f}")
    print(f"  [S4 REAL FUNDING] Turbo250G: ${p_s4_turbo['final_eq']:>10,.2f} ({p_s4_turbo['roi_pct']:+7.1f}%) | DD={p_s4_turbo['max_dd_pct']:6.2f}% | PF={p_s4_turbo['pf']} | PWR={p_s4_turbo['pwr']}% | N={p_s4_turbo['trades']} | Fund=${p_s4_turbo['funding_usd']:,.0f}")
    print(f"  [S5 PESSIMISTIC ] Bao 400G : ${p_s5_400['final_eq']:>10,.2f} ({p_s5_400['roi_pct']:+7.1f}%) | DD={p_s5_400['max_dd_pct']:6.2f}% | PF={p_s5_400['pf']} | PWR={p_s5_400['pwr']}%")
    print(f"  [S5 PESSIMISTIC ] Turbo250G: ${p_s5_turbo['final_eq']:>10,.2f} ({p_s5_turbo['roi_pct']:+7.1f}%) | DD={p_s5_turbo['max_dd_pct']:6.2f}% | PF={p_s5_turbo['pf']} | PWR={p_s5_turbo['pwr']}%")

    # Breakdown by TF & Exit Mode across selected setups
    tf_counts = {tf: 0 for tf in TFS}
    mode_counts: Dict[str, int] = {}
    for c in per_coin.values():
        for r in c["setups"]:
            if r["status"] != "DROP":
                tf_counts[r["tf"]] += 1
                k = f"-{r['sl_u']}U/+{r['tp1_u']}U/+{r['tp2_u']}U ({r['mode']})"
                mode_counts[k] = mode_counts.get(k, 0) + 1
    print("  Qualified by TF:", tf_counts)
    print("  Top Sweet-Spot Exit Configs:", sorted(mode_counts.items(), key=lambda x: -x[1])[:6])

    family_rows = []
    for tf in TFS:
        for st_name in expected_setups:
            family_rows.append({"tf": tf, "setup": st_name, "stats": calc_stats(family_acc.get((tf, st_name), []))})

    payload = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "period": "2023-11-01 -> 2026-10-31 (36 tháng · Warmup 2021–2023 · S4 Funding Thật + IOC ±0.25U)",
        "total_core": tot_core,
        "total_scout": tot_scout,
        "tf_counts": tf_counts,
        "top_exit_configs": sorted(mode_counts.items(), key=lambda x: -x[1]),
        "portfolio": {
            "s4_400g": p_s4_400,
            "s4_turbo": p_s4_turbo,
            "s5_400g": p_s5_400,
            "s5_turbo": p_s5_turbo,
        },
        "family_rows": family_rows,
        "coins": per_coin,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[OK] Saved {OUT_JSON}")


if __name__ == "__main__":
    main()
