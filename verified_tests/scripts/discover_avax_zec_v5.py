"""CHICKTRADE v5 — Independent Setup Discovery for AVAXUSDT & ZECUSDT (2023-11 -> 2026-10).

Downloads 6-year (2021-01 -> 2026-10) Binance USD-M Futures 5m klines + real 8h funding rates
from data.binance.vision (2021-2023 used as full EMA200 & wave warmup), then runs the EXACT
CHICKTRADE v5 Gold-Standard S4_REAL_FUNDING + S5_PESSIMISTIC discovery pipeline:
  - 4 TFs (5m, 15m, 30m, 1h) x 3 EMAs (34, 89, 200) x 4 Touches (W1, W2, W3, W4_REV) = 48 setups/coin
  - 11 Exit Profiles (EXIT_GRID)
  - Exact Argon B pruning + M2 6-Tier Volume + Cold-Only Cooling
  - Standalone portfolio sim (Bao 400G & Auto-Turbo 250G) per coin
  - Year-by-year stability breakdown (Y1: 2023-11..2024-10, Y2: 2024-11..2025-10, Y3: 2025-11..2026-10)

Output:
  - e:/CHICKTRADE/verified_tests/results/avax_zec_v5_discovery.json
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from verified_tests.scripts.download_6year_futures_and_funding import (
    OUT_DIR,
    _dl_5m_month,
    _dl_funding_month,
)
from verified_tests.scripts import run_gold_standard_6year_audit as gs_mod
from verified_tests.scripts.run_gold_standard_6year_audit import (
    GAMMA_TF,
    MAX_HOLD_BARS,
    MIN_U_PCT,
    compute_atr14,
    extract_ema_signals_6y,
    load_coin_6y,
    sim_trade_scenario,
)
from verified_tests.scripts.run_2023_2026_s4_sweetspot import EXIT_GRID, calc_stats
from verified_tests.scripts.run_argon_volume_audit import compute_metrics, grade_argon_b
from verified_tests.scripts.build_v5_6coin_final import m2_vol_tier, simulate_v5_portfolio

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_JSON = os.path.join(ROOT, "verified_tests", "results", "avax_zec_v5_discovery.json")
CANDIDATES = ["AVAXUSDT", "ZECUSDT"]
TFS = ("5m", "15m", "30m", "1h")

PERIODS = [
    ("Y1 (2023-11..2024-10)", int(datetime(2023, 11, 1, tzinfo=timezone.utc).timestamp() * 1000), int(datetime(2024, 11, 1, tzinfo=timezone.utc).timestamp() * 1000)),
    ("Y2 (2024-11..2025-10)", int(datetime(2024, 11, 1, tzinfo=timezone.utc).timestamp() * 1000), int(datetime(2025, 11, 1, tzinfo=timezone.utc).timestamp() * 1000)),
    ("Y3 (2025-11..2026-10)", int(datetime(2025, 11, 1, tzinfo=timezone.utc).timestamp() * 1000), int(datetime(2026, 11, 1, tzinfo=timezone.utc).timestamp() * 1000)),
]


def download_coin_6y_direct(sym: str) -> None:
    """Download full 2021-01 -> 2026-10 5m klines + 8h funding from data.binance.vision."""
    out_5m = os.path.join(OUT_DIR, f"{sym}_5m_2021_2026.csv.gz")
    out_fr = os.path.join(OUT_DIR, f"{sym}_funding_2021_2026.csv.gz")

    months: List[Tuple[str, int, int]] = []
    for y in range(2021, 2027):
        end_m = 10 if y == 2026 else 12
        for m in range(1, end_m + 1):
            months.append((sym, y, m))

    if not os.path.exists(out_5m):
        print(f"  [DL 5m] {sym}: fetching {len(months)} monthly archives from data.binance.vision...")
        with ThreadPoolExecutor(max_workers=16) as pool:
            parts = [d for d in pool.map(_dl_5m_month, months) if len(d) > 0]
        full = pd.concat(parts, ignore_index=True)
        full = full.drop_duplicates("open_time_ms").sort_values("open_time_ms").reset_index(drop=True)
        full.to_csv(out_5m, index=False, compression="gzip")
        t0 = pd.to_datetime(full["open_time_ms"].iloc[0], unit="ms", utc=True).strftime("%Y-%m-%d")
        t1 = pd.to_datetime(full["open_time_ms"].iloc[-1], unit="ms", utc=True).strftime("%Y-%m-%d")
        print(f"  [5m OK] {sym}: {len(full):,} bars ({t0} -> {t1})")
    else:
        print(f"  [5m CACHE] {sym} already cached")

    if not os.path.exists(out_fr):
        print(f"  [DL FR] {sym}: fetching {len(months)} monthly real 8h funding archives...")
        with ThreadPoolExecutor(max_workers=16) as pool:
            parts_fr = [d for d in pool.map(_dl_funding_month, months) if len(d) > 0]
        fr_df = pd.concat(parts_fr, ignore_index=True) if parts_fr else pd.DataFrame(columns=["calc_time_ms", "funding_rate"])
        if len(fr_df) > 0:
            fr_df = fr_df.drop_duplicates("calc_time_ms").sort_values("calc_time_ms").reset_index(drop=True)
        fr_df.to_csv(out_fr, index=False, compression="gzip")
        print(f"  [FR OK] {sym}: {len(fr_df):,} real 8h funding prints")
    else:
        print(f"  [FR CACHE] {sym} already cached")


def yearly_breakdown(trades: List[dict]) -> Dict[str, dict]:
    out = {}
    for label, ts0, ts1 in PERIODS:
        sub = [t for t in trades if ts0 <= t["open_ts"] < ts1]
        out[label] = calc_stats(sub)
    return out


def compact_port(p: dict) -> dict:
    return {k: v for k, v in p.items() if k != "curve"}


def main():
    global CANDIDATES, OUT_JSON
    if len(sys.argv) > 1:
        CANDIDATES = [s.upper() for s in sys.argv[1:]]
        slug = "_".join(s.lower() for s in CANDIDATES)
        OUT_JSON = os.path.join(ROOT, "verified_tests", "results", f"{slug}_v5_discovery.json")

    print(f"=== STEP 1: DOWNLOAD 2021-2026 FUTURES 5M + REAL 8H FUNDING ({', '.join(CANDIDATES)}) ===")
    for sym in CANDIDATES:
        download_coin_6y_direct(sym)

    print("\n=== STEP 2: LOAD DATA & EXTRACT CONVERGED EMA SIGNALS ===")
    load_syms = ["BTCUSDT"] + CANDIDATES
    market: Dict[str, Dict[str, pd.DataFrame]] = {}
    funding_map: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    with ThreadPoolExecutor(max_workers=3) as pool:
        futs = {sym: pool.submit(load_coin_6y, sym) for sym in load_syms}
        for sym, fut in futs.items():
            m_d, f_ts, f_rt = fut.result()
            market[sym] = m_d
            funding_map[sym] = (f_ts, f_rt)

    gs_mod.SIX_COINS = CANDIDATES
    sig_map = extract_ema_signals_6y(market, tfs=TFS, max_wave=4)

    t_start = int(datetime(2023, 11, 1, tzinfo=timezone.utc).timestamp() * 1000)
    t_end = int(datetime(2026, 11, 1, tzinfo=timezone.utc).timestamp() * 1000)
    expected_setups = [f"EMA{e}_W{w}" for e in (34, 89, 200) for w in (1, 2, 3)] + [f"EMA{e}_W4_REV" for e in (34, 89, 200)]

    print("\n=== STEP 3: FULL S4_REAL_FUNDING DISCOVERY + ARGON B / M2 AUDIT ===")
    results = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "period": "2023-11-01 -> 2026-10-31 (36M · Warmup 2021-2023 · S4_REAL_FUNDING + S5_PESSIMISTIC)",
        "coins": {},
    }

    for sym in CANDIDATES:
        df4 = market[sym]["4h"]
        df4_sub = df4[(df4["open_time_ms"] >= t_start) & (df4["open_time_ms"] < t_end)]
        macro4 = ((0.5 * compute_atr14(df4)) / df4["close"]).ewm(span=20, adjust=False).mean()
        macro4_sub = macro4.loc[df4_sub.index]
        u_avg = {tf: round(float(max(MIN_U_PCT, macro4_sub.mean() * GAMMA_TF[tf]) * 100.0), 3) for tf in TFS}
        fr_ts, fr_rate = funding_map[sym]

        # Price regime per year
        price_regime = {}
        for label, ts0, ts1 in PERIODS:
            sub4 = df4[(df4["open_time_ms"] >= ts0) & (df4["open_time_ms"] < ts1)]
            if len(sub4) > 0:
                price_regime[label] = {
                    "open": round(float(sub4["open"].iloc[0]), 3),
                    "low": round(float(sub4["low"].min()), 3),
                    "high": round(float(sub4["high"].max()), 3),
                    "close": round(float(sub4["close"].iloc[-1]), 3),
                    "u_15m_avg_pct": round(float(max(MIN_U_PCT, macro4.loc[sub4.index].mean() * GAMMA_TF["15m"]) * 100.0), 3),
                }

        all_setups = []
        v5_trades_s4 = []
        v5_trades_s5 = []
        v5_missed_s4 = 0
        v5_missed_s5 = 0

        for tf in TFS:
            df = market[sym][tf]
            op, hi, lo, cl, ts = df["open"].values, df["high"].values, df["low"].values, df["close"].values, df["open_time_ms"].values
            mb = MAX_HOLD_BARS[tf]

            for st_name in expected_setups:
                sigs = [s for s in sig_map.get((sym, tf, st_name), []) if t_start <= s["open_ts"] < t_end]
                if not sigs:
                    continue

                best_sc = -1e12
                best_cfg = EXIT_GRID[0]
                best_tr: List[dict] = []
                best_miss = 0
                best_st = calc_stats([])

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
                # S5 stress check
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

                # Base S4/S5 gate (exact same as run_2023_2026_s4_sweetspot.py)
                if best_st["n"] >= 25 and best_st["pwr"] >= 56.0 and best_st["pf"] >= 1.28 and best_st["net_u"] >= 25.0 and best_st["avg_u"] >= 0.40 and s5_st["net_u"] > 0:
                    base_status = "CORE"
                elif best_st["n"] >= 15 and best_st["pwr"] >= 53.5 and best_st["pf"] >= 1.15 and best_st["net_u"] >= 12.0 and best_st["avg_u"] >= 0.25 and s5_st["net_u"] > 0:
                    base_status = "SCOUT"
                else:
                    base_status = "DROP"

                row = {
                    "sym": sym, "tf": tf, "setup": st_name, "status": base_status,
                    "sl_u": sl_u, "tp1_u": tp1_u, "tp2_u": tp2_u, "mode": mode,
                    "s4": best_st, "s5": s5_st,
                }
                m = compute_metrics(row)
                ab_tier, ab_vm, ab_sc, ab_reason = grade_argon_b(row, m)
                m2_vm, m2_label = m2_vol_tier(best_st["pwr"], best_st["n"]) if ab_tier != "DROP" else (0.0, "DROP")

                yr_stats = yearly_breakdown(best_tr)
                pos_years = sum(1 for yst in yr_stats.values() if yst["net_u"] > 0)

                entry = {
                    "sym": sym, "tf": tf, "setup": st_name,
                    "base_status": base_status,
                    "argon_b_tier": ab_tier,
                    "m2_tier": m2_label,
                    "m2_vol": m2_vm,
                    "score": ab_sc,
                    "reason": ab_reason,
                    "sl_u": sl_u, "tp1_u": tp1_u, "tp2_u": tp2_u, "mode": mode,
                    "metrics": m,
                    "s4": best_st,
                    "s5": s5_st,
                    "yearly": yr_stats,
                    "pos_years": pos_years,
                }
                all_setups.append(entry)

                if ab_tier != "DROP":
                    v5_missed_s4 += best_miss
                    v5_missed_s5 += s5_miss
                    for tr in best_tr:
                        v5_trades_s4.append({**tr, "vol_mult": m2_vm, "score": ab_sc})
                    for tr in s5_tr:
                        v5_trades_s5.append({**tr, "vol_mult": m2_vm, "score": ab_sc})

        qualified = sorted([x for x in all_setups if x["argon_b_tier"] != "DROP"], key=lambda x: -x["score"])
        base_pass = [x for x in all_setups if x["base_status"] != "DROP"]

        p_s4_400 = simulate_v5_portfolio(v5_trades_s4, v5_missed_s4, 400, 250, False, 3) if v5_trades_s4 else {}
        p_s4_turbo = simulate_v5_portfolio(v5_trades_s4, v5_missed_s4, 400, 250, True, 3) if v5_trades_s4 else {}
        p_s5_400 = simulate_v5_portfolio(v5_trades_s5, v5_missed_s5, 400, 250, False, 3) if v5_trades_s5 else {}
        p_s5_turbo = simulate_v5_portfolio(v5_trades_s5, v5_missed_s5, 400, 250, True, 3) if v5_trades_s5 else {}

        results["coins"][sym] = {
            "u_avg": u_avg,
            "price_regime": price_regime,
            "n_base_pass": len(base_pass),
            "n_v5_qualified": len(qualified),
            "portfolio": {
                "s4_400g": compact_port(p_s4_400),
                "s4_turbo": compact_port(p_s4_turbo),
                "s5_400g": compact_port(p_s5_400),
                "s5_turbo": compact_port(p_s5_turbo),
            },
            "yearly_qualified_trades": yearly_breakdown(v5_trades_s4),
            "qualified_setups": qualified,
            "all_48_count": len(all_setups),
        }

        print(f"\n================ {sym} ================")
        print("  Price Regime by Year:")
        for lbl, pr in price_regime.items():
            print(f"    {lbl}: Open={pr['open']} | Low={pr['low']} | High={pr['high']} | Close={pr['close']} | 1U_15m={pr['u_15m_avg_pct']}%")
        print(f"  Base S4/S5 Pass: {len(base_pass)}/48 | Argon B (v5) Qualified: {len(qualified)}/48")
        if p_s4_400:
            print(f"  [S4] Bao 400G : ${p_s4_400['final_eq']:>10,.2f} ({p_s4_400['roi_pct']:+7.1f}%) | DD={p_s4_400['max_dd_pct']:6.2f}% | PF={p_s4_400['pf']} | PWR={p_s4_400['pwr']}% | N={p_s4_400['trades']} | Fund=${p_s4_400['funding_usd']:,.0f}")
            print(f"  [S4] Turbo250G: ${p_s4_turbo['final_eq']:>10,.2f} ({p_s4_turbo['roi_pct']:+7.1f}%) | DD={p_s4_turbo['max_dd_pct']:6.2f}% | PF={p_s4_turbo['pf']} | PWR={p_s4_turbo['pwr']}% | N={p_s4_turbo['trades']}")
            print(f"  [S5] Bao 400G : ${p_s5_400['final_eq']:>10,.2f} ({p_s5_400['roi_pct']:+7.1f}%) | DD={p_s5_400['max_dd_pct']:6.2f}% | PF={p_s5_400['pf']} | PWR={p_s5_400['pwr']}%")
        print("  Yearly Raw Qualified Trades (S4):")
        for lbl, yst in yearly_breakdown(v5_trades_s4).items():
            print(f"    {lbl}: N={yst['n']:>3} | WR={yst['wr']:>5.1f}% | PWR={yst['pwr']:>5.1f}% | PF={yst['pf']:>4.2f} | NetU={yst['net_u']:>+7.1f}U | Fund={yst['fund_u']:>+5.1f}U")
        print("  Discovered v5 Setups:")
        for q in qualified:
            y1 = q["yearly"][PERIODS[0][0]]["net_u"]
            y2 = q["yearly"][PERIODS[1][0]]["net_u"]
            y3 = q["yearly"][PERIODS[2][0]]["net_u"]
            print(f"    {q['tf']:<3} {q['setup']:<13} | {q['m2_tier']:<11} | -{q['sl_u']:.0f}/+{q['tp1_u']}/+{q['tp2_u']:.0f} {q['mode']:<15} | N={q['s4']['n']:>2} PWR={q['s4']['pwr']:>5.1f}% PF={q['s4']['pf']:>4.2f} Net={q['s4']['net_u']:>+6.1f}U S5={q['s5']['net_u']:>+6.1f}U | Y1={y1:>+5.1f} Y2={y2:>+5.1f} Y3={y3:>+5.1f} ({q['pos_years']}/3)")

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] Saved {OUT_JSON}")


if __name__ == "__main__":
    main()
