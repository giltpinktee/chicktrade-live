"""CHICKTRADE — Argon Quant Volume Allocation & Setup Pruning Audit (2023-11 -> 2026-10).

Audits all 288 setups (focusing on the 113 previously qualified CORE/SCOUT setups)
from `s4_2023_2026_sweetspot_288.json` using strict institutional criteria:
  - Edge-per-Risk (EPR = s4.avg_u / sl_u): normalizes expectancy across SL 5U/6U/7U.
  - S5 Resilience (s5.net_u / s4.net_u) & S5 PF: rejects fee/slippage-fragile setups.
  - Sample Reliability (N): forbids S+ (1.25x) when N < 25 to prevent small-sample noise.
  - Anti-Fee-Grind Rule: drops high-frequency (N >= 60) setups with thin avg_u / EPR / PF.
  - Risk-Normalized SL 7U Cap: scales down average SL 7U setups (0.75x * 7U = 5.25U risk).

Compares on identical S4_REAL_FUNDING & S5_PESSIMISTIC simulators (Bao 400G & Auto-Turbo 250G):
  1. Baseline (113 setups, 2-tier 1.0x / 0.5x)
  2. Argon A (Pruned + 4-tier smart volume 1.25x / 1.00x / 0.75x / 0.50x, keeping elite 5m)
  3. Argon B (Strict Elite — drops all 5m + prunes all marginal/fee-grind setups, 15m/30m/1h only)

Outputs:
  - e:/CHICKTRADE/verified_tests/results/argon_volume_proposal.json
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from verified_tests.scripts.run_gold_standard_6year_audit import (
    SIX_COINS,
    MAX_HOLD_BARS,
    load_coin_6y,
    extract_ema_signals_6y,
    sim_trade_scenario,
    simulate_portfolio,
)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IN_JSON = os.path.join(ROOT, "verified_tests", "results", "s4_2023_2026_sweetspot_288.json")
OUT_JSON = os.path.join(ROOT, "verified_tests", "results", "argon_volume_proposal.json")
TFS = ("5m", "15m", "30m", "1h")


def compute_metrics(r: dict) -> dict:
    s4, s5 = r["s4"], r["s5"]
    sl_u = float(r["sl_u"])
    epr = s4["avg_u"] / sl_u if sl_u > 0 else 0.0
    s5_res = (s5["net_u"] / s4["net_u"]) if s4["net_u"] > 0 else 0.0
    return {
        "epr": round(epr, 4),
        "s5_res": round(s5_res, 4),
        "n": int(s4["n"]),
        "pwr": float(s4["pwr"]),
        "pf": float(s4["pf"]),
        "avg_u": float(s4["avg_u"]),
        "net_u": float(s4["net_u"]),
        "s5_pf": float(s5["pf"]),
        "s5_net_u": float(s5["net_u"]),
        "sl_u": sl_u,
    }


def grade_argon_a(r: dict, m: dict) -> Tuple[str, float, float, str]:
    """Argon A: Pruned + 4-tier smart volume (1.25x / 1.00x / 0.75x / 0.50x).

    Returns (tier, vol_mult, priority_score, reason).
    """
    if r["status"] == "DROP":
        return "DROP", 0.0, -99.0, "Already DROP in S4 sweep (failed base S4/S5 gate)"

    n, pwr, pf, avg_u, net_u = m["n"], m["pwr"], m["pf"], m["avg_u"], m["net_u"]
    epr, s5_res, s5_pf, sl_u = m["epr"], m["s5_res"], m["s5_pf"], m["sl_u"]

    # Hard Drop 1: S5 fragility (slippage/fee destroys >32% of net_u or S5 PF < 1.15)
    if s5_res < 0.68 or s5_pf < 1.15:
        return "DROP", 0.0, -50.0, f"S5 fragile (S5res={s5_res:.2f}, S5_PF={s5_pf:.2f})"

    # Hard Drop 2: Anti-Fee-Grind (high frequency N >= 60 with thin edge)
    if n >= 60 and (epr < 0.11 or avg_u < 0.65 or pf < 1.28):
        return "DROP", 0.0, -50.0, f"Fee-grind trap (N={n}, avg_u={avg_u:.2f}, EPR={epr:.3f}, PF={pf:.2f})"

    # Hard Drop 3: Thin Edge-per-Risk or thin PF across any N
    if epr < 0.095 or pf < 1.22 or avg_u < 0.45:
        return "DROP", 0.0, -50.0, f"Thin edge/risk (EPR={epr:.3f}, avg_u={avg_u:.2f}, PF={pf:.2f})"

    # Hard Drop 4: 5m noise filter (5m requires higher bar to justify fee/microstructure risk)
    if r["tf"] == "5m" and (epr < 0.14 or pf < 1.35 or s5_res < 0.80):
        return "DROP", 0.0, -50.0, f"5m fee/noise risk (EPR={epr:.3f}, PF={pf:.2f}, S5res={s5_res:.2f})"

    # Composite quality score for portfolio slot priority (higher EPR, PWR, S5 resilience first)
    q_score = epr * 180.0 + (pwr - 50.0) * 1.8 + min(pf, 3.5) * 12.0 + s5_res * 20.0 + min(n, 80) * 0.15

    # Tier S+ (1.25x): Elite Edge-per-Risk, proven sample N >= 25, rock-solid S5
    # SL 7U requires stricter EPR (>=0.25) to offset +40% SL risk per trade
    s_plus_epr_min = 0.25 if sl_u >= 7.0 else 0.21
    if n >= 25 and epr >= s_plus_epr_min and pwr >= 62.5 and pf >= 1.65 and s5_res >= 0.88 and s5_pf >= 1.55:
        return (
            "S+",
            1.25,
            round(q_score + 40.0, 2),
            f"Elite S+ (N={n}, EPR={epr:.3f}, PWR={pwr:.1f}%, PF={pf:.2f}, S5res={s5_res:.2f})",
        )

    # Tier A (1.00x): Strong Core, proven sample N >= 25, solid EPR & S5
    # SL 7U capped at B (0.75x) unless EPR >= 0.18 and PF >= 1.50
    if n >= 25 and pwr >= 57.5 and pf >= 1.36 and s5_res >= 0.80 and s5_pf >= 1.28:
        if sl_u >= 7.0 and (epr < 0.18 or pf < 1.50):
            return (
                "B",
                0.75,
                round(q_score + 10.0, 2),
                f"SL 7U risk-capped to 0.75x (risk=5.25U; EPR={epr:.3f}, PF={pf:.2f})",
            )
        if epr >= 0.14:
            return (
                "A",
                1.00,
                round(q_score + 25.0, 2),
                f"Strong Core A (N={n}, EPR={epr:.3f}, PWR={pwr:.1f}%, PF={pf:.2f})",
            )

    # Tier B (0.75x): Solid mid-tier OR high-edge small-sample (15 <= N < 25) capped for statistical safety
    if n < 25:
        if epr >= 0.20 and pwr >= 62.0 and pf >= 1.60 and s5_res >= 0.88:
            return (
                "B",
                0.75,
                round(q_score + 12.0, 2),
                f"High-edge small-sample cap (N={n}<25, EPR={epr:.3f}, PF={pf:.2f})",
            )
        return (
            "C",
            0.50,
            round(q_score, 2),
            f"Small-sample scout (N={n}<25, EPR={epr:.3f}, PF={pf:.2f})",
        )

    if epr >= 0.115 and pwr >= 56.0 and pf >= 1.28 and s5_res >= 0.76:
        return (
            "B",
            0.75,
            round(q_score + 10.0, 2),
            f"Mid-tier B / wide SL (N={n}, SL={sl_u:.0f}U, EPR={epr:.3f}, PF={pf:.2f})",
        )

    # Tier C (0.50x): Viable Scout
    return (
        "C",
        0.50,
        round(q_score, 2),
        f"Scout C (N={n}, EPR={epr:.3f}, PWR={pwr:.1f}%, PF={pf:.2f})",
    )


def grade_argon_b(r: dict, m: dict) -> Tuple[str, float, float, str]:
    """Argon B: Strict Elite — Drops ALL 5m + removes all marginal/fee-grind setups (15m/30m/1h only)."""
    if r["status"] == "DROP":
        return "DROP", 0.0, -99.0, "Already DROP in S4 sweep"

    if r["tf"] == "5m":
        return "DROP", 0.0, -60.0, "Argon B: 5m timeframe excluded (microstructure/fee noise)"

    tier_a, vm_a, sc_a, reason_a = grade_argon_a(r, m)
    if tier_a == "DROP":
        return "DROP", 0.0, -50.0, reason_a

    # Argon B stricter pruning: drop Tier C scouts with low EPR (<0.13) or weak S5 (<0.80)
    if tier_a == "C" and (m["epr"] < 0.13 or m["s5_res"] < 0.80 or m["pf"] < 1.32):
        return "DROP", 0.0, -45.0, f"Argon B strict prune (low edge/S5: EPR={m['epr']:.3f}, PF={m['pf']:.2f}, S5res={m['s5_res']:.2f})"

    # Also drop N >= 50 setups if EPR < 0.135 (keep only crisp high-efficiency flow)
    if m["n"] >= 50 and m["epr"] < 0.135:
        return "DROP", 0.0, -45.0, f"Argon B strict anti-grind (N={m['n']}, EPR={m['epr']:.3f})"

    return tier_a, vm_a, sc_a, reason_a


def run_setup_trades(
    market: Dict[str, Dict[str, pd.DataFrame]],
    funding_map: Dict[str, Tuple[np.ndarray, np.ndarray]],
    sig_map: Dict[Tuple[str, str, str], List[dict]],
    setup_row: dict,
    t_start: int,
    t_end: int,
    scenario: str,
) -> Tuple[List[dict], int]:
    sym, tf, st_name = setup_row["sym"], setup_row["tf"], setup_row["setup"]
    sigs = [s for s in sig_map.get((sym, tf, st_name), []) if t_start <= s["open_ts"] < t_end]
    if not sigs:
        return [], 0
    df = market[sym][tf]
    op = df["open"].values
    hi = df["high"].values
    lo = df["low"].values
    cl = df["close"].values
    ts = df["open_time_ms"].values
    fr_ts, fr_rate = funding_map[sym]
    sl_u = float(setup_row["sl_u"])
    tp1_u = float(setup_row["tp1_u"])
    tp2_u = float(setup_row["tp2_u"])
    mode = setup_row["mode"]
    mb = MAX_HOLD_BARS[tf]

    busy_idx = -1
    tl = []
    miss = 0
    for s in sigs:
        if s["idx"] <= busy_idx:
            continue
        res = sim_trade_scenario(op, hi, lo, cl, ts, fr_ts, fr_rate, s, sl_u, tp1_u, tp2_u, mode, mb, scenario)
        if res is None:
            continue
        if res.get("ioc_missed"):
            miss += 1
            continue
        busy_idx = s["idx"] + res["bars_held"]
        tl.append(res)
    return tl, miss


def compact_port(p: dict) -> dict:
    return {k: v for k, v in p.items() if k != "curve"}


def main():
    with open(IN_JSON, "r", encoding="utf-8") as f:
        src = json.load(f)

    all_rows = []
    for sym in SIX_COINS:
        for r in src["coins"][sym]["setups"]:
            all_rows.append(r)
    print(f"Loaded {len(all_rows)} setups from {IN_JSON}")

    print("Loading 6-year 5m data + exact 8h Binance Funding...")
    market: Dict[str, Dict[str, pd.DataFrame]] = {}
    funding_map: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    with ThreadPoolExecutor(max_workers=6) as pool:
        futs = {sym: pool.submit(load_coin_6y, sym) for sym in SIX_COINS}
        for sym, fut in futs.items():
            m_d, f_ts, f_rt = fut.result()
            market[sym] = m_d
            funding_map[sym] = (f_ts, f_rt)

    print("Extracting converged EMA 34/89/200 signals (W1-W3 + W4_REV)...")
    sig_map = extract_ema_signals_6y(market, tfs=TFS, max_wave=4)

    t_start = int(datetime(2023, 11, 1, tzinfo=timezone.utc).timestamp() * 1000)
    t_end = int(datetime(2026, 11, 1, tzinfo=timezone.utc).timestamp() * 1000)

    # Pre-simulate trades once for all 113 previously active setups under S4 & S5
    cache_s4: Dict[Tuple[str, str, str], Tuple[List[dict], int]] = {}
    cache_s5: Dict[Tuple[str, str, str], Tuple[List[dict], int]] = {}
    for r in all_rows:
        if r["status"] != "DROP":
            key = (r["sym"], r["tf"], r["setup"])
            cache_s4[key] = run_setup_trades(market, funding_map, sig_map, r, t_start, t_end, "S4_REAL_FUNDING")
            cache_s5[key] = run_setup_trades(market, funding_map, sig_map, r, t_start, t_end, "S5_PESSIMISTIC")

    # Build graded audit records for all 288 setups
    audited_setups = []
    pools = {
        "baseline": {"s4": [], "s5": [], "m4": 0, "m5": 0},
        "argon_a": {"s4": [], "s5": [], "m4": 0, "m5": 0},
        "argon_b": {"s4": [], "s5": [], "m4": 0, "m5": 0},
    }

    for r in all_rows:
        m = compute_metrics(r)
        tier_a, vm_a, sc_a, reason_a = grade_argon_a(r, m)
        tier_b, vm_b, sc_b, reason_b = grade_argon_b(r, m)
        key = (r["sym"], r["tf"], r["setup"])

        rec = {
            "sym": r["sym"],
            "tf": r["tf"],
            "setup": r["setup"],
            "sl_u": r["sl_u"],
            "tp1_u": r["tp1_u"],
            "tp2_u": r["tp2_u"],
            "mode": r["mode"],
            "baseline_status": r["status"],
            "baseline_vol": r["vol_mult"],
            "metrics": m,
            "argon_a": {"tier": tier_a, "vol_mult": vm_a, "score": sc_a, "reason": reason_a},
            "argon_b": {"tier": tier_b, "vol_mult": vm_b, "score": sc_b, "reason": reason_b},
        }
        audited_setups.append(rec)

        if r["status"] != "DROP":
            tr4, m4 = cache_s4[key]
            tr5, m5 = cache_s5[key]

            # Baseline
            pools["baseline"]["m4"] += m4
            pools["baseline"]["m5"] += m5
            for t in tr4:
                pools["baseline"]["s4"].append({**t, "vol_mult": r["vol_mult"], "score": r["score"]})
            for t in tr5:
                pools["baseline"]["s5"].append({**t, "vol_mult": r["vol_mult"], "score": r["score"]})

            # Argon A
            if vm_a > 0:
                pools["argon_a"]["m4"] += m4
                pools["argon_a"]["m5"] += m5
                for t in tr4:
                    pools["argon_a"]["s4"].append({**t, "vol_mult": vm_a, "score": sc_a})
                for t in tr5:
                    pools["argon_a"]["s5"].append({**t, "vol_mult": vm_a, "score": sc_a})

            # Argon B
            if vm_b > 0:
                pools["argon_b"]["m4"] += m4
                pools["argon_b"]["m5"] += m5
                for t in tr4:
                    pools["argon_b"]["s4"].append({**t, "vol_mult": vm_b, "score": sc_b})
                for t in tr5:
                    pools["argon_b"]["s5"].append({**t, "vol_mult": vm_b, "score": sc_b})

    results = {}
    for name in ("baseline", "argon_a", "argon_b"):
        p = pools[name]
        results[name] = {
            "s4_400g": compact_port(simulate_portfolio(p["s4"], p["m4"], 400, 250, False, 3, True)),
            "s4_turbo": compact_port(simulate_portfolio(p["s4"], p["m4"], 400, 250, True, 3, True)),
            "s5_400g": compact_port(simulate_portfolio(p["s5"], p["m5"], 400, 250, False, 3, True)),
            "s5_turbo": compact_port(simulate_portfolio(p["s5"], p["m5"], 400, 250, True, 3, True)),
        }

    # Tier breakdown helper
    def tier_counts(plan_key: str) -> dict:
        cnt = {"S+": 0, "A": 0, "B": 0, "C": 0, "DROP": 0}
        tf_cnt = {tf: 0 for tf in TFS}
        for rec in audited_setups:
            t = rec[plan_key]["tier"]
            cnt[t] = cnt.get(t, 0) + 1
            if t != "DROP":
                tf_cnt[rec["tf"]] += 1
        cnt["active_total"] = cnt["S+"] + cnt["A"] + cnt["B"] + cnt["C"]
        cnt["by_tf"] = tf_cnt
        return cnt

    summary_counts = {
        "baseline": {
            "CORE": sum(1 for r in audited_setups if r["baseline_status"] == "CORE"),
            "SCOUT": sum(1 for r in audited_setups if r["baseline_status"] == "SCOUT"),
            "DROP": sum(1 for r in audited_setups if r["baseline_status"] == "DROP"),
            "active_total": sum(1 for r in audited_setups if r["baseline_status"] != "DROP"),
        },
        "argon_a": tier_counts("argon_a"),
        "argon_b": tier_counts("argon_b"),
    }

    print("\n" + "=" * 105)
    print("ARGON QUANT — PORTFOLIO BENCHMARK (2023-11 -> 2026-10 | S4_REAL_FUNDING & S5_PESSIMISTIC)")
    print("=" * 105)
    for label, key in [("BASELINE (113 | 1.0x/0.5x)", "baseline"), ("ARGON A  (4-tier + 5m elite)", "argon_a"), ("ARGON B  (Strict Elite, no 5m)", "argon_b")]:
        c = summary_counts[key]
        print(f"\n>>> {label} | Active={c['active_total']} | Breakdown={c}")
        for sc_label, sc_key in [
            ("S4 Bao 400G  ", "s4_400g"),
            ("S4 Turbo 250G", "s4_turbo"),
            ("S5 Bao 400G  ", "s5_400g"),
            ("S5 Turbo 250G", "s5_turbo"),
        ]:
            r = results[key][sc_key]
            print(
                f"  [{sc_label}] Eq=${r['final_eq']:>10,.2f} ({r['roi_pct']:+7.1f}%) | "
                f"DD={r['max_dd_pct']:6.2f}% | Calmar={r['calmar']:5.2f} | "
                f"PF={r['pf']:.2f} | PWR={r['pwr']:4.1f}% | WR={r['wr']:4.1f}% | N={r['trades']:4d} | Fund=${r['funding_usd']:,.0f}"
            )

    # Print S+ setups and pruned setups from the 113 baseline
    s_plus = [r for r in audited_setups if r["argon_a"]["tier"] == "S+"]
    s_plus.sort(key=lambda x: -x["metrics"]["epr"])
    print(f"\n--- ARGON A: TIER S+ (1.25x) [{len(s_plus)} setups] ---")
    for r in s_plus:
        m = r["metrics"]
        print(f"  {r['sym']:8s} {r['tf']:3s} {r['setup']:12s} SL={r['sl_u']:.0f} {r['mode']:15s} N={m['n']:3d} PWR={m['pwr']:4.1f}% PF={m['pf']:.2f} avg={m['avg_u']:.3f} EPR={m['epr']:.3f} S5res={m['s5_res']:.2f}")

    pruned_a = [r for r in audited_setups if r["baseline_status"] != "DROP" and r["argon_a"]["tier"] == "DROP"]
    pruned_a.sort(key=lambda x: -x["metrics"]["n"])
    print(f"\n--- ARGON A: PRUNED FROM 113 BASELINE [{len(pruned_a)} setups dropped] ---")
    for r in pruned_a:
        m = r["metrics"]
        print(f"  {r['sym']:8s} {r['tf']:3s} {r['setup']:12s} was={r['baseline_status']:5s} SL={r['sl_u']:.0f} N={m['n']:3d} avg={m['avg_u']:.3f} EPR={m['epr']:.3f} PF={m['pf']:.2f} S5res={m['s5_res']:.2f} -> {r['argon_a']['reason']}")

    payload = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "period": src["period"],
        "methodology": {
            "edge_per_risk": "s4.avg_u / sl_u (normalizes expectancy across SL 5U/6U/7U)",
            "s5_resilience": "s5.net_u / s4.net_u (measures survival under pessimistic slippage/fees)",
            "small_sample_cap": "N < 25 forbidden from S+ (1.25x) and A (1.00x); capped at B (0.75x) or C (0.50x)",
            "anti_fee_grind": "N >= 60 with EPR < 0.11 or avg_u < 0.65 or PF < 1.28 -> DROP",
            "sl7_risk_cap": "SL 7U carries +40% risk/trade vs SL 5U; capped at 0.75x (5.25U risk) unless EPR >= 0.18 & PF >= 1.50 (A) or EPR >= 0.25 (S+)",
        },
        "counts": summary_counts,
        "portfolio_comparison": results,
        "setups": audited_setups,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] Saved {OUT_JSON}")


if __name__ == "__main__":
    main()
