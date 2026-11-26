"""CHICKTRADE — Argon Quant Head-to-Head Benchmark of 4 Volume Mechanisms on 53 Argon B Setups.

Compares 4 volume sizing mechanisms on the exact 53 Argon B setups (15m/30m/1h) under:
  - Simulators: S4_REAL_FUNDING & S5_PESSIMISTIC
  - Modes: Bao 400G & Auto-Turbo 250G
  - Two independent windows:
      * Main 36M: 2023-11-01 -> 2026-10-31
      * Blind OOS 33M: 2021-02-01 -> 2023-10-31 (strict out-of-sample stability check)

Mechanisms:
  - M1: Argon B Static 4-Tier (1.25x / 1.00x / 0.75x / 0.50x)
  - M2: Legacy Multi-Tier Static 6-Step (0.25x / 0.50x / 0.75x / 1.00x / 1.25x / 1.50x) by PWR zones
  - M3: Legacy Dynamic Adaptive Volume (0.25x -> 1.50x) updating per-setup vol_mult by rolling PWR & win/SL streaks
  - M4: Hybrid SL-Normalized Multi-Step (0.25x -> 1.50x scaled by 5.0/sl_u + per-setup cold cooling + graduated DD cooling)

Outputs:
  - e:/CHICKTRADE/verified_tests/results/argon_volume_mechanisms_ab.json
"""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from verified_tests.scripts.run_gold_standard_6year_audit import (
    SIX_COINS,
    TF_PRIORITY,
    MAX_HOLD_BARS,
    load_coin_6y,
    extract_ema_signals_6y,
    sim_trade_scenario,
)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROPOSAL_JSON = os.path.join(ROOT, "verified_tests", "results", "argon_volume_proposal.json")
OUT_JSON = os.path.join(ROOT, "verified_tests", "results", "argon_volume_mechanisms_ab.json")
TFS = ("15m", "30m", "1h")
STEPS_6 = (0.25, 0.50, 0.75, 1.00, 1.25, 1.50)


def snap_step(val: float, steps=STEPS_6) -> float:
    return min(steps, key=lambda x: abs(x - val))


def m2_static_vol(rec: dict) -> float:
    """M2: Legacy 6-tier static volume (0.25x..1.50x) based on PWR zones + sample reliability."""
    m = rec["metrics"]
    pwr, n = m["pwr"], m["n"]
    if pwr >= 70.0 and n >= 25:
        return 1.50
    if pwr >= 65.0 and n >= 25:
        return 1.25
    if pwr >= 61.0 and n >= 25:
        return 1.00
    if pwr >= 58.0 or (pwr >= 65.0 and n < 25):
        return 0.75
    if pwr >= 55.0:
        return 0.50
    return 0.25


def m4_base_vol(rec: dict) -> float:
    """M4: Hybrid SL-Normalized multi-step base volume (0.25x..1.50x scaled by 5.0 / sl_u)."""
    m = rec["metrics"]
    tier = rec["argon_b"]["tier"]
    sl_u = float(rec["sl_u"])
    sl_norm = 5.0 / max(sl_u, 1.0)
    if tier == "S+":
        if m["epr"] >= 0.28 and m["pwr"] >= 66.0 and m["s5_res"] >= 0.90:
            raw = 1.50 * sl_norm
        else:
            raw = 1.25 * sl_norm
    elif tier == "A":
        raw = 1.00 * sl_norm
    elif tier == "B":
        raw = 0.75 * sl_norm
    else:
        raw = 0.50 * sl_norm
    return snap_step(max(0.25, min(1.50, raw)))


def simulate_portfolio_mech(
    trades: List[dict],
    ioc_missed: int = 0,
    mechanism: str = "M1",
    buffer_base: float = 400.0,
    turbo_buffer: float = 250.0,
    use_turbo: bool = False,
    max_same_dir: int = 3,
    use_2sl_brake: bool = True,
    init_cap: float = 10_000.0,
) -> dict:
    """Exact Gold-Standard portfolio simulator extended cleanly for M1/M2/M3/M4 (zero look-ahead)."""
    ordered = sorted(trades, key=lambda x: (x["open_ts"], -x.get("score", 0.0), TF_PRIORITY.get(x["tf"], 0)))
    eq = init_cap
    peak = init_cap
    max_dd = 0.0
    step = 0
    tier_cap = init_cap
    turbo_active = False
    consec_sl = 0
    shadow_paused = False
    shadow_until = 0
    open_pos: List[dict] = []
    closed: List[dict] = []
    total_funding = 0.0

    # Per-setup online state (updated ONLY when a trade closes -> zero look-ahead)
    setup_state: Dict[Tuple[str, str, str], dict] = {}

    def get_st(key: Tuple[str, str, str]) -> dict:
        if key not in setup_state:
            setup_state[key] = {"win_streak": 0, "sl_streak": 0, "hist": deque(maxlen=8)}
        return setup_state[key]

    def settle(now_ts: int):
        nonlocal eq, peak, max_dd, step, tier_cap, turbo_active, consec_sl, shadow_paused, shadow_until, total_funding
        open_pos.sort(key=lambda x: x["close_ts"])
        keep = []
        for p in open_pos:
            if p["close_ts"] <= now_ts:
                eq += p["pnl_usd"]
                total_funding += p["funding_usd"]
                peak = max(peak, eq)
                dd = (eq - peak) / peak * 100.0
                max_dd = min(max_dd, dd)
                while eq >= init_cap * (1.08 ** (step + 1)):
                    step += 1
                    tier_cap = init_cap * (1.08 ** step)
                while step > 0 and eq < tier_cap * 0.90:
                    step -= 1
                    tier_cap = init_cap * (1.08 ** step)
                if use_turbo:
                    if (not turbo_active) and eq >= init_cap * 1.25:
                        turbo_active = True
                    elif turbo_active and eq < init_cap * 1.15:
                        turbo_active = False
                if use_2sl_brake:
                    if p["outcome"] == "SL" and p.get("eff_base_mult", p.get("vol_mult", 1.0)) > 0.45:
                        consec_sl += 1
                        if consec_sl >= 2:
                            shadow_paused = True
                            shadow_until = p["close_ts"] + 72 * 3_600_000
                            consec_sl = 0
                    elif p["pnl_usd"] > 0:
                        consec_sl = 0

                # Update per-setup online stats on close
                skey = (p["sym"], p["tf"], p["setup"])
                st = get_st(skey)
                st["hist"].append(p["pnl_u"])
                if p["pnl_u"] > 0:
                    st["win_streak"] += 1
                    st["sl_streak"] = 0
                elif p["outcome"] == "SL":
                    st["sl_streak"] += 1
                    st["win_streak"] = 0
                else:
                    # Non-SL small loss resets win_streak
                    st["win_streak"] = 0

                closed.append(p)
            else:
                keep.append(p)
        open_pos[:] = keep

    for t in ordered:
        settle(t["open_ts"])
        if eq <= 0:
            break
        if shadow_paused:
            if t["open_ts"] >= shadow_until:
                shadow_paused = False
                consec_sl = 0
            else:
                if t["pnl_u"] > 0:
                    shadow_paused = False
                    consec_sl = 0
                continue
        if len(open_pos) >= 5 or any(p["sym"] == t["sym"] for p in open_pos):
            continue
        if sum(1 for p in open_pos if p["dir"] == t["dir"]) >= max_same_dir:
            continue

        base_m = float(t["vol_mult"])
        skey = (t["sym"], t["tf"], t["setup"])
        st = get_st(skey)

        if mechanism == "M3":
            # Legacy Dynamic Adaptive: hot streak / high rolling PWR boosts up to 1.50x; cold drops to 0.25x-0.50x
            hist = list(st["hist"])
            if len(hist) >= 5:
                gp_h = sum(x for x in hist if x > 0)
                gl_h = sum(abs(x) for x in hist if x <= 0)
                roll_pwr = gp_h / max(gp_h + gl_h, 1e-9) * 100.0
            else:
                roll_pwr = 60.0

            if st["sl_streak"] >= 3 or (len(hist) >= 5 and roll_pwr < 40.0):
                dyn_m = snap_step(max(0.25, base_m * 0.40))
            elif st["sl_streak"] == 2 or (len(hist) >= 5 and roll_pwr < 48.0):
                dyn_m = snap_step(max(0.25, base_m * 0.65))
            elif st["win_streak"] >= 4 or (len(hist) >= 6 and roll_pwr >= 68.0):
                dyn_m = snap_step(min(1.50, base_m + 0.25))
            else:
                dyn_m = base_m
            eff_m = dyn_m
        elif mechanism == "M4":
            # Hybrid SL-Normalized + asymmetric cold cooling (never over-leverage on hot streak)
            dyn_m = base_m
            if st["sl_streak"] >= 3:
                dyn_m = 0.25
            elif st["sl_streak"] == 2:
                dyn_m = snap_step(max(0.25, base_m - 0.25))
            eff_m = dyn_m
        else:
            # M1 / M2: Static multipliers
            eff_m = base_m

        eff_base_mult = eff_m

        if t.get("elevated_vol"):
            eff_m *= 0.5

        cur_dd = (eq - peak) / peak * 100.0
        if cur_dd <= -6.0:
            eff_m = min(eff_m, 0.25)
        elif mechanism == "M4" and cur_dd <= -3.5:
            # Graduated DD cooling before hitting -6% hard brake
            eff_m *= 0.75

        cur_buf = turbo_buffer if (use_turbo and turbo_active) else buffer_base
        if cur_buf == turbo_buffer:
            open_sl = sum(p["worst_sl_usd"] for p in open_pos)
            new_sl = (tier_cap * eff_m) * (t["sl_u"] / cur_buf) * 1.05
            if eq - (open_sl + new_sl) < init_cap * 1.10:
                cur_buf = buffer_base

        u_usd = (tier_cap * eff_m) / cur_buf
        notional = min(u_usd / max(t["u_frac"], 1e-6), 300_000.0 if t["sym"] in ("BTCUSDT", "ETHUSDT") else 150_000.0)
        u_usd = notional * t["u_frac"]
        if (sum(p["notional"] / 20.0 for p in open_pos) + notional / 20.0) > eq * 0.75:
            continue

        open_pos.append({
            **t,
            "eff_base_mult": eff_base_mult,
            "eff_mult_used": round(eff_m, 3),
            "notional": notional,
            "pnl_usd": t["pnl_u"] * u_usd,
            "funding_usd": t["funding_u"] * u_usd,
            "worst_sl_usd": u_usd * t["sl_u"] * 1.05,
        })

    settle(10**18)
    gp = sum(p["pnl_usd"] for p in closed if p["pnl_usd"] > 0)
    gl = sum(abs(p["pnl_usd"]) for p in closed if p["pnl_usd"] <= 0)
    wins = sum(1 for p in closed if p["pnl_usd"] > 0)
    n = len(closed)
    roi = (eq / init_cap - 1.0) * 100.0
    calmar = abs(roi / max_dd) if max_dd < -0.01 else 0.0
    avg_mult = float(np.mean([p["eff_mult_used"] for p in closed])) if closed else 0.0
    return {
        "trades": n,
        "ioc_missed": ioc_missed,
        "wr": round(wins / max(n, 1) * 100.0, 1),
        "pwr": round(gp / max(gp + gl, 1e-9) * 100.0, 1),
        "pf": round(gp / max(gl, 1e-9), 2) if gl > 1e-9 else 0.0,
        "final_eq": round(eq, 2),
        "roi_pct": round(roi, 2),
        "max_dd_pct": round(max_dd, 2),
        "calmar": round(calmar, 2),
        "max_step": step,
        "funding_usd": round(total_funding, 2),
        "avg_eff_mult": round(avg_mult, 3),
    }


def extract_window_trades(
    market: Dict[str, Dict[str, pd.DataFrame]],
    funding_map: Dict[str, Tuple[np.ndarray, np.ndarray]],
    sig_map: Dict[Tuple[str, str, str], List[dict]],
    setups_53: List[dict],
    t_start: int,
    t_end: int,
    scenario: str,
) -> Tuple[Dict[Tuple[str, str, str], List[dict]], int]:
    by_key: Dict[Tuple[str, str, str], List[dict]] = {}
    tot_miss = 0
    for rec in setups_53:
        sym, tf, st_name = rec["sym"], rec["tf"], rec["setup"]
        sigs = [s for s in sig_map.get((sym, tf, st_name), []) if t_start <= s["open_ts"] < t_end]
        if not sigs:
            by_key[(sym, tf, st_name)] = []
            continue
        df = market[sym][tf]
        op, hi, lo, cl, ts = df["open"].values, df["high"].values, df["low"].values, df["close"].values, df["open_time_ms"].values
        fr_ts, fr_rate = funding_map[sym]
        sl_u, tp1_u, tp2_u, mode = float(rec["sl_u"]), float(rec["tp1_u"]), float(rec["tp2_u"]), rec["mode"]
        mb = MAX_HOLD_BARS[tf]
        busy_idx = -1
        tl = []
        for s in sigs:
            if s["idx"] <= busy_idx:
                continue
            res = sim_trade_scenario(op, hi, lo, cl, ts, fr_ts, fr_rate, s, sl_u, tp1_u, tp2_u, mode, mb, scenario)
            if res is None:
                continue
            if res.get("ioc_missed"):
                tot_miss += 1
                continue
            busy_idx = s["idx"] + res["bars_held"]
            tl.append(res)
        by_key[(sym, tf, st_name)] = tl
    return by_key, tot_miss


def main():
    with open(PROPOSAL_JSON, "r", encoding="utf-8") as f:
        prop = json.load(f)

    setups_53 = [r for r in prop["setups"] if r["argon_b"]["tier"] != "DROP"]
    print(f"Loaded {len(setups_53)} Argon B setups from {PROPOSAL_JSON}")

    # Assign base multipliers for M1..M4
    setup_vols = []
    for rec in setups_53:
        v1 = float(rec["argon_b"]["vol_mult"])
        v2 = m2_static_vol(rec)
        v3 = v1  # M3 starts from Argon B base and adapts dynamically (0.25x..1.50x)
        v4 = m4_base_vol(rec)
        setup_vols.append({
            "sym": rec["sym"], "tf": rec["tf"], "setup": rec["setup"],
            "sl_u": rec["sl_u"], "tier_b": rec["argon_b"]["tier"],
            "score": rec["argon_b"]["score"],
            "pwr": rec["metrics"]["pwr"], "epr": rec["metrics"]["epr"], "n": rec["metrics"]["n"],
            "M1": v1, "M2": v2, "M3_init": v3, "M4_base": v4,
        })

    # Print multiplier distribution across the 53 setups
    for mk, col in [("M1 (4-tier static)", "M1"), ("M2 (6-tier PWR static)", "M2"), ("M4 (SL-norm base)", "M4_base")]:
        dist: Dict[float, int] = {}
        for sv in setup_vols:
            dist[sv[col]] = dist.get(sv[col], 0) + 1
        print(f"  {mk:24s}: {dict(sorted(dist.items()))}")

    print("\nLoading 6-year 5m data + exact 8h Binance Funding...")
    market: Dict[str, Dict[str, pd.DataFrame]] = {}
    funding_map: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    with ThreadPoolExecutor(max_workers=6) as pool:
        futs = {sym: pool.submit(load_coin_6y, sym) for sym in SIX_COINS}
        for sym, fut in futs.items():
            m_d, f_ts, f_rt = fut.result()
            market[sym] = m_d
            funding_map[sym] = (f_ts, f_rt)

    print("Extracting converged EMA 34/89/200 signals (15m/30m/1h)...")
    sig_map = extract_ema_signals_6y(market, tfs=TFS, max_wave=4)

    windows = {
        "main_2023_2026": (
            int(datetime(2023, 11, 1, tzinfo=timezone.utc).timestamp() * 1000),
            int(datetime(2026, 11, 1, tzinfo=timezone.utc).timestamp() * 1000),
            "2023-11-01 -> 2026-10-31 (36M Main)",
        ),
        "blind_2021_2023": (
            int(datetime(2021, 2, 1, tzinfo=timezone.utc).timestamp() * 1000),
            int(datetime(2023, 11, 1, tzinfo=timezone.utc).timestamp() * 1000),
            "2021-02-01 -> 2023-10-31 (33M Blind OOS)",
        ),
    }

    vol_map = {(sv["sym"], sv["tf"], sv["setup"]): sv for sv in setup_vols}
    all_results = {}

    for w_key, (t_start, t_end, w_label) in windows.items():
        print("\n" + "=" * 112)
        print(f"WINDOW: {w_label}")
        print("=" * 112)
        tr_s4_map, miss_s4 = extract_window_trades(market, funding_map, sig_map, setups_53, t_start, t_end, "S4_REAL_FUNDING")
        tr_s5_map, miss_s5 = extract_window_trades(market, funding_map, sig_map, setups_53, t_start, t_end, "S5_PESSIMISTIC")

        w_res = {}
        for mech, col in [("M1", "M1"), ("M2", "M2"), ("M3", "M3_init"), ("M4", "M4_base")]:
            s4_list = []
            s5_list = []
            for key, sv in vol_map.items():
                vm = sv[col]
                sc = sv["score"]
                for t in tr_s4_map.get(key, []):
                    s4_list.append({**t, "vol_mult": vm, "score": sc})
                for t in tr_s5_map.get(key, []):
                    s5_list.append({**t, "vol_mult": vm, "score": sc})

            w_res[mech] = {
                "s4_400g": simulate_portfolio_mech(s4_list, miss_s4, mech, 400, 250, False, 3, True),
                "s4_turbo": simulate_portfolio_mech(s4_list, miss_s4, mech, 400, 250, True, 3, True),
                "s5_400g": simulate_portfolio_mech(s5_list, miss_s5, mech, 400, 250, False, 3, True),
                "s5_turbo": simulate_portfolio_mech(s5_list, miss_s5, mech, 400, 250, True, 3, True),
            }

        all_results[w_key] = w_res

        for sc_label, sc_key in [
            ("S4 Bao 400G  ", "s4_400g"),
            ("S4 Turbo 250G", "s4_turbo"),
            ("S5 Bao 400G  ", "s5_400g"),
            ("S5 Turbo 250G", "s5_turbo"),
        ]:
            print(f"\n  [{sc_label}]")
            for mech, mdesc in [
                ("M1", "Argon B 4-Tier Static  "),
                ("M2", "Legacy 6-Tier Static   "),
                ("M3", "Legacy Dynamic Adaptive"),
                ("M4", "Hybrid SL-Norm + Cool  "),
            ]:
                r = w_res[mech][sc_key]
                print(
                    f"    {mech} ({mdesc}): Eq=${r['final_eq']:>10,.2f} ({r['roi_pct']:+7.1f}%) | "
                    f"DD={r['max_dd_pct']:6.2f}% | Calmar={r['calmar']:6.2f} | "
                    f"PF={r['pf']:.2f} | PWR={r['pwr']:4.1f}% | N={r['trades']:4d} | avgM={r['avg_eff_mult']:.2f}x"
                )

    payload = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "universe": "53 Argon B Elite Setups (15m/30m/1h)",
        "mechanisms": {
            "M1": "Argon B Static 4-Tier (1.25x / 1.00x / 0.75x / 0.50x)",
            "M2": "Legacy 6-Tier Static by PWR zones (0.25x / 0.50x / 0.75x / 1.00x / 1.25x / 1.50x)",
            "M3": "Legacy Dynamic Adaptive (0.25x..1.50x) updating per-setup vol by rolling 8-trade PWR & win/SL streaks",
            "M4": "Hybrid SL-Normalized 6-Step (0.25x..1.50x scaled by 5.0/sl_u) + per-setup cold cooling + graduated DD cooling (-3.5%)",
        },
        "setup_vol_table": setup_vols,
        "results": all_results,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] Saved {OUT_JSON}")


if __name__ == "__main__":
    main()
