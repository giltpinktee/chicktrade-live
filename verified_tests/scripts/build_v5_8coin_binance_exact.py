"""CHICKTRADE v5.1 — 8 COIN (79 Setup Tinh Hoa) · Binance USD-M Futures Exact Simulator & HTML Builder.

Incorporates ALL 7 Argon Pre-Flight Binance Realism Fixes:
  1. Hard cap MAX 5 concurrent open positions, 1 open position/symbol (One-Way Mode), <= 3 same-direction.
  2. 100% Causal Shadow-Pause (2-SL Brake): winning shadow trades only lift pause at their actual close_ts.
  3. Exact Binance timestamps: open_ts = entry bar open (i+1), close_ts = exit bar CLOSE, tp1_ts = TP1 bar CLOSE.
  4. True Binance LIMIT IOC (+/-0.25U at nxt_o): cancels immediately if nxt_o exceeds limit (no GTC dipping).
  5. Conservative SL-before-TP + Gap-Through SL fill + NO same-bar TP1+TP2 (Leg2 starts at j+1).
  6. Real 8h Binance Funding integrated at exact calc_time_ms (1.0x before tp1_ts, 0.5x strictly after tp1_ts).
  7. Real Binance 20x Margin & Notional Caps (BTC/ETH $300k, BNB/SOL/XRP/LINK $150k, NEAR $100k, ZEC $50k, <=5x eq),
     avail_eq = eq - sum(open worst_sl_usd), initial margin <= 75% avail_eq, total concurrent worst SL <= 12% eq,
     per-symbol slippage (1bp majors, 3bps ZEC/NEAR in S4; +0.10U extra slip on entry & SL in S5).

Outputs:
  - e:/CHICKTRADE/verified_tests/results/chicktrade_v5_8coin_79setups_rules.json
  - e:/CHICKTRADE/core/reports/chicktrade_new_setup_v5_8coin.html
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from verified_tests.scripts import run_gold_standard_6year_audit as gs_mod
from verified_tests.scripts.run_gold_standard_6year_audit import (
    GAMMA_TF,
    MAKER_FEE,
    MAX_HOLD_BARS,
    MIN_U_PCT,
    TAKER_FEE,
    TF_MS,
    TF_PRIORITY,
    compute_atr14,
    extract_ema_signals_6y,
    load_coin_6y,
    real_funding_u,
)
from verified_tests.scripts.build_v5_6coin_final import m2_vol_tier

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RULES_6C_JSON = os.path.join(ROOT, "verified_tests", "results", "chicktrade_v5_53setups_rules.json")
ZEC_JSON = os.path.join(ROOT, "verified_tests", "results", "avax_zec_v5_discovery.json")
NEAR_JSON = os.path.join(ROOT, "verified_tests", "results", "nearusdt_v5_discovery.json")
OUT_RULES_JSON = os.path.join(ROOT, "verified_tests", "results", "chicktrade_v5_8coin_79setups_rules.json")
OUT_HTML = os.path.join(ROOT, "core", "reports", "chicktrade_new_setup_v5_8coin.html")

SIX_COINS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "LINKUSDT", "XRPUSDT", "SOLUSDT"]
EIGHT_COINS = SIX_COINS + ["ZECUSDT", "NEARUSDT"]
TFS = ("15m", "30m", "1h")

# Argon-approved Binance 20x per-symbol notional caps & slippage
NOTIONAL_CAP_20X = {
    "BTCUSDT": 300_000.0,
    "ETHUSDT": 300_000.0,
    "BNBUSDT": 150_000.0,
    "SOLUSDT": 150_000.0,
    "XRPUSDT": 150_000.0,
    "LINKUSDT": 150_000.0,
    "NEARUSDT": 100_000.0,
    "ZECUSDT": 50_000.0,
}
SLIP_BY_SYM = {
    "BTCUSDT": 0.0001,
    "ETHUSDT": 0.0001,
    "BNBUSDT": 0.0001,
    "SOLUSDT": 0.0001,
    "XRPUSDT": 0.0001,
    "LINKUSDT": 0.0001,
    "NEARUSDT": 0.0003,
    "ZECUSDT": 0.0003,
}

# Argon Y3 Recent-Regime Survival Prune list
PRUNED_Y3 = {
    ("ZECUSDT", "15m", "EMA200_W2"),
    ("NEARUSDT", "1h", "EMA34_W4_REV"),
}


def sim_trade_binance_exact(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    ts: np.ndarray,
    fr_ts: np.ndarray,
    fr_rate: np.ndarray,
    sig: dict,
    sl_u: float,
    tp1_u: float,
    tp2_u: float,
    mode: str,
    max_bars: int,
    scenario: str,
) -> Optional[dict]:
    """Strict Binance USD-M Futures trade simulator (Argon audited, zero lookahead/optimistic bias)."""
    i = sig["idx"]
    n_bars = len(opens)
    if i + 2 >= n_bars:
        return None

    sym = sig["sym"]
    tf = sig["tf"]
    tf_ms = TF_MS[tf]
    d = sig["dir"]
    u_pct = float(sig["u_frac"])
    sig_px = float(sig["entry"])
    u_px = float(sig["u_px"])
    if u_px <= 0:
        return None

    # True Binance LIMIT IOC (+/- 0.25U) at opens[i+1]: executes immediately or cancels immediately
    nxt_o = float(opens[i + 1])
    if d == 1:
        lim_px = sig_px + 0.25 * u_px
        if nxt_o > lim_px:
            return {"ioc_missed": True}
    else:
        lim_px = sig_px - 0.25 * u_px
        if nxt_o < lim_px:
            return {"ioc_missed": True}

    fill_px = nxt_o
    open_ts = int(ts[i + 1])  # Exact entry bar open timestamp
    slip_bps = SLIP_BY_SYM.get(sym, 0.0002)
    extra_s5 = 0.10 if scenario == "S5_PESSIMISTIC" else 0.0

    entry_cost_u = ((TAKER_FEE + slip_bps) / u_pct) + extra_s5
    taker_exit_u = ((TAKER_FEE + slip_bps) / u_pct) + extra_s5
    maker_exit_u = MAKER_FEE / u_pct

    sl_px = fill_px - sl_u * u_px if d == 1 else fill_px + sl_u * u_px
    tp1_px = fill_px + tp1_u * u_px if d == 1 else fill_px - tp1_u * u_px
    tp2_px = fill_px + tp2_u * u_px if d == 1 else fill_px - tp2_u * u_px

    if mode == "PARTIAL_BE":
        l2_sl_px = fill_px
    elif mode == "PARTIAL_M1U":
        l2_sl_px = fill_px - 1.0 * u_px if d == 1 else fill_px + 1.0 * u_px
    elif mode == "PARTIAL_NO_MOVE":
        l2_sl_px = sl_px
    else:
        l2_sl_px = sl_px

    is_partial = mode.startswith("PARTIAL_")
    tp1_hit = False
    tp1_ts: Optional[int] = None
    leg1_net_u = 0.0
    leg2_net_u = 0.0
    active_sl = sl_px
    exit_idx = min(i + max_bars, n_bars - 1)
    outcome = "TIMEOUT"

    for j in range(i + 1, min(i + 1 + max_bars, n_bars)):
        bo = float(opens[j])
        bh = float(highs[j])
        bl = float(lows[j])

        if not tp1_hit:
            # Conservative: SL checked strictly before TP1; gap-through SL modeled via bo
            sl_hit = (bl <= active_sl) if d == 1 else (bh >= active_sl)
            if sl_hit:
                raw_px = min(bo, active_sl) if d == 1 else max(bo, active_sl)
                raw_u = ((raw_px - fill_px) / u_px) if d == 1 else ((fill_px - raw_px) / u_px)
                leg1_net_u = raw_u - entry_cost_u - taker_exit_u
                exit_idx = j
                outcome = "SL"
                break

            t1_hit = (bh >= tp1_px) if d == 1 else (bl <= tp1_px)
            if t1_hit:
                if not is_partial:
                    leg1_net_u = tp1_u - entry_cost_u - maker_exit_u
                    exit_idx = j
                    outcome = "TP1_ONLY"
                    break
                tp1_hit = True
                tp1_ts = int(ts[j] + tf_ms)  # Conservative: 1.0x funding until TP1 bar closes
                leg1_net_u = 0.5 * (tp1_u - entry_cost_u - maker_exit_u)
                active_sl = l2_sl_px
                # Argon fix: NEVER claim TP2 on same bar as TP1 — defer Leg2 strictly to j + 1
                continue
        else:
            # Leg 2 active on j > tp1_bar: conservative SL checked before TP2, with gap-through SL
            sl2_hit = (bl <= active_sl) if d == 1 else (bh >= active_sl)
            if sl2_hit:
                raw_px = min(bo, active_sl) if d == 1 else max(bo, active_sl)
                raw_u = ((raw_px - fill_px) / u_px) if d == 1 else ((fill_px - raw_px) / u_px)
                leg2_net_u = 0.5 * (raw_u - entry_cost_u - taker_exit_u)
                exit_idx = j
                outcome = "TP1_L2SL"
                break

            t2_hit = (bh >= tp2_px) if d == 1 else (bl <= tp2_px)
            if t2_hit:
                leg2_net_u = 0.5 * (tp2_u - entry_cost_u - maker_exit_u)
                exit_idx = j
                outcome = "TP1_TP2"
                break
    else:
        ex_px = float(closes[exit_idx])
        raw_u = ((ex_px - fill_px) / u_px) if d == 1 else ((fill_px - ex_px) / u_px)
        if not tp1_hit:
            leg1_net_u = raw_u - entry_cost_u - taker_exit_u
        else:
            leg2_net_u = 0.5 * (raw_u - entry_cost_u - taker_exit_u)
            outcome = "TP1_TO"

    close_ts = int(ts[exit_idx] + tf_ms)  # Exact exit bar CLOSE timestamp
    fund_cost_u = real_funding_u(fr_ts, fr_rate, open_ts, close_ts, d, u_pct, tp1_ts)
    pnl_u = leg1_net_u + leg2_net_u - fund_cost_u

    return {
        "sym": sym,
        "tf": tf,
        "setup": sig["setup"],
        "dir": d,
        "open_ts": open_ts,
        "close_ts": close_ts,
        "bars_held": max(1, exit_idx - i),
        "u_pct": u_pct,
        "sl_u": sl_u,
        "elevated_vol": bool(sig.get("elevated_vol", False)),
        "pnl_u": pnl_u,
        "funding_u": fund_cost_u,
        "outcome": outcome,
        "ioc_missed": False,
    }


def simulate_v5_binance_exact(
    trades: List[dict],
    ioc_missed: int = 0,
    buffer_base: float = 400.0,
    turbo_buffer: float = 250.0,
    use_turbo: bool = False,
    max_open: int = 5,
    max_same_dir: int = 3,
    init_cap: float = 10_000.0,
    use_dd_throttle: bool = True,
) -> dict:
    """Strict Binance One-Way Cross 20x Portfolio Simulator (Max 5 Open, Causal Shadow Pause)."""
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

    setup_sl_streak: Dict[Tuple[str, str, str], int] = {}
    open_pos: List[dict] = []
    closed: List[dict] = []
    curve: List[dict] = []
    total_funding = 0.0
    yearly: Dict[str, dict] = {}

    # Audit telemetry
    cool_triggers = 0
    cap5_skipped = 0
    same_dir_skipped = 0
    sym_lock_skipped = 0
    margin_skipped = 0
    risk12_skipped = 0
    shadow_skips = 0
    max_concurrent_seen = 0
    max_same_dir_seen = 0
    max_open_risk_pct = 0.0

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
                    elif turbo_active and eq <= init_cap * 1.10:
                        turbo_active = False

                sk = (p["sym"], p["tf"], p["setup"])
                if p["outcome"] == "SL":
                    setup_sl_streak[sk] = setup_sl_streak.get(sk, 0) + 1
                elif p["pnl_usd"] > 0:
                    setup_sl_streak[sk] = 0

                if p["outcome"] == "SL" and p.get("base_vol", 1.0) >= 0.75:
                    consec_sl += 1
                    if consec_sl >= 2:
                        shadow_paused = True
                        shadow_until = p["close_ts"] + 72 * 3_600_000
                        consec_sl = 0
                elif p["pnl_usd"] > 0:
                    consec_sl = 0

                closed.append(p)
                dt_str = datetime.fromtimestamp(p["close_ts"] / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
                yr = dt_str[:4]
                ys = yearly.setdefault(yr, {"trades": 0, "wins": 0, "gp": 0.0, "gl": 0.0, "pnl_usd": 0.0, "end_eq": eq})
                ys["trades"] += 1
                ys["pnl_usd"] += p["pnl_usd"]
                ys["end_eq"] = eq
                if p["pnl_usd"] > 0:
                    ys["wins"] += 1
                    ys["gp"] += p["pnl_usd"]
                else:
                    ys["gl"] += abs(p["pnl_usd"])
                curve.append({"dt": dt_str, "eq": round(eq, 2)})
            else:
                keep.append(p)
        open_pos[:] = keep

    for t in ordered:
        now = t["open_ts"]
        settle(now)

        # Causal Shadow-Pause check: pause only lifts once now >= shadow_until!
        if shadow_paused:
            if now >= shadow_until:
                shadow_paused = False
            else:
                shadow_skips += 1
                # A winning shadow trade can only lift the pause at its ACTUAL close_ts (zero lookahead!)
                if t["pnl_u"] > 0:
                    shadow_until = min(shadow_until, t["close_ts"])
                continue

        # Hard concurrency caps: MAX 5 open, 1 per symbol (One-Way Mode), <= 3 same-direction
        if len(open_pos) >= max_open:
            cap5_skipped += 1
            continue
        if any(p["sym"] == t["sym"] for p in open_pos):
            sym_lock_skipped += 1
            continue
        same_dir_cnt = sum(1 for p in open_pos if p["dir"] == t["dir"])
        if same_dir_cnt >= max_same_dir:
            same_dir_skipped += 1
            continue

        # Base M2 6-tier volume + Cold-Only Cooling (never boosts above static cap)
        base_m = float(t["vol_mult"])
        sk = (t["sym"], t["tf"], t["setup"])
        streak = setup_sl_streak.get(sk, 0)
        if streak >= 4:
            vm = min(base_m, 0.25)
            cool_triggers += 1
        elif streak == 3:
            vm = min(base_m, 0.50)
            cool_triggers += 1
        else:
            vm = base_m

        # Optional portfolio DD throttle (disabled when Pure S2 Step-Locked Volume is active)
        if use_dd_throttle:
            dd_now = (eq - peak) / peak * 100.0 if peak > 0 else 0.0
            if dd_now <= -6.0:
                vm = min(vm, 0.25)
            elif dd_now <= -3.5:
                vm = vm * 0.75

        # Elevated volatility guard
        if t.get("elevated_vol", False):
            vm = vm * 0.5

        buf = turbo_buffer if (use_turbo and turbo_active) else buffer_base
        if use_turbo and turbo_active:
            worst_5sl = 30.0 * (tier_cap / turbo_buffer)
            if eq - worst_5sl < init_cap * 1.10:
                buf = buffer_base

        u_usd = (tier_cap / buf) * vm
        notional = u_usd / max(t["u_pct"], 0.001)
        sym_cap = NOTIONAL_CAP_20X.get(t["sym"], 100_000.0)
        max_notional = min(sym_cap, eq * 5.0)
        if notional > max_notional:
            notional = max_notional
            u_usd = notional * t["u_pct"]
        if notional < 20.0:
            continue

        new_worst_sl_usd = u_usd * float(t["sl_u"])
        open_worst_sl_usd = sum(p["worst_sl_usd"] for p in open_pos)

        # Argon Fix: Concurrent worst-case SL cap <= 12% of equity (liquidation-proof at 20x)
        if (open_worst_sl_usd + new_worst_sl_usd) > eq * 0.12:
            risk12_skipped += 1
            continue

        # Argon Fix: Conservative available equity after open worst-case SL for 20x margin check
        avail_eq = eq - open_worst_sl_usd
        used_margin = sum(p["notional"] / 20.0 for p in open_pos)
        if avail_eq <= 0 or (used_margin + notional / 20.0) > avail_eq * 0.75:
            margin_skipped += 1
            continue

        open_pos.append({
            **t,
            "base_vol": base_m,
            "eff_vol": vm,
            "notional": notional,
            "worst_sl_usd": new_worst_sl_usd,
            "pnl_usd": t["pnl_u"] * u_usd,
            "funding_usd": t.get("funding_u", 0.0) * u_usd,
        })
        if len(open_pos) > max_concurrent_seen:
            max_concurrent_seen = len(open_pos)
        cur_same = sum(1 for p in open_pos if p["dir"] == t["dir"])
        if cur_same > max_same_dir_seen:
            max_same_dir_seen = cur_same
        cur_risk_pct = (open_worst_sl_usd + new_worst_sl_usd) / max(eq, 1.0) * 100.0
        if cur_risk_pct > max_open_risk_pct:
            max_open_risk_pct = cur_risk_pct

    settle(10**18)

    n = len(closed)
    wins = sum(1 for p in closed if p["pnl_usd"] > 0)
    gp = sum(p["pnl_usd"] for p in closed if p["pnl_usd"] > 0)
    gl = sum(abs(p["pnl_usd"]) for p in closed if p["pnl_usd"] <= 0)
    roi = (eq - init_cap) / init_cap * 100.0
    calmar = roi / abs(max_dd) if abs(max_dd) > 0.01 else 0.0

    yearly_list = []
    for yr in sorted(yearly.keys()):
        y = yearly[yr]
        yn = max(y["trades"], 1)
        yearly_list.append({
            "year": yr,
            "trades": y["trades"],
            "wr": round(y["wins"] / yn * 100.0, 1),
            "pwr": round(y["gp"] / max(y["gp"] + y["gl"], 1e-9) * 100.0, 1),
            "pf": round(y["gp"] / max(y["gl"], 1e-9), 2),
            "pnl_usd": round(y["pnl_usd"], 2),
            "end_eq": round(y["end_eq"], 2),
        })

    if len(curve) > 450:
        idxs = np.linspace(0, len(curve) - 1, 450, dtype=int)
        curve = [curve[k] for k in idxs]

    return {
        "final_eq": round(eq, 2),
        "roi_pct": round(roi, 1),
        "max_dd_pct": round(max_dd, 2),
        "calmar": round(calmar, 1),
        "trades": n,
        "wr": round(wins / max(n, 1) * 100.0, 1),
        "pwr": round(gp / max(gp + gl, 1e-9) * 100.0, 1),
        "pf": round(gp / max(gl, 1e-9), 2),
        "funding_usd": round(total_funding, 2),
        "ioc_missed": ioc_missed,
        "max_step": step,
        "cool_triggers": cool_triggers,
        "avg_mult": round(float(np.mean([p["eff_vol"] for p in closed])) if closed else 0.0, 2),
        "telemetry": {
            "max_concurrent_seen": max_concurrent_seen,
            "max_same_dir_seen": max_same_dir_seen,
            "max_open_risk_pct": round(max_open_risk_pct, 2),
            "cap5_skipped": cap5_skipped,
            "sym_lock_skipped": sym_lock_skipped,
            "same_dir_skipped": same_dir_skipped,
            "margin_skipped": margin_skipped,
            "risk12_skipped": risk12_skipped,
            "shadow_skips": shadow_skips,
            "ioc_missed": ioc_missed,
        },
        "yearly": yearly_list,
        "curve": curve,
    }


def load_rules_sets() -> Tuple[List[dict], List[dict], List[dict]]:
    """Returns (rules_6c_53, rules_8c_79_pruned, rules_8c_81_full)."""
    with open(RULES_6C_JSON, "r", encoding="utf-8") as f:
        r6_src = json.load(f)
    rules_6c = []
    for r in r6_src["rules"]:
        rules_6c.append({
            "sym": r["sym"],
            "tf": r["tf"],
            "setup": r["setup"],
            "vol_mult": float(r["vol_mult"]),
            "m2_tier": r["m2_tier"],
            "is_must": bool(r["is_must"]),
            "sl_u": float(r["sl_u"]),
            "tp1_u": float(r["tp1_u"]),
            "tp2_u": float(r["tp2_u"]),
            "mode": r["mode"],
            "score": float(r["score"]),
            "source": "v5_6coin",
        })

    new_2c = []
    for path, sym in ((ZEC_JSON, "ZECUSDT"), (NEAR_JSON, "NEARUSDT")):
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        for q in d["coins"][sym]["qualified_setups"]:
            vm = float(q["m2_vol"])
            new_2c.append({
                "sym": sym,
                "tf": q["tf"],
                "setup": q["setup"],
                "vol_mult": vm,
                "m2_tier": q["m2_tier"],
                "is_must": vm >= 1.25,
                "sl_u": float(q["sl_u"]),
                "tp1_u": float(q["tp1_u"]),
                "tp2_u": float(q["tp2_u"]),
                "mode": q["mode"],
                "score": float(q["score"]),
                "source": "v51_expansion",
            })

    rules_8c_81 = rules_6c + new_2c
    rules_8c_79 = rules_6c + [r for r in new_2c if (r["sym"], r["tf"], r["setup"]) not in PRUNED_Y3]
    return rules_6c, rules_8c_79, rules_8c_81


def eval_rules_binance_exact(
    rules: List[dict],
    market: Dict[str, Dict[str, pd.DataFrame]],
    funding_map: Dict[str, Tuple[np.ndarray, np.ndarray]],
    sig_map: Dict[Tuple[str, str, str], List[dict]],
    t_start: int,
    t_end: int,
) -> Tuple[List[dict], int, List[dict], int, Dict[Tuple[str, str, str], dict]]:
    """Simulate all trades for a ruleset under strict Binance-Exact S4 & S5."""
    s4_all, s5_all = [], []
    m4_tot, m5_tot = 0, 0
    per_setup_stats: Dict[Tuple[str, str, str], dict] = {}

    for r in rules:
        sym, tf, st = r["sym"], r["tf"], r["setup"]
        sigs = [s for s in sig_map.get((sym, tf, st), []) if t_start <= s["open_ts"] < t_end]
        df = market[sym][tf]
        op, hi, lo, cl, ts = df["open"].values, df["high"].values, df["low"].values, df["close"].values, df["open_time_ms"].values
        fr_ts, fr_rate = funding_map[sym]
        mb = MAX_HOLD_BARS[tf]

        setup_res = {}
        for sc_name, out_list in (("S4_REAL_FUNDING", s4_all), ("S5_PESSIMISTIC", s5_all)):
            busy = -1
            tl = []
            miss = 0
            for s in sigs:
                if s["idx"] <= busy:
                    continue
                res = sim_trade_binance_exact(op, hi, lo, cl, ts, fr_ts, fr_rate, s, r["sl_u"], r["tp1_u"], r["tp2_u"], r["mode"], mb, sc_name)
                if res is None:
                    continue
                if res.get("ioc_missed"):
                    miss += 1
                    continue
                busy = s["idx"] + res["bars_held"]
                tl.append(res)
                out_list.append({**res, "vol_mult": r["vol_mult"], "score": r["score"]})
            if sc_name == "S4_REAL_FUNDING":
                m4_tot += miss
            else:
                m5_tot += miss

            n = len(tl)
            wins = sum(1 for x in tl if x["pnl_u"] > 0)
            gp = sum(x["pnl_u"] for x in tl if x["pnl_u"] > 0)
            gl = sum(abs(x["pnl_u"]) for x in tl if x["pnl_u"] <= 0)
            net = gp - gl
            setup_res[sc_name] = {
                "n": n,
                "missed": miss,
                "wr": round(wins / max(n, 1) * 100.0, 1),
                "pwr": round(gp / max(gp + gl, 1e-9) * 100.0, 1),
                "pf": round(gp / max(gl, 1e-9), 2) if gl > 1e-9 else (9.99 if gp > 0 else 0.0),
                "net_u": round(net, 2),
                "avg_u": round(net / max(n, 1), 3),
            }
        per_setup_stats[(sym, tf, st)] = setup_res

    return s4_all, m4_tot, s5_all, m5_tot, per_setup_stats


def main():
    rules_6c, rules_79, rules_81 = load_rules_sets()
    print(f"Loaded rulesets: 6-Coin={len(rules_6c)} | 8-Coin Pruned (Argon)={len(rules_79)} | 8-Coin Full={len(rules_81)}")

    print("Loading 6-year 5m data + exact 8h Binance Funding for ALL 8 COINS...")
    market: Dict[str, Dict[str, pd.DataFrame]] = {}
    funding_map: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        futs = {sym: pool.submit(load_coin_6y, sym) for sym in EIGHT_COINS}
        for sym, fut in futs.items():
            m_d, f_ts, f_rt = fut.result()
            market[sym] = m_d
            funding_map[sym] = (f_ts, f_rt)

    print("Extracting converged EMA 34/89/200 signals (15m/30m/1h) for ALL 8 COINS...")
    gs_mod.SIX_COINS = EIGHT_COINS
    sig_map = extract_ema_signals_6y(market, tfs=TFS, max_wave=4)

    t_start = int(datetime(2023, 11, 1, tzinfo=timezone.utc).timestamp() * 1000)
    t_end = int(datetime(2026, 11, 1, tzinfo=timezone.utc).timestamp() * 1000)

    # 1. Run 6-Coin Baseline on exact same Binance-Exact engine
    s4_6, m4_6, s5_6, m5_6, _ = eval_rules_binance_exact(rules_6c, market, funding_map, sig_map, t_start, t_end)
    p6_s4_400 = simulate_v5_binance_exact(s4_6, m4_6, 400, 250, False, 5, 3)
    p6_s4_tur = simulate_v5_binance_exact(s4_6, m4_6, 400, 250, True, 5, 3)
    p6_s5_400 = simulate_v5_binance_exact(s5_6, m5_6, 400, 250, False, 5, 3)
    p6_s5_tur = simulate_v5_binance_exact(s5_6, m5_6, 400, 250, True, 5, 3)

    # 2. Run 8-Coin Official (79 setups, Argon-approved)
    s4_79, m4_79, s5_79, m5_79, stats_79 = eval_rules_binance_exact(rules_79, market, funding_map, sig_map, t_start, t_end)
    p79_s4_400 = simulate_v5_binance_exact(s4_79, m4_79, 400, 250, False, 5, 3)
    p79_s4_tur = simulate_v5_binance_exact(s4_79, m4_79, 400, 250, True, 5, 3)
    p79_s5_400 = simulate_v5_binance_exact(s5_79, m5_79, 400, 250, False, 5, 3)
    p79_s5_tur = simulate_v5_binance_exact(s5_79, m5_79, 400, 250, True, 5, 3)

    # 3. Quick A/B check on 8-Coin Full (81 setups)
    s4_81, m4_81, s5_81, m5_81, _ = eval_rules_binance_exact(rules_81, market, funding_map, sig_map, t_start, t_end)
    p81_s4_400 = simulate_v5_binance_exact(s4_81, m4_81, 400, 250, False, 5, 3)
    p81_s4_tur = simulate_v5_binance_exact(s4_81, m4_81, 400, 250, True, 5, 3)
    p81_s5_400 = simulate_v5_binance_exact(s5_81, m5_81, 400, 250, False, 5, 3)

    print("\n================ BINANCE-EXACT v5.1 HEAD-TO-HEAD (MAX 5 OPEN · CAUSAL SHADOW · TRUE IOC · GAP-SL) ================")
    for label, p4, pt, s5_4 in [
        ("6-Coin v5  (53 setups)", p6_s4_400, p6_s4_tur, p6_s5_400),
        ("8-Coin v5.1 (79 Argon)", p79_s4_400, p79_s4_tur, p79_s5_400),
        ("8-Coin Full (81 setup)", p81_s4_400, p81_s4_tur, p81_s5_400),
    ]:
        print(f"  {label}:")
        print(f"    [S4] Bao 400G : ${p4['final_eq']:>10,.2f} ({p4['roi_pct']:+7.1f}%) | DD={p4['max_dd_pct']:6.2f}% | Calmar={p4['calmar']:>5.1f} | PF={p4['pf']} | PWR={p4['pwr']}% | N={p4['trades']} | Fund=${p4['funding_usd']:,.0f}")
        print(f"    [S4] Turbo250G: ${pt['final_eq']:>10,.2f} ({pt['roi_pct']:+7.1f}%) | DD={pt['max_dd_pct']:6.2f}% | Calmar={pt['calmar']:>5.1f} | PF={pt['pf']} | PWR={pt['pwr']}% | N={pt['trades']} | Fund=${pt['funding_usd']:,.0f}")
        print(f"    [S5] Bao 400G : ${s5_4['final_eq']:>10,.2f} ({s5_4['roi_pct']:+7.1f}%) | DD={s5_4['max_dd_pct']:6.2f}% | PF={s5_4['pf']} | PWR={s5_4['pwr']}%")
    t4 = p79_s4_400["telemetry"]
    print(f"  [AUDIT TELEMETRY 8-Coin 79 S4 400G]: max_open={t4['max_concurrent_seen']}/5 | max_same_dir={t4['max_same_dir_seen']}/3 | max_open_risk={t4['max_open_risk_pct']}%/12% | cap5_skip={t4['cap5_skipped']} | sym_skip={t4['sym_lock_skipped']} | dir_skip={t4['same_dir_skipped']} | risk12_skip={t4['risk12_skipped']} | margin_skip={t4['margin_skipped']} | shadow_skip={t4['shadow_skips']} | ioc_miss={t4['ioc_missed']}")

    # Compute 1U averages per coin
    u_avg_map = {}
    for sym in EIGHT_COINS:
        df4 = market[sym]["4h"]
        df4_sub = df4[(df4["open_time_ms"] >= t_start) & (df4["open_time_ms"] < t_end)]
        macro4 = ((0.5 * compute_atr14(df4)) / df4["close"]).ewm(span=20, adjust=False).mean()
        macro4_sub = macro4.loc[df4_sub.index]
        u_avg_map[sym] = {tf: round(float(max(MIN_U_PCT, macro4_sub.mean() * GAMMA_TF[tf]) * 100.0), 3) for tf in TFS}

    # Build coin-grouped data & export rules JSON
    enriched_79 = []
    coins_payload = {}
    tier_dist: Dict[str, int] = {}

    for r in rules_79:
        st_res = stats_79[(r["sym"], r["tf"], r["setup"])]
        s4 = st_res["S4_REAL_FUNDING"]
        s5 = st_res["S5_PESSIMISTIC"]
        epr = round(s4["avg_u"] / r["sl_u"], 3) if r["sl_u"] > 0 else 0.0
        s5_res_pct = round((s5["net_u"] / s4["net_u"] * 100.0) if s4["net_u"] > 0 else 0.0, 1)
        row = {
            **r,
            "s4": s4,
            "s5": s5,
            "epr": epr,
            "s5_res_pct": s5_res_pct,
        }
        enriched_79.append(row)
        tier_dist[r["m2_tier"]] = tier_dist.get(r["m2_tier"], 0) + 1

    for sym in EIGHT_COINS:
        c_rows = sorted([x for x in enriched_79 if x["sym"] == sym], key=lambda x: (-x["vol_mult"], -x["s4"]["pwr"], -x["s4"]["net_u"]))
        coins_payload[sym] = {
            "count": len(c_rows),
            "must_count": sum(1 for x in c_rows if x["is_must"]),
            "avg_vol": round(float(np.mean([x["vol_mult"] for x in c_rows])), 2) if c_rows else 0.0,
            "tot_net_s4": round(sum(x["s4"]["net_u"] for x in c_rows), 1),
            "u_avg": u_avg_map[sym],
            "notional_cap": int(NOTIONAL_CAP_20X[sym]),
            "slip_bps": round(SLIP_BY_SYM[sym] * 10000, 1),
            "setups": c_rows,
        }

    rules_export = {
        "version": "CHICKTRADE_NEW_SETUP_v5.1_8COIN_79SETUPS_BINANCE_EXACT",
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "period": "2023-11-01 -> 2026-10-31 (36M · Warmup 2021-2023 · Binance-Exact S4 & S5)",
        "coins": EIGHT_COINS,
        "total_setups": len(enriched_79),
        "execution_rules": {
            "max_concurrent_open": 5,
            "max_per_symbol": 1,
            "max_same_direction": 3,
            "causal_shadow_pause": True,
            "true_ioc_limit_u": 0.25,
            "gap_through_sl": True,
            "no_same_bar_tp1_tp2": True,
            "max_concurrent_sl_risk_pct": 12.0,
            "notional_cap_20x": NOTIONAL_CAP_20X,
        },
        "tier_dist": tier_dist,
        "comparison": {
            "c6_53": {
                "s4_400g": {k: v for k, v in p6_s4_400.items() if k != "curve"},
                "s4_turbo": {k: v for k, v in p6_s4_tur.items() if k != "curve"},
                "s5_400g": {k: v for k, v in p6_s5_400.items() if k != "curve"},
                "s5_turbo": {k: v for k, v in p6_s5_tur.items() if k != "curve"},
            },
            "c8_79": {
                "s4_400g": {k: v for k, v in p79_s4_400.items() if k != "curve"},
                "s4_turbo": {k: v for k, v in p79_s4_tur.items() if k != "curve"},
                "s5_400g": {k: v for k, v in p79_s5_400.items() if k != "curve"},
                "s5_turbo": {k: v for k, v in p79_s5_tur.items() if k != "curve"},
            },
        },
        "setups": enriched_79,
    }
    with open(OUT_RULES_JSON, "w", encoding="utf-8") as f:
        json.dump(rules_export, f, ensure_ascii=False, indent=2)
    print(f"[OK] Saved 8-Coin locked rules JSON: {OUT_RULES_JSON}")

    html_data = {
        "generated_utc": rules_export["generated_utc"],
        "tier_dist": tier_dist,
        "portfolio": {
            "s4_400g": p79_s4_400,
            "s4_turbo": p79_s4_tur,
            "s5_400g": p79_s5_400,
            "s5_turbo": p79_s5_tur,
        },
        "baseline_6c": {
            "s4_400g": {k: v for k, v in p6_s4_400.items() if k != "curve"},
            "s4_turbo": {k: v for k, v in p6_s4_tur.items() if k != "curve"},
            "s5_400g": {k: v for k, v in p6_s5_400.items() if k != "curve"},
            "s5_turbo": {k: v for k, v in p6_s5_tur.items() if k != "curve"},
        },
        "coins": coins_payload,
    }
    build_html(html_data)


def build_html(data: dict) -> None:
    data_js = json.dumps(data, ensure_ascii=False)
    html = f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CHICKTRADE NEW SETUP v5.1 — 8 COIN (79 Setup · Binance-Exact Max 5 Lệnh)</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600;700&display=swap" rel="stylesheet">
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
  :root {{
    --bg: #F8FAFC; --card: #FFFFFF; --border: #E2E8F0; --text: #0F172A; --muted: #64748B;
    --emerald: #059669; --amber: #D97706; --sky: #0284C7; --violet: #6D28D9; --rose: #DC2626;
  }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; font-family: 'Inter', -apple-system, sans-serif; background: var(--bg); color: var(--text); font-size: 13.5px; line-height: 1.5; }}
  code {{ font-family: 'JetBrains Mono', monospace; background: #F1F5F9; padding: 1px 5px; border-radius: 4px; font-size: 12px; color: #0F172A; }}
  .hero-bar {{ background: linear-gradient(135deg, #0F172A 0%, #1E293B 60%, #064E3B 100%); color: #FFF; padding: 22px 24px 18px; border-bottom: 3px solid var(--emerald); }}
  .hero-inner {{ max-width: 1340px; margin: 0 auto; display: flex; justify-content: space-between; align-items: flex-start; gap: 16px; flex-wrap: wrap; }}
  .hero-title h1 {{ margin: 0 0 6px; font-size: 20px; font-weight: 800; letter-spacing: -0.02em; }}
  .hero-title p {{ margin: 0; font-size: 12.8px; color: #CBD5E1; }}
  .hero-title code {{ background: rgba(255,255,255,0.12); color: #A7F3D0; }}
  .hero-actions {{ display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }}
  .ver-badge {{ background: rgba(16,185,129,0.2); border: 1px solid rgba(16,185,129,0.45); color: #A7F3D0; padding: 5px 11px; border-radius: 999px; font-size: 11.5px; font-weight: 700; font-family: 'JetBrains Mono', monospace; }}
  .dict-toggle-btn {{ background: rgba(255,255,255,0.12); color: #FFF; border: 1px solid rgba(255,255,255,0.25); padding: 6px 12px; border-radius: 8px; cursor: pointer; font-weight: 600; font-size: 12px; }}
  .dict-toggle-btn:hover {{ background: rgba(255,255,255,0.22); }}
  .nav-tabs {{ background: #FFF; border-bottom: 1px solid var(--border); position: sticky; top: 0; z-index: 40; }}
  .nav-tabs-inner {{ max-width: 1340px; margin: 0 auto; display: flex; gap: 4px; padding: 8px 20px; overflow-x: auto; }}
  .tab-btn {{ border: 1px solid transparent; background: transparent; color: #475569; padding: 7px 12px; border-radius: 8px; font-weight: 600; font-size: 12.5px; cursor: pointer; white-space: nowrap; }}
  .tab-btn:hover {{ background: #F1F5F9; color: var(--text); }}
  .tab-btn.active {{ background: #ECFDF5; color: #065F46; border-color: #A7F3D0; }}
  .tab-btn.new-coin {{ border-color: #FDE68A; }}
  .page-shell {{ max-width: 1340px; margin: 0 auto; padding: 20px 20px 48px; }}
  .tab-panel {{ display: none; }}
  .tab-panel.active {{ display: block; }}
  .glossary-drawer {{ display: none; background: #FFF; border: 1px solid #CBD5E1; border-left: 4px solid var(--emerald); border-radius: 12px; padding: 16px 20px; margin-bottom: 18px; }}
  .glossary-drawer.open {{ display: block; }}
  .glossary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 14px; margin-top: 12px; }}
  .g-item {{ background: #F8FAFC; border: 1px solid var(--border); border-radius: 8px; padding: 11px 13px; font-size: 12.3px; color: #334155; }}
  .g-item strong {{ display: block; color: var(--text); margin-bottom: 4px; }}
  .filter-bar {{ display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 16px; }}
  .f-pill {{ border: 1px solid #CBD5E1; background: #FFF; color: #334155; padding: 7px 13px; border-radius: 999px; font-weight: 600; font-size: 12.3px; cursor: pointer; }}
  .f-pill.active {{ background: #0F172A; color: #FFF; border-color: #0F172A; }}
  .dual-mode-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 16px; margin-bottom: 18px; }}
  .mode-card {{ background: #FFF; border: 1px solid var(--border); border-radius: 12px; padding: 16px 18px; box-shadow: 0 1px 2px rgba(15,23,42,0.04); }}
  .mode-card h3 {{ margin: 0 0 6px; font-size: 15.5px; display: flex; justify-content: space-between; align-items: center; gap: 8px; }}
  .mode-metrics {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-top: 12px; }}
  .m-item {{ background: #F8FAFC; border: 1px solid var(--border); border-radius: 9px; padding: 10px 12px; }}
  .m-item span {{ display: block; font-size: 11.2px; color: var(--muted); font-weight: 600; }}
  .m-item strong {{ display: block; font-size: 16.5px; margin: 2px 0; font-family: 'JetBrains Mono', monospace; }}
  .m-item small {{ font-size: 11.2px; color: #475569; }}
  .section-card {{ background: #FFF; border: 1px solid var(--border); border-radius: 12px; padding: 18px 20px; margin-bottom: 18px; box-shadow: 0 1px 2px rgba(15,23,42,0.04); }}
  .section-head {{ display: flex; justify-content: space-between; align-items: flex-start; gap: 12px; flex-wrap: wrap; margin-bottom: 12px; }}
  .section-head h3 {{ margin: 0 0 3px; font-size: 15.5px; }}
  .section-sub {{ margin: 0; font-size: 12.2px; color: var(--muted); }}
  .grid-2 {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 14px; }}
  .grid-4 {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin-bottom: 16px; }}
  .grid-5 {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 10px; }}
  .audit-strip {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 10px; }}
  .tbl-wrap {{ overflow-x: auto; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 12.5px; }}
  th {{ background: #F8FAFC; color: #475569; font-weight: 700; text-align: left; padding: 9px 10px; border-bottom: 2px solid var(--border); white-space: nowrap; }}
  td {{ padding: 8px 10px; border-bottom: 1px solid #F1F5F9; white-space: nowrap; }}
  tr:hover td {{ background: #F8FAFC; }}
  .num {{ font-family: 'JetBrains Mono', monospace; }}
  .pos {{ color: var(--emerald); font-weight: 700; }}
  .neg {{ color: var(--rose); font-weight: 700; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 11px; font-weight: 700; font-family: 'JetBrains Mono', monospace; }}
  .b-v150 {{ background: #FEF3C7; color: #92400E; }}
  .b-v125 {{ background: #D1FAE5; color: #065F46; }}
  .b-v100 {{ background: #E0F2FE; color: #075985; }}
  .b-v075 {{ background: #EDE9FE; color: #5B21B6; }}
  .b-v050 {{ background: #F1F5F9; color: #475569; }}
  .must-badge {{ display: inline-block; margin-left: 6px; background: #FEF3C7; color: #B45309; border: 1px solid #FCD34D; font-size: 10px; font-weight: 800; padding: 1px 6px; border-radius: 4px; }}
  .new-tag {{ display: inline-block; margin-left: 5px; background: #DCFCE7; color: #166534; font-size: 10px; font-weight: 800; padding: 1px 6px; border-radius: 4px; }}
  .calc-controls {{ display: flex; gap: 14px; flex-wrap: wrap; margin-bottom: 14px; }}
  .calc-field {{ display: flex; flex-direction: column; gap: 4px; min-width: 180px; }}
  .calc-field label {{ font-size: 12px; font-weight: 600; color: #475569; }}
  .calc-field input, .calc-field select {{ padding: 8px 10px; border: 1px solid #CBD5E1; border-radius: 8px; font-family: 'JetBrains Mono', monospace; font-size: 13px; }}
  .mech-card {{ background: #F8FAFC; border: 1px solid var(--border); border-top: 3px solid var(--sky); border-radius: 10px; padding: 14px 16px; }}
  .mech-card h4 {{ margin: 0 0 6px; font-size: 14px; }}
  .mech-card p {{ margin: 0; font-size: 12.5px; color: #475569; line-height: 1.55; }}
</style>
</head>
<body>

<header class="hero-bar">
  <div class="hero-inner">
    <div class="hero-title">
      <h1>🏆 CHICKTRADE NEW SETUP v5.1 — 8 COIN (79 Setup Tinh Hoa · Binance-Exact Max 5 Lệnh)</h1>
      <p>6 Coin gốc (<code>53</code>) + <b>ZEC</b> (<code>11</code>) + <b>NEAR</b> (<code>15</code>) · Kiểm định 36 tháng (<code>2023-11 → 2026-10</code>) · Chuẩn <b>Binance USD-M Futures thật</b> đã qua Argon kiểm toán (0 nhìn trước tương lai)</p>
    </div>
    <div class="hero-actions">
      <button type="button" class="dict-toggle-btn" onclick="toggleGlossary()">📖 Tiêu Chí Binance-Exact</button>
      <span class="ver-badge">v5.1 · 8 COINS · 79 SETUPS · <span id="gen-utc">--</span></span>
    </div>
  </div>
</header>

<nav class="nav-tabs">
  <div class="nav-tabs-inner">
    <button class="tab-btn active" onclick="switchTab('tab-overview', this)">🌐 Tổng Quan (8 Coin)</button>
    <button class="tab-btn" onclick="switchTab('BTCUSDT', this)">🟠 BTC</button>
    <button class="tab-btn" onclick="switchTab('ETHUSDT', this)">🔵 ETH</button>
    <button class="tab-btn" onclick="switchTab('BNBUSDT', this)">🟡 BNB</button>
    <button class="tab-btn" onclick="switchTab('LINKUSDT', this)">🔷 LINK</button>
    <button class="tab-btn" onclick="switchTab('XRPUSDT', this)">⚪ XRP</button>
    <button class="tab-btn" onclick="switchTab('SOLUSDT', this)">🟣 SOL</button>
    <button class="tab-btn new-coin" onclick="switchTab('ZECUSDT', this)">🛡️ ZEC (Mới)</button>
    <button class="tab-btn new-coin" onclick="switchTab('NEARUSDT', this)">⚡ NEAR (Mới)</button>
    <button class="tab-btn" onclick="switchTab('tab-mech', this)">🧮 Cơ Chế &amp; Máy Tính S2</button>
  </div>
</nav>

<main class="page-shell">

  <section id="glossary-drawer" class="glossary-drawer">
    <div style="display:flex;justify-content:space-between;align-items:center;">
      <h3 style="margin:0;font-size:15px;">🛡️ 6 Tiêu Chí Khớp Lệnh Thật Trên Binance USD-M Futures (Argon Đã Kiểm Toán)</h3>
      <button type="button" class="dict-toggle-btn" style="color:#0F172A;border-color:#CBD5E1;" onclick="toggleGlossary()">✕ Đóng</button>
    </div>
    <div class="glossary-grid">
      <div class="g-item"><strong>1. Khóa Cứng Tối Đa 5 Lệnh Mở Đồng Thời</strong>Tối đa <b>5 lệnh toàn quỹ</b>, tối đa <b>1 lệnh/Coin</b> (One-Way Mode), tối đa <b>3 lệnh cùng chiều</b> LONG hoặc SHORT. Tổng rủi ro SL đồng thời của 5 lệnh không vượt quá <b>12% vốn</b>.</div>
      <div class="g-item"><strong>2. Phanh 2-SL Nhân Quả Tuyệt Đối (0 Lookahead)</strong>Dính 2 SL gốc (nhóm ≥0.75×) → tạm dừng đánh nháp tối đa 72h. Lệnh nháp thắng chỉ được mở khóa phanh đúng tại thời điểm <code>close_ts</code> (đóng nến thoát lệnh), tuyệt đối không mở sớm.</div>
      <div class="g-item"><strong>3. LIMIT IOC ±0.25U Thật &amp; Gap-Through SL</strong>Vào lệnh đúng giá mở nến <code>i+1</code>; nếu giá mở vượt biên <code>±0.25U</code> → hủy ngay lập tức (không chờ rút râu). SL bị nhảy gap tính đúng giá gap xấu hơn. Chặn ăn kép TP1+TP2 trong cùng 1 nến.</div>
      <div class="g-item"><strong>4. Funding 8h Lịch Sử Thật &amp; Margin 20× Sàn</strong>Trừ/cộng funding thật từng mốc 8h từ <code>open_ts</code> (mở nến vào) tới <code>close_ts</code> (đóng nến thoát), giảm 50% sau khi nến TP1 đóng. Giới hạn notional 20× từng mã (BTC/ETH $300k · NEAR $100k · ZEC $50k).</div>
    </div>
  </section>

  <!-- TAB OVERVIEW -->
  <section id="tab-overview" class="tab-panel active">
    <div class="filter-bar" id="sc-bar">
      <button class="f-pill active" onclick="setSc('s4', this)">⚡ S4 Thực Chiến Chuẩn Binance (Taker 0.05% · Slip 1–3bps · IOC thật · Funding 8h thật)</button>
      <button class="f-pill" onclick="setSc('s5', this)">🛡️ S5 Kiểm Chứng Chịu Đòn (Cộng thêm trượt xấu +0.10U cả vào &amp; SL)</button>
    </div>

    <div class="dual-mode-grid" id="dual-cards"></div>

    <div class="section-card">
      <div class="section-head">
        <div>
          <h3>🔍 Chứng Minh Khớp Lệnh Thật Binance (Argon Telemetry · Khóa Cứng ≤ 5 Lệnh)</h3>
          <p class="section-sub">Các chỉ số đo thực tế trong suốt 36 tháng mô phỏng (Bao 400G) — chứng minh không vượt trần lệnh, không vượt trần rủi ro 12%, không nhìn trước tương lai</p>
        </div>
      </div>
      <div class="audit-strip" id="telemetry-strip"></div>
    </div>

    <div class="section-card">
      <div class="section-head">
        <div>
          <h3>⚖️ Đối Chiếu Trực Tiếp: 6 Coin Gốc (53 Setup) vs 8 Coin Mới (79 Setup · Thêm ZEC + NEAR)</h3>
          <p class="section-sub">Chạy trên <b>cùng 1 động cơ Binance-Exact</b> (Max 5 lệnh, Phanh 2-SL nhân quả, IOC thật, Gap-SL, Margin 20× theo từng mã)</p>
        </div>
      </div>
      <div class="tbl-wrap">
        <table>
          <thead>
            <tr>
              <th>Phiên Bản</th><th>Chế Độ</th><th>Vốn Cuối ($10k →)</th><th>ROI 36M</th><th>MaxDD</th><th>Calmar</th><th>Số Lệnh</th><th>WR</th><th>PWR</th><th>PF</th><th>Funding Thật</th>
            </tr>
          </thead>
          <tbody id="compare-tbody"></tbody>
        </table>
      </div>
    </div>

    <div class="section-card">
      <div class="section-head">
        <div>
          <h3>📈 Đường Cong Tài Sản 36 Tháng — 8 COIN (2023–2026 · Bao 400G vs Auto-Turbo 250G)</h3>
          <p class="section-sub">Vốn $10,000 · 20× Cross · Max 5 lệnh mở đồng thời · Lãi kép S2 (+8%/-10%) · Van Hạ Lạnh Cold-Only</p>
        </div>
      </div>
      <div id="eq-chart" style="width:100%;height:370px;"></div>
    </div>

    <div class="section-card">
      <div class="section-head">
        <div>
          <h3>🎚️ Phân Bổ 6 Nấc Volume M2 Trên 79 Setup (8 Coin)</h3>
          <p class="section-sub">Dồn lực vào siêu phẩm (PWR ≥ 65%) và hạ gọn các setup biên vừa về 0.50×–0.75×</p>
        </div>
      </div>
      <div class="grid-5" id="vol-dist-grid"></div>
    </div>

    <div class="section-card">
      <div class="section-head">
        <div>
          <h3>📅 Hiệu Suất Từng Năm (2023 → 2026 · 8 Coin · Bao 400G vs Auto-Turbo 250G)</h3>
          <p class="section-sub">Dương bền vững xuyên suốt mọi năm dưới chuẩn khớp lệnh thật Binance</p>
        </div>
      </div>
      <div class="tbl-wrap">
        <table>
          <thead>
            <tr>
              <th>Năm</th><th>Số Lệnh</th><th>WR Lệnh</th><th>Thắng Tiền (PWR)</th><th>PF</th>
              <th>Lãi/Lỗ Bao 400G</th><th>Vốn Cuối 400G</th><th>Lãi/Lỗ Turbo 250G</th><th>Vốn Cuối Turbo</th>
            </tr>
          </thead>
          <tbody id="yearly-tbody"></tbody>
        </table>
      </div>
    </div>

    <div class="section-card">
      <div class="section-head">
        <div>
          <h3>🪙 Tổng Hợp 8 Coin Trong Bộ CHICKTRADE v5.1 (79 Setup)</h3>
          <p class="section-sub">Bấm vào từng dòng hoặc tab phía trên để xem đầy đủ từng setup, điểm ngọt SL/TP1/TP2 và nấc Volume M2</p>
        </div>
      </div>
      <div class="tbl-wrap">
        <table>
          <thead>
            <tr><th>Coin</th><th>Số Setup</th><th>THẤY LÀ QUẤT (≥1.25×)</th><th>Vol TB</th><th>Tổng Net S4 (U)</th><th>Trần 20×</th><th>Slip S4</th><th>1U M15</th><th>1U M30</th><th>1U 1H</th></tr>
          </thead>
          <tbody id="coin-sum-tbody"></tbody>
        </table>
      </div>
    </div>
  </section>

  <!-- COIN TAB -->
  <section id="tab-coin" class="tab-panel">
    <div class="grid-4" id="coin-kpis"></div>
    <div class="section-card">
      <div class="section-head">
        <div>
          <h3 id="coin-tbl-title">📋 Danh Sách Setup v5.1 Đã Chốt</h3>
          <p class="section-sub">Thông số chạy lại trên động cơ Binance-Exact (IOC thật, Gap-SL, không ăn kép cùng nến, Funding 8h thật)</p>
        </div>
        <div class="filter-bar" style="margin-bottom:0;">
          <button class="f-pill c-tf active" onclick="setCoinTf('ALL', this)">Tất cả (15m/30m/1h)</button>
          <button class="f-pill c-tf" onclick="setCoinTf('1h', this)">1H</button>
          <button class="f-pill c-tf" onclick="setCoinTf('30m', this)">30m</button>
          <button class="f-pill c-tf" onclick="setCoinTf('15m', this)">15m</button>
        </div>
      </div>
      <div class="tbl-wrap">
        <table>
          <thead>
            <tr>
              <th>Khung</th><th>Setup</th><th>Mốc Volume M2</th><th>Điểm Ngọt (SL / TP1 / TP2 · Kiểu)</th>
              <th>N</th><th>WR</th><th>PWR (S4)</th><th>PF</th><th>TB/Lệnh</th><th>Edge/Risk</th><th>Bền S5%</th><th>Net S4</th><th>Net S5</th>
            </tr>
          </thead>
          <tbody id="coin-tbody"></tbody>
        </table>
      </div>
    </div>
  </section>

  <!-- MECHANISM & S2 TAB -->
  <section id="tab-mech" class="tab-panel">
    <div class="section-card">
      <div class="section-head">
        <div>
          <h3>🧮 Máy Tính Bậc Lãi Kép S2 (Bậc 0 → 200 · 6 Nấc Volume M2)</h3>
          <p class="section-sub">Tính chính xác tiền 1U và rủi ro SL cho từng bậc vốn và từng nấc Volume (0.25× → 1.50×)</p>
        </div>
      </div>
      <div class="calc-controls">
        <div class="calc-field"><label>Vốn gốc ($)</label><input type="number" id="s2-base" value="10000" min="500" step="1000" oninput="renderS2()"></div>
        <div class="calc-field"><label>Bậc S2 (0 → 200)</label><input type="number" id="s2-step" value="0" min="0" max="200" step="1" oninput="renderS2()"></div>
        <div class="calc-field"><label>Nấc Volume M2</label>
          <select id="s2-vol" onchange="renderS2()">
            <option value="1.5">1.50× (S++ · PWR ≥ 70%)</option>
            <option value="1.25">1.25× (S+ · PWR ≥ 65%)</option>
            <option value="1.0" selected>1.00× (A · PWR ≥ 61%)</option>
            <option value="0.75">0.75× (B · PWR ≥ 58%)</option>
            <option value="0.5">0.50× (C · PWR ≥ 55%)</option>
            <option value="0.25">0.25× (D / Hạ lạnh)</option>
          </select>
        </div>
      </div>
      <div class="grid-4" id="s2-out"></div>
    </div>
    <div class="section-card">
      <div class="section-head"><div><h3>🛡️ Quy Tắc Vận Hành Khóa Cứng Binance-Exact v5.1</h3></div></div>
      <div class="grid-2">
        <div class="mech-card" style="border-top-color:#059669;"><h4>1. Khóa Cứng Max 5 Lệnh Mở &amp; 12% Tổng Rủi Ro</h4><p>Trên toàn bộ 8 coin, tại mọi thời điểm chỉ giữ tối đa <b>5 vị thế mở</b>, tối đa <b>1 vị thế/coin</b> (Binance One-Way Mode), tối đa <b>3 vị thế cùng chiều</b>. Tổng rủi ro SL đồng thời của toàn bộ vị thế đang mở bị khóa trần ≤ <b>12% vốn</b>.</p></div>
        <div class="mech-card" style="border-top-color:#D97706;"><h4>2. Thang 6 Nấc M2 + Van Chỉ Hạ Lạnh (Cold-Only)</h4><p>Mỗi setup có trần Volume tĩnh cố định theo Profit WR (<code>0.50× → 1.50×</code>). Dính 3 SL liên tiếp tự hạ về <code>0.50×</code>, 4 SL về <code>0.25×</code> tới khi thắng lại; không bao giờ đẩy nóng vượt trần.</p></div>
        <div class="mech-card" style="border-top-color:#0284C7;"><h4>3. Auto-Turbo 250G (+25% Bật · Sàn 110%)</h4><p>Mặc định chạy <b>Bao 400G</b>. Chỉ kích hoạt <b>Auto-Turbo 250G</b> khi vốn ≥ <code>$12,500 (+25%)</code> và kiểm tra trước mỗi lệnh sao cho 5 lệnh SL cùng lúc vẫn giữ ≥ <code>$11,000 (110%)</code>.</p></div>
        <div class="mech-card" style="border-top-color:#6D28D9;"><h4>4. Phanh 2-SL Nhân Quả &amp; Giới Hạn Notional 20×</h4><p>Dính 2 SL gốc liên tiếp (nhóm ≥0.75×) bật phanh đánh nháp tối đa 72h (chỉ mở khóa khi lệnh nháp đóng cửa có lãi). Áp đúng trần notional 20× sàn Binance cho từng mã (ZEC $50k, NEAR $100k, Alt $150k, BTC/ETH $300k).</p></div>
      </div>
    </div>
  </section>

</main>

<script>
const D = {data_js};
let scMode = 's4';
let activeCoin = 'BTCUSDT';
let coinTf = 'ALL';

document.getElementById('gen-utc').textContent = D.generated_utc;
function fmtUsd(v) {{ return '$' + Number(v || 0).toLocaleString('en-US', {{ minimumFractionDigits: 2, maximumFractionDigits: 2 }}); }}
function fmtSign(v, sfx='') {{ const n = Number(v || 0); return (n >= 0 ? '+' : '') + n.toFixed(1) + sfx; }}
function toggleGlossary() {{ document.getElementById('glossary-drawer').classList.toggle('open'); }}

function switchTab(id, btn) {{
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  if (btn) btn.classList.add('active');
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  if (id === 'tab-overview' || id === 'tab-mech') {{
    document.getElementById(id).classList.add('active');
    if (id === 'tab-overview') setTimeout(() => {{ try {{ Plotly.Plots.resize('eq-chart'); }} catch (e) {{}} }}, 40);
  }} else {{
    activeCoin = id;
    document.getElementById('tab-coin').classList.add('active');
    renderCoinPanel(id);
  }}
}}

function jumpToCoin(sym) {{
  const target = Array.from(document.querySelectorAll('.tab-btn')).find(b => b.textContent.includes(sym.replace('USDT', '')));
  switchTab(sym, target);
  window.scrollTo({{ top: 0, behavior: 'smooth' }});
}}

function setSc(m, btn) {{
  scMode = m;
  document.querySelectorAll('#sc-bar .f-pill').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  renderDualChartYearly();
}}

function volBadgeClass(v) {{
  if (v >= 1.5) return 'b-v150';
  if (v >= 1.25) return 'b-v125';
  if (v >= 1.0) return 'b-v100';
  if (v >= 0.75) return 'b-v075';
  return 'b-v050';
}}

function renderDualChartYearly() {{
  const p4 = scMode === 's4' ? D.portfolio.s4_400g : D.portfolio.s5_400g;
  const pt = scMode === 's4' ? D.portfolio.s4_turbo : D.portfolio.s5_turbo;
  const b4 = scMode === 's4' ? D.baseline_6c.s4_400g : D.baseline_6c.s5_400g;
  const bt = scMode === 's4' ? D.baseline_6c.s4_turbo : D.baseline_6c.s5_turbo;
  const lbl = scMode === 's4' ? 'S4 Binance-Exact' : 'S5 Trượt Xấu Nhất';

  document.getElementById('dual-cards').innerHTML = `
    <div class="mode-card" style="border-top:4px solid #059669;">
      <h3><span>🛡️ Bao 400G · ${{lbl}}</span><span class="badge b-v125">An Toàn · Max 5 Lệnh</span></h3>
      <div style="font-size:12.3px;color:#475569;">79 Setup (8 Coin) · Hạ lạnh ${{p4.cool_triggers}} lần · Funding thật $${{Math.round(p4.funding_usd).toLocaleString()}} · Max mở ${{p4.telemetry.max_concurrent_seen}}/5</div>
      <div class="mode-metrics">
        <div class="m-item"><span>Vốn Cuối 36M ($10k →)</span><strong class="pos">${{fmtUsd(p4.final_eq)}}</strong><small>${{fmtSign(p4.roi_pct, '%')}} · Bậc ${{p4.max_step}}</small></div>
        <div class="m-item"><span>MaxDD / Calmar</span><strong class="neg">${{p4.max_dd_pct}}%</strong><small>Calmar <b>${{p4.calmar}}</b> · ${{p4.trades}} lệnh</small></div>
        <div class="m-item"><span>WR / Thắng Tiền (PWR)</span><strong>${{p4.wr}}% / <span class="pos">${{p4.pwr}}%</span></strong><small>PF <b>${{p4.pf}}</b> · Vol TB ${{p4.avg_mult}}×</small></div>
      </div>
    </div>
    <div class="mode-card" style="border-top:4px solid #D97706;">
      <h3><span>⚡ Auto-Turbo 250G · ${{lbl}}</span><span class="badge b-v150">Bật +25% · Sàn 110%</span></h3>
      <div style="font-size:12.3px;color:#475569;">Bật từ $12,500 · Max mở ${{pt.telemetry.max_concurrent_seen}}/5 · Rủi ro mở đỉnh ${{pt.telemetry.max_open_risk_pct}}% · Funding $${{Math.round(pt.funding_usd).toLocaleString()}}</div>
      <div class="mode-metrics">
        <div class="m-item"><span>Vốn Cuối 36M ($10k →)</span><strong style="color:#B45309;">${{fmtUsd(pt.final_eq)}}</strong><small>${{fmtSign(pt.roi_pct, '%')}} · Bậc ${{pt.max_step}}</small></div>
        <div class="m-item"><span>MaxDD / Calmar</span><strong class="neg">${{pt.max_dd_pct}}%</strong><small>Calmar <b>${{pt.calmar}}</b> · ${{pt.trades}} lệnh</small></div>
        <div class="m-item"><span>WR / Thắng Tiền (PWR)</span><strong>${{pt.wr}}% / <span class="pos">${{pt.pwr}}%</span></strong><small>PF <b>${{pt.pf}}</b> · Vol TB ${{pt.avg_mult}}×</small></div>
      </div>
    </div>
  `;

  const tel = p4.telemetry;
  document.getElementById('telemetry-strip').innerHTML = `
    <div class="m-item" style="border-top:3px solid #059669;"><span>Lệnh Mở Đồng Thời Đỉnh</span><strong class="pos">${{tel.max_concurrent_seen}} / 5</strong><small>Chặn vượt trần 5: ${{tel.cap5_skipped}} kèo</small></div>
    <div class="m-item" style="border-top:3px solid #0284C7;"><span>Cùng Chiều Tối Đa</span><strong>${{tel.max_same_dir_seen}} / 3</strong><small>Chặn cùng chiều: ${{tel.same_dir_skipped}} · Trùng coin: ${{tel.sym_lock_skipped}}</small></div>
    <div class="m-item" style="border-top:3px solid #D97706;"><span>Rủi Ro SL Mở Đỉnh</span><strong>${{tel.max_open_risk_pct}}% / 12%</strong><small>Chặn rủi ro >12%: ${{tel.risk12_skipped}} · Margin: ${{tel.margin_skipped}}</small></div>
    <div class="m-item" style="border-top:3px solid #6D28D9;"><span>Hủy IOC &amp; Phanh 2-SL</span><strong>${{tel.ioc_missed}} / ${{tel.shadow_skips}}</strong><small>Hủy IOC ±0.25U / Nghỉ phanh nhân quả</small></div>
  `;

  const cmpRows = [
    ['6 Coin Gốc (53 setup)', 'Bao 400G', b4],
    ['8 Coin Mới (79 setup · +ZEC +NEAR)', 'Bao 400G', p4],
    ['6 Coin Gốc (53 setup)', 'Auto-Turbo 250G', bt],
    ['8 Coin Mới (79 setup · +ZEC +NEAR)', 'Auto-Turbo 250G', pt],
  ];
  document.getElementById('compare-tbody').innerHTML = cmpRows.map(([name, mode, x]) => {{
    const hi = name.startsWith('8 Coin') ? 'style="background:#F0FDF4;font-weight:700;"' : '';
    return `
      <tr ${{hi}}>
        <td>${{name}}</td>
        <td>${{mode}}</td>
        <td class="num pos">${{fmtUsd(x.final_eq)}}</td>
        <td class="num pos">${{fmtSign(x.roi_pct, '%')}}</td>
        <td class="num neg">${{x.max_dd_pct}}%</td>
        <td class="num"><b>${{x.calmar}}</b></td>
        <td class="num">${{x.trades}}</td>
        <td class="num">${{x.wr}}%</td>
        <td class="num pos">${{x.pwr}}%</td>
        <td class="num"><b>${{x.pf}}</b></td>
        <td class="num">$${{Math.round(x.funding_usd).toLocaleString()}}</td>
      </tr>
    `;
  }}).join('');

  Plotly.newPlot('eq-chart', [
    {{ x: p4.curve.map(c=>c.dt), y: p4.curve.map(c=>c.eq), name: `8 Coin · Bao 400G (${{lbl}})`, type: 'scatter', mode: 'lines', fill: 'tozeroy', fillcolor: 'rgba(5,150,105,0.08)', line: {{ color: '#059669', width: 2.6 }} }},
    {{ x: pt.curve.map(c=>c.dt), y: pt.curve.map(c=>c.eq), name: `8 Coin · Auto-Turbo 250G (${{lbl}})`, type: 'scatter', mode: 'lines', line: {{ color: '#D97706', width: 2.3, dash: 'dot' }} }}
  ], {{
    paper_bgcolor: '#FFFFFF', plot_bgcolor: '#F8FAFC', margin: {{ t: 20, r: 20, l: 76, b: 36 }},
    font: {{ family: 'Inter, sans-serif', size: 11.5, color: '#475569' }},
    xaxis: {{ gridcolor: '#E2E8F0' }}, yaxis: {{ gridcolor: '#E2E8F0', tickprefix: '$', type: 'log' }},
    legend: {{ orientation: 'h', y: 1.12 }}
  }}, {{ responsive: true, displayModeBar: false }});

  const turMap = {{}};
  for (const r of pt.yearly) turMap[r.year] = r;
  document.getElementById('yearly-tbody').innerHTML = p4.yearly.map(r => {{
    const t = turMap[r.year] || {{ pnl_usd: 0, end_eq: 0 }};
    return `
      <tr>
        <td class="num"><b>${{r.year}}</b></td>
        <td class="num">${{r.trades}}</td>
        <td class="num">${{r.wr}}%</td>
        <td class="num"><b class="${{r.pwr>=55?'pos':''}}">${{r.pwr}}%</b></td>
        <td class="num"><b>${{r.pf}}</b></td>
        <td class="num ${{r.pnl_usd>=0?'pos':'neg'}}">${{r.pnl_usd>=0?'+':''}}${{fmtUsd(r.pnl_usd)}}</td>
        <td class="num"><b>${{fmtUsd(r.end_eq)}}</b></td>
        <td class="num ${{t.pnl_usd>=0?'pos':'neg'}}">${{t.pnl_usd>=0?'+':''}}${{fmtUsd(t.pnl_usd)}}</td>
        <td class="num"><b>${{fmtUsd(t.end_eq)}}</b></td>
      </tr>
    `;
  }}).join('');
}}

function renderOverview() {{
  renderDualChartYearly();
  const tiers = [
    ['S++ (1.50×)', 'PWR ≥ 70%', '#D97706'],
    ['S+ (1.25×)', 'PWR 65–70%', '#059669'],
    ['A (1.00×)', 'PWR 61–65%', '#0284C7'],
    ['B (0.75×)', 'PWR 58–61%', '#6D28D9'],
    ['C (0.50×)', 'PWR 55–58%', '#64748B'],
  ];
  document.getElementById('vol-dist-grid').innerHTML = tiers.map(([k, desc, col]) => `
    <div class="m-item" style="background:#FFFFFF;border-top:3px solid ${{col}};">
      <span>${{k}}</span>
      <strong style="color:${{col}};">${{D.tier_dist[k] || 0}} setup</strong>
      <small>${{desc}}</small>
    </div>
  `).join('');

  const syms = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'LINKUSDT', 'XRPUSDT', 'SOLUSDT', 'ZECUSDT', 'NEARUSDT'];
  document.getElementById('coin-sum-tbody').innerHTML = syms.map(sym => {{
    const c = D.coins[sym];
    const isNew = (sym === 'ZECUSDT' || sym === 'NEARUSDT') ? '<span class="new-tag">MỚI</span>' : '';
    return `
      <tr style="cursor:pointer;" onclick="jumpToCoin('${{sym}}')">
        <td class="num"><b>${{sym}}</b>${{isNew}}</td>
        <td class="num"><b>${{c.count}}</b></td>
        <td class="num" style="color:#D97706;font-weight:700;">${{c.must_count}}</td>
        <td class="num"><b>${{c.avg_vol}}×</b></td>
        <td class="num pos">+${{c.tot_net_s4}}U</td>
        <td class="num">$${{(c.notional_cap/1000).toFixed(0)}}k</td>
        <td class="num">${{c.slip_bps}} bps</td>
        <td class="num">${{c.u_avg['15m']}}%</td>
        <td class="num">${{c.u_avg['30m']}}%</td>
        <td class="num">${{c.u_avg['1h']}}%</td>
      </tr>
    `;
  }}).join('');
}}

function setCoinTf(tf, btn) {{
  coinTf = tf;
  document.querySelectorAll('.c-tf').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  renderCoinTable(activeCoin);
}}

function renderCoinPanel(sym) {{
  const c = D.coins[sym];
  document.getElementById('coin-tbl-title').textContent = `📋 ${{sym}} — ${{c.count}} Setup Tinh Hoa Đã Chốt (v5.1 Binance-Exact)`;
  document.getElementById('coin-kpis').innerHTML = `
    <div class="m-item" style="background:#FFFFFF;border-top:3px solid #059669;"><span>Số Setup (${{sym}})</span><strong class="pos">${{c.count}} setup</strong><small>${{c.must_count}} kèo THẤY LÀ QUẤT · Trần 20× $${{(c.notional_cap/1000).toFixed(0)}}k</small></div>
    <div class="m-item" style="background:#FFFFFF;border-top:3px solid #D97706;"><span>Hệ Số Volume TB</span><strong>${{c.avg_vol}}×</strong><small>Theo thang 6 nấc M2 · Slip ${{c.slip_bps}} bps</small></div>
    <div class="m-item" style="background:#FFFFFF;border-top:3px solid #0284C7;"><span>Tổng Net S4 (36M)</span><strong class="pos">+${{c.tot_net_s4}}U</strong><small>Đã trừ phí Taker + IOC thật + Funding 8h</small></div>
    <div class="m-item" style="background:#FFFFFF;border-top:3px solid #6D28D9;"><span>1U TB (15m / 30m / 1h)</span><strong style="font-size:13.5px;">${{c.u_avg['15m']}}% / ${{c.u_avg['30m']}}% / ${{c.u_avg['1h']}}%</strong><small>Thước đo 4H Macro ATR</small></div>
  `;
  renderCoinTable(sym);
}}

function renderCoinTable(sym) {{
  const c = D.coins[sym];
  const rows = c.setups.filter(r => coinTf === 'ALL' || r.tf === coinTf);
  document.getElementById('coin-tbody').innerHTML = rows.map(r => {{
    const s4 = r.s4, s5 = r.s5;
    const must = r.is_must ? '<span class="must-badge">THẤY LÀ QUẤT</span>' : '';
    return `
      <tr>
        <td class="num"><b>${{r.tf.toUpperCase()}}</b></td>
        <td class="num"><b>${{r.setup}}</b>${{must}}</td>
        <td><span class="badge ${{volBadgeClass(r.vol_mult)}}">${{r.m2_tier}}</span></td>
        <td class="num">-${{r.sl_u}} / +${{r.tp1_u}} / +${{r.tp2_u}}U · ${{r.mode}}</td>
        <td class="num">${{s4.n}}</td>
        <td class="num">${{s4.wr}}%</td>
        <td class="num pos">${{s4.pwr}}%</td>
        <td class="num"><b>${{s4.pf}}</b></td>
        <td class="num pos">+${{Number(s4.avg_u).toFixed(2)}}U</td>
        <td class="num"><b>${{r.epr}}</b></td>
        <td class="num pos">${{r.s5_res_pct}}%</td>
        <td class="num pos">+${{s4.net_u}}U</td>
        <td class="num pos">+${{s5.net_u}}U</td>
      </tr>
    `;
  }}).join('');
}}

function renderS2() {{
  const base = Math.max(500, Number(document.getElementById('s2-base').value || 10000));
  const step = Math.max(0, Math.min(200, Math.floor(Number(document.getElementById('s2-step').value || 0))));
  const vol = Number(document.getElementById('s2-vol').value || 1.0);
  const tierCap = base * Math.pow(1.08, step);
  const nextCap = base * Math.pow(1.08, step + 1);
  const u400 = (tierCap * vol) / 400;
  const u250 = (tierCap * vol) / 250;
  document.getElementById('s2-out').innerHTML = `
    <div class="m-item"><span>Mốc Vốn Bậc ${{step}}</span><strong class="pos">${{fmtUsd(tierCap)}}</strong><small>Lên Bậc ${{step+1}}: ${{fmtUsd(nextCap)}}</small></div>
    <div class="m-item"><span>1U ở ${{vol.toFixed(2)}}× (400G / 250G)</span><strong>${{fmtUsd(u400)}} / ${{fmtUsd(u250)}}</strong><small>Khóa cố định suốt Bậc ${{step}}</small></div>
    <div class="m-item"><span>Rủi ro SL 5U (400G / 250G)</span><strong class="neg">-${{fmtUsd(u400*5)}} / -${{fmtUsd(u250*5)}}</strong><small>${{(5*vol/4).toFixed(2)}}% (400G) · ${{(5*vol/2.5).toFixed(2)}}% (250G)</small></div>
    <div class="m-item"><span>Rủi ro SL 6U (400G / 250G)</span><strong class="neg">-${{fmtUsd(u400*6)}} / -${{fmtUsd(u250*6)}}</strong><small>${{(6*vol/4).toFixed(2)}}% (400G) · ${{(6*vol/2.5).toFixed(2)}}% (250G)</small></div>
  `;
}}

renderOverview();
renderS2();
</script>
</body>
</html>
"""
    os.makedirs(os.path.dirname(OUT_HTML), exist_ok=True)
    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[OK] Built NEW SETUP v5.1 — 8 COIN HTML: {OUT_HTML} ({len(html):,} bytes)")


if __name__ == "__main__":
    main()
