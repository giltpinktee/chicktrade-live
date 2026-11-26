"""CHICKTRADE v6.0 — Master 7-Coin Binance-Exact Engine, Rules Sync & Full Interactive HTML Builder.

Self-contained v6.0 release builder:
  - Universe: 7 Coins (BTC, ETH, BNB, LINK, XRP, SOL, NEAR) · 68 ACTIVE + 1 BENCH_WATCH
  - Volume: V3 Barbell Alpha (0.25x -> 1.75x by PWR + Edge/Risk EPR) + Cold-Only Cooling
  - Execution: Argon-audited Binance USD-M Futures Exact (Max 5 Open, 1/symbol One-Way Mode,
    <=3 Same-Direction, 100% Causal Shadow-Pause, True LIMIT IOC +/-0.25U, Gap-Through SL,
    No Same-Bar TP1+TP2, Real 8h Binance Funding, Per-Symbol 20x Notional Caps & 12% Risk Cap)
  - Compounding: Pure S2 Unlimited Stepped Compounding (Locked Volume inside each step)
    · Mode 1: Bao 400G (Safe)
    · Mode 2: Auto-Turbo 250G (ON >= +25%, Armor Floor 110% -> revert 400G)
  - Lifecycle: Slow Bayesian Online Learning + Shadow Bench-Watch (`core/live_learning.py`)

Outputs:
  - e:/CHICKTRADE/verified_tests/results/chicktrade_v6_7coin_rules.json
  - e:/CHICKTRADE/core/core_strategy_rules.json (v6.0 synced for bot/ & cloud_bot/)
  - e:/CHICKTRADE/core/live_learning_state.json (68 ACTIVE + 1 BENCH_WATCH seeded)
  - e:/CHICKTRADE/core/reports/chicktrade_v6_7coin_master_report.html
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from core.live_learning import (
    load_or_init_learning_state,
    save_learning_state,
    v3_barbell_tier,
)
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

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_6C_RULES = os.path.join(ROOT, "verified_tests", "results", "chicktrade_v5_53setups_rules.json")
SRC_NEAR = os.path.join(ROOT, "verified_tests", "results", "nearusdt_v5_discovery.json")
OUT_V6_JSON = os.path.join(ROOT, "verified_tests", "results", "chicktrade_v6_7coin_rules.json")
OUT_CORE_RULES = os.path.join(ROOT, "core", "core_strategy_rules.json")
OUT_HTML = os.path.join(ROOT, "core", "reports", "chicktrade_v6_7coin_master_report.html")

SIX_COINS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "LINKUSDT", "XRPUSDT", "SOLUSDT"]
SEVEN_COINS = SIX_COINS + ["NEARUSDT"]
TFS = ("15m", "30m", "1h")

NOTIONAL_CAP_20X = {
    "BTCUSDT": 300_000.0,
    "ETHUSDT": 300_000.0,
    "BNBUSDT": 150_000.0,
    "SOLUSDT": 150_000.0,
    "XRPUSDT": 150_000.0,
    "LINKUSDT": 150_000.0,
    "NEARUSDT": 100_000.0,
}
SLIP_BY_SYM = {
    "BTCUSDT": 0.0001,
    "ETHUSDT": 0.0001,
    "BNBUSDT": 0.0001,
    "SOLUSDT": 0.0001,
    "XRPUSDT": 0.0001,
    "LINKUSDT": 0.0001,
    "NEARUSDT": 0.0003,
}
BENCH_WATCH_SETUPS = {("NEARUSDT", "1h", "EMA34_W4_REV")}


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
    """Strict Binance USD-M Futures trade simulator (Argon audited, zero lookahead)."""
    i = sig["idx"]
    n_bars = len(opens)
    if i + 2 >= n_bars:
        return None
    sym, tf, d = sig["sym"], sig["tf"], sig["dir"]
    tf_ms = TF_MS[tf]
    u_pct = float(sig["u_frac"])
    sig_px = float(sig["entry"])
    u_px = float(sig["u_px"])
    if u_px <= 0:
        return None

    nxt_o = float(opens[i + 1])
    if d == 1:
        if nxt_o > sig_px + 0.25 * u_px:
            return {"ioc_missed": True}
    else:
        if nxt_o < sig_px - 0.25 * u_px:
            return {"ioc_missed": True}

    fill_px = nxt_o
    open_ts = int(ts[i + 1])
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
        bo, bh, bl = float(opens[j]), float(highs[j]), float(lows[j])
        if not tp1_hit:
            if (bl <= active_sl) if d == 1 else (bh >= active_sl):
                raw_px = min(bo, active_sl) if d == 1 else max(bo, active_sl)
                raw_u = ((raw_px - fill_px) / u_px) if d == 1 else ((fill_px - raw_px) / u_px)
                leg1_net_u = raw_u - entry_cost_u - taker_exit_u
                exit_idx = j
                outcome = "SL"
                break
            if (bh >= tp1_px) if d == 1 else (bl <= tp1_px):
                if not is_partial:
                    leg1_net_u = tp1_u - entry_cost_u - maker_exit_u
                    exit_idx = j
                    outcome = "TP1_ONLY"
                    break
                tp1_hit = True
                tp1_ts = int(ts[j] + tf_ms)
                leg1_net_u = 0.5 * (tp1_u - entry_cost_u - maker_exit_u)
                active_sl = l2_sl_px
                continue
        else:
            if (bl <= active_sl) if d == 1 else (bh >= active_sl):
                raw_px = min(bo, active_sl) if d == 1 else max(bo, active_sl)
                raw_u = ((raw_px - fill_px) / u_px) if d == 1 else ((fill_px - raw_px) / u_px)
                leg2_net_u = 0.5 * (raw_u - entry_cost_u - taker_exit_u)
                exit_idx = j
                outcome = "TP1_L2SL"
                break
            if (bh >= tp2_px) if d == 1 else (bl <= tp2_px):
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

    close_ts = int(ts[exit_idx] + tf_ms)
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


def simulate_v6_portfolio(
    trades: List[dict],
    ioc_missed: int = 0,
    buffer_base: float = 400.0,
    turbo_buffer: float = 250.0,
    use_turbo: bool = False,
    max_open: int = 5,
    max_same_dir: int = 3,
    init_cap: float = 10_000.0,
) -> dict:
    """v6.0 Portfolio Simulator: S2 Step-Locked Volume + 400G/Turbo 250G (110% Armor) + Cold-Only Cooling."""
    ordered = sorted(trades, key=lambda x: (x["open_ts"], -x.get("score", 0.0), TF_PRIORITY.get(x["tf"], 0)))
    eq, peak, max_dd = init_cap, init_cap, 0.0
    step, tier_cap = 0, init_cap
    turbo_active = False
    consec_sl, shadow_paused, shadow_until = 0, False, 0
    setup_sl_streak: Dict[Tuple[str, str, str], int] = {}
    open_pos: List[dict] = []
    closed: List[dict] = []
    curve: List[dict] = []
    total_funding = 0.0
    yearly: Dict[str, dict] = {}

    cool_triggers = cap5_skipped = same_dir_skipped = sym_lock_skipped = 0
    margin_skipped = risk12_skipped = shadow_skips = 0
    max_concurrent_seen = max_same_dir_seen = 0
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
                max_dd = min(max_dd, (eq - peak) / peak * 100.0)

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

                if p["outcome"] == "SL" and p["base_vol"] >= 0.75:
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
        if shadow_paused:
            if now >= shadow_until:
                shadow_paused = False
            else:
                shadow_skips += 1
                if t["pnl_u"] > 0:
                    shadow_until = min(shadow_until, t["close_ts"])
                continue

        if len(open_pos) >= max_open:
            cap5_skipped += 1
            continue
        if any(p["sym"] == t["sym"] for p in open_pos):
            sym_lock_skipped += 1
            continue
        if sum(1 for p in open_pos if p["dir"] == t["dir"]) >= max_same_dir:
            same_dir_skipped += 1
            continue

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

        if t.get("elevated_vol", False):
            vm = vm * 0.5

        buf = turbo_buffer if (use_turbo and turbo_active) else buffer_base
        if use_turbo and turbo_active:
            if eq - 30.0 * (tier_cap / turbo_buffer) < init_cap * 1.10:
                buf = buffer_base

        u_usd = (tier_cap / buf) * vm
        notional = u_usd / max(t["u_pct"], 0.001)
        max_notional = min(NOTIONAL_CAP_20X.get(t["sym"], 100_000.0), eq * 5.0)
        if notional > max_notional:
            notional = max_notional
            u_usd = notional * t["u_pct"]
        if notional < 20.0:
            continue

        new_sl = u_usd * float(t["sl_u"])
        open_sl = sum(p["worst_sl_usd"] for p in open_pos)
        if (open_sl + new_sl) > eq * 0.12:
            risk12_skipped += 1
            continue
        avail_eq = eq - open_sl
        used_margin = sum(p["notional"] / 20.0 for p in open_pos)
        if avail_eq <= 0 or (used_margin + notional / 20.0) > avail_eq * 0.75:
            margin_skipped += 1
            continue

        open_pos.append({
            **t,
            "base_vol": base_m,
            "eff_vol": vm,
            "notional": notional,
            "worst_sl_usd": new_sl,
            "pnl_usd": t["pnl_u"] * u_usd,
            "funding_usd": t.get("funding_u", 0.0) * u_usd,
        })
        max_concurrent_seen = max(max_concurrent_seen, len(open_pos))
        max_same_dir_seen = max(max_same_dir_seen, sum(1 for p in open_pos if p["dir"] == t["dir"]))
        max_open_risk_pct = max(max_open_risk_pct, (open_sl + new_sl) / max(eq, 1.0) * 100.0)

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


def load_source_setups() -> Tuple[List[dict], List[dict], List[dict]]:
    with open(SRC_6C_RULES, "r", encoding="utf-8") as f:
        r6_src = json.load(f)
    rules_6c = []
    for r in r6_src["rules"]:
        rules_6c.append({
            "sym": r["sym"], "tf": r["tf"], "setup": r["setup"],
            "sl_u": float(r["sl_u"]), "tp1_u": float(r["tp1_u"]), "tp2_u": float(r["tp2_u"]),
            "mode": r["mode"], "score": float(r["score"]),
        })

    with open(SRC_NEAR, "r", encoding="utf-8") as f:
        near_d = json.load(f)
    near_active, bench_list = [], []
    for q in near_d["coins"]["NEARUSDT"]["qualified_setups"]:
        item = {
            "sym": "NEARUSDT", "tf": q["tf"], "setup": q["setup"],
            "sl_u": float(q["sl_u"]), "tp1_u": float(q["tp1_u"]), "tp2_u": float(q["tp2_u"]),
            "mode": q["mode"], "score": float(q["score"]),
        }
        if ("NEARUSDT", q["tf"], q["setup"]) in BENCH_WATCH_SETUPS:
            bench_list.append({**item, "bench_reason": "Y3 regime bleed (-21.2U, PF 0.27) — đang theo dõi lệnh nháp chờ hồi sinh"})
        else:
            near_active.append(item)
    return rules_6c, rules_6c + near_active, bench_list


def sync_core_rules(enriched_68: List[dict]) -> None:
    cham_map = {
        "W1": "Chạm 1 (Thuận xu hướng)", "W2": "Chạm 2 (Thuận xu hướng)",
        "W3": "Chạm 3 (Thuận xu hướng)", "W4_REV": "Chạm 4 (Đảo chiều)",
    }
    arch_map = {
        "PARTIAL_NO_MOVE": ("PARTIAL_NO_MOVE", 0.5, 0.0, "Giữ nguyên SL gốc"),
        "PARTIAL_M1U": ("PARTIAL_MOVE_SL", 0.5, -1.0, "Dời SL -1U"),
        "PARTIAL_BE": ("PARTIAL_BE", 0.5, 0.0, "Dời SL Hòa vốn (BE)"),
        "ONLY_TP1": ("ONLY_TP1", 1.0, 0.0, "Chốt toàn bộ tại TP1"),
        "ONLY_1R": ("ONLY_1R", 1.0, 0.0, "Chốt toàn bộ tại 1R"),
    }
    rules_out = []
    for r in enriched_68:
        ema_part, w_part = r["setup"].split("_", 1)
        vm = float(r["vol_mult"])
        is_core = vm >= 1.0
        exit_arch, tp1_frac, l2_sl, l2_lbl = arch_map.get(r["mode"], (r["mode"], 0.5, 0.0, r["mode"]))
        s4 = r["s4"]
        n = int(s4["n"])
        rules_out.append({
            "coin": r["sym"], "tf": r["tf"], "ema": ema_part,
            "cham": cham_map.get(w_part, w_part), "setup_code": r["setup"],
            "status": "CORE" if is_core else "SCOUT", "is_scout": not is_core,
            "v5_tier": r["m2_tier"], "is_must_order": bool(r["is_must"]),
            "exit_arch": exit_arch, "best_exit_raw": r["mode"], "tp1_frac": tp1_frac,
            "sl_u": float(r["sl_u"]), "tp1_u": float(r["tp1_u"]), "tp2_u": float(r["tp2_u"]),
            "leg2_sl_u": l2_sl, "leg2_sl_label": l2_lbl, "allowed_direction": "BOTH",
            "adaptive_volume": {
                "init_vol_mult": vm, "hot_streak_max_mult": vm,
                "cold_streak_3sl_mult": min(vm, 0.50) if vm >= 1.0 else min(vm, 0.25),
                "cold_only_cooling": True, "tier": r["m2_tier"],
            },
            "prior_1y": {
                "trades": n, "wins": int(round(n * float(s4["wr"]) / 100.0)),
                "wr_pct": float(s4["wr"]), "profit_wr_pct": float(s4["pwr"]),
                "pf_1y": float(s4["pf"]), "net_u": float(s4["net_u"]),
                "avg_u": float(s4["avg_u"]), "epr": float(r["epr"]),
                "s5_res_pct": float(r["s5_res_pct"]),
                "setup_mdd_u": float(r["mdd_u"]), "max_csl": int(r["max_csl"]), "rf": float(r["rf"]),
            },
        })

    doc = {
        "version": "v6.0-7coin-68active-1bench-v3barbell-binance-exact",
        "updated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "description": "CHICKTRADE v6.0 — 7 Coins (BTC/ETH/BNB/LINK/XRP/SOL/NEAR) · 68 ACTIVE + 1 BENCH_WATCH · V3 Barbell Alpha (0.25x-1.75x) · S2-Locked 400G/Turbo 250G · Slow Bayesian + Shadow Bench-Watch",
        "unit_config": {
            "mode": "U3_HTF_MACRO_ATR", "htf": "4h", "atr_period": 14, "ema_smooth": 20,
            "gamma_tf": {"5m": 0.17, "15m": 0.29, "30m": 0.41, "1h": 0.59}, "min_u_pct": 0.0008,
        },
        "risk_config": {
            "max_concurrent": 5, "max_same_direction": 3, "max_concurrent_sl_risk_pct": 12.0,
            "buffer_default": 400, "buffer_turbo": 250, "turbo_trigger_pct": 25.0,
            "turbo_floor_pct": 110.0, "s2_step_up_pct": 8.0, "s2_step_down_pct": -10.0,
        },
        "total_rules": len(rules_out),
        "rules": rules_out,
    }
    with open(OUT_CORE_RULES, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
    print(f"[OK] Synced {OUT_CORE_RULES} (v6.0 · {len(rules_out)} rules)")


def main():
    rules_6c, rules_68, bench_list = load_source_setups()
    print(f"CHICKTRADE v6.0 Master Build: {len(rules_68)} ACTIVE + {len(bench_list)} BENCH_WATCH across 7 Coins")

    market, funding_map = {}, {}
    with ThreadPoolExecutor(max_workers=7) as pool:
        futs = {sym: pool.submit(load_coin_6y, sym) for sym in SEVEN_COINS}
        for sym, fut in futs.items():
            m_d, f_ts, f_rt = fut.result()
            market[sym], funding_map[sym] = m_d, (f_ts, f_rt)

    gs_mod.SIX_COINS = SEVEN_COINS
    sig_map = extract_ema_signals_6y(market, tfs=TFS, max_wave=4)
    t_start = int(datetime(2023, 11, 1, tzinfo=timezone.utc).timestamp() * 1000)
    t_end = int(datetime(2026, 11, 1, tzinfo=timezone.utc).timestamp() * 1000)

    # Evaluate all 68 active + 1 bench setup under Binance-Exact S4 & S5
    s4_all, s5_all = [], []
    m4_tot = m5_tot = 0
    per_setup = {}

    for r in rules_68 + bench_list:
        sym, tf, st = r["sym"], r["tf"], r["setup"]
        sigs = [s for s in sig_map.get((sym, tf, st), []) if t_start <= s["open_ts"] < t_end]
        df = market[sym][tf]
        op, hi, lo, cl, ts = df["open"].values, df["high"].values, df["low"].values, df["close"].values, df["open_time_ms"].values
        fr_ts, fr_rate = funding_map[sym]
        mb = MAX_HOLD_BARS[tf]
        is_bench = (sym, tf, st) in BENCH_WATCH_SETUPS

        sc_out = {}
        for sc_name in ("S4_REAL_FUNDING", "S5_PESSIMISTIC"):
            busy = -1
            tl, miss = [], 0
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
            n = len(tl)
            wins = sum(1 for x in tl if x["pnl_u"] > 0)
            gp = sum(x["pnl_u"] for x in tl if x["pnl_u"] > 0)
            gl = sum(abs(x["pnl_u"]) for x in tl if x["pnl_u"] <= 0)
            net = gp - gl
            eq_u = pk_u = mdd_u = 0.0
            csl = max_csl = 0
            for x in sorted(tl, key=lambda z: z["close_ts"]):
                eq_u += x["pnl_u"]
                pk_u = max(pk_u, eq_u)
                mdd_u = max(mdd_u, pk_u - eq_u)
                if x["pnl_u"] <= 0:
                    csl += 1
                    max_csl = max(max_csl, csl)
                else:
                    csl = 0
            sc_out[sc_name] = {
                "n": n, "missed": miss, "wr": round(wins / max(n, 1) * 100.0, 1),
                "pwr": round(gp / max(gp + gl, 1e-9) * 100.0, 1),
                "pf": round(gp / max(gl, 1e-9), 2) if gl > 1e-9 else (9.99 if gp > 0 else 0.0),
                "net_u": round(net, 2), "avg_u": round(net / max(n, 1), 3),
                "mdd_u": round(mdd_u, 1), "max_csl": max_csl,
                "rf": round(net / max(mdd_u, r["sl_u"]), 2),
                "trades": tl,
            }
        per_setup[(sym, tf, st)] = sc_out

        if not is_bench:
            s4 = sc_out["S4_REAL_FUNDING"]
            epr = s4["avg_u"] / r["sl_u"] if r["sl_u"] > 0 else 0.0
            vm, tier_lbl = v3_barbell_tier(s4["pwr"], epr, s4["n"])
            r["vol_mult"], r["m2_tier"], r["is_must"] = vm, tier_lbl, vm >= 1.50
            m4_tot += s4["missed"]
            m5_tot += sc_out["S5_PESSIMISTIC"]["missed"]
            for tr in s4["trades"]:
                s4_all.append({**tr, "vol_mult": vm, "score": r["score"]})
            for tr in sc_out["S5_PESSIMISTIC"]["trades"]:
                s5_all.append({**tr, "vol_mult": vm, "score": r["score"]})

    # Run 6-coin baseline & 7-coin v6.0 portfolios
    keys_6c = {(r["sym"], r["tf"], r["setup"]) for r in rules_6c}
    s4_6 = [t for t in s4_all if (t["sym"], t["tf"], t["setup"]) in keys_6c]
    s5_6 = [t for t in s5_all if (t["sym"], t["tf"], t["setup"]) in keys_6c]
    m4_6 = sum(per_setup[k]["S4_REAL_FUNDING"]["missed"] for k in keys_6c)
    m5_6 = sum(per_setup[k]["S5_PESSIMISTIC"]["missed"] for k in keys_6c)

    p6_s4_400 = simulate_v6_portfolio(s4_6, m4_6, 400, 250, False, 5, 3)
    p6_s4_tur = simulate_v6_portfolio(s4_6, m4_6, 400, 250, True, 5, 3)
    p6_s5_400 = simulate_v6_portfolio(s5_6, m5_6, 400, 250, False, 5, 3)
    p6_s5_tur = simulate_v6_portfolio(s5_6, m5_6, 400, 250, True, 5, 3)

    p7_s4_400 = simulate_v6_portfolio(s4_all, m4_tot, 400, 250, False, 5, 3)
    p7_s4_350 = simulate_v6_portfolio(s4_all, m4_tot, 400, 350, True, 5, 3)
    p7_s4_320 = simulate_v6_portfolio(s4_all, m4_tot, 400, 320, True, 5, 3)
    p7_s4_tur = simulate_v6_portfolio(s4_all, m4_tot, 400, 250, True, 5, 3)

    p7_s5_400 = simulate_v6_portfolio(s5_all, m5_tot, 400, 250, False, 5, 3)
    p7_s5_350 = simulate_v6_portfolio(s5_all, m5_tot, 400, 350, True, 5, 3)
    p7_s5_320 = simulate_v6_portfolio(s5_all, m5_tot, 400, 320, True, 5, 3)
    p7_s5_tur = simulate_v6_portfolio(s5_all, m5_tot, 400, 250, True, 5, 3)

    print("\n================ CHICKTRADE v6.0 MASTER RESULTS (7 COINS · 68 ACTIVE + 1 BENCH) ================")
    print(f"  [S4] Bao 400G : ${p7_s4_400['final_eq']:>10,.2f} ({p7_s4_400['roi_pct']:+7.1f}%) | DD={p7_s4_400['max_dd_pct']:6.2f}% | Cal={p7_s4_400['calmar']:>5.1f} | PF={p7_s4_400['pf']} | PWR={p7_s4_400['pwr']}% | N={p7_s4_400['trades']}")
    print(f"  [S4] Turbo350G: ${p7_s4_350['final_eq']:>10,.2f} ({p7_s4_350['roi_pct']:+7.1f}%) | DD={p7_s4_350['max_dd_pct']:6.2f}% | Cal={p7_s4_350['calmar']:>5.1f} | PF={p7_s4_350['pf']}")
    print(f"  [S4] Turbo320G: ${p7_s4_320['final_eq']:>10,.2f} ({p7_s4_320['roi_pct']:+7.1f}%) | DD={p7_s4_320['max_dd_pct']:6.2f}% | Cal={p7_s4_320['calmar']:>5.1f} | PF={p7_s4_320['pf']}")
    print(f"  [S4] Turbo250G: ${p7_s4_tur['final_eq']:>10,.2f} ({p7_s4_tur['roi_pct']:+7.1f}%) | DD={p7_s4_tur['max_dd_pct']:6.2f}% | Cal={p7_s4_tur['calmar']:>5.1f} | PF={p7_s4_tur['pf']} | N={p7_s4_tur['trades']}")
    print(f"  [S5] Bao 400G : ${p7_s5_400['final_eq']:>10,.2f} ({p7_s5_400['roi_pct']:+7.1f}%) | DD={p7_s5_400['max_dd_pct']:6.2f}% | PF={p7_s5_400['pf']}")
    print(f"  [S5] Turbo250G: ${p7_s5_tur['final_eq']:>10,.2f} ({p7_s5_tur['roi_pct']:+7.1f}%) | DD={p7_s5_tur['max_dd_pct']:6.2f}% | PF={p7_s5_tur['pf']}")

    # Compute 1U averages
    u_avg_map = {}
    for sym in SEVEN_COINS:
        df4 = market[sym]["4h"]
        df4_sub = df4[(df4["open_time_ms"] >= t_start) & (df4["open_time_ms"] < t_end)]
        macro4 = ((0.5 * compute_atr14(df4)) / df4["close"]).ewm(span=20, adjust=False).mean()
        macro4_sub = macro4.loc[df4_sub.index]
        u_avg_map[sym] = {tf: round(float(max(MIN_U_PCT, macro4_sub.mean() * GAMMA_TF[tf]) * 100.0), 3) for tf in TFS}

    enriched_68, bench_enriched = [], []
    tier_dist: Dict[str, int] = {}
    for r in rules_68:
        k = (r["sym"], r["tf"], r["setup"])
        s4 = {kk: vv for kk, vv in per_setup[k]["S4_REAL_FUNDING"].items() if kk != "trades"}
        s5 = {kk: vv for kk, vv in per_setup[k]["S5_PESSIMISTIC"].items() if kk != "trades"}
        epr = round(s4["avg_u"] / r["sl_u"], 3) if r["sl_u"] > 0 else 0.0
        s5_res = round((s5["net_u"] / s4["net_u"] * 100.0) if s4["net_u"] > 0 else 0.0, 1)
        row = {**r, "lifecycle": "ACTIVE", "s4": s4, "s5": s5, "epr": epr, "s5_res_pct": s5_res,
               "mdd_u": s4["mdd_u"], "max_csl": s4["max_csl"], "rf": s4["rf"]}
        enriched_68.append(row)
        tier_dist[r["m2_tier"]] = tier_dist.get(r["m2_tier"], 0) + 1

    for b in bench_list:
        k = (b["sym"], b["tf"], b["setup"])
        s4 = {kk: vv for kk, vv in per_setup[k]["S4_REAL_FUNDING"].items() if kk != "trades"}
        s5 = {kk: vv for kk, vv in per_setup[k]["S5_PESSIMISTIC"].items() if kk != "trades"}
        bench_enriched.append({**b, "lifecycle": "BENCH_WATCH", "vol_mult": 0.0, "m2_tier": "BENCH (0.00×)", "s4": s4, "s5": s5})

    coins_payload = {}
    for sym in SEVEN_COINS:
        c_rows = sorted([x for x in enriched_68 if x["sym"] == sym], key=lambda x: (-x["vol_mult"], -x["rf"], -x["s4"]["pwr"]))
        c_bench = [x for x in bench_enriched if x["sym"] == sym]
        coins_payload[sym] = {
            "count": len(c_rows),
            "bench_count": len(c_bench),
            "must_count": sum(1 for x in c_rows if x["is_must"]),
            "avg_vol": round(float(np.mean([x["vol_mult"] for x in c_rows])), 2) if c_rows else 0.0,
            "tot_net_s4": round(sum(x["s4"]["net_u"] for x in c_rows), 1),
            "u_avg": u_avg_map[sym],
            "notional_cap": int(NOTIONAL_CAP_20X[sym]),
            "slip_bps": round(SLIP_BY_SYM[sym] * 10000, 1),
            "setups": c_rows,
            "bench_setups": c_bench,
        }

    # Export v6.0 JSON, sync core_strategy_rules.json, and re-seed live_learning_state.json
    gen_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    v6_export = {
        "version": "CHICKTRADE_v6.0_7COIN_68ACTIVE_1BENCH",
        "generated_utc": gen_utc,
        "coins": SEVEN_COINS,
        "total_active": len(enriched_68),
        "total_bench": len(bench_enriched),
        "tier_dist": tier_dist,
        "buffer_frontier": [
            {"buf": "Bao 400G", "rem_5sl_125": "115.6%", "s4": {k: v for k, v in p7_s4_400.items() if k != "curve"}, "s5": {k: v for k, v in p7_s5_400.items() if k != "curve"}},
            {"buf": "Turbo 350G", "rem_5sl_125": "114.3%", "s4": {k: v for k, v in p7_s4_350.items() if k != "curve"}, "s5": {k: v for k, v in p7_s5_350.items() if k != "curve"}},
            {"buf": "Turbo 320G", "rem_5sl_125": "113.3%", "s4": {k: v for k, v in p7_s4_320.items() if k != "curve"}, "s5": {k: v for k, v in p7_s5_320.items() if k != "curve"}},
            {"buf": "Turbo 250G", "rem_5sl_125": "110.0%", "s4": {k: v for k, v in p7_s4_tur.items() if k != "curve"}, "s5": {k: v for k, v in p7_s5_tur.items() if k != "curve"}},
        ],
        "rules": enriched_68,
        "bench_watch": bench_enriched,
    }
    with open(OUT_V6_JSON, "w", encoding="utf-8") as f:
        json.dump(v6_export, f, ensure_ascii=False, indent=2)
    print(f"[OK] Saved {OUT_V6_JSON}")

    sync_core_rules(enriched_68)
    ls = load_or_init_learning_state()
    save_learning_state(ls)
    print(f"[OK] Synced live_learning_state.json ({len(ls['setups'])} setups)")

    html_data = {
        "generated_utc": gen_utc,
        "tier_dist": tier_dist,
        "portfolio": {
            "s4_400g": p7_s4_400, "s4_turbo": p7_s4_tur,
            "s5_400g": p7_s5_400, "s5_turbo": p7_s5_tur,
        },
        "baseline_6c": {
            "s4_400g": {k: v for k, v in p6_s4_400.items() if k != "curve"},
            "s4_turbo": {k: v for k, v in p6_s4_tur.items() if k != "curve"},
            "s5_400g": {k: v for k, v in p6_s5_400.items() if k != "curve"},
            "s5_turbo": {k: v for k, v in p6_s5_tur.items() if k != "curve"},
        },
        "buffer_frontier": v6_export["buffer_frontier"],
        "coins": coins_payload,
        "bench_watch": bench_enriched,
    }
    build_html_v6(html_data)


def build_html_v6(data: dict) -> None:
    data_js = json.dumps(data, ensure_ascii=False)
    html = f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CHICKTRADE v6.0 MASTER — 7 COIN (68 Active + 1 Bench-Watch · Binance-Exact)</title>
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
  .hero-bar {{ background: linear-gradient(135deg, #0F172A 0%, #1E293B 55%, #065F46 100%); color: #FFF; padding: 22px 24px 18px; border-bottom: 3px solid #10B981; }}
  .hero-inner {{ max-width: 1360px; margin: 0 auto; display: flex; justify-content: space-between; align-items: flex-start; gap: 16px; flex-wrap: wrap; }}
  .hero-title h1 {{ margin: 0 0 6px; font-size: 20.5px; font-weight: 800; letter-spacing: -0.02em; }}
  .hero-title p {{ margin: 0; font-size: 12.8px; color: #CBD5E1; }}
  .hero-title code {{ background: rgba(255,255,255,0.14); color: #A7F3D0; }}
  .hero-actions {{ display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }}
  .ver-badge {{ background: rgba(16,185,129,0.22); border: 1px solid rgba(16,185,129,0.5); color: #A7F3D0; padding: 5px 12px; border-radius: 999px; font-size: 11.5px; font-weight: 700; font-family: 'JetBrains Mono', monospace; }}
  .nav-tabs {{ background: #FFF; border-bottom: 1px solid var(--border); position: sticky; top: 0; z-index: 40; }}
  .nav-tabs-inner {{ max-width: 1360px; margin: 0 auto; display: flex; gap: 4px; padding: 8px 20px; overflow-x: auto; }}
  .tab-btn {{ border: 1px solid transparent; background: transparent; color: #475569; padding: 7px 12px; border-radius: 8px; font-weight: 600; font-size: 12.4px; cursor: pointer; white-space: nowrap; }}
  .tab-btn:hover {{ background: #F1F5F9; color: var(--text); }}
  .tab-btn.active {{ background: #ECFDF5; color: #065F46; border-color: #A7F3D0; }}
  .tab-btn.highlight {{ border-color: #BAE6FD; background: #F0F9FF; color: #0369A1; }}
  .tab-btn.highlight.active {{ background: #0284C7; color: #FFF; border-color: #0284C7; }}
  .page-shell {{ max-width: 1360px; margin: 0 auto; padding: 20px 20px 56px; }}
  .tab-panel {{ display: none; }}
  .tab-panel.active {{ display: block; }}
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
  .grid-3 {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 12px; }}
  .grid-4 {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin-bottom: 16px; }}
  .grid-6 {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(165px, 1fr)); gap: 10px; }}
  .tbl-wrap {{ overflow-x: auto; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 12.4px; }}
  th {{ background: #F8FAFC; color: #475569; font-weight: 700; text-align: left; padding: 9px 10px; border-bottom: 2px solid var(--border); white-space: nowrap; }}
  td {{ padding: 8px 10px; border-bottom: 1px solid #F1F5F9; white-space: nowrap; }}
  tr:hover td {{ background: #F8FAFC; }}
  .num {{ font-family: 'JetBrains Mono', monospace; }}
  .pos {{ color: var(--emerald); font-weight: 700; }}
  .neg {{ color: var(--rose); font-weight: 700; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 11px; font-weight: 700; font-family: 'JetBrains Mono', monospace; }}
  .b-v175 {{ background: #FEF3C7; color: #92400E; border: 1px solid #F59E0B; }}
  .b-v150 {{ background: #D1FAE5; color: #065F46; }}
  .b-v100 {{ background: #E0F2FE; color: #075985; }}
  .b-v075 {{ background: #EDE9FE; color: #5B21B6; }}
  .b-v050 {{ background: #F1F5F9; color: #475569; }}
  .b-v025 {{ background: #FEE2E2; color: #991B1B; }}
  .b-bench {{ background: #FEF9C3; color: #854D0E; border: 1px solid #FACC15; }}
  .must-badge {{ display: inline-block; margin-left: 6px; background: #FEF3C7; color: #B45309; border: 1px solid #FCD34D; font-size: 10px; font-weight: 800; padding: 1px 6px; border-radius: 4px; }}
  .new-tag {{ display: inline-block; margin-left: 5px; background: #DCFCE7; color: #166534; font-size: 10px; font-weight: 800; padding: 1px 6px; border-radius: 4px; }}
  .calc-controls {{ display: flex; gap: 14px; flex-wrap: wrap; margin-bottom: 14px; }}
  .calc-field {{ display: flex; flex-direction: column; gap: 4px; min-width: 180px; }}
  .calc-field label {{ font-size: 12px; font-weight: 600; color: #475569; }}
  .calc-field input, .calc-field select {{ padding: 8px 10px; border: 1px solid #CBD5E1; border-radius: 8px; font-family: 'JetBrains Mono', monospace; font-size: 13px; }}
  .mech-card {{ background: #F8FAFC; border: 1px solid var(--border); border-top: 3px solid var(--sky); border-radius: 10px; padding: 14px 16px; }}
  .mech-card h4 {{ margin: 0 0 6px; font-size: 14px; }}
  .mech-card p, .mech-card li {{ margin: 0; font-size: 12.5px; color: #475569; line-height: 1.58; }}
  .mech-card ul {{ margin: 6px 0 0; padding-left: 18px; }}
  .lesson-box {{ background: #FFFBEB; border: 1px solid #FDE68A; border-left: 4px solid #D97706; border-radius: 10px; padding: 14px 16px; }}
  .lesson-box li {{ margin-bottom: 6px; font-size: 12.8px; color: #334155; }}
</style>
</head>
<body>

<header class="hero-bar">
  <div class="hero-inner">
    <div class="hero-title">
      <h1>🏆 CHICKTRADE v6.0 MASTER — 7 COIN (68 Active + 1 Bench-Watch · Binance-Exact)</h1>
      <p>Hệ thống hoàn chỉnh: <b>V3 Barbell Volume (<code>0.25×–1.75×</code>)</b> · <b>Khóa cứng Bậc S2 (`400G ↔ Turbo 250G`, Giáp 110%)</b> · <b>Tự Học Chậm + Ghế Dự Bị Hồi Sinh</b> · Kiểm định 36 tháng (<code>2023-11 → 2026-10</code>)</p>
    </div>
    <div class="hero-actions">
      <span class="ver-badge">v6.0 MASTER · 7 COINS · 68+1 SETUPS · <span id="gen-utc">--</span></span>
    </div>
  </div>
</header>

<nav class="nav-tabs">
  <div class="nav-tabs-inner">
    <button class="tab-btn active" onclick="switchTab('tab-overview', this)">🌐 Tổng Quan v6.0</button>
    <button class="tab-btn highlight" onclick="switchTab('tab-framework', this)">📐 Kiến Trúc, Tiêu Chí &amp; Điểm Ngọt</button>
    <button class="tab-btn" onclick="switchTab('BTCUSDT', this)">🟠 BTC</button>
    <button class="tab-btn" onclick="switchTab('ETHUSDT', this)">🔵 ETH</button>
    <button class="tab-btn" onclick="switchTab('BNBUSDT', this)">🟡 BNB</button>
    <button class="tab-btn" onclick="switchTab('LINKUSDT', this)">🔷 LINK</button>
    <button class="tab-btn" onclick="switchTab('XRPUSDT', this)">⚪ XRP</button>
    <button class="tab-btn" onclick="switchTab('SOLUSDT', this)">🟣 SOL</button>
    <button class="tab-btn" onclick="switchTab('NEARUSDT', this)">⚡ NEAR</button>
    <button class="tab-btn" onclick="switchTab('tab-mech', this)">🧮 Máy Tính S2 &amp; Ghế Dự Bị</button>
  </div>
</nav>

<main class="page-shell">

  <!-- TAB 1: OVERVIEW -->
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
          <p class="section-sub">Đo thực tế xuyên suốt 36 tháng (Bao 400G) — không vượt trần 5 lệnh, không vượt trần rủi ro 12%, 0 nhìn trước tương lai</p>
        </div>
      </div>
      <div class="grid-4" id="telemetry-strip" style="margin-bottom:0;"></div>
    </div>

    <div class="section-card">
      <div class="section-head">
        <div>
          <h3>⚖️ Đối Chiếu Trực Tiếp: 6 Coin Gốc (53 Setup) vs 7 Coin v6.0 (68 Setup · Thêm NEAR)</h3>
          <p class="section-sub">Chạy trên <b>cùng 1 động cơ Binance-Exact</b> (Khóa cứng Volume theo Bậc S2 · V3 Barbell 0.25×–1.75×)</p>
        </div>
      </div>
      <div class="tbl-wrap">
        <table>
          <thead><tr><th>Phiên Bản</th><th>Chế Độ</th><th>Vốn Cuối ($10k →)</th><th>ROI 36M</th><th>MaxDD</th><th>Calmar</th><th>Số Lệnh</th><th>WR</th><th>PWR</th><th>PF</th><th>Funding Thật</th></tr></thead>
          <tbody id="compare-tbody"></tbody>
        </table>
      </div>
    </div>

    <div class="section-card">
      <div class="section-head">
        <div>
          <h3>📈 Đường Cong Tài Sản 36 Tháng — CHICKTRADE v6.0 (Bao 400G vs Auto-Turbo 250G)</h3>
          <p class="section-sub">Vốn $10,000 · 20× Cross · Max 5 lệnh mở đồng thời · Lãi kép S2 (+8%/-10%) · Van Hạ Lạnh Cold-Only</p>
        </div>
      </div>
      <div id="eq-chart" style="width:100%;height:370px;"></div>
    </div>

    <div class="section-card">
      <div class="section-head">
        <div>
          <h3>🎚️ Phân Bổ 6 Nấc Volume V3 Barbell Alpha Trên 68 Setup Active</h3>
          <p class="section-sub">Dồn lực vào siêu phẩm Edge/Risk dày (<code>1.50×–1.75×</code>), ép kèo SL rộng/biên mỏng về <code>0.25×–0.50×</code> (Vol tĩnh TB chỉ <code>0.82×</code>)</p>
        </div>
      </div>
      <div class="grid-6" id="vol-dist-grid"></div>
    </div>

    <div class="section-card">
      <div class="section-head">
        <div>
          <h3>📅 Hiệu Suất Từng Năm (2023 → 2026 · Bao 400G vs Auto-Turbo 250G)</h3>
          <p class="section-sub">Tăng trưởng ổn định qua cả 4 năm dương lịch dưới chuẩn khớp lệnh thật Binance</p>
        </div>
      </div>
      <div class="tbl-wrap">
        <table>
          <thead><tr><th>Năm</th><th>Số Lệnh</th><th>WR Lệnh</th><th>Thắng Tiền (PWR)</th><th>PF</th><th>Lãi/Lỗ Bao 400G</th><th>Vốn Cuối 400G</th><th>Lãi/Lỗ Turbo 250G</th><th>Vốn Cuối Turbo</th></tr></thead>
          <tbody id="yearly-tbody"></tbody>
        </table>
      </div>
    </div>

    <div class="section-card">
      <div class="section-head">
        <div>
          <h3>🪙 Tổng Hợp 7 Coin Trong Bộ CHICKTRADE v6.0 (68 Active + 1 Bench-Watch)</h3>
          <p class="section-sub">Bấm vào từng dòng hoặc tab coin phía trên để xem chi tiết từng setup, Recovery Factor và MaxDD riêng</p>
        </div>
      </div>
      <div class="tbl-wrap">
        <table>
          <thead><tr><th>Coin</th><th>Active</th><th>Bench</th><th>Siêu Phẩm (≥1.50×)</th><th>Vol TB</th><th>Tổng Net S4 (U)</th><th>Trần 20×</th><th>Slip S4</th><th>1U M15</th><th>1U M30</th><th>1U 1H</th></tr></thead>
          <tbody id="coin-sum-tbody"></tbody>
        </table>
      </div>
    </div>
  </section>

  <!-- TAB 2: ARCHITECTURE, CRITERIA & SWEET SPOTS (NEW IN v6.0!) -->
  <section id="tab-framework" class="tab-panel">
    <div class="section-card">
      <div class="section-head">
        <div>
          <h3>🏗️ 1. Quy Trình Tạo Ra Hệ Thống CHICKTRADE v6.0 (Từ 336 Ứng Viên → 68 Tinh Hoa + 1 Dự Bị)</h3>
          <p class="section-sub">Mọi quyết định loại bỏ hay giữ lại đều có bằng chứng định lượng trên dữ liệu thật 2021–2026 (warmup 2 năm, test 36 tháng)</p>
        </div>
      </div>
      <div class="grid-3">
        <div class="mech-card" style="border-top-color:#059669;">
          <h4>Bước 1 · Quét Khám Phá 336 Ứng Viên</h4>
          <p>Quét toàn diện 7 Coin × 4 khung (<code>5m/15m/30m/1h</code>) × 3 EMA (<code>34/89/200</code>) × 4 lần chạm (<code>W1/W2/W3/W4_REV</code>) × 11 kịch bản thoát lệnh (<code>PARTIAL_M1U, PARTIAL_NO_MOVE, PARTIAL_BE, ONLY_TP1, ONLY_1R</code>). Độc lập kiểm định thêm <code>AVAX</code>, <code>ZEC</code>, <code>NEAR</code>.</p>
        </div>
        <div class="mech-card" style="border-top-color:#0284C7;">
          <h4>Bước 2 · Lọc Phễu 5 Cửa Ải (Argon Gate)</h4>
          <p>Loại sạch khung <code>5m</code> (nhiễu phí), loại kèo cày lệnh biên mỏng (<code>N ≥ 60</code> mà <code>EPR &lt; 0.11</code>), loại kèo yếu trước trượt giá (<code>S5res &lt; 68%</code>). Loại <code>AVAX</code> (loãng danh mục) &amp; <code>ZEC</code> (nghẽn trần 5 lệnh + rủi ro Privacy); giữ <b>NEAR</b> giúp kéo DD giảm từ <code>-10.1% → -8.56%</code>.</p>
        </div>
        <div class="mech-card" style="border-top-color:#D97706;">
          <h4>Bước 3 · Chuẩn Hóa Binance-Exact &amp; V3 Barbell</h4>
          <p>Khóa 7 chuẩn khớp lệnh thật (Max 5 lệnh, 1/coin, ≤ 3 cùng chiều, phanh 2-SL nhân quả 0 lookahead, IOC thật, Gap-SL, không ăn kép TP1+TP2 cùng nến, Funding 8h thật). Tái phân bổ Volume <b>V3 Barbell</b> tăng gấp đôi lợi nhuận mà không tăng đòn bẩy trung bình.</p>
        </div>
      </div>
    </div>

    <div class="section-card">
      <div class="section-head">
        <div>
          <h3>📊 2. Bản Đồ 7 Chỉ Số &amp; Tương Quan Thực Nghiệm (68 Setup)</h3>
          <p class="section-sub">Cơ sở toán học giải thích vì sao CHICKTRADE v6.0 xếp hạng Volume theo <b>EPR + PWR</b> chứ không nhìn mỗi Winrate lệnh</p>
        </div>
      </div>
      <div class="tbl-wrap">
        <table>
          <thead><tr><th>Chỉ Số</th><th>Tương Quan Với Drawdown Riêng (`MaxDD_U`)</th><th>Tương Quan Với Hiệu Suất Lãi/DD (`Recovery Factor`)</th><th>Ý Nghĩa Thực Chiến</th></tr></thead>
          <tbody>
            <tr><td><b><code>EPR = TB_U / SL_U</code></b></td><td class="num pos">-0.408 (giảm DD)</td><td class="num pos"><b>+0.735 (#1 tuyệt đối)</b></td><td>Đo biên lãi trên mỗi 1U rủi ro SL — phân biệt kèo SL 5U khỏe vs SL 7U nặng</td></tr>
            <tr><td><b><code>PWR</code> (Thắng Tiền)</b></td><td class="num pos">-0.468 (giảm DD)</td><td class="num pos"><b>+0.649 (#2)</b></td><td><code>GP / (GP + GL)</code> — chuẩn hơn WR lệnh vì hệ thống có 2 chân chốt lời TP1+TP2</td></tr>
            <tr><td><b><code>Net_U</code> (Tổng Lãi)</b></td><td class="num">+0.366</td><td class="num pos">+0.613 (#3)</td><td>Lãi tổng lớn phải đi kèm <code>EPR</code> dày mới tốt (tránh bẫy cày 150 lệnh mỏng)</td></tr>
            <tr><td><b><code>Max SL Liên Tiếp</code></b></td><td class="num neg"><b>+0.672 (#1 gây DD)</b></td><td class="num neg">-0.280</td><td>Chuỗi SL dài là thủ phạm số 1 kéo sụt quỹ → tự động bị ghìm volume + Van Hạ Lạnh</td></tr>
            <tr><td><b><code>N</code> (Tần Suất Lệnh)</b></td><td class="num neg"><b>+0.554 (#2 gây DD)</b></td><td class="num">+0.252</td><td><code>N &lt; 15</code>: thiếu mẫu (loại); <code>N &lt; 25</code>: cấm đi 1.50×–1.75×; <code>N ≥ 60</code> mỏng: loại</td></tr>
            <tr><td><b><code>SL_U</code> (5 / 6 / 7U)</b></td><td class="num neg"><b>+0.421 (#3 gây DD)</b></td><td class="num neg">-0.325</td><td>SL 7U dễ kéo DD sâu hơn SL 5U nếu đi cùng hệ số Volume</td></tr>
            <tr><td><b><code>S5 Resilience</code></b></td><td class="num">Kháng trượt giá</td><td class="num pos">Bắt buộc ≥ 68%</td><td>Giữ được ≥ 68% lãi khi cộng thêm trượt giá xấu +0.10U cả chiều vào và SL</td></tr>
          </tbody>
        </table>
      </div>
    </div>

    <div class="section-card">
      <div class="section-head">
        <div>
          <h3>🎯 3. Đường Biên Điểm Ngọt Bao Giá (400G / 350G / 320G / 250G · Khóa Cứng Volume Bậc S2)</h3>
          <p class="section-sub">Định luật tuyến tính: <code>MaxDD ≈ 11.56% × (400 / Bao_Giá)</code> — chọn đúng Bao Giá khớp khẩu vị rủi ro của bạn</p>
        </div>
      </div>
      <div class="tbl-wrap">
        <table>
          <thead><tr><th>Mức Bao Giá</th><th>Thua 1 Combo 5 SL (30U) tại 125% → Còn</th><th>S4 Vốn Cuối</th><th>S4 ROI</th><th>S4 MaxDD</th><th>S4 Calmar</th><th>S4 PF</th><th>S5 Vốn Cuối</th><th>S5 MaxDD</th></tr></thead>
          <tbody id="frontier-tbody"></tbody>
        </table>
      </div>
    </div>

    <div class="section-card">
      <div class="section-head">
        <div>
          <h3>💡 4. Năm Bài Học Xương Máu (Cẩm Nang Chọn Setup &amp; Đi Volume Chuẩn Quỹ)</h3>
          <p class="section-sub">Đúc kết từ toàn bộ quá trình kiểm toán CHICKTRADE v1 → v6.0</p>
        </div>
      </div>
      <div class="lesson-box">
        <ul style="margin:0;padding-left:18px;">
          <li><b>Bài học 1 — Nhìn cặp <code>EPR + PWR</code>, không nhìn mỗi Winrate lệnh:</b> Kèo <code>WR 49%</code> nhưng <code>EPR ≥ 0.28</code> vẫn là siêu phẩm <code>1.75×</code>; ngược lại kèo lãi to nhờ cày 175 lệnh mỏng sẽ làm nghẽn trần 5 lệnh và phá hỏng toàn rổ.</li>
          <li><b>Bài học 2 — Cấm đi Volume lớn khi mẫu nhỏ (<code>N &lt; 25</code>):</b> Dù PWR 75% nhưng dưới 25 lệnh/3 năm vẫn chỉ đi tối đa <code>0.75×–1.00×</code> để tránh xác suất ảo.</li>
          <li><b>Bài học 3 — Chiến lược Tạ Đòn (Barbell) luôn thắng dàn đều:</b> Giữ Volume tĩnh trung bình chỉ <code>0.82×</code> nhưng dồn <code>1.50×–1.75×</code> vào nhóm siêu phẩm và ép kèo SL 7U về <code>0.50×</code> giúp <b>tăng gấp đôi lợi nhuận mà DD lại giảm</b>.</li>
          <li><b>Bài học 4 — Không "giết vĩnh viễn" setup gặp chu kỳ xấu:</b> Đưa vào <b>Ghế Dự Bị (<code>BENCH_WATCH</code>)</b> đánh nháp <code>0.0×</code> để ghi nhận lệnh tương lai; khi 6 lệnh nháp hồi sinh (<code>Net ≥ +6U, PWR ≥ 58%, PF ≥ 1.35</code>) sẽ mở lại thử việc <code>0.50×</code> trong 5 lệnh rồi mới trả về đánh chính.</li>
          <li><b>Bài học 5 — Tách biệt bài toán Volume (hiệu suất) và Bao Giá (khẩu vị DD):</b> Volume V3 Barbell đã đẩy PF lên <code>1.66</code>; còn muốn DD <code>~11.5%</code> chạy <b>400G</b>, muốn DD <code>~13–14%</code> chạy <b>320G–350G</b>, muốn lãi cực đại <code>$618k</code> chấp nhận DD <code>-19%</code> chạy <b>Turbo 250G</b>.</li>
        </ul>
      </div>
    </div>
  </section>

  <!-- COIN TAB -->
  <section id="tab-coin" class="tab-panel">
    <div class="grid-4" id="coin-kpis"></div>
    <div class="section-card">
      <div class="section-head">
        <div>
          <h3 id="coin-tbl-title">📋 Danh Sách Setup v6.0 Đã Chốt</h3>
          <p class="section-sub">Bao gồm Recovery Factor (<code>Net_U / MaxDD_U</code>), MaxDD riêng, Max SL liên tiếp và độ bền trượt giá S5</p>
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
              <th>Khung</th><th>Setup</th><th>Hạng Volume V3</th><th>Điểm Ngọt (SL / TP1 / TP2 · Kiểu)</th>
              <th>N</th><th>WR</th><th>PWR</th><th>PF</th><th>EPR</th><th>MaxDD(U)</th><th>MaxSL</th><th>RF</th><th>Bền S5%</th><th>Net S4</th><th>Net S5</th>
            </tr>
          </thead>
          <tbody id="coin-tbody"></tbody>
        </table>
      </div>
    </div>
    <div class="section-card" id="coin-bench-card" style="display:none;border-left:4px solid #EAB308;">
      <div class="section-head">
        <div>
          <h3>🪑 Ghế Dự Bị Theo Dõi Lệnh Nháp (BENCH_WATCH · Volume = 0.00×)</h3>
          <p class="section-sub">Tạm dừng vào tiền thật, tự động ghi nhận lệnh nháp tương lai — mở lại Thử Việc 0.50× khi 6 lệnh nháp đạt Net ≥ +6U, PWR ≥ 58%, PF ≥ 1.35</p>
        </div>
      </div>
      <div class="tbl-wrap">
        <table>
          <thead><tr><th>Khung</th><th>Setup</th><th>Trạng Thái</th><th>Điểm Ngọt</th><th>N (3Y)</th><th>PWR (3Y)</th><th>Net S4 (3Y)</th><th>Lý Do Tạm Dừng</th></tr></thead>
          <tbody id="coin-bench-tbody"></tbody>
        </table>
      </div>
    </div>
  </section>

  <!-- MECHANISM, S2 & BENCH-WATCH TAB -->
  <section id="tab-mech" class="tab-panel">
    <div class="section-card">
      <div class="section-head">
        <div>
          <h3>🧮 Máy Tính Bậc Lãi Kép S2 Không Giới Hạn (Bậc 0 → 200 · 6 Nấc V3 Barbell)</h3>
          <p class="section-sub">Volume khóa cứng trong từng bậc S2 (`+8%` lên bậc / `-10%` lùi bậc) — không cắt nhỏ lắt nhắt theo equity</p>
        </div>
      </div>
      <div class="calc-controls">
        <div class="calc-field"><label>Vốn gốc ($)</label><input type="number" id="s2-base" value="10000" min="500" step="1000" oninput="renderS2()"></div>
        <div class="calc-field"><label>Bậc S2 (0 → 200)</label><input type="number" id="s2-step" value="0" min="0" max="200" step="1" oninput="renderS2()"></div>
        <div class="calc-field"><label>Nấc Volume V3 Barbell</label>
          <select id="s2-vol" onchange="renderS2()">
            <option value="1.75">1.75× (S++ · PWR ≥ 69% &amp; EPR ≥ 0.28)</option>
            <option value="1.5">1.50× (S+ · PWR ≥ 65.5% &amp; EPR ≥ 0.22)</option>
            <option value="1.0" selected>1.00× (A · PWR ≥ 61.5% &amp; EPR ≥ 0.17)</option>
            <option value="0.75">0.75× (B · PWR ≥ 58.5% &amp; EPR ≥ 0.13)</option>
            <option value="0.5">0.50× (C · EPR ≥ 0.10)</option>
            <option value="0.25">0.25× (D · SL rộng / Hạ lạnh)</option>
          </select>
        </div>
      </div>
      <div class="grid-4" id="s2-out"></div>
    </div>

    <div class="section-card">
      <div class="section-head">
        <div>
          <h3>🔄 Vòng Đời Tự Học Chậm &amp; Ghế Dự Bị Hồi Sinh (`core/live_learning.py`)</h3>
          <p class="section-sub">Ghi nhận 100% lịch sử chạy thật vào <code>core/live_learning_state.json</code> để thích nghi dài hạn mà không bị nhiễu ngắn hạn</p>
        </div>
      </div>
      <div class="grid-3">
        <div class="mech-card" style="border-top-color:#059669;">
          <h4>1. ACTIVE (68 Setup · Đánh Tiền Thật)</h4>
          <p>Khởi đầu bằng ký ức 3 năm (<code>prior = 60</code>). Sau mỗi lệnh thật đóng, cập nhật <code>PWR_live</code> &amp; <code>EPR_live</code> (tối thiểu <code>15</code> lệnh thật mới đổi bậc). Nếu <code>10</code> lệnh gần nhất lỗ <code>≤ -12U</code> &amp; <code>PF &lt; 0.75</code> → chuyển sang <code>BENCH_WATCH</code>.</p>
        </div>
        <div class="mech-card" style="border-top-color:#D97706;">
          <h4>2. BENCH_WATCH (Ghế Dự Bị · 0.00×)</h4>
          <p>Tạm dừng vào tiền thật nhưng vẫn ghi nhận 100% lệnh nháp tương lai. Khi <code>6</code> lệnh nháp gần nhất đạt <b>Net ≥ +6U, PWR ≥ 58%, PF ≥ 1.35</b> và lệnh cuối thắng → tự động mở khóa lên <code>PROBATION</code>.</p>
        </div>
        <div class="mech-card" style="border-top-color:#0284C7;">
          <h4>3. PROBATION (Thử Việc · Trần 0.50×)</h4>
          <p>Đánh thật nhẹ tối đa <code>0.50×</code> trong <code>5</code> lệnh đầu. Nếu 5 lệnh kết thúc có lãi (<code>Net &gt; 0</code>) → tốt nghiệp trả về đúng bậc <code>ACTIVE</code>; nếu tái lỗ <code>≤ -8U</code> → quay lại <code>BENCH_WATCH</code> ngay.</p>
        </div>
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

function switchTab(id, btn) {{
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  if (btn) btn.classList.add('active');
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  if (id === 'tab-overview' || id === 'tab-framework' || id === 'tab-mech') {{
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
  if (v >= 1.75) return 'b-v175';
  if (v >= 1.5) return 'b-v150';
  if (v >= 1.0) return 'b-v100';
  if (v >= 0.75) return 'b-v075';
  if (v >= 0.5) return 'b-v050';
  return 'b-v025';
}}

function renderDualChartYearly() {{
  const p4 = scMode === 's4' ? D.portfolio.s4_400g : D.portfolio.s5_400g;
  const pt = scMode === 's4' ? D.portfolio.s4_turbo : D.portfolio.s5_turbo;
  const b4 = scMode === 's4' ? D.baseline_6c.s4_400g : D.baseline_6c.s5_400g;
  const bt = scMode === 's4' ? D.baseline_6c.s4_turbo : D.baseline_6c.s5_turbo;
  const lbl = scMode === 's4' ? 'S4 Binance-Exact' : 'S5 Trượt Xấu Nhất';

  document.getElementById('dual-cards').innerHTML = `
    <div class="mode-card" style="border-top:4px solid #059669;">
      <h3><span>🛡️ Bao 400G · ${{lbl}}</span><span class="badge b-v150">An Toàn · S2 Khóa Cứng</span></h3>
      <div style="font-size:12.3px;color:#475569;">68 Active + 1 Bench · Hạ lạnh ${{p4.cool_triggers}} lần · Funding $${{Math.round(p4.funding_usd).toLocaleString()}} · Max mở ${{p4.telemetry.max_concurrent_seen}}/5</div>
      <div class="mode-metrics">
        <div class="m-item"><span>Vốn Cuối 36M ($10k →)</span><strong class="pos">${{fmtUsd(p4.final_eq)}}</strong><small>${{fmtSign(p4.roi_pct, '%')}} · Bậc ${{p4.max_step}}</small></div>
        <div class="m-item"><span>MaxDD / Calmar</span><strong class="neg">${{p4.max_dd_pct}}%</strong><small>Calmar <b>${{p4.calmar}}</b> · ${{p4.trades}} lệnh</small></div>
        <div class="m-item"><span>WR / Thắng Tiền (PWR)</span><strong>${{p4.wr}}% / <span class="pos">${{p4.pwr}}%</span></strong><small>PF <b>${{p4.pf}}</b> · Vol TB ${{p4.avg_mult}}×</small></div>
      </div>
    </div>
    <div class="mode-card" style="border-top:4px solid #D97706;">
      <h3><span>⚡ Auto-Turbo 250G · ${{lbl}}</span><span class="badge b-v175">Bật +25% · Giáp 110%</span></h3>
      <div style="font-size:12.3px;color:#475569;">Bật từ $12,500 · Giáp 110% về 400G · Max mở ${{pt.telemetry.max_concurrent_seen}}/5 · Funding $${{Math.round(pt.funding_usd).toLocaleString()}}</div>
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
    ['7 Coin v6.0 (68 setup · +NEAR)', 'Bao 400G', p4],
    ['6 Coin Gốc (53 setup)', 'Auto-Turbo 250G', bt],
    ['7 Coin v6.0 (68 setup · +NEAR)', 'Auto-Turbo 250G', pt],
  ];
  document.getElementById('compare-tbody').innerHTML = cmpRows.map(([name, mode, x]) => {{
    const hi = name.startsWith('7 Coin') ? 'style="background:#F0FDF4;font-weight:700;"' : '';
    return `<tr ${{hi}}><td>${{name}}</td><td>${{mode}}</td><td class="num pos">${{fmtUsd(x.final_eq)}}</td><td class="num pos">${{fmtSign(x.roi_pct, '%')}}</td><td class="num neg">${{x.max_dd_pct}}%</td><td class="num"><b>${{x.calmar}}</b></td><td class="num">${{x.trades}}</td><td class="num">${{x.wr}}%</td><td class="num pos">${{x.pwr}}%</td><td class="num"><b>${{x.pf}}</b></td><td class="num">$${{Math.round(x.funding_usd).toLocaleString()}}</td></tr>`;
  }}).join('');

  Plotly.newPlot('eq-chart', [
    {{ x: p4.curve.map(c=>c.dt), y: p4.curve.map(c=>c.eq), name: `7 Coin v6.0 · Bao 400G (${{lbl}})`, type: 'scatter', mode: 'lines', fill: 'tozeroy', fillcolor: 'rgba(5,150,105,0.08)', line: {{ color: '#059669', width: 2.6 }} }},
    {{ x: pt.curve.map(c=>c.dt), y: pt.curve.map(c=>c.eq), name: `7 Coin v6.0 · Auto-Turbo 250G (${{lbl}})`, type: 'scatter', mode: 'lines', line: {{ color: '#D97706', width: 2.3, dash: 'dot' }} }}
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
    return `<tr><td class="num"><b>${{r.year}}</b></td><td class="num">${{r.trades}}</td><td class="num">${{r.wr}}%</td><td class="num"><b class="${{r.pwr>=55?'pos':''}}">${{r.pwr}}%</b></td><td class="num"><b>${{r.pf}}</b></td><td class="num ${{r.pnl_usd>=0?'pos':'neg'}}">${{r.pnl_usd>=0?'+':''}}${{fmtUsd(r.pnl_usd)}}</td><td class="num"><b>${{fmtUsd(r.end_eq)}}</b></td><td class="num ${{t.pnl_usd>=0?'pos':'neg'}}">${{t.pnl_usd>=0?'+':''}}${{fmtUsd(t.pnl_usd)}}</td><td class="num"><b>${{fmtUsd(t.end_eq)}}</b></td></tr>`;
  }}).join('');
}}

function renderOverview() {{
  renderDualChartYearly();
  const tiers = [
    ['S++ (1.75×)', 'PWR≥69% · EPR≥0.28', '#B45309'],
    ['S+ (1.50×)', 'PWR≥65.5% · EPR≥0.22', '#059669'],
    ['A (1.00×)', 'PWR≥61.5% · EPR≥0.17', '#0284C7'],
    ['B (0.75×)', 'PWR≥58.5% · EPR≥0.13', '#6D28D9'],
    ['C (0.50×)', 'EPR ≥ 0.10', '#475569'],
    ['D (0.25×)', 'SL rộng / Hạ lạnh', '#DC2626'],
  ];
  document.getElementById('vol-dist-grid').innerHTML = tiers.map(([k, desc, col]) => `
    <div class="m-item" style="background:#FFFFFF;border-top:3px solid ${{col}};">
      <span>${{k}}</span><strong style="color:${{col}};">${{D.tier_dist[k] || 0}} setup</strong><small>${{desc}}</small>
    </div>
  `).join('');

  const syms = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'LINKUSDT', 'XRPUSDT', 'SOLUSDT', 'NEARUSDT'];
  document.getElementById('coin-sum-tbody').innerHTML = syms.map(sym => {{
    const c = D.coins[sym];
    const isNew = (sym === 'NEARUSDT') ? '<span class="new-tag">MỚI</span>' : '';
    const bTag = c.bench_count > 0 ? `<span class="badge b-bench">${{c.bench_count}}</span>` : '0';
    return `<tr style="cursor:pointer;" onclick="jumpToCoin('${{sym}}')"><td class="num"><b>${{sym}}</b>${{isNew}}</td><td class="num"><b>${{c.count}}</b></td><td class="num">${{bTag}}</td><td class="num" style="color:#D97706;font-weight:700;">${{c.must_count}}</td><td class="num"><b>${{c.avg_vol}}×</b></td><td class="num pos">+${{c.tot_net_s4}}U</td><td class="num">$${{(c.notional_cap/1000).toFixed(0)}}k</td><td class="num">${{c.slip_bps}} bps</td><td class="num">${{c.u_avg['15m']}}%</td><td class="num">${{c.u_avg['30m']}}%</td><td class="num">${{c.u_avg['1h']}}%</td></tr>`;
  }}).join('');

  document.getElementById('frontier-tbody').innerHTML = D.buffer_frontier.map(f => `
    <tr>
      <td><b>${{f.buf}}</b></td><td class="num pos">${{f.rem_5sl_125}}</td>
      <td class="num pos">${{fmtUsd(f.s4.final_eq)}}</td><td class="num pos">${{fmtSign(f.s4.roi_pct, '%')}}</td>
      <td class="num neg">${{f.s4.max_dd_pct}}%</td><td class="num"><b>${{f.s4.calmar}}</b></td><td class="num"><b>${{f.s4.pf}}</b></td>
      <td class="num pos">${{fmtUsd(f.s5.final_eq)}}</td><td class="num neg">${{f.s5.max_dd_pct}}%</td>
    </tr>
  `).join('');
}}

function setCoinTf(tf, btn) {{
  coinTf = tf;
  document.querySelectorAll('.c-tf').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  renderCoinTable(activeCoin);
}}

function renderCoinPanel(sym) {{
  const c = D.coins[sym];
  document.getElementById('coin-tbl-title').textContent = `📋 ${{sym}} — ${{c.count}} Setup Active (CHICKTRADE v6.0)`;
  document.getElementById('coin-kpis').innerHTML = `
    <div class="m-item" style="background:#FFFFFF;border-top:3px solid #059669;"><span>Active / Bench (${{sym}})</span><strong class="pos">${{c.count}} / ${{c.bench_count}}</strong><small>${{c.must_count}} siêu phẩm ≥1.50× · Trần 20× $${{(c.notional_cap/1000).toFixed(0)}}k</small></div>
    <div class="m-item" style="background:#FFFFFF;border-top:3px solid #D97706;"><span>Hệ Số Volume TB</span><strong>${{c.avg_vol}}×</strong><small>V3 Barbell · Slip ${{c.slip_bps}} bps</small></div>
    <div class="m-item" style="background:#FFFFFF;border-top:3px solid #0284C7;"><span>Tổng Net S4 (36M)</span><strong class="pos">+${{c.tot_net_s4}}U</strong><small>Đã trừ Taker + IOC thật + Funding 8h</small></div>
    <div class="m-item" style="background:#FFFFFF;border-top:3px solid #6D28D9;"><span>1U TB (15m / 30m / 1h)</span><strong style="font-size:13.5px;">${{c.u_avg['15m']}}% / ${{c.u_avg['30m']}}% / ${{c.u_avg['1h']}}%</strong><small>Thước đo 4H Macro ATR</small></div>
  `;
  renderCoinTable(sym);

  const bCard = document.getElementById('coin-bench-card');
  if (c.bench_setups && c.bench_setups.length > 0) {{
    bCard.style.display = 'block';
    document.getElementById('coin-bench-tbody').innerHTML = c.bench_setups.map(b => `
      <tr>
        <td class="num"><b>${{b.tf.toUpperCase()}}</b></td><td class="num"><b>${{b.setup}}</b></td>
        <td><span class="badge b-bench">BENCH_WATCH (0.00×)</span></td>
        <td class="num">-${{b.sl_u}} / +${{b.tp1_u}} / +${{b.tp2_u}}U · ${{b.mode}}</td>
        <td class="num">${{b.s4.n}}</td><td class="num">${{b.s4.pwr}}%</td><td class="num">+${{b.s4.net_u}}U</td>
        <td style="white-space:normal;">${{b.bench_reason}}</td>
      </tr>
    `).join('');
  }} else {{
    bCard.style.display = 'none';
  }}
}}

function renderCoinTable(sym) {{
  const c = D.coins[sym];
  const rows = c.setups.filter(r => coinTf === 'ALL' || r.tf === coinTf);
  document.getElementById('coin-tbody').innerHTML = rows.map(r => {{
    const s4 = r.s4, s5 = r.s5;
    const must = r.is_must ? '<span class="must-badge">SIÊU PHẨM</span>' : '';
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
        <td class="num"><b>${{r.epr}}</b></td>
        <td class="num neg">-${{r.mdd_u}}U</td>
        <td class="num">${{r.max_csl}}</td>
        <td class="num pos"><b>${{r.rf}}</b></td>
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
    print(f"[OK] Built CHICKTRADE v6.0 MASTER HTML: {OUT_HTML} ({len(html):,} bytes)")


if __name__ == "__main__":
    main()
