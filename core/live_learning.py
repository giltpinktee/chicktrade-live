"""CHICKTRADE v5.1 — Slow Bayesian Online Volume Learner & Shadow Bench-Watch Lifecycle.

3-State Setup Lifecycle:
  1. ACTIVE: Trades real capital using V3 Barbell Alpha (0.25x -> 1.75x) + Cold-Only Cooling.
     - Automatically demotes to BENCH_WATCH if rolling last 10 real trades bleed deeply
       (rolling_10_net_u <= -12.0U and rolling_10_pf < 0.75).
  2. BENCH_WATCH (Shadow Probation): Takes ZERO real capital (vol_mult = 0.0), but continues
     tracking every hypothetical future signal (`record_shadow_trade`) to monitor recovery.
     - Automatically promotes to PROBATION when the last 6 shadow trades prove regime recovery:
       (shadow_6_net_u >= +6.0U, shadow_6_pwr >= 58.0%, shadow_6_pf >= 1.35, last trade > 0).
  3. PROBATION: Resumes real trading capped at 0.50x for 5 real trades.
     - If 5 real probation trades finish Net > 0 -> returns to full ACTIVE V3 Barbell tier.
     - If probation bleeds <= -8.0U -> returns to BENCH_WATCH immediately.
"""
from datetime import datetime, timezone
import json
import os
from typing import Dict, List, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RULES_PATH = os.path.join(ROOT, "core", "core_strategy_rules.json")
STATE_PATH = os.path.join(ROOT, "core", "live_learning_state.json")

PRIOR_WEIGHT = 60.0
MIN_LIVE_TRADES = 15

# Lifecycle thresholds
BENCH_ROLLING_N = 10
BENCH_DEMOTE_NET_U = -12.0
BENCH_DEMOTE_PF = 0.75

SHADOW_WINDOW_N = 6
SHADOW_PROMOTE_NET_U = 6.0
SHADOW_PROMOTE_PWR = 58.0
SHADOW_PROMOTE_PF = 1.35

PROBATION_TRADES_REQUIRED = 5
PROBATION_VOL_CAP = 0.50
PROBATION_FAIL_NET_U = -8.0


def v3_barbell_tier(pwr: float, epr: float, n: int) -> Tuple[float, str]:
    """V3 Barbell Alpha 6-tier Volume (0.25x -> 1.75x by PWR + Edge-per-Risk EPR)."""
    if pwr >= 69.0 and epr >= 0.28 and n >= 25:
        return 1.75, "S++ (1.75×)"
    if pwr >= 65.5 and epr >= 0.22 and n >= 25:
        return 1.50, "S+ (1.50×)"
    if pwr >= 61.5 and epr >= 0.17 and n >= 20:
        return 1.00, "A (1.00×)"
    if pwr >= 58.5 and epr >= 0.13:
        return 0.75, "B (0.75×)"
    if epr >= 0.10:
        return 0.50, "C (0.50×)"
    return 0.25, "D (0.25×)"


def _setup_key(coin: str, tf: str, setup_code: str) -> str:
    return f"{coin}|{tf}|{setup_code}"


def _window_stats(pnls: List[float]) -> Tuple[float, float, float]:
    """Return (net_u, pwr_pct, pf) over a list of trade PnLs in U."""
    if not pnls:
        return 0.0, 0.0, 0.0
    gp = sum(x for x in pnls if x > 0)
    gl = sum(abs(x) for x in pnls if x <= 0)
    net = gp - gl
    pwr = (gp / max(gp + gl, 1e-9)) * 100.0
    pf = (gp / gl) if gl > 1e-9 else (9.99 if gp > 0 else 0.0)
    return round(net, 3), round(pwr, 1), round(pf, 2)


def _new_setup_entry(
    sl_u: float,
    pwr: float,
    avg_u: float,
    init_vm: float,
    tier: str,
    lifecycle: str = "ACTIVE",
    bench_reason: str = "",
) -> dict:
    p_frac = max(0.05, min(0.95, pwr / 100.0))
    avg_mag = abs(avg_u) / max(abs(2.0 * p_frac - 1.0), 0.15)
    return {
        "lifecycle": lifecycle,  # ACTIVE | BENCH_WATCH | PROBATION
        "bench_reason": bench_reason,
        "prior_n": PRIOR_WEIGHT,
        "prior_gp": round(PRIOR_WEIGHT * p_frac * avg_mag, 4),
        "prior_gl": round(PRIOR_WEIGHT * (1.0 - p_frac) * avg_mag, 4),
        "prior_net_u": round(PRIOR_WEIGHT * avg_u, 4),
        "sl_u": sl_u,
        "live_n": 0,
        "live_gp": 0.0,
        "live_gl": 0.0,
        "live_net_u": 0.0,
        "recent_live_u": [],
        "shadow_n": 0,
        "shadow_recent_u": [],
        "probation_n": 0,
        "probation_net_u": 0.0,
        "static_vol_mult": init_vm,
        "learned_vol_mult": init_vm,
        "learned_tier": tier,
        "updated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    }


def load_or_init_learning_state(rules_path: str = RULES_PATH, state_path: str = STATE_PATH) -> Dict[str, dict]:
    """Load persisted live learning state, seeding active setups & initial BENCH_WATCH setups."""
    state: Dict[str, dict] = {
        "version": "v5.1-slow-bayesian-benchwatch",
        "prior_weight": PRIOR_WEIGHT,
        "min_live_trades": MIN_LIVE_TRADES,
        "setups": {},
    }
    if os.path.exists(state_path):
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict) and isinstance(loaded.get("setups"), dict):
                state = loaded
        except Exception:
            pass

    if os.path.exists(rules_path):
        with open(rules_path, "r", encoding="utf-8") as f:
            rules_doc = json.load(f)
        for r in rules_doc.get("rules", []):
            k = _setup_key(r["coin"], r["tf"], r["setup_code"])
            if k not in state["setups"]:
                pr = r.get("prior_1y", {})
                state["setups"][k] = _new_setup_entry(
                    sl_u=float(r.get("sl_u", 6.0)),
                    pwr=float(pr.get("profit_wr_pct", 58.0)),
                    avg_u=float(pr.get("avg_u", 0.7)),
                    init_vm=float(r.get("adaptive_volume", {}).get("init_vol_mult", 0.75)),
                    tier=r.get("v5_tier", "B (0.75×)"),
                    lifecycle="ACTIVE",
                )
            else:
                # Ensure newly added lifecycle fields exist on older state files
                s = state["setups"][k]
                s.setdefault("lifecycle", "ACTIVE")
                s.setdefault("bench_reason", "")
                s.setdefault("recent_live_u", [])
                s.setdefault("shadow_n", 0)
                s.setdefault("shadow_recent_u", [])
                s.setdefault("probation_n", 0)
                s.setdefault("probation_net_u", 0.0)

    # Seed Y3-demoted NEAR setup into BENCH_WATCH instead of deleting it forever
    bench_near_k = _setup_key("NEARUSDT", "1h", "EMA34_W4_REV")
    if bench_near_k not in state["setups"]:
        state["setups"][bench_near_k] = _new_setup_entry(
            sl_u=6.0,
            pwr=57.5,
            avg_u=0.85,
            init_vm=0.50,
            tier="C (0.50×)",
            lifecycle="BENCH_WATCH",
            bench_reason="Y3 regime bleed (-21.2U, PF 0.27) — tracking shadow trades for reactivation",
        )
    return state


def save_learning_state(state: Dict[str, dict], state_path: str = STATE_PATH) -> None:
    state["version"] = "v5.1-slow-bayesian-benchwatch"
    state["updated_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    os.makedirs(os.path.dirname(state_path), exist_ok=True)
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def get_effective_volume(state: Dict[str, dict], coin: str, tf: str, setup_code: str, fallback_vol: float = 0.75) -> Tuple[float, str]:
    """Return (vol_mult, lifecycle). Returns 0.0 when setup is in BENCH_WATCH."""
    s = state.get("setups", {}).get(_setup_key(coin, tf, setup_code))
    if s is None:
        return fallback_vol, "ACTIVE"
    lc = s.get("lifecycle", "ACTIVE")
    if lc == "BENCH_WATCH":
        return 0.0, "BENCH_WATCH"
    vm = float(s.get("learned_vol_mult", fallback_vol))
    if lc == "PROBATION":
        return min(vm, PROBATION_VOL_CAP), "PROBATION"
    return vm, "ACTIVE"


def record_shadow_trade(state: Dict[str, dict], coin: str, tf: str, setup_code: str, pnl_u: float) -> Tuple[str, dict]:
    """Record a hypothetical (non-executed) trade for a BENCH_WATCH setup and check for reactivation."""
    k = _setup_key(coin, tf, setup_code)
    s = state.get("setups", {}).get(k)
    if s is None:
        return "UNKNOWN", {}

    s["shadow_n"] = int(s.get("shadow_n", 0)) + 1
    hist: List[float] = list(s.get("shadow_recent_u", []))
    hist.append(round(float(pnl_u), 4))
    hist = hist[-SHADOW_WINDOW_N:]
    s["shadow_recent_u"] = hist
    net_u, pwr, pf = _window_stats(hist)
    s["shadow_window_stats"] = {"n": len(hist), "net_u": net_u, "pwr": pwr, "pf": pf}
    s["updated_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    if (
        s.get("lifecycle") == "BENCH_WATCH"
        and len(hist) >= SHADOW_WINDOW_N
        and net_u >= SHADOW_PROMOTE_NET_U
        and pwr >= SHADOW_PROMOTE_PWR
        and pf >= SHADOW_PROMOTE_PF
        and pnl_u > 0
    ):
        s["lifecycle"] = "PROBATION"
        s["probation_n"] = 0
        s["probation_net_u"] = 0.0
        s["bench_reason"] = f"Reactivated to PROBATION (0.50x) after {SHADOW_WINDOW_N} shadow trades: Net={net_u:+.1f}U, PWR={pwr}%, PF={pf}"
    return s["lifecycle"], s["shadow_window_stats"]


def record_closed_trade(
    state: Dict[str, dict],
    coin: str,
    tf: str,
    setup_code: str,
    pnl_u: float,
    sl_u: float = 6.0,
    static_vol_mult: float = 0.75,
) -> Tuple[float, str, str]:
    """Update a setup's live Bayesian history with a real closed trade. Returns (vol_mult, tier, lifecycle)."""
    k = _setup_key(coin, tf, setup_code)
    s = state.setdefault("setups", {}).get(k)
    if s is None:
        s = _new_setup_entry(sl_u, 58.0, 0.6, static_vol_mult, "B (0.75×)", "ACTIVE")
        state["setups"][k] = s

    s["live_n"] = int(s.get("live_n", 0)) + 1
    s["live_net_u"] = round(float(s.get("live_net_u", 0.0)) + float(pnl_u), 4)
    if pnl_u > 0:
        s["live_gp"] = round(float(s.get("live_gp", 0.0)) + float(pnl_u), 4)
    else:
        s["live_gl"] = round(float(s.get("live_gl", 0.0)) + abs(float(pnl_u)), 4)

    rec: List[float] = list(s.get("recent_live_u", []))
    rec.append(round(float(pnl_u), 4))
    rec = rec[-BENCH_ROLLING_N:]
    s["recent_live_u"] = rec

    # Slow Bayesian tier update
    tot_n = float(s["prior_n"]) + float(s["live_n"])
    tot_gp = float(s["prior_gp"]) + float(s["live_gp"])
    tot_gl = float(s["prior_gl"]) + float(s["live_gl"])
    tot_net = float(s["prior_net_u"]) + float(s["live_net_u"])
    pwr_live = (tot_gp / max(tot_gp + tot_gl, 1e-9)) * 100.0
    epr_live = (tot_net / max(tot_n, 1.0)) / max(float(s["sl_u"]), 1.0)
    if s["live_n"] >= int(state.get("min_live_trades", MIN_LIVE_TRADES)):
        vm, tier = v3_barbell_tier(pwr_live, epr_live, int(tot_n))
        s["learned_vol_mult"] = vm
        s["learned_tier"] = tier
    s["effective_pwr"] = round(pwr_live, 2)
    s["effective_epr"] = round(epr_live, 3)

    # Lifecycle transitions
    lc = s.get("lifecycle", "ACTIVE")
    if lc == "PROBATION":
        s["probation_n"] = int(s.get("probation_n", 0)) + 1
        s["probation_net_u"] = round(float(s.get("probation_net_u", 0.0)) + float(pnl_u), 4)
        if s["probation_net_u"] <= PROBATION_FAIL_NET_U:
            s["lifecycle"] = "BENCH_WATCH"
            s["shadow_recent_u"] = []
            s["bench_reason"] = f"Probation failed ({s['probation_net_u']:+.1f}U) — returned to BENCH_WATCH"
        elif s["probation_n"] >= PROBATION_TRADES_REQUIRED and s["probation_net_u"] > 0:
            s["lifecycle"] = "ACTIVE"
            s["bench_reason"] = f"Graduated PROBATION ({s['probation_n']} trades, {s['probation_net_u']:+.1f}U) -> ACTIVE"
    elif lc == "ACTIVE" and len(rec) >= BENCH_ROLLING_N:
        r_net, r_pwr, r_pf = _window_stats(rec)
        if r_net <= BENCH_DEMOTE_NET_U and r_pf < BENCH_DEMOTE_PF:
            s["lifecycle"] = "BENCH_WATCH"
            s["shadow_recent_u"] = []
            s["bench_reason"] = f"Auto-benched on rolling {BENCH_ROLLING_N} live trades (Net={r_net:+.1f}U, PF={r_pf})"

    s["updated_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    eff_vm, eff_lc = get_effective_volume(state, coin, tf, setup_code, float(s["learned_vol_mult"]))
    return eff_vm, str(s["learned_tier"]), eff_lc
