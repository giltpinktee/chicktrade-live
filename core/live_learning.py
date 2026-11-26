"""CHICKTRADE v5.1 — Slow Bayesian Online Volume Learner.

Starts from each setup's 3-year Gold-Standard prior (2023-2026) and continuously records
every closed real/paper trade into `core/live_learning_state.json`.
Uses a heavy prior weight (prior_weight=60) and requires at least `min_live_trades=15`
before allowing a V3 Barbell volume tier change — preventing short-term whipsaw while
automatically adapting to genuine multi-month regime shifts.
"""
from datetime import datetime, timezone
import json
import os
from typing import Dict, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RULES_PATH = os.path.join(ROOT, "core", "core_strategy_rules.json")
STATE_PATH = os.path.join(ROOT, "core", "live_learning_state.json")

PRIOR_WEIGHT = 60.0
MIN_LIVE_TRADES = 15


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


def load_or_init_learning_state(rules_path: str = RULES_PATH, state_path: str = STATE_PATH) -> Dict[str, dict]:
    """Load persisted live learning state, seeding any missing setups from 3-year prior."""
    state: Dict[str, dict] = {"version": "v5.1-slow-bayesian", "prior_weight": PRIOR_WEIGHT, "min_live_trades": MIN_LIVE_TRADES, "setups": {}}
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
                pwr = float(pr.get("profit_wr_pct", 58.0))
                avg_u = float(pr.get("avg_u", 0.7))
                sl_u = float(r.get("sl_u", 6.0))
                p_frac = max(0.05, min(0.95, pwr / 100.0))
                avg_mag = abs(avg_u) / max(abs(2.0 * p_frac - 1.0), 0.15)
                init_vm = float(r.get("adaptive_volume", {}).get("init_vol_mult", 0.75))
                state["setups"][k] = {
                    "prior_n": PRIOR_WEIGHT,
                    "prior_gp": round(PRIOR_WEIGHT * p_frac * avg_mag, 4),
                    "prior_gl": round(PRIOR_WEIGHT * (1.0 - p_frac) * avg_mag, 4),
                    "prior_net_u": round(PRIOR_WEIGHT * avg_u, 4),
                    "sl_u": sl_u,
                    "live_n": 0,
                    "live_gp": 0.0,
                    "live_gl": 0.0,
                    "live_net_u": 0.0,
                    "static_vol_mult": init_vm,
                    "learned_vol_mult": init_vm,
                    "learned_tier": r.get("v5_tier", "B (0.75×)"),
                    "updated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                }
    return state


def save_learning_state(state: Dict[str, dict], state_path: str = STATE_PATH) -> None:
    state["updated_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    os.makedirs(os.path.dirname(state_path), exist_ok=True)
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def record_closed_trade(
    state: Dict[str, dict],
    coin: str,
    tf: str,
    setup_code: str,
    pnl_u: float,
    sl_u: float = 6.0,
    static_vol_mult: float = 0.75,
) -> Tuple[float, str]:
    """Update a setup's live Bayesian history with a newly closed trade and return its learned (vol_mult, tier)."""
    k = _setup_key(coin, tf, setup_code)
    s = state.setdefault("setups", {}).get(k)
    if s is None:
        s = {
            "prior_n": PRIOR_WEIGHT, "prior_gp": PRIOR_WEIGHT * 0.58 * 2.0, "prior_gl": PRIOR_WEIGHT * 0.42 * 2.0,
            "prior_net_u": PRIOR_WEIGHT * 0.6, "sl_u": sl_u,
            "live_n": 0, "live_gp": 0.0, "live_gl": 0.0, "live_net_u": 0.0,
            "static_vol_mult": static_vol_mult, "learned_vol_mult": static_vol_mult, "learned_tier": "B (0.75×)",
        }
        state["setups"][k] = s

    s["live_n"] = int(s.get("live_n", 0)) + 1
    s["live_net_u"] = round(float(s.get("live_net_u", 0.0)) + float(pnl_u), 4)
    if pnl_u > 0:
        s["live_gp"] = round(float(s.get("live_gp", 0.0)) + float(pnl_u), 4)
    else:
        s["live_gl"] = round(float(s.get("live_gl", 0.0)) + abs(float(pnl_u)), 4)

    tot_n = float(s["prior_n"]) + float(s["live_n"])
    tot_gp = float(s["prior_gp"]) + float(s["live_gp"])
    tot_gl = float(s["prior_gl"]) + float(s["live_gl"])
    tot_net = float(s["prior_net_u"]) + float(s["live_net_u"])
    pwr_live = (tot_gp / max(tot_gp + tot_gl, 1e-9)) * 100.0
    epr_live = (tot_net / max(tot_n, 1.0)) / max(float(s["sl_u"]), 1.0)

    # Guard: require >= MIN_LIVE_TRADES before updating volume tier from static prior
    if s["live_n"] >= int(state.get("min_live_trades", MIN_LIVE_TRADES)):
        vm, tier = v3_barbell_tier(pwr_live, epr_live, int(tot_n))
        s["learned_vol_mult"] = vm
        s["learned_tier"] = tier
    s["effective_pwr"] = round(pwr_live, 2)
    s["effective_epr"] = round(epr_live, 3)
    s["updated_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return float(s["learned_vol_mult"]), str(s["learned_tier"])
