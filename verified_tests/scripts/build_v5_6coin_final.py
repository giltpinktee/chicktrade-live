"""CHICKTRADE v5.0 — Locked 53-Setup (Argon B) + M2 6-Tier Volume + Cold-Only Cooling.

Runs exact 2023-11-01 -> 2026-10-31 Gold-Standard S4_REAL_FUNDING & S5_PESSIMISTIC simulations,
exports locked v5 rules JSON, and builds Master-Style 6-Coin HTML:
  - e:/CHICKTRADE/verified_tests/results/chicktrade_v5_53setups_rules.json
  - e:/CHICKTRADE/core/reports/chicktrade_new_setup_v5_6coin.html
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
    GAMMA_TF,
    TF_PRIORITY,
    MAX_HOLD_BARS,
    MIN_U_PCT,
    load_coin_6y,
    compute_atr14,
    extract_ema_signals_6y,
    sim_trade_scenario,
)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ARGON_JSON = os.path.join(ROOT, "verified_tests", "results", "argon_volume_proposal.json")
OUT_RULES_JSON = os.path.join(ROOT, "verified_tests", "results", "chicktrade_v5_53setups_rules.json")
OUT_HTML = os.path.join(ROOT, "core", "reports", "chicktrade_new_setup_v5_6coin.html")
TFS = ("15m", "30m", "1h")


def m2_vol_tier(pwr: float, n: int) -> Tuple[float, str]:
    """Locked M2 6-tier Volume scale by Profit WR (with N>=25 guard for 1.25x/1.50x)."""
    if pwr >= 70.0 and n >= 25:
        return 1.50, "S++ (1.50×)"
    if pwr >= 65.0 and n >= 25:
        return 1.25, "S+ (1.25×)"
    if pwr >= 61.0 and n >= 20:
        return 1.00, "A (1.00×)"
    if pwr >= 58.0:
        return 0.75, "B (0.75×)"
    if pwr >= 55.0:
        return 0.50, "C (0.50×)"
    return 0.25, "D (0.25×)"


def simulate_v5_portfolio(
    trades: List[dict],
    ioc_missed: int = 0,
    buffer_base: float = 400.0,
    turbo_buffer: float = 250.0,
    use_turbo: bool = False,
    max_same_dir: int = 3,
    init_cap: float = 10_000.0,
) -> dict:
    """Gold-Standard v5 Portfolio Sim: M2 6-tier base + Cold-Only Cooling (never boosts above static cap)."""
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

    # Per-setup consecutive SL tracker for Cold-Only Cooling
    setup_sl_streak: Dict[Tuple[str, str, str], int] = {}
    open_pos: List[dict] = []
    closed: List[dict] = []
    curve: List[dict] = []
    total_funding = 0.0
    yearly: Dict[str, dict] = {}
    cool_triggers = 0

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
                dt_s = datetime.fromtimestamp(p["close_ts"] / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d")
                yr = dt_s[:4]
                yrec = yearly.setdefault(yr, {"trades": 0, "wins": 0, "gp": 0.0, "gl": 0.0, "pnl": 0.0, "end_eq": eq})
                yrec["trades"] += 1
                if p["pnl_usd"] > 0:
                    yrec["wins"] += 1
                    yrec["gp"] += p["pnl_usd"]
                else:
                    yrec["gl"] += abs(p["pnl_usd"])
                yrec["pnl"] += p["pnl_usd"]
                yrec["end_eq"] = round(eq, 2)

                if not curve or curve[-1]["dt"] != dt_s:
                    curve.append({"dt": dt_s, "eq": round(eq, 2), "dd": round(dd, 2)})
                else:
                    curve[-1]["eq"] = round(eq, 2)
                    curve[-1]["dd"] = round(min(curve[-1]["dd"], dd), 2)
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
        eff_m = base_m
        # Cold-Only Cooling: if this setup hit >= 3 consecutive SLs, cut volume to 0.25x-0.50x (never boost above base_m!)
        sk = (t["sym"], t["tf"], t["setup"])
        if setup_sl_streak.get(sk, 0) >= 3:
            eff_m = min(eff_m, 0.25 if base_m <= 0.75 else 0.50)
            cool_triggers += 1

        if t.get("elevated_vol"):
            eff_m *= 0.5
        cur_dd = (eq - peak) / peak * 100.0
        if cur_dd <= -6.0:
            eff_m = min(eff_m, 0.25)
        elif cur_dd <= -3.5:
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
            **t, "base_vol": base_m, "eff_vol": round(eff_m, 3),
            "notional": notional, "pnl_usd": t["pnl_u"] * u_usd,
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

    yr_list = []
    for yr in sorted(yearly.keys()):
        r = yearly[yr]
        yr_list.append({
            "year": yr,
            "trades": r["trades"],
            "wr": round(r["wins"] / max(r["trades"], 1) * 100.0, 1),
            "pwr": round(r["gp"] / max(r["gp"] + r["gl"], 1e-9) * 100.0, 1),
            "pf": round(r["gp"] / max(r["gl"], 1e-9), 2) if r["gl"] > 1e-9 else 9.99,
            "pnl_usd": round(r["pnl"], 2),
            "end_eq": r["end_eq"],
        })

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
        "avg_mult": round(float(np.mean([p["eff_vol"] for p in closed])) if closed else 0.0, 2),
        "cool_triggers": cool_triggers,
        "funding_usd": round(total_funding, 2),
        "yearly": yr_list,
        "curve": curve,
    }


def main():
    with open(ARGON_JSON, encoding="utf-8") as f:
        argon = json.load(f)
    with open(os.path.join(ROOT, "verified_tests", "results", "s4_2023_2026_sweetspot_288.json"), encoding="utf-8") as f:
        s4_base = json.load(f)
    wr_lookup = {}
    for sym_k, c_obj in s4_base.get("coins", {}).items():
        for sr in c_obj.get("setups", []):
            wr_lookup[(sym_k, sr["tf"], sr["setup"])] = float(sr.get("s4", {}).get("wr", 0.0))

    # Extract the 53 locked Argon B setups and assign M2 6-tier Volume
    raw_setups = argon.get("setups", [])
    locked_53 = []
    tier_dist: Dict[str, int] = {}

    for r in raw_setups:
        ab = r.get("argon_b") or {}
        b_tier = ab.get("tier", "DROP")
        if b_tier in (None, "DROP", ""):
            continue
        m = r.get("metrics", {})
        pwr = float(m.get("pwr", 0.0))
        n = int(m.get("n", 0))
        vol_m, m2_lbl = m2_vol_tier(pwr, n)
        tier_dist[m2_lbl] = tier_dist.get(m2_lbl, 0) + 1
        sl_u = float(r["sl_u"])
        avg_u = float(m.get("avg_u", 0.0))
        epr = round(float(m.get("epr", avg_u / max(sl_u, 1.0))), 3)
        s5_res = round(float(m.get("s5_res", 0.0)) * 100.0, 1)
        is_must = vol_m >= 1.25 or b_tier == "S+"
        s4_obj = {
            "n": n,
            "wr": wr_lookup.get((r["sym"], r["tf"], r["setup"]), 0.0),
            "pwr": pwr,
            "pf": float(m.get("pf", 0.0)),
            "avg_u": avg_u,
            "net_u": float(m.get("net_u", 0.0)),
        }
        s5_obj = {
            "pf": float(m.get("s5_pf", 0.0)),
            "net_u": float(m.get("s5_net_u", 0.0)),
        }
        locked_53.append({
            "sym": r["sym"],
            "tf": r["tf"],
            "setup": r["setup"],
            "argon_tier": b_tier,
            "m2_tier": m2_lbl,
            "vol_mult": vol_m,
            "sl_u": sl_u,
            "tp1_u": float(r["tp1_u"]),
            "tp2_u": float(r["tp2_u"]),
            "mode": r["mode"],
            "is_must": is_must,
            "epr": epr,
            "s5_res_pct": s5_res,
            "score": round(vol_m * 40.0 + pwr * 0.6 + s4_obj["net_u"] * 0.3, 2),
            "s4": s4_obj,
            "s5": s5_obj,
        })

    print(f"Loaded {len(locked_53)} locked v5 setups | M2 6-Tier Dist: {tier_dist}")
    assert len(locked_53) == 53, f"Expected 53 Argon B setups, got {len(locked_53)}"

    print("Loading 6-year 5m data + real 8h Binance Funding for 2023-11 -> 2026-10 simulation...")
    market: Dict[str, Dict[str, pd.DataFrame]] = {}
    funding_map: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    with ThreadPoolExecutor(max_workers=6) as pool:
        futs = {sym: pool.submit(load_coin_6y, sym) for sym in SIX_COINS}
        for sym, fut in futs.items():
            m_d, f_ts, f_rt = fut.result()
            market[sym] = m_d
            funding_map[sym] = (f_ts, f_rt)

    sig_map = extract_ema_signals_6y(market, tfs=TFS, max_wave=4)
    t_start = int(datetime(2023, 11, 1, tzinfo=timezone.utc).timestamp() * 1000)
    t_end = int(datetime(2026, 11, 1, tzinfo=timezone.utc).timestamp() * 1000)

    def _run_sc(sc: str) -> Tuple[List[dict], int]:
        trades = []
        missed = 0
        for r in locked_53:
            sym, tf, st = r["sym"], r["tf"], r["setup"]
            sigs = [s for s in sig_map.get((sym, tf, st), []) if t_start <= s["open_ts"] < t_end]
            df = market[sym][tf]
            op, hi, lo, cl, ts = df["open"].values, df["high"].values, df["low"].values, df["close"].values, df["open_time_ms"].values
            fr_ts, fr_rate = funding_map[sym]
            mb = MAX_HOLD_BARS[tf]
            busy = -1
            for s in sigs:
                if s["idx"] <= busy:
                    continue
                res = sim_trade_scenario(op, hi, lo, cl, ts, fr_ts, fr_rate, s, r["sl_u"], r["tp1_u"], r["tp2_u"], r["mode"], mb, sc)
                if res is None:
                    continue
                if res.get("ioc_missed"):
                    missed += 1
                    continue
                busy = s["idx"] + res["bars_held"]
                res["vol_mult"] = r["vol_mult"]
                res["score"] = r["score"]
                trades.append(res)
        return trades, missed

    tr_s4, miss_s4 = _run_sc("S4_REAL_FUNDING")
    tr_s5, miss_s5 = _run_sc("S5_PESSIMISTIC")

    p_s4_400 = simulate_v5_portfolio(tr_s4, miss_s4, 400, 250, False, 3)
    p_s4_tur = simulate_v5_portfolio(tr_s4, miss_s4, 400, 250, True, 3)
    p_s5_400 = simulate_v5_portfolio(tr_s5, miss_s5, 400, 250, False, 3)
    p_s5_tur = simulate_v5_portfolio(tr_s5, miss_s5, 400, 250, True, 3)

    print("\n=== CHICKTRADE NEW SETUP v5 — 6 COIN (53 Setups · M2 6-Tier + Cold-Only Cooling) ===")
    print(f"  [S4 REAL FUNDING] Bao 400G : ${p_s4_400['final_eq']:>10,.2f} ({p_s4_400['roi_pct']:+7.1f}%) | DD={p_s4_400['max_dd_pct']:6.2f}% | Calmar={p_s4_400['calmar']:6.1f} | PF={p_s4_400['pf']} | PWR={p_s4_400['pwr']}% | N={p_s4_400['trades']} | Fund=${p_s4_400['funding_usd']:,.0f}")
    print(f"  [S4 REAL FUNDING] Turbo250G: ${p_s4_tur['final_eq']:>10,.2f} ({p_s4_tur['roi_pct']:+7.1f}%) | DD={p_s4_tur['max_dd_pct']:6.2f}% | Calmar={p_s4_tur['calmar']:6.1f} | PF={p_s4_tur['pf']} | PWR={p_s4_tur['pwr']}% | N={p_s4_tur['trades']}")
    print(f"  [S5 PESSIMISTIC ] Bao 400G : ${p_s5_400['final_eq']:>10,.2f} ({p_s5_400['roi_pct']:+7.1f}%) | DD={p_s5_400['max_dd_pct']:6.2f}% | Calmar={p_s5_400['calmar']:6.1f} | PF={p_s5_400['pf']} | PWR={p_s5_400['pwr']}%")
    print(f"  [S5 PESSIMISTIC ] Turbo250G: ${p_s5_tur['final_eq']:>10,.2f} ({p_s5_tur['roi_pct']:+7.1f}%) | DD={p_s5_tur['max_dd_pct']:6.2f}% | Calmar={p_s5_tur['calmar']:6.1f} | PF={p_s5_tur['pf']} | PWR={p_s5_tur['pwr']}%")

    # Per-coin metadata + 1U avg
    coins_out = {}
    for sym in SIX_COINS:
        df4 = market[sym]["4h"]
        df4_sub = df4[(df4["open_time_ms"] >= t_start) & (df4["open_time_ms"] < t_end)]
        macro4 = ((0.5 * compute_atr14(df4)) / df4["close"]).ewm(span=20, adjust=False).mean().loc[df4_sub.index]
        u_avg = {tf: round(float(max(MIN_U_PCT, macro4.mean() * GAMMA_TF[tf]) * 100.0), 3) for tf in TFS}
        c_list = sorted([r for r in locked_53 if r["sym"] == sym], key=lambda x: (-x["vol_mult"], -x["s4"]["pwr"], -x["s4"]["net_u"]))
        coins_out[sym] = {
            "u_avg": u_avg,
            "count": len(c_list),
            "must_count": sum(1 for r in c_list if r["is_must"]),
            "avg_vol": round(float(np.mean([r["vol_mult"] for r in c_list])) if c_list else 0.0, 2),
            "tot_net_s4": round(sum(r["s4"]["net_u"] for r in c_list), 1),
            "setups": c_list,
        }

    v5_payload = {
        "version": "v5.0-argon-b-53setups-m2-coldcool",
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "period": "2023-11-01 -> 2026-10-31 (36 tháng · Chuẩn S4 Funding 8h Thật + IOC ±0.25U)",
        "total_setups": len(locked_53),
        "tier_dist": tier_dist,
        "portfolio": {
            "s4_400g": p_s4_400,
            "s4_turbo": p_s4_tur,
            "s5_400g": p_s5_400,
            "s5_turbo": p_s5_tur,
        },
        "coins": coins_out,
        "rules": locked_53,
    }
    with open(OUT_RULES_JSON, "w", encoding="utf-8") as f:
        json.dump(v5_payload, f, ensure_ascii=False, indent=2)
    print(f"[OK] Saved locked v5 ruleset JSON: {OUT_RULES_JSON}")

    build_html(v5_payload)


def build_html(data: dict) -> None:
    data_js = json.dumps(data, ensure_ascii=False)
    html = f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CHICKTRADE NEW SETUP v5 — 6 COIN (53 Setup Tinh Hoa · 6 Nấc Volume M2 + Hạ Lạnh · 2023–2026)</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500;600;700&display=swap" rel="stylesheet">
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  :root {{
    --bg: #F8FAFC; --card: #FFFFFF; --border: #E2E8F0; --border-strong: #CBD5E1;
    --text: #0F172A; --text-sec: #475569; --muted: #64748B;
    --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    --font-mono: 'JetBrains Mono', ui-monospace, Consolas, monospace;
  }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; font-family: var(--font-sans); background: var(--bg); color: var(--text); font-size: 13.5px; line-height: 1.48; }}
  .num {{ font-family: var(--font-mono); font-variant-numeric: tabular-nums; }}
  .page-shell {{ max-width: 1350px; margin: 0 auto; padding: 0 20px 60px 20px; }}
  .hero-bar {{ background: #FFFFFF; border-bottom: 1px solid #E2E8F0; box-shadow: 0 1px 2px rgba(15,23,42,0.04); padding: 18px 0 15px 0; }}
  .hero-inner {{ max-width: 1350px; margin: 0 auto; padding: 0 20px; display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 14px; }}
  .hero-title h1 {{ margin: 0; font-size: 20px; font-weight: 700; color: #0F172A; letter-spacing: -0.2px; }}
  .hero-title p {{ margin: 4px 0 0 0; font-size: 13px; color: #475569; }}
  .hero-actions {{ display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }}
  .dict-toggle-btn {{ background: #F8FAFC; border: 1px solid #CBD5E1; color: #0F172A; padding: 7px 14px; border-radius: 8px; cursor: pointer; font-weight: 600; font-size: 12.8px; }}
  .dict-toggle-btn:hover {{ background: #F1F5F9; }}
  .ver-badge {{ background: #ECFDF5; border: 1px solid #A7F3D0; color: #059669; padding: 6px 12px; border-radius: 8px; font-family: var(--font-mono); font-weight: 700; font-size: 12px; }}
  .nav-tabs {{ position: sticky; top: 0; z-index: 100; background: rgba(255,255,255,0.92); backdrop-filter: blur(8px); border-bottom: 1px solid #E2E8F0; padding: 10px 0; margin-bottom: 18px; }}
  .nav-tabs-inner {{ max-width: 1350px; margin: 0 auto; padding: 0 20px; display: flex; flex-wrap: wrap; gap: 7px; }}
  .tab-btn {{ background: #FFFFFF; border: 1px solid #CBD5E1; color: #475569; padding: 8px 14px; border-radius: 8px; cursor: pointer; font-weight: 600; font-size: 12.8px; }}
  .tab-btn:hover {{ background: #F1F5F9; color: #0F172A; }}
  .tab-btn.active {{ background: #0F172A; border-color: #0F172A; color: #FFFFFF; }}
  .tab-panel {{ display: none; }}
  .tab-panel.active {{ display: block; }}
  .glossary-drawer {{ display: none; background: #FFFFFF; border: 1px solid #CBD5E1; border-radius: 12px; padding: 18px 20px; margin: 0 0 18px 0; box-shadow: 0 4px 16px rgba(15,23,42,0.08); }}
  .glossary-drawer.open {{ display: block; }}
  .glossary-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-top: 10px; }}
  @media (max-width: 920px) {{ .glossary-grid {{ grid-template-columns: 1fr; }} }}
  .g-item {{ background: #F8FAFC; border: 1px solid #E2E8F0; border-left: 3px solid #0284C7; border-radius: 8px; padding: 11px 13px; font-size: 12.4px; color: #475569; }}
  .g-item strong {{ color: #0F172A; display: block; margin-bottom: 4px; }}
  .dual-mode-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 18px; }}
  @media (max-width: 920px) {{ .dual-mode-grid {{ grid-template-columns: 1fr; }} }}
  .mode-card {{ background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px; padding: 18px 20px; box-shadow: 0 1px 3px rgba(15,23,42,0.05); }}
  .mode-card h3 {{ margin: 0 0 8px 0; font-size: 15.5px; display: flex; justify-content: space-between; align-items: center; gap: 8px; }}
  .mode-metrics {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-top: 12px; }}
  .m-item {{ background: #F8FAFC; padding: 11px 12px; border-radius: 8px; border: 1px solid #E2E8F0; }}
  .m-item span {{ display: block; font-size: 11.4px; color: #64748B; font-weight: 600; }}
  .m-item strong {{ display: block; font-family: var(--font-mono); font-size: 15.5px; font-weight: 700; margin-top: 4px; }}
  .m-item small {{ display: block; font-size: 11px; color: #64748B; margin-top: 3px; }}
  .section-card {{ background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px; padding: 18px 20px; margin-bottom: 18px; box-shadow: 0 1px 3px rgba(15,23,42,0.05); }}
  .section-head {{ display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 10px; margin-bottom: 12px; border-bottom: 1px solid #E2E8F0; padding-bottom: 10px; }}
  .section-head h3 {{ margin: 0; font-size: 15.5px; }}
  .section-sub {{ font-size: 12.2px; color: #64748B; margin: 2px 0 0 0; }}
  .grid-5 {{ display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 10px; }}
  @media (max-width: 920px) {{ .grid-5 {{ grid-template-columns: repeat(2, 1fr); }} }}
  .grid-4 {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin-bottom: 14px; }}
  @media (max-width: 920px) {{ .grid-4 {{ grid-template-columns: repeat(2, 1fr); }} }}
  .grid-2 {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }}
  @media (max-width: 820px) {{ .grid-2 {{ grid-template-columns: 1fr; }} }}
  .filter-bar {{ display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 12px; }}
  .f-pill {{ border: 1px solid #CBD5E1; background: #FFFFFF; color: #475569; border-radius: 999px; padding: 4px 11px; font-size: 11.8px; font-weight: 600; cursor: pointer; }}
  .f-pill.active {{ background: #0F172A; color: #FFFFFF; border-color: #0F172A; }}
  .tbl-wrap {{ overflow-x: auto; border: 1px solid #E2E8F0; border-radius: 10px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 12.2px; }}
  th {{ background: #F8FAFC; color: #475569; text-align: left; padding: 9px 10px; font-size: 11px; font-weight: 700; text-transform: uppercase; border-bottom: 1px solid #E2E8F0; white-space: nowrap; }}
  td {{ padding: 8px 10px; border-bottom: 1px solid #F1F5F9; white-space: nowrap; }}
  tr:hover td {{ background: #F8FAFC; }}
  .badge {{ display: inline-block; padding: 2px 7px; border-radius: 5px; font-size: 10.6px; font-weight: 700; }}
  .b-v150 {{ background: #FEF3C7; color: #92400E; border: 1px solid #F59E0B; }}
  .b-v125 {{ background: #ECFDF5; color: #047857; border: 1px solid #34D399; }}
  .b-v100 {{ background: #F0F9FF; color: #0369A1; border: 1px solid #7DD3FC; }}
  .b-v075 {{ background: #F5F3FF; color: #6D28D9; border: 1px solid #C4B5FD; }}
  .b-v050 {{ background: #F1F5F9; color: #475569; border: 1px solid #CBD5E1; }}
  .pos {{ color: #059669; font-weight: 700; }}
  .neg {{ color: #DC2626; font-weight: 700; }}
  code {{ font-family: var(--font-mono); font-size: 12px; background: #F1F5F9; border: 1px solid #E2E8F0; padding: 1px 5px; border-radius: 4px; }}
  @keyframes legendary-shimmer {{ 0% {{ transform: translateX(-140%); }} 45%, 100% {{ transform: translateX(180%); }} }}
  .must-badge {{ position: relative; display: inline-block; overflow: hidden; padding: 2px 7px; border-radius: 5px; font-size: 10px; font-weight: 800; vertical-align: middle; margin-left: 5px; background: linear-gradient(135deg, #F59E0B 0%, #EA580C 50%, #D97706 100%); color: #FFFFFF; border: 1px solid #FDE68A; box-shadow: 0 2px 6px rgba(245,158,11,0.35); }}
  .must-badge::after {{ content: ""; position: absolute; top: 0; left: 0; width: 60%; height: 100%; background: linear-gradient(110deg, transparent 20%, rgba(255,255,255,0.65) 50%, transparent 80%); animation: legendary-shimmer 3.2s ease-in-out infinite; }}
  .calc-controls {{ display: flex; flex-wrap: wrap; gap: 14px; align-items: end; margin-bottom: 14px; background: #F8FAFC; padding: 12px 14px; border-radius: 10px; border: 1px solid #E2E8F0; }}
  .calc-field label {{ display: block; font-size: 11.5px; font-weight: 600; color: #475569; margin-bottom: 4px; }}
  .calc-field input, .calc-field select {{ padding: 7px 10px; border: 1px solid #CBD5E1; border-radius: 7px; font-family: var(--font-mono); font-size: 13px; background: #FFFFFF; min-width: 140px; }}
  .mech-card {{ background: #F8FAFC; border: 1px solid #E2E8F0; border-top: 3px solid #0284C7; border-radius: 10px; padding: 14px 16px; }}
  .mech-card h4 {{ margin: 0 0 6px 0; font-size: 14px; }}
  .mech-card p {{ margin: 0; font-size: 12.6px; color: #475569; line-height: 1.55; }}
</style>
</head>
<body>

<header class="hero-bar">
  <div class="hero-inner">
    <div class="hero-title">
      <h1>🏆 CHICKTRADE NEW SETUP v5 — 6 COIN (53 Setup Tinh Hoa · 6 Nấc Volume M2 + Van Hạ Lạnh)</h1>
      <p>Kiểm định chuẩn 36 tháng (<code>2023-11 → 2026-10</code>) · Bỏ hẳn <code>5m</code> &amp; 60 kèo cày phí · Chuẩn <code>S4_REAL_FUNDING</code> (Taker 0.05% + IOC ±0.25U + Funding 8h thật · Đóng nến thực tế)</p>
    </div>
    <div class="hero-actions">
      <button type="button" class="dict-toggle-btn" onclick="toggleGlossary()">📖 Từ Điển v5</button>
      <span class="ver-badge">NEW SETUP v5 · 53 SETUPS · <span id="gen-utc">--</span></span>
    </div>
  </div>
</header>

<nav class="nav-tabs">
  <div class="nav-tabs-inner">
    <button class="tab-btn active" onclick="switchTab('tab-overview', this)">🌐 Tổng Quan v5 (6 Coin)</button>
    <button class="tab-btn" onclick="switchTab('BTCUSDT', this)">🟠 BTC</button>
    <button class="tab-btn" onclick="switchTab('ETHUSDT', this)">🔵 ETH</button>
    <button class="tab-btn" onclick="switchTab('BNBUSDT', this)">🟡 BNB</button>
    <button class="tab-btn" onclick="switchTab('LINKUSDT', this)">🔷 LINK</button>
    <button class="tab-btn" onclick="switchTab('XRPUSDT', this)">⚪ XRP</button>
    <button class="tab-btn" onclick="switchTab('SOLUSDT', this)">🟣 SOL</button>
    <button class="tab-btn" onclick="switchTab('tab-mech', this)">🛡️ Cơ Chế v5 &amp; Máy Tính S2</button>
  </div>
</nav>

<main class="page-shell">

  <section id="glossary-drawer" class="glossary-drawer">
    <div style="display:flex;justify-content:space-between;align-items:center;">
      <h3 style="margin:0;font-size:15px;">📖 Kiến Trúc Khóa Cứng CHICKTRADE NEW SETUP v5</h3>
      <button type="button" class="dict-toggle-btn" onclick="toggleGlossary()">✕ Đóng</button>
    </div>
    <div class="glossary-grid">
      <div class="g-item">
        <strong>1. 53 Setup Tinh Hoa (Chỉ 15m / 30m / 1H)</strong>
        Đã loại sạch 14 setup <code>5m</code> nhiễu phí và 46 kèo "nhiều lệnh nhưng biên lãi mỏng" để tránh nghẽn suất lệnh và hạn chế kích hoạt phanh 2-SL oan.
      </div>
      <div class="g-item">
        <strong>2. Thang 6 Nấc Volume M2 + Van Hạ Lạnh</strong>
        <code>PWR ≥ 70% → 1.50×</code> · <code>≥65% → 1.25×</code> · <code>≥61% → 1.00×</code> · <code>≥58% → 0.75×</code> · <code>≥55% → 0.50×</code> · <code>&lt;55% → 0.25×</code>. Dính ≥3 SL liên tiếp tự hạ về <code>0.25×–0.50×</code>, <b>không đẩy nóng vượt trần</b>.
      </div>
      <div class="g-item">
        <strong>3. Chuẩn S4_REAL_FUNDING &amp; Auto-Turbo Sàn 110%</strong>
        Đóng đúng nến thực tế, SL trước TP, Taker <code>0.05%</code>, IOC ±0.25U, Funding 8h lịch sử thật. Auto-Turbo 250G chỉ bật từ <code>+25%</code> và bảo vệ sàn <code>110%</code> khi 5 lệnh SL cùng lúc.
      </div>
    </div>
  </section>

  <!-- TAB OVERVIEW -->
  <section id="tab-overview" class="tab-panel active">
    <div class="filter-bar" id="sc-bar">
      <button class="f-pill active" onclick="setSc('s4', this)">⚡ S4 Thực Chiến Chuẩn (Taker 0.05% + IOC ±0.25U + Funding 8h thật)</button>
      <button class="f-pill" onclick="setSc('s5', this)">🛡️ S5 Kiểm Chứng Chịu Đòn (Trượt giá xấu nhất +0.10U vào &amp; SL)</button>
    </div>

    <div class="dual-mode-grid" id="dual-cards"></div>

    <div class="section-card">
      <div class="section-head">
        <div>
          <h3>📈 Đường Cong Tài Sản 36 Tháng (2023–2026 · Bao 400G vs Auto-Turbo 250G)</h3>
          <p class="section-sub">Vốn $10,000 · 20× Cross · Lãi kép S2 (+8%/-10%) · Trần ≤3 lệnh cùng chiều · Phanh 2-SL · Van Hạ Lạnh Cold-Only</p>
        </div>
      </div>
      <div id="eq-chart" style="width:100%;height:370px;"></div>
    </div>

    <div class="section-card">
      <div class="section-head">
        <div>
          <h3>⚖️ Phân Bổ 6 Nấc Volume M2 Trên 53 Setup v5</h3>
          <p class="section-sub">Dồn tiền vào đúng siêu phẩm (PWR ≥ 65%) và hạ gọn các kèo biên vừa về 0.50×–0.75× để ép MaxDD xuống dưới 9%–11%</p>
        </div>
      </div>
      <div class="grid-5" id="vol-dist-grid"></div>
    </div>

    <div class="section-card">
      <div class="section-head">
        <div>
          <h3>📅 Hiệu Suất Từng Năm (2023 → 2026 · Bao 400G vs Auto-Turbo 250G)</h3>
          <p class="section-sub">Kiểm tra sự ổn định qua từng năm thực tế trên cùng dữ liệu S4 / S5</p>
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
          <h3>🪙 Tổng Hợp 6 Coin Trong Bộ New Setup v5 (53 Setup)</h3>
          <p class="section-sub">Bấm vào từng dòng hoặc tab phía trên để xem đầy đủ danh sách setup, điểm ngọt SL/TP1/TP2 và hệ số Volume của từng Coin</p>
        </div>
      </div>
      <div class="tbl-wrap">
        <table>
          <thead>
            <tr><th>Coin</th><th>Số Setup v5</th><th>THẤY LÀ QUẤT (≥1.25×)</th><th>Volume TB</th><th>Tổng Net S4 (U)</th><th>1U M15</th><th>1U M30</th><th>1U 1H</th></tr>
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
          <h3 id="coin-tbl-title">📋 Danh Sách Setup v5 Đã Chốt</h3>
          <p class="section-sub">Bao gồm hệ số Volume M2, Điểm ngọt SL/TP1/TP2, Edge/Risk (TB_U / SL_U) và độ bền trượt giá S5 Resilience (%)</p>
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
          <h3>🧮 Máy Tính Bậc Lãi Kép S2 Không Giới Hạn (Bậc 0 → 200 · Theo 6 Nấc Volume v5)</h3>
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
      <div class="section-head"><div><h3>🛡️ Quy Tắc Vận Hành Khóa Cứng Của New Setup v5</h3></div></div>
      <div class="grid-2">
        <div class="mech-card" style="border-top-color:#059669;">
          <h4>1. Thang 6 Nấc M2 + Van Chỉ Hạ Lạnh (Cold-Only Cooling)</h4>
          <p>Mỗi setup có trần Volume tĩnh cố định theo Profit WR (<code>0.50× → 1.50×</code>). Nếu setup dính <b>3 SL liên tiếp</b>, bot tự hạ về <code>0.25×–0.50×</code> tới khi thắng lại; <b>tuyệt đối không đẩy nóng vượt trần</b> để bảo toàn sức bền trượt giá S5 (~95%+).</p>
        </div>
        <div class="mech-card" style="border-top-color:#D97706;">
          <h4>2. Hạ Nhiệt Mượt Theo Drawdown Quỹ</h4>
          <p>Khi quỹ sụt từ đỉnh <code>-3.5%</code> → nhân mượt <code>×0.75</code> volume; khi sụt <code>-6.0%</code> → khóa cứng trần <code>0.25×</code>; khi biến động nến đột biến <code>ATR_ratio ≥ 1.35</code> → tự động chia đôi volume <code>×0.5</code>.</p>
        </div>
        <div class="mech-card" style="border-top-color:#0284C7;">
          <h4>3. Auto-Turbo 250G (+25% Bật · Sàn 110%)</h4>
          <p>Mặc định chạy <b>Bao 400G</b>. Chỉ kích hoạt <b>Auto-Turbo 250G</b> khi vốn ≥ <code>$12,500 (+25%)</code> và kiểm tra trước mỗi lệnh sao cho 5 lệnh SL cùng lúc vẫn giữ ≥ <code>$11,000 (110%)</code>.</p>
        </div>
        <div class="mech-card" style="border-top-color:#6D28D9;">
          <h4>4. Trần Tương Quan ≤ 3 Lệnh Cùng Chiều + Phanh 2-SL</h4>
          <p>Tối đa 5 lệnh toàn quỹ, 1 lệnh/coin, <b>tối đa 3 lệnh cùng chiều LONG hoặc SHORT</b>; dính 2 SL gốc liên tiếp (nhóm ≥ 0.75×) bật phanh đánh nháp Shadow-Win (tối đa 72h).</p>
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
  const lbl = scMode === 's4' ? 'S4 Thực Chiến' : 'S5 Trượt Xấu Nhất';

  document.getElementById('dual-cards').innerHTML = `
    <div class="mode-card" style="border-top:4px solid #059669;">
      <h3><span>🛡️ Bao 400G · ${{lbl}}</span><span class="badge b-v125">An Toàn · Ưu Tiên</span></h3>
      <div style="font-size:12.4px;color:#475569;">53 Setup · 6 Nấc Volume M2 + Hạ Lạnh (${{p4.cool_triggers}} lần kích hoạt) · Funding thật $${{Math.round(p4.funding_usd).toLocaleString()}}</div>
      <div class="mode-metrics">
        <div class="m-item"><span>Vốn Cuối 36M ($10k →)</span><strong class="pos">${{fmtUsd(p4.final_eq)}}</strong><small>${{fmtSign(p4.roi_pct, '%')}} · Bậc ${{p4.max_step}}</small></div>
        <div class="m-item"><span>MaxDD / Calmar</span><strong class="neg">${{p4.max_dd_pct}}%</strong><small>Calmar <b>${{p4.calmar}}</b> · ${{p4.trades}} lệnh</small></div>
        <div class="m-item"><span>WR / Thắng Tiền (PWR)</span><strong>${{p4.wr}}% / <span class="pos">${{p4.pwr}}%</span></strong><small>PF <b>${{p4.pf}}</b> · Vol TB ${{p4.avg_mult}}×</small></div>
      </div>
    </div>
    <div class="mode-card" style="border-top:4px solid #D97706;">
      <h3><span>⚡ Auto-Turbo 250G · ${{lbl}}</span><span class="badge b-v150">Bật +25% · Sàn 110%</span></h3>
      <div style="font-size:12.4px;color:#475569;">Bật tự động khi vốn ≥ $12,500 · 5 lệnh SL cùng lúc vẫn giữ ≥ $11,000 · Funding thật $${{Math.round(pt.funding_usd).toLocaleString()}}</div>
      <div class="mode-metrics">
        <div class="m-item"><span>Vốn Cuối 36M ($10k →)</span><strong style="color:#B45309;">${{fmtUsd(pt.final_eq)}}</strong><small>${{fmtSign(pt.roi_pct, '%')}} · Bậc ${{pt.max_step}}</small></div>
        <div class="m-item"><span>MaxDD / Calmar</span><strong class="neg">${{pt.max_dd_pct}}%</strong><small>Calmar <b>${{pt.calmar}}</b> · ${{pt.trades}} lệnh</small></div>
        <div class="m-item"><span>WR / Thắng Tiền (PWR)</span><strong>${{pt.wr}}% / <span class="pos">${{pt.pwr}}%</span></strong><small>PF <b>${{pt.pf}}</b> · Vol TB ${{pt.avg_mult}}×</small></div>
      </div>
    </div>
  `;

  Plotly.newPlot('eq-chart', [
    {{ x: p4.curve.map(c=>c.dt), y: p4.curve.map(c=>c.eq), name: `Bao 400G (${{lbl}})`, type: 'scatter', mode: 'lines', fill: 'tozeroy', fillcolor: 'rgba(5,150,105,0.08)', line: {{ color: '#059669', width: 2.6 }} }},
    {{ x: pt.curve.map(c=>c.dt), y: pt.curve.map(c=>c.eq), name: `Auto-Turbo 250G (${{lbl}})`, type: 'scatter', mode: 'lines', line: {{ color: '#D97706', width: 2.3, dash: 'dot' }} }}
  ], {{
    paper_bgcolor: '#FFFFFF', plot_bgcolor: '#F8FAFC', margin: {{ t: 20, r: 20, l: 72, b: 36 }},
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

  const syms = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'LINKUSDT', 'XRPUSDT', 'SOLUSDT'];
  document.getElementById('coin-sum-tbody').innerHTML = syms.map(sym => {{
    const c = D.coins[sym];
    return `
      <tr style="cursor:pointer;" onclick="jumpToCoin('${{sym}}')">
        <td class="num"><b>${{sym}}</b></td>
        <td class="num"><b>${{c.count}}</b></td>
        <td class="num" style="color:#D97706;font-weight:700;">${{c.must_count}}</td>
        <td class="num"><b>${{c.avg_vol}}×</b></td>
        <td class="num pos">+${{c.tot_net_s4}}U</td>
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
  document.getElementById('coin-tbl-title').textContent = `📋 ${{sym}} — ${{c.count}} Setup Tinh Hoa Đã Chốt (New Setup v5)`;
  document.getElementById('coin-kpis').innerHTML = `
    <div class="m-item" style="background:#FFFFFF;border-top:3px solid #059669;"><span>Số Setup v5 (${{sym}})</span><strong class="pos">${{c.count}} setup</strong><small>${{c.must_count}} kèo THẤY LÀ QUẤT</small></div>
    <div class="m-item" style="background:#FFFFFF;border-top:3px solid #D97706;"><span>Hệ Số Volume TB</span><strong>${{c.avg_vol}}×</strong><small>Theo thang 6 nấc M2</small></div>
    <div class="m-item" style="background:#FFFFFF;border-top:3px solid #0284C7;"><span>Tổng Net S4 (36M)</span><strong class="pos">+${{c.tot_net_s4}}U</strong><small>Đã trừ phí Taker + Funding thật</small></div>
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
    print(f"[OK] Built NEW SETUP v5 — 6 COIN HTML: {OUT_HTML} ({len(html):,} bytes)")


if __name__ == "__main__":
    main()
