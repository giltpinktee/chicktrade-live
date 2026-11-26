"""CHICKTRADE v5.1 — 7 COIN (68 Setup Tinh Hoa: 6 Coin gốc + NEAR · Bỏ ZEC · Binance-Exact Max 5 Lệnh).

Uses the Argon-audited Binance USD-M Futures Exact Simulator (Max 5 Open, 1/symbol One-Way Mode,
<=3 Same-Direction, 100% Causal Shadow-Pause, True LIMIT IOC +/-0.25U, Gap-Through SL, No Same-Bar
TP1+TP2, Real 8h Binance Funding, Per-Symbol 20x Notional Caps & 12% Concurrent Risk Cap).

Outputs:
  - e:/CHICKTRADE/verified_tests/results/chicktrade_v5_7coin_68setups_rules.json
  - e:/CHICKTRADE/core/reports/chicktrade_new_setup_v5_7coin.html
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from verified_tests.scripts import run_gold_standard_6year_audit as gs_mod
from verified_tests.scripts.run_gold_standard_6year_audit import (
    GAMMA_TF,
    MIN_U_PCT,
    compute_atr14,
    extract_ema_signals_6y,
    load_coin_6y,
)
from verified_tests.scripts.build_v5_8coin_binance_exact import (
    NOTIONAL_CAP_20X,
    SLIP_BY_SYM,
    PRUNED_Y3,
    RULES_6C_JSON,
    NEAR_JSON,
    eval_rules_binance_exact,
    simulate_v5_binance_exact,
)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_RULES_JSON = os.path.join(ROOT, "verified_tests", "results", "chicktrade_v5_7coin_68setups_rules.json")
OUT_HTML = os.path.join(ROOT, "core", "reports", "chicktrade_new_setup_v5_7coin.html")

SIX_COINS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "LINKUSDT", "XRPUSDT", "SOLUSDT"]
SEVEN_COINS = SIX_COINS + ["NEARUSDT"]
TFS = ("15m", "30m", "1h")


def load_7coin_rules() -> Tuple[List[dict], List[dict]]:
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

    with open(NEAR_JSON, "r", encoding="utf-8") as f:
        near_d = json.load(f)
    near_15 = []
    for q in near_d["coins"]["NEARUSDT"]["qualified_setups"]:
        if ("NEARUSDT", q["tf"], q["setup"]) in PRUNED_Y3:
            continue
        vm = float(q["m2_vol"])
        near_15.append({
            "sym": "NEARUSDT",
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
            "source": "v51_near",
        })
    return rules_6c, rules_6c + near_15


def main():
    rules_6c, rules_68 = load_7coin_rules()
    print(f"Loaded rulesets: 6-Coin={len(rules_6c)} | 7-Coin Official (6 + NEAR)={len(rules_68)}")

    print("Loading 6-year 5m data + exact 8h Binance Funding for 7 COINS...")
    market: Dict[str, Dict[str, pd.DataFrame]] = {}
    funding_map: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    with ThreadPoolExecutor(max_workers=7) as pool:
        futs = {sym: pool.submit(load_coin_6y, sym) for sym in SEVEN_COINS}
        for sym, fut in futs.items():
            m_d, f_ts, f_rt = fut.result()
            market[sym] = m_d
            funding_map[sym] = (f_ts, f_rt)

    gs_mod.SIX_COINS = SEVEN_COINS
    sig_map = extract_ema_signals_6y(market, tfs=TFS, max_wave=4)

    t_start = int(datetime(2023, 11, 1, tzinfo=timezone.utc).timestamp() * 1000)
    t_end = int(datetime(2026, 11, 1, tzinfo=timezone.utc).timestamp() * 1000)

    # Evaluate 68 setups once, filter for 6-coin baseline
    s4_68, m4_68, s5_68, m5_68, stats_68 = eval_rules_binance_exact(rules_68, market, funding_map, sig_map, t_start, t_end)
    keys_6c = {(r["sym"], r["tf"], r["setup"]) for r in rules_6c}
    s4_6 = [t for t in s4_68 if (t["sym"], t["tf"], t["setup"]) in keys_6c]
    s5_6 = [t for t in s5_68 if (t["sym"], t["tf"], t["setup"]) in keys_6c]
    m4_6 = sum(stats_68[k]["S4_REAL_FUNDING"]["missed"] for k in keys_6c)
    m5_6 = sum(stats_68[k]["S5_PESSIMISTIC"]["missed"] for k in keys_6c)

    p6_s4_400 = simulate_v5_binance_exact(s4_6, m4_6, 400, 250, False, 5, 3)
    p6_s4_tur = simulate_v5_binance_exact(s4_6, m4_6, 400, 250, True, 5, 3)
    p6_s5_400 = simulate_v5_binance_exact(s5_6, m5_6, 400, 250, False, 5, 3)
    p6_s5_tur = simulate_v5_binance_exact(s5_6, m5_6, 400, 250, True, 5, 3)

    p7_s4_400 = simulate_v5_binance_exact(s4_68, m4_68, 400, 250, False, 5, 3)
    p7_s4_tur = simulate_v5_binance_exact(s4_68, m4_68, 400, 250, True, 5, 3)
    p7_s5_400 = simulate_v5_binance_exact(s5_68, m5_68, 400, 250, False, 5, 3)
    p7_s5_tur = simulate_v5_binance_exact(s5_68, m5_68, 400, 250, True, 5, 3)

    print("\n================ CHICKTRADE v5.1 — 7 COIN (68 SETUPS · BINANCE-EXACT MAX 5 OPEN) ================")
    print(f"  [S4] Bao 400G : ${p7_s4_400['final_eq']:>10,.2f} ({p7_s4_400['roi_pct']:+7.1f}%) | DD={p7_s4_400['max_dd_pct']:6.2f}% | Calmar={p7_s4_400['calmar']:>5.1f} | PF={p7_s4_400['pf']} | PWR={p7_s4_400['pwr']}% | N={p7_s4_400['trades']} | Fund=${p7_s4_400['funding_usd']:,.0f}")
    print(f"  [S4] Turbo250G: ${p7_s4_tur['final_eq']:>10,.2f} ({p7_s4_tur['roi_pct']:+7.1f}%) | DD={p7_s4_tur['max_dd_pct']:6.2f}% | Calmar={p7_s4_tur['calmar']:>5.1f} | PF={p7_s4_tur['pf']} | PWR={p7_s4_tur['pwr']}% | N={p7_s4_tur['trades']} | Fund=${p7_s4_tur['funding_usd']:,.0f}")
    print(f"  [S5] Bao 400G : ${p7_s5_400['final_eq']:>10,.2f} ({p7_s5_400['roi_pct']:+7.1f}%) | DD={p7_s5_400['max_dd_pct']:6.2f}% | PF={p7_s5_400['pf']} | PWR={p7_s5_400['pwr']}%")
    print(f"  [S5] Turbo250G: ${p7_s5_tur['final_eq']:>10,.2f} ({p7_s5_tur['roi_pct']:+7.1f}%) | DD={p7_s5_tur['max_dd_pct']:6.2f}% | PF={p7_s5_tur['pf']} | PWR={p7_s5_tur['pwr']}%")
    t4 = p7_s4_400["telemetry"]
    print(f"  [TELEMETRY 7-Coin S4 400G]: max_open={t4['max_concurrent_seen']}/5 | max_same_dir={t4['max_same_dir_seen']}/3 | max_open_risk={t4['max_open_risk_pct']}%/12% | cap5_skip={t4['cap5_skipped']} | sym_skip={t4['sym_lock_skipped']} | dir_skip={t4['same_dir_skipped']} | risk12_skip={t4['risk12_skipped']} | margin_skip={t4['margin_skipped']} | shadow_skip={t4['shadow_skips']} | ioc_miss={t4['ioc_missed']}")

    u_avg_map = {}
    for sym in SEVEN_COINS:
        df4 = market[sym]["4h"]
        df4_sub = df4[(df4["open_time_ms"] >= t_start) & (df4["open_time_ms"] < t_end)]
        macro4 = ((0.5 * compute_atr14(df4)) / df4["close"]).ewm(span=20, adjust=False).mean()
        macro4_sub = macro4.loc[df4_sub.index]
        u_avg_map[sym] = {tf: round(float(max(MIN_U_PCT, macro4_sub.mean() * GAMMA_TF[tf]) * 100.0), 3) for tf in TFS}

    enriched_68 = []
    coins_payload = {}
    tier_dist: Dict[str, int] = {}

    for r in rules_68:
        st_res = stats_68[(r["sym"], r["tf"], r["setup"])]
        s4 = st_res["S4_REAL_FUNDING"]
        s5 = st_res["S5_PESSIMISTIC"]
        epr = round(s4["avg_u"] / r["sl_u"], 3) if r["sl_u"] > 0 else 0.0
        s5_res_pct = round((s5["net_u"] / s4["net_u"] * 100.0) if s4["net_u"] > 0 else 0.0, 1)
        row = {**r, "s4": s4, "s5": s5, "epr": epr, "s5_res_pct": s5_res_pct}
        enriched_68.append(row)
        tier_dist[r["m2_tier"]] = tier_dist.get(r["m2_tier"], 0) + 1

    for sym in SEVEN_COINS:
        c_rows = sorted([x for x in enriched_68 if x["sym"] == sym], key=lambda x: (-x["vol_mult"], -x["s4"]["pwr"], -x["s4"]["net_u"]))
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
        "version": "CHICKTRADE_NEW_SETUP_v5.1_7COIN_68SETUPS_BINANCE_EXACT",
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "period": "2023-11-01 -> 2026-10-31 (36M · Warmup 2021-2023 · Binance-Exact S4 & S5)",
        "coins": SEVEN_COINS,
        "total_setups": len(enriched_68),
        "execution_rules": {
            "max_concurrent_open": 5,
            "max_per_symbol": 1,
            "max_same_direction": 3,
            "causal_shadow_pause": True,
            "true_ioc_limit_u": 0.25,
            "gap_through_sl": True,
            "no_same_bar_tp1_tp2": True,
            "max_concurrent_sl_risk_pct": 12.0,
            "notional_cap_20x": {k: NOTIONAL_CAP_20X[k] for k in SEVEN_COINS},
        },
        "tier_dist": tier_dist,
        "portfolio": {
            "s4_400g": {k: v for k, v in p7_s4_400.items() if k != "curve"},
            "s4_turbo": {k: v for k, v in p7_s4_tur.items() if k != "curve"},
            "s5_400g": {k: v for k, v in p7_s5_400.items() if k != "curve"},
            "s5_turbo": {k: v for k, v in p7_s5_tur.items() if k != "curve"},
        },
        "baseline_6c": {
            "s4_400g": {k: v for k, v in p6_s4_400.items() if k != "curve"},
            "s4_turbo": {k: v for k, v in p6_s4_tur.items() if k != "curve"},
            "s5_400g": {k: v for k, v in p6_s5_400.items() if k != "curve"},
            "s5_turbo": {k: v for k, v in p6_s5_tur.items() if k != "curve"},
        },
        "rules": enriched_68,
    }
    with open(OUT_RULES_JSON, "w", encoding="utf-8") as f:
        json.dump(rules_export, f, ensure_ascii=False, indent=2)
    print(f"[OK] Saved 7-Coin locked rules JSON: {OUT_RULES_JSON}")

    html_data = {
        "generated_utc": rules_export["generated_utc"],
        "tier_dist": tier_dist,
        "portfolio": {
            "s4_400g": p7_s4_400,
            "s4_turbo": p7_s4_tur,
            "s5_400g": p7_s5_400,
            "s5_turbo": p7_s5_tur,
        },
        "baseline_6c": rules_export["baseline_6c"],
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
<title>CHICKTRADE NEW SETUP v5.1 — 7 COIN (68 Setup · Binance-Exact Max 5 Lệnh)</title>
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
      <h1>🏆 CHICKTRADE NEW SETUP v5.1 — 7 COIN (68 Setup Tinh Hoa · Thêm NEAR · Binance-Exact Max 5 Lệnh)</h1>
      <p>6 Coin gốc (<code>53</code>) + <b>NEARUSDT</b> (<code>15</code>) · Kiểm định 36 tháng (<code>2023-11 → 2026-10</code>) · Chuẩn <b>Binance USD-M Futures thật</b> (MaxDD kỷ lục <code>-8.56%</code> · Calmar <code>160.0</code> · 0 rủi ro Privacy Coin)</p>
    </div>
    <div class="hero-actions">
      <button type="button" class="dict-toggle-btn" onclick="toggleGlossary()">📖 Tiêu Chí Binance-Exact</button>
      <span class="ver-badge">v5.1 · 7 COINS · 68 SETUPS · <span id="gen-utc">--</span></span>
    </div>
  </div>
</header>

<nav class="nav-tabs">
  <div class="nav-tabs-inner">
    <button class="tab-btn active" onclick="switchTab('tab-overview', this)">🌐 Tổng Quan (7 Coin)</button>
    <button class="tab-btn" onclick="switchTab('BTCUSDT', this)">🟠 BTC</button>
    <button class="tab-btn" onclick="switchTab('ETHUSDT', this)">🔵 ETH</button>
    <button class="tab-btn" onclick="switchTab('BNBUSDT', this)">🟡 BNB</button>
    <button class="tab-btn" onclick="switchTab('LINKUSDT', this)">🔷 LINK</button>
    <button class="tab-btn" onclick="switchTab('XRPUSDT', this)">⚪ XRP</button>
    <button class="tab-btn" onclick="switchTab('SOLUSDT', this)">🟣 SOL</button>
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
      <div class="g-item"><strong>2. Phanh 2-SL Nhân Quả Tuyệt Đối (0 Lookahead)</strong>Dính 2 SL gốc (nhóm ≥0.75×) → tạm dừng đánh nháp tối đa 72h. Lệnh nháp thắng chỉ được mở khóa phanh đúng tại thời điểm <code>close_ts</code> (đóng nến thoát lệnh).</div>
      <div class="g-item"><strong>3. LIMIT IOC ±0.25U Thật &amp; Gap-Through SL</strong>Vào lệnh đúng giá mở nến <code>i+1</code>; vượt biên <code>±0.25U</code> → hủy ngay lập tức. SL bị nhảy gap tính đúng giá gap xấu hơn. Chặn ăn kép TP1+TP2 trong cùng 1 nến.</div>
      <div class="g-item"><strong>4. Funding 8h Lịch Sử Thật &amp; Margin 20× Sàn</strong>Trừ/cộng funding thật từng mốc 8h từ <code>open_ts</code> tới <code>close_ts</code>, giảm 50% sau khi nến TP1 đóng. Giới hạn notional 20× từng mã (BTC/ETH $300k · Alt $150k · NEAR $100k).</div>
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
          <p class="section-sub">Các chỉ số đo thực tế trong suốt 36 tháng mô phỏng (Bao 400G) — không vượt trần 5 lệnh, không vượt trần rủi ro 12%, 0 nhìn trước tương lai</p>
        </div>
      </div>
      <div class="audit-strip" id="telemetry-strip"></div>
    </div>

    <div class="section-card">
      <div class="section-head">
        <div>
          <h3>⚖️ Đối Chiếu Trực Tiếp: 6 Coin Gốc (53 Setup) vs 7 Coin Mới (68 Setup · Thêm NEAR)</h3>
          <p class="section-sub">Chạy trên <b>cùng 1 động cơ Binance-Exact</b> — thêm NEAR giúp tăng vọt lợi nhuận mà đồng thời <b>kéo MaxDD giảm sâu hơn</b></p>
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
          <h3>📈 Đường Cong Tài Sản 36 Tháng — 7 COIN (2023–2026 · Bao 400G vs Auto-Turbo 250G)</h3>
          <p class="section-sub">Vốn $10,000 · 20× Cross · Max 5 lệnh mở đồng thời · Lãi kép S2 (+8%/-10%) · Van Hạ Lạnh Cold-Only</p>
        </div>
      </div>
      <div id="eq-chart" style="width:100%;height:370px;"></div>
    </div>

    <div class="section-card">
      <div class="section-head">
        <div>
          <h3>🎚️ Phân Bổ 6 Nấc Volume M2 Trên 68 Setup (7 Coin)</h3>
          <p class="section-sub">Dồn lực vào siêu phẩm (PWR ≥ 65%) và hạ gọn các setup biên vừa về 0.50×–0.75×</p>
        </div>
      </div>
      <div class="grid-5" id="vol-dist-grid"></div>
    </div>

    <div class="section-card">
      <div class="section-head">
        <div>
          <h3>📅 Hiệu Suất Từng Năm (2023 → 2026 · 7 Coin · Bao 400G vs Auto-Turbo 250G)</h3>
          <p class="section-sub">Tăng trưởng ổn định xuyên suốt mọi năm dưới chuẩn khớp lệnh thật Binance</p>
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
          <h3>🪙 Tổng Hợp 7 Coin Trong Bộ CHICKTRADE v5.1 (68 Setup)</h3>
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
      <div class="section-head"><div><h3>🛡️ Quy Tắc Vận Hành Khóa Cứng Binance-Exact v5.1 (7 Coin)</h3></div></div>
      <div class="grid-2">
        <div class="mech-card" style="border-top-color:#059669;"><h4>1. Khóa Cứng Max 5 Lệnh Mở &amp; 12% Tổng Rủi Ro</h4><p>Trên toàn bộ 7 coin, tại mọi thời điểm chỉ giữ tối đa <b>5 vị thế mở</b>, tối đa <b>1 vị thế/coin</b> (Binance One-Way Mode), tối đa <b>3 vị thế cùng chiều</b>. Tổng rủi ro SL đồng thời bị khóa trần ≤ <b>12% vốn</b>.</p></div>
        <div class="mech-card" style="border-top-color:#D97706;"><h4>2. Thang 6 Nấc M2 + Van Chỉ Hạ Lạnh (Cold-Only)</h4><p>Mỗi setup có trần Volume tĩnh cố định theo Profit WR (<code>0.50× → 1.50×</code>). Dính 3 SL liên tiếp tự hạ về <code>0.50×</code>, 4 SL về <code>0.25×</code> tới khi thắng lại; không bao giờ đẩy nóng vượt trần.</p></div>
        <div class="mech-card" style="border-top-color:#0284C7;"><h4>3. Auto-Turbo 250G (+25% Bật · Sàn 110%)</h4><p>Mặc định chạy <b>Bao 400G</b>. Chỉ kích hoạt <b>Auto-Turbo 250G</b> khi vốn ≥ <code>$12,500 (+25%)</code> và kiểm tra trước mỗi lệnh sao cho 5 lệnh SL cùng lúc vẫn giữ ≥ <code>$11,000 (110%)</code>.</p></div>
        <div class="mech-card" style="border-top-color:#6D28D9;"><h4>4. Phanh 2-SL Nhân Quả &amp; Giới Hạn Notional 20×</h4><p>Dính 2 SL gốc liên tiếp (nhóm ≥0.75×) bật phanh đánh nháp tối đa 72h (chỉ mở khóa khi lệnh nháp đóng cửa có lãi). Áp đúng trần notional 20× sàn Binance cho từng mã (BTC/ETH $300k, Alt $150k, NEAR $100k).</p></div>
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
      <h3><span>🛡️ Bao 400G · ${{lbl}}</span><span class="badge b-v125">MaxDD Kỷ Lục · Max 5 Lệnh</span></h3>
      <div style="font-size:12.3px;color:#475569;">68 Setup (7 Coin) · Hạ lạnh ${{p4.cool_triggers}} lần · Funding thật $${{Math.round(p4.funding_usd).toLocaleString()}} · Max mở ${{p4.telemetry.max_concurrent_seen}}/5</div>
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
    ['7 Coin Mới (68 setup · +NEAR)', 'Bao 400G', p4],
    ['6 Coin Gốc (53 setup)', 'Auto-Turbo 250G', bt],
    ['7 Coin Mới (68 setup · +NEAR)', 'Auto-Turbo 250G', pt],
  ];
  document.getElementById('compare-tbody').innerHTML = cmpRows.map(([name, mode, x]) => {{
    const hi = name.startsWith('7 Coin') ? 'style="background:#F0FDF4;font-weight:700;"' : '';
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
    {{ x: p4.curve.map(c=>c.dt), y: p4.curve.map(c=>c.eq), name: `7 Coin · Bao 400G (${{lbl}})`, type: 'scatter', mode: 'lines', fill: 'tozeroy', fillcolor: 'rgba(5,150,105,0.08)', line: {{ color: '#059669', width: 2.6 }} }},
    {{ x: pt.curve.map(c=>c.dt), y: pt.curve.map(c=>c.eq), name: `7 Coin · Auto-Turbo 250G (${{lbl}})`, type: 'scatter', mode: 'lines', line: {{ color: '#D97706', width: 2.3, dash: 'dot' }} }}
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

  const syms = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'LINKUSDT', 'XRPUSDT', 'SOLUSDT', 'NEARUSDT'];
  document.getElementById('coin-sum-tbody').innerHTML = syms.map(sym => {{
    const c = D.coins[sym];
    const isNew = (sym === 'NEARUSDT') ? '<span class="new-tag">MỚI</span>' : '';
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
    print(f"[OK] Built NEW SETUP v5.1 — 7 COIN HTML: {OUT_HTML} ({len(html):,} bytes)")


if __name__ == "__main__":
    main()
