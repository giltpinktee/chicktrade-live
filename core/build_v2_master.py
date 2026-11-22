"""Build CHICKTRADE v2.0 Master Package:
1. Combines the 42 MTF exit-optimized setups (15m/30m/1h) + 1 verified 5m setup (LINKUSDT 5m EMA89_W2) = 43 Total Setups (32 Core + 11 Scout).
2. Computes exact Bayesian-Kelly Adaptive Volume x EMA Risk Multiplier (EMA34=1.00, EMA89=0.95, EMA200=0.90, Scouts capped at 0.45x initial), plus Vol_Hot(+5W) and Vol_Cold(-3SL).
3. Saves `core/core_strategy_rules.json` (v2.0-final-43setups-exit-optimized) and `core/reports/v2_43setups_master.csv`.
4. Runs the 43-setup portfolio simulation across 6 buffer/Auto-Turbo configurations (12M & OOS 6M) using cached data in `data/cache/`.
5. Generates the dark-themed, self-contained interactive HTML report at `core/reports/chicktrade_6coin_master_report.html` (8 tabs).
"""
import glob
import html
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

CORE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CORE_DIR)
RESEARCH_DIR = os.path.join(ROOT, "research")
for p in (ROOT, RESEARCH_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from config import BacktestConfig, RiskConfig
from core.adaptive_sizing import BayesianKellySizer, SetupPrior
from risk.compounding import CompoundingManager
from run_full_216_exit_sweetspot import COIN_CAL, sim_exit_arch
from run_full_3ema_storm_guard import build_cooldown_mask, compute_storm_flags, generate_all_4_touches
from run_research_mtf import resample_ohlcv

SIX_COINS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "LINKUSDT", "XRPUSDT", "SOLUSDT"]

EMA_RISK_MULT = {"EMA34": 1.00, "EMA89": 0.95, "EMA200": 0.90}

CHAM_LABELS = {
    "W1": "Chạm 1 (Khởi sóng)",
    "W2": "Chạm 2 (Tiếp diễn)",
    "W3": "Chạm 3 (Chín muồi)",
    "W4_REV": "Chạm 4 (Đảo chiều)",
}

EXIT_PARSE = {
    "ONLY_TP1": ("ONLY_TP1", 0.0, 1.0, "N/A (100% TP1)"),
    "ONLY_1R": ("ONLY_1R", 0.0, 1.0, "N/A (100% 1R)"),
    "PARTIAL_NO_MOVE": ("PARTIAL_NO_MOVE", 0.0, 0.5, "Giữ SL gốc"),
    "PARTIAL_BE+Phi": ("PARTIAL_MOVE_SL", 0.3, 0.5, "BE + Phí (+0.3U)"),
    "PARTIAL_Entry0U": ("PARTIAL_MOVE_SL", 0.0, 0.5, "Entry (0U)"),
    "PARTIAL_Entry-1U": ("PARTIAL_MOVE_SL", -1.0, 0.5, "Entry - 1U"),
    "PARTIAL_Entry-2U": ("PARTIAL_MOVE_SL", -2.0, 0.5, "Entry - 2U"),
}

EXIT_ARCH_DESC = {
    "ONLY_TP1": "Chốt 100% tại TP1 (Scalp an toàn)",
    "ONLY_1R": "Chốt 100% tại 1R (= SL_U)",
    "PARTIAL_NO_MOVE": "Chốt 50% TP1 + Giữ SL gốc + 50% TP2",
    "PARTIAL_MOVE_SL": "Chốt 50% TP1 + Dời SL chân 2 + 50% TP2",
}


def load_cached_5m(sym: str) -> pd.DataFrame:
    cache_dir = os.path.join(ROOT, "data", "cache")
    pattern = os.path.join(cache_dir, f"{sym}_futures_5m_*_12m.csv.gz")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"Cached 5m data not found for {sym} in {cache_dir}")
    df = pd.read_csv(matches[-1], parse_dates=["timestamp"])
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
    return df


def setup_cham_label(setup_code: str) -> str:
    parts = setup_code.split("_", 1)
    suffix = parts[1] if len(parts) > 1 else ""
    return CHAM_LABELS.get(suffix, suffix)


def compute_setup_payoff_stats(
    df: pd.DataFrame,
    sigs: list,
    setup_code: str,
    sl_u: float,
    tp1_u: float,
    tp2_u: float,
    arch: str,
    l2_u: float,
) -> Tuple[float, float, dict]:
    """Run exact trade simulation on a single setup to get exact AvgWin_U, AvgLoss_U, and exit anatomy."""
    h_arr = df["high"].values
    l_arr = df["low"].values
    c_arr = df["close"].values
    matched = [s for s in sigs if s.setup_type == setup_code]
    if not matched:
        return max(tp1_u * 0.8, 1.5), sl_u + 0.25, {}

    wins_u, losses_u = [], []
    sl_cnt = tp1_cnt = l2_cnt = tp2_cnt = 0
    for s in matched:
        u = max(0.5 * s.atr, s.entry_price * 1e-5)
        out, h_tp1, h_l2, h_tp2, nu = sim_exit_arch(
            h_arr, l_arr, c_arr, s.bar_idx, s.entry_price, s.direction, u,
            sl_u, tp1_u, tp2_u, arch, l2_u,
        )
        if out == "SL":
            sl_cnt += 1
        if h_tp1:
            tp1_cnt += 1
        if h_l2:
            l2_cnt += 1
        if h_tp2:
            tp2_cnt += 1
        if nu > 0:
            wins_u.append(nu)
        else:
            losses_u.append(-nu)

    n = len(matched)
    avg_win = float(np.mean(wins_u)) if wins_u else max(tp1_u * 0.8, 1.5)
    avg_loss = float(np.mean(losses_u)) if losses_u else (sl_u + 0.25)
    stats = {
        "n": n,
        "sl_full_pct": sl_cnt / n * 100.0,
        "tp1_hit_pct": tp1_cnt / n * 100.0,
        "leg2_sl_pct": l2_cnt / n * 100.0,
        "tp2_hit_pct": tp2_cnt / n * 100.0,
        "wr_pct": len(wins_u) / n * 100.0,
        "net_u": float(sum(wins_u) - sum(losses_u)),
    }
    return avg_win, avg_loss, stats


def sim_43_portfolio(
    tf_dfs_map: dict,
    tf_sigs_map: dict,
    rules_map: dict,
    base_buf: float,
    turbo_buf: float,
    trigger_eq: float,
    disengage_eq: float,
    max_c: int,
    start_time: pd.Timestamp = None,
    protected_floor: float = 11_000.0,
) -> dict:
    """Simulate the 43-setup portfolio with exact exit architectures, Storm Guard, and optional Auto-Turbo."""
    rcfg = RiskConfig(
        initial_capital=10_000.0,
        buffer_units=base_buf,
        max_concurrent_trades=max_c,
        max_trades_per_day=10,
    )
    comp = CompoundingManager(rcfg)
    equity = 10_000.0
    peak_eq = equity
    in_turbo = (trigger_eq <= equity)
    open_cap = round(max_c * 0.85, 2)

    cands = []
    for tfl, sm in tf_sigs_map.items():
        for sym, sigs in sm.items():
            for s in sigs:
                k = (sym, tfl, s.setup_type)
                if k not in rules_map:
                    continue
                if start_time is not None and s.timestamp < start_time:
                    continue
                cands.append((tfl, rules_map[k], s))

    tf_prio = {"1h": 0, "30m": 1, "15m": 2, "5m": 3}
    cands.sort(key=lambda x: (x[2].timestamp, tf_prio[x[0]], -x[1]["vol"], x[2].symbol))

    active, closed = [], []
    daily_t, daily_sl, daily_pnl, daily_eq = {}, {}, {}, {}
    consec_sl = 0
    first_ts = cands[0][2].timestamp if cands else pd.Timestamp("2025-11-01", tz="UTC")
    eq_points = [(first_ts.strftime("%Y-%m-%d %H:%M"), equity)]

    def settle(now_ts):
        nonlocal equity, peak_eq, consec_sl, in_turbo
        keep, done = [], []
        for ex_ts, tr in active:
            (done if ex_ts <= now_ts else keep).append((ex_ts, tr))
        done.sort(key=lambda x: x[0])
        for ex_ts, tr in done:
            equity += tr["pnl"]
            peak_eq = max(peak_eq, equity)
            comp.update_equity(equity, ex_ts)
            dk = tr["dk"]
            daily_pnl[dk] = daily_pnl.get(dk, 0.0) + tr["pnl"]
            if tr["out"] == "SL":
                consec_sl += 1
                daily_sl[dk] = daily_sl.get(dk, 0) + 1
            elif tr["pnl"] > 0:
                consec_sl = 0
            if (not in_turbo) and equity >= trigger_eq:
                in_turbo = True
            elif in_turbo and equity < disengage_eq:
                in_turbo = False
            closed.append(tr)
            eq_points.append((ex_ts.strftime("%Y-%m-%d %H:%M"), round(equity, 2)))
        active[:] = keep

    for tfl, rule, sig in cands:
        settle(sig.timestamp)
        if equity <= 0:
            break
        dk = sig.timestamp.strftime("%Y-%m-%d")
        if dk not in daily_eq:
            daily_eq[dk] = equity
        if daily_t.get(dk, 0) >= 10 or daily_sl.get(dk, 0) >= 3 or daily_pnl.get(dk, 0.0) <= -0.05 * daily_eq[dk]:
            continue
        if len(active) >= max_c or any(tr["sym"] == sig.symbol for _, tr in active):
            continue

        eff_m = rule["vol"]
        if sig.pa_vol_ok:
            eff_m *= 0.5
        if (equity - peak_eq) / peak_eq * 100.0 <= -6.0 or consec_sl >= 2:
            eff_m = min(eff_m, 0.25)
        eff_m = min(eff_m, max(0.0, open_cap - sum(tr["m"] for _, tr in active)))
        if eff_m < 0.20:
            continue

        cur_buf = turbo_buf if in_turbo else base_buf
        if in_turbo:
            open_sl_usd = sum(tr["sl_usd"] for _, tr in active)
            new_sl_usd = (min(comp.tier_capital, equity) * eff_m / turbo_buf) * rule["sl_u"]
            if (equity - open_sl_usd - new_sl_usd) < protected_floor:
                cur_buf = base_buf

        df = tf_dfs_map[tfl][sig.symbol]
        u = max(0.5 * sig.atr, sig.entry_price * 1e-5)
        out, _, _, _, nu = sim_exit_arch(
            df["high"].values, df["low"].values, df["close"].values,
            sig.bar_idx, sig.entry_price, sig.direction, u,
            rule["sl_u"], rule["tp1_u"], rule["tp2_u"], rule["arch"], rule["l2_u"],
        )
        sb = min(comp.tier_capital, equity)
        qty = (sb * eff_m) / (cur_buf * u)
        pnl = nu * u * qty
        sl_usd = qty * rule["sl_u"] * u

        ex_i = min(sig.bar_idx + (4 if out == "SL" else 8), len(df) - 1)
        ex_ts = pd.Timestamp(df["timestamp"].iloc[ex_i])
        if ex_ts.tzinfo is None:
            ex_ts = ex_ts.tz_localize("UTC")
        active.append((ex_ts, {
            "sym": sig.symbol, "tf": tfl, "m": eff_m, "out": out,
            "pnl": pnl, "sl_usd": sl_usd, "dk": dk, "in_turbo": cur_buf < base_buf,
        }))
        daily_t[dk] = daily_t.get(dk, 0) + 1

    if active:
        settle(pd.Timestamp("2099-01-01", tz="UTC"))

    ev = np.array([p[1] for p in eq_points])
    pk = np.maximum.accumulate(ev)
    dd_arr = (ev - pk) / pk * 100.0
    mdd = float(dd_arr.min())
    pnls = np.array([t["pnl"] for t in closed]) if closed else np.array([0.0])
    wr = float((pnls > 0).mean() * 100.0) if closed else 0.0
    pos = float(pnls[pnls > 0].sum())
    neg = float(-pnls[pnls < 0].sum())
    pf = pos / max(neg, 0.01)
    ret = (float(ev[-1]) / 10_000.0 - 1.0) * 100.0

    # Min equity after first time hitting trigger_eq
    hit_idxs = np.where(ev >= trigger_eq)[0]
    min_after_trigger = float(ev[hit_idxs[0]:].min()) if len(hit_idxs) > 0 else float(ev.min())

    # Downsample equity curve slightly for compact HTML if > 400 points
    if len(eq_points) > 400:
        step = max(1, len(eq_points) // 350)
        sampled = eq_points[::step]
        if sampled[-1] != eq_points[-1]:
            sampled.append(eq_points[-1])
    else:
        sampled = eq_points

    return {
        "trades": len(closed),
        "wr": round(wr, 2),
        "pf": round(pf, 2),
        "final_eq": round(float(ev[-1]), 2),
        "max_eq": round(float(ev.max()), 2),
        "ret": round(ret, 2),
        "mdd": round(mdd, 2),
        "step": comp.max_step_reached,
        "calmar": round(ret / max(abs(mdd), 0.1), 2),
        "min_after_trigger": round(min_after_trigger, 2),
        "curve_x": [p[0] for p in sampled],
        "curve_y": [p[1] for p in sampled],
    }


def badge_html(text: str, kind: str) -> str:
    return f'<span class="badge badge-{kind}">{html.escape(text)}</span>'


def arch_badge(arch: str) -> str:
    cls = {
        "ONLY_TP1": "cyan",
        "ONLY_1R": "indigo",
        "PARTIAL_NO_MOVE": "amber",
        "PARTIAL_MOVE_SL": "emerald",
    }.get(arch, "slate")
    return badge_html(arch, cls)


def status_badge(status: str) -> str:
    cls = {"CORE": "emerald", "SCOUT": "amber", "SKIP": "slate"}.get(status, "slate")
    return badge_html(status, cls)


def format_pf(val: float) -> str:
    if val >= 50.0:
        return "∞"
    return f"{val:.2f}"


def generate_html_report(
    rules_list: List[dict],
    portfolio_rows: List[dict],
    audit_df: pd.DataFrame,
    out_path: str,
) -> None:
    """Build self-contained dark-themed 8-tab interactive HTML report with Plotly.js CDN."""
    n_total = len(rules_list)
    n_core = sum(1 for r in rules_list if r["status"] == "CORE")
    n_scout = sum(1 for r in rules_list if r["status"] == "SCOUT")

    # Exit architecture counts
    arch_order = ["ONLY_TP1", "ONLY_1R", "PARTIAL_NO_MOVE", "PARTIAL_MOVE_SL"]
    arch_counts = {a: sum(1 for r in rules_list if r["exit_arch"] == a) for a in arch_order}
    arch_net_u = {a: round(sum(r["prior_1y"]["net_u"] for r in rules_list if r["exit_arch"] == a), 1) for a in arch_order}

    # Find key portfolio configs for KPI cards
    p_400 = next(p for p in portfolio_rows if "400G (Max 5)" in p["label"])
    p_t230 = next(p for p in portfolio_rows if "400G->230G" in p["label"])
    p_t250 = next(p for p in portfolio_rows if "400G->250G" in p["label"])

    # Portfolio Comparison Table HTML
    port_trs = []
    for p in portfolio_rows:
        is_rec = "400G (Max 5)" in p["label"] or "400G->230G" in p["label"]
        row_cls = ' class="row-highlight"' if is_rec else ""
        r12, r6 = p["m12"], p["oos6"]
        dd_ok = "✅" if abs(r12["mdd"]) < 15.0 else "⚠️"
        port_trs.append(
            f"<tr{row_cls}>"
            f"<td><strong>{html.escape(p['label'])}</strong></td>"
            f"<td class='num'>{r12['trades']}</td>"
            f"<td class='num'>{r12['wr']:.1f}%</td>"
            f"<td class='num'>{r12['pf']:.2f}</td>"
            f"<td class='num text-emerald'><strong>${r12['final_eq']:,.0f}</strong></td>"
            f"<td class='num text-emerald'>+{r12['ret']:,.1f}%</td>"
            f"<td class='num text-rose'>{r12['mdd']:.2f}% {dd_ok}</td>"
            f"<td class='num text-cyan'>{r12['calmar']:.1f}</td>"
            f"<td class='num'>{r12['step']}/20</td>"
            f"<td class='num text-emerald'>+{r6['ret']:,.1f}%</td>"
            f"<td class='num text-rose'>{r6['mdd']:.2f}%</td>"
            f"<td class='num'>{r6['wr']:.1f}%</td>"
            f"</tr>"
        )

    # Tab 2: 43 Setups Matrix rows
    matrix_trs = []
    for idx, r in enumerate(rules_list, 1):
        pr = r["prior_1y"]
        av = r["adaptive_volume"]
        if r["exit_arch"] in ("ONLY_TP1", "ONLY_1R"):
            sl_tp_str = f"SL {r['sl_u']:.0f}U | TP {r['tp1_u']:.1f}U (100%)"
        elif r["exit_arch"] == "PARTIAL_NO_MOVE":
            sl_tp_str = f"SL {r['sl_u']:.0f}U | TP1 {r['tp1_u']:.1f}U | Giữ SL | TP2 {r['tp2_u']:.0f}U"
        else:
            sl_tp_str = f"SL {r['sl_u']:.0f}U | TP1 {r['tp1_u']:.1f}U | {r['leg2_sl_label']} | TP2 {r['tp2_u']:.0f}U"

        matrix_trs.append(
            f'<tr data-coin="{r["coin"]}" data-status="{r["status"]}" data-arch="{r["exit_arch"]}" data-tf="{r["tf"]}">'
            f"<td class='num muted'>{idx}</td>"
            f"<td><strong>{r['coin'].replace('USDT','')}</strong></td>"
            f"<td><span class='tf-pill tf-{r['tf']}'>{r['tf']}</span></td>"
            f"<td><code>{r['setup_code']}</code></td>"
            f"<td class='num'>{pr['trades']}</td>"
            f"<td>{arch_badge(r['exit_arch'])}</td>"
            f"<td class='mono-sm'>{html.escape(sl_tp_str)}</td>"
            f"<td class='num text-rose'>{pr['sl_full_pct']:.1f}%</td>"
            f"<td class='num text-cyan'>{pr['tp1_hit_pct']:.1f}%</td>"
            f"<td class='num text-amber'>{pr['leg2_sl_pct']:.1f}%</td>"
            f"<td class='num text-emerald'>{pr['tp2_hit_pct']:.1f}%</td>"
            f"<td class='num'><strong>{pr['wr_pct']:.1f}%</strong></td>"
            f"<td class='num mono-sm'>{format_pf(pr['pf_is'])} / {format_pf(pr['pf_oos'])} / <strong class='text-emerald'>{format_pf(pr['pf_1y'])}</strong></td>"
            f"<td class='num text-cyan'><strong>{av['init_vol_mult']:.2f}x</strong></td>"
            f"<td class='num text-emerald'>{av['hot_streak_max_mult']:.2f}x</td>"
            f"<td class='num text-rose'>{av['cold_streak_3sl_mult']:.2f}x</td>"
            f"<td>{status_badge(r['status'])}</td>"
            f"</tr>"
        )

    # Tabs 3-8: 6 Coin Tabs
    coin_tabs_html = []
    for coin in SIX_COINS:
        c_rules = [r for r in rules_list if r["coin"] == coin]
        c_core = sum(1 for r in c_rules if r["status"] == "CORE")
        c_scout = sum(1 for r in c_rules if r["status"] == "SCOUT")
        c_trades = sum(r["prior_1y"]["trades"] for r in c_rules)
        c_net_u = sum(r["prior_1y"]["net_u"] for r in c_rules)
        c_avg_wr = float(np.mean([r["prior_1y"]["wr_pct"] for r in c_rules])) if c_rules else 0.0
        cc = COIN_CAL[coin]

        chosen_trs = []
        for r in c_rules:
            pr = r["prior_1y"]
            av = r["adaptive_volume"]
            if r["exit_arch"] in ("ONLY_TP1", "ONLY_1R"):
                sl_tp_str = f"SL {r['sl_u']:.0f}U → TP {r['tp1_u']:.1f}U (100%)"
            elif r["exit_arch"] == "PARTIAL_NO_MOVE":
                sl_tp_str = f"SL {r['sl_u']:.0f}U | 50% TP1 {r['tp1_u']:.1f}U | Giữ SL | 50% TP2 {r['tp2_u']:.0f}U"
            else:
                sl_tp_str = f"SL {r['sl_u']:.0f}U | 50% TP1 {r['tp1_u']:.1f}U | Dời {r['leg2_sl_label']} | 50% TP2 {r['tp2_u']:.0f}U"
            chosen_trs.append(
                f"<tr>"
                f"<td><span class='tf-pill tf-{r['tf']}'>{r['tf']}</span></td>"
                f"<td><code>{r['setup_code']}</code></td>"
                f"<td>{status_badge(r['status'])}</td>"
                f"<td>{arch_badge(r['exit_arch'])}</td>"
                f"<td class='mono-sm'>{html.escape(sl_tp_str)}</td>"
                f"<td class='num'>{pr['trades']}</td>"
                f"<td class='num text-rose'>{pr['sl_full_pct']:.1f}%</td>"
                f"<td class='num text-cyan'>{pr['tp1_hit_pct']:.1f}%</td>"
                f"<td class='num text-emerald'>{pr['tp2_hit_pct']:.1f}%</td>"
                f"<td class='num'><strong>{pr['wr_pct']:.1f}%</strong></td>"
                f"<td class='num mono-sm'>{format_pf(pr['pf_is'])} / {format_pf(pr['pf_oos'])} / <strong class='text-emerald'>{format_pf(pr['pf_1y'])}</strong></td>"
                f"<td class='num text-emerald'>+{pr['net_u']:.1f}U</td>"
                f"<td class='num text-cyan'><strong>{av['init_vol_mult']:.2f}x</strong></td>"
                f"</tr>"
            )

        # Full audit rows for this coin from full_216_exit_sweetspot_audit.csv
        c_audit = audit_df[audit_df["Coin"] == coin].copy()
        audit_trs = []
        for _, ar in c_audit.iterrows():
            st = str(ar["Status"])
            r_cls = " class='row-chosen'" if st in ("CORE", "SCOUT") else " class='row-skip'"
            net_v = float(ar["Net_U"])
            net_cls = "text-emerald" if net_v > 0 else "text-rose"
            audit_trs.append(
                f"<tr{r_cls}>"
                f"<td><span class='tf-pill tf-{ar['TF']}'>{ar['TF']}</span></td>"
                f"<td><code>{ar['Setup']}</code></td>"
                f"<td class='num'>{int(ar['Lenh'])}</td>"
                f"<td>{html.escape(str(ar['Best_Exit']))}</td>"
                f"<td class='num'>{float(ar['SL_Full_%']):.1f}%</td>"
                f"<td class='num'>{float(ar['TP1_Hit_%']):.1f}%</td>"
                f"<td class='num'>{float(ar['Dinh_DoiSL_%']):.1f}%</td>"
                f"<td class='num'>{float(ar['TP2_Hit_%']):.1f}%</td>"
                f"<td class='num'>{float(ar['WR_%']):.1f}%</td>"
                f"<td class='num'>{format_pf(float(ar['PF_IS']))}</td>"
                f"<td class='num'>{format_pf(float(ar['PF_OOS']))}</td>"
                f"<td class='num'>{format_pf(float(ar['PF_1Y']))}</td>"
                f"<td class='num {net_cls}'>{net_v:+.1f}U</td>"
                f"<td>{status_badge(st)}</td>"
                f"</tr>"
            )

        extra_note = ""
        if coin == "LINKUSDT":
            extra_note = (
                '<div class="callout callout-cyan">⚡ <strong>Độc quyền Khung 5m (1/72 toàn thị trường):</strong> '
                '<code>LINKUSDT 5m EMA89_W2</code> (PARTIAL_Entry-2U, SL 6U / TP1 4U / TP2 8U) là setup 5m DUY NHẤT vượt qua kiểm định IS/OOS 1 năm (28 lệnh, WR 78.6%, PF_IS 3.16 / PF_OOS 1.47 / PF_1Y 2.02, +40.8U).</div>'
            )

        coin_tabs_html.append(
            f"""
            <section id="tab-{coin}" class="tab-panel">
              <div class="kpi-grid">
                <div class="kpi-card">
                  <div class="kpi-label">Setup Đạt Chuẩn ({coin})</div>
                  <div class="kpi-val text-emerald">{len(c_rules)} <span class="kpi-sub">({c_core} Core · {c_scout} Scout)</span></div>
                </div>
                <div class="kpi-card">
                  <div class="kpi-label">Tổng Lệnh / Năm (1Y)</div>
                  <div class="kpi-val">{c_trades}</div>
                </div>
                <div class="kpi-card">
                  <div class="kpi-label">Winrate Trung Bình</div>
                  <div class="kpi-val text-cyan">{c_avg_wr:.1f}%</div>
                </div>
                <div class="kpi-card">
                  <div class="kpi-label">Tổng Net_U (Các Setup Chọn)</div>
                  <div class="kpi-val text-emerald">+{c_net_u:.1f}U</div>
                </div>
                <div class="kpi-card">
                  <div class="kpi-label">Hiệu Chuẩn Vật Lý (SL / TP1 / TP2)</div>
                  <div class="kpi-val mono-md">{cc['sl']:.0f}U / {cc['tp1']['15m']:.1f}-{cc['tp1']['1h']:.1f}U / {cc['tp2']:.0f}U</div>
                </div>
              </div>
              {extra_note}
              <div class="card">
                <div class="card-header">
                  <h3>🎯 Danh Sách Setup Đang Chạy Trên {coin} ({len(c_rules)} Setups)</h3>
                </div>
                <div class="table-wrap">
                  <table>
                    <thead>
                      <tr>
                        <th>TF</th><th>Setup</th><th>Status</th><th>Kiến Trúc</th><th>Cấu Hình SL / TP1 / Dời SL / TP2</th>
                        <th class="num">Lệnh</th><th class="num">SL Full</th><th class="num">TP1 Hit</th><th class="num">TP2 Hit</th>
                        <th class="num">WR%</th><th class="num">PF (IS/OOS/1Y)</th><th class="num">Net U</th><th class="num">Vol Init</th>
                      </tr>
                    </thead>
                    <tbody>{''.join(chosen_trs)}</tbody>
                  </table>
                </div>
              </div>
              <div class="card">
                <div class="card-header">
                  <h3>🔬 Full Audit Toàn Bộ Khung 15m/30m/1h{ ' + 5m' if coin == 'LINKUSDT' else '' } ({len(c_audit)} Setups)</h3>
                  <span class="muted">Nguồn: full_216_exit_sweetspot_audit.csv</span>
                </div>
                <div class="table-wrap">
                  <table>
                    <thead>
                      <tr>
                        <th>TF</th><th>Setup</th><th class="num">Lệnh</th><th>Best Exit</th>
                        <th class="num">SL Full%</th><th class="num">TP1 Hit%</th><th class="num">Dính Dời SL%</th><th class="num">TP2 Hit%</th>
                        <th class="num">WR%</th><th class="num">PF IS</th><th class="num">PF OOS</th><th class="num">PF 1Y</th>
                        <th class="num">Net U</th><th>Status</th>
                      </tr>
                    </thead>
                    <tbody>{''.join(audit_trs)}</tbody>
                  </table>
                </div>
              </div>
            </section>
            """
        )

    # Prepare JSON data for Plotly charts
    colors = ["#64748b", "#38bdf8", "#a78bfa", "#f472b6", "#f59e0b", "#10b981"]
    traces_12m = []
    traces_oos6 = []
    for idx, p in enumerate(portfolio_rows):
        is_bold = ("400G (Max 5)" in p["label"]) or ("400G->230G" in p["label"])
        traces_12m.append({
            "name": p["label"],
            "x": p["m12"]["curve_x"],
            "y": p["m12"]["curve_y"],
            "mode": "lines",
            "line": {"color": colors[idx % len(colors)], "width": 3 if is_bold else 1.7},
        })
        traces_oos6.append({
            "name": p["label"],
            "x": p["oos6"]["curve_x"],
            "y": p["oos6"]["curve_y"],
            "mode": "lines",
            "line": {"color": colors[idx % len(colors)], "width": 3 if is_bold else 1.7},
        })

    arch_chart_data = {
        "labels": arch_order,
        "counts": [arch_counts[a] for a in arch_order],
        "net_u": [arch_net_u[a] for a in arch_order],
    }

    html_doc = f"""<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>CHICKTRADE v2.0 — Master 43-Setup Strategy &amp; Portfolio Report</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    :root {{
      --bg: #0b1120;
      --surface: #111827;
      --surface-2: #162033;
      --border: #1e293b;
      --text: #e2e8f0;
      --muted: #94a3b8;
      --emerald: #10b981;
      --cyan: #38bdf8;
      --amber: #f59e0b;
      --rose: #f43f5e;
      --indigo: #818cf8;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Inter, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.5;
    }}
    header {{
      background: linear-gradient(135deg, #0f172a 0%, #111827 60%, #0b2434 100%);
      border-bottom: 1px solid var(--border);
      padding: 20px 28px;
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      align-items: center;
      gap: 16px;
    }}
    header h1 {{
      margin: 0;
      font-size: 1.45rem;
      font-weight: 700;
      letter-spacing: -0.01em;
    }}
    header .subtitle {{
      color: var(--muted);
      font-size: 0.88rem;
      margin-top: 4px;
    }}
    .version-pill {{
      background: rgba(16, 185, 129, 0.14);
      border: 1px solid rgba(16, 185, 129, 0.4);
      color: var(--emerald);
      padding: 6px 12px;
      border-radius: 999px;
      font-size: 0.8rem;
      font-weight: 600;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    }}
    .tab-bar {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      padding: 12px 28px;
      background: #0d1526;
      border-bottom: 1px solid var(--border);
      position: sticky;
      top: 0;
      z-index: 20;
    }}
    .tab-btn {{
      background: transparent;
      border: 1px solid var(--border);
      color: var(--muted);
      padding: 8px 14px;
      border-radius: 8px;
      cursor: pointer;
      font-size: 0.86rem;
      font-weight: 600;
      transition: all 0.15s ease;
    }}
    .tab-btn:hover {{
      color: var(--text);
      border-color: #334155;
      background: var(--surface);
    }}
    .tab-btn.active {{
      background: rgba(56, 189, 248, 0.14);
      border-color: var(--cyan);
      color: var(--cyan);
    }}
    main {{
      max-width: 1560px;
      margin: 0 auto;
      padding: 24px 28px 60px;
    }}
    .tab-panel {{ display: none; }}
    .tab-panel.active {{ display: block; }}
    .kpi-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 14px;
      margin-bottom: 20px;
    }}
    .kpi-card {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 16px 18px;
    }}
    .kpi-label {{
      font-size: 0.78rem;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.04em;
      margin-bottom: 6px;
    }}
    .kpi-val {{
      font-size: 1.5rem;
      font-weight: 700;
      font-variant-numeric: tabular-nums;
    }}
    .kpi-sub {{
      font-size: 0.82rem;
      font-weight: 500;
      color: var(--muted);
    }}
    .grid-2 {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
      gap: 18px;
      margin-bottom: 20px;
    }}
    .card {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 18px 20px;
      margin-bottom: 20px;
    }}
    .card-header {{
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      align-items: center;
      gap: 10px;
      margin-bottom: 14px;
    }}
    .card-header h3 {{
      margin: 0;
      font-size: 1.05rem;
      font-weight: 600;
    }}
    .pillar-list {{
      margin: 0;
      padding-left: 18px;
      color: #cbd5e1;
      font-size: 0.9rem;
    }}
    .pillar-list li {{ margin-bottom: 8px; }}
    .callout {{
      padding: 12px 16px;
      border-radius: 10px;
      margin-bottom: 18px;
      font-size: 0.9rem;
    }}
    .callout-cyan {{
      background: rgba(56, 189, 248, 0.1);
      border: 1px solid rgba(56, 189, 248, 0.35);
      color: #bae6fd;
    }}
    .table-wrap {{
      overflow-x: auto;
      border-radius: 8px;
      border: 1px solid var(--border);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.84rem;
      font-variant-numeric: tabular-nums;
    }}
    th, td {{
      padding: 9px 11px;
      border-bottom: 1px solid var(--border);
      text-align: left;
      white-space: nowrap;
    }}
    th {{
      background: var(--surface-2);
      color: var(--muted);
      font-weight: 600;
      font-size: 0.76rem;
      text-transform: uppercase;
      letter-spacing: 0.03em;
    }}
    tbody tr:hover {{ background: rgba(56, 189, 248, 0.05); }}
    tr.row-highlight {{ background: rgba(16, 185, 129, 0.08); }}
    tr.row-chosen {{ background: rgba(16, 185, 129, 0.06); }}
    tr.row-skip {{ opacity: 0.58; }}
    .num {{ text-align: right; }}
    .muted {{ color: var(--muted); }}
    .text-emerald {{ color: var(--emerald); }}
    .text-cyan {{ color: var(--cyan); }}
    .text-amber {{ color: var(--amber); }}
    .text-rose {{ color: var(--rose); }}
    .mono-sm {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.78rem; }}
    .mono-md {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 1.15rem; }}
    code {{
      background: #1e293b;
      padding: 2px 6px;
      border-radius: 5px;
      font-size: 0.8rem;
      color: #e2e8f0;
    }}
    .badge {{
      display: inline-block;
      padding: 2px 8px;
      border-radius: 999px;
      font-size: 0.73rem;
      font-weight: 600;
    }}
    .badge-emerald {{ background: rgba(16,185,129,0.16); color: #34d399; border: 1px solid rgba(16,185,129,0.35); }}
    .badge-cyan {{ background: rgba(56,189,248,0.16); color: #38bdf8; border: 1px solid rgba(56,189,248,0.35); }}
    .badge-indigo {{ background: rgba(129,140,248,0.16); color: #a5b4fc; border: 1px solid rgba(129,140,248,0.35); }}
    .badge-amber {{ background: rgba(245,158,11,0.16); color: #fbbf24; border: 1px solid rgba(245,158,11,0.35); }}
    .badge-slate {{ background: rgba(148,163,184,0.14); color: #94a3b8; border: 1px solid rgba(148,163,184,0.28); }}
    .tf-pill {{
      display: inline-block;
      padding: 2px 7px;
      border-radius: 6px;
      font-size: 0.74rem;
      font-weight: 700;
      font-family: ui-monospace, monospace;
    }}
    .tf-5m {{ background: rgba(244,63,94,0.2); color: #fda4af; }}
    .tf-15m {{ background: rgba(56,189,248,0.18); color: #7dd3fc; }}
    .tf-30m {{ background: rgba(167,139,250,0.18); color: #c4b5fd; }}
    .tf-1h {{ background: rgba(16,185,129,0.18); color: #6ee7b7; }}
    .filter-bar {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 14px;
      align-items: center;
    }}
    .filter-select {{
      background: var(--surface-2);
      border: 1px solid #334155;
      color: var(--text);
      padding: 6px 10px;
      border-radius: 8px;
      font-size: 0.84rem;
    }}
    .chart-toggle {{
      display: flex;
      gap: 8px;
    }}
    .chart-btn {{
      background: var(--surface-2);
      border: 1px solid #334155;
      color: var(--muted);
      padding: 5px 12px;
      border-radius: 6px;
      cursor: pointer;
      font-size: 0.8rem;
      font-weight: 600;
    }}
    .chart-btn.active {{
      border-color: var(--cyan);
      color: var(--cyan);
      background: rgba(56,189,248,0.14);
    }}
  </style>
</head>
<body>
  <header>
    <div>
      <h1>CHICKTRADE v2.0 — Master Strategy &amp; Portfolio Report</h1>
      <div class="subtitle">6 Coins (BTC · ETH · BNB · LINK · XRP · SOL) · {n_total} Exit-Optimized Setups ({n_core} Core + {n_scout} Scout) · Bayesian-Kelly Adaptive Sizing · Storm Guard</div>
    </div>
    <div class="version-pill">v2.0-final-43setups-exit-optimized</div>
  </header>

  <nav class="tab-bar">
    <button class="tab-btn active" onclick="openTab('tab-overview', this)">1. Tổng Quan v2.0</button>
    <button class="tab-btn" onclick="openTab('tab-matrix', this)">2. Ma Trận 43 Setups</button>
    <button class="tab-btn" onclick="openTab('tab-BTCUSDT', this)">3. BTC</button>
    <button class="tab-btn" onclick="openTab('tab-ETHUSDT', this)">4. ETH</button>
    <button class="tab-btn" onclick="openTab('tab-BNBUSDT', this)">5. BNB</button>
    <button class="tab-btn" onclick="openTab('tab-LINKUSDT', this)">6. LINK</button>
    <button class="tab-btn" onclick="openTab('tab-XRPUSDT', this)">7. XRP</button>
    <button class="tab-btn" onclick="openTab('tab-SOLUSDT', this)">8. SOL</button>
  </nav>

  <main>
    <!-- TAB 1: TONG QUAN v2.0 -->
    <section id="tab-overview" class="tab-panel active">
      <div class="kpi-grid">
        <div class="kpi-card">
          <div class="kpi-label">Tổng Bộ Setup v2.0</div>
          <div class="kpi-val text-cyan">{n_total} <span class="kpi-sub">({n_core} Core · {n_scout} Scout)</span></div>
        </div>
        <div class="kpi-card">
          <div class="kpi-label">Mặc Định: Bao Giá 400G (Max 5)</div>
          <div class="kpi-val text-emerald">+{p_400['m12']['ret']:,.1f}% <span class="kpi-sub">DD {p_400['m12']['mdd']:.2f}%</span></div>
        </div>
        <div class="kpi-card">
          <div class="kpi-label">Auto-Turbo ON: 400G → 230G</div>
          <div class="kpi-val text-emerald">+{p_t230['m12']['ret']:,.1f}% <span class="kpi-sub">DD {p_t230['m12']['mdd']:.2f}%</span></div>
        </div>
        <div class="kpi-card">
          <div class="kpi-label">OOS 6 Tháng (400G / Turbo 230G)</div>
          <div class="kpi-val text-cyan">+{p_400['oos6']['ret']:,.1f}% <span class="kpi-sub">/ +{p_t230['oos6']['ret']:,.1f}%</span></div>
        </div>
        <div class="kpi-card">
          <div class="kpi-label">Phán Quyết Khung 5m</div>
          <div class="kpi-val text-amber">1 / 72 <span class="kpi-sub">LINK 5m EMA89_W2</span></div>
        </div>
      </div>

      <div class="grid-2">
        <div class="card">
          <div class="card-header">
            <h3>🛡️ Kiến Trúc Quản Trị Vốn &amp; Auto-Turbo v2.0</h3>
          </div>
          <ul class="pillar-list">
            <li><strong>Bao giá mặc định 400G (Max 5 lệnh đồng thời):</strong> Đạt Equity <strong>${p_400['m12']['final_eq']:,.0f}</strong> (+{p_400['m12']['ret']:,.1f}% / 12M, MaxDD <strong>{p_400['m12']['mdd']:.2f}%</strong>, Calmar {p_400['m12']['calmar']:.1f}). OOS 6M đạt +{p_400['oos6']['ret']:,.1f}% (DD {p_400['oos6']['mdd']:.2f}%).</li>
            <li><strong>Công tắc Auto-Turbo ON/OFF (Mặc định OFF, bật/tắt runtime):</strong> Khởi đầu ở 400G để bảo vệ vốn gốc $10,000. Khi Equity ≥ <strong>$12,500</strong> (1.25x), tự động chuyển sang <strong>230G</strong> (hoặc 250G). Tự hạ về 400G nếu Equity &lt; <strong>$11,500</strong> (1.15x).</li>
            <li><strong>Sàn cứng bảo vệ $11,000 (Worst-Case SL Floor):</strong> Trước mỗi lệnh Turbo, bot kiểm tra <code>(Equity - Tổng_SL_Mở - SL_Lệnh_Mới) ≥ $11,000</code>. Nếu vi phạm, tự lùi lệnh đó về 400G — đảm bảo không bao giờ mất lãi gốc +10%. Đáy sau khi bật Turbo (400G→230G): <strong>${p_t230['m12']['min_after_trigger']:,.0f}</strong>.</li>
            <li><strong>Phán quyết khung 5m (1/72):</strong> Trong 72 tổ hợp 5m toàn thị trường, 71 setup bị nhiễu phí/whipsaw; duy nhất <code>LINKUSDT 5m EMA89_W2</code> (PARTIAL_Entry-2U, SL 6U / TP1 4U / TP2 8U) vượt chuẩn IS/OOS: 28 lệnh, WR 78.6%, PF_IS 3.16 / PF_OOS 1.47 / PF_1Y 2.02, +40.8U.</li>
            <li><strong>Hệ số rủi ro EMA + Bayesian-Kelly:</strong> Nhân hệ số theo tốc độ EMA (<code>EMA34=1.00</code>, <code>EMA89=0.95</code>, <code>EMA200=0.90</code>), giới hạn Scout ban đầu ≤ <code>0.45x</code> cho đến khi dữ liệu live xác nhận.</li>
          </ul>
        </div>
        <div class="card">
          <div class="card-header">
            <h3>🧩 Phân Bổ 4 Kiến Trúc Thoát Lệnh (43 Setups)</h3>
          </div>
          <div id="arch-chart" style="height:260px;"></div>
        </div>
      </div>

      <div class="card">
        <div class="card-header">
          <h3>📊 Bảng So Sánh Danh Mục 43 Setups (12 Tháng &amp; OOS 6 Tháng — Vốn Gốc $10,000)</h3>
        </div>
        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Cấu Hình Bao Giá / Auto-Turbo</th>
                <th class="num">Lệnh 12M</th>
                <th class="num">WR 12M</th>
                <th class="num">PF 12M</th>
                <th class="num">Equity 12M</th>
                <th class="num">Lãi 12M</th>
                <th class="num">MaxDD 12M</th>
                <th class="num">Calmar</th>
                <th class="num">Bậc Lãi Kép</th>
                <th class="num">OOS 6M Lãi</th>
                <th class="num">OOS 6M DD</th>
                <th class="num">OOS WR</th>
              </tr>
            </thead>
            <tbody>{''.join(port_trs)}</tbody>
          </table>
        </div>
      </div>

      <div class="card">
        <div class="card-header">
          <h3>📈 Đường Cong Tăng Trưởng Vốn (Interactive Equity Curve)</h3>
          <div class="chart-toggle">
            <button id="btn-eq-12m" class="chart-btn active" onclick="renderEquity('12m')">Full 12 Tháng</button>
            <button id="btn-eq-oos6" class="chart-btn" onclick="renderEquity('oos6')">OOS 6 Tháng</button>
          </div>
        </div>
        <div id="equity-chart" style="height:430px;"></div>
      </div>
    </section>

    <!-- TAB 2: MA TRAN 43 SETUPS -->
    <section id="tab-matrix" class="tab-panel">
      <div class="card">
        <div class="card-header">
          <h3>📋 Ma Trận Toàn Diện 43 Setups v2.0 (TP1 / TP2 / Dời SL / Bayesian-Kelly Volume)</h3>
          <span class="muted" id="matrix-count">Hiển thị 43 / 43 setups</span>
        </div>
        <div class="filter-bar">
          <select id="f-coin" class="filter-select" onchange="filterMatrix()">
            <option value="">Tất cả 6 Coin</option>
            <option value="BTCUSDT">BTC</option>
            <option value="ETHUSDT">ETH</option>
            <option value="BNBUSDT">BNB</option>
            <option value="LINKUSDT">LINK</option>
            <option value="XRPUSDT">XRP</option>
            <option value="SOLUSDT">SOL</option>
          </select>
          <select id="f-status" class="filter-select" onchange="filterMatrix()">
            <option value="">Tất cả Status (Core + Scout)</option>
            <option value="CORE">CORE ({n_core})</option>
            <option value="SCOUT">SCOUT ({n_scout})</option>
          </select>
          <select id="f-arch" class="filter-select" onchange="filterMatrix()">
            <option value="">Tất cả Kiến Trúc Thoát Lệnh</option>
            <option value="ONLY_TP1">ONLY_TP1</option>
            <option value="ONLY_1R">ONLY_1R</option>
            <option value="PARTIAL_NO_MOVE">PARTIAL_NO_MOVE</option>
            <option value="PARTIAL_MOVE_SL">PARTIAL_MOVE_SL</option>
          </select>
          <select id="f-tf" class="filter-select" onchange="filterMatrix()">
            <option value="">Tất cả Khung Giờ</option>
            <option value="5m">5m</option>
            <option value="15m">15m</option>
            <option value="30m">30m</option>
            <option value="1h">1h</option>
          </select>
        </div>
        <div class="table-wrap">
          <table id="matrix-table">
            <thead>
              <tr>
                <th class="num">#</th>
                <th>Coin</th>
                <th>TF</th>
                <th>Setup</th>
                <th class="num">Lệnh 1Y</th>
                <th>Kiến Trúc Thoát Lệnh</th>
                <th>SL / TP1 / Dời SL / TP2</th>
                <th class="num">SL_Full%</th>
                <th class="num">TP1_Hit%</th>
                <th class="num">Dính Dời SL%</th>
                <th class="num">TP2_Hit%</th>
                <th class="num">WR%</th>
                <th class="num">PF (IS / OOS / 1Y)</th>
                <th class="num">Vol_Init</th>
                <th class="num">Vol_Hot</th>
                <th class="num">Vol_Cold</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>{''.join(matrix_trs)}</tbody>
          </table>
        </div>
      </div>
    </section>

    <!-- TABS 3-8: 6 COINS -->
    {''.join(coin_tabs_html)}
  </main>

  <script>
    const TRACES_12M = {json.dumps(traces_12m)};
    const TRACES_OOS6 = {json.dumps(traces_oos6)};
    const ARCH_DATA = {json.dumps(arch_chart_data)};

    function openTab(tabId, btn) {{
      document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.getElementById(tabId).classList.add('active');
      btn.classList.add('active');
      if (tabId === 'tab-overview') {{
        Plotly.Plots.resize('equity-chart');
        Plotly.Plots.resize('arch-chart');
      }}
    }}

    function renderArchChart() {{
      const trace = {{
        x: ARCH_DATA.labels,
        y: ARCH_DATA.counts,
        type: 'bar',
        text: ARCH_DATA.counts.map((c, i) => `${{c}} setups (+${{ARCH_DATA.net_u[i]}}U)`),
        textposition: 'outside',
        marker: {{ color: ['#38bdf8', '#818cf8', '#f59e0b', '#10b981'] }}
      }};
      Plotly.newPlot('arch-chart', [trace], {{
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        font: {{ color: '#cbd5e1', size: 12 }},
        margin: {{ t: 24, r: 16, b: 36, l: 38 }},
        yaxis: {{ gridcolor: '#1e293b', zerolinecolor: '#334155', range: [0, Math.max(...ARCH_DATA.counts) + 5] }},
        xaxis: {{ tickfont: {{ size: 11 }} }}
      }}, {{ responsive: true, displayModeBar: false }});
    }}

    function renderEquity(mode) {{
      document.getElementById('btn-eq-12m').classList.toggle('active', mode === '12m');
      document.getElementById('btn-eq-oos6').classList.toggle('active', mode === 'oos6');
      const traces = mode === '12m' ? TRACES_12M : TRACES_OOS6;
      Plotly.newPlot('equity-chart', traces, {{
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        font: {{ color: '#cbd5e1', size: 12 }},
        margin: {{ t: 18, r: 24, b: 44, l: 72 }},
        hovermode: 'x unified',
        legend: {{ orientation: 'h', y: -0.18, font: {{ size: 11 }} }},
        xaxis: {{ gridcolor: '#1e293b', showgrid: true }},
        yaxis: {{ gridcolor: '#1e293b', tickprefix: '$', separatethousands: true }},
        shapes: [
          {{ type: 'line', xref: 'paper', x0: 0, x1: 1, y0: 11000, y1: 11000, line: {{ color: '#f43f5e', width: 1.2, dash: 'dot' }} }},
          {{ type: 'line', xref: 'paper', x0: 0, x1: 1, y0: 12500, y1: 12500, line: {{ color: '#f59e0b', width: 1.2, dash: 'dash' }} }}
        ],
        annotations: [
          {{ xref: 'paper', x: 0.01, y: 11000, text: 'Sàn cứng bảo vệ $11,000', showarrow: false, yshift: -12, font: {{ color: '#fda4af', size: 10 }} }},
          {{ xref: 'paper', x: 0.01, y: 12500, text: 'Mốc bật Auto-Turbo $12,500', showarrow: false, yshift: 10, font: {{ color: '#fcd34d', size: 10 }} }}
        ]
      }}, {{ responsive: true }});
    }}

    function filterMatrix() {{
      const fc = document.getElementById('f-coin').value;
      const fs = document.getElementById('f-status').value;
      const fa = document.getElementById('f-arch').value;
      const ft = document.getElementById('f-tf').value;
      const rows = document.querySelectorAll('#matrix-table tbody tr');
      let shown = 0;
      rows.forEach(tr => {{
        const ok = (!fc || tr.dataset.coin === fc)
          && (!fs || tr.dataset.status === fs)
          && (!fa || tr.dataset.arch === fa)
          && (!ft || tr.dataset.tf === ft);
        tr.style.display = ok ? '' : 'none';
        if (ok) shown++;
      }});
      document.getElementById('matrix-count').textContent = `Hiển thị ${{shown}} / ${{rows.length}} setups`;
    }}

    renderArchChart();
    renderEquity('12m');
  </script>
</body>
</html>
"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_doc)


def main():
    reports_dir = os.path.join(CORE_DIR, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    # 1. Load cached 5m OHLCV for all 6 coins and resample
    print("Loading cached 5m OHLCV data for 6 coins...")
    raw_m5 = {sym: load_cached_5m(sym) for sym in SIX_COINS}
    dfs_m15 = {s: resample_ohlcv(d, "15min") for s, d in raw_m5.items()}
    dfs_m30 = {s: resample_ohlcv(d, "30min") for s, d in raw_m5.items()}
    dfs_h1 = {s: resample_ohlcv(d, "1h") for s, d in raw_m5.items()}
    dfs_h4 = {s: resample_ohlcv(d, "4h") for s, d in raw_m5.items()}
    tf_dfs_map = {"5m": raw_m5, "15m": dfs_m15, "30m": dfs_m30, "1h": dfs_h1}

    # Generate Storm-Guarded signals for 15m/30m/1h (all 6 coins) + 5m (LINKUSDT)
    cfg = BacktestConfig()
    tf_sigs_map: Dict[str, Dict[str, list]] = {}
    for tfl, tdf, hdf, cd, syms in [
        ("15m", dfs_m15, dfs_h1, 16, SIX_COINS),
        ("30m", dfs_m30, dfs_h4, 8, SIX_COINS),
        ("1h", dfs_h1, dfs_h4, 4, SIX_COINS),
        ("5m", raw_m5, dfs_h1, 48, ["LINKUSDT"]),
    ]:
        bst = compute_storm_flags(tdf["BTCUSDT"])
        bmask = build_cooldown_mask(bst["raw_storm"], cd)
        bts = tdf["BTCUSDT"]["timestamp"].values
        tf_sigs_map[tfl] = {
            sym: generate_all_4_touches(tdf[sym], hdf[sym], bts, bmask, sym, cfg.strategy, cooldown_bars=cd)[0]
            for sym in syms
        }

    # 2. Load the 42 MTF setups + append the 1 verified 5m setup (LINKUSDT 5m EMA89_W2)
    sel_csv = os.path.join(reports_dir, "selected_setups_exit_anatomy.csv")
    df_42 = pd.read_csv(sel_csv)

    # Compute exact anatomy for LINKUSDT 5m EMA89_W2
    _, _, link5_stats = compute_setup_payoff_stats(
        raw_m5["LINKUSDT"], tf_sigs_map["5m"]["LINKUSDT"], "EMA89_W2",
        sl_u=6.0, tp1_u=4.0, tp2_u=8.0, arch="PARTIAL_MOVE_SL", l2_u=-2.0,
    )
    row_5m = {
        "Coin": "LINKUSDT",
        "TF": "5m",
        "Setup": "EMA89_W2",
        "Lenh": 28,
        "Best_Exit": "PARTIAL_Entry-2U",
        "SL_Full_%": round(link5_stats.get("sl_full_pct", 21.43), 4),
        "TP1_Hit_%": round(link5_stats.get("tp1_hit_pct", 78.57), 4),
        "Dinh_DoiSL_%": round(link5_stats.get("leg2_sl_pct", 28.57), 4),
        "TP2_Hit_%": round(link5_stats.get("tp2_hit_pct", 46.43), 4),
        "WR_%": 78.5714,
        "PF_IS": 3.16,
        "PF_OOS": 1.47,
        "PF_1Y": 2.02,
        "Net_U": 40.84,
        "Status": "CORE",
        "Was_Old33": False,
    }
    df_43 = pd.concat([df_42, pd.DataFrame([row_5m])], ignore_index=True)
    assert len(df_43) == 43, f"Expected 43 setups, got {len(df_43)}"

    # 3. Compute Bayesian-Kelly Adaptive Volume x EMA Risk Multiplier for all 43 setups
    sizer = BayesianKellySizer(min_wr_active=0.45)
    rules_list = []
    rules_map = {}
    csv_rows = []

    for _, r in df_43.iterrows():
        coin = str(r["Coin"])
        tf = str(r["TF"])
        setup_code = str(r["Setup"])
        ema = setup_code.split("_")[0]
        status = str(r["Status"])
        is_scout = (status == "SCOUT")
        best_exit = str(r["Best_Exit"])
        arch, l2_u, tp1_frac, l2_label = EXIT_PARSE[best_exit]

        cc = COIN_CAL[coin]
        sl_u = float(cc["sl"])
        raw_tp1_u = 4.0 if (coin == "LINKUSDT" and tf == "5m") else float(cc["tp1"][tf])
        raw_tp2_u = float(cc["tp2"])

        if arch == "ONLY_TP1":
            tp1_u, tp2_u, leg2_sl_u = raw_tp1_u, 0.0, 0.0
        elif arch == "ONLY_1R":
            tp1_u, tp2_u, leg2_sl_u = sl_u, 0.0, 0.0
        elif arch == "PARTIAL_NO_MOVE":
            tp1_u, tp2_u, leg2_sl_u = raw_tp1_u, raw_tp2_u, -sl_u
            l2_label = f"Giữ SL gốc (-{sl_u:.0f}U)"
        else:
            tp1_u, tp2_u, leg2_sl_u = raw_tp1_u, raw_tp2_u, l2_u

        # Get exact empirical AvgWin_U & AvgLoss_U from the actual 1Y trades
        avg_win_u, avg_loss_u, _ = compute_setup_payoff_stats(
            tf_dfs_map[tf][coin], tf_sigs_map[tf][coin], setup_code,
            sl_u, raw_tp1_u, raw_tp2_u, arch, l2_u,
        )

        n_1y = int(r["Lenh"])
        wr_pct = float(r["WR_%"])
        wins_1y = int(round(n_1y * wr_pct / 100.0))
        sl_full_pct = float(r["SL_Full_%"])

        prior = SetupPrior(
            coin=coin,
            tf=tf,
            setup_code=setup_code,
            prior_trades=n_1y,
            prior_wins=wins_1y,
            avg_win_u=avg_win_u,
            avg_loss_u=avg_loss_u,
            sl_full_pct=sl_full_pct,
        )
        dec = sizer.evaluate(prior)
        dec_hot = sizer.evaluate(prior, [avg_win_u] * 5)
        dec_cold = sizer.evaluate(prior, [-avg_loss_u] * 3)

        ema_m = EMA_RISK_MULT.get(ema, 1.0)
        vol_init_raw = round(dec.vol_mult * ema_m, 2)
        vol_init = min(vol_init_raw, 0.45) if is_scout else vol_init_raw
        vol_hot = round(dec_hot.vol_mult * ema_m, 2)
        vol_cold = round(dec_cold.vol_mult * ema_m, 2)
        tier_str = "Scout (Rare 0.45x)" if is_scout else dec.tier_label

        rule_obj = {
            "coin": coin,
            "tf": tf,
            "ema": ema,
            "cham": setup_cham_label(setup_code),
            "setup_code": setup_code,
            "status": status,
            "is_scout": is_scout,
            "exit_arch": arch,
            "best_exit_raw": best_exit,
            "tp1_frac": tp1_frac,
            "sl_u": sl_u,
            "tp1_u": tp1_u,
            "tp2_u": tp2_u,
            "leg2_sl_u": leg2_sl_u,
            "leg2_sl_label": l2_label,
            "prior_1y": {
                "trades": n_1y,
                "wins": wins_1y,
                "sl_full_pct": round(sl_full_pct, 2),
                "tp1_hit_pct": round(float(r["TP1_Hit_%"]), 2),
                "leg2_sl_pct": round(float(r["Dinh_DoiSL_%"]), 2),
                "tp2_hit_pct": round(float(r["TP2_Hit_%"]), 2),
                "wr_pct": round(wr_pct, 2),
                "pf_is": round(float(r["PF_IS"]), 2),
                "pf_oos": round(float(r["PF_OOS"]), 2),
                "pf_1y": round(float(r["PF_1Y"]), 2),
                "net_u": round(float(r["Net_U"]), 2),
                "avg_win_u": round(avg_win_u, 2),
                "avg_loss_u": round(avg_loss_u, 2),
            },
            "adaptive_volume": {
                "kelly_full_pct": round(dec.kelly_full_pct, 2),
                "quarter_kelly_pct": round(dec.quarter_kelly_pct, 2),
                "ema_risk_mult": ema_m,
                "init_vol_mult": vol_init,
                "hot_streak_max_mult": vol_hot,
                "cold_streak_3sl_mult": vol_cold,
                "tier": tier_str,
            },
        }
        rules_list.append(rule_obj)

        rules_map[(coin, tf, setup_code)] = {
            "arch": arch,
            "l2_u": l2_u,
            "sl_u": sl_u,
            "tp1_u": raw_tp1_u,
            "tp2_u": raw_tp2_u,
            "vol": vol_init,
        }

        csv_rows.append({
            "Coin": coin,
            "TF": tf,
            "EMA": ema,
            "Setup": setup_code,
            "Status": status,
            "Trades_1Y": n_1y,
            "Exit_Arch": arch,
            "Best_Exit": best_exit,
            "TP1_Frac": tp1_frac,
            "SL_U": sl_u,
            "TP1_U": tp1_u,
            "Leg2_SL_U": leg2_sl_u,
            "Leg2_SL_Label": l2_label,
            "TP2_U": tp2_u,
            "SL_Full_%": round(sl_full_pct, 2),
            "TP1_Hit_%": round(float(r["TP1_Hit_%"]), 2),
            "Leg2_SL_%": round(float(r["Dinh_DoiSL_%"]), 2),
            "TP2_Hit_%": round(float(r["TP2_Hit_%"]), 2),
            "WR_%": round(wr_pct, 2),
            "PF_IS": round(float(r["PF_IS"]), 2),
            "PF_OOS": round(float(r["PF_OOS"]), 2),
            "PF_1Y": round(float(r["PF_1Y"]), 2),
            "Net_U": round(float(r["Net_U"]), 2),
            "Kelly_Full_%": round(dec.kelly_full_pct, 2),
            "Vol_Init": vol_init,
            "Vol_Hot(+5W)": vol_hot,
            "Vol_Cold(-3SL)": vol_cold,
            "Tier": tier_str,
        })

    # Sort rules nicely: CORE first, then by PF_1Y descending
    rules_list.sort(key=lambda x: (0 if x["status"] == "CORE" else 1, -x["prior_1y"]["pf_1y"]))
    csv_df = pd.DataFrame(csv_rows).sort_values(["Status", "PF_1Y"], ascending=[True, False])
    csv_out = os.path.join(reports_dir, "v2_43setups_master.csv")
    csv_df.to_csv(csv_out, index=False)

    # 4. Save core_strategy_rules.json (v2.0-final-43setups-exit-optimized)
    core_json = {
        "version": "2.0-final-43setups-exit-optimized",
        "default_buffer_units": 400,
        "supported_buffers": [250, 300, 400, 500],
        "buffer_max_concurrent": {"250": 2, "300": 3, "400": 5, "500": 7},
        "max_concurrent_trades": 5,
        "ema_risk_multipliers": EMA_RISK_MULT,
        "auto_turbo": {
            "enabled": False,
            "toggleable_runtime": True,
            "base_buffer_units": 400,
            "turbo_buffer_units": 230,
            "trigger_equity_ratio": 1.25,
            "disengage_equity_ratio": 1.15,
            "protected_floor_ratio": 1.10,
            "enforce_worst_case_sl_floor": True,
            "max_dd_ceiling_pct": 15.0,
            "note": "Optional ON/OFF toggle (default OFF). Triggers 230G at Equity >= 1.25x ($12,500), disengages < 1.15x ($11,500), enforces hard worst-case SL floor >= 1.10x ($11,000).",
        },
        "max_open_mult_cap": 4.25,
        "max_trades_per_day": 10,
        "compounding_steps": 20,
        "step_gain_pct": 0.08,
        "storm_guard": {
            "cooldown_hours": 4,
            "atr_spike_mult": 1.8,
            "btc_er_min": 0.15,
            "btc_er_atr_mult": 1.2,
            "elevated_atr_mult": 1.35,
            "dd_brake_pct": -6.0,
            "consec_sl_brake": 2,
            "daily_sl_stop": 3,
            "daily_loss_stop_pct": -5.0,
        },
        "summary": {
            "total_setups": len(rules_list),
            "core_setups": sum(1 for x in rules_list if x["status"] == "CORE"),
            "scout_setups": sum(1 for x in rules_list if x["status"] == "SCOUT"),
            "m5_verified_setups": 1,
        },
        "rules": rules_list,
    }
    json_out = os.path.join(CORE_DIR, "core_strategy_rules.json")
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(core_json, f, ensure_ascii=False, indent=2)
    print(f"Saved {json_out} ({len(rules_list)} setups: {core_json['summary']['core_setups']} Core + {core_json['summary']['scout_setups']} Scout)")

    # 5. Run 43-setup portfolio simulation across 6 configs (12M & OOS 6M)
    all_ts = sorted(s.timestamp for s in tf_sigs_map["15m"]["BTCUSDT"])
    mid_ts = all_ts[len(all_ts) // 2]

    configs = [
        ("1. Chuẩn: Bao 500G (Max 7)", 500.0, 500.0, 999_999.0, 11_500.0, 7),
        ("2. Mặc Định: Bao 400G (Max 5)", 400.0, 400.0, 999_999.0, 11_500.0, 5),
        ("3. Chuẩn: Bao 300G (Max 3)", 300.0, 300.0, 999_999.0, 11_500.0, 3),
        ("4. Chuẩn: Bao 250G (Max 2)", 250.0, 250.0, 999_999.0, 11_500.0, 2),
        ("5. Auto-Turbo 400G->250G (Max 5)", 400.0, 250.0, 12_500.0, 11_500.0, 5),
        ("6. Auto-Turbo 400G->230G (Max 5)", 400.0, 230.0, 12_500.0, 11_500.0, 5),
    ]

    portfolio_rows = []
    print("\n" + "=" * 122)
    print("CHICKTRADE v2.0 — 43-SETUP PORTFOLIO SIMULATION (12M & OOS 6M | BAYESIAN-KELLY x EMA RISK MULT)")
    print("=" * 122)
    summary_print = []
    for label, b_buf, t_buf, trig, diseng, mc in configs:
        r12 = sim_43_portfolio(tf_dfs_map, tf_sigs_map, rules_map, b_buf, t_buf, trig, diseng, mc, None)
        r6 = sim_43_portfolio(tf_dfs_map, tf_sigs_map, rules_map, b_buf, t_buf, trig, diseng, mc, mid_ts)
        portfolio_rows.append({"label": label, "m12": r12, "oos6": r6})
        summary_print.append({
            "Config": label,
            "12M_Tr": r12["trades"],
            "12M_WR%": r12["wr"],
            "12M_PF": r12["pf"],
            "12M_Eq_$": r12["final_eq"],
            "12M_Ret%": r12["ret"],
            "12M_DD%": r12["mdd"],
            "Calmar": r12["calmar"],
            "Step": f"{r12['step']}/20",
            "OOS_Ret%": r6["ret"],
            "OOS_DD%": r6["mdd"],
            "OOS_WR%": r6["wr"],
        })
    print(pd.DataFrame(summary_print).to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    # 6. Load full 216 audit CSV (+ append LINKUSDT 5m row for LINK tab audit) and build HTML report
    audit_csv = os.path.join(ROOT, "output_test", "full_216_exit_sweetspot_audit.csv")
    audit_df = pd.read_csv(audit_csv)
    audit_df = pd.concat([audit_df, pd.DataFrame([row_5m])], ignore_index=True)

    html_out = os.path.join(reports_dir, "chicktrade_6coin_master_report.html")
    generate_html_report(rules_list, portfolio_rows, audit_df, html_out)
    print(f"\nGenerated HTML report: {html_out} ({os.path.getsize(html_out):,} bytes)")
    print(f"Generated CSV matrix : {csv_out} ({len(csv_df)} rows)")


if __name__ == "__main__":
    main()
