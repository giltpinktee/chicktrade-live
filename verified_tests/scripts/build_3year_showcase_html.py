"""Build CHICKTRADE v3.0 36-Month Showcase HTML Report (U3_HTF_MACRO_ATR + S2 50-Step + Liquidity Cap).

36-Month Compounding Showcase (01/11/2023 -> 31/10/2026) with True Walk-Forward Blind Year 3:
  - 58 setups are discovered strictly on Train 2Y (2023-11-01 -> 2025-10-31) under
    U3_HTF_MACRO_ATR (1U anchored to 4H EMA20 macro ATR * gamma_tf, 100% zero lookahead).
  - Compounding: S2_LOCKED_HYSTERESIS_10PCT (50 steps, +8%/step, -10% hysteresis) +
    live per-trade liquidity cap ($300,000 BTC/ETH, $150,000 Altcoins).
  - Blind Year 3 (2025-11-01 -> 2026-10-31) is 100% Out-Of-Sample (+37.13% Auto-Turbo / +28.57% Bao 400G).
  - Full 36M continuous compounding: +9,934.9% ($1,003,492, MaxDD -13.33%, Calmar 745.5) Auto-Turbo 250G
    and +4,510.3% ($461,028, MaxDD -10.05%, Calmar 448.8) Bao 400G.

Outputs ONLY to:
  - e:\CHICKTRADE\core\reports\chicktrade_3year_showcase_report.html
Does NOT modify chicktrade_6coin_master_report.html, core_strategy_rules.json, or bot/.
"""
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
VERIFIED_DIR = os.path.dirname(SCRIPTS_DIR)
ROOT = os.path.dirname(VERIFIED_DIR)
for p in (ROOT, SCRIPTS_DIR, os.path.join(ROOT, "old", "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from config import BacktestConfig
from run_full_216_exit_sweetspot import COIN_CAL, sim_exit_arch
from run_full_3ema_storm_guard import build_cooldown_mask, compute_storm_flags, generate_all_4_touches
from run_invariant_1u_benchmark import (
    GAMMA_4H,
    discover_dual_split_rules_arrays,
    precompute_invariant_u_arrays,
)
from run_50step_s2_check import sim_s2_steps
from run_master_6coin_html import SIX_COINS
from run_pct_vs_atr_3year_study import filter_sigs, load_3year_cached_m5
from run_research_mtf import resample_ohlcv

REPORTS_DIR = os.path.join(ROOT, "core", "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)
OUT_HTML = os.path.join(REPORTS_DIR, "chicktrade_3year_showcase_report.html")


def sim_u3_with_trace(
    tf_dfs_map: dict,
    tf_sigs_map: dict,
    rules_map: dict,
    u3_arrays: Dict[str, Dict[str, np.ndarray]],
    base_buffer: float = 400.0,
    auto_turbo_buffer: Optional[float] = None,
    hard_floor_eq: float = 11_000.0,
    core_only_sl_trigger: bool = True,
    window_start_ts: pd.Timestamp = pd.Timestamp("2023-11-01 00:00:00", tz="UTC"),
    max_step: int = 50,
    apply_liq_cap: bool = True,
) -> Tuple[dict, List[dict], List[dict]]:
    """Mirror sim_s2_steps identically while capturing exact timestamped equity & DD curve."""
    step = 0
    tier_capital = 10_000.0
    equity = 10_000.0
    peak_eq = equity
    open_cap = 4.25
    in_turbo = False

    cands = []
    for tfl, sm in tf_sigs_map.items():
        for sym, sigs in sm.items():
            for s in sigs:
                k = (sym, tfl, s.setup_type)
                if k in rules_map:
                    cands.append((tfl, rules_map[k], s))

    tf_prio = {"1h": 0, "30m": 1, "15m": 2, "5m": 3}
    cands.sort(key=lambda x: (x[2].timestamp, tf_prio[x[0]], -x[1]["vol"], x[2].symbol))

    active, closed = [], []
    daily_t, daily_pnl, daily_eq = {}, {}, {}
    eq_curve = [equity]
    curve_points = [{"ts": window_start_ts.isoformat(), "eq": round(equity, 2), "dd": 0.0}]

    port_consec_sl = 0
    port_shadow_paused = False
    port_paused_until = pd.Timestamp("1970-01-01", tz="UTC")

    def settle(now_ts: pd.Timestamp):
        nonlocal equity, peak_eq, step, tier_capital, port_consec_sl, port_shadow_paused, port_paused_until, in_turbo
        keep, done = [], []
        for ex_ts, tr in active:
            (done if ex_ts <= now_ts else keep).append((ex_ts, tr))
        done.sort(key=lambda x: x[0])
        for ex_ts, tr in done:
            equity += tr["pnl"]
            peak_eq = max(peak_eq, equity)
            while step < max_step and equity >= 10_000.0 * (1.08 ** (step + 1)):
                step += 1
                tier_capital = 10_000.0 * (1.08 ** step)
            while step > 0 and equity < tier_capital * 0.90:
                step -= 1
                tier_capital = 10_000.0 * (1.08 ** step)
            dk = tr["dk"]
            daily_pnl[dk] = daily_pnl.get(dk, 0.0) + tr["pnl"]

            if auto_turbo_buffer is not None:
                if (not in_turbo) and equity >= 12_500.0:
                    in_turbo = True
                elif in_turbo and equity < 11_500.0:
                    in_turbo = False

            if tr["out"] == "SL":
                if (not core_only_sl_trigger) or tr["base_vol"] > 0.45:
                    port_consec_sl += 1
                    if port_consec_sl >= 2:
                        port_consec_sl = 0
                        port_shadow_paused = True
                        port_paused_until = ex_ts + pd.Timedelta(hours=72)
            elif tr["pnl"] > 0:
                port_consec_sl = 0

            tr["exit_ts"] = ex_ts
            closed.append(tr)
            eq_curve.append(equity)
            dd_pct = (equity - peak_eq) / peak_eq * 100.0 if peak_eq > 0 else 0.0
            curve_points.append({
                "ts": ex_ts.isoformat(),
                "eq": round(equity, 2),
                "dd": round(dd_pct, 4),
            })
        active[:] = keep

    for tfl, rule, sig in cands:
        settle(sig.timestamp)
        if equity <= 0:
            break
        dk = sig.timestamp.strftime("%Y-%m-%d")
        if dk not in daily_eq:
            daily_eq[dk] = equity

        df = tf_dfs_map[tfl][sig.symbol]
        u = float(u3_arrays[tfl][sig.symbol][sig.bar_idx])

        if port_shadow_paused:
            if sig.timestamp >= port_paused_until:
                port_shadow_paused = False
            else:
                _, _, _, _, nu_sh = sim_exit_arch(
                    df["high"].values, df["low"].values, df["close"].values,
                    sig.bar_idx, sig.entry_price, sig.direction, u,
                    rule["sl_u"], rule["tp1_u"], rule["tp2_u"], rule["arch"], rule["l2_u"],
                )
                if nu_sh > 0:
                    port_shadow_paused = False
                continue

        if daily_t.get(dk, 0) >= 10 or daily_pnl.get(dk, 0.0) <= -0.05 * daily_eq[dk]:
            continue
        if len(active) >= 5 or any(tr["sym"] == sig.symbol for _, tr in active):
            continue

        eff_m = rule["vol"]
        if sig.pa_vol_ok:
            eff_m *= 0.5

        cur_dd_pct = (equity - peak_eq) / peak_eq * 100.0
        if cur_dd_pct <= -6.0:
            eff_m = min(eff_m, 0.25)

        eff_m = min(eff_m, max(0.0, open_cap - sum(tr["m"] for _, tr in active)))
        if eff_m < 0.20:
            continue

        sb = tier_capital
        cur_buf = base_buffer
        if auto_turbo_buffer is not None and in_turbo:
            cur_buf = auto_turbo_buffer
            open_sl_risk = sum(tr["worst_sl_usd"] for _, tr in active)
            new_sl_risk = (sb * eff_m) * (rule["sl_u"] / cur_buf) * 1.05
            if equity - (open_sl_risk + new_sl_risk) < hard_floor_eq:
                cur_buf = base_buffer

        out, _, _, _, nu = sim_exit_arch(
            df["high"].values, df["low"].values, df["close"].values,
            sig.bar_idx, sig.entry_price, sig.direction, u,
            rule["sl_u"], rule["tp1_u"], rule["tp2_u"], rule["arch"], rule["l2_u"],
        )
        qty = (sb * eff_m) / (cur_buf * u)
        if apply_liq_cap:
            liq_cap = 300_000.0 if sig.symbol in ("BTCUSDT", "ETHUSDT") else 150_000.0
            if qty * sig.entry_price > liq_cap:
                qty = liq_cap / sig.entry_price
        pnl = nu * u * qty
        worst_sl_usd = qty * (rule["sl_u"] * u) * 1.05

        ex_i = min(sig.bar_idx + (4 if out == "SL" else 8), len(df) - 1)
        ex_ts = pd.Timestamp(df["timestamp"].iloc[ex_i])
        if ex_ts.tzinfo is None:
            ex_ts = ex_ts.tz_localize("UTC")

        active.append((ex_ts, {
            "sym": sig.symbol,
            "dir": sig.direction,
            "base_vol": rule["vol"],
            "m": eff_m,
            "out": out,
            "pnl": float(pnl),
            "dk": dk,
            "worst_sl_usd": worst_sl_usd,
        }))
        daily_t[dk] = daily_t.get(dk, 0) + 1

    if active:
        settle(pd.Timestamp("2099-01-01", tz="UTC"))

    ev = np.array(eq_curve)
    pk = np.maximum.accumulate(ev)
    mdd = float(((ev - pk) / pk * 100.0).min())
    pnls = np.array([t["pnl"] for t in closed])
    wr = float((pnls > 0).mean() * 100.0) if len(pnls) else 0.0
    pf = float(pnls[pnls > 0].sum() / max(-pnls[pnls < 0].sum(), 0.01)) if len(pnls) else 0.0
    ret = (float(ev[-1]) / 10_000.0 - 1.0) * 100.0
    metrics = {
        "trades": len(closed),
        "wr": wr,
        "pf": pf,
        "final_eq": float(ev[-1]),
        "ret": ret,
        "mdd": mdd,
        "calmar": ret / max(abs(mdd), 0.1),
    }
    return metrics, closed, curve_points


def build_monthly_heatmap_36m(curve_points: List[dict]) -> Tuple[Dict[int, Dict[int, Optional[dict]]], Dict[int, dict], int, int]:
    """Compute all 36 months (2023-11 -> 2026-10) ret%, dollar PnL, and % on $10k."""
    df_c = pd.DataFrame(curve_points)
    df_c["ts"] = pd.to_datetime(df_c["ts"], utc=True)
    df_c["ym"] = df_c["ts"].dt.strftime("%Y-%m")

    periods = pd.period_range("2023-11", "2026-10", freq="M")
    month_data: Dict[int, Dict[int, Optional[dict]]] = {y: {m: None for m in range(1, 13)} for y in (2023, 2024, 2025, 2026)}
    year_start_eq: Dict[int, float] = {}
    year_end_eq: Dict[int, float] = {}

    prev_eq = 10_000.0
    green_cnt = 0
    total_cnt = 0

    for per in periods:
        y, m = per.year, per.month
        ym_str = str(per)
        if y not in year_start_eq:
            year_start_eq[y] = prev_eq
        sub = df_c[df_c["ym"] == ym_str]
        end_eq = float(sub["eq"].iloc[-1]) if len(sub) else prev_eq
        pnl_usd = end_eq - prev_eq
        m_ret = (pnl_usd / prev_eq) * 100.0 if prev_eq > 0 else 0.0
        base_pct = (pnl_usd / 10_000.0) * 100.0
        month_data[y][m] = {
            "ret_pct": round(m_ret, 2),
            "pnl_usd": round(pnl_usd, 0),
            "base_pct": round(base_pct, 1),
        }
        year_end_eq[y] = end_eq
        prev_eq = end_eq
        total_cnt += 1
        if m_ret >= 0:
            green_cnt += 1

    annual_data = {}
    for y in (2023, 2024, 2025, 2026):
        y_pnl = year_end_eq[y] - year_start_eq[y]
        annual_data[y] = {
            "ret_pct": round((y_pnl / year_start_eq[y]) * 100.0, 2),
            "pnl_usd": round(y_pnl, 0),
            "base_pct": round((y_pnl / 10_000.0) * 100.0, 1),
        }
    return month_data, annual_data, green_cnt, total_cnt


def heat_cell_colors(val: Optional[float]) -> str:
    if val is None:
        return "background:#F8FAFC;color:#94A3B8;"
    if val >= 15.0:
        return "background:#065F46;color:#FFFFFF;"
    if val >= 8.0:
        return "background:#059669;color:#FFFFFF;"
    if val >= 3.0:
        return "background:#A7F3D0;color:#065F46;"
    if val >= 0.0:
        return "background:#ECFDF5;color:#047857;"
    if val >= -3.0:
        return "background:#FFF1F2;color:#BE123C;"
    return "background:#FFE4E6;color:#9F1239;"


def fmt_usd_short(v: float) -> str:
    sign = "+" if v >= 0 else "-"
    return f"{sign}${abs(v):,.0f}"


def render_heatmap_tbody(month_data: Dict[int, Dict[int, Optional[dict]]], annual_data: Dict[int, dict], view_mode: str) -> str:
    rows = []
    year_labels = {
        2023: "2023 <small>(T11–T12 · Train)</small>",
        2024: "2024 <small>(12T · Train)</small>",
        2025: "2025 <small>(T1–10 Train · T11–12 Mù)</small>",
        2026: "2026 <small>(T1–T10 · Thi Mù OOS)</small>",
    }
    for y in (2023, 2024, 2025, 2026):
        cells = [f'<td class="hm-year">{year_labels[y]}</td>']
        for m in range(1, 13):
            d = month_data[y][m]
            if d is None:
                cells.append('<td class="hm-cell num" style="background:#F8FAFC;color:#94A3B8;">&mdash;</td>')
                continue
            ret_v = d["ret_pct"]
            pnl_v = d["pnl_usd"]
            base_v = d["base_pct"]
            sty = heat_cell_colors(ret_v)
            is_oos_cell = (y == 2025 and m >= 11) or (y == 2026)
            if is_oos_cell:
                sty += "box-shadow:inset 0 0 0 1.5px #0284C7;"
            sub_col = "rgba(255,255,255,0.88)" if ret_v >= 8.0 else ("#475569" if ret_v >= 0 else "#9F1239")
            if view_mode == "pct":
                top_txt = f"+{ret_v:.1f}%" if ret_v >= 0 else f"{ret_v:.1f}%"
                sub_txt = fmt_usd_short(pnl_v)
            else:
                top_txt = fmt_usd_short(pnl_v)
                sub_txt = f"({base_v:+,.1f}% gốc)"
            cells.append(
                f'<td class="hm-cell num" style="{sty}">'
                f'<div class="hm-top">{top_txt}</div>'
                f'<div class="hm-sub" style="color:{sub_col};">{sub_txt}</div>'
                f"</td>"
            )
        ad = annual_data[y]
        ann_ret = ad["ret_pct"]
        ann_pnl = ad["pnl_usd"]
        ann_base = ad["base_pct"]
        ann_col = "#059669" if ann_ret >= 0 else "#DC2626"
        ann_bg = "#ECFDF5" if ann_ret >= 0 else "#FFF1F2"
        if view_mode == "pct":
            ann_top = f"+{ann_ret:,.1f}%" if ann_ret >= 0 else f"{ann_ret:,.1f}%"
            ann_sub = fmt_usd_short(ann_pnl)
        else:
            ann_top = fmt_usd_short(ann_pnl)
            ann_sub = f"({ann_base:+,.1f}% gốc)"
        cells.append(
            f'<td class="hm-ann num" style="background:{ann_bg};color:{ann_col};">'
            f'<div class="hm-top">{ann_top}</div>'
            f'<div class="hm-sub" style="color:#475569;">{ann_sub}</div>'
            f"</td>"
        )
        rows.append(f"<tr>{''.join(cells)}</tr>")
    return "".join(rows)


def render_u3_macro_table_html(tf_dfs_map: dict, u3_arrays: Dict[str, Dict[str, np.ndarray]]) -> str:
    """Compact 6-row table showing 36M median U3_HTF_MACRO_ATR 1U%, SL%, TP1%, TP2%."""
    rows = []
    for sym in SIX_COINS:
        cc = COIN_CAL[sym]
        sl_u = float(cc["sl"])
        tp2_u = float(cc["tp2"])
        u_med = {}
        for tfl in ("15m", "30m", "1h"):
            close_np = tf_dfs_map[tfl][sym]["close"].values[205:]
            u_np = u3_arrays[tfl][sym][205:]
            u_med[tfl] = float(np.nanmedian(u_np / close_np) * 100.0)
        tp1_15 = float(cc["tp1"]["15m"])
        tp1_30 = float(cc["tp1"]["30m"])
        tp1_1h = float(cc["tp1"]["1h"])

        rows.append(
            f"""<tr>
                <td style="text-align:left;"><strong>{sym}</strong></td>
                <td class="num"><strong style="color:#0369A1;">{u_med['15m']:.3f}%</strong> <small style="color:#64748B;">(SL {sl_u*u_med['15m']:.2f}% &middot; TP1 {tp1_15*u_med['15m']:.2f}% &middot; TP2 {tp2_u*u_med['15m']:.2f}%)</small></td>
                <td class="num"><strong style="color:#6D28D9;">{u_med['30m']:.3f}%</strong> <small style="color:#64748B;">(SL {sl_u*u_med['30m']:.2f}% &middot; TP1 {tp1_30*u_med['30m']:.2f}% &middot; TP2 {tp2_u*u_med['30m']:.2f}%)</small></td>
                <td class="num"><strong style="color:#059669;">{u_med['1h']:.3f}%</strong> <small style="color:#64748B;">(SL {sl_u*u_med['1h']:.2f}% &middot; TP1 {tp1_1h*u_med['1h']:.2f}% &middot; TP2 {tp2_u*u_med['1h']:.2f}%)</small></td>
                <td class="num" style="color:#475569;">SL <b>{sl_u:.0f}U</b> &middot; TP2 <b>{tp2_u:.0f}U</b></td>
            </tr>"""
        )
    return "".join(rows)


def main():
    print("Loading 36-month cached 5m futures data (2023-11-01 -> 2026-10-31)...")
    raw_m5 = load_3year_cached_m5()
    cfg = BacktestConfig()
    dfs_m15 = {s: resample_ohlcv(d, "15min") for s, d in raw_m5.items()}
    dfs_m30 = {s: resample_ohlcv(d, "30min") for s, d in raw_m5.items()}
    dfs_h1 = {s: resample_ohlcv(d, "1h") for s, d in raw_m5.items()}
    dfs_h4 = {s: resample_ohlcv(d, "4h") for s, d in raw_m5.items()}
    tf_dfs_map = {"5m": raw_m5, "15m": dfs_m15, "30m": dfs_m30, "1h": dfs_h1}

    u_arrays, _ = precompute_invariant_u_arrays(tf_dfs_map, dfs_h4)
    u3_arrays = u_arrays["U3_HTF_MACRO_ATR"]

    print("Generating 4-touch signals across 36M...")
    tf_sigs_map = {}
    for tfl, tdf, hdf, cd in [("5m", raw_m5, dfs_h1, 48), ("15m", dfs_m15, dfs_h1, 16), ("30m", dfs_m30, dfs_h4, 8), ("1h", dfs_h1, dfs_h4, 4)]:
        bst = compute_storm_flags(tdf["BTCUSDT"])
        bmask = build_cooldown_mask(bst["raw_storm"], cd)
        bts = tdf["BTCUSDT"]["timestamp"].values
        tf_sigs_map[tfl] = {
            sym: generate_all_4_touches(tdf[sym], hdf[sym], bts, bmask, sym, cfg.strategy, cooldown_bars=cd)[0]
            for sym in SIX_COINS
        }

    ts_y1_start = pd.Timestamp("2023-11-01 00:00:00", tz="UTC")
    ts_y1_end = pd.Timestamp("2024-10-31 23:59:59", tz="UTC")
    ts_y2_start = pd.Timestamp("2024-11-01 00:00:00", tz="UTC")
    ts_y2_end = pd.Timestamp("2025-10-31 23:59:59", tz="UTC")
    ts_y3_start = pd.Timestamp("2025-11-01 00:00:00", tz="UTC")
    ts_y3_end = pd.Timestamp("2026-10-31 23:59:59", tz="UTC")

    # 1. Discover v3.0 58 setups strictly on Train 2Y (2023-11 -> 2025-10) under U3_HTF_MACRO_ATR
    sigs_y1 = filter_sigs(tf_sigs_map, ts_y1_start, ts_y1_end)
    sigs_y2 = filter_sigs(tf_sigs_map, ts_y2_start, ts_y2_end)
    sigs_train_2y = filter_sigs(tf_sigs_map, ts_y1_start, ts_y2_end)
    sigs_blind_y3 = filter_sigs(tf_sigs_map, ts_y3_start, ts_y3_end)
    sigs_36m = filter_sigs(tf_sigs_map, ts_y1_start, ts_y3_end)

    wf_rules = discover_dual_split_rules_arrays(tf_dfs_map, sigs_train_2y, "U3_HTF_MACRO_ATR", u3_arrays, ts_y1_end, min_trades=10)
    n_setups = len(wf_rules)
    assert n_setups == 58, f"Expected 58 v3.0 setups, got {n_setups}"
    print(f"Trained {n_setups} v3.0 setups strictly on Train 2Y (11/2023 -> 10/2025) under U3_HTF_MACRO_ATR.")

    # 2. Full 36M compounding curve traces + parity assertions (S2 50-Step + Liquidity Cap)
    f36_400, _, pts_400 = sim_u3_with_trace(
        tf_dfs_map, sigs_36m, wf_rules, u3_arrays, 400.0, None,
        window_start_ts=ts_y1_start, max_step=50, apply_liq_cap=True,
    )
    f36_250, _, pts_250 = sim_u3_with_trace(
        tf_dfs_map, sigs_36m, wf_rules, u3_arrays, 400.0, 250.0,
        window_start_ts=ts_y1_start, max_step=50, apply_liq_cap=True,
    )

    ref_400 = sim_s2_steps(tf_dfs_map, sigs_36m, wf_rules, u3_arrays, 50, True, ts_y1_start, ts_y3_end, 400.0, None)
    ref_250 = sim_s2_steps(tf_dfs_map, sigs_36m, wf_rules, u3_arrays, 50, True, ts_y1_start, ts_y3_end, 400.0, 250.0)
    assert abs(f36_400["final_eq"] - ref_400["final_eq"]) < 0.05
    assert abs(f36_250["final_eq"] - ref_250["final_eq"]) < 0.05

    # Individual periods (reset $10,000, S2 50-Step + Liquidity Cap)
    y1_400 = sim_s2_steps(tf_dfs_map, sigs_y1, wf_rules, u3_arrays, 50, True, ts_y1_start, ts_y1_end, 400.0, None)
    y1_250 = sim_s2_steps(tf_dfs_map, sigs_y1, wf_rules, u3_arrays, 50, True, ts_y1_start, ts_y1_end, 400.0, 250.0)
    y2_400 = sim_s2_steps(tf_dfs_map, sigs_y2, wf_rules, u3_arrays, 50, True, ts_y2_start, ts_y2_end, 400.0, None)
    y2_250 = sim_s2_steps(tf_dfs_map, sigs_y2, wf_rules, u3_arrays, 50, True, ts_y2_start, ts_y2_end, 400.0, 250.0)
    t2_400 = sim_s2_steps(tf_dfs_map, sigs_train_2y, wf_rules, u3_arrays, 50, True, ts_y1_start, ts_y2_end, 400.0, None)
    t2_250 = sim_s2_steps(tf_dfs_map, sigs_train_2y, wf_rules, u3_arrays, 50, True, ts_y1_start, ts_y2_end, 400.0, 250.0)
    y3_400 = sim_s2_steps(tf_dfs_map, sigs_blind_y3, wf_rules, u3_arrays, 50, True, ts_y3_start, ts_y3_end, 400.0, None)
    y3_250 = sim_s2_steps(tf_dfs_map, sigs_blind_y3, wf_rules, u3_arrays, 50, True, ts_y3_start, ts_y3_end, 400.0, 250.0)

    period_specs = [
        ("Năm 1 (11/2023 – 10/2024)", "Train · Reset $10k", y1_400, y1_250, "train"),
        ("Năm 2 (11/2024 – 10/2025)", "Train · Reset $10k", y2_400, y2_250, "train"),
        ("Gộp 2 Năm Train (11/2023 – 10/2025)", "Lãi kép 24T Train", t2_400, t2_250, "train"),
        ("Năm 3 Thi Mù OOS (11/2025 – 10/2026)", "100% Blind OOS · Reset $10k", y3_400, y3_250, "blind"),
        ("Tổng 36 Tháng Liên Tục (11/2023 – 10/2026)", "Lãi kép 36 tháng", f36_400, f36_250, "total"),
    ]

    period_rows_html = []
    for p_title, p_tag, r400, r250, p_kind in period_specs:
        pnl_400 = r400["final_eq"] - 10_000.0
        pnl_250 = r250["final_eq"] - 10_000.0
        base_pct_400 = pnl_400 / 10_000.0 * 100.0
        base_pct_250 = pnl_250 / 10_000.0 * 100.0
        row_cls = ' class="row-total"' if p_kind == "total" else (' style="background:#F0F9FF;"' if p_kind == "blind" else "")
        tag_cls = "tag-total" if p_kind == "total" else "tag-blind"
        c400 = "#15803D" if p_kind == "total" else ("#0369A1" if p_kind == "blind" else "#059669")
        c250 = "#EA580C" if p_kind == "total" else ("#0369A1" if p_kind == "blind" else "#059669")

        period_rows_html.append(
            f"""<tr{row_cls}>
                <td><strong>{p_title}</strong> <span class="{tag_cls}">{p_tag}</span></td>
                <td class="num">{r400['trades']:,d}</td>
                <td class="num">{r400['wr']:.1f}%</td>
                <td class="num" style="color:{c400};font-weight:700;">{r400['pf']:.2f}</td>
                <td class="num" style="color:{c400};font-weight:800;">+{r400['ret']:,.2f}%</td>
                <td class="num" style="color:{c400};font-weight:700;">{fmt_usd_short(pnl_400)} <small style="color:#475569;font-weight:500;">({base_pct_400:+,.2f}% gốc)</small></td>
                <td class="num" style="color:#DC2626;">{r400['mdd']:.2f}%</td>
                <td class="num" style="color:{c250};font-weight:800;">+{r250['ret']:,.2f}%</td>
                <td class="num" style="color:{c250};font-weight:700;">{fmt_usd_short(pnl_250)} <small style="color:#475569;font-weight:500;">({base_pct_250:+,.2f}% gốc)</small></td>
                <td class="num" style="color:#DC2626;">{r250['mdd']:.2f}%</td>
            </tr>"""
        )

    hm_400, ann_400, green_400, tot_m = build_monthly_heatmap_36m(pts_400)
    hm_250, ann_250, green_250, _ = build_monthly_heatmap_36m(pts_250)

    pnl36_400 = f36_400["final_eq"] - 10_000.0
    pnl36_250 = f36_250["final_eq"] - 10_000.0
    mult_400 = f36_400["final_eq"] / 10_000.0
    mult_250 = f36_250["final_eq"] / 10_000.0

    x_400 = [p["ts"] for p in pts_400]
    y_eq_400 = [p["eq"] for p in pts_400]
    y_dd_400 = [p["dd"] for p in pts_400]

    x_250 = [p["ts"] for p in pts_250]
    y_eq_250 = [p["eq"] for p in pts_250]
    y_dd_250 = [p["dd"] for p in pts_250]

    html_doc = f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CHICKTRADE v3.0 — Báo Cáo Tổng Quan 36 Tháng (U3_HTF_MACRO_ATR · 58 Setup)</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600;700;800&display=swap" rel="stylesheet">
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  :root {{
    --bg: #F8FAFC;
    --card: #FFFFFF;
    --border: #E2E8F0;
    --border-strong: #CBD5E1;
    --text: #0F172A;
    --text-sec: #475569;
    --muted: #64748B;
    --emerald: #059669;
    --emerald-deep: #15803D;
    --amber-deep: #C2410C;
    --rose-deep: #DC2626;
    --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    --font-mono: 'JetBrains Mono', ui-monospace, SFMono-Regular, Consolas, monospace;
  }}
  * {{ box-sizing: border-box; }}
  html, body {{
    margin: 0;
    padding: 0;
    width: 100%;
    max-width: 100vw;
    overflow-x: hidden;
    font-family: var(--font-sans);
    background: var(--bg);
    color: var(--text);
    line-height: 1.5;
    font-size: 13.5px;
    -webkit-font-smoothing: antialiased;
  }}
  .num {{ font-variant-numeric: tabular-nums; font-family: var(--font-mono); }}
  .nw {{ white-space: nowrap; display: inline-block; }}

  /* 1. Executive Hero Header */
  .exec-hero {{
    background: #FFFFFF;
    border-bottom: 1px solid var(--border);
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
    padding: 24px 0 20px 0;
    margin-bottom: 22px;
  }}
  .shell {{
    max-width: 1340px;
    margin: 0 auto;
    padding: 0 22px;
  }}
  .hero-top {{
    display: flex;
    flex-wrap: wrap;
    justify-content: space-between;
    align-items: flex-start;
    gap: 16px;
  }}
  .eyebrow {{
    display: inline-flex;
    align-items: center;
    gap: 8px;
    font-size: 11.5px;
    font-weight: 700;
    letter-spacing: 0.6px;
    text-transform: uppercase;
    color: #059669;
    background: #ECFDF5;
    border: 1px solid #A7F3D0;
    padding: 4px 10px;
    border-radius: 6px;
    margin-bottom: 10px;
  }}
  .hero-top h1 {{
    margin: 0;
    font-size: 23px;
    font-weight: 800;
    letter-spacing: -0.4px;
    color: #0F172A;
    line-height: 1.24;
  }}
  .hero-sub {{
    margin: 6px 0 0 0;
    font-size: 13.5px;
    color: #475569;
  }}

  @keyframes shimmer-sweep {{
    0% {{ transform: translateX(-140%); }}
    45%, 100% {{ transform: translateX(180%); }}
  }}
  .oos-verified-badge {{
    position: relative;
    overflow: hidden;
    background: linear-gradient(135deg, #059669 0%, #047857 55%, #065F46 100%);
    color: #FFFFFF;
    border: 1px solid #6EE7B7;
    border-radius: 10px;
    padding: 12px 18px;
    box-shadow: 0 4px 14px rgba(5, 150, 105, 0.22);
    min-width: 310px;
  }}
  .oos-verified-badge::after {{
    content: "";
    position: absolute;
    top: 0; left: 0;
    width: 55%; height: 100%;
    background: linear-gradient(110deg, transparent 20%, rgba(255,255,255,0.40) 50%, transparent 80%);
    animation: shimmer-sweep 3.4s ease-in-out infinite;
    pointer-events: none;
  }}
  .oos-verified-badge .b-lbl {{
    display: block;
    font-size: 10.8px;
    font-weight: 700;
    letter-spacing: 0.45px;
    text-transform: uppercase;
    opacity: 0.94;
  }}
  .oos-verified-badge .b-val {{
    display: block;
    font-family: var(--font-mono);
    font-size: 17px;
    font-weight: 800;
    margin-top: 2px;
  }}
  .oos-verified-badge .b-sub {{
    display: block;
    font-size: 11.5px;
    opacity: 0.94;
    margin-top: 2px;
  }}

  .meta-strip {{
    margin-top: 16px;
    display: flex;
    flex-wrap: wrap;
    gap: 8px 18px;
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    padding: 9px 14px;
    font-size: 12px;
    color: #334155;
  }}
  .meta-strip b {{ color: #0F172A; }}
  .meta-strip code {{
    font-family: var(--font-mono);
    font-size: 11.5px;
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    color: #0F172A;
    padding: 1px 5px;
    border-radius: 4px;
  }}

  /* 2. Two KPI Cards */
  .kpi-dual-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 18px;
    margin-bottom: 20px;
  }}
  @media (max-width: 900px) {{
    .kpi-dual-grid {{ grid-template-columns: 1fr; }}
  }}
  .tear-kpi-card {{
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 14px;
    padding: 22px 24px;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.05), 0 8px 24px rgba(15, 23, 42, 0.04);
  }}
  .tear-kpi-head {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 10px;
    margin-bottom: 12px;
  }}
  .tear-kpi-head h2 {{
    margin: 0;
    font-size: 15.5px;
    font-weight: 700;
    color: #0F172A;
  }}
  .pill-primary {{
    background: #ECFDF5;
    color: #047857;
    border: 1px solid #A7F3D0;
    padding: 3px 10px;
    border-radius: 6px;
    font-size: 11px;
    font-weight: 700;
  }}
  .pill-turbo {{
    background: #FFEDD5;
    color: #C2410C;
    border: 1px solid #FDBA74;
    padding: 3px 10px;
    border-radius: 6px;
    font-size: 11px;
    font-weight: 700;
  }}
  .hero-figure-row {{
    display: flex;
    flex-wrap: wrap;
    align-items: baseline;
    gap: 12px;
    margin-bottom: 6px;
  }}
  .hero-eq {{
    font-family: var(--font-mono);
    font-size: 29px;
    font-weight: 800;
    letter-spacing: -0.6px;
  }}
  .hero-ret-badge {{
    font-family: var(--font-mono);
    font-size: 14.5px;
    font-weight: 700;
    padding: 3px 10px;
    border-radius: 7px;
  }}
  .tear-desc {{
    margin: 0 0 16px 0;
    font-size: 12.8px;
    color: #475569;
  }}
  .sub-stats-grid {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
  }}
  @media (max-width: 640px) {{
    .sub-stats-grid {{ grid-template-columns: repeat(2, 1fr); }}
  }}
  .sub-stat {{
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 9px;
    padding: 10px 12px;
  }}
  .sub-stat span {{
    display: block;
    font-size: 11px;
    font-weight: 600;
    color: #64748B;
  }}
  .sub-stat strong {{
    display: block;
    font-family: var(--font-mono);
    font-size: 15px;
    font-weight: 700;
    margin-top: 3px;
    color: #0F172A;
  }}

  /* Generic Section Card */
  .card {{
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 14px;
    padding: 20px 22px;
    margin-bottom: 20px;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.05), 0 8px 24px rgba(15, 23, 42, 0.04);
  }}
  .card-head {{
    display: flex;
    flex-wrap: wrap;
    justify-content: space-between;
    align-items: center;
    gap: 10px;
    padding-bottom: 12px;
    margin-bottom: 14px;
    border-bottom: 1px solid #E2E8F0;
  }}
  .card-head h3 {{
    margin: 0;
    font-size: 16px;
    font-weight: 700;
    color: #0F172A;
  }}
  .card-head p {{
    margin: 2px 0 0 0;
    font-size: 12.4px;
    color: #64748B;
  }}
  .controls-row {{
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    align-items: center;
  }}

  .toggle-group {{
    display: inline-flex;
    background: #F1F5F9;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    padding: 3px;
    gap: 3px;
  }}
  .t-btn {{
    background: transparent;
    border: none;
    color: #475569;
    font-family: var(--font-sans);
    font-size: 12px;
    font-weight: 600;
    padding: 5px 11px;
    border-radius: 6px;
    cursor: pointer;
    transition: all 0.14s ease;
  }}
  .t-btn:hover {{ color: #0F172A; }}
  .t-btn.active {{
    background: #0F172A;
    color: #FFFFFF;
  }}

  /* Tables (0 Horizontal Scroll) */
  table.clean-tbl {{
    width: 100%;
    table-layout: fixed;
    border-collapse: collapse;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    overflow: hidden;
  }}
  table.clean-tbl th, table.clean-tbl td {{
    border: 1px solid #E2E8F0;
    padding: 10px 8px;
    font-size: 12.2px;
    vertical-align: middle;
    word-wrap: break-word;
  }}
  table.clean-tbl th {{
    background: #F1F5F9;
    color: #334155;
    font-weight: 700;
    font-size: 11.4px;
    text-align: right;
  }}
  table.clean-tbl th:first-child, table.clean-tbl td:first-child {{
    text-align: left;
  }}
  table.clean-tbl td {{ text-align: right; color: #0F172A; }}
  tr.row-total {{ background: #F0FDF4; border-top: 2px solid #86EFAC; }}
  .tag-blind {{
    display: inline-block;
    font-size: 10.2px;
    font-weight: 700;
    color: #0369A1;
    background: #E0F2FE;
    border: 1px solid #BAE6FD;
    padding: 1px 6px;
    border-radius: 5px;
    margin-left: 4px;
  }}
  .tag-total {{
    display: inline-block;
    font-size: 10.2px;
    font-weight: 700;
    color: #047857;
    background: #D1FAE5;
    border: 1px solid #6EE7B7;
    padding: 1px 6px;
    border-radius: 5px;
    margin-left: 4px;
  }}

  /* Monthly Heatmap Table */
  table.hm-tbl {{
    width: 100%;
    table-layout: fixed;
    border-collapse: collapse;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    overflow: hidden;
  }}
  table.hm-tbl th, table.hm-tbl td {{
    border: 1px solid #E2E8F0;
    padding: 8px 3px;
    text-align: center;
    font-size: 11.5px;
  }}
  table.hm-tbl th {{
    background: #F1F5F9;
    color: #334155;
    font-weight: 700;
  }}
  .hm-top {{ font-weight: 700; line-height: 1.25; }}
  .hm-sub {{ font-size: 10px; font-weight: 500; line-height: 1.2; margin-top: 2px; }}
  .hm-year {{
    text-align: left !important;
    padding-left: 10px !important;
    font-weight: 700;
    color: #0F172A;
    background: #F8FAFC;
  }}
  .hm-year small {{
    display: block;
    font-weight: 500;
    font-size: 10px;
    color: #64748B;
  }}
  .hm-ann {{
    font-weight: 800;
    border-left: 2px solid #CBD5E1 !important;
  }}

  /* 5. Safety Architecture Cards */
  .arch-grid {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
    margin-bottom: 36px;
  }}
  @media (max-width: 1000px) {{
    .arch-grid {{ grid-template-columns: repeat(2, 1fr); }}
  }}
  @media (max-width: 600px) {{
    .arch-grid {{ grid-template-columns: 1fr; }}
  }}
  .arch-card {{
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-top: 3px solid #059669;
    border-radius: 12px;
    padding: 16px 18px;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
  }}
  .arch-card h4 {{
    margin: 0 0 6px 0;
    font-size: 14px;
    font-weight: 700;
    color: #0F172A;
  }}
  .arch-card p {{
    margin: 0;
    font-size: 12.4px;
    color: #475569;
    line-height: 1.48;
  }}
  .footer-note {{
    text-align: center;
    font-size: 11.8px;
    color: #64748B;
    padding-bottom: 36px;
  }}
</style>
</head>
<body>

<!-- 1. EXECUTIVE HERO HEADER (v3.0 — 36M SHOWCASE + BLIND Y3 OOS) -->
<header class="exec-hero">
  <div class="shell">
    <div class="hero-top">
      <div>
        <div class="eyebrow">CHICKTRADE v3.0 &middot; {n_setups} Setups &middot; S2 50 Bậc + Cap $300k/$150k + U3_HTF_MACRO_ATR</div>
        <h1>CHICKTRADE v3.0 &mdash; BÁO CÁO TỔNG QUAN 36 THÁNG (U3_HTF_MACRO_ATR &middot; S2 50 BẬC)</h1>
        <p class="hero-sub"><strong>S2 &middot; 50 Bậc Lãi Kép (đệm -10% mới hạ bậc) + Trần Thanh Khoản ($300k BTC/ETH, $150k Alt) + U3_HTF_MACRO_ATR (1U biến thiên theo nến 4H)</strong>: Train 2Y (<code>11/2023&rarr;10/2025</code>) &amp; Thi Mù Năm 3 (<code>11/2025&rarr;10/2026</code>)</p>
      </div>
      <div class="oos-verified-badge">
        <span class="b-lbl">VƯỢT BÀI THI MÙ NĂM 3 (11/2025&ndash;10/2026)</span>
        <span class="b-val num">+{y3_250['ret']:.2f}% (Auto-Turbo, DD {y3_250['mdd']:.2f}%) / +{y3_400['ret']:.2f}% (Bao 400G, DD {y3_400['mdd']:.2f}%)</span>
        <span class="b-sub num">Full 36M: +{f36_250['ret']:,.1f}% (${f36_250['final_eq']:,.0f}) Turbo &middot; +{f36_400['ret']:,.1f}% (${f36_400['final_eq']:,.0f}) 400G</span>
      </div>
    </div>

    <div class="meta-strip">
      <span><b>Toàn chuỗi 36 tháng:</b> <span class="nw"><code>01/11/2023 &rarr; 31/10/2026</code></span> <span class="nw">(2 Năm Train <code>11/2023&ndash;10/2025</code> + 1 Năm Thi Mù OOS <code>11/2025&ndash;10/2026</code> &middot; 6 Coin Futures &middot; Vốn <code>$10,000</code>)</span></span>
      <span><b>Lãi kép &amp; Thanh khoản:</b> <span class="nw"><code>S2 · 50 Bậc (+8%/bậc, đệm -10%)</code></span> &middot; <span class="nw">Cap <code>$300k BTC/ETH · $150k Alt</code></span></span>
      <span><b>Thước đo 1U &amp; Phí:</b> <span class="nw"><code>U3_HTF_MACRO_ATR (4H EMA20 &times; &gamma;_tf)</code></span> &middot; <span class="nw">Phí <code>0.02%/0.05%</code> + trượt <code>0.01%</code></span></span>
    </div>
  </div>
</header>

<main class="shell">

  <!-- 2. TWO HERO KPI CARDS (36M COMPOUNDING: AUTO-TURBO 250G & BAO 400G) -->
  <section class="kpi-dual-grid">
    <div class="tear-kpi-card" style="border-top: 4px solid #EA580C;">
      <div class="tear-kpi-head">
        <h2>Chế Độ Tăng Tốc Auto-Turbo (400G &rarr; 250G)</h2>
        <span class="pill-turbo">Khóa Sàn Cứng &ge; $11,000</span>
      </div>
      <div class="hero-figure-row">
        <span class="hero-eq" style="color:#C2410C;">$10,000 &rarr; ${f36_250['final_eq']:,.0f}</span>
        <span class="hero-ret-badge" style="background:#FFF7ED;color:#C2410C;border:1px solid #FED7AA;">+{f36_250['ret']:,.1f}% (+${pnl36_250:,.0f})</span>
      </div>
      <p class="tear-desc">Chạy thông suốt 36 tháng (11/2023 &ndash; 10/2026) trên 1 tài khoản chung $10,000: nhân <strong>&times;{mult_250:.2f} vốn</strong> với sụt giảm tối đa chỉ <strong>{f36_250['mdd']:.2f}%</strong> và <strong>Calmar {f36_250['calmar']:.1f}</strong> (Blind Y3: <strong>+{y3_250['ret']:.2f}%</strong>).</p>
      <div class="sub-stats-grid">
        <div class="sub-stat"><span>Số Lệnh 36M</span><strong>{f36_250['trades']:,d}</strong></div>
        <div class="sub-stat"><span>PF / WR / Calmar</span><strong style="color:#059669;">{f36_250['pf']:.2f} / {f36_250['wr']:.1f}% / {f36_250['calmar']:.1f}</strong></div>
        <div class="sub-stat"><span>Max Drawdown 36M</span><strong style="color:#DC2626;">{f36_250['mdd']:.2f}%</strong></div>
        <div class="sub-stat"><span>Tháng Có Lãi</span><strong style="color:#059669;">{green_250}/{tot_m} ({green_250/tot_m*100:.0f}%)</strong></div>
      </div>
    </div>

    <div class="tear-kpi-card" style="border-top: 4px solid #059669;">
      <div class="tear-kpi-head">
        <h2>Chế Độ Tiêu Chuẩn An Toàn (Bao 400 Giá)</h2>
        <span class="pill-primary">Mặc Định An Toàn Tối Đa</span>
      </div>
      <div class="hero-figure-row">
        <span class="hero-eq" style="color:#15803D;">$10,000 &rarr; ${f36_400['final_eq']:,.0f}</span>
        <span class="hero-ret-badge" style="background:#ECFDF5;color:#15803D;border:1px solid #A7F3D0;">+{f36_400['ret']:,.1f}% (+${pnl36_400:,.0f})</span>
      </div>
      <p class="tear-desc">Đường cong vốn êm mượt nhất 36 tháng: nhân <strong>&times;{mult_400:.2f} vốn</strong> với Max Drawdown kỷ lục chỉ <strong>{f36_400['mdd']:.2f}%</strong> và <strong>Calmar {f36_400['calmar']:.1f}</strong> (Blind Y3: <strong>+{y3_400['ret']:.2f}%</strong>, DD {y3_400['mdd']:.2f}%).</p>
      <div class="sub-stats-grid">
        <div class="sub-stat"><span>Số Lệnh 36M</span><strong>{f36_400['trades']:,d}</strong></div>
        <div class="sub-stat"><span>PF / WR / Calmar</span><strong style="color:#059669;">{f36_400['pf']:.2f} / {f36_400['wr']:.1f}% / {f36_400['calmar']:.1f}</strong></div>
        <div class="sub-stat"><span>Max Drawdown 36M</span><strong style="color:#DC2626;">{f36_400['mdd']:.2f}%</strong></div>
        <div class="sub-stat"><span>Tháng Có Lãi</span><strong style="color:#059669;">{green_400}/{tot_m} ({green_400/tot_m*100:.0f}%)</strong></div>
      </div>
    </div>
  </section>

  <!-- 3. INTERACTIVE 36-MONTH EQUITY & DRAWDOWN CHART (TRAIN 2Y vs BLIND Y3 OOS) -->
  <section class="card">
    <div class="card-head">
      <div>
        <h3>Đường Cong Tăng Trưởng Tài Sản &amp; Drawdown 36 Tháng (01/11/2023 &rarr; 31/10/2026)</h3>
        <p>Phân vùng rõ <strong>2 Năm Train (11/2023 &ndash; 10/2025)</strong> và <strong>1 Năm Thi Mù OOS (11/2025 &ndash; 10/2026)</strong> &middot; Bao 400G vs Auto-Turbo 250G</p>
      </div>
      <div class="toggle-group">
        <button type="button" class="t-btn active" id="btn-scale-linear" onclick="setChartScale('linear', this)">Thang Tuyến Tính ($)</button>
        <button type="button" class="t-btn" id="btn-scale-log" onclick="setChartScale('log', this)">Thang Logarit (&times; Vốn)</button>
      </div>
    </div>
    <div id="chart-36m-equity-dd" style="width:100%;height:540px;"></div>
  </section>

  <!-- 4A. 5-ROW PERIOD SUMMARY TABLE -->
  <section class="card">
    <div class="card-head">
      <div>
        <h3>Bảng Tổng Kết Từng Năm &amp; Chuỗi 36 Tháng (Train 2Y &rarr; Thi Mù Năm 3 OOS)</h3>
        <p>Hiển thị đầy đủ <strong>% Lãi kỳ đó</strong>, <strong>Lãi USD (+$)</strong> và <strong>% Trên Vốn Gốc $10,000</strong></p>
      </div>
    </div>
    <table class="clean-tbl">
      <colgroup>
        <col style="width:25%">
        <col style="width:6%">
        <col style="width:6%">
        <col style="width:5%">
        <col style="width:8%">
        <col style="width:15%">
        <col style="width:7%">
        <col style="width:8%">
        <col style="width:13%">
        <col style="width:7%">
      </colgroup>
      <thead>
        <tr>
          <th>Giai Đoạn Kiểm Chứng</th>
          <th>Lệnh</th>
          <th>WR</th>
          <th>PF</th>
          <th>400G (%)</th>
          <th>400G Lãi ($ &amp; % Gốc)</th>
          <th>400G DD</th>
          <th>Turbo (%)</th>
          <th>Turbo Lãi ($ &amp; % Gốc)</th>
          <th>Turbo DD</th>
        </tr>
      </thead>
      <tbody>
        {''.join(period_rows_html)}
      </tbody>
    </table>
  </section>

  <!-- 4B. 36-MONTH RETURNS HEATMAP (DEFAULT: BAO 400G) -->
  <section class="card">
    <div class="card-head">
      <div>
        <h3>Bảng Nhiệt Lợi Nhuận 36 Tháng (11/2023 &rarr; 10/2026 &middot; Viền xanh dương = Năm 3 Thi Mù OOS)</h3>
        <p id="hm-subtitle"><strong>Bao 400G Chuẩn</strong>: <strong style="color:#059669;">{green_400}/{tot_m} tháng có lãi ({green_400/tot_m*100:.1f}%)</strong> &middot; Auto-Turbo 250G: {green_250}/{tot_m} tháng có lãi ({green_250/tot_m*100:.1f}%)</p>
      </div>
      <div class="controls-row">
        <div class="toggle-group">
          <button type="button" class="t-btn active" onclick="setHmBuffer('400', this)">Bao 400G Chuẩn</button>
          <button type="button" class="t-btn" onclick="setHmBuffer('250', this)">Auto-Turbo 250G</button>
        </div>
        <div class="toggle-group">
          <button type="button" class="t-btn active" onclick="setHmView('pct', this)">% Theo Tháng</button>
          <button type="button" class="t-btn" onclick="setHmView('base', this)">Lãi $ &amp; % Vốn Gốc ($10k)</button>
        </div>
      </div>
    </div>

    <div id="hm-wrap-400-pct">
      <table class="hm-tbl">
        <colgroup><col style="width:14%"><col style="width:6.5%"><col style="width:6.5%"><col style="width:6.5%"><col style="width:6.5%"><col style="width:6.5%"><col style="width:6.5%"><col style="width:6.5%"><col style="width:6.5%"><col style="width:6.5%"><col style="width:6.5%"><col style="width:6.5%"><col style="width:6.5%"><col style="width:8%"></colgroup>
        <thead><tr><th>Năm</th><th>T1</th><th>T2</th><th>T3</th><th>T4</th><th>T5</th><th>T6</th><th>T7</th><th>T8</th><th>T9</th><th>T10</th><th>T11</th><th>T12</th><th>Cả Năm</th></tr></thead>
        <tbody>{render_heatmap_tbody(hm_400, ann_400, 'pct')}</tbody>
      </table>
    </div>
    <div id="hm-wrap-400-base" style="display:none;">
      <table class="hm-tbl">
        <colgroup><col style="width:14%"><col style="width:6.5%"><col style="width:6.5%"><col style="width:6.5%"><col style="width:6.5%"><col style="width:6.5%"><col style="width:6.5%"><col style="width:6.5%"><col style="width:6.5%"><col style="width:6.5%"><col style="width:6.5%"><col style="width:6.5%"><col style="width:6.5%"><col style="width:8%"></colgroup>
        <thead><tr><th>Năm</th><th>T1</th><th>T2</th><th>T3</th><th>T4</th><th>T5</th><th>T6</th><th>T7</th><th>T8</th><th>T9</th><th>T10</th><th>T11</th><th>T12</th><th>Cả Năm</th></tr></thead>
        <tbody>{render_heatmap_tbody(hm_400, ann_400, 'base')}</tbody>
      </table>
    </div>
    <div id="hm-wrap-250-pct" style="display:none;">
      <table class="hm-tbl">
        <colgroup><col style="width:14%"><col style="width:6.5%"><col style="width:6.5%"><col style="width:6.5%"><col style="width:6.5%"><col style="width:6.5%"><col style="width:6.5%"><col style="width:6.5%"><col style="width:6.5%"><col style="width:6.5%"><col style="width:6.5%"><col style="width:6.5%"><col style="width:6.5%"><col style="width:8%"></colgroup>
        <thead><tr><th>Năm</th><th>T1</th><th>T2</th><th>T3</th><th>T4</th><th>T5</th><th>T6</th><th>T7</th><th>T8</th><th>T9</th><th>T10</th><th>T11</th><th>T12</th><th>Cả Năm</th></tr></thead>
        <tbody>{render_heatmap_tbody(hm_250, ann_250, 'pct')}</tbody>
      </table>
    </div>
    <div id="hm-wrap-250-base" style="display:none;">
      <table class="hm-tbl">
        <colgroup><col style="width:14%"><col style="width:6.5%"><col style="width:6.5%"><col style="width:6.5%"><col style="width:6.5%"><col style="width:6.5%"><col style="width:6.5%"><col style="width:6.5%"><col style="width:6.5%"><col style="width:6.5%"><col style="width:6.5%"><col style="width:6.5%"><col style="width:6.5%"><col style="width:8%"></colgroup>
        <thead><tr><th>Năm</th><th>T1</th><th>T2</th><th>T3</th><th>T4</th><th>T5</th><th>T6</th><th>T7</th><th>T8</th><th>T9</th><th>T10</th><th>T11</th><th>T12</th><th>Cả Năm</th></tr></thead>
        <tbody>{render_heatmap_tbody(hm_250, ann_250, 'base')}</tbody>
      </table>
    </div>
  </section>

  <!-- 4C. COMPACT 6-ROW U3_HTF_MACRO_ATR 1U / SL / TP1 / TP2 CONVERSION TABLE -->
  <section class="card">
    <div class="card-head">
      <div>
        <h3>Bảng Quy Đổi 1U / SL / TP1 / TP2 Theo Chuẩn v3.0 (<code>U3_HTF_MACRO_ATR</code>)</h3>
        <p>Trung vị 36 tháng của <code>1U(t) = Close(t) &times; max(0.08%, EMA20_4H(0.5&times;ATR14_4H/Close_4H) &times; &gamma;_tf)</code> với <code>&gamma;_15m={GAMMA_4H['15m']}, &gamma;_30m={GAMMA_4H['30m']}, &gamma;_1h={GAMMA_4H['1h']}</code></p>
      </div>
    </div>
    <table class="clean-tbl">
      <colgroup>
        <col style="width:12%">
        <col style="width:24%">
        <col style="width:24%">
        <col style="width:24%">
        <col style="width:16%">
      </colgroup>
      <thead>
        <tr>
          <th>Đồng Coin</th>
          <th>Khung 15m (1U% &middot; SL/TP1/TP2%)</th>
          <th>Khung 30m (1U% &middot; SL/TP1/TP2%)</th>
          <th>Khung 1h (1U% &middot; SL/TP1/TP2%)</th>
          <th>Chuẩn SL / TP2 (U)</th>
        </tr>
      </thead>
      <tbody>
        {render_u3_macro_table_html(tf_dfs_map, u3_arrays)}
      </tbody>
    </table>
  </section>

  <!-- 5. 4 SAFETY & U3_HTF_MACRO_ATR EXPLANATION CARDS -->
  <section class="arch-grid">
    <div class="arch-card" style="border-top-color:#0284C7;">
      <h4>1. <code>U3_HTF_MACRO_ATR</code> (1U Biến Thiên 4H)</h4>
      <p><code>1U</code> biến thiên mượt theo nến <code>4H</code> đã đóng (<code>EMA20 &times; &gamma;_tf</code>, sàn <code>0.08%</code>) — loại bỏ nhiễu ngắn hạn và tự thích ứng qua mọi chu kỳ.</p>
    </div>
    <div class="arch-card" style="border-top-color:#059669;">
      <h4>2. <code>S2 · 50 Bậc</code> Lãi Kép (Đệm -10%)</h4>
      <p>Khóa cứng volume theo bậc đang đứng (tăng bậc mỗi <code>+8%</code>, tối đa 50 bậc) và chỉ hạ 1 bậc khi tụt quá <code>-10%</code> dưới mốc bậc, giúp hồi phục DD nhanh.</p>
    </div>
    <div class="arch-card" style="border-top-color:#EA580C;">
      <h4>3. Trần Thanh Khoản ($300k / $150k)</h4>
      <p>Áp trần thực chiến mỗi lệnh tối đa <code>$300,000</code> cho BTC/ETH và <code>$150,000</code> cho Altcoin, kèm Auto-Turbo 250G khóa cứng sàn bảo vệ <code>$11,000</code>.</p>
    </div>
    <div class="arch-card" style="border-top-color:#6D28D9;">
      <h4>4. 58 Setup Xuyên 3 Năm &amp; Phanh 2-SL</h4>
      <p>Gồm <strong>38 Core (0.85&times;)</strong> và <strong>20 Core Mở Rộng (0.55&times;)</strong> vượt lọc kép 2 năm Train, kèm phanh <code>2-SL &rarr; Shadow-Win</code> bảo vệ danh mục 6 Coin.</p>
    </div>
  </section>

  <footer class="footer-note">
    CHICKTRADE v3.0 (<code>U3_HTF_MACRO_ATR</code> &middot; <code>S2 50 Bậc</code> &middot; Cap <code>$300k/$150k</code>) &middot; 36 tháng (01/11/2023 &ndash; 31/10/2026) &middot; 58 Setup Train 2Y, Thi Mù OOS Năm 3.
  </footer>

</main>

<script>
  const x400 = {json.dumps(x_400)};
  const yEq400 = {json.dumps(y_eq_400)};
  const yDd400 = {json.dumps(y_dd_400)};
  const x250 = {json.dumps(x_250)};
  const yEq250 = {json.dumps(y_eq_250)};
  const yDd250 = {json.dumps(y_dd_250)};

  const traces = [
    {{
      name: 'Auto-Turbo 250G ($10k \u2192 ${f36_250["final_eq"]:,.0f} · +{f36_250["ret"]:,.1f}%)',
      x: x250,
      y: yEq250,
      type: 'scatter',
      mode: 'lines',
      line: {{ color: '#EA580C', width: 2.6 }},
      xaxis: 'x',
      yaxis: 'y',
      hovertemplate: '<b>Auto-Turbo 250G</b><br>Ngày: %{{x|%d/%m/%Y}}<br>Tài sản: <b>$%{{y:,.0f}}</b><extra></extra>'
    }},
    {{
      name: 'Bao 400G Chuẩn ($10k \u2192 ${f36_400["final_eq"]:,.0f} · +{f36_400["ret"]:,.1f}%)',
      x: x400,
      y: yEq400,
      type: 'scatter',
      mode: 'lines',
      line: {{ color: '#059669', width: 2.5 }},
      xaxis: 'x',
      yaxis: 'y',
      hovertemplate: '<b>Bao 400G Chuẩn</b><br>Ngày: %{{x|%d/%m/%Y}}<br>Tài sản: <b>$%{{y:,.0f}}</b><extra></extra>'
    }},
    {{
      name: 'Drawdown Bao 400G (%)',
      x: x400,
      y: yDd400,
      type: 'scatter',
      mode: 'lines',
      fill: 'tozeroy',
      fillcolor: 'rgba(5, 150, 105, 0.12)',
      line: {{ color: '#059669', width: 1.5 }},
      xaxis: 'x',
      yaxis: 'y2',
      hovertemplate: '<b>DD Bao 400G</b>: %{{y:.2f}}%<extra></extra>'
    }},
    {{
      name: 'Drawdown Turbo 250G (%)',
      x: x250,
      y: yDd250,
      type: 'scatter',
      mode: 'lines',
      fill: 'tozeroy',
      fillcolor: 'rgba(234, 88, 12, 0.08)',
      line: {{ color: '#EA580C', width: 1.2 }},
      xaxis: 'x',
      yaxis: 'y2',
      hovertemplate: '<b>DD Turbo 250G</b>: %{{y:.2f}}%<extra></extra>'
    }}
  ];

  const baseLayout = {{
    paper_bgcolor: '#FFFFFF',
    plot_bgcolor: '#F8FAFC',
    font: {{ color: '#334155', size: 12, family: 'Inter, sans-serif' }},
    hovermode: 'x unified',
    legend: {{ orientation: 'h', y: 1.06, x: 0 }},
    margin: {{ t: 36, r: 24, l: 78, b: 42 }},
    xaxis: {{
      gridcolor: '#E2E8F0',
      showspikes: true,
      spikecolor: '#94A3B8',
      spikethickness: 1
    }},
    yaxis: {{
      title: 'Tài Sản Tài Khoản ($ USD)',
      domain: [0.34, 1.0],
      gridcolor: '#E2E8F0',
      tickprefix: '$',
      type: 'linear'
    }},
    yaxis2: {{
      title: 'Drawdown (%)',
      domain: [0.0, 0.27],
      gridcolor: '#E2E8F0',
      ticksuffix: '%',
      range: [-16, 1]
    }},
    shapes: [
      {{
        type: 'rect',
        xref: 'x', yref: 'paper',
        x0: '2025-11-01T00:00:00Z', x1: '2026-10-31T23:55:00Z',
        y0: 0, y1: 1,
        fillcolor: 'rgba(2, 132, 199, 0.06)',
        line: {{ width: 0 }}
      }},
      {{
        type: 'line',
        xref: 'x', yref: 'paper',
        x0: '2025-11-01T00:00:00Z', x1: '2025-11-01T00:00:00Z',
        y0: 0, y1: 1,
        line: {{ color: '#0284C7', width: 2.0, dash: 'dash' }}
      }}
    ],
    annotations: [
      {{
        x: '2024-11-01T00:00:00Z',
        y: 0.97,
        xref: 'x', yref: 'paper',
        text: '<b>2 NĂM TRAIN (11/2023 \u2013 10/2025)</b>',
        showarrow: false,
        font: {{ size: 11.5, color: '#047857' }},
        bgcolor: '#ECFDF5',
        bordercolor: '#A7F3D0',
        borderwidth: 1,
        borderpad: 4
      }},
      {{
        x: '2026-05-01T00:00:00Z',
        y: 0.97,
        xref: 'x', yref: 'paper',
        text: '<b>1 NĂM THI MÙ OOS (11/2025 \u2013 10/2026)</b>',
        showarrow: false,
        font: {{ size: 11.5, color: '#0369A1' }},
        bgcolor: '#F0F9FF',
        bordercolor: '#BAE6FD',
        borderwidth: 1,
        borderpad: 4
      }}
    ]
  }};

  Plotly.newPlot('chart-36m-equity-dd', traces, baseLayout, {{ responsive: true, displayModeBar: false }});

  function setChartScale(scaleType, btn) {{
    document.getElementById('btn-scale-linear').classList.toggle('active', scaleType === 'linear');
    document.getElementById('btn-scale-log').classList.toggle('active', scaleType === 'log');
    Plotly.relayout('chart-36m-equity-dd', {{ 'yaxis.type': scaleType }});
  }}

  let curHmBuf = '400';
  let curHmView = 'pct';

  function renderHmVisibility() {{
    for (const b of ['400', '250']) {{
      for (const v of ['pct', 'base']) {{
        document.getElementById('hm-wrap-' + b + '-' + v).style.display = (b === curHmBuf && v === curHmView) ? '' : 'none';
      }}
    }}
    const sub = document.getElementById('hm-subtitle');
    if (curHmBuf === '400') {{
      sub.innerHTML = '<strong>Bao 400G Chuẩn</strong>: <strong style="color:#059669;">{green_400}/{tot_m} tháng có lãi ({green_400/tot_m*100:.1f}%)</strong> &middot; Auto-Turbo 250G: {green_250}/{tot_m} tháng có lãi ({green_250/tot_m*100:.1f}%)';
    }} else {{
      sub.innerHTML = '<strong>Auto-Turbo 250G</strong>: <strong style="color:#059669;">{green_250}/{tot_m} tháng có lãi ({green_250/tot_m*100:.1f}%)</strong> &middot; Bao 400G: {green_400}/{tot_m} tháng có lãi ({green_400/tot_m*100:.1f}%)';
    }}
  }}

  function setHmBuffer(buf, btn) {{
    btn.parentElement.querySelectorAll('.t-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    curHmBuf = buf;
    renderHmVisibility();
  }}

  function setHmView(vw, btn) {{
    btn.parentElement.querySelectorAll('.t-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    curHmView = vw;
    renderHmVisibility();
  }}
</script>
</body>
</html>
"""

    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(html_doc)
    print(f"[DONE] Built v3.0 36-Month Showcase HTML Report (U3_HTF_MACRO_ATR, {n_setups} setups) at: {OUT_HTML}")


if __name__ == "__main__":
    main()
