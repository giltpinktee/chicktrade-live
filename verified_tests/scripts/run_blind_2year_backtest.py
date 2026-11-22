"""True Blind Out-Of-Sample (OOS) 2-Year Backtest (2023-11-01 -> 2025-10-31).

Zero overlap with the 1-year training/OOS window (2025-11-01 -> 2026-10-31).
Reads the EXACT frozen 75 setups and Rule 1B directional locks from:
  - e:\CHICKTRADE\core\core_strategy_rules.json
Runs the exact sim_portfolio_advanced engine from run_alpha_discovery_argon.py without
modifying any files in core/, bot/, or existing verified_tests scripts.

Outputs:
  - e:\CHICKTRADE\verified_tests\results\blind_2year_summary.csv
  - e:\CHICKTRADE\verified_tests\results\blind_2year_breakdown.csv
  - e:\CHICKTRADE\verified_tests\results\blind_2year_curves.json
"""
from concurrent.futures import ThreadPoolExecutor
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

from config import BacktestConfig, RiskConfig
from data.binance_loader import download_symbol_1y
from risk.compounding import CompoundingManager
from run_alpha_discovery_argon import sim_portfolio_advanced
from run_full_216_exit_sweetspot import sim_exit_arch
from run_full_3ema_storm_guard import build_cooldown_mask, compute_storm_flags, generate_all_4_touches
from run_master_6coin_html import SIX_COINS
from run_research_mtf import resample_ohlcv

RULES_PATH = os.path.join(ROOT, "core", "core_strategy_rules.json")
RESULTS_DIR = os.path.join(VERIFIED_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_frozen_rules() -> Tuple[dict, dict, dict, Dict[Tuple[str, str, str], int]]:
    """Load exact frozen 75 setups, 46 (v2.1) subset, 39 CORE-only subset, and Rule 1B dir_filter."""
    with open(RULES_PATH, encoding="utf-8") as f:
        cfg = json.load(f)

    rules_75: dict = {}
    rules_46: dict = {}
    rules_39_core: dict = {}
    dir_filter: Dict[Tuple[str, str, str], int] = {}

    for idx, r in enumerate(cfg["rules"]):
        k = (r["coin"], r["tf"], r["setup_code"])
        spec = {
            "arch": r["exit_arch"],
            "l2_u": float(r["leg2_sl_u"]),
            "sl_u": float(r["sl_u"]),
            "tp1_u": float(r["tp1_u"]),
            "tp2_u": float(r["tp2_u"]),
            "vol": float(r["adaptive_volume"]["init_vol_mult"]),
            "status": str(r["status"]),
        }
        rules_75[k] = spec
        if idx < 46:
            rules_46[k] = spec
        if r["status"] in ("CORE", "CORE_EXP"):
            rules_39_core[k] = spec

        ad = r.get("allowed_direction", "BOTH")
        if ad == "LONG_ONLY":
            dir_filter[k] = 1
        elif ad == "SHORT_ONLY":
            dir_filter[k] = -1

    return rules_75, rules_46, rules_39_core, dir_filter


def filter_sigs_by_window(
    tf_sigs_map: dict,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> dict:
    """Return a filtered tf_sigs_map strictly within [start_ts, end_ts] inclusive."""
    out = {}
    for tfl, sm in tf_sigs_map.items():
        out[tfl] = {
            sym: [s for s in sigs if start_ts <= s.timestamp <= end_ts]
            for sym, sigs in sm.items()
        }
    return out


def sim_portfolio_with_trace(
    tf_dfs_map: dict,
    tf_sigs_map: dict,
    rules_map: dict,
    dir_filter_map: Optional[Dict[Tuple[str, str, str], int]] = None,
    base_buffer: float = 400.0,
    auto_turbo_buffer: Optional[float] = None,
    hard_floor_eq: float = 11_000.0,
    core_only_sl_trigger: bool = True,
    window_start_ts: Optional[pd.Timestamp] = None,
) -> Tuple[dict, List[dict], List[dict]]:
    """Mirror sim_portfolio_advanced identically while capturing closed trade records and timestamped equity curve."""
    rcfg = RiskConfig(initial_capital=10_000.0, buffer_units=base_buffer, max_concurrent_trades=5, max_trades_per_day=10)
    comp = CompoundingManager(rcfg)
    equity = 10_000.0
    peak_eq = equity
    open_cap = 4.25
    in_turbo = False
    ever_triggered_turbo = False
    min_eq_after_turbo = 10_000.0

    cands = []
    for tfl, sm in tf_sigs_map.items():
        for sym, sigs in sm.items():
            for s in sigs:
                k = (sym, tfl, s.setup_type)
                if k not in rules_map:
                    continue
                if dir_filter_map and k in dir_filter_map and s.direction != dir_filter_map[k]:
                    continue
                cands.append((tfl, rules_map[k], s))

    tf_prio = {"1h": 0, "30m": 1, "15m": 2, "5m": 3}
    cands.sort(key=lambda x: (x[2].timestamp, tf_prio[x[0]], -x[1]["vol"], x[2].symbol))

    active, closed = [], []
    daily_t, daily_pnl, daily_eq = {}, {}, {}
    eq_curve = [equity]
    init_ts_str = window_start_ts.isoformat() if window_start_ts is not None else "2023-11-01T00:00:00+00:00"
    curve_points = [{"timestamp": init_ts_str, "equity": round(equity, 2), "dd_pct": 0.0}]

    port_consec_sl = 0
    port_shadow_paused = False
    port_paused_until = pd.Timestamp("1970-01-01", tz="UTC")

    def settle(now_ts: pd.Timestamp):
        nonlocal equity, peak_eq, port_consec_sl, port_shadow_paused, port_paused_until
        nonlocal in_turbo, ever_triggered_turbo, min_eq_after_turbo
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

            if ever_triggered_turbo:
                min_eq_after_turbo = min(min_eq_after_turbo, equity)

            if auto_turbo_buffer is not None:
                if (not in_turbo) and equity >= 12_500.0:
                    in_turbo = True
                    ever_triggered_turbo = True
                    min_eq_after_turbo = min(min_eq_after_turbo, equity)
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
                "timestamp": ex_ts.isoformat(),
                "equity": round(equity, 2),
                "dd_pct": round(dd_pct, 4),
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
        u = max(0.5 * sig.atr, sig.entry_price * 1e-5)

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

        cur_buf = base_buffer
        if auto_turbo_buffer is not None and in_turbo:
            cur_buf = auto_turbo_buffer
            sb_test = min(comp.tier_capital, equity)
            open_sl_risk = sum(tr["worst_sl_usd"] for _, tr in active)
            new_sl_risk = (sb_test * eff_m) * (rule["sl_u"] / cur_buf) * 1.05
            if equity - (open_sl_risk + new_sl_risk) < hard_floor_eq:
                cur_buf = base_buffer

        out, _, _, _, nu = sim_exit_arch(
            df["high"].values, df["low"].values, df["close"].values,
            sig.bar_idx, sig.entry_price, sig.direction, u,
            rule["sl_u"], rule["tp1_u"], rule["tp2_u"], rule["arch"], rule["l2_u"],
        )
        sb = min(comp.tier_capital, equity)
        qty = (sb * eff_m) / (cur_buf * u)
        pnl = nu * u * qty
        worst_sl_usd = (sb * eff_m) * (rule["sl_u"] / cur_buf) * 1.05

        ex_i = min(sig.bar_idx + (4 if out == "SL" else 8), len(df) - 1)
        ex_ts = pd.Timestamp(df["timestamp"].iloc[ex_i])
        if ex_ts.tzinfo is None:
            ex_ts = ex_ts.tz_localize("UTC")

        active.append((ex_ts, {
            "sym": sig.symbol,
            "tf": tfl,
            "setup_code": sig.setup_type,
            "status": rule["status"],
            "dir": sig.direction,
            "base_vol": rule["vol"],
            "m": eff_m,
            "out": out,
            "nu": float(nu),
            "pnl": float(pnl),
            "dk": dk,
            "entry_ts": sig.timestamp,
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
    pf = float(pnls[pnls > 0].sum() / max(-pnls[pnls < 0].sum(), 0.01))
    ret = (float(ev[-1]) / 10_000.0 - 1.0) * 100.0
    metrics = {
        "trades": len(closed),
        "wr": wr,
        "pf": pf,
        "final_eq": float(ev[-1]),
        "ret": ret,
        "mdd": mdd,
        "calmar": ret / max(abs(mdd), 0.1),
        "min_after_turbo": min_eq_after_turbo,
    }
    return metrics, closed, curve_points


def build_monthly_returns(curve_points: List[dict], closed_trades: List[dict]) -> List[dict]:
    """Compute monthly equity, monthly_ret_pct, trades, and max intramonth DD from curve_points."""
    df_c = pd.DataFrame(curve_points)
    df_c["ts"] = pd.to_datetime(df_c["timestamp"], utc=True)
    df_c["month"] = df_c["ts"].dt.strftime("%Y-%m")

    trades_by_m: Dict[str, int] = {}
    for tr in closed_trades:
        mk = tr["exit_ts"].strftime("%Y-%m")
        trades_by_m[mk] = trades_by_m.get(mk, 0) + 1

    months = pd.period_range("2023-11", "2025-10", freq="M")
    monthly = []
    prev_eq = 10_000.0
    for per in months:
        m_str = str(per)
        sub = df_c[df_c["month"] == m_str]
        end_eq = float(sub["equity"].iloc[-1]) if len(sub) else prev_eq
        m_ret = (end_eq / prev_eq - 1.0) * 100.0 if prev_eq > 0 else 0.0
        min_dd = float(sub["dd_pct"].min()) if len(sub) else 0.0
        monthly.append({
            "month": m_str,
            "start_eq": round(prev_eq, 2),
            "end_eq": round(end_eq, 2),
            "monthly_ret_pct": round(m_ret, 2),
            "trades": trades_by_m.get(m_str, 0),
            "dd_pct": round(min_dd, 2),
        })
        prev_eq = end_eq
    return monthly


def calc_group_metrics(pnls_u: List[float], pnls_usd: Optional[List[float]] = None) -> dict:
    """Compute trades, trade WR %, profit WR %, PF, net_u, and optional net_usd."""
    n = len(pnls_u)
    if n == 0:
        return {"trades": 0, "wr_pct": 0.0, "profit_wr_pct": 0.0, "pf": 0.0, "net_u": 0.0, "net_usd": 0.0}
    au = np.array(pnls_u, dtype=float)
    wr_pct = float((au > 0).mean() * 100.0)
    gp_u = float(au[au > 0].sum())
    gl_u = float(-au[au < 0].sum())
    profit_wr_pct = float(gp_u / (gp_u + gl_u) * 100.0) if (gp_u + gl_u) > 0 else 0.0
    pf_u = float(gp_u / max(gl_u, 0.01)) if gl_u > 0.01 else (99.99 if gp_u > 0 else 0.0)

    if pnls_usd is not None and len(pnls_usd) == n:
        ausd = np.array(pnls_usd, dtype=float)
        gp_usd = float(ausd[ausd > 0].sum())
        gl_usd = float(-ausd[ausd < 0].sum())
        pf_val = float(gp_usd / max(gl_usd, 0.01)) if gl_usd > 0.01 else (99.99 if gp_usd > 0 else 0.0)
        net_usd = float(ausd.sum())
    else:
        pf_val = pf_u
        net_usd = 0.0

    return {
        "trades": n,
        "wr_pct": round(wr_pct, 2),
        "profit_wr_pct": round(profit_wr_pct, 2),
        "pf": round(pf_val, 2),
        "net_u": round(float(au.sum()), 2),
        "net_usd": round(net_usd, 2),
    }


def main():
    rules_75, rules_46, rules_39_core, dir_filter = load_frozen_rules()
    print(f"Loaded frozen rules: 75={len(rules_75)} | 46(v2.1)={len(rules_46)} | 39(CORE+CORE_EXP)={len(rules_39_core)} | DirLocks={len(dir_filter)}")

    # 1. Parallel download 24 months ending 2025-10 (2023-11-01 -> 2025-10-31)
    print("\nDownloading 24 months (2023-11 -> 2025-10) 5m Binance Futures OHLCV for 6 coins in parallel...")
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {sym: ex.submit(download_symbol_1y, sym, "5m", 24, "futures", (2025, 10)) for sym in SIX_COINS}
        raw_m5 = {sym: futs[sym].result() for sym in SIX_COINS}

    for sym in SIX_COINS:
        df = raw_m5[sym].sort_values("timestamp").reset_index(drop=True)
        raw_m5[sym] = df
        t_min, t_max = df["timestamp"].min(), df["timestamp"].max()
        print(f"  {sym:8s}: {len(df):,} bars | {t_min} -> {t_max}")
        assert str(t_min).startswith("2023-11-01"), f"Unexpected start date for {sym}: {t_min}"
        assert str(t_max).startswith("2025-10-31"), f"Unexpected end date for {sym}: {t_max}"

    # 2. Resample & Generate 4-touch signals across 24M
    cfg = BacktestConfig()
    dfs_m15 = {s: resample_ohlcv(d, "15min") for s, d in raw_m5.items()}
    dfs_m30 = {s: resample_ohlcv(d, "30min") for s, d in raw_m5.items()}
    dfs_h1 = {s: resample_ohlcv(d, "1h") for s, d in raw_m5.items()}
    dfs_h4 = {s: resample_ohlcv(d, "4h") for s, d in raw_m5.items()}
    tf_dfs_map = {"5m": raw_m5, "15m": dfs_m15, "30m": dfs_m30, "1h": dfs_h1}

    tf_sigs_map = {}
    for tfl, tdf, hdf, cd in [("5m", raw_m5, dfs_h1, 48), ("15m", dfs_m15, dfs_h1, 16), ("30m", dfs_m30, dfs_h4, 8), ("1h", dfs_h1, dfs_h4, 4)]:
        bst = compute_storm_flags(tdf["BTCUSDT"])
        bmask = build_cooldown_mask(bst["raw_storm"], cd)
        bts = tdf["BTCUSDT"]["timestamp"].values
        tf_sigs_map[tfl] = {
            sym: generate_all_4_touches(tdf[sym], hdf[sym], bts, bmask, sym, cfg.strategy, cooldown_bars=cd)[0]
            for sym in SIX_COINS
        }

    # 3. Define 4 required time windows
    windows = [
        ("1_Full_24M (2023-11-01 -> 2025-10-31)", pd.Timestamp("2023-11-01 00:00:00", tz="UTC"), pd.Timestamp("2025-10-31 23:59:59", tz="UTC")),
        ("2_Year_1   (2023-11-01 -> 2024-10-31)", pd.Timestamp("2023-11-01 00:00:00", tz="UTC"), pd.Timestamp("2024-10-31 23:59:59", tz="UTC")),
        ("3_Year_2   (2024-11-01 -> 2025-10-31)", pd.Timestamp("2024-11-01 00:00:00", tz="UTC"), pd.Timestamp("2025-10-31 23:59:59", tz="UTC")),
        ("4_Cal_2024 (2024-01-01 -> 2024-12-31)", pd.Timestamp("2024-01-01 00:00:00", tz="UTC"), pd.Timestamp("2024-12-31 23:59:59", tz="UTC")),
    ]

    summary_rows = []
    for w_name, w_start, w_end in windows:
        wsigs = filter_sigs_by_window(tf_sigs_map, w_start, w_end)
        for mode_name, turbo_buf in [("Bao_400G", None), ("Auto_Turbo_250G", 250.0)]:
            res = sim_portfolio_advanced(
                tf_dfs_map, wsigs, rules_75,
                start_time=None,
                dir_filter_map=dir_filter,
                base_buffer=400.0,
                auto_turbo_buffer=turbo_buf,
                hard_floor_eq=11_000.0,
                core_only_sl_trigger=True,
            )
            summary_rows.append({
                "Experiment": "75_Setups_v2.3",
                "Window": w_name,
                "Mode": mode_name,
                "Trades": res["trades"],
                "Trade_WR_%": round(res["wr"], 2),
                "PF": round(res["pf"], 2),
                "Final_Eq_$": round(res["final_eq"], 2),
                "Net_Profit_$": round(res["final_eq"] - 10_000.0, 2),
                "Return_%": round(res["ret"], 2),
                "MaxDD_%": round(res["mdd"], 2),
                "Calmar": round(res["calmar"], 2),
                "Min_Eq_After_Turbo_$": round(res["min_after_turbo"], 2),
            })

    # 4. Cross-check 75 Setups (v2.3) vs 46 Setups (v2.1) vs CORE-only (39 Setups) on Full 24M
    full_sigs = filter_sigs_by_window(tf_sigs_map, windows[0][1], windows[0][2])
    for exp_label, rmap in [("Compare_46_Setups_v2.1", rules_46), ("Compare_39_CORE_Only", rules_39_core)]:
        for mode_name, turbo_buf in [("Bao_400G", None), ("Auto_Turbo_250G", 250.0)]:
            res = sim_portfolio_advanced(
                tf_dfs_map, full_sigs, rmap,
                start_time=None,
                dir_filter_map=dir_filter,
                base_buffer=400.0,
                auto_turbo_buffer=turbo_buf,
                hard_floor_eq=11_000.0,
                core_only_sl_trigger=True,
            )
            summary_rows.append({
                "Experiment": exp_label,
                "Window": windows[0][0],
                "Mode": mode_name,
                "Trades": res["trades"],
                "Trade_WR_%": round(res["wr"], 2),
                "PF": round(res["pf"], 2),
                "Final_Eq_$": round(res["final_eq"], 2),
                "Net_Profit_$": round(res["final_eq"] - 10_000.0, 2),
                "Return_%": round(res["ret"], 2),
                "MaxDD_%": round(res["mdd"], 2),
                "Calmar": round(res["calmar"], 2),
                "Min_Eq_After_Turbo_$": round(res["min_after_turbo"], 2),
            })

    df_sum = pd.DataFrame(summary_rows)
    sum_path = os.path.join(RESULTS_DIR, "blind_2year_summary.csv")
    df_sum.to_csv(sum_path, index=False)

    # 5. Exact trace for 24M curves (Bao 400G & Auto-Turbo 250G) and cross-verify against sim_portfolio_advanced
    tr_400, closed_400, pts_400 = sim_portfolio_with_trace(
        tf_dfs_map, full_sigs, rules_75,
        dir_filter_map=dir_filter, base_buffer=400.0, auto_turbo_buffer=None,
        core_only_sl_trigger=True, window_start_ts=windows[0][1],
    )
    tr_250, closed_250, pts_250 = sim_portfolio_with_trace(
        tf_dfs_map, full_sigs, rules_75,
        dir_filter_map=dir_filter, base_buffer=400.0, auto_turbo_buffer=250.0,
        hard_floor_eq=11_000.0, core_only_sl_trigger=True, window_start_ts=windows[0][1],
    )
    # Parity check against sim_portfolio_advanced
    ref_400 = [r for r in summary_rows if r["Experiment"] == "75_Setups_v2.3" and r["Window"].startswith("1_Full") and r["Mode"] == "Bao_400G"][0]
    ref_250 = [r for r in summary_rows if r["Experiment"] == "75_Setups_v2.3" and r["Window"].startswith("1_Full") and r["Mode"] == "Auto_Turbo_250G"][0]
    assert abs(tr_400["final_eq"] - ref_400["Final_Eq_$"]) < 0.05, f"400G parity mismatch: {tr_400['final_eq']} vs {ref_400['Final_Eq_$']}"
    assert abs(tr_250["final_eq"] - ref_250["Final_Eq_$"]) < 0.05, f"250G parity mismatch: {tr_250['final_eq']} vs {ref_250['Final_Eq_$']}"

    curves_payload = {
        "period": "2023-11-01 -> 2025-10-31",
        "initial_capital": 10_000.0,
        "bao_400g": {
            "summary": {k: round(v, 4) if isinstance(v, float) else v for k, v in tr_400.items()},
            "monthly": build_monthly_returns(pts_400, closed_400),
            "curve": pts_400,
        },
        "auto_turbo_250g": {
            "summary": {k: round(v, 4) if isinstance(v, float) else v for k, v in tr_250.items()},
            "monthly": build_monthly_returns(pts_250, closed_250),
            "curve": pts_250,
        },
    }
    curves_path = os.path.join(RESULTS_DIR, "blind_2year_curves.json")
    with open(curves_path, "w", encoding="utf-8") as f:
        json.dump(curves_payload, f, indent=2)

    # 6. Detailed 2-year breakdown by Coin (6 coins) & by Setup Status Group (6 groups)
    # Both PORTFOLIO_400G (executed in 6-coin portfolio) & STANDALONE_ALL_SIGNALS (raw setup edge)
    standalone_trades = []
    for (sym, tfl, stype), rule in sorted(rules_75.items()):
        df = tf_dfs_map[tfl][sym]
        h_arr, l_arr, c_arr = df["high"].values, df["low"].values, df["close"].values
        for s in full_sigs[tfl][sym]:
            if s.setup_type != stype:
                continue
            if (sym, tfl, stype) in dir_filter and s.direction != dir_filter[(sym, tfl, stype)]:
                continue
            u = max(0.5 * s.atr, s.entry_price * 1e-5)
            _, _, _, _, nu = sim_exit_arch(
                h_arr, l_arr, c_arr, s.bar_idx, s.entry_price, s.direction, u,
                rule["sl_u"], rule["tp1_u"], rule["tp2_u"], rule["arch"], rule["l2_u"],
            )
            standalone_trades.append({
                "sym": sym,
                "tf": tfl,
                "setup_code": stype,
                "status": rule["status"],
                "nu": float(nu),
            })

    breakdown_rows = []
    status_order = ["CORE", "CORE_EXP", "SCOUT", "SNIPER", "SCOUT_EXP", "HOT_ADAPTIVE"]

    # Portfolio-executed breakdown (Bao 400G)
    for sym in SIX_COINS:
        sub = [t for t in closed_400 if t["sym"] == sym]
        m = calc_group_metrics([t["nu"] for t in sub], [t["pnl"] for t in sub])
        breakdown_rows.append({"Scope": "PORTFOLIO_400G", "Dimension": "COIN", "Key": sym, "Setup_Count": sum(1 for k in rules_75 if k[0] == sym), **m})
    for st in status_order:
        sub = [t for t in closed_400 if t["status"] == st]
        m = calc_group_metrics([t["nu"] for t in sub], [t["pnl"] for t in sub])
        breakdown_rows.append({"Scope": "PORTFOLIO_400G", "Dimension": "STATUS_GROUP", "Key": st, "Setup_Count": sum(1 for v in rules_75.values() if v["status"] == st), **m})

    # Standalone raw-signal breakdown (tests true per-setup edge across all 24M signals)
    for sym in SIX_COINS:
        sub = [t for t in standalone_trades if t["sym"] == sym]
        m = calc_group_metrics([t["nu"] for t in sub])
        breakdown_rows.append({"Scope": "STANDALONE_RAW", "Dimension": "COIN", "Key": sym, "Setup_Count": sum(1 for k in rules_75 if k[0] == sym), **m})
    for st in status_order:
        sub = [t for t in standalone_trades if t["status"] == st]
        m = calc_group_metrics([t["nu"] for t in sub])
        breakdown_rows.append({"Scope": "STANDALONE_RAW", "Dimension": "STATUS_GROUP", "Key": st, "Setup_Count": sum(1 for v in rules_75.values() if v["status"] == st), **m})

    df_bd = pd.DataFrame(breakdown_rows)
    bd_path = os.path.join(RESULTS_DIR, "blind_2year_breakdown.csv")
    df_bd.to_csv(bd_path, index=False)

    print("\n" + "=" * 130)
    print("BLIND 2-YEAR OOS SUMMARY (2023-11-01 -> 2025-10-31 — ZERO OVERLAP WITH TRAINING YEAR 2025-11 -> 2026-10)")
    print("=" * 130)
    print(df_sum.to_string(index=False))
    print("\n" + "=" * 130)
    print("BLIND 2-YEAR BREAKDOWN BY COIN & SETUP GROUP (PORTFOLIO_400G vs STANDALONE_RAW)")
    print("=" * 130)
    print(df_bd.to_string(index=False))
    print(f"\nSaved:\n  - {sum_path}\n  - {bd_path}\n  - {curves_path}")


if __name__ == "__main__":
    main()
