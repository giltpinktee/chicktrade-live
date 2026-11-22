"""Independent Quantitative Audit: Trading Sessions (UTC) & Day-of-Week / Weekend Filtering.
Evaluated on the 6-Coin CHICKTRADE dataset under PORTFOLIO 2-SL -> Shadow-Win (Bao 400G, Max 5 concurrent).
Measures Winrate, PF, Avg Net U, Total Net U by UTC Session and Weekday vs Weekend,
and tests blocking / 0.5x volume throttling on BOTH Full 12M and Strict OOS 6M.
"""
import os
import sys
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

RESEARCH_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(RESEARCH_DIR)
for p in (ROOT, RESEARCH_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from config import BacktestConfig, RiskConfig
from data.binance_loader import _detect_latest_available_month, download_symbol_1y
from risk.compounding import CompoundingManager
from run_full_216_exit_sweetspot import sim_exit_arch
from run_full_3ema_storm_guard import build_cooldown_mask, compute_storm_flags, generate_all_4_touches
from run_master_6coin_html import SIX_COINS
from run_pause_resume_sweetspot import load_rules_with_hot_adaptive
from run_research_mtf import resample_ohlcv


SESSIONS_ORDER = [
    "Asia (00-07 UTC)",
    "Europe/London (08-12 UTC)",
    "US/NY Overlap (13-20 UTC)",
    "Late US (21-23 UTC)",
]
DAY_GROUPS_ORDER = ["Weekday (Mon-Fri)", "Weekend (Sat-Sun)"]
DAYS_ORDER = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _to_utc(ts: pd.Timestamp) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        return t.tz_localize("UTC")
    return t.tz_convert("UTC")


def get_session_utc(ts: pd.Timestamp) -> str:
    h = _to_utc(ts).hour
    if 0 <= h <= 7:
        return "Asia (00-07 UTC)"
    if 8 <= h <= 12:
        return "Europe/London (08-12 UTC)"
    if 13 <= h <= 20:
        return "US/NY Overlap (13-20 UTC)"
    return "Late US (21-23 UTC)"


def get_day_group(ts: pd.Timestamp) -> str:
    return "Weekend (Sat-Sun)" if _to_utc(ts).dayofweek >= 5 else "Weekday (Mon-Fri)"


def get_day_name(ts: pd.Timestamp) -> str:
    return DAYS_ORDER[_to_utc(ts).dayofweek]


def calc_u_stats(nus: np.ndarray) -> dict:
    if len(nus) == 0:
        return {"trades": 0, "wr": 0.0, "pf_u": 0.0, "avg_nu": 0.0, "tot_nu": 0.0}
    wins = nus[nus > 0]
    losses = nus[nus < 0]
    wr = float((nus > 0).mean() * 100.0)
    gross_win = float(wins.sum()) if len(wins) else 0.0
    gross_loss = float(-losses.sum()) if len(losses) else 0.0
    pf_u = gross_win / max(gross_loss, 1e-6)
    return {
        "trades": int(len(nus)),
        "wr": wr,
        "pf_u": pf_u,
        "avg_nu": float(nus.mean()),
        "tot_nu": float(nus.sum()),
    }


def sim_portfolio_2sl_shadow(
    tf_dfs_map: dict,
    tf_sigs_map: dict,
    rules_map: dict,
    filter_fn: Optional[Callable[[pd.Timestamp], float]] = None,
    start_time: Optional[pd.Timestamp] = None,
    end_time: Optional[pd.Timestamp] = None,
) -> dict:
    """Exact PORTFOLIO 2-SL -> Shadow-Win policy (Bao 400G, Max 5 concurrent)
    with optional session/day volume multiplier filter_fn(ts) -> {0.0, 0.5, 1.0}.
    """
    rcfg = RiskConfig(initial_capital=10_000.0, buffer_units=400.0, max_concurrent_trades=5, max_trades_per_day=10)
    comp = CompoundingManager(rcfg)
    equity = 10_000.0
    peak_eq = equity
    open_cap = 4.25

    cands = []
    for tfl, sm in tf_sigs_map.items():
        for sym, sigs in sm.items():
            for s in sigs:
                k = (sym, tfl, s.setup_type)
                if k not in rules_map:
                    continue
                if start_time is not None and s.timestamp < start_time:
                    continue
                if end_time is not None and s.timestamp >= end_time:
                    continue
                cands.append((tfl, rules_map[k], s))

    tf_prio = {"1h": 0, "30m": 1, "15m": 2, "5m": 3}
    cands.sort(key=lambda x: (x[2].timestamp, tf_prio[x[0]], -x[1]["vol"], x[2].symbol))

    active, closed = [], []
    daily_t, daily_pnl, daily_eq = {}, {}, {}
    eq_curve = [equity]

    consec_sl = 0
    paused_until = pd.Timestamp("1970-01-01", tz="UTC")
    shadow_paused = False
    pause_events = 0
    skipped_by_pause = 0
    filtered_out = 0

    def settle(now_ts):
        nonlocal equity, peak_eq, consec_sl, paused_until, shadow_paused, pause_events
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
                if consec_sl >= 2:
                    pause_events += 1
                    consec_sl = 0
                    shadow_paused = True
                    paused_until = ex_ts + pd.Timedelta(hours=72)
            elif tr["pnl"] > 0:
                consec_sl = 0

            closed.append(tr)
            eq_curve.append(equity)
        active[:] = keep

    for tfl, rule, sig in cands:
        settle(sig.timestamp)
        if equity <= 0:
            break

        # Session / Day filter check
        sess_mult = 1.0 if filter_fn is None else float(filter_fn(sig.timestamp))
        if sess_mult <= 0.0:
            filtered_out += 1
            continue

        dk = sig.timestamp.strftime("%Y-%m-%d")
        if dk not in daily_eq:
            daily_eq[dk] = equity

        df = tf_dfs_map[tfl][sig.symbol]
        u = max(0.5 * sig.atr, sig.entry_price * 1e-5)

        is_paused = False
        if shadow_paused:
            if sig.timestamp >= paused_until:
                shadow_paused = False
            else:
                is_paused = True

        if is_paused:
            skipped_by_pause += 1
            _, _, _, _, nu_sh = sim_exit_arch(
                df["high"].values, df["low"].values, df["close"].values,
                sig.bar_idx, sig.entry_price, sig.direction, u,
                rule["sl_u"], rule["tp1_u"], rule["tp2_u"], rule["arch"], rule["l2_u"],
            )
            if nu_sh > 0:
                shadow_paused = False
            continue

        if daily_t.get(dk, 0) >= 10 or daily_pnl.get(dk, 0.0) <= -0.05 * daily_eq[dk]:
            continue
        if len(active) >= 5 or any(tr["sym"] == sig.symbol for _, tr in active):
            continue

        eff_m = rule["vol"] * sess_mult
        if sig.pa_vol_ok:
            eff_m *= 0.5
        if (equity - peak_eq) / peak_eq * 100.0 <= -6.0:
            eff_m = min(eff_m, 0.25)
        eff_m = min(eff_m, max(0.0, open_cap - sum(tr["m"] for _, tr in active)))
        if eff_m < 0.20:
            continue

        out, _, _, _, nu = sim_exit_arch(
            df["high"].values, df["low"].values, df["close"].values,
            sig.bar_idx, sig.entry_price, sig.direction, u,
            rule["sl_u"], rule["tp1_u"], rule["tp2_u"], rule["arch"], rule["l2_u"],
        )
        sb = min(comp.tier_capital, equity)
        qty = (sb * eff_m) / (400.0 * u)
        pnl = nu * u * qty

        ex_i = min(sig.bar_idx + (4 if out == "SL" else 8), len(df) - 1)
        ex_ts = pd.Timestamp(df["timestamp"].iloc[ex_i])
        if ex_ts.tzinfo is None:
            ex_ts = ex_ts.tz_localize("UTC")

        active.append((
            ex_ts,
            {
                "ts": _to_utc(sig.timestamp),
                "session": get_session_utc(sig.timestamp),
                "day_group": get_day_group(sig.timestamp),
                "day_name": get_day_name(sig.timestamp),
                "sym": sig.symbol,
                "tfl": tfl,
                "setup": sig.setup_type,
                "m": eff_m,
                "out": out,
                "nu": float(nu),
                "pnl": float(pnl),
                "dk": dk,
            },
        ))
        daily_t[dk] = daily_t.get(dk, 0) + 1

    if active:
        settle(pd.Timestamp("2099-01-01", tz="UTC"))

    ev = np.array(eq_curve)
    pk = np.maximum.accumulate(ev)
    mdd = float(((ev - pk) / pk * 100.0).min())
    pnls = np.array([t["pnl"] for t in closed])
    nus = np.array([t["nu"] for t in closed])
    wr = float((pnls > 0).mean() * 100.0) if len(pnls) else 0.0
    pf_usd = float(pnls[pnls > 0].sum() / max(-pnls[pnls < 0].sum(), 0.01)) if len(pnls) else 0.0
    u_st = calc_u_stats(nus)
    ret = (float(ev[-1]) / 10_000.0 - 1.0) * 100.0
    return {
        "trades": len(closed),
        "pauses": pause_events,
        "skipped": skipped_by_pause,
        "filtered_out": filtered_out,
        "wr": wr,
        "pf": pf_usd,
        "pf_u": u_st["pf_u"],
        "avg_nu": u_st["avg_nu"],
        "tot_nu": u_st["tot_nu"],
        "ret": ret,
        "mdd": mdd,
        "calmar": ret / max(abs(mdd), 0.1),
        "closed": closed,
    }


def evaluate_raw_signals(tf_dfs_map: dict, tf_sigs_map: dict, rules_map: dict, mid_ts: pd.Timestamp) -> pd.DataFrame:
    """Evaluate all raw candidate signals (no portfolio concurrency/pausing) to inspect intrinsic session/day edge."""
    rows = []
    for tfl, sm in tf_sigs_map.items():
        for sym, sigs in sm.items():
            df = tf_dfs_map[tfl][sym]
            highs, lows, closes = df["high"].values, df["low"].values, df["close"].values
            for s in sigs:
                k = (sym, tfl, s.setup_type)
                if k not in rules_map:
                    continue
                rule = rules_map[k]
                u = max(0.5 * s.atr, s.entry_price * 1e-5)
                out, _, _, _, nu = sim_exit_arch(
                    highs, lows, closes,
                    s.bar_idx, s.entry_price, s.direction, u,
                    rule["sl_u"], rule["tp1_u"], rule["tp2_u"], rule["arch"], rule["l2_u"],
                )
                rows.append({
                    "ts": _to_utc(s.timestamp),
                    "is_oos": s.timestamp >= mid_ts,
                    "session": get_session_utc(s.timestamp),
                    "day_group": get_day_group(s.timestamp),
                    "day_name": get_day_name(s.timestamp),
                    "nu": float(nu),
                })
    return pd.DataFrame(rows)


def build_breakdown_rows(
    section_name: str,
    df_12m: pd.DataFrame,
    df_is: pd.DataFrame,
    df_oos: pd.DataFrame,
    col: str,
    order: List[str],
) -> List[dict]:
    out_rows = []
    for item in order:
        s12 = calc_u_stats(df_12m.loc[df_12m[col] == item, "nu"].values)
        sis = calc_u_stats(df_is.loc[df_is[col] == item, "nu"].values)
        soos = calc_u_stats(df_oos.loc[df_oos[col] == item, "nu"].values)
        out_rows.append({
            "Section": section_name,
            "Item": item,
            "Trades_12M": s12["trades"],
            "WR_12M%": s12["wr"],
            "PF_12M": s12["pf_u"],
            "Avg_NetU_12M": s12["avg_nu"],
            "Total_NetU_12M": s12["tot_nu"],
            "Ret_12M%": np.nan,
            "MaxDD_12M%": np.nan,
            "Calmar_12M": np.nan,
            "Trades_IS6M": sis["trades"],
            "WR_IS6M%": sis["wr"],
            "PF_IS6M": sis["pf_u"],
            "Avg_NetU_IS6M": sis["avg_nu"],
            "Total_NetU_IS6M": sis["tot_nu"],
            "Trades_OOS6M": soos["trades"],
            "WR_OOS6M%": soos["wr"],
            "PF_OOS6M": soos["pf_u"],
            "Avg_NetU_OOS6M": soos["avg_nu"],
            "Total_NetU_OOS6M": soos["tot_nu"],
            "Ret_OOS6M%": np.nan,
            "MaxDD_OOS6M%": np.nan,
            "Calmar_OOS6M": np.nan,
        })
    return out_rows


def main():
    rules_map = load_rules_with_hot_adaptive()
    cfg = BacktestConfig()
    end_ym = _detect_latest_available_month("BTCUSDT", "5m")

    raw_m5 = {sym: download_symbol_1y(sym, "5m", 12, "futures", end_ym) for sym in SIX_COINS}
    dfs_m15 = {s: resample_ohlcv(d, "15min") for s, d in raw_m5.items()}
    dfs_m30 = {s: resample_ohlcv(d, "30min") for s, d in raw_m5.items()}
    dfs_h1 = {s: resample_ohlcv(d, "1h") for s, d in raw_m5.items()}
    dfs_h4 = {s: resample_ohlcv(d, "4h") for s, d in raw_m5.items()}
    tf_dfs_map = {"5m": raw_m5, "15m": dfs_m15, "30m": dfs_m30, "1h": dfs_h1}

    tf_sigs_map = {}
    for tfl, tdf, hdf, cd in [
        ("5m", raw_m5, dfs_h1, 48),
        ("15m", dfs_m15, dfs_h1, 16),
        ("30m", dfs_m30, dfs_h4, 8),
        ("1h", dfs_h1, dfs_h4, 4),
    ]:
        bst = compute_storm_flags(tdf["BTCUSDT"])
        bmask = build_cooldown_mask(bst["raw_storm"], cd)
        bts = tdf["BTCUSDT"]["timestamp"].values
        tf_sigs_map[tfl] = {
            sym: generate_all_4_touches(tdf[sym], hdf[sym], bts, bmask, sym, cfg.strategy, cooldown_bars=cd)[0]
            for sym in SIX_COINS
        }

    all_ts = sorted(s.timestamp for s in tf_sigs_map["15m"]["BTCUSDT"])
    mid_ts = all_ts[len(all_ts) // 2]
    print(f"Dataset split: Full 12M start={all_ts[0]} | IS/OOS Mid={mid_ts} | End={all_ts[-1]}")

    # 1. Baseline PORTFOLIO 2-SL -> Shadow-Win runs (12M, IS 6M, OOS 6M)
    base_12m = sim_portfolio_2sl_shadow(tf_dfs_map, tf_sigs_map, rules_map, None, None, None)
    base_is = sim_portfolio_2sl_shadow(tf_dfs_map, tf_sigs_map, rules_map, None, None, mid_ts)
    base_oos = sim_portfolio_2sl_shadow(tf_dfs_map, tf_sigs_map, rules_map, None, mid_ts, None)

    df_exec_12m = pd.DataFrame(base_12m["closed"])
    df_exec_is = pd.DataFrame(base_is["closed"])
    df_exec_oos = pd.DataFrame(base_oos["closed"])

    # 2. Raw signals breakdown
    df_raw = evaluate_raw_signals(tf_dfs_map, tf_sigs_map, rules_map, mid_ts)
    df_raw_is = df_raw[~df_raw["is_oos"]]
    df_raw_oos = df_raw[df_raw["is_oos"]]

    all_csv_rows = []

    # Diagnostic Breakdowns
    exec_sess_rows = build_breakdown_rows("EXEC_SESSION", df_exec_12m, df_exec_is, df_exec_oos, "session", SESSIONS_ORDER)
    exec_wd_rows = build_breakdown_rows("EXEC_WEEKDAY_WEEKEND", df_exec_12m, df_exec_is, df_exec_oos, "day_group", DAY_GROUPS_ORDER)
    exec_dow_rows = build_breakdown_rows("EXEC_DAY_OF_WEEK", df_exec_12m, df_exec_is, df_exec_oos, "day_name", DAYS_ORDER)
    raw_sess_rows = build_breakdown_rows("RAW_SESSION", df_raw, df_raw_is, df_raw_oos, "session", SESSIONS_ORDER)
    raw_wd_rows = build_breakdown_rows("RAW_WEEKDAY_WEEKEND", df_raw, df_raw_is, df_raw_oos, "day_group", DAY_GROUPS_ORDER)

    all_csv_rows.extend(exec_sess_rows)
    all_csv_rows.extend(exec_wd_rows)
    all_csv_rows.extend(exec_dow_rows)
    all_csv_rows.extend(raw_sess_rows)
    all_csv_rows.extend(raw_wd_rows)

    def print_breakdown(title: str, rows: List[dict]):
        print("\n" + "=" * 128)
        print(title)
        print("=" * 128)
        cols = [
            "Item",
            "Trades_12M", "WR_12M%", "PF_12M", "Avg_NetU_12M", "Total_NetU_12M",
            "Trades_IS6M", "PF_IS6M", "Avg_NetU_IS6M",
            "Trades_OOS6M", "WR_OOS6M%", "PF_OOS6M", "Avg_NetU_OOS6M", "Total_NetU_OOS6M",
        ]
        print(pd.DataFrame(rows)[cols].to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    print_breakdown("1A. EXECUTED TRADES (PORTFOLIO 2-SL -> SHADOW-WIN) BY UTC SESSION", exec_sess_rows)
    print_breakdown("1B. EXECUTED TRADES (PORTFOLIO 2-SL -> SHADOW-WIN) BY WEEKDAY vs WEEKEND", exec_wd_rows)
    print_breakdown("1C. EXECUTED TRADES (PORTFOLIO 2-SL -> SHADOW-WIN) BY DAY OF WEEK", exec_dow_rows)
    print_breakdown("2A. RAW CANDIDATE SIGNALS (PRE-PORTFOLIO) BY UTC SESSION", raw_sess_rows)
    print_breakdown("2B. RAW CANDIDATE SIGNALS (PRE-PORTFOLIO) BY WEEKDAY vs WEEKEND", raw_wd_rows)

    # Identify worst session in IS 6M and Full 12M (by Avg_NetU)
    worst_is_sess = min(exec_sess_rows, key=lambda r: r["Avg_NetU_IS6M"])["Item"]
    worst_12m_sess = min(exec_sess_rows, key=lambda r: r["Avg_NetU_12M"])["Item"]
    print(f"\nWorst session by Executed Avg Net U -> IS 6M: {worst_is_sess} | Full 12M: {worst_12m_sess}")

    # 3. Filter & Volume-Throttling Policy Tests
    policies: List[Tuple[str, Optional[Callable[[pd.Timestamp], float]]]] = [
        ("0. Baseline (No Filter)", None),
        ("1. Block Weekend (Sat-Sun)", lambda ts: 0.0 if get_day_group(ts) == "Weekend (Sat-Sun)" else 1.0),
        ("2. Half Vol 0.5x Weekend (Sat-Sun)", lambda ts: 0.5 if get_day_group(ts) == "Weekend (Sat-Sun)" else 1.0),
        ("3. Block Asia (00-07 UTC)", lambda ts: 0.0 if get_session_utc(ts) == "Asia (00-07 UTC)" else 1.0),
        ("4. Half Vol 0.5x Asia (00-07 UTC)", lambda ts: 0.5 if get_session_utc(ts) == "Asia (00-07 UTC)" else 1.0),
        ("5. Block Europe/London (08-12 UTC)", lambda ts: 0.0 if get_session_utc(ts) == "Europe/London (08-12 UTC)" else 1.0),
        ("6. Half Vol 0.5x Europe/London (08-12 UTC)", lambda ts: 0.5 if get_session_utc(ts) == "Europe/London (08-12 UTC)" else 1.0),
        ("7. Block US/NY Overlap (13-20 UTC)", lambda ts: 0.0 if get_session_utc(ts) == "US/NY Overlap (13-20 UTC)" else 1.0),
        ("8. Half Vol 0.5x US/NY Overlap (13-20 UTC)", lambda ts: 0.5 if get_session_utc(ts) == "US/NY Overlap (13-20 UTC)" else 1.0),
        ("9. Block Late US (21-23 UTC)", lambda ts: 0.0 if get_session_utc(ts) == "Late US (21-23 UTC)" else 1.0),
        ("10. Half Vol 0.5x Late US (21-23 UTC)", lambda ts: 0.5 if get_session_utc(ts) == "Late US (21-23 UTC)" else 1.0),
        (
            f"11. Block Worst-IS ({worst_is_sess}) + Weekend",
            lambda ts, ws=worst_is_sess: 0.0 if (get_session_utc(ts) == ws or get_day_group(ts) == "Weekend (Sat-Sun)") else 1.0,
        ),
        (
            f"12. Half Vol 0.5x Worst-IS ({worst_is_sess}) + Weekend",
            lambda ts, ws=worst_is_sess: 0.5 if (get_session_utc(ts) == ws or get_day_group(ts) == "Weekend (Sat-Sun)") else 1.0,
        ),
    ]
    if worst_12m_sess != worst_is_sess:
        policies.extend([
            (
                f"13. Block Worst-12M ({worst_12m_sess}) + Weekend",
                lambda ts, ws=worst_12m_sess: 0.0 if (get_session_utc(ts) == ws or get_day_group(ts) == "Weekend (Sat-Sun)") else 1.0,
            ),
            (
                f"14. Half Vol 0.5x Worst-12M ({worst_12m_sess}) + Weekend",
                lambda ts, ws=worst_12m_sess: 0.5 if (get_session_utc(ts) == ws or get_day_group(ts) == "Weekend (Sat-Sun)") else 1.0,
            ),
        ])

    policy_rows = []
    for label, fn in policies:
        r12 = sim_portfolio_2sl_shadow(tf_dfs_map, tf_sigs_map, rules_map, fn, None, None)
        ris = sim_portfolio_2sl_shadow(tf_dfs_map, tf_sigs_map, rules_map, fn, None, mid_ts)
        roos = sim_portfolio_2sl_shadow(tf_dfs_map, tf_sigs_map, rules_map, fn, mid_ts, None)
        policy_rows.append({
            "Section": "POLICY_BACKTEST",
            "Item": label,
            "Trades_12M": r12["trades"],
            "WR_12M%": r12["wr"],
            "PF_12M": r12["pf"],
            "Avg_NetU_12M": r12["avg_nu"],
            "Total_NetU_12M": r12["tot_nu"],
            "Ret_12M%": r12["ret"],
            "MaxDD_12M%": r12["mdd"],
            "Calmar_12M": r12["calmar"],
            "Trades_IS6M": ris["trades"],
            "WR_IS6M%": ris["wr"],
            "PF_IS6M": ris["pf"],
            "Avg_NetU_IS6M": ris["avg_nu"],
            "Total_NetU_IS6M": ris["tot_nu"],
            "Trades_OOS6M": roos["trades"],
            "WR_OOS6M%": roos["wr"],
            "PF_OOS6M": roos["pf"],
            "Avg_NetU_OOS6M": roos["avg_nu"],
            "Total_NetU_OOS6M": roos["tot_nu"],
            "Ret_OOS6M%": roos["ret"],
            "MaxDD_OOS6M%": roos["mdd"],
            "Calmar_OOS6M": roos["calmar"],
        })

    all_csv_rows.extend(policy_rows)

    print("\n" + "=" * 140)
    print("3. PORTFOLIO 2-SL -> SHADOW-WIN (BAO 400G, MAX 5) | SESSION & WEEKEND FILTERING BACKTEST (12M vs OOS 6M)")
    print("=" * 140)
    pcols = [
        "Item",
        "Trades_12M", "WR_12M%", "PF_12M", "Ret_12M%", "MaxDD_12M%", "Calmar_12M",
        "Trades_OOS6M", "WR_OOS6M%", "PF_OOS6M", "Ret_OOS6M%", "MaxDD_OOS6M%", "Calmar_OOS6M",
    ]
    print(pd.DataFrame(policy_rows)[pcols].to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    out_path = os.path.join(ROOT, "output_test", "session_day_audit.csv")
    pd.DataFrame(all_csv_rows).to_csv(out_path, index=False)
    print(f"\nSaved audit CSV to: {out_path}")


if __name__ == "__main__":
    main()
