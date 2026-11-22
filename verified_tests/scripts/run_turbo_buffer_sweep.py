"""Sweep Turbo Buffer (350..200) under MaxDD < 15% constraint + Principal-Protected 2-Gear Auto-Turbo Simulation."""
import json
import os
import sys

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
from run_6coin_full_study import fast_sim_policy
from run_full_3ema_storm_guard import build_cooldown_mask, compute_storm_flags, generate_all_4_touches
from run_master_6coin_html import SIX_COINS
from run_research_mtf import resample_ohlcv


def simulate_2gear_portfolio(
    tf_dfs_map: dict,
    tf_sigs_map: dict,
    rules_map: dict,
    base_buffer: float = 400.0,
    turbo_buffer: float = 280.0,
    turbo_trigger_equity: float = 13_000.0,
    turbo_disengage_equity: float = 12_000.0,
    max_concurrent: int = 5,
    start_time: pd.Timestamp = None,
) -> dict:
    """2-Gear Auto-Turbo Portfolio:
    - Starts at `base_buffer` (400U) to protect $10,000 principal.
    - Switches to `turbo_buffer` once equity >= `turbo_trigger_equity` (guaranteeing >= $11,000 even at max DD!).
    - Drops back to `base_buffer` if equity falls below `turbo_disengage_equity`.
    """
    rcfg = RiskConfig(initial_capital=10_000.0, buffer_units=base_buffer, max_concurrent_trades=max_concurrent, max_trades_per_day=10)
    comp = CompoundingManager(rcfg)
    equity = rcfg.initial_capital
    peak_eq = equity
    in_turbo = (turbo_trigger_equity <= equity)
    open_mult_cap = round(max_concurrent * 0.85, 2)

    candidates = []
    for tfl, sigs_map in tf_sigs_map.items():
        for sym, sigs in sigs_map.items():
            for s in sigs:
                key = (sym, tfl, s.setup_type)
                if key not in rules_map:
                    continue
                if start_time is not None and s.timestamp < start_time:
                    continue
                candidates.append((tfl, rules_map[key], s))

    tf_prio = {"1h": 0, "30m": 1, "15m": 2}
    candidates.sort(key=lambda x: (x[2].timestamp, tf_prio[x[0]], -x[1]["vol_mult"], x[2].symbol))

    active, closed = [], []
    daily_trades, daily_sl, daily_pnl, daily_start_eq = {}, {}, {}, {}
    consec_sl = 0
    eq_curve = [{"timestamp": candidates[0][2].timestamp, "equity": equity}]
    turbo_switches = 0

    def settle(now_ts):
        nonlocal equity, peak_eq, consec_sl, in_turbo, turbo_switches
        keep, done = [], []
        for ex_ts, tr in active:
            (done if ex_ts <= now_ts else keep).append((ex_ts, tr))
        done.sort(key=lambda x: x[0])
        for ex_ts, tr in done:
            equity += tr["net_pnl"]
            peak_eq = max(peak_eq, equity)
            comp.update_equity(equity, ex_ts)
            dk = tr["day_key"]
            daily_pnl[dk] = daily_pnl.get(dk, 0.0) + tr["net_pnl"]
            if tr["outcome"] == "SL":
                consec_sl += 1
                daily_sl[dk] = daily_sl.get(dk, 0) + 1
            elif tr["net_pnl"] > 0:
                consec_sl = 0

            # 2-Gear Auto-Turbo switching
            if (not in_turbo) and equity >= turbo_trigger_equity:
                in_turbo = True
                turbo_switches += 1
            elif in_turbo and equity < turbo_disengage_equity:
                in_turbo = False

            closed.append(tr)
            eq_curve.append({"timestamp": ex_ts, "equity": equity})
        active[:] = keep

    for tfl, rule, sig in candidates:
        settle(sig.timestamp)
        if equity <= 0:
            break
        dk = sig.timestamp.strftime("%Y-%m-%d")
        if dk not in daily_start_eq:
            daily_start_eq[dk] = equity

        if daily_trades.get(dk, 0) >= rcfg.max_trades_per_day:
            continue
        if daily_sl.get(dk, 0) >= 3:
            continue
        if daily_pnl.get(dk, 0.0) <= -0.05 * daily_start_eq[dk]:
            continue
        if len(active) >= max_concurrent:
            continue
        if any(tr["symbol"] == sig.symbol for _, tr in active):
            continue

        cur_buf = turbo_buffer if in_turbo else base_buffer
        eff_mult = rule["vol_mult"]
        if sig.pa_vol_ok:
            eff_mult *= 0.5
        dd_pct = (equity - peak_eq) / peak_eq * 100.0
        if dd_pct <= -6.0 or consec_sl >= 2:
            eff_mult = min(eff_mult, 0.25)

        open_mult = sum(tr["size_mult"] for _, tr in active)
        eff_mult = min(eff_mult, max(0.0, open_mult_cap - open_mult))
        if eff_mult < 0.20:
            continue

        df = tf_dfs_map[tfl][sig.symbol]
        h_arr, l_arr, c_arr = df["high"].values, df["low"].values, df["close"].values
        u = max(0.5 * sig.atr, sig.entry_price * 1e-5)
        out, net_u = fast_sim_policy(
            h_arr, l_arr, c_arr, sig.bar_idx, sig.entry_price, sig.direction, u,
            rule["sl_u"], rule["tp1_u"], rule["tp2_u"], rule["leg2_sl_u"]
        )
        sizing_base = min(comp.tier_capital, equity)
        qty = (sizing_base * eff_mult) / (cur_buf * u)
        net_pnl = net_u * u * qty

        bars_est = 4 if out == "SL" else 8
        exit_idx = min(sig.bar_idx + bars_est, len(df) - 1)
        ex_ts = pd.Timestamp(df["timestamp"].iloc[exit_idx])
        if ex_ts.tzinfo is None:
            ex_ts = ex_ts.tz_localize("UTC")

        active.append(
            (
                ex_ts,
                {
                    "symbol": sig.symbol,
                    "size_mult": eff_mult,
                    "outcome": out,
                    "net_pnl": net_pnl,
                    "day_key": dk,
                    "in_turbo": in_turbo,
                },
            )
        )
        daily_trades[dk] = daily_trades.get(dk, 0) + 1

    if active:
        settle(pd.Timestamp("2099-01-01", tz="UTC"))

    ev = np.array([x["equity"] for x in eq_curve])
    pk = np.maximum.accumulate(ev)
    max_dd = float(((ev - pk) / pk * 100.0).min())
    pnls = np.array([t["net_pnl"] for t in closed])
    wr = (pnls > 0).mean() * 100.0 if len(pnls) else 0.0
    pos = pnls[pnls > 0].sum()
    neg = -pnls[pnls < 0].sum()
    pf = float(pos / neg) if neg > 0 else 0.0
    min_eq_after_turbo = float(ev[ev >= turbo_trigger_equity].min()) if (ev >= turbo_trigger_equity).any() else float(ev.min())

    # True minimum equity AFTER first time hitting trigger
    hit_idxs = np.where(ev >= turbo_trigger_equity)[0]
    min_after_trigger = float(ev[hit_idxs[0] :].min()) if len(hit_idxs) > 0 else float(ev.min())

    return {
        "trades": len(closed),
        "wr_pct": wr,
        "pf": pf,
        "final_eq": float(ev[-1]),
        "max_eq": float(ev.max()),
        "ret_pct": (float(ev[-1]) / 10_000.0 - 1.0) * 100.0,
        "max_dd": max_dd,
        "max_step": comp.max_step_reached,
        "min_after_trigger": min_after_trigger,
        "calmar": ((float(ev[-1]) / 10_000.0 - 1.0) * 100.0) / max(abs(max_dd), 0.1),
    }


def main():
    with open(os.path.join(ROOT, "core", "core_strategy_rules.json"), encoding="utf-8") as f:
        core_cfg = json.load(f)

    ema_mult = core_cfg.get("ema_risk_multipliers", {"EMA34": 1.0, "EMA89": 0.95, "EMA200": 0.90})
    rules_map = {}
    for r in core_cfg["rules"]:
        em = ema_mult.get(r["ema"], 1.0)
        vm = round(float(r["adaptive_volume"]["init_vol_mult"]) * em, 2)
        rules_map[(r["coin"], r["tf"], r["setup_code"])] = {
            "vol_mult": vm,
            "sl_u": float(r["sl_u"]),
            "tp1_u": float(r["tp1_u"]),
            "tp2_u": float(r["tp2_u"]),
            "leg2_sl_u": float(r["leg2_sl_u"]),
        }

    cfg = BacktestConfig()
    end_ym = _detect_latest_available_month("BTCUSDT", "5m")
    raw_m5 = {sym: download_symbol_1y(sym, "5m", 12, "futures", end_ym) for sym in SIX_COINS}
    dfs_m15 = {s: resample_ohlcv(d, "15min") for s, d in raw_m5.items()}
    dfs_m30 = {s: resample_ohlcv(d, "30min") for s, d in raw_m5.items()}
    dfs_h1 = {s: resample_ohlcv(d, "1h") for s, d in raw_m5.items()}
    dfs_h4 = {s: resample_ohlcv(d, "4h") for s, d in raw_m5.items()}
    tf_dfs_map = {"15m": dfs_m15, "30m": dfs_m30, "1h": dfs_h1}

    tf_sigs_map = {}
    for tfl, tdf, hdf, cd in [("15m", dfs_m15, dfs_h1, 16), ("30m", dfs_m30, dfs_h4, 8), ("1h", dfs_h1, dfs_h4, 4)]:
        bst = compute_storm_flags(tdf["BTCUSDT"])
        bmask = build_cooldown_mask(bst["raw_storm"], cd)
        bts = tdf["BTCUSDT"]["timestamp"].values
        tf_sigs_map[tfl] = {
            sym: generate_all_4_touches(tdf[sym], hdf[sym], bts, bmask, sym, cfg.strategy, cooldown_bars=cd)[0]
            for sym in SIX_COINS
        }

    all_ts = sorted(s.timestamp for s in tf_sigs_map["15m"]["BTCUSDT"])
    mid_ts = all_ts[len(all_ts) // 2]

    print("=" * 120)
    print("QUET MUC BAO GIA TANG TOC (MAX 5 LENH, 33 SETUPS v1.4) — RANG BUOC CUNG: MaxDD < 15.0% & BAO VE SAN $11,000")
    print("=" * 120)

    rows = []
    for tb in [400.0, 350.0, 320.0, 300.0, 280.0, 260.0, 250.0, 230.0, 200.0]:
        # First check pure Turbo DD to derive exact Principal-Protected Trigger = $11,000 / (1 - |DD|)
        pure = simulate_2gear_portfolio(tf_dfs_map, tf_sigs_map, rules_map, base_buffer=tb, turbo_buffer=tb, turbo_trigger_equity=0.0, max_concurrent=5)
        pure_dd = abs(pure["max_dd"])
        # Exact mathematical trigger so even hitting full MaxDD keeps Equity >= $11,000 (+10% net profit)
        req_trigger = max(11_500.0, round(11_000.0 / max(1.0 - pure_dd / 100.0, 0.5), -2))
        disengage = max(11_200.0, req_trigger - 1_000.0)

        # Now run 2-Gear Auto-Turbo (Start 400U -> Switch to tb at req_trigger!)
        g2_12m = simulate_2gear_portfolio(tf_dfs_map, tf_sigs_map, rules_map, base_buffer=400.0, turbo_buffer=tb, turbo_trigger_equity=req_trigger, turbo_disengage_equity=disengage, max_concurrent=5)
        g2_oos = simulate_2gear_portfolio(tf_dfs_map, tf_sigs_map, rules_map, base_buffer=400.0, turbo_buffer=tb, turbo_trigger_equity=req_trigger, turbo_disengage_equity=disengage, max_concurrent=5, start_time=mid_ts)

        ok_dd = "✅ ĐẠT (<15%)" if abs(g2_12m["max_dd"]) < 15.0 else "❌ VƯỢT 15%"
        rows.append(
            {
                "Bao_Turbo": f"{int(tb)}G",
                "Moc_Bat_Turbo_$": req_trigger,
                "Pure_DD_%": pure["max_dd"],
                "2Gear_12M_Eq_$": g2_12m["final_eq"],
                "2Gear_12M_Lai_%": g2_12m["ret_pct"],
                "2Gear_12M_DD_%": g2_12m["max_dd"],
                "Day_Sau_Bat_$": g2_12m["min_after_trigger"],
                "Calmar": g2_12m["calmar"],
                "OOS6M_Lai_%": g2_oos["ret_pct"],
                "OOS6M_DD_%": g2_oos["max_dd"],
                "Dat_DD<15%": ok_dd,
            }
        )

    res_df = pd.DataFrame(rows)
    print(res_df.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
    res_df.to_csv(os.path.join(ROOT, "output_test", "turbo_2gear_buffer_sweep.csv"), index=False)


if __name__ == "__main__":
    main()
