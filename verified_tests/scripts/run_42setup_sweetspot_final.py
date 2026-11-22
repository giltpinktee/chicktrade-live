"""Recompute Portfolio Sweet Spots (Buffer 500/400/300/250 + 2-Gear Auto-Turbo MaxDD<15%) on the 42 Exit-Optimized Setups."""
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
from run_full_3ema_storm_guard import build_cooldown_mask, compute_storm_flags, generate_all_4_touches
from run_full_216_exit_sweetspot import COIN_CAL, sim_exit_arch
from run_master_6coin_html import SIX_COINS
from run_research_mtf import resample_ohlcv

EXIT_PARSE = {
    "ONLY_TP1": ("ONLY_TP1", 0.0),
    "ONLY_1R": ("ONLY_1R", 0.0),
    "PARTIAL_NO_MOVE": ("PARTIAL_NO_MOVE", 0.0),
    "PARTIAL_BE+Phi": ("PARTIAL_MOVE_SL", 0.3),
    "PARTIAL_Entry0U": ("PARTIAL_MOVE_SL", 0.0),
    "PARTIAL_Entry-1U": ("PARTIAL_MOVE_SL", -1.0),
    "PARTIAL_Entry-2U": ("PARTIAL_MOVE_SL", -2.0),
}
EMA_MULT = {"EMA34": 1.00, "EMA89": 0.95, "EMA200": 0.90}


def wr_to_vol(wr: float, is_scout: bool, ema: str) -> float:
    if wr >= 80.0:
        base = 1.20
    elif wr >= 75.0:
        base = 1.00
    elif wr >= 70.0:
        base = 0.80
    elif wr >= 65.0:
        base = 0.65
    else:
        base = 0.50
    v = base * EMA_MULT.get(ema, 1.0)
    return round(min(v, 0.45) if is_scout else v, 2)


def sim_42_portfolio(tf_dfs_map, tf_sigs_map, rules_map, base_buf, turbo_buf, trigger_eq, disengage_eq, max_c, start_time=None):
    rcfg = RiskConfig(initial_capital=10_000.0, buffer_units=base_buf, max_concurrent_trades=max_c, max_trades_per_day=10)
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

    tf_prio = {"1h": 0, "30m": 1, "15m": 2}
    cands.sort(key=lambda x: (x[2].timestamp, tf_prio[x[0]], -x[1]["vol"], x[2].symbol))

    active, closed = [], []
    daily_t, daily_sl, daily_pnl, daily_eq = {}, {}, {}, {}
    consec_sl = 0
    eq_curve = [equity]

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
            eq_curve.append(equity)
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

        # Enforce hard floor >= $11,000 if all open SLs swept while in Turbo
        cur_buf = turbo_buf if in_turbo else base_buf
        if in_turbo:
            open_sl_usd = sum(tr["sl_usd"] for _, tr in active)
            new_sl_usd = (min(comp.tier_capital, equity) * eff_m / turbo_buf) * rule["sl_u"]
            if (equity - open_sl_usd - new_sl_usd) < 11_000.0:
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
        active.append((ex_ts, {"sym": sig.symbol, "m": eff_m, "out": out, "pnl": pnl, "sl_usd": sl_usd, "dk": dk}))
        daily_t[dk] = daily_t.get(dk, 0) + 1

    if active:
        settle(pd.Timestamp("2099-01-01", tz="UTC"))

    ev = np.array(eq_curve)
    pk = np.maximum.accumulate(ev)
    mdd = float(((ev - pk) / pk * 100.0).min())
    pnls = np.array([t["pnl"] for t in closed])
    wr = (pnls > 0).mean() * 100.0 if len(pnls) else 0.0
    pf = float(pnls[pnls > 0].sum() / max(-pnls[pnls < 0].sum(), 0.01))
    ret = (float(ev[-1]) / 10_000.0 - 1.0) * 100.0
    return {
        "trades": len(closed), "wr": wr, "pf": pf, "final_eq": float(ev[-1]),
        "max_eq": float(ev.max()), "ret": ret, "mdd": mdd, "step": comp.max_step_reached,
        "calmar": ret / max(abs(mdd), 0.1),
    }


def main():
    sel_df = pd.read_csv(os.path.join(ROOT, "core", "reports", "selected_setups_exit_anatomy.csv"))
    rules_map = {}
    for _, r in sel_df.iterrows():
        sym, tfl, stype = r["Coin"], r["TF"], r["Setup"]
        arch, l2_u = EXIT_PARSE[r["Best_Exit"]]
        cc = COIN_CAL[sym]
        ema = stype.split("_")[0]
        vol = wr_to_vol(float(r["WR_%"]), r["Status"] == "SCOUT", ema)
        rules_map[(sym, tfl, stype)] = {
            "arch": arch, "l2_u": l2_u, "sl_u": cc["sl"], "tp1_u": cc["tp1"][tfl], "tp2_u": cc["tp2"], "vol": vol,
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

    print("=" * 116)
    print("DIEM NGOT MOI TREN BO 42 SETUP (DA TOI UU KIEN TRUC THOAT LENH RIENG TUNG CON: ONLY_TP1 / ONLY_1R / PARTIAL)")
    print("=" * 116)
    rows = []
    for label, b_buf, t_buf, trig, mc in [
        ("1. Chuan: Bao 500G (Max 7)", 500.0, 500.0, 999999, 7),
        ("2. Chuan: Bao 400G (Max 5)", 400.0, 400.0, 999999, 5),
        ("3. Chuan: Bao 300G (Max 3)", 300.0, 300.0, 999999, 3),
        ("4. Chuan: Bao 250G (Max 2)", 250.0, 250.0, 999999, 2),
        ("5. Turbo: 400G -> 280G @ $12.5k (Max 5)", 400.0, 280.0, 12500, 5),
        ("6. Turbo: 400G -> 250G @ $12.5k (Max 5)", 400.0, 250.0, 12500, 5),
        ("7. Turbo: 400G -> 230G @ $12.5k (Max 5)", 400.0, 230.0, 12500, 5),
    ]:
        r12 = sim_42_portfolio(tf_dfs_map, tf_sigs_map, rules_map, b_buf, t_buf, trig, 11500, mc, None)
        r6 = sim_42_portfolio(tf_dfs_map, tf_sigs_map, rules_map, b_buf, t_buf, trig, 11500, mc, mid_ts)
        rows.append(
            {
                "Cau_Hinh": label,
                "12M_Lenh": r12["trades"], "12M_WR%": r12["wr"], "12M_PF": r12["pf"],
                "12M_Eq_$": r12["final_eq"], "12M_Lai%": r12["ret"], "12M_DD%": r12["mdd"],
                "Calmar": r12["calmar"], "Bac": f"{r12['step']}/20",
                "OOS_WR%": r6["wr"], "OOS_Lai%": r6["ret"], "OOS_DD%": r6["mdd"],
            }
        )
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:,.2f}"))


if __name__ == "__main__":
    main()
