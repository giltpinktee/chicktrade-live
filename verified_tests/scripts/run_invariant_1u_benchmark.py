"""3-Year Scale-Invariant & Regime-Adaptive 1U Benchmark (2023-11-01 -> 2026-10-31).

Compares 4 Zero-Lookahead Scale-Invariant 1U conventions (U1..U4) head-to-head against
M0_PURE_ATR and PER_COIN_PCT_1.0x across 3 tests (Bao 400G & Auto-Turbo 250G, 6 coins, $10k):
  - U1_ROLLING_MEDIAN_30D:
      base_pct(t) = max(0.0008, rolling_median_30d(0.5 * ATR14 / close))
      win = {'5m': 8640, '15m': 2880, '30m': 1440, '1h': 720}[tf], min_periods=20
      u(t) = close(t) * base_pct(t)
  - U2_CLAMPED_HYBRID_ATR:
      raw_pct(t) = 0.5 * ATR14(t) / close(t)
      u_pct(t) = clip(raw_pct(t), 0.80 * base_pct(t), 1.25 * base_pct(t))
      u(t) = close(t) * max(0.0008, u_pct(t))
  - U3_HTF_MACRO_ATR:
      On 4h: macro_pct_4h = EMA(20, adjust=False) of (0.5 * ATR14_4h / close_4h),
      shifted +1 closed 4h bar (zero lookahead) and backward-asof mapped to 5m/15m/30m/1h,
      scaled by gamma = {'5m': 0.17, '15m': 0.29, '30m': 0.41, '1h': 0.59}[tf]:
      u(t) = close(t) * max(0.0008, macro_pct_4h_closed * gamma)
  - U4_STRUCTURAL_WAVE_BLEND:
      spread_pct(t) = clip(0.16 * abs(EMA34(t) - EMA200(t)) / close(t),
                           0.75 * base_pct(t), 1.35 * base_pct(t))
      u(t) = close(t) * max(0.0008, 0.60 * base_pct(t) + 0.40 * spread_pct(t))

Tests:
  - Test A: Frozen 75 Setups (v2.3) + Rule 1B dir_filter on 2Y_Blind & 1Y_Recent.
  - Test B: Walk-Forward (Train 2Y 11/2023->10/2025 -> Blind Test Y3 11/2025->10/2026 + Full 36M).
  - Test C: Reverse Blind (Train 1Y 11/2025->10/2026 -> Blind Test Y1, Y2, 2Y 11/2023->10/2025 + Full 36M).

Output:
  - e:\\CHICKTRADE\\verified_tests\\results\\invariant_1u_benchmark.csv
"""
import os
import sys
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
VERIFIED_DIR = os.path.dirname(SCRIPTS_DIR)
ROOT = os.path.dirname(VERIFIED_DIR)
for p in (ROOT, SCRIPTS_DIR, os.path.join(ROOT, "old", "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from config import BacktestConfig, RiskConfig
from risk.compounding import CompoundingManager
from run_288_with_shadow_win_recheck import EXIT_CANDS
from run_full_216_exit_sweetspot import COIN_CAL, sim_exit_arch
from run_full_3ema_storm_guard import build_cooldown_mask, compute_storm_flags, generate_all_4_touches
from run_master_6coin_html import SIX_COINS
from run_pct_vs_atr_3year_study import filter_sigs, load_3year_cached_m5, load_frozen_75
from run_per_coin_pct_3year_study import measure_per_coin_u_pct
from run_research_mtf import resample_ohlcv

RESULTS_DIR = os.path.join(VERIFIED_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

MIN_U_PCT = 0.0008
WIN_30D = {"5m": 8640, "15m": 2880, "30m": 1440, "1h": 720}
GAMMA_4H = {"5m": 0.17, "15m": 0.29, "30m": 0.41, "1h": 0.59}

ALL_MODES = [
    "U1_ROLLING_MEDIAN_30D",
    "U2_CLAMPED_HYBRID_ATR",
    "U3_HTF_MACRO_ATR",
    "U4_STRUCTURAL_WAVE_BLEND",
    "PER_COIN_PCT_1.0x",
    "M0_PURE_ATR",
]


def _compute_atr14(df: pd.DataFrame) -> pd.Series:
    """Wilder ATR(14) computed identically to existing verified_tests scripts."""
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat(
        [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / 14, adjust=False).mean()


def precompute_invariant_u_arrays(
    tf_dfs_map: dict,
    dfs_h4: Dict[str, pd.DataFrame],
) -> Tuple[Dict[str, Dict[str, Dict[str, np.ndarray]]], pd.DataFrame]:
    """Precompute per-candle 1U arrays for U1..U4 (100% zero lookahead) and diagnostic stats."""
    u_arrays: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {
        m: {tf: {} for tf in ("5m", "15m", "30m", "1h")}
        for m in ALL_MODES[:4]
    }
    diag_rows = []

    # Step 1: 4h macro percentage per coin, shifted by +1 closed 4h bar for strict zero lookahead.
    h4_macro_closed: Dict[str, pd.DataFrame] = {}
    for sym in SIX_COINS:
        df4 = dfs_h4[sym]
        atr14_4h = _compute_atr14(df4)
        raw_4h = (0.5 * atr14_4h) / df4["close"]
        macro_4h = raw_4h.ewm(span=20, adjust=False).mean()
        tmp = pd.DataFrame({
            "timestamp": pd.to_datetime(df4["timestamp"], utc=True),
            "macro_pct_4h": macro_4h.shift(1).values,
        }).sort_values("timestamp")
        h4_macro_closed[sym] = tmp

    # Step 2: Compute U1..U4 arrays per timeframe and coin.
    for tfl in ("5m", "15m", "30m", "1h"):
        win = WIN_30D[tfl]
        gamma = GAMMA_4H[tfl]
        for sym in SIX_COINS:
            df = tf_dfs_map[tfl][sym]
            close = df["close"].astype(float)
            atr14 = _compute_atr14(df)
            raw_pct = (0.5 * atr14) / close

            # Zero-lookahead rolling median over past 30 days; fallback to causal raw_pct then MIN_U_PCT.
            roll_med = raw_pct.rolling(window=win, min_periods=20).median()
            base_pct = np.maximum(MIN_U_PCT, roll_med.ffill().fillna(raw_pct).fillna(MIN_U_PCT).values)

            close_np = close.values
            raw_np = raw_pct.fillna(MIN_U_PCT).values

            # U1: Rolling 30D Median
            u1_pct = base_pct
            u_arrays["U1_ROLLING_MEDIAN_30D"][tfl][sym] = close_np * u1_pct

            # U2: Clamped Hybrid ATR around 30D baseline
            u2_pct = np.maximum(MIN_U_PCT, np.clip(raw_np, 0.80 * base_pct, 1.25 * base_pct))
            u_arrays["U2_CLAMPED_HYBRID_ATR"][tfl][sym] = close_np * u2_pct

            # U3: HTF 4h Macro ATR (asof backward onto closed 4h bar)
            tf_ts = pd.DataFrame({"timestamp": pd.to_datetime(df["timestamp"], utc=True)})
            merged = pd.merge_asof(tf_ts, h4_macro_closed[sym], on="timestamp", direction="backward")
            macro_mapped = merged["macro_pct_4h"].ffill().fillna(MIN_U_PCT / max(gamma, 1e-6)).values
            u3_pct = np.maximum(MIN_U_PCT, macro_mapped * gamma)
            u_arrays["U3_HTF_MACRO_ATR"][tfl][sym] = close_np * u3_pct

            # U4: Structural Wave Blend (EMA34 vs EMA200 spread)
            ema34 = close.ewm(span=34, adjust=False).mean().values
            ema200 = close.ewm(span=200, adjust=False).mean().values
            spread_raw = 0.16 * np.abs(ema34 - ema200) / close_np
            spread_pct = np.clip(spread_raw, 0.75 * base_pct, 1.35 * base_pct)
            u4_pct = np.maximum(MIN_U_PCT, 0.60 * base_pct + 0.40 * spread_pct)
            u_arrays["U4_STRUCTURAL_WAVE_BLEND"][tfl][sym] = close_np * u4_pct

            # Diagnostic medians after warmup (bar >= 205)
            valid = slice(205, None)
            emp_gamma = float(np.nanmedian(raw_np[valid] / np.maximum(macro_mapped[valid], 1e-9)))
            diag_rows.append({
                "Coin": sym,
                "TF": tfl,
                "Med_Raw_%": round(float(np.nanmedian(raw_np[valid])) * 100.0, 4),
                "Med_U1_%": round(float(np.nanmedian(u1_pct[valid])) * 100.0, 4),
                "Med_U2_%": round(float(np.nanmedian(u2_pct[valid])) * 100.0, 4),
                "Med_U3_%": round(float(np.nanmedian(u3_pct[valid])) * 100.0, 4),
                "Med_U4_%": round(float(np.nanmedian(u4_pct[valid])) * 100.0, 4),
                "Gamma_Spec": gamma,
                "Gamma_Emp_Med": round(emp_gamma, 4),
            })

    return u_arrays, pd.DataFrame(diag_rows)


def build_per_coin_u_arrays(
    tf_dfs_map: dict,
    per_coin_table: Dict[str, Dict[str, float]],
) -> Dict[str, Dict[str, np.ndarray]]:
    """Build per-candle 1U arrays for PER_COIN_PCT_1.0x using the train-measured table."""
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for tfl in ("5m", "15m", "30m", "1h"):
        out[tfl] = {}
        for sym in SIX_COINS:
            close_np = tf_dfs_map[tfl][sym]["close"].values
            pct = per_coin_table[sym][tfl]
            out[tfl][sym] = np.maximum(close_np * pct, close_np * 1e-5)
    return out


def get_u_for_signal(
    mode_name: str,
    tfl: str,
    sig,
    u_mode_arrays: Optional[Dict[str, Dict[str, np.ndarray]]],
) -> float:
    """Resolve 1U in price units at signal entry (O(1) array lookup or pure ATR)."""
    if mode_name == "M0_PURE_ATR":
        return max(0.5 * sig.atr, sig.entry_price * 1e-5)
    return float(u_mode_arrays[tfl][sig.symbol][sig.bar_idx])


def sim_portfolio_u_arrays(
    tf_dfs_map: dict,
    tf_sigs_map: dict,
    rules_map: dict,
    mode_name: str,
    u_mode_arrays: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
    dir_filter_map: Optional[Dict[Tuple[str, str, str], int]] = None,
    base_buffer: float = 400.0,
    auto_turbo_buffer: Optional[float] = None,
    hard_floor_eq: float = 11_000.0,
    core_only_sl_trigger: bool = True,
) -> dict:
    """Portfolio simulation matching sim_portfolio_custom_u with array-backed 1U."""
    rcfg = RiskConfig(
        initial_capital=10_000.0,
        buffer_units=base_buffer,
        max_concurrent_trades=5,
        max_trades_per_day=10,
    )
    comp = CompoundingManager(rcfg)
    equity = 10_000.0
    peak_eq = equity
    open_cap = 4.25
    in_turbo = False

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

    port_consec_sl = 0
    port_shadow_paused = False
    port_paused_until = pd.Timestamp("1970-01-01", tz="UTC")

    def settle(now_ts: pd.Timestamp):
        nonlocal equity, peak_eq, port_consec_sl, port_shadow_paused, port_paused_until, in_turbo
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

        df = tf_dfs_map[tfl][sig.symbol]
        u = get_u_for_signal(mode_name, tfl, sig, u_mode_arrays)

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
    return {
        "trades": len(closed),
        "wr": wr,
        "pf": pf,
        "final_eq": float(ev[-1]),
        "ret": ret,
        "mdd": mdd,
        "calmar": ret / max(abs(mdd), 0.1),
    }


def discover_dual_split_rules_arrays(
    tf_dfs_map: dict,
    train_sigs: dict,
    mode_name: str,
    u_mode_arrays: Optional[Dict[str, Dict[str, np.ndarray]]],
    split_ts: pd.Timestamp,
    min_trades: int = 10,
) -> dict:
    """Scan all 216 setups (15m/30m/1h) on train_sigs, picking best exit_arch/l2_u passing both halves."""
    discovered = {}
    for sym in SIX_COINS:
        cc = COIN_CAL[sym]
        sl_u = float(cc["sl"])
        tp2_u = float(cc["tp2"])
        for tfl in ("15m", "30m", "1h"):
            tp1_u = float(cc["tp1"][tfl])
            df = tf_dfs_map[tfl][sym]
            h_arr, l_arr, c_arr = df["high"].values, df["low"].values, df["close"].values
            sigs_by_setup = {}
            for s in train_sigs[tfl][sym]:
                sigs_by_setup.setdefault(s.setup_type, []).append(s)

            for stype, sigs in sorted(sigs_by_setup.items()):
                if len(sigs) < min_trades:
                    continue
                best_qual = None
                for _, arch, l2_u in EXIT_CANDS:
                    if arch == "PARTIAL_MOVE_SL" and l2_u < 0 and tp1_u <= abs(l2_u) + 0.6:
                        continue
                    h1_u, h2_u, all_u = [], [], []
                    for s in sigs:
                        u = get_u_for_signal(mode_name, tfl, s, u_mode_arrays)
                        _, _, _, _, nu = sim_exit_arch(
                            h_arr, l_arr, c_arr, s.bar_idx, s.entry_price, s.direction, u,
                            sl_u, tp1_u, tp2_u, arch, l2_u,
                        )
                        all_u.append(nu)
                        (h1_u if s.timestamp <= split_ts else h2_u).append(nu)

                    def _pf(arr):
                        if not arr:
                            return 0.0
                        a = np.array(arr, dtype=float)
                        gp, gl = float(a[a > 0].sum()), float(-a[a < 0].sum())
                        return gp / gl if gl > 0.01 else (99.99 if gp > 0 else 0.0)

                    pf1 = _pf(h1_u)
                    pf2 = _pf(h2_u)
                    pfall = _pf(all_u)
                    net_all = float(sum(all_u))
                    if pf1 >= 1.05 and pf2 >= 1.05 and pfall >= 1.20 and net_all > 0:
                        cand = (net_all, pfall, arch, float(l2_u))
                        if best_qual is None or cand[0] > best_qual[0]:
                            best_qual = cand

                if best_qual is not None:
                    _, pfall, arch, l2_u = best_qual
                    vol_m = 0.85 if pfall >= 1.50 else 0.55
                    discovered[(sym, tfl, stype)] = {
                        "arch": arch,
                        "l2_u": l2_u,
                        "sl_u": sl_u,
                        "tp1_u": tp1_u,
                        "tp2_u": tp2_u,
                        "vol": vol_m,
                    }
    return discovered


def _append_result(
    rows: list,
    test_name: str,
    mode_name: str,
    setups_count: int,
    window_name: str,
    buf_name: str,
    res: dict,
):
    rows.append({
        "Test": test_name,
        "U_Mode": mode_name,
        "Setups_Count": setups_count,
        "Window": window_name,
        "Buffer_Mode": buf_name,
        "Trades": res["trades"],
        "WR_%": round(res["wr"], 2),
        "PF": round(res["pf"], 2),
        "Return_%": round(res["ret"], 2),
        "MaxDD_%": round(res["mdd"], 2),
        "Calmar": round(res["calmar"], 2),
        "Final_Eq_$": round(res["final_eq"], 2),
    })


def main():
    print("Loading 36-month cached 5m futures data (2023-11-01 -> 2026-10-31)...")
    raw_m5 = load_3year_cached_m5()
    cfg = BacktestConfig()
    dfs_m15 = {s: resample_ohlcv(d, "15min") for s, d in raw_m5.items()}
    dfs_m30 = {s: resample_ohlcv(d, "30min") for s, d in raw_m5.items()}
    dfs_h1 = {s: resample_ohlcv(d, "1h") for s, d in raw_m5.items()}
    dfs_h4 = {s: resample_ohlcv(d, "4h") for s, d in raw_m5.items()}
    tf_dfs_map = {"5m": raw_m5, "15m": dfs_m15, "30m": dfs_m30, "1h": dfs_h1}

    print("Precomputing Scale-Invariant 1U arrays (U1..U4, 100% Zero Lookahead)...")
    u_arrays, diag_df = precompute_invariant_u_arrays(tf_dfs_map, dfs_h4)
    print("\n" + "=" * 120)
    print("DIAGNOSTIC: 36M MEDIAN 1U_% BY FORMULA & EMPIRICAL TF/4H GAMMA RATIO (15m/30m/1h)")
    print("=" * 120)
    print(diag_df[diag_df["TF"].isin(["15m", "30m", "1h"])].to_string(index=False))

    print("\nGenerating 4-touch signals across 36M...")
    tf_sigs_map = {}
    for tfl, tdf, hdf, cd in [("5m", raw_m5, dfs_h1, 48), ("15m", dfs_m15, dfs_h1, 16), ("30m", dfs_m30, dfs_h4, 8), ("1h", dfs_h1, dfs_h4, 4)]:
        bst = compute_storm_flags(tdf["BTCUSDT"])
        bmask = build_cooldown_mask(bst["raw_storm"], cd)
        bts = tdf["BTCUSDT"]["timestamp"].values
        tf_sigs_map[tfl] = {
            sym: generate_all_4_touches(tdf[sym], hdf[sym], bts, bmask, sym, cfg.strategy, cooldown_bars=cd)[0]
            for sym in SIX_COINS
        }

    ts_2y_start = pd.Timestamp("2023-11-01 00:00:00", tz="UTC")
    ts_y1_end = pd.Timestamp("2024-10-31 23:59:59", tz="UTC")
    ts_y2_start = pd.Timestamp("2024-11-01 00:00:00", tz="UTC")
    ts_2y_end = pd.Timestamp("2025-10-31 23:59:59", tz="UTC")
    ts_y3_start = pd.Timestamp("2025-11-01 00:00:00", tz="UTC")
    ts_3y_end = pd.Timestamp("2026-10-31 23:59:59", tz="UTC")
    ts_y3_mid = pd.Timestamp("2026-04-30 23:59:59", tz="UTC")

    # Zero-lookahead per-coin percentage tables measured strictly on each experiment's Train window
    pct_tbl_train_2y = measure_per_coin_u_pct(tf_dfs_map, ts_2y_start, ts_2y_end)
    pct_tbl_train_y3 = measure_per_coin_u_pct(tf_dfs_map, ts_y3_start, ts_3y_end)
    pct_arrays_train_2y = build_per_coin_u_arrays(tf_dfs_map, pct_tbl_train_2y)
    pct_arrays_train_y3 = build_per_coin_u_arrays(tf_dfs_map, pct_tbl_train_y3)

    sigs_y1 = filter_sigs(tf_sigs_map, ts_2y_start, ts_y1_end)
    sigs_y2 = filter_sigs(tf_sigs_map, ts_y2_start, ts_2y_end)
    sigs_2y = filter_sigs(tf_sigs_map, ts_2y_start, ts_2y_end)
    sigs_y3 = filter_sigs(tf_sigs_map, ts_y3_start, ts_3y_end)
    sigs_36m = filter_sigs(tf_sigs_map, ts_2y_start, ts_3y_end)

    rules_75, dir_filter = load_frozen_75()
    all_rows = []

    def _resolve_arrays(mode_name: str, pct_arr: dict) -> Optional[dict]:
        if mode_name == "M0_PURE_ATR":
            return None
        if mode_name == "PER_COIN_PCT_1.0x":
            return pct_arr
        return u_arrays[mode_name]

    # =========================================================================
    # TEST A: Frozen 75 Setups (v2.3) + Rule 1B dir_filter
    # =========================================================================
    print("\nRunning Test A (75 Setups v2.3)...")
    for mode_name in ALL_MODES:
        m_arr = _resolve_arrays(mode_name, pct_arrays_train_y3)
        for w_name, wsigs in [
            ("2Y_Blind (11/2023->10/2025)", sigs_2y),
            ("1Y_Recent (11/2025->10/2026)", sigs_y3),
        ]:
            for buf_name, turbo_buf in [("Bao_400G", None), ("Auto_Turbo_250G", 250.0)]:
                r = sim_portfolio_u_arrays(
                    tf_dfs_map, wsigs, rules_75, mode_name, m_arr,
                    dir_filter_map=dir_filter, base_buffer=400.0, auto_turbo_buffer=turbo_buf,
                )
                _append_result(all_rows, "TestA_75_Setups_v2.3", mode_name, len(rules_75), w_name, buf_name, r)

    # =========================================================================
    # TEST B: Walk-Forward (Train 2Y 11/2023->10/2025 -> Blind Test Y3 11/2025->10/2026 + Full 36M)
    # =========================================================================
    print("\nRunning Test B (Walk-Forward: Train 2Y -> Blind Y3 + Full 36M)...")
    for mode_name in ALL_MODES:
        m_arr = _resolve_arrays(mode_name, pct_arrays_train_2y)
        wf_rules = discover_dual_split_rules_arrays(tf_dfs_map, sigs_2y, mode_name, m_arr, ts_y1_end, min_trades=10)
        n085 = sum(1 for v in wf_rules.values() if v["vol"] >= 0.80)
        n055 = len(wf_rules) - n085
        print(f"  Test B [{mode_name:24s}]: {len(wf_rules):2d} setups ({n085:2d} @ 0.85x, {n055:2d} @ 0.55x)")
        for w_name, wsigs in [
            ("1_Train_2Y (11/2023->10/2025)", sigs_2y),
            ("2_Blind_Y3 (11/2025->10/2026)", sigs_y3),
            ("3_Full_36M (11/2023->10/2026)", sigs_36m),
        ]:
            for buf_name, turbo_buf in [("Bao_400G", None), ("Auto_Turbo_250G", 250.0)]:
                r = sim_portfolio_u_arrays(
                    tf_dfs_map, wsigs, wf_rules, mode_name, m_arr,
                    dir_filter_map=None, base_buffer=400.0, auto_turbo_buffer=turbo_buf,
                )
                _append_result(all_rows, "TestB_WalkForward_MultiYear", mode_name, len(wf_rules), w_name, buf_name, r)

    # =========================================================================
    # TEST C: Reverse Blind (Train 1Y 11/2025->10/2026 -> Blind Y1, Blind Y2, Blind 2Y, Full 36M)
    # =========================================================================
    print("\nRunning Test C (Reverse Blind: Train Y3 -> Blind Y1, Y2, 2Y + Full 36M)...")
    for mode_name in ALL_MODES:
        m_arr = _resolve_arrays(mode_name, pct_arrays_train_y3)
        wf_rules_c = discover_dual_split_rules_arrays(tf_dfs_map, sigs_y3, mode_name, m_arr, ts_y3_mid, min_trades=8)
        n085 = sum(1 for v in wf_rules_c.values() if v["vol"] >= 0.80)
        n055 = len(wf_rules_c) - n085
        print(f"  Test C [{mode_name:24s}]: {len(wf_rules_c):2d} setups ({n085:2d} @ 0.85x, {n055:2d} @ 0.55x)")
        for w_name, wsigs in [
            ("1_Train_Y3 (11/2025->10/2026)", sigs_y3),
            ("2_Blind_Y1 (11/2023->10/2024)", sigs_y1),
            ("3_Blind_Y2 (11/2024->10/2025)", sigs_y2),
            ("4_Blind_2Y (11/2023->10/2025)", sigs_2y),
            ("5_Full_36M (11/2023->10/2026)", sigs_36m),
        ]:
            for buf_name, turbo_buf in [("Bao_400G", None), ("Auto_Turbo_250G", 250.0)]:
                r = sim_portfolio_u_arrays(
                    tf_dfs_map, wsigs, wf_rules_c, mode_name, m_arr,
                    dir_filter_map=None, base_buffer=400.0, auto_turbo_buffer=turbo_buf,
                )
                _append_result(all_rows, "TestC_TrainY3_Blind2Y", mode_name, len(wf_rules_c), w_name, buf_name, r)

    df_out = pd.DataFrame(all_rows)
    out_csv = os.path.join(RESULTS_DIR, "invariant_1u_benchmark.csv")
    df_out.to_csv(out_csv, index=False)

    for test_key, title in [
        ("TestA_75_Setups_v2.3", "TEST A: 75 SETUPS v2.3 — U1..U4 vs PER_COIN_PCT_1.0x vs M0_PURE_ATR"),
        ("TestB_WalkForward_MultiYear", "TEST B: WALK-FORWARD (TRAIN 2Y 11/2023->10/2025 -> BLIND Y3 11/2025->10/2026 + 36M)"),
        ("TestC_TrainY3_Blind2Y", "TEST C: REVERSE BLIND (TRAIN Y3 11/2025->10/2026 -> BLIND Y1, Y2, 2Y + 36M)"),
    ]:
        print("\n" + "=" * 136)
        print(title)
        print("=" * 136)
        print(df_out[df_out["Test"] == test_key].to_string(index=False))

    print(f"\nSaved results to: {out_csv}")


if __name__ == "__main__":
    main()
