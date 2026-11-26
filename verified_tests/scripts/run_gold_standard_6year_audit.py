"""CHICKTRADE — Gold-Standard 6-Year Institutional Audit Engine (2021-01 -> 2026-10).

Zero shortcuts, zero assumptions, full microstructure fidelity:
  1. Real 6-year Binance USD-M Futures 5m OHLCV resampled cleanly to 15m/30m/1h/4h.
  2. Real historical 8-hour Binance Funding Rates integrated at every exact funding timestamp
     while a position is open (-dir * notional * funding_rate).
  3. Exact LIMIT IOC +/- 0.25U fill model checked on the next bar's open/high/low.
  4. Conservative SL-before-TP intra-bar ordering; real exit timestamp (NO 4/8-bar shortcut).
  5. Full S1 -> S5 stress ladder (S1_REAL_EXIT, S2_TAKER, S3_IOC, S4_REAL_FUNDING, S5_PESSIMISTIC).
  6. Strict Cross-Period Blind Test (2023-2026 <-> 2021-2023) + 8-Fold Rolling Walk-Forward (2023-01 -> 2026-10).
  7. Head-to-head Engine Benchmark under identical S4_REAL_FUNDING costs:
       - v3.3 58 Setups (EMA)
       - v4.0 124 Setups (EMA Touch 1-5)
       - Walk-Forward EMA (8 folds)
       - CHICKTRADE-TREND 4H (Donchian 100 + 6xATR Trailing, LONG-only, Fixed & Walk-Forward)
       - CHICKTRADE-EMA-TRAIL 4H (EMA34/89 Pullback + 5xATR Trailing Stop, LONG-only & Both-Side)
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DIR_6Y = os.path.join(ROOT, "data", "cache", "6y")
V3_RULES = os.path.join(ROOT, "core", "core_strategy_rules.json")
V4_JSON = os.path.join(ROOT, "verified_tests", "results", "v4_proposed_setups_3year.json")
OUT_JSON = os.path.join(ROOT, "verified_tests", "results", "gold_standard_6year_audit.json")

SIX_COINS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "LINKUSDT", "XRPUSDT", "SOLUSDT"]
TFS = ("15m", "30m", "1h")
GAMMA_TF = {"5m": 0.17, "15m": 0.29, "30m": 0.41, "1h": 0.59}
TF_COOLDOWN = {"5m": 48, "15m": 16, "30m": 8, "1h": 4}
TF_PRIORITY = {"1h": 0, "30m": 1, "15m": 2, "5m": 3}
TF_MS = {"5m": 300_000, "15m": 900_000, "30m": 1_800_000, "1h": 3_600_000, "4h": 14_400_000}
MAX_HOLD_BARS = {"5m": 1440, "15m": 960, "30m": 480, "1h": 360}

TAKER_FEE = 0.0005
MAKER_FEE = 0.0002
SLIP_BPS = 0.0001
MIN_U_PCT = 0.0008


def load_coin_6y(sym: str) -> Tuple[Dict[str, pd.DataFrame], np.ndarray, np.ndarray]:
    df5 = pd.read_csv(os.path.join(DIR_6Y, f"{sym}_5m_2021_2026.csv.gz"))
    df5["timestamp"] = pd.to_datetime(df5["open_time_ms"], unit="ms", utc=True)
    df5["close_time_ms"] = df5["open_time_ms"] + TF_MS["5m"] - 1

    out = {"5m": df5}
    for tf, rule in (("15m", "15min"), ("30m", "30min"), ("1h", "1h"), ("4h", "4h")):
        g = df5.set_index("timestamp").resample(rule).agg({
            "open_time_ms": "first", "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
        }).dropna().reset_index()
        g["open_time_ms"] = g["open_time_ms"].astype("int64")
        g["close_time_ms"] = g["open_time_ms"] + TF_MS[tf] - 1
        out[tf] = g

    fr = pd.read_csv(os.path.join(DIR_6Y, f"{sym}_funding_2021_2026.csv.gz"))
    fr_ts = fr["calc_time_ms"].values.astype("int64") if len(fr) else np.array([], dtype="int64")
    fr_rate = fr["funding_rate"].values.astype("float64") if len(fr) else np.array([], dtype="float64")
    return out, fr_ts, fr_rate


def compute_atr14(df: pd.DataFrame) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / 14, adjust=False).mean()


def compute_adx14(df: pd.DataFrame) -> np.ndarray:
    h, l, c = df["high"], df["low"], df["close"]
    plus_dm = h.diff().clip(lower=0.0)
    minus_dm = (-l.diff()).clip(lower=0.0)
    plus_dm = plus_dm.where(plus_dm > minus_dm, 0.0)
    minus_dm = minus_dm.where(minus_dm > plus_dm, 0.0)
    atr = compute_atr14(df)
    plus_di = 100.0 * plus_dm.ewm(alpha=1.0 / 14, adjust=False).mean() / atr.replace(0, np.nan)
    minus_di = 100.0 * minus_dm.ewm(alpha=1.0 / 14, adjust=False).mean() / atr.replace(0, np.nan)
    dx = (100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)).fillna(0.0)
    return dx.ewm(alpha=1.0 / 14, adjust=False).mean().values


def compute_storm_cooldown(df: pd.DataFrame, cooldown_bars: int) -> Tuple[np.ndarray, np.ndarray]:
    h, l, c, o = df["high"], df["low"], df["close"], df["open"]
    atr = compute_atr14(df)
    atr_ma50 = atr.rolling(50, min_periods=10).mean()
    atr_ratio = (atr / atr_ma50.replace(0, np.nan)).fillna(1.0)
    crange = (h - l).replace(0, np.nan)
    body = (c - o).abs()
    wick_frac = (1.0 - body / crange).fillna(0.0)
    sweep = (crange >= 2.8 * atr) | ((crange >= 2.0 * atr) & (wick_frac >= 0.70))
    net_move = (c - c.shift(10)).abs()
    path_len = c.diff().abs().rolling(10, min_periods=10).sum().replace(0, np.nan)
    er = (net_move / path_len).fillna(0.5)
    raw = (atr_ratio >= 1.8) | sweep | ((er < 0.15) & (atr_ratio >= 1.20))
    return (raw.astype(int).rolling(cooldown_bars, min_periods=1).max() > 0).values, atr_ratio.values


def real_funding_u(
    fr_ts: np.ndarray,
    fr_rate: np.ndarray,
    open_ts: int,
    close_ts: int,
    d: int,
    u_frac: float,
    tp1_ts: Optional[int] = None,
) -> float:
    """Exact historical Binance 8h funding impact expressed in U (positive return = cost deducted from PnL)."""
    if len(fr_ts) == 0:
        return 0.0
    lo = int(np.searchsorted(fr_ts, open_ts, side="right"))
    hi = int(np.searchsorted(fr_ts, close_ts, side="right"))
    if hi <= lo:
        return 0.0
    total_frac = 0.0
    for k in range(lo, hi):
        size_w = 0.5 if (tp1_ts is not None and fr_ts[k] > tp1_ts) else 1.0
        # Long pays when funding_rate > 0; Short receives when funding_rate > 0
        total_frac += d * fr_rate[k] * size_w
    return float(total_frac / max(u_frac, 1e-6))


def extract_ema_signals_6y(
    market: Dict[str, Dict[str, pd.DataFrame]],
    tfs: Tuple[str, ...] = ("5m", "15m", "30m", "1h"),
    max_wave: int = 5,
) -> Dict[Tuple[str, str, str], List[dict]]:
    """Extract all continuous converged EMA 34/89/200 W1..W5 (+W4/W5 REV) signals across 6 years."""
    sig_map: Dict[Tuple[str, str, str], List[dict]] = {}
    for tf in tfs:
        htf_name = "1h" if tf in ("5m", "15m") else "4h"
        cd_bars = TF_COOLDOWN[tf]
        btc_df = market["BTCUSDT"][tf]
        btc_cd, _ = compute_storm_cooldown(btc_df, cd_bars)
        btc_ts = btc_df["open_time_ms"].values
        gamma = GAMMA_TF[tf]

        for sym in SIX_COINS:
            df = market[sym][tf]
            hdf = market[sym][htf_name]
            df4 = market[sym]["4h"]

            closes = df["close"].values
            opens = df["open"].values
            highs = df["high"].values
            lows = df["low"].values
            ts_ms = df["open_time_ms"].values
            close_ms = df["close_time_ms"].values

            ema34 = df["close"].ewm(span=34, adjust=False).mean().values
            ema89 = df["close"].ewm(span=89, adjust=False).mean().values
            ema200 = df["close"].ewm(span=200, adjust=False).mean().values
            atr = compute_atr14(df).values
            adx = compute_adx14(df)
            local_cd, atr_ratio = compute_storm_cooldown(df, cd_bars)

            htf_e34 = hdf["close"].ewm(span=34, adjust=False).mean()
            htf_e200 = hdf["close"].ewm(span=200, adjust=False).mean()
            htf_tr = np.where((hdf["close"] > htf_e200) & (htf_e34 > htf_e200), 1, np.where((hdf["close"] < htf_e200) & (htf_e34 < htf_e200), -1, 0))
            htf_close_ts = hdf["close_time_ms"].values

            atr4 = compute_atr14(df4)
            macro4 = ((0.5 * atr4) / df4["close"]).ewm(span=20, adjust=False).mean().values
            h4_close_ts = df4["close_time_ms"].values

            ema_specs = [(34, ema34, 0.8, 3), (89, ema89, 1.0, 5), (200, ema200, 1.2, 8)]
            for ema_p, et_arr, imp_mult, min_gap in ema_specs:
                trend = 0
                wave_count = 0
                armed = False
                last_touch = -999
                for i in range(205, len(df) - 2):
                    c, o, h, l, a = closes[i], opens[i], highs[i], lows[i], atr[i]
                    e34, e89, e200, et = ema34[i], ema89[i], ema200[i], et_arr[i]
                    if np.isnan(a) or a <= 0 or np.isnan(e200):
                        continue
                    if ema_p == 34:
                        new_trend = 1 if (c > e200 and e34 > e89 > e200) else (-1 if (c < e200 and e34 < e89 < e200) else 0)
                    elif ema_p == 89:
                        new_trend = 1 if (c > e200 and e89 > e200) else (-1 if (c < e200 and e89 < e200) else 0)
                    else:
                        new_trend = 1 if (e89 > e200 and c >= e200 - 0.8 * a) else (-1 if (e89 < e200 and c <= e200 + 0.8 * a) else 0)
                    if new_trend != trend:
                        trend = new_trend
                        wave_count = 0
                        armed = False
                        last_touch = -999
                        continue
                    if trend == 0:
                        continue
                    if trend == 1 and (h - et) >= imp_mult * a and l > et:
                        armed = True
                    elif trend == -1 and (et - l) >= imp_mult * a and h < et:
                        armed = True
                    if not armed or (i - last_touch) < min_gap:
                        continue
                    touch_buf, breach_buf = 0.20 * a, 0.65 * a
                    touched = (l <= et + touch_buf and c >= et - breach_buf) if trend == 1 else (h >= et - touch_buf and c <= et + breach_buf)
                    if not touched:
                        continue
                    wave_count += 1
                    armed = False
                    last_touch = i
                    if wave_count > max_wave or adx[i] < 20.0 or local_cd[i]:
                        continue
                    b_idx = int(np.searchsorted(btc_ts, ts_ms[i], side="right")) - 1
                    if b_idx >= 0 and btc_cd[b_idx]:
                        continue
                    h_idx = int(np.searchsorted(htf_close_ts, close_ms[i], side="right")) - 1
                    if h_idx < 0:
                        continue
                    ht = int(htf_tr[h_idx])
                    h4_idx = int(np.searchsorted(h4_close_ts, close_ms[i], side="right")) - 1
                    if h4_idx < 0:
                        continue
                    u_frac = max(MIN_U_PCT, float(macro4[h4_idx]) * gamma)
                    base = {
                        "idx": i, "sym": sym, "tf": tf, "open_ts": int(ts_ms[i]),
                        "entry": float(c), "u_frac": u_frac, "u_px": float(c * u_frac),
                        "elevated_vol": bool(atr_ratio[i] >= 1.35),
                    }
                    if ht == trend:
                        st = f"EMA{ema_p}_W{wave_count}"
                        sig_map.setdefault((sym, tf, st), []).append({**base, "setup": st, "dir": trend})
                    if wave_count in (4, 5):
                        st = f"EMA{ema_p}_W{wave_count}_REV"
                        sig_map.setdefault((sym, tf, st), []).append({**base, "setup": st, "dir": -trend})
    return sig_map


def sim_trade_scenario(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    ts_ms: np.ndarray,
    fr_ts: np.ndarray,
    fr_rate: np.ndarray,
    s: dict,
    sl_u: float,
    tp1_u: float,
    tp2_u: float,
    exit_mode: str,
    max_bars: int,
    scenario: str = "S4_REAL_FUNDING",
) -> Optional[dict]:
    """Simulate a single trade under S1_REAL_EXIT / S2_TAKER / S3_IOC / S4_REAL_FUNDING / S5_PESSIMISTIC."""
    i = s["idx"]
    if i + 1 >= len(highs):
        return None
    d = s["dir"]
    p_sig = s["entry"]
    u_px = s["u_px"]
    u_frac = s["u_frac"]

    # IOC +/- 0.25U fill check on bar i+1
    use_ioc = scenario in ("S3_IOC", "S4_REAL_FUNDING", "S5_PESSIMISTIC")
    fill_px = p_sig
    if use_ioc:
        nxt_o, nxt_h, nxt_l = opens[i + 1], highs[i + 1], lows[i + 1]
        lim_px = p_sig + d * 0.25 * u_px
        if d == 1:
            if nxt_o > lim_px:
                if nxt_l > lim_px:
                    return {"ioc_missed": True}
                fill_px = lim_px
            else:
                fill_px = nxt_o
        else:
            if nxt_o < lim_px:
                if nxt_h < lim_px:
                    return {"ioc_missed": True}
                fill_px = lim_px
            else:
                fill_px = nxt_o

    extra_slip_u = 0.10 if scenario == "S5_PESSIMISTIC" else 0.0
    if extra_slip_u > 0:
        fill_px = fill_px + d * extra_slip_u * u_px

    # Structural SL/TP anchored to signal price P_sig (matching live bot!), BE anchored to fill_px
    sl_px = p_sig - d * sl_u * u_px
    tp1_px = p_sig + d * tp1_u * u_px
    tp2_px = p_sig + d * tp2_u * u_px
    if exit_mode == "PARTIAL_BE":
        l2_sl_px = fill_px * (1.0 + d * 2.0 * TAKER_FEE)
    elif exit_mode == "PARTIAL_M1U":
        l2_sl_px = p_sig - d * 1.0 * u_px
    else:
        l2_sl_px = sl_px

    entry_fee = MAKER_FEE if scenario == "S1_REAL_EXIT" else TAKER_FEE
    entry_cost_u = entry_fee / u_frac
    maker_exit_u = MAKER_FEE / u_frac
    taker_exit_u = (TAKER_FEE + SLIP_BPS) / u_frac + extra_slip_u

    tp1_hit = False
    tp1_ts: Optional[int] = None
    active_sl = sl_px
    pnl_u = 0.0
    outcome = "TIMEOUT"
    max_j = min(len(highs), i + max_bars + 1)
    exit_idx = max_j - 1

    for j in range(i + 1, max_j):
        bh, bl = highs[j], lows[j]
        if exit_mode in ("ONLY_TP1", "ONLY_1R"):
            tgt_px = tp1_px if exit_mode == "ONLY_TP1" else (p_sig + d * sl_u * u_px)
            if (d == 1 and bl <= sl_px) or (d == -1 and bh >= sl_px):
                pnl_u = d * (sl_px - fill_px) / u_px - entry_cost_u - taker_exit_u
                outcome = "SL"
                exit_idx = j
                break
            if (d == 1 and bh >= tgt_px) or (d == -1 and bl <= tgt_px):
                pnl_u = d * (tgt_px - fill_px) / u_px - entry_cost_u - maker_exit_u
                outcome = exit_mode
                exit_idx = j
                break
        else:
            if not tp1_hit:
                if (d == 1 and bl <= active_sl) or (d == -1 and bh >= active_sl):
                    pnl_u = d * (active_sl - fill_px) / u_px - entry_cost_u - taker_exit_u
                    outcome = "SL"
                    exit_idx = j
                    break
                if (d == 1 and bh >= tp1_px) or (d == -1 and bl <= tp1_px):
                    tp1_hit = True
                    tp1_ts = int(ts_ms[j])
                    pnl_u += 0.5 * (d * (tp1_px - fill_px) / u_px - entry_cost_u - maker_exit_u)
                    active_sl = l2_sl_px
                    if (d == 1 and bh >= tp2_px) or (d == -1 and bl <= tp2_px):
                        pnl_u += 0.5 * (d * (tp2_px - fill_px) / u_px - entry_cost_u - maker_exit_u)
                        outcome = "TP2"
                        exit_idx = j
                        break
            else:
                if (d == 1 and bl <= active_sl) or (d == -1 and bh >= active_sl):
                    pnl_u += 0.5 * (d * (active_sl - fill_px) / u_px - entry_cost_u - taker_exit_u)
                    outcome = "LEG2_SL"
                    exit_idx = j
                    break
                if (d == 1 and bh >= tp2_px) or (d == -1 and bl <= tp2_px):
                    pnl_u += 0.5 * (d * (tp2_px - fill_px) / u_px - entry_cost_u - maker_exit_u)
                    outcome = "TP2"
                    exit_idx = j
                    break
    else:
        rem_u = d * (closes[exit_idx] - fill_px) / u_px - entry_cost_u - taker_exit_u
        pnl_u += (0.5 if tp1_hit else 1.0) * rem_u

    f_u = 0.0
    if scenario in ("S4_REAL_FUNDING", "S5_PESSIMISTIC"):
        f_u = real_funding_u(fr_ts, fr_rate, s["open_ts"], int(ts_ms[exit_idx]), d, u_frac, tp1_ts)
        pnl_u -= f_u

    return {
        "ioc_missed": False,
        "sym": s["sym"], "tf": s["tf"], "setup": s["setup"], "dir": d,
        "open_ts": s["open_ts"], "close_ts": int(ts_ms[exit_idx]),
        "bars_held": int(exit_idx - i), "entry": fill_px, "u_frac": u_frac,
        "sl_u": sl_u, "outcome": outcome,
        "pnl_u": round(float(pnl_u), 4), "funding_u": round(float(f_u), 4),
        "elevated_vol": s["elevated_vol"],
    }


def simulate_portfolio(
    trades: List[dict],
    ioc_missed: int = 0,
    buffer_base: float = 400.0,
    turbo_buffer: float = 250.0,
    use_turbo: bool = False,
    max_same_dir: int = 3,
    use_2sl_brake: bool = True,
    init_cap: float = 10_000.0,
) -> dict:
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
    open_pos: List[dict] = []
    closed: List[dict] = []
    curve: List[dict] = []
    total_funding = 0.0

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
                if use_2sl_brake:
                    if p["outcome"] == "SL" and p.get("vol_mult", 1.0) > 0.45:
                        consec_sl += 1
                        if consec_sl >= 2:
                            shadow_paused = True
                            shadow_until = p["close_ts"] + 72 * 3_600_000
                            consec_sl = 0
                    elif p["pnl_usd"] > 0:
                        consec_sl = 0
                closed.append(p)
                dt_s = datetime.fromtimestamp(p["close_ts"] / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d")
                if not curve or curve[-1]["dt"] != dt_s:
                    curve.append({"dt": dt_s, "eq": round(eq, 2)})
                else:
                    curve[-1]["eq"] = round(eq, 2)
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

        eff_m = float(t.get("vol_mult", 1.0))
        if t.get("elevated_vol"):
            eff_m *= 0.5
        if (eq - peak) / peak * 100.0 <= -6.0:
            eff_m = min(eff_m, 0.25)

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
            **t, "notional": notional, "pnl_usd": t["pnl_u"] * u_usd,
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
    tot_att = n + ioc_missed
    return {
        "trades": n,
        "ioc_missed": ioc_missed,
        "ioc_miss_pct": round(ioc_missed / max(tot_att, 1) * 100.0, 2),
        "wr": round(wins / max(n, 1) * 100.0, 1),
        "pwr": round(gp / max(gp + gl, 1e-9) * 100.0, 1),
        "pf": round(gp / max(gl, 1e-9), 2) if gl > 1e-9 else 0.0,
        "final_eq": round(eq, 2),
        "roi_pct": round(roi, 2),
        "max_dd_pct": round(max_dd, 2),
        "calmar": round(calmar, 2),
        "max_step": step,
        "funding_usd": round(total_funding, 2),
        "curve": curve,
    }


def eval_rule_set_window(
    market: Dict[str, Dict[str, pd.DataFrame]],
    funding_map: Dict[str, Tuple[np.ndarray, np.ndarray]],
    sig_map: Dict[Tuple[str, str, str], List[dict]],
    rules: List[dict],
    start_ms: int,
    end_ms: int,
    scenario: str,
) -> Tuple[List[dict], int]:
    trades = []
    missed = 0
    for r in rules:
        sym, tf, st = r["sym"], r["tf"], r["setup"]
        sigs = sig_map.get((sym, tf, st), [])
        if not sigs:
            continue
        df = market[sym][tf]
        op, hi, lo, cl, ts = df["open"].values, df["high"].values, df["low"].values, df["close"].values, df["open_time_ms"].values
        fr_ts, fr_rate = funding_map[sym]
        mb = MAX_HOLD_BARS[tf]
        busy_idx = -1
        for s in sigs:
            if s["open_ts"] < start_ms or s["open_ts"] >= end_ms:
                continue
            if s["idx"] <= busy_idx:
                continue
            res = sim_trade_scenario(op, hi, lo, cl, ts, fr_ts, fr_rate, s, r["sl_u"], r["tp1_u"], r["tp2_u"], r["mode"], mb, scenario)
            if res is None:
                continue
            if res.get("ioc_missed"):
                missed += 1
                continue
            busy_idx = s["idx"] + res["bars_held"]
            res["vol_mult"] = r["vol_mult"]
            res["score"] = r.get("score", 1.0)
            trades.append(res)
    return trades, missed


def run_donchian_4h_engine(
    market: Dict[str, Dict[str, pd.DataFrame]],
    funding_map: Dict[str, Tuple[np.ndarray, np.ndarray]],
    start_ms: int,
    end_ms: int,
    lookback: int = 100,
    atr_mult: float = 6.0,
    scenario: str = "S4_REAL_FUNDING",
) -> List[dict]:
    """4H Donchian Breakout + ATR Trailing Stop (LONG-only) under exact S4_REAL_FUNDING costs."""
    trades = []
    for sym in SIX_COINS:
        df = market[sym]["4h"]
        op, hi, lo, cl, ts = df["open"].values, df["high"].values, df["low"].values, df["close"].values, df["open_time_ms"].values
        atr = compute_atr14(df).values
        dc_high = pd.Series(hi).shift(1).rolling(lookback).max().values
        fr_ts, fr_rate = funding_map[sym]

        i = max(lookback + 5, 200)
        while i < len(df) - 2:
            if ts[i] < start_ms or ts[i] >= end_ms:
                i += 1
                continue
            if np.isnan(dc_high[i]) or np.isnan(atr[i]) or atr[i] <= 0:
                i += 1
                continue
            if cl[i] > dc_high[i]:
                # Enter on next bar open (realistic Taker + slip)
                fill_px = op[i + 1] * (1.0 + SLIP_BPS)
                a0 = atr[i]
                # Express risk unit 1U = 1 * ATR / fill_px so SL = atr_mult * U (clean 1:1 sizing with Bao 400G!)
                u_px = a0
                u_frac = u_px / fill_px
                peak_h = fill_px
                trail_sl = peak_h - atr_mult * a0
                exit_idx = len(df) - 1
                exit_px = cl[exit_idx]
                outcome = "TIMEOUT"

                for j in range(i + 1, len(df)):
                    if lo[j] <= trail_sl:
                        exit_px = min(op[j], trail_sl) * (1.0 - SLIP_BPS)
                        exit_idx = j
                        outcome = "TRAIL_SL" if exit_px > fill_px else "SL"
                        break
                    if hi[j] > peak_h:
                        peak_h = hi[j]
                        trail_sl = max(trail_sl, peak_h - atr_mult * atr[j])

                entry_fee = TAKER_FEE / u_frac
                exit_fee = (TAKER_FEE + SLIP_BPS) / u_frac
                f_u = real_funding_u(fr_ts, fr_rate, int(ts[i + 1]), int(ts[exit_idx]), 1, u_frac) if scenario in ("S4_REAL_FUNDING", "S5_PESSIMISTIC") else 0.0
                pnl_u = (exit_px - fill_px) / u_px - entry_fee - exit_fee - f_u
                trades.append({
                    "sym": sym, "tf": "1h", "setup": f"DONCHIAN_{lookback}_{atr_mult}ATR", "dir": 1,
                    "open_ts": int(ts[i + 1]), "close_ts": int(ts[exit_idx]),
                    "entry": fill_px, "u_frac": u_frac, "sl_u": atr_mult,
                    "outcome": outcome, "pnl_u": round(float(pnl_u), 4), "funding_u": round(float(f_u), 4),
                    "vol_mult": 1.0, "score": 10.0, "elevated_vol": False,
                })
                i = exit_idx + 1
            else:
                i += 1
    return trades


def run_ema_trail_4h_engine(
    market: Dict[str, Dict[str, pd.DataFrame]],
    funding_map: Dict[str, Tuple[np.ndarray, np.ndarray]],
    start_ms: int,
    end_ms: int,
    atr_mult: float = 5.0,
    long_only: bool = True,
    scenario: str = "S4_REAL_FUNDING",
) -> List[dict]:
    """4H EMA34/89 Pullback + ATR Trailing Stop (lets winners run without fixed TP cap) under S4_REAL_FUNDING."""
    trades = []
    for sym in SIX_COINS:
        df = market[sym]["4h"]
        op, hi, lo, cl, ts = df["open"].values, df["high"].values, df["low"].values, df["close"].values, df["open_time_ms"].values
        ema34 = df["close"].ewm(span=34, adjust=False).mean().values
        ema89 = df["close"].ewm(span=89, adjust=False).mean().values
        ema200 = df["close"].ewm(span=200, adjust=False).mean().values
        atr = compute_atr14(df).values
        adx = compute_adx14(df)
        fr_ts, fr_rate = funding_map[sym]

        i = 205
        while i < len(df) - 2:
            if ts[i] < start_ms or ts[i] >= end_ms:
                i += 1
                continue
            a0 = atr[i]
            if np.isnan(a0) or a0 <= 0 or adx[i] < 20.0:
                i += 1
                continue
            d = 0
            if cl[i] > ema200[i] and ema34[i] > ema89[i] > ema200[i] and lo[i] <= ema34[i] + 0.25 * a0 and cl[i] > ema34[i]:
                d = 1
            elif (not long_only) and cl[i] < ema200[i] and ema34[i] < ema89[i] < ema200[i] and hi[i] >= ema34[i] - 0.25 * a0 and cl[i] < ema34[i]:
                d = -1
            if d == 0:
                i += 1
                continue

            fill_px = op[i + 1] * (1.0 + d * SLIP_BPS)
            u_px = a0
            u_frac = u_px / fill_px
            extreme = fill_px
            trail_sl = extreme - d * atr_mult * a0
            exit_idx = len(df) - 1
            exit_px = cl[exit_idx]
            outcome = "TIMEOUT"

            for j in range(i + 1, len(df)):
                if (d == 1 and lo[j] <= trail_sl) or (d == -1 and hi[j] >= trail_sl):
                    raw_ex = min(op[j], trail_sl) if d == 1 else max(op[j], trail_sl)
                    exit_px = raw_ex * (1.0 - d * SLIP_BPS)
                    exit_idx = j
                    outcome = "TRAIL_SL" if d * (exit_px - fill_px) > 0 else "SL"
                    break
                if d == 1 and hi[j] > extreme:
                    extreme = hi[j]
                    trail_sl = max(trail_sl, extreme - atr_mult * atr[j])
                elif d == -1 and lo[j] < extreme:
                    extreme = lo[j]
                    trail_sl = min(trail_sl, extreme + atr_mult * atr[j])

            f_u = real_funding_u(fr_ts, fr_rate, int(ts[i + 1]), int(ts[exit_idx]), d, u_frac) if scenario in ("S4_REAL_FUNDING", "S5_PESSIMISTIC") else 0.0
            pnl_u = d * (exit_px - fill_px) / u_px - (2.0 * TAKER_FEE + SLIP_BPS) / u_frac - f_u
            trades.append({
                "sym": sym, "tf": "1h", "setup": f"EMA34_TRAIL_{atr_mult}ATR", "dir": d,
                "open_ts": int(ts[i + 1]), "close_ts": int(ts[exit_idx]),
                "entry": fill_px, "u_frac": u_frac, "sl_u": atr_mult,
                "outcome": outcome, "pnl_u": round(float(pnl_u), 4), "funding_u": round(float(f_u), 4),
                "vol_mult": 1.0, "score": 10.0, "elevated_vol": False,
            })
            i = exit_idx + 1
    return trades


def main():
    print("Loading 6-year (2021-01 -> 2026-10) 5m data + exact 8h Binance Funding Rates...")
    market: Dict[str, Dict[str, pd.DataFrame]] = {}
    funding_map: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    with ThreadPoolExecutor(max_workers=6) as pool:
        futs = {sym: pool.submit(load_coin_6y, sym) for sym in SIX_COINS}
        for sym, fut in futs.items():
            m_d, f_ts, f_rt = fut.result()
            market[sym] = m_d
            funding_map[sym] = (f_ts, f_rt)

    print("Extracting converged 6-year EMA signals across 5m/15m/30m/1h...")
    sig_map = extract_ema_signals_6y(market, tfs=("5m", "15m", "30m", "1h"), max_wave=5)

    # Load v3.3 58 setups & v4.0 124 setups
    with open(V3_RULES, encoding="utf-8") as f:
        v3_raw = json.load(f)["rules"]
    v3_rules = []
    for r in v3_raw:
        arch = r["exit_arch"]
        mode = "PARTIAL_BE" if "BE" in r.get("leg2_sl_label", "") else ("PARTIAL_M1U" if arch == "PARTIAL_MOVE_SL" else arch)
        v3_rules.append({
            "sym": r["coin"], "tf": r["tf"], "setup": r["setup_code"],
            "sl_u": float(r["sl_u"]), "tp1_u": float(r["tp1_u"]), "tp2_u": float(r["tp2_u"]),
            "mode": mode, "vol_mult": float(r["adaptive_volume"]["init_vol_mult"]), "score": 10.0,
        })

    with open(V4_JSON, encoding="utf-8") as f:
        v4_raw = json.load(f)["coins"]
    v4_rules = []
    for sym, c in v4_raw.items():
        for r in c["setups"]:
            if r["status"] in ("CORE", "SCOUT"):
                v4_rules.append({
                    "sym": sym, "tf": r["tf"], "setup": r["setup"],
                    "sl_u": float(r["sl_u"]), "tp1_u": float(r["tp1_u"]), "tp2_u": float(r["tp2_u"]),
                    "mode": r["mode"], "vol_mult": float(r["vol_mult"]), "score": float(r["score"]),
                })

    t_2021 = int(datetime(2021, 2, 1, tzinfo=timezone.utc).timestamp() * 1000)
    t_2023_11 = int(datetime(2023, 11, 1, tzinfo=timezone.utc).timestamp() * 1000)
    t_2025_11 = int(datetime(2025, 11, 1, tzinfo=timezone.utc).timestamp() * 1000)
    t_2026_11 = int(datetime(2026, 11, 1, tzinfo=timezone.utc).timestamp() * 1000)

    # 1. S1 -> S5 Stress Ladder on v4.0 (124 setups) & v3.3 (58 setups) across 2021-2023 BLIND vs 2023-2026 IS
    scenarios = ["S1_REAL_EXIT", "S2_TAKER", "S3_IOC", "S4_REAL_FUNDING", "S5_PESSIMISTIC"]
    stress_results = {"v4_124": {}, "v3_58": {}}

    print("\n=== PROTOCOL A: 6-YEAR S1 -> S5 STRESS LADDER (2021-2023 BLIND OOS vs 2023-2026) ===")
    for tag, rules in (("v4_124", v4_rules), ("v3_58", v3_rules)):
        for w_name, w0, w1 in (("BLIND_2021_2023", t_2021, t_2023_11), ("IS_2023_2026", t_2023_11, t_2026_11)):
            stress_results[tag][w_name] = {}
            for sc in scenarios:
                tr, miss = eval_rule_set_window(market, funding_map, sig_map, rules, w0, w1, sc)
                p400 = simulate_portfolio(tr, miss, 400, 250, False, 3, True)
                p400_nocap = simulate_portfolio(tr, miss, 400, 250, False, 5, True)
                stress_results[tag][w_name][sc] = {"bao_400g": p400, "bao_400g_5dir": p400_nocap}
                if sc in ("S1_REAL_EXIT", "S3_IOC", "S4_REAL_FUNDING", "S5_PESSIMISTIC"):
                    print(
                        f"  {tag:6s} | {w_name:15s} | {sc:15s} | N={p400['trades']:4d} IOC_miss={p400['ioc_miss_pct']:4.1f}% "
                        f"| PF={p400['pf']:4.2f} PWR={p400['pwr']:4.1f}% ROI={p400['roi_pct']:+7.1f}% DD={p400['max_dd_pct']:6.2f}% Fund=${p400['funding_usd']:,.0f}"
                    )

    # 2. Protocol B: 8-Fold Rolling Walk-Forward (24M Train -> 6M Blind Test, 2023-01 -> 2026-10) under S4_REAL_FUNDING
    print("\n=== PROTOCOL B: 8-FOLD ROLLING WALK-FORWARD EMA (2023-01 -> 2026-10, S4_REAL_FUNDING) ===")
    wf_folds = [
        ("2021-01", "2023-01", "2023-07"),
        ("2021-07", "2023-07", "2024-01"),
        ("2022-01", "2024-01", "2024-07"),
        ("2022-07", "2024-07", "2025-01"),
        ("2023-01", "2025-01", "2025-07"),
        ("2023-07", "2025-07", "2026-01"),
        ("2024-01", "2026-01", "2026-07"),
        ("2024-05", "2026-05", "2026-11"),
    ]
    def _ym_ms(ym: str) -> int:
        y, m = map(int, ym.split("-"))
        return int(datetime(y, m, 1, tzinfo=timezone.utc).timestamp() * 1000)

    # Pre-sim all M15/M30/1H W1-W4 signals once with standard PARTIAL_M1U (6U/4U/12U) under S4_REAL_FUNDING for honest WF
    all_base_rules = [
        {"sym": sym, "tf": tf, "setup": st, "sl_u": 6.0, "tp1_u": 4.0, "tp2_u": 12.0, "mode": "PARTIAL_M1U", "vol_mult": 1.0, "score": 1.0}
        for sym in SIX_COINS for tf in ("15m", "30m", "1h")
        for st in ([f"EMA{e}_W{w}" for e in (34, 89, 200) for w in (1, 2, 3)] + [f"EMA{e}_W4_REV" for e in (34, 89, 200)])
    ]
    pre_tr, pre_miss = eval_rule_set_window(market, funding_map, sig_map, all_base_rules, t_2021, t_2026_11, "S4_REAL_FUNDING")
    by_key: Dict[Tuple[str, str, str], List[dict]] = {}
    for t in pre_tr:
        by_key.setdefault((t["sym"], t["tf"], t["setup"]), []).append(t)

    wf_oos_trades = []
    wf_fold_stats = []
    for tr0_s, te0_s, te1_s in wf_folds:
        tr0, te0, te1 = _ym_ms(tr0_s), _ym_ms(te0_s), _ym_ms(te1_s)
        sel_keys = set()
        for k, lst in by_key.items():
            sub = [x for x in lst if tr0 <= x["open_ts"] < te0]
            if len(sub) < 12:
                continue
            gp = sum(x["pnl_u"] for x in sub if x["pnl_u"] > 0)
            gl = sum(abs(x["pnl_u"]) for x in sub if x["pnl_u"] <= 0)
            pwr = gp / max(gp + gl, 1e-9) * 100.0
            pf = gp / max(gl, 1e-9)
            if pwr >= 54.0 and pf >= 1.18 and (gp - gl) >= 10.0:
                sel_keys.add(k)
        fold_tr = [x for k in sel_keys for x in by_key.get(k, []) if te0 <= x["open_ts"] < te1]
        fp = simulate_portfolio(fold_tr, 0, 400, 250, False, 3, True)
        wf_fold_stats.append({"test_window": f"{te0_s} -> {te1_s}", "n_selected": len(sel_keys), "trades": fp["trades"], "pf": fp["pf"], "roi_pct": fp["roi_pct"], "dd": fp["max_dd_pct"]})
        wf_oos_trades.extend(fold_tr)
        print(f"  WF Fold {te0_s}->{te1_s} | Sel={len(sel_keys):2d} | N={fp['trades']:3d} | PF={fp['pf']:4.2f} | ROI={fp['roi_pct']:+6.2f}% | DD={fp['max_dd_pct']:5.2f}%")

    wf_ema_port = simulate_portfolio(wf_oos_trades, 0, 400, 250, False, 3, True)
    print(f"  [WF EMA TOTAL 2023-2026] N={wf_ema_port['trades']} | PF={wf_ema_port['pf']} | PWR={wf_ema_port['pwr']}% | ROI={wf_ema_port['roi_pct']:+.2f}% | DD={wf_ema_port['max_dd_pct']}%")

    # 3. Protocol C: Benchmark Trend & Trailing Engines on EXACT same 6-year S4_REAL_FUNDING simulator!
    print("\n=== PROTOCOL C: ENGINE BENCHMARK UNDER IDENTICAL S4_REAL_FUNDING (6 YEARS 2021-2026) ===")
    engines_out = {}
    for eng_name, fn in [
        ("DONCHIAN_4H_100_6ATR_LONG", lambda s, e: run_donchian_4h_engine(market, funding_map, s, e, 100, 6.0, "S4_REAL_FUNDING")),
        ("EMA34_4H_5ATR_TRAIL_LONG", lambda s, e: run_ema_trail_4h_engine(market, funding_map, s, e, 5.0, True, "S4_REAL_FUNDING")),
        ("EMA34_4H_5ATR_TRAIL_BOTH", lambda s, e: run_ema_trail_4h_engine(market, funding_map, s, e, 5.0, False, "S4_REAL_FUNDING")),
    ]:
        engines_out[eng_name] = {}
        for w_name, w0, w1 in [
            ("FULL_6Y_2021_2026", t_2021, t_2026_11),
            ("Y1_Y3_2021_2023", t_2021, t_2023_11),
            ("Y4_Y6_2023_2026", t_2023_11, t_2026_11),
            ("BLIND_Y6_2025_2026", t_2025_11, t_2026_11),
        ]:
            tr = fn(w0, w1)
            # For trend-following trailing-stop engines, disable 2-SL pause (low WR ~40% is normal for trend following!)
            p400 = simulate_portfolio(tr, 0, 400, 250, False, 5, False)
            ptur = simulate_portfolio(tr, 0, 400, 250, True, 5, False)
            engines_out[eng_name][w_name] = {"bao_400g": p400, "turbo_250g": ptur}
            print(
                f"  {eng_name:26s} | {w_name:18s} | N={p400['trades']:3d} WR={p400['wr']:4.1f}% PF={p400['pf']:4.2f} "
                f"| 400G={p400['roi_pct']:+7.1f}% (DD {p400['max_dd_pct']:5.1f}%) | Turbo={ptur['roi_pct']:+7.1f}% (DD {ptur['max_dd_pct']:5.1f}%) | Fund=${p400['funding_usd']:,.0f}"
            )

    payload = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "period": "2021-01-01 -> 2026-10-31 (6 năm · 6 Coin · Funding 8h Binance thật · IOC ±0.25U)",
        "stress_ladder": stress_results,
        "wf_ema": {"folds": wf_fold_stats, "total": wf_ema_port},
        "engines": engines_out,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] Saved Gold-Standard 6-Year Audit to {OUT_JSON}")


if __name__ == "__main__":
    main()
