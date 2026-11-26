"""CHICKTRADE v3.3 — XAUUSD (Gold) Full 4-TF Quant Onboarding & 100x Leverage Study (2020-2024 Train + 2025 Blind OOS).

Downloads real 5m Gold (PAXGUSDT 1:1 physical gold) monthly archives from Binance Vision (2020-09 -> 2025-12),
resamples exact 5m/15m/30m/1h/4h bars, applies Forex Mon-Fri session filter + IC Markets Raw Spread XAUUSD fee model
(0.012% RT), sweeps SL/TP1/TP2/Leg2-SL sweet spots, audits all 48 setups (4 TFs x 12 EMA setups), runs 100x leverage
portfolio simulations (Bao 400G / 250G / 150G + S2 Compounding), and exports:
  - coins/XAUUSD/xauusd_study_results.json
  - coins/XAUUSD/audit_48.csv
  - core/reports/chicktrade_xauusd_100x_report.html
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import io
import json
import math
import os
import zipfile
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
XAU_DIR = os.path.join(ROOT, "coins", "XAUUSD")
DATA_DIR = os.path.join(XAU_DIR, "data")
CACHE_5M = os.path.join(DATA_DIR, "xauusd_5m_2020_2025.csv.gz")
OUT_JSON = os.path.join(XAU_DIR, "xauusd_study_results.json")
OUT_CSV = os.path.join(XAU_DIR, "audit_48.csv")
OUT_HTML = os.path.join(ROOT, "core", "reports", "chicktrade_xauusd_100x_report.html")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(os.path.dirname(OUT_HTML), exist_ok=True)

GAMMA_TF = {"5m": 0.17, "15m": 0.29, "30m": 0.41, "1h": 0.59}
TF_COOLDOWN = {"5m": 48, "15m": 16, "30m": 8, "1h": 4}
TF_PRIORITY = {"1h": 0, "30m": 1, "15m": 2, "5m": 3}

# IC Markets Raw Spread XAUUSD realistic cost:
# Spread ~$0.12 (~0.004%) + Commission $7/lot (~0.003%) + Slippage (~0.005%) = 0.012% RT (0.006% per side)
IC_SIDE_COST = 0.00006
MIN_U_PCT_GOLD = 0.0004  # 0.04% floor (tailored for Gold's lower macro volatility than Crypto)


def _download_month_5m(ym: Tuple[int, int]) -> pd.DataFrame:
    y, m = ym
    url = f"https://data.binance.vision/data/spot/monthly/klines/PAXGUSDT/5m/PAXGUSDT-5m-{y:04d}-{m:02d}.zip"
    try:
        r = requests.get(url, timeout=20)
        if r.status_code != 200:
            return pd.DataFrame()
        with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
            name = zf.namelist()[0]
            with zf.open(name) as f:
                df = pd.read_csv(
                    f,
                    header=None,
                    usecols=[0, 1, 2, 3, 4, 5, 6],
                    names=["open_time_ms", "open", "high", "low", "close", "volume", "close_time_ms"],
                )
        # Normalize microsecond timestamps (Binance Vision 2025+ uses microseconds on spot!)
        if len(df) > 0 and int(df["open_time_ms"].iloc[0]) > 10**14:
            df["open_time_ms"] = (df["open_time_ms"] // 1000).astype("int64")
            df["close_time_ms"] = (df["close_time_ms"] // 1000).astype("int64")
        return df
    except Exception as exc:
        print(f"[WARN] Failed {y}-{m:02d}: {exc}")
        return pd.DataFrame()


def load_or_download_5m() -> pd.DataFrame:
    if os.path.exists(CACHE_5M):
        print(f"[CACHE] Loading {CACHE_5M}...")
        df = pd.read_csv(CACHE_5M)
        df["timestamp"] = pd.to_datetime(df["open_time_ms"], unit="ms", utc=True)
        return df

    months: List[Tuple[int, int]] = []
    for y in range(2020, 2026):
        start_m = 9 if y == 2020 else 1
        for m in range(start_m, 13):
            months.append((y, m))

    print(f"[DL] Downloading {len(months)} monthly 5m archives (2020-09 -> 2025-12) in parallel...")
    with ThreadPoolExecutor(max_workers=12) as pool:
        dfs = list(pool.map(_download_month_5m, months))

    dfs = [d for d in dfs if len(d) > 0]
    full = pd.concat(dfs, ignore_index=True)
    full["open_time_ms"] = full["open_time_ms"].astype("int64")
    full["close_time_ms"] = full["close_time_ms"].astype("int64")
    for col in ("open", "high", "low", "close", "volume"):
        full[col] = full[col].astype("float64")
    full = full.drop_duplicates("open_time_ms").sort_values("open_time_ms").reset_index(drop=True)
    full.to_csv(CACHE_5M, index=False, compression="gzip")
    full["timestamp"] = pd.to_datetime(full["open_time_ms"], unit="ms", utc=True)
    print(f"[OK] Saved {len(full):,} 5m bars to {CACHE_5M}")
    return full


def resample_tf(df5: pd.DataFrame, rule: str) -> pd.DataFrame:
    if rule == "5min":
        return df5.copy().reset_index(drop=True)
    g = df5.set_index("timestamp").resample(rule).agg({
        "open_time_ms": "first",
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
        "close_time_ms": "last",
    }).dropna().reset_index()
    g["open_time_ms"] = g["open_time_ms"].astype("int64")
    g["close_time_ms"] = g["close_time_ms"].astype("int64")
    return g


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


def compute_storm_cooldown(df: pd.DataFrame, cooldown_bars: int) -> np.ndarray:
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
    return (raw.astype(int).rolling(cooldown_bars, min_periods=1).max() > 0).values


def forex_open_mask(ts_series: pd.Series) -> np.ndarray:
    wd = ts_series.dt.weekday.values
    hr = ts_series.dt.hour.values
    closed = (wd == 5) | ((wd == 4) & (hr >= 22)) | ((wd == 6) & (hr < 22))
    return ~closed


def extract_signals_for_tf(
    df: pd.DataFrame,
    hdf: pd.DataFrame,
    df4h: pd.DataFrame,
    tf: str,
) -> List[dict]:
    """Extract all raw CHICKTRADE v3.3 signals for a TF with zero-lookahead 4H U3_HTF_MACRO_ATR."""
    closes = df["close"].values
    opens = df["open"].values
    highs = df["high"].values
    lows = df["low"].values
    ts_ms = df["open_time_ms"].values
    close_ms = df["close_time_ms"].values
    fx_open = forex_open_mask(df["timestamp"])

    ema34 = df["close"].ewm(span=34, adjust=False).mean().values
    ema89 = df["close"].ewm(span=89, adjust=False).mean().values
    ema200 = df["close"].ewm(span=200, adjust=False).mean().values
    atr = compute_atr14(df).values
    adx = compute_adx14(df)
    local_cd = compute_storm_cooldown(df, TF_COOLDOWN[tf])

    htf_e34 = hdf["close"].ewm(span=34, adjust=False).mean()
    htf_e200 = hdf["close"].ewm(span=200, adjust=False).mean()
    htf_tr = np.where((hdf["close"] > htf_e200) & (htf_e34 > htf_e200), 1, np.where((hdf["close"] < htf_e200) & (htf_e34 < htf_e200), -1, 0))
    htf_close_ts = hdf["close_time_ms"].values

    atr4 = compute_atr14(df4h)
    macro_4h = ((0.5 * atr4) / df4h["close"]).ewm(span=20, adjust=False).mean().values
    h4_close_ts = df4h["close_time_ms"].values
    gamma = GAMMA_TF[tf]

    signals: List[dict] = []
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
            if wave_count > 4 or not fx_open[i] or adx[i] < 20.0 or local_cd[i]:
                continue

            h_idx = int(np.searchsorted(htf_close_ts, close_ms[i], side="right")) - 1
            if h_idx < 0:
                continue
            ht = int(htf_tr[h_idx])
            if wave_count <= 3:
                if ht != trend:
                    continue
                d = trend
                stype = f"EMA{ema_p}_W{wave_count}"
            else:
                d = -trend
                stype = f"EMA{ema_p}_W4_REV"

            h4_idx = int(np.searchsorted(h4_close_ts, close_ms[i], side="right")) - 1
            if h4_idx < 0:
                continue
            u_frac = max(MIN_U_PCT_GOLD, float(macro_4h[h4_idx]) * gamma)
            signals.append({
                "idx": i,
                "tf": tf,
                "setup": stype,
                "dir": d,
                "open_ts": int(ts_ms[i]),
                "entry": float(c),
                "u_frac": float(u_frac),
                "u_px": float(c * u_frac),
            })

    return signals


def evaluate_signals(
    df: pd.DataFrame,
    signals: List[dict],
    sl_u: float,
    tp1_u: float,
    tp2_u: float,
    exit_mode: str = "PARTIAL_BE",  # PARTIAL_BE | PARTIAL_M1U | ONLY_TP1
    side_cost: float = IC_SIDE_COST,
) -> List[dict]:
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    ts_ms = df["open_time_ms"].values
    n_bars = len(df)

    # Group non-overlapping per (tf, setup)
    busy_by_setup: Dict[str, int] = {}
    out: List[dict] = []

    for s in signals:
        st = s["setup"]
        i = s["idx"]
        if i <= busy_by_setup.get(st, -1):
            continue

        d = s["dir"]
        entry = s["entry"]
        u_px = s["u_px"]
        u_frac = s["u_frac"]
        cost_u = 2.0 * side_cost / u_frac

        sl_px = entry - d * sl_u * u_px
        tp1_px = entry + d * tp1_u * u_px
        tp2_px = entry + d * tp2_u * u_px
        if exit_mode == "PARTIAL_BE":
            l2_sl_px = entry * (1.0 + d * 2.0 * side_cost)
            l2_sl_u = 0.0
        elif exit_mode == "PARTIAL_M1U":
            l2_sl_px = entry - d * 1.0 * u_px
            l2_sl_u = -1.0
        else:
            l2_sl_px = sl_px
            l2_sl_u = -sl_u

        tp1_hit = False
        active_sl = sl_px
        pnl_u = 0.0
        outcome = "TIMEOUT"
        max_j = min(n_bars, i + 201)
        exit_idx = max_j - 1

        for j in range(i + 1, max_j):
            bh, bl = highs[j], lows[j]
            if exit_mode == "ONLY_TP1":
                if (d == 1 and bl <= sl_px) or (d == -1 and bh >= sl_px):
                    pnl_u = -sl_u - cost_u
                    outcome = "SL"
                    exit_idx = j
                    break
                if (d == 1 and bh >= tp1_px) or (d == -1 and bl <= tp1_px):
                    pnl_u = tp1_u - cost_u
                    outcome = "TP1_FULL"
                    exit_idx = j
                    break
            else:
                if not tp1_hit:
                    if (d == 1 and bl <= active_sl) or (d == -1 and bh >= active_sl):
                        pnl_u = -sl_u - cost_u
                        outcome = "SL"
                        exit_idx = j
                        break
                    if (d == 1 and bh >= tp1_px) or (d == -1 and bl <= tp1_px):
                        tp1_hit = True
                        pnl_u += 0.5 * (tp1_u - cost_u)
                        active_sl = l2_sl_px
                        if (d == 1 and bh >= tp2_px) or (d == -1 and bl <= tp2_px):
                            pnl_u += 0.5 * (tp2_u - cost_u)
                            outcome = "TP2"
                            exit_idx = j
                            break
                else:
                    if (d == 1 and bl <= active_sl) or (d == -1 and bh >= active_sl):
                        pnl_u += 0.5 * (l2_sl_u - cost_u)
                        outcome = "LEG2_SL"
                        exit_idx = j
                        break
                    if (d == 1 and bh >= tp2_px) or (d == -1 and bl <= tp2_px):
                        pnl_u += 0.5 * (tp2_u - cost_u)
                        outcome = "TP2"
                        exit_idx = j
                        break
        else:
            rem_u = d * (closes[exit_idx] - entry) / u_px - cost_u
            pnl_u += (0.5 if tp1_hit else 1.0) * rem_u

        busy_by_setup[st] = exit_idx
        out.append({
            "tf": s["tf"],
            "setup": st,
            "dir": "LONG" if d == 1 else "SHORT",
            "open_ts": s["open_ts"],
            "close_ts": int(ts_ms[exit_idx]),
            "entry": entry,
            "u_pct": u_frac * 100.0,
            "sl_u": sl_u,
            "tp1_u": tp1_u,
            "tp2_u": tp2_u,
            "exit_mode": exit_mode,
            "outcome": outcome,
            "pnl_u": round(float(pnl_u), 4),
        })
    return out


def calc_metrics(trades: List[dict]) -> dict:
    n = len(trades)
    if n == 0:
        return {"n": 0, "wins": 0, "wr": 0.0, "pwr": 0.0, "pf": 0.0, "net_u": 0.0, "avg_w": 0.0, "avg_l": 0.0}
    wins = [t["pnl_u"] for t in trades if t["pnl_u"] > 0]
    losses = [abs(t["pnl_u"]) for t in trades if t["pnl_u"] <= 0]
    gp = sum(wins)
    gl = sum(losses)
    return {
        "n": n,
        "wins": len(wins),
        "wr": round(len(wins) / n * 100.0, 1),
        "pwr": round(gp / max(gp + gl, 1e-9) * 100.0, 1),
        "pf": round(gp / max(gl, 1e-9), 2) if gl > 1e-9 else (9.99 if gp > 0 else 0.0),
        "net_u": round(gp - gl, 2),
        "avg_w": round(gp / max(len(wins), 1), 2),
        "avg_l": round(gl / max(len(losses), 1), 2),
    }


def simulate_portfolio_100x(
    trades: List[dict],
    buffer_units: float,
    leverage: float = 100.0,
    init_cap: float = 10_000.0,
    use_2sl_brake: bool = True,
) -> dict:
    """Simulate realistic 100x IC Markets XAUUSD account with S2 Unlimited Compounding (+8%/-10%),
    margin pre-check, lot size tracking, and 2-SL Shadow-Win brake."""
    ordered = sorted(trades, key=lambda x: (x["open_ts"], TF_PRIORITY[x["tf"]]))
    eq = init_cap
    peak = init_cap
    max_dd = 0.0
    step = 0
    tier_cap = init_cap
    consec_sl = 0
    shadow_paused = False
    busy_ts = 0

    taken = []
    curve = []
    max_margin_pct = 0.0
    avg_lots = []
    avg_sl_pct_eq = []
    liq_events = 0

    for t in ordered:
        if t["open_ts"] < busy_ts:
            continue
        if shadow_paused:
            if t["pnl_u"] > 0:
                shadow_paused = False
                consec_sl = 0
            continue

        u_usd = tier_cap / buffer_units
        u_frac = t["u_pct"] / 100.0
        notional = u_usd / max(u_frac, 1e-6)
        # IC Markets XAUUSD institutional liquidity cap: $5,000,000 (~14 lots at $3500)
        notional = min(notional, 5_000_000.0)
        eff_u_usd = notional * u_frac
        req_margin = notional / leverage
        margin_pct = req_margin / max(eq, 1.0) * 100.0
        if margin_pct > 75.0:
            # Cap notional to 75% initial margin
            notional = eq * 0.75 * leverage
            eff_u_usd = notional * u_frac
            req_margin = notional / leverage
            margin_pct = 75.0

        max_margin_pct = max(max_margin_pct, margin_pct)
        lots = notional / (t["entry"] * 100.0)  # 1 Standard Lot = 100 oz XAUUSD
        avg_lots.append(lots)
        sl_risk_pct = (eff_u_usd * t["sl_u"]) / max(eq, 1.0) * 100.0
        avg_sl_pct_eq.append(sl_risk_pct)

        pnl_usd = t["pnl_u"] * eff_u_usd
        eq += pnl_usd
        if eq <= 0:
            eq = 0.0
            liq_events += 1
            break

        peak = max(peak, eq)
        dd = (eq - peak) / peak * 100.0
        max_dd = min(max_dd, dd)

        # S2 Unlimited Compounding (+8% up, -10% hysteresis down)
        while eq >= init_cap * (1.08 ** (step + 1)):
            step += 1
            tier_cap = init_cap * (1.08 ** step)
        while step > 0 and eq < tier_cap * 0.90:
            step -= 1
            tier_cap = init_cap * (1.08 ** step)

        if use_2sl_brake:
            if t["outcome"] == "SL":
                consec_sl += 1
                if consec_sl >= 2:
                    shadow_paused = True
                    consec_sl = 0
            elif t["pnl_u"] > 0:
                consec_sl = 0

        busy_ts = t["close_ts"]
        dt_str = datetime.fromtimestamp(t["close_ts"] / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d")
        taken.append({**t, "pnl_usd": round(pnl_usd, 2), "eq": round(eq, 2), "lots": round(lots, 3)})
        curve.append({"dt": dt_str, "eq": round(eq, 2), "dd": round(dd, 2)})

    m = calc_metrics(taken)
    roi = (eq / init_cap - 1.0) * 100.0
    calmar = abs(roi / max_dd) if max_dd < -0.01 else 0.0
    return {
        "buffer": int(buffer_units),
        "leverage": int(leverage),
        "trades": m["n"],
        "wr": m["wr"],
        "pwr": m["pwr"],
        "pf": m["pf"],
        "net_u": m["net_u"],
        "final_eq": round(eq, 2),
        "roi_pct": round(roi, 2),
        "max_dd_pct": round(max_dd, 2),
        "calmar": round(calmar, 2),
        "max_step": step,
        "max_margin_pct": round(max_margin_pct, 1),
        "avg_lots": round(float(np.mean(avg_lots)) if avg_lots else 0.0, 2),
        "avg_sl_risk_pct": round(float(np.mean(avg_sl_pct_eq)) if avg_sl_pct_eq else 0.0, 2),
        "liq_events": liq_events,
        "curve": curve,
    }


def main():
    df5 = load_or_download_5m()
    tfs_data = {
        "5m": df5,
        "15m": resample_tf(df5, "15min"),
        "30m": resample_tf(df5, "30min"),
        "1h": resample_tf(df5, "1h"),
        "4h": resample_tf(df5, "4h"),
    }
    for k, v in tfs_data.items():
        print(f"  {k:3s}: {len(v):7,d} bars | {v['timestamp'].iloc[0].strftime('%Y-%m-%d')} -> {v['timestamp'].iloc[-1].strftime('%Y-%m-%d')}")

    train_start_ms = int(datetime(2020, 10, 1, tzinfo=timezone.utc).timestamp() * 1000)
    train_end_ms = int(datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    oos_end_ms = int(datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)

    # 4H Macro ATR stats
    df4 = tfs_data["4h"]
    macro4 = ((0.5 * compute_atr14(df4)) / df4["close"]).ewm(span=20, adjust=False).mean()
    m4_train = macro4[(df4["open_time_ms"] >= train_start_ms) & (df4["open_time_ms"] < train_end_ms)]
    u_stats = {}
    for tf in ("5m", "15m", "30m", "1h"):
        u_avg = float(m4_train.mean() * GAMMA_TF[tf] * 100.0)
        u_min = float(max(MIN_U_PCT_GOLD, m4_train.min() * GAMMA_TF[tf]) * 100.0)
        u_max = float(m4_train.max() * GAMMA_TF[tf] * 100.0)
        u_stats[tf] = {"avg_pct": round(u_avg, 4), "min_pct": round(u_min, 4), "max_pct": round(u_max, 4)}
    print("\n1U Stats (2020-2024):", u_stats)

    # Extract raw signals once per TF
    raw_sigs: Dict[str, List[dict]] = {}
    for tf in ("5m", "15m", "30m", "1h"):
        htf = "1h" if tf in ("5m", "15m") else "4h"
        raw_sigs[tf] = extract_signals_for_tf(tfs_data[tf], tfs_data[htf], tfs_data["4h"], tf)
        n_tr = sum(1 for s in raw_sigs[tf] if train_start_ms <= s["open_ts"] < train_end_ms)
        n_oo = sum(1 for s in raw_sigs[tf] if train_end_ms <= s["open_ts"] < oos_end_ms)
        print(f"  Signals {tf:3s}: Train(2020-2024)={n_tr:4d} | Blind OOS(2025)={n_oo:3d}")

    # Step A: Sweet-Spot Grid Search on 2020-2024 per TF
    print("\n=== SWEET-SPOT GRID SEARCH (2020-2024 Train) ===")
    grid_results: Dict[str, List[dict]] = {}
    best_cfg_by_tf: Dict[str, dict] = {}
    sl_grid = [4.0, 5.0, 6.0]
    tp1_grid = [3.0, 3.5, 4.0]
    tp2_grid = [8.0, 10.0, 12.0]
    modes = ["PARTIAL_BE", "PARTIAL_M1U", "ONLY_TP1"]

    for tf in ("5m", "15m", "30m", "1h"):
        df = tfs_data[tf]
        sigs = raw_sigs[tf]
        tf_rows = []
        for sl_u in sl_grid:
            for tp1_u in tp1_grid:
                for mode in modes:
                    tp2_list = [tp1_u] if mode == "ONLY_TP1" else tp2_grid
                    for tp2_u in tp2_list:
                        ev = evaluate_signals(df, sigs, sl_u, tp1_u, tp2_u, mode)
                        tr_only = [t for t in ev if train_start_ms <= t["open_ts"] < train_end_ms]
                        m = calc_metrics(tr_only)
                        score = m["net_u"] * (m["pwr"] / 50.0) if m["n"] >= 30 else -9999.0
                        tf_rows.append({
                            "tf": tf, "sl_u": sl_u, "tp1_u": tp1_u, "tp2_u": tp2_u, "mode": mode,
                            **m, "score": round(score, 2),
                        })
        tf_rows.sort(key=lambda x: -x["score"])
        grid_results[tf] = tf_rows[:12]
        best = tf_rows[0]
        best_cfg_by_tf[tf] = best
        print(
            f"  BEST {tf:3s}: SL={best['sl_u']}U TP1={best['tp1_u']}U TP2={best['tp2_u']}U ({best['mode']}) "
            f"| N={best['n']} WR={best['wr']}% ProfitWR={best['pwr']}% PF={best['pf']} Net={best['net_u']:+.1f}U"
        )

    # Step B: Full 48-Setup Audit using each TF's Sweet-Spot config (Train 2020-2024 + Blind 2025 OOS)
    print("\n=== FULL 48-SETUP AUDIT (2020-2024 Train -> 2025 Blind OOS) ===")
    audit_rows: List[dict] = []
    all_trades_train: List[dict] = []
    all_trades_oos: List[dict] = []
    all_trades_full: List[dict] = []

    for tf in ("5m", "15m", "30m", "1h"):
        cfg = best_cfg_by_tf[tf]
        ev = evaluate_signals(tfs_data[tf], raw_sigs[tf], cfg["sl_u"], cfg["tp1_u"], cfg["tp2_u"], cfg["mode"])
        by_setup: Dict[str, List[dict]] = {}
        for t in ev:
            by_setup.setdefault(t["setup"], []).append(t)

        all_setups = [f"EMA{p}_W{w}" for p in (34, 89, 200) for w in (1, 2, 3)] + [f"EMA{p}_W4_REV" for p in (34, 89, 200)]
        for st in all_setups:
            lst = by_setup.get(st, [])
            tr_l = [x for x in lst if train_start_ms <= x["open_ts"] < train_end_ms]
            oo_l = [x for x in lst if train_end_ms <= x["open_ts"] < oos_end_ms]
            mt = calc_metrics(tr_l)
            mo = calc_metrics(oo_l)

            # Strict CHICKTRADE grading criteria on 2020-2024 Train ONLY (zero OOS leakage!)
            if mt["n"] >= 18 and mt["pwr"] >= 56.0 and mt["pf"] >= 1.28 and mt["net_u"] >= 20.0:
                status = "CORE"
                vol_mult = 1.0
            elif mt["n"] >= 10 and mt["pwr"] >= 53.0 and mt["pf"] >= 1.13 and mt["net_u"] >= 8.0:
                status = "SCOUT"
                vol_mult = 0.5
            else:
                status = "DROP"
                vol_mult = 0.0

            score = (24.0 if status == "CORE" else (8.0 if status == "SCOUT" else -50.0)) + (mt["net_u"] / 60.0) * 45.0 + (mt["pwr"] / 100.0) * 40.0 + min(mt["pf"], 3.5) * 3.0
            audit_rows.append({
                "tf": tf,
                "setup": st,
                "sl_u": cfg["sl_u"],
                "tp1_u": cfg["tp1_u"],
                "tp2_u": cfg["tp2_u"],
                "mode": cfg["mode"],
                "status": status,
                "vol_mult": vol_mult,
                "score": round(score, 2),
                "train_n": mt["n"],
                "train_wr": mt["wr"],
                "train_pwr": mt["pwr"],
                "train_pf": mt["pf"],
                "train_net_u": mt["net_u"],
                "oos_n": mo["n"],
                "oos_wr": mo["wr"],
                "oos_pwr": mo["pwr"],
                "oos_pf": mo["pf"],
                "oos_net_u": mo["net_u"],
            })
            if status in ("CORE", "SCOUT"):
                for x in tr_l:
                    all_trades_train.append({**x, "status": status, "vol_mult": vol_mult})
                for x in oo_l:
                    all_trades_oos.append({**x, "status": status, "vol_mult": vol_mult})
                for x in tr_l + oo_l:
                    all_trades_full.append({**x, "status": status, "vol_mult": vol_mult})

    # Mark Top 8 CORE/SCOUT setups as THẤY LÀ QUẤT
    active_sorted = sorted([r for r in audit_rows if r["status"] in ("CORE", "SCOUT")], key=lambda x: -x["score"])
    must_set = {(r["tf"], r["setup"]) for r in active_sorted[:8]}
    for r in audit_rows:
        r["is_must"] = (r["tf"], r["setup"]) in must_set

    audit_rows.sort(key=lambda x: (-1 if x["status"] == "CORE" else (0 if x["status"] == "SCOUT" else 1), -x["score"]))
    pd.DataFrame(audit_rows).to_csv(OUT_CSV, index=False)

    n_core = sum(1 for r in audit_rows if r["status"] == "CORE")
    n_scout = sum(1 for r in audit_rows if r["status"] == "SCOUT")
    print(f"  Qualified Setups: CORE={n_core}, SCOUT={n_scout}, TOTAL={n_core + n_scout} / 48")
    for r in active_sorted[:12]:
        tag = "★" if r["is_must"] else " "
        print(
            f"  {tag} {r['status']:5s} {r['tf']:3s} {r['setup']:13s} | "
            f"2020-24: N={r['train_n']:3d} WR={r['train_wr']:5.1f}% PWR={r['train_pwr']:5.1f}% PF={r['train_pf']:4.2f} Net={r['train_net_u']:+6.1f}U | "
            f"2025 OOS: N={r['oos_n']:2d} WR={r['oos_wr']:5.1f}% Net={r['oos_net_u']:+5.1f}U"
        )

    # Step C: 100x Leverage Portfolio Sims (Train 2020-2024, Blind OOS 2025, Full 2020-2025)
    print("\n=== 100x LEVERAGE PORTFOLIO SIMULATIONS ($10,000, IC Markets Raw Spread XAUUSD) ===")
    sims_train = {}
    sims_oos = {}
    sims_full = {}
    for buf in (400, 250, 150):
        sims_train[str(buf)] = simulate_portfolio_100x(all_trades_train, buffer_units=buf, leverage=100.0)
        sims_oos[str(buf)] = simulate_portfolio_100x(all_trades_oos, buffer_units=buf, leverage=100.0)
        sims_full[str(buf)] = simulate_portfolio_100x(all_trades_full, buffer_units=buf, leverage=100.0)
        st = sims_train[str(buf)]
        so = sims_oos[str(buf)]
        print(
            f"  Bao {buf}G @ 100x | 2020-24: ${st['final_eq']:>10,.2f} ({st['roi_pct']:+6.1f}%) DD={st['max_dd_pct']:5.2f}% "
            f"Calmar={st['calmar']:5.1f} MaxMargin={st['max_margin_pct']:4.1f}% AvgLot={st['avg_lots']:.2f} "
            f"| 2025 Blind OOS: {so['roi_pct']:+6.2f}% DD={so['max_dd_pct']:5.2f}% ({so['trades']} lệnh, WR {so['wr']}%, PWR {so['pwr']}%)"
        )

    # Yearly breakdown for Bao 400G & Bao 250G on Full 2020-2025
    yearly = {}
    for yr in range(2020, 2026):
        y0 = int(datetime(yr, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
        y1 = int(datetime(yr + 1, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
        yt = [t for t in all_trades_full if y0 <= t["open_ts"] < y1]
        s400 = simulate_portfolio_100x(yt, buffer_units=400, leverage=100.0)
        s250 = simulate_portfolio_100x(yt, buffer_units=250, leverage=100.0)
        yearly[str(yr)] = {
            "trades": s400["trades"],
            "wr": s400["wr"],
            "pwr": s400["pwr"],
            "pf": s400["pf"],
            "net_u": s400["net_u"],
            "roi_400g": s400["roi_pct"],
            "dd_400g": s400["max_dd_pct"],
            "roi_250g": s250["roi_pct"],
            "dd_250g": s250["max_dd_pct"],
        }

    # Build summary payload
    payload = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "symbol": "XAUUSD",
        "data_source": "PAXGUSDT 5m (1:1 Physical Gold · Forex Mon-Fri Session · IC Markets Raw Spread 0.012% RT)",
        "train_period": "2020-10-01 -> 2024-12-31 (4.25 năm)",
        "oos_period": "2025-01-01 -> 2025-12-31 (1 năm Blind Out-of-Sample)",
        "leverage": 100,
        "u_stats": u_stats,
        "best_cfg_by_tf": best_cfg_by_tf,
        "grid_top_by_tf": grid_results,
        "audit_48": audit_rows,
        "sims_train": sims_train,
        "sims_oos": sims_oos,
        "sims_full": sims_full,
        "yearly": yearly,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] Saved results to {OUT_JSON} & {OUT_CSV}")


if __name__ == "__main__":
    main()
