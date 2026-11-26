"""CHICKTRADE v3.0 Cloud Paper Trading Engine ($0/month GitHub Actions + Pages).

Runs headless every 15 minutes (or `--once` locally/CI), fetches live Binance USD-M
Futures klines & exchangeInfo, computes zero-lookahead U3_HTF_MACRO_ATR rates with
4H-cycle deltas, manages open positions, scans the 58 v3.0 setups, enforces S2 (50-step,
-10% hysteresis, $300k/$150k liquidity cap), Bao 400G / optional Auto-Turbo 250G,
PORTFOLIO 2-SL -> Shadow-Win brake, 20x Cross margin pre-check, and writes persistent
dashboard state to `e:\CHICKTRADE\docs\data\live_dashboard.json`.

Does NOT modify `bot/` or `core/`.
"""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

CLOUD_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CLOUD_DIR)
RULES_PATH = os.path.join(ROOT, "core", "core_strategy_rules.json")
OUT_JSON = os.path.join(ROOT, "docs", "data", "live_dashboard.json")
OUT_CHART_JSON = os.path.join(ROOT, "docs", "data", "chart_candles.json")

SIX_COINS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "LINKUSDT", "XRPUSDT", "SOLUSDT", "NEARUSDT"]
FAPI_BASES = [
    "https://fapi.binance.com",
    "https://fapi1.binance.com",
    "https://fapi2.binance.com",
    "https://fapi3.binance.com",
]

MAKER_FEE = 0.0002
TAKER_FEE = 0.0005
SLIP_BPS = 0.0001  # 1 bps = 0.01%

FALLBACK_EXCHANGE_FILTERS = {
    "BTCUSDT": {"stepSize": 0.001, "tickSize": 0.1, "minNotional": 100.0},
    "ETHUSDT": {"stepSize": 0.001, "tickSize": 0.01, "minNotional": 20.0},
    "BNBUSDT": {"stepSize": 0.01, "tickSize": 0.01, "minNotional": 5.0},
    "LINKUSDT": {"stepSize": 0.01, "tickSize": 0.001, "minNotional": 5.0},
    "XRPUSDT": {"stepSize": 0.1, "tickSize": 0.0001, "minNotional": 5.0},
    "SOLUSDT": {"stepSize": 0.01, "tickSize": 0.01, "minNotional": 5.0},
    "NEARUSDT": {"stepSize": 1.0, "tickSize": 0.001, "minNotional": 5.0},
}

TF_COOLDOWN = {"15m": 16, "30m": 8, "1h": 4}
TF_PRIORITY = {"1h": 0, "30m": 1, "15m": 2, "5m": 3}
TF_MS = {"15m": 900_000, "30m": 1_800_000, "1h": 3_600_000, "4h": 14_400_000}


def _fapi_get(path: str, params: Optional[dict] = None, timeout: int = 12) -> Any:
    """Query Binance Futures public REST API across mirror bases with clean error handling."""
    last_err = None
    for base in FAPI_BASES:
        try:
            r = requests.get(f"{base}{path}", params=params, timeout=timeout)
            if r.status_code == 200:
                return r.json()
            last_err = RuntimeError(f"HTTP {r.status_code} from {base}{path}: {r.text[:160]}")
        except Exception as exc:
            last_err = exc
    raise RuntimeError(f"All Binance FAPI mirrors failed for {path}: {last_err}")


def fetch_exchange_filters() -> Dict[str, Dict[str, float]]:
    """Fetch stepSize, tickSize, minNotional from /fapi/v1/exchangeInfo with safe fallback."""
    filters = {k: dict(v) for k, v in FALLBACK_EXCHANGE_FILTERS.items()}
    try:
        data = _fapi_get("/fapi/v1/exchangeInfo", timeout=10)
        for s_info in data.get("symbols", []):
            sym = s_info.get("symbol")
            if sym not in filters:
                continue
            for flt in s_info.get("filters", []):
                ftype = flt.get("filterType")
                if ftype == "LOT_SIZE":
                    filters[sym]["stepSize"] = float(flt.get("stepSize", filters[sym]["stepSize"]))
                elif ftype == "PRICE_FILTER":
                    filters[sym]["tickSize"] = float(flt.get("tickSize", filters[sym]["tickSize"]))
                elif ftype in ("MIN_NOTIONAL", "NOTIONAL"):
                    mn = flt.get("notional") or flt.get("minNotional")
                    if mn is not None:
                        filters[sym]["minNotional"] = float(mn)
    except Exception as exc:
        print(f"[WARN] exchangeInfo fallback used: {exc}")
    return filters


def floor_step(val: float, step: float) -> float:
    """Round down strictly to exchange stepSize / tickSize."""
    if step <= 0:
        return val
    prec = max(0, int(round(-math.log10(step))))
    floored = math.floor(val / step + 1e-9) * step
    return round(floored, prec)


def _fetch_klines_with_fallback(sym: str, interval: str, limit: int) -> List[List[Any]]:
    """Fetch klines with resilient fallback for US cloud runners (Binance FAPI -> Binance Vision -> OKX -> Gate)."""
    hdrs = {"User-Agent": "curl/8.4.0", "Accept": "application/json"}
    errs: List[str] = []

    # 1. Binance Futures mirrors (works locally / non-US)
    for base in FAPI_BASES:
        try:
            r = requests.get(f"{base}/fapi/v1/klines", params={"symbol": sym, "interval": interval, "limit": min(limit, 1500)}, headers=hdrs, timeout=10)
            if r.status_code == 200:
                arr = r.json()
                if isinstance(arr, list) and len(arr) >= 50:
                    return arr
            errs.append(f"BF:{r.status_code}")
        except Exception as exc:
            errs.append(f"BF:err")

    # 2. Binance Vision Global CDN (never geo-blocked on GitHub Actions US runners, up to 1000/call -> paginate for 1200)
    try:
        step_ms = TF_MS.get(interval, 900_000)
        r1 = requests.get(
            "https://data-api.binance.vision/api/v3/klines",
            params={"symbol": sym, "interval": interval, "limit": min(limit, 1000)},
            headers=hdrs,
            timeout=12,
        )
        if r1.status_code == 200:
            arr1 = r1.json()
            if isinstance(arr1, list) and len(arr1) >= 50:
                if limit > 1000 and len(arr1) == 1000:
                    extra = min(limit - 1000, 1000)
                    end_t = int(arr1[0][0]) - 1
                    r0 = requests.get(
                        "https://data-api.binance.vision/api/v3/klines",
                        params={"symbol": sym, "interval": interval, "endTime": end_t, "limit": extra},
                        headers=hdrs,
                        timeout=12,
                    )
                    if r0.status_code == 200 and isinstance(r0.json(), list):
                        arr1 = r0.json() + arr1
                return arr1
        errs.append(f"BV:{r1.status_code}")
    except Exception:
        errs.append("BV:err")

    # 3. OKX Perpetual Swap API (300/call, paginate up to 3 pages)
    base_coin = sym.replace("USDT", "")
    try:
        okx_bar = {"15m": "15m", "30m": "30m", "1h": "1H", "4h": "4H"}.get(interval, "1H")
        step_ms = TF_MS.get(interval, 3_600_000)
        collected: List[List[Any]] = []
        after_ts: Optional[str] = None
        for _ in range(4):
            params: Dict[str, Any] = {"instId": f"{base_coin}-USDT-SWAP", "bar": okx_bar, "limit": 300}
            if after_ts:
                params["after"] = after_ts
            ro = requests.get("https://www.okx.com/api/v5/market/candles", params=params, headers=hdrs, timeout=10)
            if ro.status_code != 200:
                errs.append(f"OKX:{ro.status_code}")
                break
            rows = ro.json().get("data", [])
            if not rows:
                break
            collected.extend(rows)
            after_ts = str(rows[-1][0])
            if len(collected) >= limit:
                break
        if len(collected) >= 50:
            collected.reverse()
            return [
                [int(c[0]), c[1], c[2], c[3], c[4], c[5], int(c[0]) + step_ms - 1]
                for c in collected[-limit:]
            ]
    except Exception:
        errs.append("OKX:err")

    # 4. Gate.io Futures API
    try:
        step_ms = TF_MS.get(interval, 3_600_000)
        rg = requests.get(
            "https://api.gateio.ws/api/v4/futures/usdt/candlesticks",
            params={"contract": f"{base_coin}_USDT", "interval": interval, "limit": min(limit, 1000)},
            headers=hdrs,
            timeout=10,
        )
        if rg.status_code == 200:
            arr = rg.json()
            if isinstance(arr, list) and len(arr) >= 50:
                return [
                    [int(c["t"]) * 1000, str(c["o"]), str(c["h"]), str(c["l"]), str(c["c"]), str(c.get("v", 0)), int(c["t"]) * 1000 + step_ms - 1]
                    for c in arr
                ]
        errs.append(f"GT:{rg.status_code}")
    except Exception:
        errs.append("GT:err")

    raise RuntimeError(f"All market data sources failed for {sym} {interval}: {', '.join(errs)}")


def fetch_symbol_klines(sym: str, interval: str, limit: int = 1200) -> pd.DataFrame:
    """Fetch USD-M Futures klines (with resilient CDN/exchange fallback) and return OHLCV DataFrame."""
    raw = _fetch_klines_with_fallback(sym, interval, limit)
    rows = []
    for k in raw:
        rows.append({
            "open_time_ms": int(k[0]),
            "timestamp": pd.to_datetime(int(k[0]), unit="ms", utc=True),
            "open": float(k[1]),
            "high": float(k[2]),
            "low": float(k[3]),
            "close": float(k[4]),
            "volume": float(k[5]),
            "close_time_ms": int(k[6]),
        })
    df = pd.DataFrame(rows).drop_duplicates("open_time_ms").sort_values("open_time_ms").reset_index(drop=True)
    return df


def fetch_all_market_data(now_ms: int) -> Tuple[Dict[str, Dict[str, pd.DataFrame]], Dict[str, float]]:
    """Parallel-fetch 1200 bars (15m/30m/1h) & 600 bars (4h) for converged EMA200 & stable wave counts."""
    tasks = [(sym, tf, 600 if tf == "4h" else 1200) for sym in SIX_COINS for tf in ("15m", "30m", "1h", "4h")]
    closed_data: Dict[str, Dict[str, pd.DataFrame]] = {sym: {} for sym in SIX_COINS}
    live_prices: Dict[str, float] = {}

    def _run(task):
        sym, tf, lim = task
        return sym, tf, fetch_symbol_klines(sym, tf, lim)

    with ThreadPoolExecutor(max_workers=6) as pool:
        for sym, tf, df in pool.map(_run, tasks):
            if tf == "15m" and len(df) > 0:
                live_prices[sym] = float(df["close"].iloc[-1])
            closed_df = df[df["close_time_ms"] <= now_ms].reset_index(drop=True)
            closed_data[sym][tf] = closed_df

    return closed_data, live_prices


def compute_atr14(df: pd.DataFrame) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat(
        [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / 14, adjust=False).mean()


def compute_adx14(df: pd.DataFrame) -> np.ndarray:
    high, low, close = df["high"], df["low"], df["close"]
    plus_dm = high.diff().clip(lower=0.0)
    minus_dm = (-low.diff()).clip(lower=0.0)
    plus_dm = plus_dm.where(plus_dm > minus_dm, 0.0)
    minus_dm = minus_dm.where(minus_dm > plus_dm, 0.0)
    atr = compute_atr14(df)
    plus_di = 100.0 * plus_dm.ewm(alpha=1.0 / 14, adjust=False).mean() / atr.replace(0, np.nan)
    minus_di = 100.0 * minus_dm.ewm(alpha=1.0 / 14, adjust=False).mean() / atr.replace(0, np.nan)
    dx = (100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)).fillna(0.0)
    return dx.ewm(alpha=1.0 / 14, adjust=False).mean().values


def compute_storm_cooldown(df: pd.DataFrame, cooldown_bars: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (cooldown_mask, atr_ratio) matching v6.1 3-Tier Storm Guard T1 (4h / 12h / 24h)."""
    b_light = max(1, int(cooldown_bars))
    b_med = max(1, int(cooldown_bars) * 3)
    b_heavy = max(1, int(cooldown_bars) * 6)
    high, low, close, open_ = df["high"], df["low"], df["close"], df["open"]
    atr = compute_atr14(df)
    atr_ma50 = atr.rolling(50, min_periods=10).mean()
    atr_ratio = (atr / atr_ma50.replace(0, np.nan)).fillna(1.0)
    crange = (high - low).replace(0, np.nan)
    body = (close - open_).abs()
    wick_frac = (1.0 - body / crange).fillna(0.0)
    net_move = (close - close.shift(10)).abs()
    path_len = close.diff().abs().rolling(10, min_periods=10).sum().replace(0, np.nan)
    er = (net_move / path_len).fillna(0.5)

    heavy = (atr_ratio >= 2.8) | (crange >= 4.0 * atr)
    med = (atr_ratio >= 2.2) | (crange >= 2.8 * atr) | ((crange >= 2.0 * atr) & (wick_frac >= 0.70))
    light = (atr_ratio >= 1.8) | ((er < 0.15) & (atr_ratio >= 1.20))

    cd_mask = (
        (light.astype(int).rolling(b_light, min_periods=1).max() > 0)
        | (med.astype(int).rolling(b_med, min_periods=1).max() > 0)
        | (heavy.astype(int).rolling(b_heavy, min_periods=1).max() > 0)
    ).values
    return cd_mask, atr_ratio.values


def compute_u3_rates(
    closed_data: Dict[str, Dict[str, pd.DataFrame]],
    live_prices: Dict[str, float],
    unit_cfg: dict,
) -> Tuple[List[dict], Dict[str, Dict[str, float]]]:
    """Compute U3_HTF_MACRO_ATR for current & previous closed 4H bar across all 6 coins."""
    gamma_map = unit_cfg.get("gamma_tf", {"5m": 0.17, "15m": 0.29, "30m": 0.41, "1h": 0.59})
    min_u_pct = float(unit_cfg.get("min_u_pct", 0.0008))
    ema_smooth = int(unit_cfg.get("ema_smooth", 20))

    coin_cal_sl_tp2 = {
        "BTCUSDT": (5.0, 12.0, {"15m": 4.0, "30m": 4.0, "1h": 3.5}),
        "ETHUSDT": (6.0, 10.0, {"15m": 3.5, "30m": 3.5, "1h": 3.5}),
        "BNBUSDT": (6.0, 10.0, {"15m": 4.0, "30m": 3.5, "1h": 3.5}),
        "LINKUSDT": (6.0, 8.0, {"15m": 3.5, "30m": 3.5, "1h": 3.5}),
        "XRPUSDT": (6.0, 8.0, {"15m": 3.5, "30m": 3.5, "1h": 3.5}),
        "SOLUSDT": (6.0, 8.0, {"15m": 3.5, "30m": 3.5, "1h": 3.5}),
        "NEARUSDT": (6.0, 15.0, {"15m": 4.0, "30m": 4.0, "1h": 4.0}),
    }

    rates_rows: List[dict] = []
    u_price_map: Dict[str, Dict[str, float]] = {sym: {} for sym in SIX_COINS}

    for sym in SIX_COINS:
        df4 = closed_data[sym]["4h"]
        atr4 = compute_atr14(df4)
        macro_4h = ((0.5 * atr4) / df4["close"]).ewm(span=ema_smooth, adjust=False).mean()
        cur_macro = float(macro_4h.iloc[-1])
        prev_macro = float(macro_4h.iloc[-2]) if len(macro_4h) >= 2 else cur_macro
        px = live_prices.get(sym, float(df4["close"].iloc[-1]))

        sl_u, tp2_u, tp1_map = coin_cal_sl_tp2[sym]
        row: Dict[str, Any] = {
            "coin": sym,
            "price": round(px, 6),
            "macro_4h_pct": round(cur_macro * 100.0, 4),
            "prev_macro_4h_pct": round(prev_macro * 100.0, 4),
            "h4_closed_utc": df4["timestamp"].iloc[-1].strftime("%Y-%m-%d %H:%M"),
            "sl_u": sl_u,
            "tp2_u": tp2_u,
            "tf_details": {},
        }

        for tf in ("15m", "30m", "1h"):
            g = float(gamma_map.get(tf, 0.29))
            cur_u_frac = max(min_u_pct, cur_macro * g)
            prev_u_frac = max(min_u_pct, prev_macro * g)
            cur_pct = round(cur_u_frac * 100.0, 4)
            prev_pct = round(prev_u_frac * 100.0, 4)
            delta_pct = round(cur_pct - prev_pct, 4)

            u_price_map[sym][tf] = px * cur_u_frac
            tp1_u = float(tp1_map[tf])
            row[f"u_pct_{tf}"] = cur_pct
            row[f"prev_u_pct_{tf}"] = prev_pct
            row[f"delta_pct_{tf}"] = delta_pct
            row["tf_details"][tf] = {
                "u_pct": cur_pct,
                "prev_u_pct": prev_pct,
                "delta_pct": delta_pct,
                "u_usd_price": round(px * cur_u_frac, 6),
                "sl_pct": round(sl_u * cur_pct, 3),
                "tp1_u": tp1_u,
                "tp1_pct": round(tp1_u * cur_pct, 3),
                "tp2_pct": round(tp2_u * cur_pct, 3),
                "doi_sl_1u_pct": round(-1.0 * cur_pct, 3),
            }

        rates_rows.append(row)

    return rates_rows, u_price_map


def detect_latest_signals(
    closed_data: Dict[str, Dict[str, pd.DataFrame]],
    rules_map: Dict[Tuple[str, str, str], dict],
) -> List[dict]:
    """Scan all 6 coins x (15m, 30m, 1h) for setups firing on the latest strictly closed candle."""
    found: List[dict] = []

    for tf in ("15m", "30m", "1h"):
        htf_name = "1h" if tf == "15m" else "4h"
        cd_bars = TF_COOLDOWN[tf]
        btc_df = closed_data["BTCUSDT"][tf]
        btc_cd, _ = compute_storm_cooldown(btc_df, cd_bars)
        btc_ts = btc_df["open_time_ms"].values

        for sym in SIX_COINS:
            df = closed_data[sym][tf]
            hdf = closed_data[sym][htf_name]
            if len(df) < 210 or len(hdf) < 205:
                continue

            closes = df["close"].values
            opens = df["open"].values
            highs = df["high"].values
            lows = df["low"].values
            ts_ms = df["open_time_ms"].values
            ema34 = df["close"].ewm(span=34, adjust=False).mean().values
            ema89 = df["close"].ewm(span=89, adjust=False).mean().values
            ema200 = df["close"].ewm(span=200, adjust=False).mean().values
            atr = compute_atr14(df).values
            adx = compute_adx14(df)
            local_cd, atr_ratio_arr = compute_storm_cooldown(df, cd_bars)

            crange = np.where((highs - lows) > 0, highs - lows, np.nan)
            lower_wick = np.nan_to_num((np.minimum(opens, closes) - lows) / crange, nan=0.0)
            upper_wick = np.nan_to_num((highs - np.maximum(opens, closes)) / crange, nan=0.0)

            htf_e34 = hdf["close"].ewm(span=34, adjust=False).mean()
            htf_e200 = hdf["close"].ewm(span=200, adjust=False).mean()
            htf_tr_raw = np.where(
                (hdf["close"] > htf_e200) & (htf_e34 > htf_e200),
                1,
                np.where((hdf["close"] < htf_e200) & (htf_e34 < htf_e200), -1, 0),
            )
            htf_ts = hdf["open_time_ms"].values
            # Strict zero lookahead: use only HTF bars whose close_time <= tf bar close_time
            htf_close_ts = hdf["close_time_ms"].values
            last_idx = len(df) - 1
            last_close_ms = int(df["close_time_ms"].iloc[last_idx])

            ema_specs = [
                (34, ema34, 0.8, 3),
                (89, ema89, 1.0, 5),
                (200, ema200, 1.2, 8),
            ]

            for ema_p, et_arr, imp_mult, min_gap in ema_specs:
                trend = 0
                wave_count = 0
                armed = False
                last_touch = -999

                for i in range(205, len(df)):
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
                    if trend == 1:
                        touched = (l <= et + touch_buf) and (c >= et - breach_buf)
                    else:
                        touched = (h >= et - touch_buf) and (c <= et + breach_buf)
                    if not touched:
                        continue

                    wave_count += 1
                    armed = False
                    last_touch = i
                    if wave_count > 4 or i != last_idx:
                        continue

                    if adx[i] < 20.0:
                        continue

                    # Closed HTF bar lookup
                    h_idx = int(np.searchsorted(htf_close_ts, last_close_ms, side="right")) - 1
                    if h_idx < 0:
                        continue
                    ht = int(htf_tr_raw[h_idx])

                    if wave_count <= 3:
                        if ht != trend:
                            continue
                        direction = trend
                        stype = f"EMA{ema_p}_W{wave_count}"
                    else:
                        direction = -trend
                        stype = f"EMA{ema_p}_W4_REV"

                    if (sym, tf, stype) not in rules_map:
                        continue

                    b_idx = int(np.searchsorted(btc_ts, ts_ms[i], side="right")) - 1
                    btc_in_storm = bool(btc_cd[b_idx]) if b_idx >= 0 else False
                    if local_cd[i] or btc_in_storm:
                        continue

                    found.append({
                        "symbol": sym,
                        "tf": tf,
                        "setup_type": stype,
                        "direction": "LONG" if direction == 1 else "SHORT",
                        "dir_int": direction,
                        "signal_ts_ms": int(ts_ms[i]),
                        "signal_close_ms": last_close_ms,
                        "signal_utc": df["timestamp"].iloc[i].strftime("%Y-%m-%d %H:%M"),
                        "entry_price": float(c),
                        "elevated_vol": bool(atr_ratio_arr[i] >= 1.35),
                        "rule": rules_map[(sym, tf, stype)],
                    })

    found.sort(key=lambda x: (TF_PRIORITY[x["tf"]], -x["rule"]["adaptive_volume"]["init_vol_mult"], x["symbol"]))
    return found


def _calc_leg_pnl(entry: float, exit_px: float, qty: float, d: int, exit_maker: bool) -> float:
    slip_px = exit_px * (1.0 - d * SLIP_BPS) if not exit_maker else exit_px
    gross = d * (slip_px - entry) * qty
    # Live LIMIT IOC entry pays Taker fee (0.05%); exit pays Maker (0.02%) on TP or Taker (0.05%) on SL
    fees = (entry * qty * TAKER_FEE) + (slip_px * qty * (MAKER_FEE if exit_maker else TAKER_FEE))
    return float(gross - fees)


def _apply_s2_step(state: dict) -> None:
    """Update S2_LOCKED_HYSTERESIS_10PCT (unlimited steps, +8%/step, -10% hysteresis)."""
    acc = state["account"]
    eq = float(acc["equity"])
    step = int(acc["step"])
    tier_cap = float(acc["tier_capital"])
    while eq >= 10_000.0 * (1.08 ** (step + 1)):
        step += 1
        tier_cap = 10_000.0 * (1.08 ** step)
    while step > 0 and eq < tier_cap * 0.90:
        step -= 1
        tier_cap = 10_000.0 * (1.08 ** step)
    acc["step"] = step
    acc["tier_capital"] = round(tier_cap, 2)


def _fmt_px(px: float) -> str:
    apx = abs(float(px))
    if apx >= 1000:
        return f"${apx:,.1f}"
    if apx >= 10:
        return f"${apx:,.2f}"
    if apx >= 1:
        return f"${apx:,.3f}"
    return f"${apx:,.4f}"


def _fmt_signed_usd(val: float) -> str:
    v = float(val)
    return f"+${v:,.2f}" if v >= 0 else f"-${abs(v):,.2f}"


def _fmt_duration_vi(opened_utc: str, closed_utc: str) -> str:
    try:
        t0 = datetime.strptime(opened_utc[:16], "%Y-%m-%d %H:%M")
        t1 = datetime.strptime(closed_utc[:16], "%Y-%m-%d %H:%M")
        mins = max(0, int((t1 - t0).total_seconds() // 60))
        days, rem = divmod(mins, 1440)
        hrs, m = divmod(rem, 60)
        if days > 0:
            return f"{days} ngày {hrs} giờ {m:02d} phút"
        if hrs > 0:
            return f"{hrs} giờ {m:02d} phút"
        return f"{m} phút"
    except Exception:
        return "--"


def _s2_progress_info(acc: dict) -> Tuple[int, float, float, float, float]:
    step = int(acc.get("step", 0))
    eq = float(acc.get("equity", 10_000.0))
    tier_cap = float(acc.get("tier_capital", 10_000.0))
    buf = float(acc.get("buffer_units", 400))
    unit_usd = tier_cap / max(buf, 1.0)
    cur_base = 10_000.0 * (1.08 ** step)
    next_eq = 10_000.0 * (1.08 ** (step + 1))
    span = max(next_eq - cur_base, 1.0)
    prog_pct = max(0.0, min(100.0, (eq - cur_base) / span * 100.0))
    rem_usd = max(0.0, next_eq - eq)
    return step, unit_usd, prog_pct, next_eq, rem_usd


def _brake_status_text(acc: dict) -> str:
    if acc.get("shadow_paused"):
        return "Đang phanh 2-SL (Shadow-Win)"
    return f"An toàn ({int(acc.get('portfolio_consec_sl', 0))}/2 SL)"


def build_quickchart_png(
    sym: str,
    tf: str,
    candles_rows: List[List[float]],
    pos: Optional[dict] = None,
) -> Optional[bytes]:
    """Render 800x460 PNG candlestick + EMA 34/89/200 + Entry/SL/TP1/TP2 lines + blue Entry dot via QuickChart."""
    if not candles_rows or len(candles_rows) < 10:
        return None
    tail = candles_rows[-60:]
    n_show = len(tail)
    labels: List[str] = []
    wicks: List[List[float]] = []
    bodies: List[List[float]] = []
    colors: List[str] = []
    e34_show: List[float] = []
    e89_show: List[float] = []
    e200_show: List[float] = []

    y_min = float("inf")
    y_max = float("-inf")
    for r in tail:
        y_min = min(y_min, float(r[3]), float(r[5]), float(r[6]), float(r[7]))
        y_max = max(y_max, float(r[2]), float(r[5]), float(r[6]), float(r[7]))

    entry_idx = -1
    if pos:
        tf_sec = TF_MS.get(tf, 3_600_000) // 1000
        raw_entry_sec = int(pos.get("entry_bar_ts") or 0)
        aligned_sec = (raw_entry_sec // tf_sec) * tf_sec if raw_entry_sec > 0 else 0
        if aligned_sec and int(tail[0][0]) <= aligned_sec <= int(tail[-1][0]):
            for i, r in enumerate(tail):
                t_i = int(r[0])
                if t_i == aligned_sec or (t_i <= aligned_sec and (i == n_show - 1 or int(tail[i + 1][0]) > aligned_sec)):
                    entry_idx = i
                    break
        if entry_idx < 0:
            entry_idx = n_show - 1
        for k in ("entry_price", "active_sl_price", "tp1_price", "tp2_price"):
            v = float(pos.get(k) or 0.0)
            if v > 0:
                y_min = min(y_min, v)
                y_max = max(y_max, v)

    pad_range = max((y_max - y_min) * 0.05, y_max * 0.001)
    min_body = max((y_max - y_min) * 0.0025, y_max * 0.00015)

    for r in tail:
        t_sec = int(r[0])
        o, h, l, c = float(r[1]), float(r[2]), float(r[3]), float(r[4])
        dt_vn = datetime.fromtimestamp(t_sec + 7 * 3600, tz=timezone.utc)
        lbl = dt_vn.strftime("%d/%m %Hh") if tf == "4h" else dt_vn.strftime("%H:%M")
        labels.append(lbl)
        wicks.append([l, h])
        if abs(c - o) < min_body:
            bodies.append([o - min_body * 0.5, o + min_body * 0.5])
        else:
            bodies.append([min(o, c), max(o, c)])
        # Keep original green/red candle colors (never recolor the entry candle)
        colors.append("#059669" if c >= o else "#DC2626")
        e34_show.append(float(r[5]))
        e89_show.append(float(r[6]))
        e200_show.append(float(r[7]))

    datasets: List[dict] = [
        {"type": "line", "label": "EMA 34", "data": e34_show, "borderColor": "#0284C7", "borderWidth": 2, "pointRadius": 0, "fill": False, "tension": 0.15, "order": 1},
        {"type": "line", "label": "EMA 89", "data": e89_show, "borderColor": "#D97706", "borderWidth": 2, "pointRadius": 0, "fill": False, "tension": 0.15, "order": 1},
        {"type": "line", "label": "EMA 200", "data": e200_show, "borderColor": "#7C3AED", "borderWidth": 2.2, "pointRadius": 0, "fill": False, "tension": 0.15, "order": 1},
        {"type": "bar", "label": "Wick", "data": wicks, "backgroundColor": colors, "barThickness": 2, "grouped": False, "order": 2},
        {"type": "bar", "label": "Candle", "data": bodies, "backgroundColor": colors, "barPercentage": 0.72, "categoryPercentage": 0.88, "grouped": False, "order": 3},
    ]

    if pos:
        def _add_hline(lbl: str, price: float, col: str, dashed: bool) -> None:
            if price <= 0:
                return
            ds: Dict[str, Any] = {
                "type": "line",
                "label": lbl,
                "data": [price] * n_show,
                "borderColor": col,
                "borderWidth": 1.6 if dashed else 2,
                "pointRadius": 0,
                "fill": False,
                "order": 0,
            }
            if dashed:
                ds["borderDash"] = [6, 4]
            datasets.append(ds)

        _add_hline(f"Entry {_fmt_px(pos['entry_price'])}", float(pos["entry_price"]), "#2563EB", True)
        _add_hline(f"SL {_fmt_px(pos['active_sl_price'])}", float(pos["active_sl_price"]), "#DC2626", False)
        _add_hline(f"TP1 {_fmt_px(pos['tp1_price'])}", float(pos["tp1_price"]), "#059669", True)
        _add_hline(f"TP2 {_fmt_px(pos['tp2_price'])}", float(pos["tp2_price"]), "#15803D", False)

        if entry_idx >= 0:
            dot_data: List[Optional[float]] = [None] * n_show
            dot_data[entry_idx] = float(pos["entry_price"])
            datasets.append({
                "type": "line",
                "label": "Entry Point",
                "data": dot_data,
                "showLine": False,
                "pointRadius": 7,
                "pointBackgroundColor": "#2563EB",
                "pointBorderColor": "#FFFFFF",
                "pointBorderWidth": 2,
                "order": 0,
            })

    title_extra = f" · {pos['direction']} {pos['setup_code']}" if pos else ""
    chart_cfg = {
        "type": "bar",
        "data": {"labels": labels, "datasets": datasets},
        "options": {
            "plugins": {
                "title": {
                    "display": True,
                    "text": f"{sym} · {tf.upper()} ({n_show} nến · EMA 34/89/200{title_extra})",
                    "color": "#0F172A",
                    "font": {"size": 14, "weight": "bold"},
                },
                "legend": {
                    "display": True,
                    "labels": {
                        "color": "#334155",
                        "font": {"size": 11},
                    },
                },
            },
            "scales": {
                "x": {"ticks": {"color": "#475569", "maxTicksLimit": 10, "maxRotation": 0}, "grid": {"color": "#F1F5F9"}},
                "y": {"position": "right", "min": y_min - pad_range, "max": y_max + pad_range, "ticks": {"color": "#334155"}, "grid": {"color": "#E2E8F0"}},
            },
        },
    }

    try:
        res = requests.post(
            "https://quickchart.io/chart",
            json={"version": "3", "width": 800, "height": 460, "backgroundColor": "#FFFFFF", "format": "png", "chart": chart_cfg},
            timeout=15,
        )
        if res.status_code == 200 and res.content:
            return res.content
    except Exception as exc:
        print(f"[WARN] QuickChart PNG build failed for {sym} {tf}: {exc}")
    return None


def _send_telegram(event_or_text: Any, chart_candles_map: Optional[Dict[str, Dict[str, List[List[float]]]]] = None) -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    gas_url = os.environ.get("GAS_BROADCAST_URL", "").strip()
    gas_secret = os.environ.get("GAS_BROADCAST_SECRET", "").strip()

    if isinstance(event_or_text, dict):
        text = str(event_or_text.get("text", ""))
        event_type = str(event_or_text.get("event_type", "")).strip()
        pos_id = str(event_or_text.get("pos_id", "")).strip()
        reply_to_msg_id = event_or_text.get("reply_to_msg_id")
        pos_ref = event_or_text.get("pos_ref")
    else:
        text = str(event_or_text)
        event_type = ""
        pos_id = ""
        reply_to_msg_id = None
        pos_ref = None

    if not text:
        return

    png_bytes: Optional[bytes] = None
    photo_filename = "chart.png"
    if event_type == "OPEN" and pos_ref is not None and chart_candles_map:
        sym = str(pos_ref.get("symbol", ""))
        tf = str(pos_ref.get("tf", ""))
        rows = chart_candles_map.get(sym, {}).get(tf, [])
        if rows:
            png_bytes = build_quickchart_png(sym, tf, rows, pos_ref)
            photo_filename = f"{sym}_{tf}.png"

    if token and chat_id:
        sent_ok = False
        if png_bytes and len(text) <= 1020:
            try:
                data_form: Dict[str, Any] = {
                    "chat_id": chat_id,
                    "caption": text,
                    "parse_mode": "HTML",
                }
                if reply_to_msg_id:
                    data_form["reply_to_message_id"] = str(int(reply_to_msg_id))
                    data_form["allow_sending_without_reply"] = "true"
                res = requests.post(
                    f"https://api.telegram.org/bot{token}/sendPhoto",
                    data=data_form,
                    files={"photo": (photo_filename, png_bytes, "image/png")},
                    timeout=15,
                )
                rj = res.json() if res.ok else {}
                if rj.get("ok"):
                    sent_ok = True
                    if rj.get("result", {}).get("message_id") and event_type == "OPEN" and pos_ref is not None:
                        pos_ref["tg_open_msg_id"] = int(rj["result"]["message_id"])
            except Exception as exc:
                print(f"[WARN] Telegram sendPhoto failed, falling back to text: {exc}")

        if not sent_ok:
            payload: Dict[str, Any] = {
                "chat_id": chat_id,
                "text": text,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            }
            if reply_to_msg_id:
                payload["reply_to_message_id"] = int(reply_to_msg_id)
                payload["allow_sending_without_reply"] = True
            try:
                res = requests.post(f"https://api.telegram.org/bot{token}/sendMessage", json=payload, timeout=10)
                rj = res.json() if res.ok else {}
                if rj.get("ok") and rj.get("result", {}).get("message_id") and event_type == "OPEN" and pos_ref is not None:
                    pos_ref["tg_open_msg_id"] = int(rj["result"]["message_id"])
            except Exception as exc:
                print(f"[WARN] Telegram notify failed: {exc}")

    if gas_url.startswith("https://"):
        b_payload: Dict[str, Any] = {
            "action": "broadcast",
            "secret": gas_secret,
            "text": text,
            "exclude_chat_id": chat_id,
        }
        if event_type:
            b_payload["event_type"] = event_type
        if pos_id:
            b_payload["pos_id"] = pos_id
        if png_bytes:
            b_payload["photo_base64"] = base64.b64encode(png_bytes).decode("ascii")
            b_payload["photo_filename"] = photo_filename
        try:
            requests.post(gas_url, json=b_payload, timeout=15)
        except Exception as exc:
            print(f"[WARN] Broadcast failed: {exc}")


def _init_state(turbo_enabled: bool) -> dict:
    return {
        "version": "v3.0-u3-htf-macro-4h",
        "updated_at_utc": "",
        "next_m15_close_ms": 0,
        "next_h4_close_ms": 0,
        "last_daily_recap_date": "",
        "last_weekly_recap_week": "",
        "account": {
            "initial_capital": 10_000.0,
            "equity": 10_000.0,
            "peak_equity": 10_000.0,
            "max_dd_pct": 0.0,
            "roi_pct": 0.0,
            "step": 0,
            "tier_capital": 10_000.0,
            "sizing_policy": "S2_LOCKED_HYSTERESIS_10PCT (Không giới hạn bậc · Cap $300k/$150k)",
            "buffer_units": 400,
            "turbo_enabled": turbo_enabled,
            "turbo_active": False,
            "leverage": 20,
            "portfolio_consec_sl": 0,
            "shadow_paused": False,
            "shadow_paused_until_ms": 0,
        },
        "stats": {
            "total_trades": 0,
            "wins": 0,
            "wr_pct": 0.0,
            "profit_wr_pct": 0.0,
            "pf": 0.0,
            "net_pnl_usd": 0.0,
            "gross_profit_usd": 0.0,
            "gross_loss_usd": 0.0,
        },
        "live_u3_rates": [],
        "open_positions": [],
        "shadow_positions": [],
        "closed_trades": [],
        "equity_curve": [{"ts": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"), "eq": 10_000.0, "dd": 0.0}],
        "processed_signal_keys": [],
        "daily_tracker": {"date": "", "trades": 0, "pnl": 0.0, "start_eq": 10_000.0},
    }


def compute_must_order_score(r: dict) -> float:
    """Compute composite THẤY LÀ QUẤT score matching core/build_v21_ui.py."""
    p = r.get("prior_1y", {})
    net_u = float(p.get("net_u", 0.0))
    pwr = float(p.get("profit_wr_pct", 0.0))
    pf = float(p.get("pf_1y", 0.0))
    st = str(r.get("status", ""))
    bonus = 24.0 if st == "CORE" else (12.0 if st == "CORE_EXP" else (5.0 if st in ("SCOUT", "HOT_ADAPTIVE", "SCOUT_EXP") else 0.0))
    return bonus + (net_u / 90.0) * 55.0 + (pwr / 100.0) * 45.0 + min(pf, 3.5) * 2.0


def compute_top_must_keys(rules_list: List[dict], top_n: int = 15) -> set:
    """Select Top 15 setups as THẤY LÀ QUẤT (is_must_order)."""
    ranked = sorted(rules_list, key=lambda r: -compute_must_order_score(r))
    return {(r["coin"], r["tf"], r["setup_code"]) for r in ranked[:top_n]}


def compute_setup_live_stats(
    rules_map: Dict[Tuple[str, str, str], dict],
    closed_trades: List[dict],
) -> Dict[Tuple[str, str, str], dict]:
    """Compute real-time setup WR & Profit WR from 3Y prior + CLOSED trades ONLY (never open positions)."""
    acc_map: Dict[Tuple[str, str, str], dict] = {}
    for k, r in rules_map.items():
        p = r.get("prior_1y", {})
        base_n = int(p.get("trades", 0))
        base_wins = int(p.get("wins", 0))
        pwr = float(p.get("profit_wr_pct", 50.0)) / 100.0
        net_u = float(p.get("net_u", 0.0))
        if abs(2.0 * pwr - 1.0) > 0.002 and net_u != 0.0:
            tot_u = abs(net_u / (2.0 * pwr - 1.0))
            gp_u = pwr * tot_u
            gl_u = (1.0 - pwr) * tot_u
        else:
            avg_w = float(p.get("avg_win_u", 0.0))
            avg_l = float(p.get("avg_loss_u", 0.0))
            gp_u = avg_w * base_wins
            gl_u = avg_l * max(0, base_n - base_wins)
        acc_map[k] = {"n": base_n, "wins": base_wins, "gp_u": gp_u, "gl_u": gl_u}

    for tr in closed_trades:
        k = (tr.get("symbol"), tr.get("tf"), tr.get("setup_code"))
        if k not in acc_map:
            continue
        rec = acc_map[k]
        rec["n"] += 1
        pnl_usd = float(tr.get("pnl_usd", 0.0))
        if pnl_usd > 0:
            rec["wins"] += 1
        if "pnl_u" in tr and tr["pnl_u"] is not None:
            pnl_u = float(tr["pnl_u"])
        else:
            u_px = float(tr.get("u_price", 0.0))
            qty = float(tr.get("qty", 0.0))
            pnl_u = pnl_usd / (u_px * qty) if (u_px > 0 and qty > 0) else (pnl_usd / max(25.0 * float(tr.get("vol_mult", 1.0)), 1.0))
        if pnl_u > 0:
            rec["gp_u"] += pnl_u
        else:
            rec["gl_u"] += abs(pnl_u)

    out: Dict[Tuple[str, str, str], dict] = {}
    for k, rec in acc_map.items():
        n = rec["n"]
        tot_u = rec["gp_u"] + rec["gl_u"]
        out[k] = {
            "total_samples": n,
            "live_wr_pct": round(rec["wins"] / n * 100.0, 1) if n > 0 else 0.0,
            "live_profit_wr_pct": round(rec["gp_u"] / tot_u * 100.0, 1) if tot_u > 1e-6 else 0.0,
        }
    return out


def compute_expected_pnl_fields(pos: dict, equity: float) -> None:
    """Compute exact fee-adjusted expected Win (TP1 / TP2) & Loss (SL) USD for an open position."""
    entry = float(pos["entry_price"])
    qty = float(pos["qty"])
    d = 1 if pos["direction"] == "LONG" else -1
    arch = pos.get("exit_arch", "PARTIAL_MOVE_SL")
    sl_px = float(pos["sl_price"])
    active_sl_px = float(pos["active_sl_price"])
    tp1_px = float(pos["tp1_price"])
    tp2_px = float(pos["tp2_price"])
    r1_px = float(pos.get("r1_price", tp1_px))

    if arch in ("ONLY_TP1", "ONLY_1R"):
        tgt = tp1_px if arch == "ONLY_TP1" else r1_px
        exp_tp1 = _calc_leg_pnl(entry, tgt, qty, d, exit_maker=True)
        exp_tp2 = exp_tp1
    else:
        leg1_tp1 = float(pos["realized_pnl_usd"]) if pos.get("tp1_hit") else _calc_leg_pnl(entry, tp1_px, qty * 0.5, d, exit_maker=True)
        leg2_tp2 = _calc_leg_pnl(entry, tp2_px, qty * 0.5, d, exit_maker=True)
        exp_tp1 = leg1_tp1
        exp_tp2 = leg1_tp1 + leg2_tp2

    if pos.get("tp1_hit"):
        exp_sl = float(pos.get("realized_pnl_usd", 0.0)) + _calc_leg_pnl(entry, active_sl_px, qty * 0.5, d, exit_maker=False)
    else:
        exp_sl = _calc_leg_pnl(entry, sl_px, qty, d, exit_maker=False)

    pos["expected_tp1_usd"] = round(float(exp_tp1), 2)
    pos["expected_tp2_usd"] = round(float(exp_tp2), 2)
    pos["expected_sl_usd"] = round(float(exp_sl), 2)
    pos["expected_sl_eq_pct"] = round((float(exp_sl) / max(float(equity), 1.0)) * 100.0, 2)


def _compute_entry_bar_ts(pos: dict) -> int:
    if pos.get("entry_bar_ts"):
        return int(pos["entry_bar_ts"])
    tf_sec = TF_MS.get(str(pos.get("tf", "1h")), 3_600_000) // 1000
    try:
        opened_str = str(pos.get("opened_utc", ""))[:16]
        dt = datetime.strptime(opened_str, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
        return (int(dt.timestamp()) // tf_sec) * tf_sec
    except Exception:
        return 0


def build_chart_candles_payload(
    closed_data: Dict[str, Dict[str, pd.DataFrame]],
    updated_at_utc: str,
    tail_n: int = 80,
) -> dict:
    """Export last 80 strictly closed candles + engine-exact EMA 34/89/200 for all 6 coins x 4 TFs."""
    candles_out: Dict[str, Dict[str, List[List[float]]]] = {}
    for sym in SIX_COINS:
        candles_out[sym] = {}
        for tf in ("15m", "30m", "1h", "4h"):
            df = closed_data.get(sym, {}).get(tf)
            if df is None or len(df) == 0:
                candles_out[sym][tf] = []
                continue
            closes = df["close"].astype(float)
            e34 = closes.ewm(span=34, adjust=False).mean()
            e89 = closes.ewm(span=89, adjust=False).mean()
            e200 = closes.ewm(span=200, adjust=False).mean()
            sub = df.iloc[-tail_n:]
            rows: List[List[float]] = []
            for idx, row in sub.iterrows():
                t_sec = int(row["open_time_ms"]) // 1000
                rows.append([
                    t_sec,
                    round(float(row["open"]), 6),
                    round(float(row["high"]), 6),
                    round(float(row["low"]), 6),
                    round(float(row["close"]), 6),
                    round(float(e34.loc[idx]), 6),
                    round(float(e89.loc[idx]), 6),
                    round(float(e200.loc[idx]), 6),
                ])
            candles_out[sym][tf] = rows
    return {"updated_at_utc": updated_at_utc, "candles": candles_out}


def enrich_positions_metadata(
    state: dict,
    rules_map: Dict[Tuple[str, str, str], dict],
    top_must_keys: set,
) -> None:
    """Update live setup WR/Profit WR (3Y prior + CLOSED ONLY), is_must_order, entry_bar_ts & expected PnL."""
    live_setup_stats = compute_setup_live_stats(rules_map, state.get("closed_trades", []))
    equity = float(state["account"]["equity"])

    for pos in state.get("open_positions", []):
        k = (pos.get("symbol"), pos.get("tf"), pos.get("setup_code"))
        st = live_setup_stats.get(k)
        if st is not None:
            pos["setup_live_wr_pct"] = st["live_wr_pct"]
            pos["setup_live_profit_wr_pct"] = st["live_profit_wr_pct"]
            pos["setup_total_samples"] = st["total_samples"]
            pos["wr_pct"] = st["live_wr_pct"]
            pos["profit_wr_pct"] = st["live_profit_wr_pct"]
        pos["is_must_order"] = bool(k in top_must_keys)
        pos["entry_bar_ts"] = _compute_entry_bar_ts(pos)
        compute_expected_pnl_fields(pos, equity)

    for tr in state.get("closed_trades", []):
        k = (tr.get("symbol"), tr.get("tf"), tr.get("setup_code"))
        st = live_setup_stats.get(k)
        if st is not None:
            tr["setup_live_wr_pct"] = st["live_wr_pct"]
            tr["setup_live_profit_wr_pct"] = st["live_profit_wr_pct"]
            tr["setup_total_samples"] = st["total_samples"]
            tr.setdefault("wr_pct", st["live_wr_pct"])
            tr.setdefault("profit_wr_pct", st["live_profit_wr_pct"])
        tr["is_must_order"] = bool(k in top_must_keys)


def load_or_init_state(turbo_enabled: bool) -> dict:
    if os.path.exists(OUT_JSON):
        try:
            with open(OUT_JSON, encoding="utf-8") as f:
                st = json.load(f)
            st.setdefault("last_daily_recap_date", "")
            st.setdefault("last_weekly_recap_week", "")
            st.setdefault("shadow_positions", [])
            st["account"]["turbo_enabled"] = turbo_enabled
            st["account"]["sizing_policy"] = "S2_LOCKED_HYSTERESIS_10PCT (Không giới hạn bậc · Cap $300k/$150k)"
            return st
        except Exception as exc:
            print(f"[WARN] Could not parse existing {OUT_JSON}, starting fresh: {exc}")
    return _init_state(turbo_enabled)


def _format_shadow_pause_msg() -> str:
    return (
        "🛡️ <b>KÍCH HOẠT PHANH AN TOÀN 2-SL (CHẾ ĐỘ ĐÁNH NHÁP SHADOW-WIN)</b>\n"
        "Danh mục vừa ghi nhận <b>2 lệnh SL gốc liên tiếp</b> ở nhóm Core.\n\n"
        "• <b>Hành động của Bot:</b> Tạm khóa vào tiền thật cho các tín hiệu mới để tránh bão thị trường.\n"
        "• <b>Điều kiện mở khóa tự động:</b> Bot vẫn theo dõi ngầm 58 Setup 24/7 — ngay khi có 1 lệnh thắng nháp (Shadow-Win chạm TP1), hệ thống sẽ tự động mở khóa tiền thật trở lại!"
    )


def _format_shadow_resume_msg(reason_line: str) -> str:
    return (
        "🟢 <b>ĐÃ MỞ KHÓA TIỀN THẬT — VƯỢT BÃO THÀNH CÔNG (SHADOW-WIN)</b>\n"
        f"{reason_line}\n\n"
        "• Chuỗi nhiễu thị trường đã qua — Bot chính thức bật lại chế độ giao dịch tiền thật 100% từ tín hiệu tiếp theo!"
    )


def _check_shadow_positions(
    state: dict,
    closed_data: Dict[str, Dict[str, pd.DataFrame]],
    now_ms: int,
    events: List[str],
) -> None:
    """Evaluate shadow positions while 2-SL brake is active; unlock on first Shadow-Win or 72h timeout."""
    acc = state["account"]
    if not acc.get("shadow_paused"):
        return

    if now_ms >= int(acc.get("shadow_paused_until_ms", 0)):
        acc["shadow_paused"] = False
        acc["shadow_paused_until_ms"] = 0
        acc["portfolio_consec_sl"] = 0
        state["shadow_positions"] = []
        events.append(_format_shadow_resume_msg("Thời gian phanh tối đa <b>72 giờ</b> đã kết thúc an toàn."))
        return

    still_shadow: List[dict] = []
    for sp in state.get("shadow_positions", []):
        sym, tf = sp["symbol"], sp["tf"]
        df = closed_data[sym][tf]
        d = 1 if sp["direction"] == "LONG" else -1
        new_bars = df[df["close_time_ms"] > int(sp["last_checked_close_ms"])]
        won_shadow = False
        lost_shadow = False
        hit_label = "TP1"
        hit_u = float(sp.get("tp1_u", 3.5))

        for _, bar in new_bars.iterrows():
            sp["last_checked_close_ms"] = int(bar["close_time_ms"])
            sp["bars_held"] = int(sp.get("bars_held", 0)) + 1
            bh, bl = float(bar["high"]), float(bar["low"])
            arch = sp.get("exit_arch", "PARTIAL_MOVE_SL")
            tp_tgt = sp["tp1_price"] if arch != "ONLY_1R" else sp.get("r1_price", sp["tp1_price"])

            if (d == 1 and bl <= sp["sl_price"]) or (d == -1 and bh >= sp["sl_price"]):
                lost_shadow = True
                break
            if (d == 1 and bh >= tp_tgt) or (d == -1 and bl <= tp_tgt):
                won_shadow = True
                hit_label = "1R" if arch == "ONLY_1R" else "TP1"
                hit_u = float(sp.get("sl_u", 5.0)) if arch == "ONLY_1R" else float(sp.get("tp1_u", 3.5))
                break
            if sp["bars_held"] >= 200:
                lost_shadow = True
                break

        if won_shadow:
            acc["shadow_paused"] = False
            acc["shadow_paused_until_ms"] = 0
            acc["portfolio_consec_sl"] = 0
            state["shadow_positions"] = []
            events.append(
                _format_shadow_resume_msg(
                    f"Lệnh nháp <b>{sym} {tf} [{sp['setup_code']}]</b> vừa chạm <code>{hit_label}</code> (<b>+{hit_u:g}U</b>)."
                )
            )
            return
        if not lost_shadow:
            still_shadow.append(sp)

    state["shadow_positions"] = still_shadow


def update_open_positions(
    state: dict,
    closed_data: Dict[str, Dict[str, pd.DataFrame]],
    live_prices: Dict[str, float],
    events: List[str],
    now_ms: int = 0,
    rules_map: Optional[Dict[Tuple[str, str, str], dict]] = None,
) -> None:
    """Check closed candles for SL / TP1 / TP2 / Leg2_SL hits on all open positions."""
    if now_ms <= 0:
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    _check_shadow_positions(state, closed_data, now_ms, events)

    acc = state["account"]
    stats = state["stats"]
    orig_open = list(state["open_positions"])
    still_open: List[dict] = []

    for idx, pos in enumerate(orig_open):
        sym, tf = pos["symbol"], pos["tf"]
        df = closed_data[sym][tf]
        d = 1 if pos["direction"] == "LONG" else -1
        new_bars = df[df["close_time_ms"] > int(pos["last_checked_close_ms"])]
        rule_obj = rules_map.get((sym, tf, pos["setup_code"])) if rules_map else None
        base_coin = sym.replace("USDT", "")
        tp1_u_val = float(pos.get("tp1_u") or (rule_obj["tp1_u"] if rule_obj else 3.5))
        tp2_u_val = float(pos.get("tp2_u") or (rule_obj["tp2_u"] if rule_obj else 12.0))

        closed_outcome = None
        close_px = float(live_prices.get(sym, pos["entry_price"]))
        close_ts_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")

        for _, bar in new_bars.iterrows():
            pos["last_checked_close_ms"] = int(bar["close_time_ms"])
            pos["bars_held"] = int(pos.get("bars_held", 0)) + 1
            bh, bl, bc = float(bar["high"]), float(bar["low"]), float(bar["close"])
            ts_str = bar["timestamp"].strftime("%Y-%m-%d %H:%M")

            arch = pos["exit_arch"]
            if arch in ("ONLY_TP1", "ONLY_1R"):
                tp_tgt = pos["tp1_price"] if arch == "ONLY_TP1" else pos["r1_price"]
                if (d == 1 and bl <= pos["sl_price"]) or (d == -1 and bh >= pos["sl_price"]):
                    closed_outcome, close_px, close_ts_str = "SL", pos["sl_price"], ts_str
                    break
                if (d == 1 and bh >= tp_tgt) or (d == -1 and bl <= tp_tgt):
                    closed_outcome = "TP1_FULL" if arch == "ONLY_TP1" else "1R_FULL"
                    close_px, close_ts_str = tp_tgt, ts_str
                    break
            else:
                if not pos["tp1_hit"]:
                    if (d == 1 and bl <= pos["sl_price"]) or (d == -1 and bh >= pos["sl_price"]):
                        closed_outcome, close_px, close_ts_str = "SL", pos["sl_price"], ts_str
                        break
                    if (d == 1 and bh >= pos["tp1_price"]) or (d == -1 and bl <= pos["tp1_price"]):
                        pos["tp1_hit"] = True
                        q1 = pos["qty"] * 0.5
                        pnl1 = _calc_leg_pnl(pos["entry_price"], pos["tp1_price"], q1, d, exit_maker=True)
                        pos["realized_pnl_usd"] = round(float(pos.get("realized_pnl_usd", 0.0)) + pnl1, 2)
                        acc["equity"] = round(float(acc["equity"]) + pnl1, 2)
                        acc["peak_equity"] = max(float(acc["peak_equity"]), float(acc["equity"]))
                        pos["active_sl_price"] = pos["leg2_sl_price"]
                        compute_expected_pnl_fields(pos, float(acc["equity"]))
                        rem_q_tp1 = round(float(pos["qty"]) * 0.5, 6)
                        l2_desc = pos.get("leg2_sl_label") or "Dời SL chân 2"
                        must_tag = " · 🔥 <b>THẤY LÀ QUẤT</b>" if pos.get("is_must_order") else ""
                        events.append({
                            "event_type": "TP1_HIT",
                            "pos_id": pos["id"],
                            "reply_to_msg_id": pos.get("tg_open_msg_id"),
                            "text": (
                                f"🎯 <b>CHỐT TP1 (50%) — {sym} · {tf.upper()} [{pos['setup_code']}]</b>{must_tag}\n"
                                f"• Chốt Chân 1: <b>{_fmt_signed_usd(pnl1)}</b> (+{tp1_u_val:g}U) tại <code>{_fmt_px(pos['tp1_price'])}</code>\n"
                                f"• 🛡️ Dời SL Chân 2 ({rem_q_tp1} {base_coin}): <code>{_fmt_px(pos['active_sl_price'])}</code> ({l2_desc})\n"
                                f"• 🚀 Đích TP2 (+{tp2_u_val:g}U): <code>{_fmt_px(pos['tp2_price'])}</code> (Tổng lãi nếu đạt: <b>{_fmt_signed_usd(pos['expected_tp2_usd'])}</b>)"
                            ),
                        })
                        if (d == 1 and bh >= pos["tp2_price"]) or (d == -1 and bl <= pos["tp2_price"]):
                            closed_outcome, close_px, close_ts_str = "TP2", pos["tp2_price"], ts_str
                            break
                else:
                    if (d == 1 and bl <= pos["active_sl_price"]) or (d == -1 and bh >= pos["active_sl_price"]):
                        closed_outcome, close_px, close_ts_str = "LEG2_SL", pos["active_sl_price"], ts_str
                        break
                    if (d == 1 and bh >= pos["tp2_price"]) or (d == -1 and bl <= pos["tp2_price"]):
                        closed_outcome, close_px, close_ts_str = "TP2", pos["tp2_price"], ts_str
                        break

            if pos["bars_held"] >= 200:
                closed_outcome, close_px, close_ts_str = "TIMEOUT", bc, ts_str
                break

        if closed_outcome is None:
            cur_px = float(live_prices.get(sym, pos["entry_price"]))
            rem_qty = pos["qty"] * (0.5 if pos["tp1_hit"] else 1.0)
            unr_pnl = d * (cur_px - pos["entry_price"]) * rem_qty
            pos["current_price"] = round(cur_px, 6)
            pos["unrealized_pnl_usd"] = round(float(unr_pnl), 2)
            pos["unrealized_pct"] = round(d * (cur_px - pos["entry_price"]) / pos["entry_price"] * 100.0, 3)
            still_open.append(pos)
            continue

        # Close remaining leg & settle account
        leg1_realized = float(pos.get("realized_pnl_usd", 0.0))
        was_tp1_hit = bool(pos.get("tp1_hit"))
        rem_qty = pos["qty"] * (0.5 if was_tp1_hit else 1.0)
        exit_maker = closed_outcome in ("TP1_FULL", "1R_FULL", "TP2")
        pnl_rem = _calc_leg_pnl(pos["entry_price"], close_px, rem_qty, d, exit_maker=exit_maker)
        tot_pnl = round(leg1_realized + pnl_rem, 2)

        eq_before = max(float(acc["equity"]), 1.0)
        acc["equity"] = round(float(acc["equity"]) + pnl_rem, 2)
        acc["peak_equity"] = max(float(acc["peak_equity"]), float(acc["equity"]))
        dd_now = (float(acc["equity"]) - float(acc["peak_equity"])) / float(acc["peak_equity"]) * 100.0
        acc["max_dd_pct"] = round(min(float(acc["max_dd_pct"]), dd_now), 2)
        _apply_s2_step(state)

        dk = close_ts_str[:10]
        if state["daily_tracker"]["date"] == dk:
            state["daily_tracker"]["pnl"] = round(float(state["daily_tracker"]["pnl"]) + tot_pnl, 2)

        if acc.get("turbo_enabled", False):
            if (not acc["turbo_active"]) and acc["equity"] >= 12_500.0:
                acc["turbo_active"] = True
            elif acc["turbo_active"] and acc["equity"] < 11_500.0:
                acc["turbo_active"] = False
            acc["buffer_units"] = 250 if acc["turbo_active"] else 400
        else:
            acc["turbo_active"] = False
            acc["buffer_units"] = 400

        triggered_shadow_pause = False
        if closed_outcome == "SL" and pos["base_vol"] > 0.45:
            acc["portfolio_consec_sl"] = int(acc.get("portfolio_consec_sl", 0)) + 1
            if acc["portfolio_consec_sl"] >= 2:
                acc["portfolio_consec_sl"] = 0
                acc["shadow_paused"] = True
                acc["shadow_paused_until_ms"] = int(pos["last_checked_close_ms"]) + 72 * 3_600_000
                triggered_shadow_pause = True
        elif tot_pnl > 0:
            acc["portfolio_consec_sl"] = 0

        stats["total_trades"] = int(stats["total_trades"]) + 1
        if tot_pnl > 0:
            stats["wins"] = int(stats["wins"]) + 1
            stats["gross_profit_usd"] = round(float(stats["gross_profit_usd"]) + tot_pnl, 2)
        else:
            stats["gross_loss_usd"] = round(float(stats["gross_loss_usd"]) + abs(tot_pnl), 2)

        gp, gl = float(stats["gross_profit_usd"]), float(stats["gross_loss_usd"])
        stats["net_pnl_usd"] = round(gp - gl, 2)
        stats["wr_pct"] = round(stats["wins"] / max(stats["total_trades"], 1) * 100.0, 2)
        stats["profit_wr_pct"] = round(gp / max(gp + gl, 0.01) * 100.0, 2)
        stats["pf"] = round(gp / max(gl, 0.01), 2) if gl > 0.01 else (99.99 if gp > 0 else 0.0)

        u_px = float(pos.get("u_price", 0.0))
        qty_f = float(pos["qty"])
        pnl_u = round(tot_pnl / (u_px * qty_f), 4) if (u_px > 0 and qty_f > 0) else 0.0
        closed_rec = {
            "id": pos["id"],
            "symbol": sym,
            "tf": tf,
            "setup_code": pos["setup_code"],
            "direction": pos["direction"],
            "entry_price": pos["entry_price"],
            "exit_price": round(close_px, 6),
            "qty": pos["qty"],
            "u_price": u_px,
            "vol_mult": pos["vol_mult"],
            "profit_wr_pct": pos.get("profit_wr_pct", 0.0),
            "wr_pct": pos.get("wr_pct", 0.0),
            "is_must_order": bool(pos.get("is_must_order", False)),
            "outcome": closed_outcome,
            "pnl_usd": tot_pnl,
            "pnl_u": pnl_u,
            "opened_utc": pos["opened_utc"],
            "closed_utc": close_ts_str,
        }
        state["closed_trades"] = ([closed_rec] + state["closed_trades"])[:500]
        state["equity_curve"].append({"ts": close_ts_str, "eq": acc["equity"], "dd": round(dd_now, 2)})
        state["equity_curve"] = state["equity_curve"][-1000:]

        # Updated 3Y+Live setup stats including this newly closed trade
        setup_pwr = float(pos.get("profit_wr_pct", 0.0))
        setup_wr = float(pos.get("wr_pct", 0.0))
        setup_samples = int(pos.get("setup_total_samples", 0)) + 1
        if rules_map:
            updated_map = compute_setup_live_stats(rules_map, state["closed_trades"])
            st_upd = updated_map.get((sym, tf, pos["setup_code"]))
            if st_upd:
                setup_pwr = st_upd["live_profit_wr_pct"]
                setup_wr = st_upd["live_wr_pct"]
                setup_samples = st_upd["total_samples"]

        must_tag = " · 🔥 <b>THẤY LÀ QUẤT</b>" if pos.get("is_must_order") else ""
        if closed_outcome == "TP2":
            title_line = f"🏆 <b>ĐÓNG LỆNH THẮNG TRỌN TP2 — {sym} · {tf.upper()} [{pos['setup_code']}]</b>{must_tag}"
        elif closed_outcome in ("TP1_FULL", "1R_FULL"):
            title_line = f"🏆 <b>ĐÓNG LỆNH CHỐT ĐÍCH {closed_outcome} — {sym} · {tf.upper()} [{pos['setup_code']}]</b>{must_tag}"
        elif closed_outcome == "LEG2_SL":
            ic = "✅" if tot_pnl > 0 else "⚠️"
            title_line = f"{ic} <b>ĐÓNG LỆNH DỜI SL CHÂN 2 — {sym} · {tf.upper()} [{pos['setup_code']}]</b>{must_tag}"
        elif closed_outcome == "TIMEOUT":
            ic = "✅" if tot_pnl > 0 else "⏱️"
            title_line = f"{ic} <b>ĐÓNG LỆNH HẾT GIỜ (TIMEOUT) — {sym} · {tf.upper()} [{pos['setup_code']}]</b>{must_tag}"
        else:
            title_line = f"❌ <b>ĐÓNG LỆNH DÍNH SL GỐC — {sym} · {tf.upper()} [{pos['setup_code']}]</b>{must_tag}"

        eq_impact_pct = (tot_pnl / eq_before) * 100.0
        if was_tp1_hit and closed_outcome in ("TP2", "LEG2_SL", "TIMEOUT"):
            detail_line = (
                f"• Chi tiết: Chân 1 chốt TP1 (<code>{_fmt_signed_usd(leg1_realized)}</code>) · "
                f"Chân 2 đóng ({closed_outcome}) tại <code>{_fmt_px(close_px)}</code> (<code>{_fmt_signed_usd(pnl_rem)}</code>)"
            )
        else:
            detail_line = (
                f"• Chi tiết: Entry <code>{_fmt_px(pos['entry_price'])}</code> → Đóng tại <code>{_fmt_px(close_px)}</code> "
                f"(KL <code>{qty_f}</code> {base_coin})"
            )

        dur_str = _fmt_duration_vi(pos["opened_utc"], close_ts_str)
        step_n, unit_usd, prog_pct, next_eq, _ = _s2_progress_info(acc)
        init_cap = float(acc.get("initial_capital", 10_000.0))
        net_from_init = float(acc["equity"]) - init_cap
        roi_now = (float(acc["equity"]) / init_cap - 1.0) * 100.0
        rem_open_cnt = len(still_open) + (len(orig_open) - 1 - idx)

        events.append({
            "event_type": "CLOSED",
            "pos_id": pos["id"],
            "reply_to_msg_id": pos.get("tg_open_msg_id"),
            "text": (
                f"{title_line}\n"
                f"• Kết quả: <b>{_fmt_signed_usd(tot_pnl)}</b> ({eq_impact_pct:+.2f}% vốn · {pnl_u:+.2f}U) · Giữ: <b>{dur_str}</b>\n"
                f"{detail_line}\n"
                f"• Balance: <b>${acc['equity']:,.2f}</b> ({roi_now:+.2f}%) · Bậc {step_n} ({prog_pct:.1f}% → Bậc {step_n + 1}) · Mở: <b>{rem_open_cnt}/5</b>\n"
                f"• Setup (3Y+Live): WR <b>{setup_wr:.1f}%</b> · Thắng tiền <b>{setup_pwr:.1f}%</b> ({setup_samples} lệnh) · Phanh: <b>{_brake_status_text(acc)}</b>"
            ),
        })
        if triggered_shadow_pause:
            events.append(_format_shadow_pause_msg())

    state["open_positions"] = still_open


def open_new_positions(
    state: dict,
    signals: List[dict],
    u_price_map: Dict[str, Dict[str, float]],
    filters: Dict[str, Dict[str, float]],
    now_ms: int,
    events: List[str],
    top_must_keys: Optional[set] = None,
    rules_map: Optional[Dict[Tuple[str, str, str], dict]] = None,
) -> None:
    """Open qualifying signals with 20x Cross Margin, stepSize floor, minNotional & S2 liq cap."""
    acc = state["account"]
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if state["daily_tracker"]["date"] != today_str:
        state["daily_tracker"] = {"date": today_str, "trades": 0, "pnl": 0.0, "start_eq": float(acc["equity"])}

    dt = state["daily_tracker"]
    seen_keys = set(state.get("processed_signal_keys", []))
    live_setup_stats = compute_setup_live_stats(rules_map, state.get("closed_trades", [])) if rules_map else {}

    for sig in signals:
        sig_key = f"{sig['symbol']}|{sig['tf']}|{sig['setup_type']}|{sig['signal_ts_ms']}"
        if sig_key in seen_keys:
            continue
        seen_keys.add(sig_key)

        sym, tf = sig["symbol"], sig["tf"]
        u = float(u_price_map[sym][tf])
        rule = sig["rule"]
        d = sig["dir_int"]
        p_sig = float(sig["entry_price"])

        if acc["shadow_paused"]:
            if now_ms >= int(acc.get("shadow_paused_until_ms", 0)):
                acc["shadow_paused"] = False
                acc["shadow_paused_until_ms"] = 0
                acc["portfolio_consec_sl"] = 0
                state["shadow_positions"] = []
                events.append(_format_shadow_resume_msg("Thời gian phanh tối đa <b>72 giờ</b> đã kết thúc an toàn."))
            else:
                tick = filters[sym]["tickSize"]
                sl_u_s, tp1_u_s = float(rule["sl_u"]), float(rule["tp1_u"])
                if not any(sp["symbol"] == sym for sp in state.get("shadow_positions", [])):
                    state.setdefault("shadow_positions", []).append({
                        "symbol": sym,
                        "tf": tf,
                        "setup_code": sig["setup_type"],
                        "direction": sig["direction"],
                        "exit_arch": rule["exit_arch"],
                        "sl_u": sl_u_s,
                        "tp1_u": tp1_u_s,
                        "sl_price": floor_step(p_sig - d * sl_u_s * u, tick),
                        "tp1_price": floor_step(p_sig + d * tp1_u_s * u, tick),
                        "r1_price": floor_step(p_sig + d * sl_u_s * u, tick),
                        "bars_held": 0,
                        "last_checked_close_ms": sig["signal_close_ms"],
                    })
                continue

        if dt["trades"] >= 10 or dt["pnl"] <= -0.05 * dt["start_eq"]:
            continue
        if len(state["open_positions"]) >= 5 or any(p["symbol"] == sym for p in state["open_positions"]):
            continue
        # Correlation guard: cap max 3 simultaneous open positions in the same direction (LONG or SHORT)
        if sum(1 for p in state["open_positions"] if p["direction"] == sig["direction"]) >= 3:
            continue

        eff_m = float(rule["adaptive_volume"]["init_vol_mult"])
        if sig["elevated_vol"]:
            eff_m *= 0.5
        cur_dd_pct = (float(acc["equity"]) - float(acc["peak_equity"])) / float(acc["peak_equity"]) * 100.0
        if cur_dd_pct <= -6.0:
            eff_m = min(eff_m, 0.25)

        open_m_sum = sum(float(p["vol_mult"]) for p in state["open_positions"])
        eff_m = min(eff_m, max(0.0, 4.25 - open_m_sum))
        if eff_m < 0.20:
            continue

        sb = float(acc["tier_capital"])
        cur_buf = 250.0 if (acc.get("turbo_active") and acc.get("turbo_enabled")) else 400.0
        sl_u, tp1_u, tp2_u = float(rule["sl_u"]), float(rule["tp1_u"]), float(rule["tp2_u"])
        l2_u = float(rule["leg2_sl_u"])

        if cur_buf == 250.0:
            open_sl_risk = sum(float(p["worst_sl_usd"]) for p in state["open_positions"])
            new_sl_risk = (sb * eff_m) * (sl_u / cur_buf) * 1.05
            if float(acc["equity"]) - (open_sl_risk + new_sl_risk) < 11_000.0:
                cur_buf = 400.0

        qty_raw = (sb * eff_m) / (cur_buf * u)
        liq_cap = 300_000.0 if sym in ("BTCUSDT", "ETHUSDT") else 150_000.0
        if qty_raw * p_sig > liq_cap:
            qty_raw = liq_cap / p_sig

        flt = filters[sym]
        qty = floor_step(qty_raw, flt["stepSize"])
        notional = qty * p_sig
        if qty <= 0 or notional < flt["minNotional"]:
            continue

        # 20x Cross Margin Pre-Check: keep total initial margin <= 75% of equity
        used_margin = sum(float(p["notional_usd"]) / 20.0 for p in state["open_positions"])
        if (used_margin + notional / 20.0) > float(acc["equity"]) * 0.75:
            continue

        tick = flt["tickSize"]
        sl_px = floor_step(p_sig - d * sl_u * u, tick)
        tp1_px = floor_step(p_sig + d * tp1_u * u, tick)
        tp2_px = floor_step(p_sig + d * tp2_u * u, tick)
        r1_px = floor_step(p_sig + d * sl_u * u, tick)
        if rule["exit_arch"] == "PARTIAL_NO_MOVE":
            l2_px = sl_px
        elif abs(l2_u) < 1e-6 or "BE" in rule.get("leg2_sl_label", ""):
            l2_px = floor_step(p_sig * (1.0 + d * 2.0 * TAKER_FEE), tick)
        else:
            l2_px = floor_step(p_sig + d * l2_u * u, tick)

        worst_sl_usd = round(qty * (sl_u * u) * 1.05, 2)
        pos_id = f"{sym}_{tf}_{sig['setup_type']}_{sig['signal_ts_ms']}"
        prior = rule.get("prior_1y", {})
        st_live = live_setup_stats.get((sym, tf, sig["setup_type"]))
        pwr_pct = st_live["live_profit_wr_pct"] if st_live else round(float(prior.get("profit_wr_pct", 0.0)), 1)
        wr_pct = st_live["live_wr_pct"] if st_live else round(float(prior.get("wr_pct", 0.0)), 1)
        samples_n = st_live["total_samples"] if st_live else int(prior.get("trades", 0))
        is_must = bool(top_must_keys and (sym, tf, sig["setup_type"]) in top_must_keys)
        pos_obj = {
            "id": pos_id,
            "symbol": sym,
            "tf": tf,
            "setup_code": sig["setup_type"],
            "direction": sig["direction"],
            "status": rule.get("status", "CORE"),
            "entry_price": round(p_sig, 6),
            "current_price": round(p_sig, 6),
            "qty": qty,
            "notional_usd": round(notional, 2),
            "vol_mult": round(eff_m, 2),
            "base_vol": float(rule["adaptive_volume"]["init_vol_mult"]),
            "profit_wr_pct": pwr_pct,
            "wr_pct": wr_pct,
            "setup_live_profit_wr_pct": pwr_pct,
            "setup_live_wr_pct": wr_pct,
            "setup_total_samples": samples_n,
            "is_must_order": is_must,
            "buffer_used": int(cur_buf),
            "u_price": round(u, 6),
            "sl_u": sl_u,
            "tp1_u": tp1_u,
            "tp2_u": tp2_u,
            "exit_arch": rule["exit_arch"],
            "sl_price": sl_px,
            "active_sl_price": sl_px,
            "leg2_sl_price": l2_px,
            "leg2_sl_label": rule.get("leg2_sl_label", ""),
            "tp1_price": tp1_px,
            "tp2_price": tp2_px,
            "r1_price": r1_px,
            "tp1_hit": False,
            "realized_pnl_usd": 0.0,
            "unrealized_pnl_usd": 0.0,
            "unrealized_pct": 0.0,
            "worst_sl_usd": worst_sl_usd,
            "bars_held": 0,
            "opened_utc": sig["signal_utc"],
            "entry_bar_ts": int(sig["signal_ts_ms"]) // 1000,
            "last_checked_close_ms": sig["signal_close_ms"],
        }
        compute_expected_pnl_fields(pos_obj, float(acc["equity"]))
        state["open_positions"].append(pos_obj)
        dt["trades"] = int(dt["trades"]) + 1

        dir_ic = "🟢" if d == 1 else "🔴"
        base_coin = sym.replace("USDT", "")
        u_pct = (u / p_sig * 100.0) if p_sig > 0 else 0.0
        mode_lbl = "Auto-Turbo 250G" if int(cur_buf) == 250 else "Bao 400G"
        rank_line = (
            f"🏆 <b>Hạng S+ (🔥 THẤY LÀ QUẤT)</b> · Volume {eff_m:.2f}×"
            if is_must
            else f"📌 Nhóm <b>{rule.get('status', 'CORE')}</b> · Volume {eff_m:.2f}×"
        )
        arch = rule["exit_arch"]
        if arch in ("ONLY_TP1", "ONLY_1R"):
            tgt_px = tp1_px if arch == "ONLY_TP1" else r1_px
            tgt_u = tp1_u if arch == "ONLY_TP1" else sl_u
            tgt_lbl = "TP1" if arch == "ONLY_TP1" else "1R"
            targets_block = (
                f"• {tgt_lbl} (Chốt 100% · +{tgt_u:g}U): <code>{_fmt_px(tgt_px)}</code>\n"
                f"• SL gốc (-{sl_u:g}U): <code>{_fmt_px(sl_px)}</code>"
            )
            exp_win_line = f"• Chạm {tgt_lbl}: <b>{_fmt_signed_usd(pos_obj['expected_tp1_usd'])}</b>"
        else:
            l2_note = rule.get("leg2_sl_label") or "Dời SL chân 2"
            targets_block = (
                f"• TP1 (Chốt 50% · +{tp1_u:g}U): <code>{_fmt_px(tp1_px)}</code>\n"
                f"• TP2 (Chốt 50% · +{tp2_u:g}U): <code>{_fmt_px(tp2_px)}</code> ({l2_note})\n"
                f"• SL gốc (-{sl_u:g}U): <code>{_fmt_px(sl_px)}</code>"
            )
            exp_win_line = (
                f"• Chạm TP1: <b>{_fmt_signed_usd(pos_obj['expected_tp1_usd'])}</b> | "
                f"Ăn trọn TP2: <b>{_fmt_signed_usd(pos_obj['expected_tp2_usd'])}</b>"
            )

        events.append({
            "event_type": "OPEN",
            "pos_id": pos_id,
            "pos_ref": pos_obj,
            "text": (
                f"{dir_ic} <b>MỞ LỆNH {sig['direction']} — {sym} · {tf.upper()} [{sig['setup_type']}]</b>\n"
                f"{rank_line}\n\n"
                f"• Giá khớp Entry: <code>{_fmt_px(p_sig)}</code> (1U = ${_fmt_px(u)[1:]} · {u_pct:.3f}%)\n"
                f"• Khối lượng: <code>{qty}</code> {base_coin} (~${notional:,.0f} · Bậc {int(acc['step'])} · {mode_lbl})\n"
                f"• Số dư hiện tại (Balance): <b>${float(acc['equity']):,.2f}</b>\n"
                f"• Lịch sử Setup (3Y+Live): Thắng tiền <b>{pwr_pct:.1f}%</b> · WR <b>{wr_pct:.1f}%</b> ({samples_n} lệnh)\n\n"
                f"🎯 <b>Mốc Quản Trị Lệnh:</b>\n"
                f"{targets_block}\n\n"
                f"💰 <b>Dự Kiến Lãi / Lỗ Tiền Thật (Sau phí):</b>\n"
                f"{exp_win_line}\n"
                f"• Dính SL gốc: <b>{_fmt_signed_usd(pos_obj['expected_sl_usd'])}</b> ({pos_obj['expected_sl_eq_pct']:+.2f}% vốn)"
            ),
        })

    state["processed_signal_keys"] = sorted(seen_keys)[-500:]


def _collect_window_closed(closed_trades: List[dict], cutoff_dt: datetime) -> List[dict]:
    out: List[dict] = []
    for tr in closed_trades:
        c_str = str(tr.get("closed_utc", ""))[:16]
        try:
            cdt = datetime.strptime(c_str, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
            if cdt >= cutoff_dt:
                out.append(tr)
        except Exception:
            continue
    return out


def _format_macro_1u_block(rates_rows: List[dict]) -> str:
    by_coin = {r["coin"].replace("USDT", ""): float(r.get("u_pct_15m", 0.0)) for r in rates_rows}
    r1 = f"• BTC: {by_coin.get('BTC', 0.0):.3f}% · ETH: {by_coin.get('ETH', 0.0):.3f}% · BNB: {by_coin.get('BNB', 0.0):.3f}%"
    r2 = f"• LINK: {by_coin.get('LINK', 0.0):.3f}% · XRP: {by_coin.get('XRP', 0.0):.3f}% · SOL: {by_coin.get('SOL', 0.0):.3f}%"
    return f"{r1}\n{r2}"


def build_recap_message(state: dict, now_dt: datetime, is_weekly: bool = False) -> str:
    """Build Mẫu 5 Daily Recap (07:00 VN) or Weekly Recap (Monday 07:00 VN)."""
    acc = state["account"]
    vn_dt = now_dt + timedelta(hours=7)
    init_cap = float(acc.get("initial_capital", 10_000.0))
    eq = float(acc.get("equity", init_cap))
    net_diff = eq - init_cap
    roi_pct = (eq / init_cap - 1.0) * 100.0
    step_n, unit_usd, _, next_eq, rem_usd = _s2_progress_info(acc)
    mode_lbl = "Auto-Turbo 250G" if acc.get("turbo_active") else f"Bao {int(acc.get('buffer_units', 400))}G"

    hours_back = 168 if is_weekly else 24
    win_closed = _collect_window_closed(state.get("closed_trades", []), now_dt - timedelta(hours=hours_back))
    w_wins = sum(1 for t in win_closed if float(t.get("pnl_usd", 0.0)) > 0)
    w_losses = len(win_closed) - w_wins
    w_pnl = sum(float(t.get("pnl_usd", 0.0)) for t in win_closed)

    open_pos = state.get("open_positions", [])
    if open_pos:
        tot_unr = sum(float(p.get("realized_pnl_usd", 0.0)) + float(p.get("unrealized_pnl_usd", 0.0)) for p in open_pos)
        top_p = open_pos[0]
        extra = f" (+{len(open_pos) - 1} lệnh khác)" if len(open_pos) > 1 else ""
        open_line = (
            f"• Lệnh đang mở: <b>{len(open_pos)}/5 lệnh</b> "
            f"({top_p['symbol']} {top_p['tf']} {top_p['direction']}{extra} · PnL tạm tính: <b>{_fmt_signed_usd(tot_unr)}</b>)"
        )
    else:
        open_line = "• Lệnh đang mở: <b>0/5 lệnh</b> (Quỹ đang giữ 100% USDT chờ tín hiệu)"

    if len(win_closed) > 0:
        period_lbl = "7 Ngày Qua" if is_weekly else "24 Giờ Qua"
        act_lbl = "trong tuần" if is_weekly else "trong ngày"
        activity_line = (
            f"• Lệnh đã đóng {act_lbl}: <b>{len(win_closed)} lệnh</b> "
            f"({w_wins} Thắng / {w_losses} Thua · Lãi ròng: <b>{_fmt_signed_usd(w_pnl)}</b>)"
        )
    else:
        period_lbl = "7 Ngày Qua" if is_weekly else "24 Giờ Qua"
        activity_line = (
            "• Thị trường đi ngang chưa chạm vùng điểm nổ EMA chuẩn — <b>0 phát sinh lệnh mới</b>, "
            "bảo toàn 100% vốn &amp; chờ nhịp chạm chuẩn tiếp theo."
        )

    header = (
        "🗓️ <b>BÁO CÁO TỔNG KẾT TUẦN — CHICKTRADE QUANT v3.3</b>"
        if is_weekly
        else "☀️ <b>BÁO CÁO TỔNG KẾT NGÀY — CHICKTRADE QUANT v3.3</b>"
    )
    iso_y, iso_w, _ = vn_dt.isocalendar()
    date_line = (
        f"📅 Tuần W{iso_w:02d}/{iso_y} · Chốt sáng Thứ Hai {vn_dt.strftime('%d/%m/%Y')} (07:00 VN · 24/7)"
        if is_weekly
        else f"📅 Ngày: {vn_dt.strftime('%d/%m/%Y')} (Hệ thống hoạt động 24/7 bình thường)"
    )

    return (
        f"{header}\n"
        f"{date_line}\n\n"
        f"💼 <b>Tài Sản &amp; Lãi Kép S2:</b>\n"
        f"• Số dư hiện tại (Balance): <b>${eq:,.2f}</b> ({roi_pct:+.2f}% · {_fmt_signed_usd(net_diff)} so với vốn gốc ${init_cap:,.0f})\n"
        f"• Bậc lãi kép: <b>Bậc {step_n}</b> ({mode_lbl} · 1U = ${unit_usd:,.2f}) — Còn +${rem_usd:,.2f} lên Bậc {step_n + 1} (${next_eq:,.2f})\n"
        f"• Sụt giảm tối đa (MaxDD): <b>{float(acc.get('max_dd_pct', 0.0)):.2f}%</b> · Phanh 2-SL: <b>{_brake_status_text(acc)}</b>\n\n"
        f"📊 <b>Hoạt Động {period_lbl}:</b>\n"
        f"{activity_line}\n"
        f"{open_line}\n\n"
        f"📐 <b>Nhịp Thở Vĩ Mô 4H (1U % Khung 15m):</b>\n"
        f"{_format_macro_1u_block(state.get('live_u3_rates', []))}"
    )


def main():
    parser = argparse.ArgumentParser(description="CHICKTRADE v3.0 Paper Engine")
    parser.add_argument("--once", action="store_true", help="Run a single update cycle and exit")
    parser.add_argument("--daily-recap", action="store_true", help="Force emit Daily Recap notification now")
    parser.add_argument("--weekly-recap", action="store_true", help="Force emit Weekly Recap notification now")
    args = parser.parse_args()

    with open(RULES_PATH, encoding="utf-8") as f:
        core_cfg = json.load(f)
    unit_cfg = core_cfg.get("unit_config", {})
    rules_map = {(r["coin"], r["tf"], r["setup_code"]): r for r in core_cfg["rules"]}
    top_must_keys = compute_top_must_keys(core_cfg["rules"], top_n=15)

    turbo_enabled = os.environ.get("TURBO_ENABLED", "0").strip().lower() in ("1", "true", "yes")
    now_dt = datetime.now(timezone.utc)
    now_ms = int(now_dt.timestamp() * 1000)

    state = load_or_init_state(turbo_enabled)
    filters = fetch_exchange_filters()
    closed_data, live_prices = fetch_all_market_data(now_ms)
    rates_rows, u_price_map = compute_u3_rates(closed_data, live_prices, unit_cfg)

    events: List[str] = []
    update_open_positions(state, closed_data, live_prices, events, now_ms=now_ms, rules_map=rules_map)
    signals = detect_latest_signals(closed_data, rules_map)
    open_new_positions(
        state, signals, u_price_map, filters, now_ms, events,
        top_must_keys=top_must_keys, rules_map=rules_map,
    )
    enrich_positions_metadata(state, rules_map, top_must_keys)

    acc = state["account"]
    acc["roi_pct"] = round((float(acc["equity"]) / float(acc["initial_capital"]) - 1.0) * 100.0, 2)
    state["live_u3_rates"] = rates_rows
    state["updated_at_utc"] = now_dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    state["next_m15_close_ms"] = ((now_ms // TF_MS["15m"]) + 1) * TF_MS["15m"]
    state["next_h4_close_ms"] = ((now_ms // TF_MS["4h"]) + 1) * TF_MS["4h"]

    # Daily & Weekly Recap at 00:00-00:30 UTC (07:00-07:30 VN)
    in_recap_window = now_dt.hour == 0 and now_dt.minute <= 30
    today_utc = now_dt.strftime("%Y-%m-%d")
    iso_y, iso_w, _ = now_dt.isocalendar()
    week_key = f"{iso_y}-W{iso_w:02d}"

    if args.daily_recap or (in_recap_window and state.get("last_daily_recap_date") != today_utc):
        events.append(build_recap_message(state, now_dt, is_weekly=False))
        state["last_daily_recap_date"] = today_utc

    if args.weekly_recap or (in_recap_window and now_dt.weekday() == 0 and state.get("last_weekly_recap_week") != week_key):
        events.append(build_recap_message(state, now_dt, is_weekly=True))
        state["last_weekly_recap_week"] = week_key

    chart_payload = build_chart_candles_payload(closed_data, state["updated_at_utc"], tail_n=80)

    for ev in events:
        _send_telegram(ev, chart_payload.get("candles"))

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

    with open(OUT_CHART_JSON, "w", encoding="utf-8") as f:
        json.dump(chart_payload, f, separators=(",", ":"), ensure_ascii=False)

    print(
        f"[OK] {state['updated_at_utc']} | Equity=${acc['equity']:,.2f} (Step {acc['step']} Unlimited) | "
        f"Open={len(state['open_positions'])} | Closed={state['stats']['total_trades']} | "
        f"Signals={len(signals)} | U3_Coins={len(rates_rows)} | Alerts={len(events)}"
    )


if __name__ == "__main__":
    main()
