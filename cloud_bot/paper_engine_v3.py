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
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
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

SIX_COINS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "LINKUSDT", "XRPUSDT", "SOLUSDT"]
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


def fetch_symbol_klines(sym: str, interval: str, limit: int = 500) -> pd.DataFrame:
    """Fetch Binance USD-M Futures klines and return strictly closed + live metadata."""
    raw = _fapi_get("/fapi/v1/klines", params={"symbol": sym, "interval": interval, "limit": limit})
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
    return pd.DataFrame(rows)


def fetch_all_market_data(now_ms: int) -> Tuple[Dict[str, Dict[str, pd.DataFrame]], Dict[str, float]]:
    """Parallel-fetch 15m/30m/1h/4h klines across 6 coins and split closed bars vs live price."""
    tasks = [(sym, tf, 300 if tf == "4h" else 500) for sym in SIX_COINS for tf in ("15m", "30m", "1h", "4h")]
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
    """Return (cooldown_mask, atr_ratio) matching backtest 2-layer Storm Guard."""
    high, low, close, open_ = df["high"], df["low"], df["close"], df["open"]
    atr = compute_atr14(df)
    atr_ma50 = atr.rolling(50, min_periods=10).mean()
    atr_ratio = (atr / atr_ma50.replace(0, np.nan)).fillna(1.0)
    crange = (high - low).replace(0, np.nan)
    body = (close - open_).abs()
    wick_frac = (1.0 - body / crange).fillna(0.0)
    sweep_candle = (crange >= 2.8 * atr) | ((crange >= 2.0 * atr) & (wick_frac >= 0.70))
    net_move = (close - close.shift(10)).abs()
    path_len = close.diff().abs().rolling(10, min_periods=10).sum().replace(0, np.nan)
    er = (net_move / path_len).fillna(0.5)
    raw_storm = (atr_ratio >= 1.8) | sweep_candle | ((er < 0.15) & (atr_ratio >= 1.20))
    cd_mask = (raw_storm.astype(int).rolling(cooldown_bars, min_periods=1).max() > 0).values
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
    fees = (entry * qty * MAKER_FEE) + (slip_px * qty * (MAKER_FEE if exit_maker else TAKER_FEE))
    return float(gross - fees)


def _apply_s2_step(state: dict) -> None:
    """Update S2_LOCKED_HYSTERESIS_10PCT (50 steps, +8%/step, -10% hysteresis)."""
    acc = state["account"]
    eq = float(acc["equity"])
    step = int(acc["step"])
    tier_cap = float(acc["tier_capital"])
    while step < 50 and eq >= 10_000.0 * (1.08 ** (step + 1)):
        step += 1
        tier_cap = 10_000.0 * (1.08 ** step)
    while step > 0 and eq < tier_cap * 0.90:
        step -= 1
        tier_cap = 10_000.0 * (1.08 ** step)
    acc["step"] = step
    acc["tier_capital"] = round(tier_cap, 2)


def _send_telegram(text: str) -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        return
    webapp_url = os.environ.get("WEBAPP_URL", "").strip()
    payload: Dict[str, Any] = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    if webapp_url.startswith("https://"):
        payload["reply_markup"] = {
            "inline_keyboard": [[{"text": "📊 Mở Dashboard Live & Bảng 1U", "web_app": {"url": webapp_url}}]]
        }
    try:
        requests.post(f"https://api.telegram.org/bot{token}/sendMessage", json=payload, timeout=10)
    except Exception as exc:
        print(f"[WARN] Telegram notify failed: {exc}")


def _init_state(turbo_enabled: bool) -> dict:
    return {
        "version": "v3.0-u3-htf-macro-4h",
        "updated_at_utc": "",
        "next_m15_close_ms": 0,
        "next_h4_close_ms": 0,
        "account": {
            "initial_capital": 10_000.0,
            "equity": 10_000.0,
            "peak_equity": 10_000.0,
            "max_dd_pct": 0.0,
            "roi_pct": 0.0,
            "step": 0,
            "tier_capital": 10_000.0,
            "sizing_policy": "S2_LOCKED_HYSTERESIS_10PCT (50 Bậc · Cap $300k/$150k)",
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


def enrich_positions_metadata(
    state: dict,
    rules_map: Dict[Tuple[str, str, str], dict],
    top_must_keys: set,
) -> None:
    """Backfill profit_wr_pct, wr_pct, and is_must_order on existing open & closed records."""
    for group_key in ("open_positions", "closed_trades"):
        for item in state.get(group_key, []):
            k = (item.get("symbol"), item.get("tf"), item.get("setup_code"))
            r = rules_map.get(k)
            if r is not None:
                p = r.get("prior_1y", {})
                item["profit_wr_pct"] = round(float(p.get("profit_wr_pct", 0.0)), 1)
                item["wr_pct"] = round(float(p.get("wr_pct", 0.0)), 1)
                item["is_must_order"] = bool(k in top_must_keys)


def load_or_init_state(turbo_enabled: bool) -> dict:
    if os.path.exists(OUT_JSON):
        try:
            with open(OUT_JSON, encoding="utf-8") as f:
                st = json.load(f)
            st["account"]["turbo_enabled"] = turbo_enabled
            return st
        except Exception as exc:
            print(f"[WARN] Could not parse existing {OUT_JSON}, starting fresh: {exc}")
    return _init_state(turbo_enabled)


def update_open_positions(
    state: dict,
    closed_data: Dict[str, Dict[str, pd.DataFrame]],
    live_prices: Dict[str, float],
    events: List[str],
) -> None:
    """Check closed candles for SL / TP1 / TP2 / Leg2_SL hits on all open positions."""
    acc = state["account"]
    stats = state["stats"]
    still_open: List[dict] = []

    for pos in state["open_positions"]:
        sym, tf = pos["symbol"], pos["tf"]
        df = closed_data[sym][tf]
        d = 1 if pos["direction"] == "LONG" else -1
        new_bars = df[df["close_time_ms"] > int(pos["last_checked_close_ms"])]

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
                        must_tag = " · 🔥 <b>THẤY LÀ QUẤT</b>" if pos.get("is_must_order") else ""
                        events.append(
                            f"🎯 <b>CHẠM TP1 — DỜI SL CHÂN 2</b> | {sym} ({tf}) {pos['direction']}{must_tag}\n"
                            f"Thắng tiền: <b>{pos.get('profit_wr_pct', 0):.1f}%</b> | WR: <b>{pos.get('wr_pct', 0):.1f}%</b>\n"
                            f"Chốt 50% @ <code>{pos['tp1_price']:.4f}</code> (+${pnl1:.2f}) · Dời SL → <code>{pos['active_sl_price']:.4f}</code>"
                        )
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
        rem_qty = pos["qty"] * (0.5 if pos["tp1_hit"] else 1.0)
        exit_maker = closed_outcome in ("TP1_FULL", "1R_FULL", "TP2")
        pnl_rem = _calc_leg_pnl(pos["entry_price"], close_px, rem_qty, d, exit_maker=exit_maker)
        tot_pnl = round(float(pos.get("realized_pnl_usd", 0.0)) + pnl_rem, 2)

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

        if closed_outcome == "SL" and pos["base_vol"] > 0.45:
            acc["portfolio_consec_sl"] = int(acc.get("portfolio_consec_sl", 0)) + 1
            if acc["portfolio_consec_sl"] >= 2:
                acc["portfolio_consec_sl"] = 0
                acc["shadow_paused"] = True
                acc["shadow_paused_until_ms"] = int(pos["last_checked_close_ms"]) + 72 * 3_600_000
                events.append("🛑 <b>PHANH 2-SL CORE → SHADOW-WIN</b> | Tạm dừng tiền thật đợi 1 lệnh thắng nháp (tối đa 72h).")
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

        closed_rec = {
            "id": pos["id"],
            "symbol": sym,
            "tf": tf,
            "setup_code": pos["setup_code"],
            "direction": pos["direction"],
            "entry_price": pos["entry_price"],
            "exit_price": round(close_px, 6),
            "qty": pos["qty"],
            "vol_mult": pos["vol_mult"],
            "profit_wr_pct": pos.get("profit_wr_pct", 0.0),
            "wr_pct": pos.get("wr_pct", 0.0),
            "is_must_order": bool(pos.get("is_must_order", False)),
            "outcome": closed_outcome,
            "pnl_usd": tot_pnl,
            "opened_utc": pos["opened_utc"],
            "closed_utc": close_ts_str,
        }
        state["closed_trades"] = ([closed_rec] + state["closed_trades"])[:500]
        state["equity_curve"].append({"ts": close_ts_str, "eq": acc["equity"], "dd": round(dd_now, 2)})
        state["equity_curve"] = state["equity_curve"][-1000:]

        emoji = "✅" if tot_pnl > 0 else "❌"
        must_tag = " · 🔥 <b>THẤY LÀ QUẤT</b>" if pos.get("is_must_order") else ""
        events.append(
            f"{emoji} <b>ĐÓNG LỆNH ({closed_outcome})</b> | {sym} ({tf}) {pos['setup_code']} {pos['direction']}{must_tag}\n"
            f"Thắng tiền: <b>{pos.get('profit_wr_pct', 0):.1f}%</b> | WR: <b>{pos.get('wr_pct', 0):.1f}%</b>\n"
            f"PnL: <b>{tot_pnl:+.2f} USD</b> · Equity: <b>${acc['equity']:,.2f}</b> (Bậc {acc['step']}/50)"
        )

    state["open_positions"] = still_open


def open_new_positions(
    state: dict,
    signals: List[dict],
    u_price_map: Dict[str, Dict[str, float]],
    filters: Dict[str, Dict[str, float]],
    now_ms: int,
    events: List[str],
    top_must_keys: Optional[set] = None,
) -> None:
    """Open qualifying signals with 20x Cross Margin, stepSize floor, minNotional & S2 liq cap."""
    acc = state["account"]
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if state["daily_tracker"]["date"] != today_str:
        state["daily_tracker"] = {"date": today_str, "trades": 0, "pnl": 0.0, "start_eq": float(acc["equity"])}

    dt = state["daily_tracker"]
    seen_keys = set(state.get("processed_signal_keys", []))

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
            else:
                continue

        if dt["trades"] >= 10 or dt["pnl"] <= -0.05 * dt["start_eq"]:
            continue
        if len(state["open_positions"]) >= 5 or any(p["symbol"] == sym for p in state["open_positions"]):
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
        pwr_pct = round(float(prior.get("profit_wr_pct", 0.0)), 1)
        wr_pct = round(float(prior.get("wr_pct", 0.0)), 1)
        is_must = bool(top_must_keys and (sym, tf, sig["setup_type"]) in top_must_keys)
        pos_obj = {
            "id": pos_id,
            "symbol": sym,
            "tf": tf,
            "setup_code": sig["setup_type"],
            "direction": sig["direction"],
            "entry_price": round(p_sig, 6),
            "current_price": round(p_sig, 6),
            "qty": qty,
            "notional_usd": round(notional, 2),
            "vol_mult": round(eff_m, 2),
            "base_vol": float(rule["adaptive_volume"]["init_vol_mult"]),
            "profit_wr_pct": pwr_pct,
            "wr_pct": wr_pct,
            "is_must_order": is_must,
            "buffer_used": int(cur_buf),
            "u_price": round(u, 6),
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
            "last_checked_close_ms": sig["signal_close_ms"],
        }
        state["open_positions"].append(pos_obj)
        dt["trades"] = int(dt["trades"]) + 1
        must_tag = " · 🔥 <b>THẤY LÀ QUẤT</b>" if is_must else ""
        events.append(
            f"🚀 <b>MỞ LỆNH MỚI</b> | {sym} ({tf}) <b>{sig['setup_type']}</b> [{sig['direction']}]{must_tag}\n"
            f"Thắng tiền: <b>{pwr_pct:.1f}%</b> | WR: <b>{wr_pct:.1f}%</b>\n"
            f"Entry: <code>{p_sig:.4f}</code> · Qty: <code>{qty}</code> (~${notional:,.0f} @ 20×)\n"
            f"SL: <code>{sl_px:.4f}</code> · TP1: <code>{tp1_px:.4f}</code> · TP2: <code>{tp2_px:.4f}</code>"
        )

    state["processed_signal_keys"] = sorted(seen_keys)[-500:]


def main():
    parser = argparse.ArgumentParser(description="CHICKTRADE v3.0 Paper Engine")
    parser.add_argument("--once", action="store_true", help="Run a single update cycle and exit")
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
    enrich_positions_metadata(state, rules_map, top_must_keys)
    filters = fetch_exchange_filters()
    closed_data, live_prices = fetch_all_market_data(now_ms)
    rates_rows, u_price_map = compute_u3_rates(closed_data, live_prices, unit_cfg)

    events: List[str] = []
    update_open_positions(state, closed_data, live_prices, events)
    signals = detect_latest_signals(closed_data, rules_map)
    open_new_positions(state, signals, u_price_map, filters, now_ms, events, top_must_keys=top_must_keys)

    acc = state["account"]
    acc["roi_pct"] = round((float(acc["equity"]) / float(acc["initial_capital"]) - 1.0) * 100.0, 2)
    state["live_u3_rates"] = rates_rows
    state["updated_at_utc"] = now_dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    state["next_m15_close_ms"] = ((now_ms // TF_MS["15m"]) + 1) * TF_MS["15m"]
    state["next_h4_close_ms"] = ((now_ms // TF_MS["4h"]) + 1) * TF_MS["4h"]

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

    for ev in events:
        _send_telegram(ev)

    print(
        f"[OK] {state['updated_at_utc']} | Equity=${acc['equity']:,.2f} (Step {acc['step']}/50) | "
        f"Open={len(state['open_positions'])} | Closed={state['stats']['total_trades']} | "
        f"Signals={len(signals)} | U3_Coins={len(rates_rows)}"
    )


if __name__ == "__main__":
    main()
