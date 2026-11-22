"""Free Binance historical 5m OHLCV downloader (data.binance.vision + REST fallback)."""
import io
import os
import time
import zipfile
from datetime import datetime, timezone
from typing import List, Optional, Tuple

import pandas as pd
import requests

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

VISION_SPOT_URL = "https://data.binance.vision/data/spot/monthly/klines/{symbol}/{interval}/{symbol}-{interval}-{year}-{month:02d}.zip"
VISION_FUTURES_URL = "https://data.binance.vision/data/futures/um/monthly/klines/{symbol}/{interval}/{symbol}-{interval}-{year}-{month:02d}.zip"
REST_FUTURES_URL = "https://fapi.binance.com/fapi/v1/klines"
REST_SPOT_URL = "https://api.binance.com/api/v3/klines"

KLINE_COLS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore",
]


def _month_list(end_year: int, end_month: int, months: int) -> List[Tuple[int, int]]:
    result = []
    y, m = end_year, end_month
    for _ in range(months):
        result.append((y, m))
        m -= 1
        if m == 0:
            m = 12
            y -= 1
    result.reverse()
    return result


def _detect_latest_available_month(symbol: str = "BTCUSDT", interval: str = "5m") -> Tuple[int, int]:
    """Probe data.binance.vision backwards from real current UTC time to find latest published monthly zip."""
    # Check Binance server time first so local clock skew doesn't break anything
    now_utc = datetime.now(timezone.utc)
    try:
        r = requests.get("https://api.binance.com/api/v3/time", timeout=8)
        if r.status_code == 200:
            server_ms = r.json()["serverTime"]
            now_utc = datetime.fromtimestamp(server_ms / 1000.0, tz=timezone.utc)
    except Exception:
        pass

    y, m = now_utc.year, now_utc.month
    for _ in range(24):
        url = VISION_FUTURES_URL.format(symbol=symbol, interval=interval, year=y, month=m)
        try:
            resp = requests.head(url, timeout=8)
            if resp.status_code == 200:
                return y, m
        except Exception:
            pass
        m -= 1
        if m == 0:
            m = 12
            y -= 1
    return 2025, 12


def _parse_klines_csv(raw_bytes: bytes) -> pd.DataFrame:
    text_head = raw_bytes[:80].decode("utf-8", errors="ignore")
    has_header = text_head.startswith("open_time")
    df = pd.read_csv(
        io.BytesIO(raw_bytes),
        header=0 if has_header else None,
        names=None if has_header else KLINE_COLS,
    )
    df = df.iloc[:, :6].copy()
    df.columns = ["open_time", "open", "high", "low", "close", "volume"]
    # Binance spot switched some 2025+ files to microseconds; normalize to milliseconds
    ts = pd.to_numeric(df["open_time"], errors="coerce")
    ts = ts.where(ts < 10**14, ts // 1000)
    df["timestamp"] = pd.to_datetime(ts, unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[["timestamp", "open", "high", "low", "close", "volume"]].dropna()


def download_symbol_1y(
    symbol: str,
    interval: str = "5m",
    months: int = 12,
    market: str = "futures",
    end_ym: Optional[Tuple[int, int]] = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Download and cache 1 year of M5 klines from Binance Data Vision (free, no API key)."""
    if end_ym is None:
        end_ym = _detect_latest_available_month("BTCUSDT", interval)

    end_y, end_m = end_ym
    cache_path = os.path.join(CACHE_DIR, f"{symbol}_{market}_{interval}_{end_y}{end_m:02d}_{months}m.csv.gz")

    if os.path.exists(cache_path) and not force_refresh:
        df = pd.read_csv(cache_path, parse_dates=["timestamp"])
        if df["timestamp"].dt.tz is None:
            df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
        return df

    url_tpl = VISION_FUTURES_URL if market == "futures" else VISION_SPOT_URL
    frames = []
    session = requests.Session()

    for y, m in _month_list(end_y, end_m, months):
        url = url_tpl.format(symbol=symbol, interval=interval, year=y, month=m)
        resp = session.get(url, timeout=30)
        if resp.status_code != 200 and market == "futures":
            # Fallback to spot archive if futures monthly file unavailable (e.g. PAXGUSDT)
            alt_url = VISION_SPOT_URL.format(symbol=symbol, interval=interval, year=y, month=m)
            resp = session.get(alt_url, timeout=30)
        if resp.status_code != 200:
            print(f"  [WARN] Missing {symbol} {y}-{m:02d} (HTTP {resp.status_code})")
            continue

        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            csv_name = zf.namelist()[0]
            raw = zf.read(csv_name)
            frames.append(_parse_klines_csv(raw))
        time.sleep(0.05)

    if not frames:
        raise RuntimeError(f"Failed to download any data for {symbol} ({market}, {interval})")

    full = pd.concat(frames, ignore_index=True)
    full = full.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    full.to_csv(cache_path, index=False, compression="gzip")
    return full
