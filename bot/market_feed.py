"""Binance USD-M Futures Public Market Data Feed (zero-repaint closed klines + live prices)."""
from __future__ import annotations

import json
import time
from typing import Dict, Iterable, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

SUPPORTED_INTERVALS = ("5m", "15m", "30m", "1h", "4h")


class BinanceMarketFeed:
    """Fetches closed klines and live ticker/mark prices from Binance USD-M Futures Public API."""

    def __init__(
        self,
        base_url: str = "https://fapi.binance.com",
        timeout: float = 10.0,
        max_retries: int = 3,
        retry_backoff: float = 0.75,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)
        self.max_retries = max(1, int(max_retries))
        self.retry_backoff = float(retry_backoff)
        self._exchange_info_cache: Optional[Dict[str, dict]] = None

    def _get_json(self, path: str, params: Optional[Dict[str, object]] = None):
        qs = f"?{urlencode(params)}" if params else ""
        url = f"{self.base_url}{path}{qs}"
        last_err: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            req = Request(url, headers={"User-Agent": "CHICKTRADE-Bot/2.0", "Accept": "application/json"})
            try:
                with urlopen(req, timeout=self.timeout) as resp:
                    payload = resp.read().decode("utf-8")
                    return json.loads(payload)
            except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_err = exc
                if attempt < self.max_retries:
                    time.sleep(self.retry_backoff * attempt)
        raise RuntimeError(f"Binance Public API request failed after {self.max_retries} attempts: {url} ({last_err})")

    def fetch_closed_klines(self, symbol: str, interval: str, limit: int = 300) -> pd.DataFrame:
        """Fetch the last `limit` strictly CLOSED klines (drops the still-forming candle for zero-repaint)."""
        if interval not in SUPPORTED_INTERVALS:
            raise ValueError(f"Unsupported interval={interval!r}; expected one of {SUPPORTED_INTERVALS}")

        # Request limit + 1 so dropping the forming candle still leaves `limit` closed bars
        req_limit = min(int(limit) + 1, 1500)
        raw = self._get_json(
            "/fapi/v1/klines",
            {"symbol": symbol.upper(), "interval": interval, "limit": req_limit},
        )
        if not raw or len(raw) < 2:
            raise RuntimeError(f"Insufficient kline data returned for {symbol} {interval}")

        # Zero-repaint rule: always drop the final candle (currently forming on Binance)
        closed_raw = raw[:-1]
        if len(closed_raw) > limit:
            closed_raw = closed_raw[-limit:]

        records = []
        for row in closed_raw:
            records.append(
                {
                    "timestamp": pd.to_datetime(int(row[0]), unit="ms", utc=True),
                    "open": float(row[1]),
                    "high": float(row[2]),
                    "low": float(row[3]),
                    "close": float(row[4]),
                    "volume": float(row[5]),
                    "close_time": pd.to_datetime(int(row[6]), unit="ms", utc=True),
                }
            )
        df = pd.DataFrame(records)
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    def fetch_last_price(self, symbol: str) -> float:
        data = self._get_json("/fapi/v1/ticker/price", {"symbol": symbol.upper()})
        return float(data["price"])

    def fetch_all_prices(self, symbols: Optional[Iterable[str]] = None) -> Dict[str, float]:
        data = self._get_json("/fapi/v1/ticker/price")
        wanted = {s.upper() for s in symbols} if symbols is not None else None
        prices: Dict[str, float] = {}
        for item in data:
            sym = item["symbol"].upper()
            if wanted is None or sym in wanted:
                prices[sym] = float(item["price"])
        return prices

    def fetch_mark_price(self, symbol: str) -> float:
        data = self._get_json("/fapi/v1/premiumIndex", {"symbol": symbol.upper()})
        return float(data["markPrice"])

    def fetch_exchange_info(self, symbols: Optional[Iterable[str]] = None, force_refresh: bool = False) -> Dict[str, dict]:
        """Fetch symbol precision filters (stepSize, tickSize, minQty, minNotional)."""
        if self._exchange_info_cache is None or force_refresh:
            data = self._get_json("/fapi/v1/exchangeInfo")
            parsed: Dict[str, dict] = {}
            for s in data.get("symbols", []):
                sym = s["symbol"].upper()
                filters = {f["filterType"]: f for f in s.get("filters", [])}
                lot = filters.get("LOT_SIZE", {})
                price_f = filters.get("PRICE_FILTER", {})
                min_not = filters.get("MIN_NOTIONAL", {})
                parsed[sym] = {
                    "symbol": sym,
                    "stepSize": float(lot.get("stepSize", 0.001)),
                    "minQty": float(lot.get("minQty", 0.001)),
                    "tickSize": float(price_f.get("tickSize", 0.01)),
                    "minNotional": float(min_not.get("notional", 5.0)),
                    "quantityPrecision": int(s.get("quantityPrecision", 3)),
                    "pricePrecision": int(s.get("pricePrecision", 2)),
                }
            self._exchange_info_cache = parsed

        if symbols is None:
            return dict(self._exchange_info_cache)
        wanted = {s.upper() for s in symbols}
        return {k: v for k, v in self._exchange_info_cache.items() if k in wanted}
