"""Download & cache 6-year (2021-01 -> 2026-10) Binance USD-M Futures 5m klines & real 8h funding rates for 6 coins.

Uses data.binance.vision public archives (fast, zero geo-block, exact Binance USD-M Futures data).
Caches into e:/CHICKTRADE/data/cache/6y/ (gitignored via data/cache/).
"""
from concurrent.futures import ThreadPoolExecutor
import io
import os
import zipfile
from typing import List, Tuple

import pandas as pd
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE_DIR = os.path.join(ROOT, "data", "cache")
OUT_DIR = os.path.join(CACHE_DIR, "6y")
os.makedirs(OUT_DIR, exist_ok=True)

SIX_COINS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "LINKUSDT", "XRPUSDT", "SOLUSDT"]


def _dl_5m_month(args: Tuple[str, int, int]) -> pd.DataFrame:
    sym, y, m = args
    url = f"https://data.binance.vision/data/futures/um/monthly/klines/{sym}/5m/{sym}-5m-{y:04d}-{m:02d}.zip"
    try:
        r = requests.get(url, timeout=25)
        if r.status_code != 200:
            return pd.DataFrame()
        with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
            with zf.open(zf.namelist()[0]) as f:
                df = pd.read_csv(f, usecols=[0, 1, 2, 3, 4, 5], header=None)
        # Drop header row if present
        if str(df.iloc[0, 0]).lower().startswith("open"):
            df = df.iloc[1:].reset_index(drop=True)
        df.columns = ["open_time_ms", "open", "high", "low", "close", "volume"]
        df["open_time_ms"] = df["open_time_ms"].astype("int64")
        for c in ("open", "high", "low", "close", "volume"):
            df[c] = df[c].astype("float64")
        return df
    except Exception:
        return pd.DataFrame()


def _dl_funding_month(args: Tuple[str, int, int]) -> pd.DataFrame:
    sym, y, m = args
    url = f"https://data.binance.vision/data/futures/um/monthly/fundingRate/{sym}/{sym}-fundingRate-{y:04d}-{m:02d}.zip"
    try:
        r = requests.get(url, timeout=20)
        if r.status_code != 200:
            return pd.DataFrame()
        with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
            with zf.open(zf.namelist()[0]) as f:
                df = pd.read_csv(f)
        # Columns: calc_time, funding_interval_hours, last_funding_rate
        t_col = df.columns[0]
        r_col = df.columns[-1]
        out = pd.DataFrame({
            "calc_time_ms": df[t_col].astype("int64"),
            "funding_rate": df[r_col].astype("float64"),
        })
        return out
    except Exception:
        return pd.DataFrame()


def build_coin_6y(sym: str) -> None:
    out_5m = os.path.join(OUT_DIR, f"{sym}_5m_2021_2026.csv.gz")
    out_fr = os.path.join(OUT_DIR, f"{sym}_funding_2021_2026.csv.gz")

    if not os.path.exists(out_5m):
        # Download 2021-01 -> 2023-10 from Binance Vision, then concat with existing 2023-11 -> 2026-10 cache!
        months_old: List[Tuple[str, int, int]] = []
        for y in (2021, 2022, 2023):
            end_m = 10 if y == 2023 else 12
            for m in range(1, end_m + 1):
                months_old.append((sym, y, m))
        with ThreadPoolExecutor(max_workers=12) as pool:
            parts = [d for d in pool.map(_dl_5m_month, months_old) if len(d) > 0]

        # Load existing 2023-11 -> 2026-10 caches
        p24 = os.path.join(CACHE_DIR, f"{sym}_futures_5m_202510_24m.csv.gz")
        p12 = os.path.join(CACHE_DIR, f"{sym}_futures_5m_202610_12m.csv.gz")
        for p in (p24, p12):
            d = pd.read_csv(p)
            ts = pd.to_datetime(d["timestamp"], utc=True)
            d["open_time_ms"] = (ts.astype("int64") // 10**6).astype("int64")
            parts.append(d[["open_time_ms", "open", "high", "low", "close", "volume"]])

        full = pd.concat(parts, ignore_index=True)
        full = full.drop_duplicates("open_time_ms").sort_values("open_time_ms").reset_index(drop=True)
        full.to_csv(out_5m, index=False, compression="gzip")
        t0 = pd.to_datetime(full["open_time_ms"].iloc[0], unit="ms", utc=True).strftime("%Y-%m-%d")
        t1 = pd.to_datetime(full["open_time_ms"].iloc[-1], unit="ms", utc=True).strftime("%Y-%m-%d")
        print(f"  [5m OK] {sym}: {len(full):,} bars ({t0} -> {t1})")
    else:
        print(f"  [5m CACHE] {sym} already cached")

    if not os.path.exists(out_fr):
        months_fr: List[Tuple[str, int, int]] = []
        for y in range(2021, 2027):
            end_m = 10 if y == 2026 else 12
            for m in range(1, end_m + 1):
                months_fr.append((sym, y, m))
        with ThreadPoolExecutor(max_workers=12) as pool:
            parts_fr = [d for d in pool.map(_dl_funding_month, months_fr) if len(d) > 0]
        fr_df = pd.concat(parts_fr, ignore_index=True) if parts_fr else pd.DataFrame(columns=["calc_time_ms", "funding_rate"])
        if len(fr_df) > 0:
            fr_df = fr_df.drop_duplicates("calc_time_ms").sort_values("calc_time_ms").reset_index(drop=True)
        fr_df.to_csv(out_fr, index=False, compression="gzip")
        print(f"  [FR OK] {sym}: {len(fr_df):,} real 8h funding prints")
    else:
        print(f"  [FR CACHE] {sym} already cached")


def main():
    print("Downloading 6-year (2021-01 -> 2026-10) Binance Futures 5m + Real Funding for 6 coins...")
    for sym in SIX_COINS:
        build_coin_6y(sym)
    print("[DONE] All 6 coins 6-year data + real funding ready!")


if __name__ == "__main__":
    main()
