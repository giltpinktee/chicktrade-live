"""Technical indicators for EMA 34-89-200 M5 strategy."""
import numpy as np
import pandas as pd

from config import StrategyConfig


def add_indicators(df: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    """Compute EMA 34/89/200, ATR(14), Volume MA(20), and candle anatomy features."""
    out = df.copy()
    close = out["close"]
    high = out["high"]
    low = out["low"]
    open_ = out["open"]

    out["ema34"] = close.ewm(span=cfg.ema_fast, adjust=False).mean()
    out["ema89"] = close.ewm(span=cfg.ema_mid, adjust=False).mean()
    out["ema200"] = close.ewm(span=cfg.ema_slow, adjust=False).mean()

    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    out["atr"] = tr.ewm(alpha=1.0 / cfg.atr_period, adjust=False).mean()

    out["vol_ma"] = out["volume"].rolling(cfg.vol_ma_period, min_periods=1).mean()
    out["vol_ratio"] = out["volume"] / out["vol_ma"].replace(0, np.nan)

    candle_range = (high - low).replace(0, np.nan)
    body_low = np.minimum(open_, close)
    body_high = np.maximum(open_, close)

    out["candle_range"] = high - low
    out["lower_wick_ratio"] = ((body_low - low) / candle_range).fillna(0.0)
    out["upper_wick_ratio"] = ((high - body_high) / candle_range).fillna(0.0)
    out["is_bullish"] = close > open_
    out["is_bearish"] = close < open_

    return out
