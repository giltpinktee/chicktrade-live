"""EMA 34-89-200 Wave/Touch Counter & Signal Generator (Method 1 - M5)."""
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from config import StrategyConfig
from strategy.indicators import add_indicators


@dataclass
class Signal:
    bar_idx: int
    timestamp: pd.Timestamp
    symbol: str
    direction: int          # +1 = LONG (Buy), -1 = SHORT (Sell)
    wave: int               # 1, 2, 3, 4 (4 represents 4+ counter-trend)
    setup_type: str         # e.g. "W1_TREND", "W2_TREND", "W2_TREND_PAVOL", "W3_TREND", "W4_REVERSAL"
    entry_price: float
    atr: float
    pa_vol_ok: bool
    ema_aligned: bool       # True if EMA34 > EMA89 > EMA200 (bull) or EMA34 < EMA89 < EMA200 (bear)


def generate_signals(df: pd.DataFrame, symbol: str, cfg: StrategyConfig) -> List[Signal]:
    """Scan M5 OHLCV DataFrame sequentially (zero lookahead) and emit wave-touch signals."""
    data = add_indicators(df, cfg)
    signals: List[Signal] = []

    n = len(data)
    if n < cfg.ema_slow + 10:
        return signals

    timestamps = data["timestamp"].values
    opens = data["open"].values
    highs = data["high"].values
    lows = data["low"].values
    closes = data["close"].values
    ema34 = data["ema34"].values
    ema89 = data["ema89"].values
    ema200 = data["ema200"].values
    atr = data["atr"].values
    vol_ratio = data["vol_ratio"].values
    lower_wick = data["lower_wick_ratio"].values
    upper_wick = data["upper_wick_ratio"].values
    candle_range = data["candle_range"].values

    trend = 0              # +1 uptrend, -1 downtrend
    wave_count = 0
    armed = False          # True once price has moved away from EMA34 by >= min_impulse_atr * ATR
    last_touch_idx = -999

    for i in range(cfg.ema_slow, n):
        c = closes[i]
        h = highs[i]
        l = lows[i]
        o = opens[i]
        e34 = ema34[i]
        e89 = ema89[i]
        e200 = ema200[i]
        a = atr[i]

        if np.isnan(a) or a <= 0 or np.isnan(e200):
            continue

        # 1. Macro trend check: EMA 200 breach resets trend & wave count
        new_trend = 0
        if c > e200 and e34 > e89:
            new_trend = 1
        elif c < e200 and e34 < e89:
            new_trend = -1

        if new_trend != trend:
            trend = new_trend
            wave_count = 0
            armed = False
            last_touch_idx = -999
            continue

        if trend == 0:
            continue

        # 2. Liquidity void / news spike filter
        if candle_range[i] > cfg.liquidity_void_atr * a:
            # Abnormal spike invalidates clean entry on this candle
            continue

        # 3. Arming condition: price must pull away from EMA34 (impulse leg) before next touch
        if trend == 1:
            if (h - e34) >= cfg.min_impulse_atr * a and l > e34:
                armed = True
        else:
            if (e34 - l) >= cfg.min_impulse_atr * a and h < e34:
                armed = True

        if not armed or (i - last_touch_idx) < cfg.min_bars_between_touches:
            continue

        # 4. Detect EMA34 touch
        touch_buf = cfg.touch_buffer_atr * a
        breach_buf = cfg.max_breach_atr * a
        ema_aligned = (e34 > e89 > e200) if trend == 1 else (e34 < e89 < e200)

        if trend == 1:
            touched = l <= (e34 + touch_buf) and c >= (e34 - breach_buf)
            pa_ok = (lower_wick[i] >= cfg.min_wick_ratio) or (c > o and c > e34)
        else:
            touched = h >= (e34 - touch_buf) and c <= (e34 + breach_buf)
            pa_ok = (upper_wick[i] >= cfg.min_wick_ratio) or (c < o and c < e34)

        if not touched:
            continue

        wave_count += 1
        armed = False
        last_touch_idx = i
        vol_ok = (not np.isnan(vol_ratio[i])) and (vol_ratio[i] >= cfg.min_vol_ratio)
        pa_vol_ok = bool(pa_ok and vol_ok)

        ts = pd.Timestamp(timestamps[i])
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")

        if wave_count <= 3:
            setup_type = f"W{wave_count}_TREND"
            direction = trend
        else:
            # Wave 4+: Counter-trend reversal ("bão / quét thanh khoản đảo chiều")
            setup_type = "W4_REVERSAL"
            direction = -trend

        signals.append(
            Signal(
                bar_idx=i,
                timestamp=ts,
                symbol=symbol,
                direction=direction,
                wave=min(wave_count, 4),
                setup_type=setup_type,
                entry_price=float(c),
                atr=float(a),
                pa_vol_ok=pa_vol_ok,
                ema_aligned=bool(ema_aligned),
            )
        )

    return signals
