"""MAE/MFE & Touch 1-2-3-4 Statistical Anatomy for EMA 34, EMA 89, EMA 200 on M5 (1-Year Data)."""
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import BacktestConfig, RiskConfig, StrategyConfig
from data.binance_loader import _detect_latest_available_month, download_symbol_1y
from strategy.indicators import add_indicators


def detect_ema_touches(df: pd.DataFrame, symbol: str, ema_period: int, cfg: StrategyConfig) -> List[dict]:
    """Detect touches 1, 2, 3, 4 on a specific EMA (34, 89, or 200) on M5 with zero lookahead."""
    data = add_indicators(df, cfg)
    n = len(data)
    if n < 250:
        return []

    closes = data["close"].values
    highs = data["high"].values
    lows = data["low"].values
    ema34 = data["ema34"].values
    ema89 = data["ema89"].values
    ema200 = data["ema200"].values
    atr = data["atr"].values
    crange = data["candle_range"].values
    ts_arr = data["timestamp"].values

    target_ema = ema34 if ema_period == 34 else (ema89 if ema_period == 89 else ema200)
    impulse_mult = 0.8 if ema_period == 34 else (1.0 if ema_period == 89 else 1.2)
    min_gap = 3 if ema_period == 34 else (5 if ema_period == 89 else 8)

    trend = 0
    wave_count = 0
    armed = False
    last_touch = -999
    events = []

    for i in range(210, n - 25):  # leave 24 bars ahead for MFE/MAE window
        c, h, l, a = closes[i], highs[i], lows[i], atr[i]
        e34, e89, e200, et = ema34[i], ema89[i], ema200[i], target_ema[i]
        if np.isnan(a) or a <= 0 or np.isnan(e200):
            continue

        # Define macro trend depending on which EMA is being tested
        if ema_period == 34:
            new_trend = 1 if (c > e200 and e34 > e89) else (-1 if (c < e200 and e34 < e89) else 0)
        elif ema_period == 89:
            new_trend = 1 if (c > e200 and e89 > e200) else (-1 if (c < e200 and e89 < e200) else 0)
        else:
            # For EMA 200: trend defined by EMA 89 relative to EMA 200 (price hasn't broken deeply through 200)
            if e89 > e200 and c >= e200 - 0.8 * a:
                new_trend = 1
            elif e89 < e200 and c <= e200 + 0.8 * a:
                new_trend = -1
            else:
                new_trend = 0

        if new_trend != trend:
            trend = new_trend
            wave_count = 0
            armed = False
            last_touch = -999
            continue

        if trend == 0:
            continue

        if crange[i] > cfg.liquidity_void_atr * a:
            continue

        # Arming: price must pull away from target EMA before counting next touch
        if trend == 1:
            if (h - et) >= impulse_mult * a and l > et:
                armed = True
        else:
            if (et - l) >= impulse_mult * a and h < et:
                armed = True

        if not armed or (i - last_touch) < min_gap:
            continue

        touch_buf = 0.20 * a
        breach_buf = 0.65 * a
        if trend == 1:
            touched = (l <= et + touch_buf) and (c >= et - breach_buf)
        else:
            touched = (h >= et - touch_buf) and (c <= et + breach_buf)

        if not touched:
            continue

        wave_count += 1
        armed = False
        last_touch = i
        if wave_count > 4:
            continue  # Focus strictly on Touches 1, 2, 3, 4

        unit_size = max(0.5 * a, c * 1e-5)
        entry = c

        # Measure short-window (12 bars = 1 hour on M5) bounce anatomy:
        # 1) Raw MAE & MFE in 1h window (12 bars)
        # 2) MFE BEFORE hitting SL (for SL = 3U, 5U, 8U) within 1h (12 bars) and 2h (24 bars)
        # 3) Bars to reach peak MFE before SL=5U
        win_1h_h = highs[i + 1 : i + 13]
        win_1h_l = lows[i + 1 : i + 13]
        if trend == 1:
            fav_1h = (win_1h_h - entry) / unit_size
            adv_1h = (entry - win_1h_l) / unit_size
        else:
            fav_1h = (entry - win_1h_l) / unit_size
            adv_1h = (win_1h_h - entry) / unit_size

        raw_mfe_1h = float(max(0.0, fav_1h.max()))
        raw_mae_1h = float(max(0.0, adv_1h.max()))

        # Compute MFE before hitting SL=5U (and SL=3U, SL=8U) within 12 bars (1h)
        def mfe_before_sl(sl_u: float, horizon: int = 12):
            wh = highs[i + 1 : i + 1 + horizon]
            wl = lows[i + 1 : i + 1 + horizon]
            best_fav = 0.0
            peak_bar = 0
            hit_sl = False
            for k in range(len(wh)):
                if trend == 1:
                    adv = (entry - wl[k]) / unit_size
                    fav = (wh[k] - entry) / unit_size
                else:
                    adv = (wh[k] - entry) / unit_size
                    fav = (entry - wl[k]) / unit_size
                if adv >= sl_u:
                    hit_sl = True
                    break
                if fav > best_fav:
                    best_fav = float(fav)
                    peak_bar = k + 1
            return best_fav, hit_sl, peak_bar

        mfe_b_sl3, sl3_hit, _ = mfe_before_sl(3.0, 12)
        mfe_b_sl5, sl5_hit, peak_bar_5 = mfe_before_sl(5.0, 12)
        mfe_b_sl8, sl8_hit, _ = mfe_before_sl(8.0, 12)
        mfe_b_sl5_2h, sl5_hit_2h, _ = mfe_before_sl(5.0, 24)

        # Fee in units (Maker 0.02% + Taker 0.05% = 0.07% round-trip conservative)
        fee_u = (entry * 0.0007) / unit_size

        events.append(
            {
                "symbol": symbol,
                "ema": f"EMA{ema_period}",
                "touch": wave_count,
                "direction": trend,
                "unit_pct": unit_size / entry * 100.0,
                "fee_u": fee_u,
                "raw_mfe_1h": raw_mfe_1h,
                "raw_mae_1h": raw_mae_1h,
                "mfe_b_sl3": mfe_b_sl3,
                "sl3_hit": sl3_hit,
                "mfe_b_sl5": mfe_b_sl5,
                "sl5_hit": sl5_hit,
                "peak_bar_5": peak_bar_5,
                "mfe_b_sl8": mfe_b_sl8,
                "sl8_hit": sl8_hit,
                "mfe_b_sl5_2h": mfe_b_sl5_2h,
                # Hit TP before SL=5U within 1h
                "hit_tp3_b_sl5": mfe_b_sl5 >= 3.0,
                "hit_tp5_b_sl5": mfe_b_sl5 >= 5.0,
                "hit_tp8_b_sl5": mfe_b_sl5 >= 8.0,
                # Hit TP before SL=8U within 1h
                "hit_tp5_b_sl8": mfe_b_sl8 >= 5.0,
                "hit_tp8_b_sl8": mfe_b_sl8 >= 8.0,
            }
        )
    return events


def summarize_anatomy(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (ema, touch), g in df.groupby(["ema", "touch"]):
        n = len(g)
        rows.append(
            {
                "EMA": ema,
                "Cham": int(touch),
                "So_Lenh": n,
                "MAE_TB_1h": g["raw_mae_1h"].mean(),
                "MAE_Med_1h": g["raw_mae_1h"].median(),
                "MFE_TB_1h": g["raw_mfe_1h"].mean(),
                "MFE_Med_1h": g["raw_mfe_1h"].median(),
                "TP_TB_TruocSL5": g["mfe_b_sl5"].mean(),
                "TP_Med_TruocSL5": g["mfe_b_sl5"].median(),
                "P75_TP_TruocSL5": g["mfe_b_sl5"].quantile(0.75),
                "Dinh_SL5_%": g["sl5_hit"].mean() * 100.0,
                "Dinh_SL3_%": g["sl3_hit"].mean() * 100.0,
                "Dinh_SL8_%": g["sl8_hit"].mean() * 100.0,
                "Dat_TP3_%": g["hit_tp3_b_sl5"].mean() * 100.0,
                "Dat_TP5_%": g["hit_tp5_b_sl5"].mean() * 100.0,
                "Dat_TP8_%": g["hit_tp8_b_sl5"].mean() * 100.0,
                "Nen_Dat_Dinh": g.loc[g["mfe_b_sl5"] > 0, "peak_bar_5"].mean(),
                "Phi_TB_U": g["fee_u"].mean(),
            }
        )
    return pd.DataFrame(rows)


def main():
    cfg = BacktestConfig()
    end_ym = _detect_latest_available_month("BTCUSDT", "5m")

    crypto_events = []
    gold_events = []

    for sym in cfg.symbols:
        df = download_symbol_1y(sym, interval="5m", months=12, market="futures", end_ym=end_ym)
        for ema_p in [34, 89, 200]:
            crypto_events.extend(detect_ema_touches(df, sym, ema_p, cfg.strategy))

    df_gold = download_symbol_1y("PAXGUSDT", interval="5m", months=12, market="spot", end_ym=end_ym)
    for ema_p in [34, 89, 200]:
        gold_events.extend(detect_ema_touches(df_gold, "PAXGUSDT", ema_p, cfg.strategy))

    cdf = pd.DataFrame(crypto_events)
    gdf = pd.DataFrame(gold_events)

    print("=" * 110)
    print("THONG KE ANATOMY CHAM 1-2-3-4 CUA EMA 34 / 89 / 200 — KHUNG M5 (CUA SO NGAN 1 GIO = 12 NEN M5)")
    print("Don vi: Unit (1 Unit = 0.5 * ATR M5). BTC+ETH+BNB (1 nam: 2025-11 -> 2026-10)")
    print("=" * 110)
    c_sum = summarize_anatomy(cdf)
    print(c_sum.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    print("\n" + "=" * 110)
    print("DOI CHIEU VANG (PAXGUSDT M5 — 1 NAM, CUA SO NGAN 1 GIO = 12 NEN M5)")
    print("=" * 110)
    g_sum = summarize_anatomy(gdf)
    print(g_sum.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    # Save CSVs
    out_dir = os.path.join(ROOT, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    c_sum.to_csv(os.path.join(out_dir, "ema_34_89_200_anatomy_crypto_m5.csv"), index=False)
    g_sum.to_csv(os.path.join(out_dir, "ema_34_89_200_anatomy_gold_m5.csv"), index=False)
    print(f"\nSaved CSVs to {out_dir}")


if __name__ == "__main__":
    main()
