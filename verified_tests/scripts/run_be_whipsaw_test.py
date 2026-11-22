"""Test Whipsaw Rate after TP1: Compare SL=Entry+Fee vs SL=Entry vs SL=Entry-1U vs Entry-2U vs Keep SL-5U."""
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import BacktestConfig
from data.binance_loader import _detect_latest_available_month, download_symbol_1y
from run_3ema_complete import generate_3ema_signals
from run_research_mtf import resample_ohlcv
from strategy.ema_wave import Signal


def sim_leg2_policy(
    df: pd.DataFrame,
    sig: Signal,
    sl_units: float,
    tp1_units: float,
    tp2_units: float,
    leg2_sl_mode: str,   # "BE_FEE", "BE_ZERO", "SL_MINUS_1U", "SL_MINUS_2U", "KEEP_SL5"
    entry_fee: float = 0.0002,
    tp_fee: float = 0.0002,
    sl_fee: float = 0.0005,
    slippage_bps: float = 1.0,
    max_hold_bars: int = 288,
) -> dict:
    unit_size = max(sig.atr * 0.5, sig.entry_price * 1e-5)
    entry = sig.entry_price
    d = sig.direction
    slip = slippage_bps / 10_000.0

    sl_orig = entry - d * sl_units * unit_size
    tp1_price = entry + d * tp1_units * unit_size
    tp2_price = entry + d * tp2_units * unit_size

    if leg2_sl_mode == "BE_FEE":
        leg2_sl = entry + d * entry * (entry_fee + sl_fee + slip)
    elif leg2_sl_mode == "BE_ZERO":
        leg2_sl = entry
    elif leg2_sl_mode == "SL_MINUS_1U":
        leg2_sl = entry - d * 1.0 * unit_size
    elif leg2_sl_mode == "SL_MINUS_2U":
        leg2_sl = entry - d * 2.0 * unit_size
    else:  # KEEP_SL5
        leg2_sl = sl_orig

    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    end_idx = min(sig.bar_idx + max_hold_bars, len(df) - 1)

    tp1_hit = False
    tp1_bar = -1
    leg1_net_u = 0.0
    fee1_u = (entry * entry_fee + tp1_price * tp_fee) / unit_size

    for j in range(sig.bar_idx + 1, end_idx + 1):
        h, l = highs[j], lows[j]
        if not tp1_hit:
            if (l <= sl_orig) if d == 1 else (h >= sl_orig):
                fill = sl_orig * (1.0 - d * slip)
                gross = d * (fill - entry) / unit_size
                fee = (entry * entry_fee + fill * sl_fee) / unit_size
                return {"outcome": "SL_FULL", "tp1_hit": False, "leg2_stopped": False, "would_hit_tp2": False, "net_u": gross - fee}

            if (h >= tp1_price) if d == 1 else (l <= tp1_price):
                tp1_hit = True
                tp1_bar = j
                leg1_net_u = 0.5 * (tp1_units - fee1_u)
                if (h >= tp2_price) if d == 1 else (l <= tp2_price):
                    fee2 = (entry * entry_fee + tp2_price * tp_fee) / unit_size
                    return {"outcome": "TP1+TP2", "tp1_hit": True, "leg2_stopped": False, "would_hit_tp2": True, "net_u": leg1_net_u + 0.5 * (tp2_units - fee2)}
        else:
            if (l <= leg2_sl) if d == 1 else (h >= leg2_sl):
                fill = leg2_sl * (1.0 - d * slip)
                gross2 = d * (fill - entry) / unit_size
                fee2 = (entry * entry_fee + fill * sl_fee) / unit_size
                # Check if price STILL would have hit TP2 later before hitting original SL5!
                would_tp2 = False
                for k in range(j + 1, end_idx + 1):
                    hk, lk = highs[k], lows[k]
                    if (lk <= sl_orig) if d == 1 else (hk >= sl_orig):
                        break
                    if (hk >= tp2_price) if d == 1 else (lk <= tp2_price):
                        would_tp2 = True
                        break
                return {
                    "outcome": "TP1+LEG2_SL",
                    "tp1_hit": True,
                    "leg2_stopped": True,
                    "would_hit_tp2": would_tp2,
                    "net_u": leg1_net_u + 0.5 * (gross2 - fee2),
                }

            if (h >= tp2_price) if d == 1 else (l <= tp2_price):
                fee2 = (entry * entry_fee + tp2_price * tp_fee) / unit_size
                return {"outcome": "TP1+TP2", "tp1_hit": True, "leg2_stopped": False, "would_hit_tp2": True, "net_u": leg1_net_u + 0.5 * (tp2_units - fee2)}

    fill = closes[end_idx] * (1.0 - d * slip)
    gross = d * (fill - entry) / unit_size
    fee = (entry * entry_fee + fill * sl_fee) / unit_size
    if tp1_hit:
        return {"outcome": "TP1+EOD", "tp1_hit": True, "leg2_stopped": False, "would_hit_tp2": False, "net_u": leg1_net_u + 0.5 * (gross - fee)}
    return {"outcome": "EOD", "tp1_hit": False, "leg2_stopped": False, "would_hit_tp2": False, "net_u": gross - fee}


def main():
    cfg = BacktestConfig()
    end_ym = _detect_latest_available_month("BTCUSDT", "5m")

    raw_m5 = {sym: download_symbol_1y(sym, interval="5m", months=12, market="futures", end_ym=end_ym) for sym in cfg.symbols}
    dfs_m15 = {sym: resample_ohlcv(df, "15min") for sym, df in raw_m5.items()}
    dfs_h1 = {sym: resample_ohlcv(df, "1h") for sym, df in raw_m5.items()}
    dfs_h4 = {sym: resample_ohlcv(df, "4h") for sym, df in raw_m5.items()}

    sigs_m15 = {sym: generate_3ema_signals(dfs_m15[sym], dfs_h1[sym], sym, cfg.strategy, min_adx=20.0, max_wave=2) for sym in cfg.symbols}
    sigs_h1 = {sym: generate_3ema_signals(dfs_h1[sym], dfs_h4[sym], sym, cfg.strategy, min_adx=20.0, max_wave=2) for sym in cfg.symbols}

    modes = [
        ("1. SL = Entry + Phi (BE_FEE)", "BE_FEE"),
        ("2. SL = Entry thuan (BE_ZERO)", "BE_ZERO"),
        ("3. SL = Entry - 1U (Van lai +1.5U)", "SL_MINUS_1U"),
        ("4. SL = Entry - 2U (Van lai +1.0U)", "SL_MINUS_2U"),
        ("5. Giu nguyen SL goc -5U", "KEEP_SL5"),
    ]

    for title, dfs, sigs_map, setup, tp1_u in [
        ("M15 EMA34_W1 (TP1=4U, TP2=10U, 428 lenh)", dfs_m15, sigs_m15, "EMA34_W1", 4.0),
        ("H1 EMA34_W2 (TP1=3U, TP2=10U, 75 lenh)", dfs_h1, sigs_h1, "EMA34_W2", 3.0),
    ]:
        print("\n" + "=" * 110)
        print(f"KIEM TRA GIAT NGUOC QUET BE SAU KHI AN TP1 — {title}")
        print("=" * 110)
        rows = []
        for mlabel, mcode in modes:
            res = []
            for sym in cfg.symbols:
                for s in sigs_map[sym]:
                    if s.setup_type == setup:
                        res.append(sim_leg2_policy(dfs[sym], s, 5.0, tp1_u, 10.0, mcode))
            rdf = pd.DataFrame(res)
            tp1_df = rdf[rdf["tp1_hit"]]
            n_tp1 = len(tp1_df)
            whipsaw_pct = tp1_df["leg2_stopped"].mean() * 100.0 if n_tp1 else 0.0
            stopped_df = tp1_df[tp1_df["leg2_stopped"]]
            regret_pct = stopped_df["would_hit_tp2"].mean() * 100.0 if len(stopped_df) else 0.0
            tp2_pct = (rdf["outcome"] == "TP1+TP2").mean() * 100.0
            wr_pct = (rdf["net_u"] > 0).mean() * 100.0
            pos = rdf.loc[rdf["net_u"] > 0, "net_u"].sum()
            neg = -rdf.loc[rdf["net_u"] < 0, "net_u"].sum()
            pf = pos / neg if neg > 0 else 0.0
            rows.append(
                {
                    "Chinh_Sach_SL_Chan_2": mlabel,
                    "WR_%": wr_pct,
                    "TP2_%": tp2_pct,
                    "Quet_SL_Chan2_%_(sauTP1)": whipsaw_pct,
                    "Bi_Ru_Oan_VanLenTP2_%": regret_pct,
                    "PF": pf,
                    "Total_Net_U": rdf["net_u"].sum(),
                }
            )
        print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:,.2f}"))


if __name__ == "__main__":
    main()
