"""Backtest Partial Take Profit (50% at TP1 + Move SL to Entry+Fee + 50% at TP2) across M5, M15, H1."""
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import BacktestConfig, RiskConfig
from data.binance_loader import _detect_latest_available_month, download_symbol_1y
from run_3ema_complete import generate_3ema_signals
from run_research_mtf import resample_ohlcv
from strategy.ema_wave import Signal


def simulate_partial_tp(
    df: pd.DataFrame,
    sig: Signal,
    sl_units: float,
    tp1_units: float,
    tp2_units: float,
    tp1_frac: float = 0.5,
    atr_to_unit_mult: float = 0.5,
    entry_fee: float = 0.0002,
    tp_fee: float = 0.0002,
    sl_fee: float = 0.0005,
    slippage_bps: float = 1.0,
    max_hold_bars: int = 288,
) -> dict:
    """Simulate 2-leg trade: close tp1_frac at TP1, move SL to Entry+Fee, close remaining at TP2 or BE."""
    unit_size = max(sig.atr * atr_to_unit_mult, sig.entry_price * 1e-5)
    entry = sig.entry_price
    d = sig.direction
    slip = slippage_bps / 10_000.0

    sl_price = entry - d * sl_units * unit_size
    tp1_price = entry + d * tp1_units * unit_size
    tp2_price = entry + d * tp2_units * unit_size
    # Entry + round-trip fee buffer so remaining leg is truly >= $0 after fees
    be_price = entry + d * entry * (entry_fee + sl_fee + slip)

    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    n = len(df)
    end_idx = min(sig.bar_idx + max_hold_bars, n - 1)

    tp1_hit = False
    leg1_net_u = 0.0
    leg2_net_u = 0.0
    outcome = "EOD"

    for j in range(sig.bar_idx + 1, end_idx + 1):
        h = highs[j]
        l = lows[j]

        if not tp1_hit:
            # Conservative: check SL before TP1
            sl_triggered = (l <= sl_price) if d == 1 else (h >= sl_price)
            if sl_triggered:
                fill = sl_price * (1.0 - d * slip)
                gross_u = d * (fill - entry) / unit_size
                fee_u = (entry * entry_fee + fill * sl_fee) / unit_size
                return {
                    "outcome": "SL_FULL",
                    "tp1_hit": False,
                    "tp2_hit": False,
                    "net_units": gross_u - fee_u,
                    "fee_units": fee_u,
                }

            tp1_triggered = (h >= tp1_price) if d == 1 else (l <= tp1_price)
            if tp1_triggered:
                tp1_hit = True
                # Close leg 1 (tp1_frac) at tp1_price
                fee1_u = (entry * entry_fee + tp1_price * tp_fee) / unit_size
                leg1_net_u = tp1_frac * (tp1_units - fee1_u)
                # Move SL of leg 2 to Entry + Fee
                sl_price = be_price

                # If TP2 also reached on this very same candle (and SL wasn't hit)
                tp2_same_bar = (h >= tp2_price) if d == 1 else (l <= tp2_price)
                if tp2_same_bar:
                    fee2_u = (entry * entry_fee + tp2_price * tp_fee) / unit_size
                    leg2_net_u = (1.0 - tp1_frac) * (tp2_units - fee2_u)
                    return {
                        "outcome": "TP1+TP2",
                        "tp1_hit": True,
                        "tp2_hit": True,
                        "net_units": leg1_net_u + leg2_net_u,
                        "fee_units": tp1_frac * fee1_u + (1.0 - tp1_frac) * fee2_u,
                    }
        else:
            # Leg 2 active with SL at Entry+Fee: conservative check BE stop before TP2
            be_triggered = (l <= sl_price) if d == 1 else (h >= sl_price)
            if be_triggered:
                fill = sl_price * (1.0 - d * slip)
                gross2_u = d * (fill - entry) / unit_size
                fee2_u = (entry * entry_fee + fill * sl_fee) / unit_size
                leg2_net_u = (1.0 - tp1_frac) * (gross2_u - fee2_u)
                fee1_u = (entry * entry_fee + tp1_price * tp_fee) / unit_size
                return {
                    "outcome": "TP1+BE",
                    "tp1_hit": True,
                    "tp2_hit": False,
                    "net_units": leg1_net_u + leg2_net_u,
                    "fee_units": tp1_frac * fee1_u + (1.0 - tp1_frac) * fee2_u,
                }

            tp2_triggered = (h >= tp2_price) if d == 1 else (l <= tp2_price)
            if tp2_triggered:
                fee2_u = (entry * entry_fee + tp2_price * tp_fee) / unit_size
                leg2_net_u = (1.0 - tp1_frac) * (tp2_units - fee2_u)
                fee1_u = (entry * entry_fee + tp1_price * tp_fee) / unit_size
                return {
                    "outcome": "TP1+TP2",
                    "tp1_hit": True,
                    "tp2_hit": True,
                    "net_units": leg1_net_u + leg2_net_u,
                    "fee_units": tp1_frac * fee1_u + (1.0 - tp1_frac) * fee2_u,
                }

    # EOD close
    fill = closes[end_idx] * (1.0 - d * slip)
    gross_u = d * (fill - entry) / unit_size
    fee_u = (entry * entry_fee + fill * sl_fee) / unit_size
    if tp1_hit:
        fee1_u = (entry * entry_fee + tp1_price * tp_fee) / unit_size
        leg2_net_u = (1.0 - tp1_frac) * (gross_u - fee_u)
        return {
            "outcome": "TP1+EOD",
            "tp1_hit": True,
            "tp2_hit": False,
            "net_units": leg1_net_u + leg2_net_u,
            "fee_units": tp1_frac * fee1_u + (1.0 - tp1_frac) * fee_u,
        }
    return {
        "outcome": "EOD",
        "tp1_hit": False,
        "tp2_hit": False,
        "net_units": gross_u - fee_u,
        "fee_units": fee_u,
    }


def eval_scheme(dfs: Dict[str, pd.DataFrame], sigs_map: Dict[str, List[Signal]], allowed_setups: List[str], sl_u: float, tp1_u: float, tp2_u: float, tp1_frac: float) -> dict:
    res = []
    for sym, sigs in sigs_map.items():
        df = dfs[sym]
        for s in sigs:
            if s.setup_type not in allowed_setups:
                continue
            r = simulate_partial_tp(df, s, sl_u, tp1_u, tp2_u, tp1_frac)
            res.append(r)
    rdf = pd.DataFrame(res)
    n = len(rdf)
    wins = (rdf["net_units"] > 0).mean() * 100.0
    tp1_rate = rdf["tp1_hit"].mean() * 100.0
    tp2_rate = rdf["tp2_hit"].mean() * 100.0
    sl_rate = (rdf["outcome"] == "SL_FULL").mean() * 100.0
    pos = rdf.loc[rdf["net_units"] > 0, "net_units"].sum()
    neg = -rdf.loc[rdf["net_units"] < 0, "net_units"].sum()
    pf = pos / neg if neg > 0 else 0.0
    return {
        "Trades": n,
        "Winrate_%": wins,
        "TP1_%": tp1_rate,
        "TP2_%": tp2_rate,
        "SL_Full_%": sl_rate,
        "PF": pf,
        "Avg_Net_U": rdf["net_units"].mean(),
        "Total_Net_U": rdf["net_units"].sum(),
        "Avg_Fee_U": rdf["fee_units"].mean(),
    }


def main():
    cfg = BacktestConfig()
    end_ym = _detect_latest_available_month("BTCUSDT", "5m")

    raw_m5 = {sym: download_symbol_1y(sym, interval="5m", months=12, market="futures", end_ym=end_ym) for sym in cfg.symbols}
    dfs_m15 = {sym: resample_ohlcv(df, "15min") for sym, df in raw_m5.items()}
    dfs_h1 = {sym: resample_ohlcv(df, "1h") for sym, df in raw_m5.items()}
    dfs_h4 = {sym: resample_ohlcv(df, "4h") for sym, df in raw_m5.items()}

    sigs_m5 = {sym: generate_3ema_signals(raw_m5[sym], dfs_h1[sym], sym, cfg.strategy, min_adx=20.0, max_wave=2) for sym in cfg.symbols}
    sigs_m15 = {sym: generate_3ema_signals(dfs_m15[sym], dfs_h1[sym], sym, cfg.strategy, min_adx=20.0, max_wave=2) for sym in cfg.symbols}
    sigs_h1 = {sym: generate_3ema_signals(dfs_h1[sym], dfs_h4[sym], sym, cfg.strategy, min_adx=20.0, max_wave=2) for sym in cfg.symbols}

    schemes = [
        ("Single: SL5 / TP5 (No Partial)", 5.0, 5.0, 5.0, 1.0),
        ("Single: SL5 / TP10 (No Partial)", 5.0, 10.0, 10.0, 1.0),
        ("Partial 50%@3U + BE_Fee + 50%@4.5U", 5.0, 3.0, 4.5, 0.5),
        ("Partial 50%@3U + BE_Fee + 50%@5.0U", 5.0, 3.0, 5.0, 0.5),
        ("Partial 50%@3U + BE_Fee + 50%@6.0U", 5.0, 3.0, 6.0, 0.5),
        ("Partial 50%@3U + BE_Fee + 50%@10.0U", 5.0, 3.0, 10.0, 0.5),
        ("Partial 50%@4U + BE_Fee + 50%@10.0U", 5.0, 4.0, 10.0, 0.5),
        ("Partial 50%@5U + BE_Fee + 50%@10.0U", 5.0, 5.0, 10.0, 0.5),
    ]

    groups = [
        ("M5 (HTF+ADX20, EMA34_W1 + EMA200_W1)", raw_m5, sigs_m5, ["EMA34_W1", "EMA200_W1"]),
        ("M15 (HTF+ADX20, EMA34_W1)", dfs_m15, sigs_m15, ["EMA34_W1"]),
        ("M15 (HTF+ADX20, EMA34_W1 + EMA200_W1)", dfs_m15, sigs_m15, ["EMA34_W1", "EMA200_W1"]),
        ("H1 (HTF+ADX20, EMA34_W2)", dfs_h1, sigs_h1, ["EMA34_W2"]),
    ]

    for gname, gdfs, gsigs, setups in groups:
        print("\n" + "=" * 105)
        print(f"SO SANH CHOT TUNG PHAN (PARTIAL TP + BE_FEE) — {gname}")
        print("=" * 105)
        rows = []
        for sname, sl_u, tp1_u, tp2_u, frac in schemes:
            m = eval_scheme(gdfs, gsigs, setups, sl_u, tp1_u, tp2_u, frac)
            rows.append({"Scheme": sname, **m})
        print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:,.2f}"))


if __name__ == "__main__":
    main()
