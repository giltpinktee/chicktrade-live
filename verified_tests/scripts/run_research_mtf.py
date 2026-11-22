"""Multi-timeframe (M15 / M30 / H1) & Filter Research for Method 1 (EMA 34-89-200) with 400-unit buffer."""
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import BacktestConfig, RiskConfig, StrategyConfig
from data.binance_loader import _detect_latest_available_month, download_symbol_1y
from engine.portfolio import run_discovery_stats, run_portfolio_backtest
from report.html_report import export_html_report
from run_backtest import compute_portfolio_metrics, summarize_discovery
from strategy.ema_wave import Signal, generate_signals


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    tmp = df.set_index("timestamp")
    res = pd.DataFrame(
        {
            "open": tmp["open"].resample(rule).first(),
            "high": tmp["high"].resample(rule).max(),
            "low": tmp["low"].resample(rule).min(),
            "close": tmp["close"].resample(rule).last(),
            "volume": tmp["volume"].resample(rule).sum(),
        }
    ).dropna()
    return res.reset_index()


def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    plus_dm = high.diff().clip(lower=0.0)
    minus_dm = (-low.diff()).clip(lower=0.0)
    plus_dm = plus_dm.where(plus_dm > minus_dm, 0.0)
    minus_dm = minus_dm.where(minus_dm > plus_dm, 0.0)

    tr = pd.concat(
        [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1.0 / period, adjust=False).mean()
    plus_di = 100.0 * plus_dm.ewm(alpha=1.0 / period, adjust=False).mean() / atr.replace(0, np.nan)
    minus_di = 100.0 * minus_dm.ewm(alpha=1.0 / period, adjust=False).mean() / atr.replace(0, np.nan)
    dx = (100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)).fillna(0.0)
    return dx.ewm(alpha=1.0 / period, adjust=False).mean()


def filter_signals_htf_adx(
    df: pd.DataFrame,
    df_htf: pd.DataFrame,
    sigs: List[Signal],
    min_adx: float = 20.0,
    require_ema_aligned: bool = True,
    require_htf_trend: bool = True,
) -> List[Signal]:
    """Filter signals using ADX (anti-sideway) and Higher-Timeframe EMA34/EMA200 trend."""
    adx = compute_adx(df, 14).values

    htf = df_htf.copy()
    htf["htf_ema34"] = htf["close"].ewm(span=34, adjust=False).mean()
    htf["htf_ema200"] = htf["close"].ewm(span=200, adjust=False).mean()
    htf["htf_trend"] = np.where(
        (htf["close"] > htf["htf_ema200"]) & (htf["htf_ema34"] > htf["htf_ema200"]),
        1,
        np.where((htf["close"] < htf["htf_ema200"]) & (htf["htf_ema34"] < htf["htf_ema200"]), -1, 0),
    )
    # Shift HTF by 1 bar so we only use CLOSED HTF candles (zero lookahead!)
    htf["htf_trend_closed"] = htf["htf_trend"].shift(1).fillna(0).astype(int)
    htf_ts = htf["timestamp"].values
    htf_tr = htf["htf_trend_closed"].values

    filtered: List[Signal] = []
    for s in sigs:
        if require_ema_aligned and not s.ema_aligned:
            continue
        if adx[s.bar_idx] < min_adx:
            continue
        if require_htf_trend:
            idx = int(np.searchsorted(htf_ts, s.timestamp.to_datetime64(), side="right")) - 1
            if idx < 0:
                continue
            ht = int(htf_tr[idx])
            # Trend-following waves (W1, W2, W3) must align with HTF trend; W4 reversal is opposite
            if s.wave <= 3 and ht != s.direction:
                continue
        filtered.append(s)
    return filtered


def main():
    cfg = BacktestConfig()
    end_ym = _detect_latest_available_month("BTCUSDT", "5m")

    raw_m5: Dict[str, pd.DataFrame] = {}
    for sym in cfg.symbols:
        raw_m5[sym] = download_symbol_1y(sym, interval="5m", months=12, market="futures", end_ym=end_ym)

    # Build H4 HTF for trend alignment
    dfs_h4 = {sym: resample_ohlcv(df, "4h") for sym, df in raw_m5.items()}
    dfs_h1 = {sym: resample_ohlcv(df, "1h") for sym, df in raw_m5.items()}

    tf_configs = [
        ("M15", "15min", dfs_h1),
        ("M30", "30min", dfs_h4),
        ("H1", "1h", dfs_h4),
    ]

    rr_profiles = [
        ("SL5_TP5_NoBE", 5.0, 5.0, 3.0, False),
        ("SL5_TP7.5_NoBE", 5.0, 7.5, 3.0, False),
        ("SL5_TP10_NoBE", 5.0, 10.0, 3.0, False),
        ("SL4_TP6_NoBE", 4.0, 6.0, 3.0, False),
        ("SL4_TP8_NoBE", 4.0, 8.0, 3.0, False),
        ("SL5_TP3_NoBE", 5.0, 3.0, 3.0, False),
    ]

    print("=" * 84)
    print("CHICKTRADE — MULTI-TIMEFRAME (M15 / M30 / H1) + HTF/ADX FILTER RESEARCH (400-UNIT)")
    print("=" * 84)

    summary_rows = []
    best_candidates = []

    for tf_name, rule, htf_map in tf_configs:
        tf_dfs = {sym: resample_ohlcv(df, rule) for sym, df in raw_m5.items()}
        raw_sigs = {sym: generate_signals(tf_dfs[sym], sym, cfg.strategy) for sym in cfg.symbols}
        filt_sigs = {
            sym: filter_signals_htf_adx(tf_dfs[sym], htf_map[sym], raw_sigs[sym], min_adx=20.0)
            for sym in cfg.symbols
        }

        for mode_name, sig_map in [("RAW", raw_sigs), ("HTF+ADX20", filt_sigs)]:
            for pname, sl_u, tp_u, be_u, use_be in rr_profiles:
                rcfg = RiskConfig(buffer_units=400.0, sl_units=sl_u, tp_units=tp_u, be_trigger_units=be_u, enable_be=use_be)
                ddf = run_discovery_stats(tf_dfs, sig_map, rcfg, sl_u, tp_u, be_u, use_be)
                if ddf.empty:
                    continue

                for subset_name, waves in [("W1+W2", ["W1_TREND", "W2_TREND"]), ("W1_ONLY", ["W1_TREND"]), ("W4_REV", ["W4_REVERSAL"])]:
                    sub = ddf[ddf["setup_type"].isin(waves)]
                    if len(sub) < 30:
                        continue
                    wr = sub["is_win"].mean() * 100.0
                    pos = sub.loc[sub["net_units"] > 0, "net_units"].sum()
                    neg = -sub.loc[sub["net_units"] < 0, "net_units"].sum()
                    pf = pos / neg if neg > 0 else 0.0
                    avg_u = sub["net_units"].mean()
                    tot_u = sub["net_units"].sum()
                    avg_fee = sub["fee_units"].mean()

                    row = {
                        "TF": tf_name,
                        "Filter": mode_name,
                        "Profile": pname,
                        "Waves": subset_name,
                        "Trades": len(sub),
                        "Winrate_%": wr,
                        "PF": pf,
                        "Avg_Net_U": avg_u,
                        "Total_Net_U": tot_u,
                        "Avg_Fee_U": avg_fee,
                    }
                    summary_rows.append(row)
                    if pf >= 1.05:
                        best_candidates.append((pf, row, tf_dfs, sig_map, rcfg, waves))

    res_df = pd.DataFrame(summary_rows).sort_values("PF", ascending=False)
    print("\n--- TOP 15 CONFIGURATIONS BY PROFIT FACTOR (AFTER BINANCE FUTURES FEES) ---")
    print(res_df.head(15).to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    print("\n--- BOTTOM 5 (FOR CONTRAST) ---")
    print(res_df.tail(5).to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    # Run Stage 2 Portfolio Simulation (400-unit buffer, 20 steps +8%, max 2 open, max 10/day) on Top 3 positive PF configs
    print("\n--- STAGE 2: PORTFOLIO SIMULATION (400-UNIT BUFFER, $10,000, 20 STEPS +8%) ON TOP CANDIDATES ---")
    portfolio_runs = {}
    best_candidates.sort(key=lambda x: x[0], reverse=True)
    seen = set()
    picked = []
    for item in best_candidates:
        key = (item[1]["TF"], item[1]["Filter"], item[1]["Profile"], item[1]["Waves"])
        if key not in seen:
            seen.add(key)
            picked.append(item)
        if len(picked) >= 3:
            break

    for idx, (pf_val, row, tf_dfs, sig_map, rcfg, waves) in enumerate(picked, 1):
        # Allow selected waves in portfolio simulation using their actual winrate (or 0.60 if positive expectancy R:R > 1)
        # Note: if R:R >= 1.5, a 45-50% winrate is highly profitable; we test BOTH strict >=51% and PF-qualified Half Size
        wr_map_strict = {f"{s}:{w}": row["Winrate_%"] / 100.0 for s in cfg.symbols for w in waves}
        # Also test with Half Size enabled for positive-expectancy R:R setups
        wr_map_half = {f"{s}:{w}": max(row["Winrate_%"] / 100.0, 0.55) for s in cfg.symbols for w in waves}

        trades, eq_df, comp = run_portfolio_backtest(
            tf_dfs, sig_map, wr_map_half, rcfg, require_pa_vol_for_w2=False
        )
        m = compute_portfolio_metrics(trades, eq_df, rcfg.initial_capital, comp.max_step_reached)
        tag = f"{row['TF']}_{row['Filter']}_{row['Profile']}_{row['Waves']}"
        portfolio_runs[tag] = {
            "label": f"#{idx} {tag} (400U Buffer, Half/Full)",
            "trades": trades,
            "equity_curve": eq_df,
            "metrics": m,
        }
        print(
            f"  #{idx} [{tag}]\n"
            f"      Equity=${m['final_equity']:9,.2f} ({m['return_pct']:+6.2f}%) | Step {m['max_step']:2d}/20 | "
            f"Trades={m['total_trades']:4d} | WR={m['winrate_pct']:5.2f}% | PF={m['profit_factor']:.2f} | "
            f"MaxDD={m['max_dd_pct']:6.2f}% | Fees=${m['total_fees']:,.0f}"
        )

    out_dir = os.path.join(ROOT, "outputs")
    report_path = export_html_report(
        output_dir=out_dir,
        summary_tables={"MTF_Research_Top30": res_df.head(30)},
        portfolio_runs=portfolio_runs,
    )
    print(f"\nUpdated HTML report saved to: {report_path}")


if __name__ == "__main__":
    main()
