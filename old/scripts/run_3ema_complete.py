"""Complete 3-EMA (34 + 89 + 200) Combined Strategy Backtest (500-Unit Buffer, 20-Step Compounding, OOS Validation)."""
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
from engine.portfolio import run_discovery_stats, run_portfolio_backtest
from report.html_report import export_html_report
from run_backtest import compute_portfolio_metrics
from run_research_mtf import compute_adx, resample_ohlcv
from strategy.ema_wave import Signal
from strategy.indicators import add_indicators


def generate_3ema_signals(
    df: pd.DataFrame,
    df_htf: pd.DataFrame,
    symbol: str,
    cfg: StrategyConfig,
    min_adx: float = 20.0,
    max_wave: int = 2,
) -> List[Signal]:
    """Generate zero-lookahead Touch 1..max_wave signals across EMA 34, EMA 89, AND EMA 200 with HTF+ADX filter."""
    data = add_indicators(df, cfg)
    adx = compute_adx(df, 14).values
    n = len(data)
    if n < 220:
        return []

    # HTF trend (strictly shifted by 1 bar -> closed HTF candle only)
    htf = df_htf.copy()
    htf["htf_ema34"] = htf["close"].ewm(span=34, adjust=False).mean()
    htf["htf_ema200"] = htf["close"].ewm(span=200, adjust=False).mean()
    htf["htf_trend"] = np.where(
        (htf["close"] > htf["htf_ema200"]) & (htf["htf_ema34"] > htf["htf_ema200"]),
        1,
        np.where((htf["close"] < htf["htf_ema200"]) & (htf["htf_ema34"] < htf["htf_ema200"]), -1, 0),
    )
    htf["htf_trend_closed"] = htf["htf_trend"].shift(1).fillna(0).astype(int)
    htf_ts = htf["timestamp"].values
    htf_tr = htf["htf_trend_closed"].values

    closes = data["close"].values
    opens = data["open"].values
    highs = data["high"].values
    lows = data["low"].values
    ema34 = data["ema34"].values
    ema89 = data["ema89"].values
    ema200 = data["ema200"].values
    atr = data["atr"].values
    crange = data["candle_range"].values
    vol_ratio = data["vol_ratio"].values
    lower_wick = data["lower_wick_ratio"].values
    upper_wick = data["upper_wick_ratio"].values
    ts_arr = data["timestamp"].values

    signals: List[Signal] = []

    # Track state independently for each EMA (34, 89, 200)
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

        for i in range(205, n):
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

            if crange[i] > cfg.liquidity_void_atr * a:
                continue

            if trend == 1:
                if (h - et) >= imp_mult * a and l > et:
                    armed = True
            else:
                if (et - l) >= imp_mult * a and h < et:
                    armed = True

            if not armed or (i - last_touch) < min_gap:
                continue

            touch_buf = 0.20 * a
            breach_buf = 0.65 * a
            if trend == 1:
                touched = (l <= et + touch_buf) and (c >= et - breach_buf)
                pa_ok = (lower_wick[i] >= cfg.min_wick_ratio) or (c > o and c > et)
            else:
                touched = (h >= et - touch_buf) and (c <= et + breach_buf)
                pa_ok = (upper_wick[i] >= cfg.min_wick_ratio) or (c < o and c < et)

            if not touched:
                continue

            wave_count += 1
            armed = False
            last_touch = i

            if wave_count > max_wave:
                continue

            # Apply ADX + HTF filter
            if adx[i] < min_adx:
                continue
            idx = int(np.searchsorted(htf_ts, ts_arr[i], side="right")) - 1
            if idx < 0 or int(htf_tr[idx]) != trend:
                continue

            ts = pd.Timestamp(ts_arr[i])
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")

            vol_ok = (not np.isnan(vol_ratio[i])) and (vol_ratio[i] >= cfg.min_vol_ratio)
            signals.append(
                Signal(
                    bar_idx=i,
                    timestamp=ts,
                    symbol=symbol,
                    direction=trend,
                    wave=wave_count,
                    setup_type=f"EMA{ema_p}_W{wave_count}",
                    entry_price=float(c),
                    atr=float(a),
                    pa_vol_ok=bool(pa_ok and vol_ok),
                    ema_aligned=True,
                )
            )

    signals.sort(key=lambda s: s.bar_idx)
    return signals


def main():
    cfg = BacktestConfig()
    end_ym = _detect_latest_available_month("BTCUSDT", "5m")

    raw_m5: Dict[str, pd.DataFrame] = {}
    for sym in cfg.symbols:
        raw_m5[sym] = download_symbol_1y(sym, interval="5m", months=12, market="futures", end_ym=end_ym)

    dfs_h1 = {sym: resample_ohlcv(df, "1h") for sym, df in raw_m5.items()}
    dfs_h4 = {sym: resample_ohlcv(df, "4h") for sym, df in raw_m5.items()}
    dfs_m15 = {sym: resample_ohlcv(df, "15min") for sym, df in raw_m5.items()}

    print("=" * 96)
    print("CHICKTRADE — 3-EMA (34 + 89 + 200) COMPLETE STRATEGY BACKTEST | BAO 500 GIA | OOS VALIDATION")
    print("=" * 96)

    all_tables = {}
    portfolio_runs = {}

    for tf_name, tf_dfs, htf_dfs, sl_u, tp_u in [
        ("M15", dfs_m15, dfs_h1, 5.0, 10.0),
        ("M15", dfs_m15, dfs_h1, 5.0, 5.0),
        ("H1", dfs_h1, dfs_h4, 5.0, 10.0),
        ("H1", dfs_h1, dfs_h4, 5.0, 5.0),
    ]:
        sigs_map = {
            sym: generate_3ema_signals(tf_dfs[sym], htf_dfs[sym], sym, cfg.strategy, min_adx=20.0, max_wave=2)
            for sym in cfg.symbols
        }
        rcfg = RiskConfig(buffer_units=500.0, sl_units=sl_u, tp_units=tp_u, enable_be=False)
        ddf = run_discovery_stats(tf_dfs, sigs_map, rcfg, sl_u, tp_u, 3.0, False)

        # Split In-Sample (first 6m) vs Out-of-Sample (last 6m)
        mid_ts = ddf["timestamp"].median()
        in_df = ddf[ddf["timestamp"] < mid_ts]
        oos_df = ddf[ddf["timestamp"] >= mid_ts]

        rows = []
        qualified_setups_oos = {}
        qualified_setups_full = {}

        for stype, g in ddf.groupby("setup_type"):
            g_in = in_df[in_df["setup_type"] == stype]
            g_oos = oos_df[oos_df["setup_type"] == stype]

            def _stats(sub):
                if len(sub) == 0:
                    return 0, 0.0, 0.0, 0.0
                wr = sub["is_win"].mean() * 100.0
                pos = sub.loc[sub["net_units"] > 0, "net_units"].sum()
                neg = -sub.loc[sub["net_units"] < 0, "net_units"].sum()
                pf = pos / neg if neg > 0 else (9.99 if pos > 0 else 0.0)
                return len(sub), wr, pf, sub["net_units"].sum()

            n_all, wr_all, pf_all, u_all = _stats(g)
            n_in, wr_in, pf_in, u_in = _stats(g_in)
            n_oo, wr_oo, pf_oo, u_oo = _stats(g_oos)

            rows.append(
                {
                    "Setup": stype,
                    "1Y_Trades": n_all,
                    "1Y_WR%": wr_all,
                    "1Y_PF": pf_all,
                    "1Y_NetU": u_all,
                    "IS_6M_PF": pf_in,
                    "OOS_6M_Trades": n_oo,
                    "OOS_6M_WR%": wr_oo,
                    "OOS_6M_PF": pf_oo,
                    "OOS_6M_NetU": u_oo,
                }
            )
            # Qualify setup for OOS trading strictly if In-Sample PF >= 1.05 (positive expectancy!)
            if pf_in >= 1.05 and n_in >= 15:
                # Assign FULL (>70% WR or PF >= 1.4) vs HALF
                qualified_setups_oos[stype] = 0.72 if (wr_in > 70.0 or pf_in >= 1.40) else 0.60
            if pf_all >= 1.05 and n_all >= 30:
                qualified_setups_full[stype] = 0.72 if (wr_all > 70.0 or pf_all >= 1.40) else 0.60

        tname = f"{tf_name}_SL{int(sl_u)}_TP{int(tp_u)}"
        tbl = pd.DataFrame(rows).sort_values("1Y_PF", ascending=False)
        all_tables[tname] = tbl

        print(f"\n--- {tname} (3-EMA Combined: EMA34 + EMA89 + EMA200, HTF+ADX20, Bao 500 Gia) ---")
        print(tbl.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

        # Stage 2 Portfolio Run (Full 12M & Strict Out-of-Sample 6M)
        if qualified_setups_full:
            tr_f, eq_f, cp_f = run_portfolio_backtest(tf_dfs, sigs_map, qualified_setups_full, rcfg, require_pa_vol_for_w2=False)
            mf = compute_portfolio_metrics(tr_f, eq_f, rcfg.initial_capital, cp_f.max_step_reached)
            portfolio_runs[f"{tname}_FULL12M"] = {
                "label": f"{tname} — Toàn bộ 12 tháng (Bao 500 giá)",
                "trades": tr_f,
                "equity_curve": eq_f,
                "metrics": mf,
            }
            print(
                f"  >> PORTFOLIO FULL 12M ({list(qualified_setups_full.keys())}): "
                f"Equity=${mf['final_equity']:,.2f} ({mf['return_pct']:+.2f}%) | Step {mf['max_step']}/20 | "
                f"Trades={mf['total_trades']} | WR={mf['winrate_pct']:.1f}% | PF={mf['profit_factor']:.2f} | "
                f"MaxDD={mf['max_dd_pct']:.2f}% | Fees=${mf['total_fees']:,.0f}"
            )

        if qualified_setups_oos:
            tr_o, eq_o, cp_o = run_portfolio_backtest(
                tf_dfs, sigs_map, qualified_setups_oos, rcfg, start_time=mid_ts, require_pa_vol_for_w2=False
            )
            mo = compute_portfolio_metrics(tr_o, eq_o, rcfg.initial_capital, cp_o.max_step_reached)
            portfolio_runs[f"{tname}_OOS6M"] = {
                "label": f"{tname} — Out-of-Sample 6 Tháng Cuối (Bao 500 giá)",
                "trades": tr_o,
                "equity_curve": eq_o,
                "metrics": mo,
            }
            print(
                f"  >> PORTFOLIO OOS   6M ({list(qualified_setups_oos.keys())}): "
                f"Equity=${mo['final_equity']:,.2f} ({mo['return_pct']:+.2f}%) | Step {mo['max_step']}/20 | "
                f"Trades={mo['total_trades']} | WR={mo['winrate_pct']:.1f}% | PF={mo['profit_factor']:.2f} | "
                f"MaxDD={mo['max_dd_pct']:.2f}% | Fees=${mo['total_fees']:,.0f}"
            )

    out_dir = os.path.join(ROOT, "outputs")
    report_path = export_html_report(out_dir, all_tables, portfolio_runs)
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
