"""CHICKTRADE — 1-Year M5 Backtest Runner (Stage 1 Discovery + Stage 2 Portfolio Compounding)."""
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

# Ensure workspace root is on sys.path
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import BacktestConfig, RiskConfig
from data.binance_loader import _detect_latest_available_month, download_symbol_1y
from engine.portfolio import TradeRecord, run_discovery_stats, run_portfolio_backtest
from report.html_report import export_html_report
from strategy.ema_wave import Signal, generate_signals


def summarize_discovery(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows = []
    for keys, g in df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        total = len(g)
        wins = int(g["is_win"].sum())
        tp_cnt = int((g["exit_reason"] == "TP").sum())
        be_cnt = int((g["exit_reason"] == "BE").sum())
        sl_cnt = int((g["exit_reason"] == "SL").sum())
        winrate = (wins / total) * 100.0 if total else 0.0
        non_loss_rate = ((tp_cnt + be_cnt) / total) * 100.0 if total else 0.0
        gross_pos = g.loc[g["net_units"] > 0, "net_units"].sum()
        gross_neg = -g.loc[g["net_units"] < 0, "net_units"].sum()
        pf = (gross_pos / gross_neg) if gross_neg > 0 else (99.0 if gross_pos > 0 else 0.0)
        avg_net_u = g["net_units"].mean()
        total_net_u = g["net_units"].sum()
        avg_fee_u = g["fee_units"].mean()

        row = {col: val for col, val in zip(group_cols, keys)}
        row.update(
            {
                "Trades": total,
                "TP": tp_cnt,
                "BE": be_cnt,
                "SL": sl_cnt,
                "Winrate_%": winrate,
                "NonLoss_%": non_loss_rate,
                "PF": pf,
                "Avg_Net_U": avg_net_u,
                "Total_Net_U": total_net_u,
                "Avg_Fee_U": avg_fee_u,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def compute_portfolio_metrics(trades: List[TradeRecord], eq_df: pd.DataFrame, initial_capital: float, max_step: int) -> dict:
    if not trades:
        return {
            "final_equity": initial_capital,
            "return_pct": 0.0,
            "max_step": 1,
            "total_trades": 0,
            "winrate_pct": 0.0,
            "profit_factor": 0.0,
            "max_dd_pct": 0.0,
            "total_fees": 0.0,
            "full_count": 0,
            "half_count": 0,
        }
    final_eq = float(eq_df["equity"].iloc[-1])
    ret_pct = (final_eq / initial_capital - 1.0) * 100.0
    net_pnls = np.array([t.net_pnl for t in trades])
    wins = int((net_pnls > 0).sum())
    wr = wins / len(trades) * 100.0
    pos = net_pnls[net_pnls > 0].sum()
    neg = -net_pnls[net_pnls < 0].sum()
    pf = float(pos / neg) if neg > 0 else (99.0 if pos > 0 else 0.0)

    eq_vals = eq_df["equity"].values
    peak = np.maximum.accumulate(eq_vals)
    dd = (eq_vals - peak) / peak * 100.0
    max_dd = float(dd.min())

    return {
        "final_equity": final_eq,
        "return_pct": ret_pct,
        "max_step": max_step,
        "total_trades": len(trades),
        "winrate_pct": wr,
        "profit_factor": pf,
        "max_dd_pct": max_dd,
        "total_fees": float(sum(t.fee_usd for t in trades)),
        "full_count": sum(1 for t in trades if t.tier_label == "FULL"),
        "half_count": sum(1 for t in trades if t.tier_label == "HALF"),
    }


def main():
    cfg = BacktestConfig()
    all_symbols = cfg.symbols + [cfg.reference_gold_symbol]

    print("=" * 72)
    print("CHICKTRADE — 1-YEAR M5 BACKTEST (EMA 34-89-200 + 20-STEP COMPOUNDING)")
    print("=" * 72)

    end_ym = _detect_latest_available_month("BTCUSDT", cfg.timeframe)
    print(f"[1/4] Downloading 12 months of {cfg.timeframe} data ending {end_ym[0]}-{end_ym[1]:02d}...")

    dfs: Dict[str, pd.DataFrame] = {}
    signals_by_symbol: Dict[str, List[Signal]] = {}

    for sym in all_symbols:
        market = "spot" if sym == "PAXGUSDT" else "futures"
        df = download_symbol_1y(sym, interval=cfg.timeframe, months=cfg.months, market=market, end_ym=end_ym)
        dfs[sym] = df
        sigs = generate_signals(df, sym, cfg.strategy)
        signals_by_symbol[sym] = sigs
        print(
            f"  - {sym:8s} ({market:7s}): {len(df):6d} bars | "
            f"{df['timestamp'].iloc[0].strftime('%Y-%m-%d')} -> {df['timestamp'].iloc[-1].strftime('%Y-%m-%d')} | "
            f"{len(sigs):5d} raw wave signals"
        )

    # Volatility / Unit Calibration Table (shows exact 200-unit drop-to-wipe per coin)
    print("\n[2/4] ATR-Normalized Unit Calibration (1 Unit = 0.5 * ATR(14) on M5, Buffer = 200 Units):")
    cal_rows = []
    for sym in all_symbols:
        sigs = signals_by_symbol[sym]
        if not sigs:
            continue
        avg_price = float(np.mean([s.entry_price for s in sigs]))
        avg_unit = float(np.mean([s.atr * cfg.risk.atr_to_unit_mult for s in sigs]))
        unit_pct = avg_unit / avg_price * 100.0
        cal_rows.append(
            {
                "Symbol": sym,
                "Avg_Price_$": avg_price,
                "Avg_1_Unit_$": avg_unit,
                "1_Unit_%": unit_pct,
                "SL_5U_$": avg_unit * 5,
                "SL_5U_%": unit_pct * 5,
                "Buffer_200U_$": avg_unit * 200,
                "Drop_To_Wipe_Full_%": unit_pct * 200,
            }
        )
    cal_df = pd.DataFrame(cal_rows)
    print(cal_df.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    # Stage 1: Empirical Discovery across SL/TP profiles (Crypto portfolio + Gold reference)
    print("\n[3/4] Stage 1 — Empirical Discovery (1-Year Independent Signal Statistics):")
    profiles = [
        ("SL5_TP3_NoBE", 5.0, 3.0, 3.0, False),
        ("SL5_TP4_BE3", 5.0, 4.0, 3.0, True),
        ("SL5_TP5_BE3", 5.0, 5.0, 3.0, True),
        ("SL5_TP5_NoBE", 5.0, 5.0, 3.0, False),
        ("SL3_TP3_NoBE", 3.0, 3.0, 3.0, False),
        ("SL3_TP5_BE3", 3.0, 5.0, 3.0, True),
        ("SL5_TP8_BE3", 5.0, 8.0, 3.0, True),
    ]

    profile_summary_rows = []
    discovery_dfs: Dict[str, pd.DataFrame] = {}

    crypto_dfs = {s: dfs[s] for s in cfg.symbols}
    crypto_sigs = {s: signals_by_symbol[s] for s in cfg.symbols}

    for pname, sl_u, tp_u, be_u, use_be in profiles:
        ddf = run_discovery_stats(dfs, signals_by_symbol, cfg.risk, sl_u, tp_u, be_u, use_be)
        discovery_dfs[pname] = ddf

        # Crypto-only W1+W2 summary
        c_w12 = ddf[(ddf["symbol"].isin(cfg.symbols)) & (ddf["setup_type"].isin(["W1_TREND", "W2_TREND"]))]
        wr = c_w12["is_win"].mean() * 100.0 if len(c_w12) else 0.0
        pos = c_w12.loc[c_w12["net_units"] > 0, "net_units"].sum()
        neg = -c_w12.loc[c_w12["net_units"] < 0, "net_units"].sum()
        pf = pos / neg if neg > 0 else 0.0
        profile_summary_rows.append(
            {
                "Profile": pname,
                "SL_U": sl_u,
                "TP_U": tp_u,
                "BE": "Yes" if use_be else "No",
                "Crypto_W1W2_Trades": len(c_w12),
                "Winrate_%": wr,
                "PF": pf,
                "Avg_Net_U": c_w12["net_units"].mean() if len(c_w12) else 0.0,
                "Total_Net_U": c_w12["net_units"].sum() if len(c_w12) else 0.0,
            }
        )

    profile_summary_df = pd.DataFrame(profile_summary_rows)
    print("\n--- Profile Comparison (BTC+ETH+BNB, Wave 1 & Wave 2) ---")
    print(profile_summary_df.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    # Detailed Wave breakdown for SL5_TP3_NoBE (high winrate scalping) and SL5_TP5_BE3 (method 1 default)
    best_by_pf = profile_summary_df.sort_values("PF", ascending=False).iloc[0]["Profile"]
    print(f"\n--- Detailed Breakdown by Wave & Symbol ({best_by_pf}) ---")
    wave_sym_df = summarize_discovery(discovery_dfs[best_by_pf], ["symbol", "setup_type"])
    print(wave_sym_df.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    print("\n--- Detailed Breakdown by Wave (SL5_TP3_NoBE — Scalping Ăn Nhanh) ---")
    wave_scalp_df = summarize_discovery(discovery_dfs["SL5_TP3_NoBE"], ["symbol", "setup_type"])
    print(wave_scalp_df.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    print(f"\n--- Filter Impact: PA+Vol & EMA Alignment on Crypto ({best_by_pf}) ---")
    crypto_best = discovery_dfs[best_by_pf][discovery_dfs[best_by_pf]["symbol"].isin(cfg.symbols)]
    filter_df = summarize_discovery(crypto_best, ["setup_type", "pa_vol_ok", "ema_aligned"])
    print(filter_df.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    # Stage 2: Portfolio 20-step Compounding Backtest on BTC+ETH+BNB
    print("\n[4/4] Stage 2 — Portfolio 20-Step Compounding Simulation ($10,000, Max 2 Open, Max 10/Day):")
    # Find midpoint timestamp for In-Sample (first 6m) vs Out-of-Sample (last 6m)
    all_ts = sorted(s.timestamp for sigs in crypto_sigs.values() for s in sigs)
    mid_ts = all_ts[len(all_ts) // 2]

    portfolio_runs = {}

    for pname, sl_u, tp_u, be_u, use_be in [
        ("SL5_TP3_NoBE", 5.0, 3.0, 3.0, False),
        ("SL5_TP5_BE3", 5.0, 5.0, 3.0, True),
        ("SL5_TP8_BE3", 5.0, 8.0, 3.0, True),
    ]:
        ddf = discovery_dfs[pname]
        c_ddf = ddf[ddf["symbol"].isin(cfg.symbols)]

        # Measure empirical winrates from FIRST 6 MONTHS only (In-Sample)
        in_sample = c_ddf[c_ddf["timestamp"] < mid_ts]
        wr_map_oos: Dict[str, float] = {}
        for (sym, stype), g in in_sample.groupby(["symbol", "setup_type"]):
            if len(g) >= 20:
                wr_map_oos[f"{sym}:{stype}"] = float(g["is_win"].mean())

        rcfg = RiskConfig(
            sl_units=sl_u,
            tp_units=tp_u,
            be_trigger_units=be_u,
            enable_be=use_be,
        )

        # Run Out-of-Sample (last 6 months) with empirical winrates from first 6 months
        trades_oos, eq_oos, comp_oos = run_portfolio_backtest(
            crypto_dfs, crypto_sigs, wr_map_oos, rcfg, start_time=mid_ts
        )
        m_oos = compute_portfolio_metrics(trades_oos, eq_oos, rcfg.initial_capital, comp_oos.max_step_reached)
        portfolio_runs[f"{pname}_OOS6M"] = {
            "label": f"{pname} — Out-of-Sample 6 Tháng Cuối (Rule ≥51%)",
            "trades": trades_oos,
            "equity_curve": eq_oos,
            "metrics": m_oos,
        }

        # Run Full 12 Months using full-year empirical winrates
        wr_map_full: Dict[str, float] = {}
        for (sym, stype), g in c_ddf.groupby(["symbol", "setup_type"]):
            if len(g) >= 20:
                wr_map_full[f"{sym}:{stype}"] = float(g["is_win"].mean())

        trades_full, eq_full, comp_full = run_portfolio_backtest(
            crypto_dfs, crypto_sigs, wr_map_full, rcfg
        )
        m_full = compute_portfolio_metrics(trades_full, eq_full, rcfg.initial_capital, comp_full.max_step_reached)
        portfolio_runs[f"{pname}_FULL12M"] = {
            "label": f"{pname} — Toàn Bộ 12 Tháng (Rule ≥51%)",
            "trades": trades_full,
            "equity_curve": eq_full,
            "metrics": m_full,
        }

        print(
            f"  [{pname:12s}] FULL 12M: Equity=${m_full['final_equity']:9,.2f} ({m_full['return_pct']:+6.2f}%) | "
            f"Step {m_full['max_step']:2d}/20 | Trades={m_full['total_trades']:4d} | WR={m_full['winrate_pct']:5.2f}% | "
            f"PF={m_full['profit_factor']:.2f} | MaxDD={m_full['max_dd_pct']:6.2f}% | Fees=${m_full['total_fees']:,.0f}"
        )
        print(
            f"  [{pname:12s}] OOS   6M: Equity=${m_oos['final_equity']:9,.2f} ({m_oos['return_pct']:+6.2f}%) | "
            f"Step {m_oos['max_step']:2d}/20 | Trades={m_oos['total_trades']:4d} | WR={m_oos['winrate_pct']:5.2f}% | "
            f"PF={m_oos['profit_factor']:.2f} | MaxDD={m_oos['max_dd_pct']:6.2f}% | Fees=${m_oos['total_fees']:,.0f}"
        )

    out_dir = os.path.join(ROOT, "outputs")
    report_path = export_html_report(
        output_dir=out_dir,
        summary_tables={
            "1_Unit_Calibration_200_Buffer": cal_df,
            "2_SL_TP_Profile_Comparison": profile_summary_df,
            f"3_Wave_By_Symbol_{best_by_pf}": wave_sym_df,
            "4_Wave_By_Symbol_SL5_TP3_Scalp": wave_scalp_df,
            f"5_Filter_Impact_{best_by_pf}": filter_df,
        },
        portfolio_runs=portfolio_runs,
    )
    print(f"\nDone! Interactive HTML report saved to: {report_path}")


if __name__ == "__main__":
    main()
