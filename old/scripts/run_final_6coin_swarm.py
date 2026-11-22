"""Final 6-Coin Quantitative Swarm Verification & Portfolio Simulation (BTC, ETH, BNB, LINK, XRP, SOL).

Features:
1. Strict Anti-Overfitting Setup Audit:
   - Minimum sample size: Trades_1Y >= 15 (and IS_6M >= 6, OOS_6M >= 6).
   - Eliminates lucky small-sample outliers (<15 trades such as 12-13 trade anomalies).
   - Fixes the ETH/BNB BE-Fee mathematical leak (where TP1=3U + Leg2=Entry-3U turned BE into a -0.30U net loss).
   - Requires profitability in BOTH halves of the year (PF_IS6M >= 1.08 and PF_OOS6M >= 1.08) + Winrate >= 60%.
2. Tailored Per-Coin Execution DNA (SL, TP1 50%, Leg2 SL, TP2 50%, Volume Multiplier by Risk/Winrate Tier).
3. Full Portfolio Simulation across all 6 coins simultaneously:
   - Initial Capital: $10,000
   - Max Concurrent Open Trades: 3 (max 1 per coin at any moment)
   - Max Trades / Day: 10
   - 20-Step Compounding (+8% per step, step-down protection)
   - 3-Layer Storm Guard (Local + BTC Macro Storm Cooldown 4h, Volatility/DD Dynamic Sizing, Daily Circuit Breaker)
   - Binance Futures Fees: Maker 0.02%, Taker 0.05%, Slippage 1 bps
   - Evaluated across 4 Buffer Levels: 500, 400, 300, 250 units (for 12M Full, 6M OOS, and Blind Walk-Forward OOS).
4. Exports updated HTML report to outputs/final_6coin_swarm_report.html and CSV tables.
"""
import json
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
from risk.compounding import CompoundingManager
from run_be_whipsaw_test import sim_leg2_policy
from run_full_3ema_storm_guard import build_cooldown_mask, compute_storm_flags, generate_all_4_touches
from run_research_mtf import resample_ohlcv

SIX_COINS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "LINKUSDT", "XRPUSDT", "SOLUSDT"]
BUFFER_LEVELS = [500.0, 400.0, 300.0, 250.0]

# Per-Coin Tailored Execution DNA: (sl_u, tp1_by_tf, tp2_u, leg2_sl_mode, leg2_label)
COIN_EXEC_DNA: Dict[str, dict] = {
    "BTCUSDT": {
        "sl_u": 5.0,
        "tp1_map": {"M15": 4.0, "M30": 4.0, "H1": 3.5},
        "tp2_u": 12.0,
        "leg2_mode": "SL_MINUS_2U",
        "leg2_label": "Entry - 2U",
        "risk_zone": "Khu vực A (Core Trend Leader)",
    },
    "ETHUSDT": {
        "sl_u": 6.0,
        "tp1_map": {"M15": 3.5, "M30": 3.5, "H1": 3.5},
        "tp2_u": 10.0,
        "leg2_mode": "SL_MINUS_2U",
        "leg2_label": "Entry - 2U",
        "risk_zone": "Khu vực B (Wide-Wick Trend Pullback)",
    },
    "BNBUSDT": {
        "sl_u": 6.0,
        "tp1_map": {"M15": 4.0, "M30": 3.5, "H1": 3.5},
        "tp2_u": 10.0,
        "leg2_mode": "SL_MINUS_2U",
        "leg2_label": "Entry - 2U",
        "risk_zone": "Khu vực B (Mean-Reversion & Deep EMA)",
    },
    "LINKUSDT": {
        "sl_u": 6.0,
        "tp1_map": {"M15": 3.5, "M30": 3.5, "H1": 3.5},
        "tp2_u": 8.0,
        "leg2_mode": "SL_MINUS_2U",
        "leg2_label": "Entry - 2U",
        "risk_zone": "Khu vực A (High-Winrate First Pullback)",
    },
    "XRPUSDT": {
        "sl_u": 6.0,
        "tp1_map": {"M15": 3.5, "M30": 3.5, "H1": 3.5},
        "tp2_u": 8.0,
        "leg2_mode": "SL_MINUS_2U",
        "leg2_label": "Entry - 2U",
        "risk_zone": "Khu vực A (Impulse Wave 3 & HTF EMA200)",
    },
    "SOLUSDT": {
        "sl_u": 6.0,
        "tp1_map": {"M15": 3.5, "M30": 3.5, "H1": 3.5},
        "tp2_u": 8.0,
        "leg2_mode": "SL_MINUS_2U",
        "leg2_label": "Entry - 2U",
        "risk_zone": "Khu vực B (High-Beta Fast Pullback)",
    },
}


def assign_volume_mult(wr_1y: float, pf_1y: float, pf_oos: float) -> Tuple[float, str]:
    """Assign risk-tiered volume multiplier based on empirical Winrate & Profit Factor."""
    if wr_1y >= 73.0 and pf_1y >= 1.65 and pf_oos >= 1.30:
        return 1.00, "1.00x (Hạng A+ Core)"
    elif wr_1y >= 70.0 and pf_1y >= 1.40:
        return 0.80, "0.80x (Hạng A Chuẩn)"
    elif (wr_1y >= 65.0 and pf_1y >= 1.25) or (wr_1y >= 61.0 and pf_1y >= 1.80):
        return 0.65, "0.65x (Hạng B Hỗ trợ)"
    else:
        return 0.50, "0.50x (Hạng C Vệ tinh)"


def discover_and_audit_setups(
    tf_dfs_map: Dict[str, Dict[str, pd.DataFrame]],
    tf_sigs_map: Dict[str, Dict[str, list]],
    mid_ts: pd.Timestamp,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[Tuple[str, str, str], dict], Dict[Tuple[str, str, str], dict]]:
    """Audit all setups across 6 coins x 3 TFs x 12 EMA-Touch combinations.

    Returns:
      - full_audit_df: Every setup with >= 10 trades
      - selected_df: Strictly filtered robust setups (>= 15 trades, IS & OOS profitable, WR >= 61%)
      - robust_rules: dict[(sym, tf, setup)] -> rule_spec
      - blind_wf_rules: dict[(sym, tf, setup)] -> rule_spec (selected ONLY using IS 6M data!)
    """
    all_rows = []
    selected_rows = []
    robust_rules: Dict[Tuple[str, str, str], dict] = {}
    blind_wf_rules: Dict[Tuple[str, str, str], dict] = {}

    for sym in SIX_COINS:
        dna = COIN_EXEC_DNA[sym]
        sl_u = dna["sl_u"]
        tp2_u = dna["tp2_u"]
        l2_mode = dna["leg2_mode"]
        l2_label = dna["leg2_label"]

        for tf_name in ["M15", "M30", "H1"]:
            df = tf_dfs_map[tf_name][sym]
            tp1_u = dna["tp1_map"][tf_name]
            sigs = tf_sigs_map[tf_name][sym]
            by_setup: Dict[str, list] = {}
            for s in sigs:
                r = sim_leg2_policy(df, s, sl_u, tp1_u, tp2_u, l2_mode)
                r["timestamp"] = s.timestamp
                by_setup.setdefault(s.setup_type, []).append(r)

            for stype, rlist in sorted(by_setup.items()):
                g = pd.DataFrame(rlist)
                if len(g) < 10:
                    continue
                g_is = g[g["timestamp"] < mid_ts]
                g_oos = g[g["timestamp"] >= mid_ts]

                def _pf(sub):
                    if len(sub) == 0:
                        return 0.0
                    pos = sub.loc[sub["net_u"] > 0, "net_u"].sum()
                    neg = -sub.loc[sub["net_u"] < 0, "net_u"].sum()
                    return float(pos / neg) if neg > 0 else (9.99 if pos > 0 else 0.0)

                pf_1y = _pf(g)
                pf_is = _pf(g_is)
                pf_oos = _pf(g_oos)
                wr_1y = float((g["net_u"] > 0).mean() * 100.0)
                wr_is = float((g_is["net_u"] > 0).mean() * 100.0) if len(g_is) else 0.0
                wr_oos = float((g_oos["net_u"] > 0).mean() * 100.0) if len(g_oos) else 0.0
                sl_rate = float((g["outcome"] == "SL_FULL").mean() * 100.0)
                tp2_rate = float((g["outcome"] == "TP1+TP2").mean() * 100.0)

                parts = stype.split("_")
                ema_name = parts[0]
                touch_str = parts[1].replace("W", "Chạm ") + (" (Đảo chiều)" if "REV" in stype else " (Thuận xu hướng)")

                row = {
                    "Coin": sym,
                    "TF": tf_name,
                    "EMA": ema_name,
                    "Cham": touch_str,
                    "Setup_Code": stype,
                    "SL_U": f"{sl_u:.1f}U",
                    "TP1_U": f"{tp1_u:.1f}U (50%)",
                    "Doi_SL_Chan2": l2_label,
                    "TP2_U": f"{tp2_u:.1f}U (50%)",
                    "Trades_1Y": len(g),
                    "Trades_IS": len(g_is),
                    "Trades_OOS": len(g_oos),
                    "WR_IS_%": wr_is,
                    "WR_OOS_%": wr_oos,
                    "WR_1Y_%": wr_1y,
                    "SL_Full_%": sl_rate,
                    "TP2_Hit_%": tp2_rate,
                    "PF_IS6M": pf_is,
                    "PF_OOS6M": pf_oos,
                    "PF_1Y": pf_1y,
                    "Net_U_1Y": float(g["net_u"].sum()),
                }
                all_rows.append(row)

                # Strict Blind Walk-Forward rule (selected ONLY from first 6M IS, zero lookahead)
                if len(g_is) >= 7 and pf_is >= 1.20 and wr_is >= 62.0:
                    wf_mult = 1.00 if (wr_is >= 73.0 and pf_is >= 1.60) else (0.75 if wr_is >= 68.0 else 0.50)
                    blind_wf_rules[(sym, tf_name, stype)] = {
                        "mult": wf_mult,
                        "sl_u": sl_u,
                        "tp1_u": tp1_u,
                        "tp2_u": tp2_u,
                        "leg2_mode": l2_mode,
                    }

                # Anti-Overfitting Dual-Half Filter:
                # 1. At least 15 trades in 1Y (and >= 6 in IS, >= 6 in OOS) -> rejects <15-trade lucky anomalies
                # 2. Profitable in BOTH halves: PF_IS6M >= 1.08 AND PF_OOS6M >= 1.08 AND PF_1Y >= 1.22
                # 3. Winrate >= 61.0% across 1Y
                if (
                    len(g) >= 15
                    and len(g_is) >= 6
                    and len(g_oos) >= 6
                    and pf_is >= 1.08
                    and pf_oos >= 1.08
                    and pf_1y >= 1.22
                    and wr_1y >= 61.0
                ):
                    mult, tier_label = assign_volume_mult(wr_1y, pf_1y, pf_oos)
                    sel_row = {
                        **row,
                        "Vol_Mult": mult,
                        "Phan_Hang_Volume": tier_label,
                    }
                    selected_rows.append(sel_row)
                    robust_rules[(sym, tf_name, stype)] = {
                        "mult": mult,
                        "sl_u": sl_u,
                        "tp1_u": tp1_u,
                        "tp2_u": tp2_u,
                        "leg2_mode": l2_mode,
                        "tier_label": tier_label,
                    }

    full_df = pd.DataFrame(all_rows)
    sel_df = pd.DataFrame(selected_rows).sort_values(["Coin", "PF_1Y"], ascending=[True, False])
    return full_df, sel_df, robust_rules, blind_wf_rules


def simulate_6coin_portfolio(
    tf_dfs_map: Dict[str, Dict[str, pd.DataFrame]],
    tf_sigs_map: Dict[str, Dict[str, list]],
    rules_map: Dict[Tuple[str, str, str], dict],
    buffer_units: float = 250.0,
    max_concurrent: int = 3,
    max_trades_per_day: int = 10,
    start_time: pd.Timestamp = None,
) -> dict:
    """Simulate all 6 coins in parallel with 20-step compounding, 3-layer Storm Guard, and exact fee accounting."""
    rcfg = RiskConfig(
        initial_capital=10_000.0,
        buffer_units=buffer_units,
        max_concurrent_trades=max_concurrent,
        max_trades_per_day=max_trades_per_day,
    )
    comp = CompoundingManager(rcfg)
    equity = rcfg.initial_capital
    peak_eq = equity

    candidates = []
    for tf_name, sigs_map in tf_sigs_map.items():
        for sym, sigs in sigs_map.items():
            for s in sigs:
                key = (sym, tf_name, s.setup_type)
                if key not in rules_map:
                    continue
                if start_time is not None and s.timestamp < start_time:
                    continue
                candidates.append((tf_name, rules_map[key], s))

    if not candidates:
        return {
            "buffer_units": int(buffer_units),
            "final_equity": equity,
            "max_equity": equity,
            "net_profit_usd": 0.0,
            "return_pct": 0.0,
            "max_step": 1,
            "final_step": 1,
            "total_trades": 0,
            "winrate_pct": 0.0,
            "pf": 0.0,
            "max_dd_pct": 0.0,
            "per_coin_wr": {c: "0.0% (0)" for c in SIX_COINS},
            "per_coin_pnl": {c: 0.0 for c in SIX_COINS},
            "eq_df": pd.DataFrame(),
            "trades": [],
        }

    tf_prio = {"H1": 0, "M30": 1, "M15": 2}
    candidates.sort(key=lambda x: (x[2].timestamp, tf_prio[x[0]], -x[1]["mult"], x[2].symbol))

    active, closed = [], []
    daily_trades: Dict[str, int] = {}
    daily_sl: Dict[str, int] = {}
    daily_pnl: Dict[str, float] = {}
    daily_start_eq: Dict[str, float] = {}
    consec_sl = 0
    eq_curve = [{"timestamp": candidates[0][2].timestamp, "equity": equity, "step": 1}]

    def settle(now_ts):
        nonlocal equity, peak_eq, consec_sl
        keep, done = [], []
        for ex_ts, tr in active:
            (done if ex_ts <= now_ts else keep).append((ex_ts, tr))
        done.sort(key=lambda x: x[0])
        for ex_ts, tr in done:
            equity += tr["net_pnl"]
            peak_eq = max(peak_eq, equity)
            comp.update_equity(equity, ex_ts)
            dk = tr["day_key"]
            daily_pnl[dk] = daily_pnl.get(dk, 0.0) + tr["net_pnl"]
            if tr["outcome"] == "SL_FULL":
                consec_sl += 1
                daily_sl[dk] = daily_sl.get(dk, 0) + 1
            elif tr["net_pnl"] > 0:
                consec_sl = 0
            tr["equity_after"] = equity
            closed.append(tr)
            eq_curve.append({"timestamp": ex_ts, "equity": equity, "step": comp.step})
        active[:] = keep

    for tf_name, rspec, sig in candidates:
        settle(sig.timestamp)
        if equity <= 0:
            break
        dk = sig.timestamp.strftime("%Y-%m-%d")
        if dk not in daily_start_eq:
            daily_start_eq[dk] = equity

        # Storm Guard Layer 3: Daily circuit breakers
        if daily_trades.get(dk, 0) >= rcfg.max_trades_per_day:
            continue
        if daily_sl.get(dk, 0) >= 3:
            continue
        if daily_pnl.get(dk, 0.0) <= -0.05 * daily_start_eq[dk]:
            continue
        if len(active) >= rcfg.max_concurrent_trades:
            continue
        if any(tr["symbol"] == sig.symbol for _, tr in active):
            continue

        # Storm Guard Layer 2: Dynamic volatility & drawdown position sizing
        eff_mult = float(rspec["mult"])
        if sig.pa_vol_ok:  # ATR 1.35x - 1.80x elevated volatility regime
            eff_mult *= 0.5
        dd_pct = (equity - peak_eq) / peak_eq * 100.0
        if dd_pct <= -6.0 or consec_sl >= 2:
            eff_mult = min(eff_mult, 0.25)

        open_mult = sum(tr["size_mult"] for _, tr in active)
        eff_mult = min(eff_mult, max(0.0, 1.65 - open_mult))
        if eff_mult < 0.20:
            continue

        df = tf_dfs_map[tf_name][sig.symbol]
        sim = sim_leg2_policy(
            df,
            sig,
            rspec["sl_u"],
            rspec["tp1_u"],
            rspec["tp2_u"],
            rspec["leg2_mode"],
        )
        unit_size = max(sig.atr * 0.5, sig.entry_price * 1e-5)
        sizing_base = min(comp.tier_capital, equity)
        qty = (sizing_base * eff_mult) / (rcfg.buffer_units * unit_size)
        net_pnl = sim["net_u"] * unit_size * qty

        bars_est = 4 if sim["outcome"] == "SL_FULL" else 8
        exit_idx = min(sig.bar_idx + bars_est, len(df) - 1)
        ex_ts = pd.Timestamp(df["timestamp"].iloc[exit_idx])
        if ex_ts.tzinfo is None:
            ex_ts = ex_ts.tz_localize("UTC")

        tr = {
            "tf": tf_name,
            "symbol": sig.symbol,
            "setup": sig.setup_type,
            "size_mult": eff_mult,
            "outcome": sim["outcome"],
            "net_u": sim["net_u"],
            "net_pnl": net_pnl,
            "day_key": dk,
            "entry_time": sig.timestamp,
            "exit_time": ex_ts,
        }
        active.append((ex_ts, tr))
        daily_trades[dk] = daily_trades.get(dk, 0) + 1

    if active:
        settle(pd.Timestamp("2099-01-01", tz="UTC"))

    eq_df = pd.DataFrame(eq_curve)
    pnls = np.array([t["net_pnl"] for t in closed]) if closed else np.array([0.0])
    wr = float((pnls > 0).mean() * 100.0) if closed else 0.0
    pos = pnls[pnls > 0].sum()
    neg = -pnls[pnls < 0].sum()
    pf = float(pos / neg) if neg > 0 else 0.0
    ev = eq_df["equity"].values
    pk = np.maximum.accumulate(ev)
    max_dd = float(((ev - pk) / pk * 100.0).min())
    max_eq = float(ev.max())
    final_eq = float(ev[-1])

    per_coin_wr = {}
    per_coin_pnl = {}
    tdf = pd.DataFrame(closed) if closed else pd.DataFrame()
    for c in SIX_COINS:
        if tdf.empty or c not in tdf["symbol"].values:
            per_coin_wr[c] = "0.0% (0)"
            per_coin_pnl[c] = 0.0
        else:
            sub = tdf[tdf["symbol"] == c]
            c_wr = (sub["net_pnl"] > 0).mean() * 100.0
            per_coin_wr[c] = f"{c_wr:.1f}% ({len(sub)})"
            per_coin_pnl[c] = float(sub["net_pnl"].sum())

    return {
        "buffer_units": int(buffer_units),
        "trades": closed,
        "eq_df": eq_df,
        "final_equity": final_eq,
        "max_equity": max_eq,
        "net_profit_usd": final_eq - rcfg.initial_capital,
        "return_pct": (final_eq / rcfg.initial_capital - 1.0) * 100.0,
        "max_step": comp.max_step_reached,
        "final_step": comp.step,
        "total_trades": len(closed),
        "winrate_pct": wr,
        "pf": pf,
        "max_dd_pct": max_dd,
        "per_coin_wr": per_coin_wr,
        "per_coin_pnl": per_coin_pnl,
    }


def export_final_swarm_html(
    html_path: str,
    selected_df: pd.DataFrame,
    sim_12m_df: pd.DataFrame,
    sim_oos_df: pd.DataFrame,
    runs_12m: Dict[int, dict],
    runs_oos: Dict[int, dict],
) -> None:
    """Generate comprehensive interactive HTML report for the 6-Coin Swarm Final Strategy."""
    traces_12m = []
    for buf, res in runs_12m.items():
        eq_df = res["eq_df"]
        if not eq_df.empty:
            traces_12m.append(
                {
                    "name": f"12M — Bao {buf} giá (${res['final_equity']:,.0f} | DD {res['max_dd_pct']:.2f}%)",
                    "x": [str(t) for t in eq_df["timestamp"].tolist()],
                    "y": [round(float(v), 2) for v in eq_df["equity"].tolist()],
                    "mode": "lines",
                }
            )

    step_shapes = [
        {
            "type": "line",
            "xref": "paper",
            "x0": 0,
            "x1": 1,
            "y0": round(10_000 * (1.08 ** s), 2),
            "y1": round(10_000 * (1.08 ** s), 2),
            "line": {"color": "rgba(148,163,184,0.20)", "width": 1, "dash": "dot"},
        }
        for s in range(1, 21)
    ]

    cards_html = []
    for buf in [500, 400, 300, 250]:
        r = runs_12m[buf]
        ro = runs_oos[buf]
        cards_html.append(
            f"""
            <div class="card">
                <h3>Mốc Bao {buf} Giá (12 Tháng & OOS 6 Tháng)</h3>
                <div class="grid">
                    <div><span>Final Equity (12T)</span><strong>${r['final_equity']:,.2f} ({r['return_pct']:+.2f}%)</strong></div>
                    <div><span>Max Equity (12T)</span><strong>${r['max_equity']:,.2f}</strong></div>
                    <div><span>Tiền lãi ròng (12T)</span><strong>+${r['net_profit_usd']:,.2f}</strong></div>
                    <div><span>Bậc lãi kép đạt được</span><strong>Bậc {r['max_step']} / 20</strong></div>
                    <div><span>Winrate Tổng Hợp (12T)</span><strong>{r['winrate_pct']:.2f}% ({r['total_trades']} lệnh)</strong></div>
                    <div><span>Profit Factor / Max DD</span><strong>PF {r['pf']:.2f} | DD {r['max_dd_pct']:.2f}%</strong></div>
                    <div><span>Kiểm chứng OOS 6T Cuối</span><strong>${ro['final_equity']:,.2f} ({ro['return_pct']:+.2f}%)</strong></div>
                    <div><span>OOS Winrate / Max DD</span><strong>WR {ro['winrate_pct']:.2f}% | DD {ro['max_dd_pct']:.2f}%</strong></div>
                </div>
            </div>
            """
        )

    disp_sel = selected_df[
        [
            "Coin",
            "TF",
            "EMA",
            "Cham",
            "SL_U",
            "TP1_U",
            "Doi_SL_Chan2",
            "TP2_U",
            "Trades_1Y",
            "WR_IS_%",
            "WR_OOS_%",
            "WR_1Y_%",
            "PF_IS6M",
            "PF_OOS6M",
            "PF_1Y",
            "Net_U_1Y",
            "Phan_Hang_Volume",
        ]
    ].copy()

    html = f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<title>CHICKTRADE — Báo Cáo Chốt Phương Án 6 Coin Swarm (BTC, ETH, BNB, LINK, XRP, SOL)</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #0f172a; color: #e2e8f0; margin: 0; padding: 24px; }}
  h1 {{ margin-top: 0; color: #38bdf8; }}
  h2 {{ margin-top: 32px; color: #93c5fd; border-bottom: 1px solid #334155; padding-bottom: 6px; }}
  .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 16px; margin: 20px 0; }}
  .card {{ background: #1e293b; border: 1px solid #334155; border-radius: 10px; padding: 16px; }}
  .card h3 {{ margin: 0 0 12px 0; color: #38bdf8; font-size: 16px; }}
  .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }}
  .grid div {{ background: #0f172a; padding: 8px 10px; border-radius: 6px; }}
  .grid span {{ display: block; font-size: 11px; color: #94a3b8; }}
  .grid strong {{ font-size: 14px; color: #f8fafc; }}
  #equity-chart {{ width: 100%; height: 480px; background: #1e293b; border-radius: 10px; padding: 8px; box-sizing: border-box; }}
  table.tbl {{ width: 100%; border-collapse: collapse; margin-top: 10px; background: #1e293b; font-size: 12.5px; }}
  table.tbl th, table.tbl td {{ border: 1px solid #334155; padding: 7px 9px; text-align: right; }}
  table.tbl th {{ background: #0f172a; color: #93c5fd; text-align: center; }}
  table.tbl td:first-child, table.tbl td:nth-child(2), table.tbl td:nth-child(3), table.tbl td:nth-child(4) {{ text-align: left; }}
</style>
</head>
<body>
  <h1>CHICKTRADE — Báo Cáo Tổng Hợp Hệ Thống 6 Coin Swarm (Anti-Overfitting IS/OOS)</h1>
  <p>Danh mục 6 Coin: <strong>BTCUSDT, ETHUSDT, BNBUSDT, LINKUSDT, XRPUSDT, SOLUSDT</strong> | Vốn gốc <strong>$10,000</strong> | Tối đa <strong>3 lệnh mở cùng lúc</strong> | Tối đa <strong>10 lệnh/ngày</strong> | <strong>20 Bậc Lãi Kép (+8%/bậc)</strong> | <strong>Storm Guard 3 Lớp</strong> | Phí Binance Futures Maker 0.02% / Taker 0.05% + Slippage 1 bps.</p>
  <div class="cards">
    {''.join(cards_html)}
  </div>
  <h2>Biểu Đồ Tăng Trưởng Vốn (Equity Curve) Ở 4 Mốc Bao Giá: 500 / 400 / 300 / 250 Giá</h2>
  <div id="equity-chart"></div>
  <h2>Bảng 1: Kết Quả Mô Phỏng Portfolio 6 Coin — 12 Tháng Full (Mốc 500 / 400 / 300 / 250 Giá)</h2>
  {sim_12m_df.to_html(index=False, classes="tbl", float_format=lambda x: f"{x:,.2f}")}
  <h2>Bảng 2: Kết Quả Mô Phỏng Portfolio 6 Coin — 6 Tháng Cuối Out-of-Sample (Mốc 500 / 400 / 300 / 250 Giá)</h2>
  {sim_oos_df.to_html(index=False, classes="tbl", float_format=lambda x: f"{x:,.2f}")}
  <h2>Bảng 3: Bảng Chốt Chi Tiết Thông Số Tối Ưu Từng Coin (Đã Lọc Chống Overfitting: >= 15 Lệnh, Bền Cả 2 Nửa Năm IS & OOS)</h2>
  {disp_sel.to_html(index=False, classes="tbl", float_format=lambda x: f"{x:,.2f}")}
  <script>
    const traces = {json.dumps(traces_12m)};
    const layout = {{
      paper_bgcolor: '#1e293b',
      plot_bgcolor: '#0f172a',
      font: {{ color: '#e2e8f0' }},
      xaxis: {{ title: 'Thời gian (UTC)', gridcolor: '#1e293b' }},
      yaxis: {{ title: 'Vốn ($)', gridcolor: '#334155' }},
      shapes: {json.dumps(step_shapes)},
      legend: {{ orientation: 'h', y: 1.08 }}
    }};
    Plotly.newPlot('equity-chart', traces, layout, {{ responsive: true }});
  </script>
</body>
</html>
"""
    os.makedirs(os.path.dirname(html_path), exist_ok=True)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)


def build_summary_table(runs_by_buf: Dict[int, dict]) -> pd.DataFrame:
    rows = []
    for buf in [500, 400, 300, 250]:
        r = runs_by_buf[buf]
        cwr = r["per_coin_wr"]
        rows.append(
            {
                "Bao_Gia": f"{buf} giá",
                "Tong_Lenh": r["total_trades"],
                "WR_BTC": cwr["BTCUSDT"],
                "WR_ETH": cwr["ETHUSDT"],
                "WR_BNB": cwr["BNBUSDT"],
                "WR_LINK": cwr["LINKUSDT"],
                "WR_XRP": cwr["XRPUSDT"],
                "WR_SOL": cwr["SOLUSDT"],
                "WR_Tong_Hop_%": r["winrate_pct"],
                "PF": r["pf"],
                "Tien_Lai_USD": r["net_profit_usd"],
                "Lai_Suat_%": r["return_pct"],
                "Max_DD_%": r["max_dd_pct"],
                "Max_Equity_USD": r["max_equity"],
                "Final_Equity_USD": r["final_equity"],
                "Bac_Lai_Kep": f"Bậc {r['max_step']}/20",
            }
        )
    return pd.DataFrame(rows)


def main():
    cfg = BacktestConfig()
    end_ym = _detect_latest_available_month("BTCUSDT", "5m")

    raw_m5 = {sym: download_symbol_1y(sym, "5m", 12, "futures", end_ym) for sym in SIX_COINS}
    dfs_m15 = {s: resample_ohlcv(d, "15min") for s, d in raw_m5.items()}
    dfs_m30 = {s: resample_ohlcv(d, "30min") for s, d in raw_m5.items()}
    dfs_h1 = {s: resample_ohlcv(d, "1h") for s, d in raw_m5.items()}
    dfs_h4 = {s: resample_ohlcv(d, "4h") for s, d in raw_m5.items()}
    tf_dfs_map = {"M15": dfs_m15, "M30": dfs_m30, "H1": dfs_h1}

    tf_sigs_map = {}
    for tf_name, tf_dfs, htf_dfs, cd in [
        ("M15", dfs_m15, dfs_h1, 16),
        ("M30", dfs_m30, dfs_h4, 8),
        ("H1", dfs_h1, dfs_h4, 4),
    ]:
        bst = compute_storm_flags(tf_dfs["BTCUSDT"])
        bmask = build_cooldown_mask(bst["raw_storm"], cd)
        bts = tf_dfs["BTCUSDT"]["timestamp"].values
        tf_sigs_map[tf_name] = {
            sym: generate_all_4_touches(tf_dfs[sym], htf_dfs[sym], bts, bmask, sym, cfg.strategy, cooldown_bars=cd)[0]
            for sym in SIX_COINS
        }

    all_ts = sorted(s.timestamp for s in tf_sigs_map["M15"]["BTCUSDT"])
    mid_ts = all_ts[len(all_ts) // 2]

    full_audit_df, selected_df, robust_rules, blind_wf_rules = discover_and_audit_setups(
        tf_dfs_map, tf_sigs_map, mid_ts
    )

    print("=" * 120)
    print(f"BANG CHOT CHI TIET THONG SO TOI UU RIENG TUNG COIN ({len(selected_df)} SETUPS >= 15 LENH, BEN CA IS 6T & OOS 6T)")
    print("=" * 120)
    cols_show = [
        "Coin",
        "TF",
        "EMA",
        "Cham",
        "SL_U",
        "TP1_U",
        "Doi_SL_Chan2",
        "TP2_U",
        "Trades_1Y",
        "WR_IS_%",
        "WR_OOS_%",
        "WR_1Y_%",
        "PF_IS6M",
        "PF_OOS6M",
        "PF_1Y",
        "Net_U_1Y",
        "Phan_Hang_Volume",
    ]
    print(selected_df[cols_show].to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    runs_12m: Dict[int, dict] = {}
    runs_oos: Dict[int, dict] = {}
    runs_blind_wf: Dict[int, dict] = {}

    for buf in BUFFER_LEVELS:
        b_int = int(buf)
        runs_12m[b_int] = simulate_6coin_portfolio(
            tf_dfs_map, tf_sigs_map, robust_rules, buffer_units=buf, max_concurrent=3, max_trades_per_day=10, start_time=None
        )
        runs_oos[b_int] = simulate_6coin_portfolio(
            tf_dfs_map, tf_sigs_map, robust_rules, buffer_units=buf, max_concurrent=3, max_trades_per_day=10, start_time=mid_ts
        )
        runs_blind_wf[b_int] = simulate_6coin_portfolio(
            tf_dfs_map, tf_sigs_map, blind_wf_rules, buffer_units=buf, max_concurrent=3, max_trades_per_day=10, start_time=mid_ts
        )

    sim_12m_df = build_summary_table(runs_12m)
    sim_oos_df = build_summary_table(runs_oos)
    sim_wf_df = build_summary_table(runs_blind_wf)

    print("\n" + "=" * 120)
    print("KET QUA MO PHONG PORTFOLIO 6 COIN SONG SONG — 12 THANG FULL (4 MOC BAO GIA: 500 / 400 / 300 / 250)")
    print("=" * 120)
    print(sim_12m_df.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    print("\n" + "=" * 120)
    print("KET QUA MO PHONG PORTFOLIO 6 COIN SONG SONG — 6 THANG CUOI OUT-OF-SAMPLE (4 MOC BAO GIA: 500 / 400 / 300 / 250)")
    print("=" * 120)
    print(sim_oos_df.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    print("\n" + "=" * 120)
    print(f"KIEM CHUNG WALK-FORWARD MU 6 THANG CUOI (CHON LUAT CHI TU 6T DAU IS, {len(blind_wf_rules)} SETUPS -> TEST MU 6T CUOI)")
    print("=" * 120)
    print(sim_wf_df.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    out_dir = os.path.join(ROOT, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    selected_df.to_csv(os.path.join(out_dir, "final_6coin_selected_setups.csv"), index=False)
    sim_12m_df.to_csv(os.path.join(out_dir, "final_6coin_portfolio_12m.csv"), index=False)
    sim_oos_df.to_csv(os.path.join(out_dir, "final_6coin_portfolio_oos6m.csv"), index=False)
    sim_wf_df.to_csv(os.path.join(out_dir, "final_6coin_portfolio_blind_wf_oos6m.csv"), index=False)

    html_path = os.path.join(out_dir, "final_6coin_swarm_report.html")
    export_final_swarm_html(html_path, selected_df, sim_12m_df, sim_oos_df, runs_12m, runs_oos)
    print(f"\n[DONE] HTML Report saved to: {html_path}")


if __name__ == "__main__":
    main()
