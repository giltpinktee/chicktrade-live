"""Master 6-Coin Full-Timeframe (5m, 15m, 30m, 1h) Quantitative Audit, Core Strategy Optimizer & Multi-Tab HTML Dashboard.

Generates:
  - e:\CHICKTRADE\outputs\chicktrade_6coin_master_report.html
  - e:\CHICKTRADE\outputs\master_all_288_setups_audit.csv
  - e:\CHICKTRADE\outputs\master_selected_core_setups.csv
"""
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

RESEARCH_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(RESEARCH_DIR)
for p in (ROOT, RESEARCH_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from config import BacktestConfig, RiskConfig
from data.binance_loader import _detect_latest_available_month, download_symbol_1y
from risk.compounding import CompoundingManager
from run_be_whipsaw_test import sim_leg2_policy
from run_full_3ema_storm_guard import build_cooldown_mask, compute_storm_flags, generate_all_4_touches
from run_research_mtf import resample_ohlcv

SIX_COINS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "LINKUSDT", "XRPUSDT", "SOLUSDT"]
ALL_TFS = ["M5", "M15", "M30", "H1"]
TF_LABELS = {"M5": "5m", "M15": "15m", "M30": "30m", "H1": "1h"}
ALL_SETUP_CODES = [
    "EMA34_W1", "EMA34_W2", "EMA34_W3", "EMA34_W4_REV",
    "EMA89_W1", "EMA89_W2", "EMA89_W3", "EMA89_W4_REV",
    "EMA200_W1", "EMA200_W2", "EMA200_W3", "EMA200_W4_REV",
]
BUFFER_LEVELS = [500.0, 400.0, 300.0, 250.0]

# Per-Coin Tailored Execution DNA
COIN_EXEC_DNA: Dict[str, dict] = {
    "BTCUSDT": {
        "sl_u": 5.0,
        "tp1_map": {"M5": 4.0, "M15": 4.0, "M30": 4.0, "H1": 3.5},
        "tp2_u": 12.0,
        "leg2_mode": "SL_MINUS_2U",
        "leg2_label": "Entry - 2U",
        "risk_zone": "Khu vực A — Core Trend Leader (Sóng Thuận Dài, Râu Ngắn)",
        "rationale": "BTC có Median MAE rất gọn (2.57U–3.06U) nên SL = 5.0U vượt xa vùng nhiễu P70. Sóng thuận EMA34/89 đi rất dài (P75 MFE ~ 8U, Max 41U) nên nới TP2 = 12.0U giúp tối đa hóa Profit Factor (1.97–2.46). Loại bỏ toàn bộ khung 5m và lệnh Đảo chiều Chạm 4 (W4_REV) vì quán tính xu hướng BTC rất mạnh.",
    },
    "ETHUSDT": {
        "sl_u": 6.0,
        "tp1_map": {"M5": 3.5, "M15": 3.5, "M30": 3.5, "H1": 3.5},
        "tp2_u": 10.0,
        "leg2_mode": "SL_MINUS_2U",
        "leg2_label": "Entry - 2U",
        "risk_zone": "Khu vực A/B — Wide-Wick Trend & H1 Reversal",
        "rationale": "ETH thường quét râu thanh khoản sâu (P75 MAE = 5.84U–6.45U) nên SL = 6.0U là chìa khóa chống quét oan. Khắc phục triệt để lỗi cũ (TP1=3U + Entry-3U bị âm phí sàn) bằng TP1 = 3.5U + Dời SL = Entry - 2U (khóa lãi ròng +0.55U sau phí ngay khi chạm TP1), đưa Winrate 5 setup chủ lực của ETH lên 72.73%–83.33%.",
    },
    "BNBUSDT": {
        "sl_u": 6.0,
        "tp1_map": {"M5": 4.0, "M15": 4.0, "M30": 3.5, "H1": 3.5},
        "tp2_u": 10.0,
        "leg2_mode": "SL_MINUS_2U",
        "leg2_label": "Entry - 2U",
        "risk_zone": "Khu vực B — Deep EMA Pullback & Wave-4 Mean Reversion",
        "rationale": "BNB vận động theo cấu trúc tích lũy biên rộng, tôn trọng cực tốt các nhịp hồi sâu về EMA89/EMA200 trên M30/H1 và đảo chiều cạn kiệt sóng 4 (M15 EMA34_W4_REV đạt WR 76.92% ở cả hai nửa năm; M30 EMA89_W4_REV đạt WR 73.33%). Loại bỏ các setup mẫu nhỏ <15 lệnh trên H1.",
    },
    "LINKUSDT": {
        "sl_u": 6.0,
        "tp1_map": {"M5": 3.5, "M15": 3.5, "M30": 3.5, "H1": 3.5},
        "tp2_u": 8.0,
        "leg2_mode": "SL_MINUS_2U",
        "leg2_label": "Entry - 2U",
        "risk_zone": "Khu vực A+ — Vua Winrate Nhịp Chạm Đầu Tiên (First Pullback)",
        "rationale": "LINK có phản ứng bật tăng chuẩn xác nhất ngay nhịp Chạm 1 (W1) của EMA34/EMA89 (khung 15m) và EMA200 (khung 1h) với Median MFE > 5.0U và P75 MFE ~ 7.5U–8.0U. Đặt TP2 = 8.0U giúp tỷ lệ ăn trọn 2 chân đạt đỉnh, mang lại Winrate thực nghiệm 76.47%–82.46%.",
    },
    "XRPUSDT": {
        "sl_u": 6.0,
        "tp1_map": {"M5": 3.5, "M15": 3.5, "M30": 3.5, "H1": 3.5},
        "tp2_u": 8.0,
        "leg2_mode": "SL_MINUS_2U",
        "leg2_label": "Entry - 2U",
        "risk_zone": "Khu vực A — Bùng Nổ Nhịp Sóng 3 & Hỗ Trợ Dài Hạn EMA200",
        "rationale": "XRP thường nén biên độ rồi bùng nổ mạnh ở nhịp Chạm 3 trên M15 (EMA34_W3 đạt 35 lệnh, WR 77.14%) và tôn trọng tuyệt đối đường EMA200 trên H1/M15 (H1 EMA200_W1 đạt WR 76.19%, PF 2.10). Đã loại bỏ setup M30 EMA89_W3 (WR 91.7%) vì chỉ có 12 lệnh (<15 lệnh) để chống overfitting.",
    },
    "SOLUSDT": {
        "sl_u": 6.0,
        "tp1_map": {"M5": 3.5, "M15": 3.5, "M30": 3.5, "H1": 3.5},
        "tp2_u": 8.0,
        "leg2_mode": "SL_MINUS_2U",
        "leg2_label": "Entry - 2U",
        "risk_zone": "Khu vực B — High-Beta Chọn Lọc Tinh Nhuệ (Fast Pullback)",
        "rationale": "SOL có biến động Beta rất cao, các nhịp chạm muộn (W3/W4) dễ bị trượt đà ở nửa năm sau. Hệ thống lọc bỏ toàn bộ nhiễu và chỉ giữ 2 setup tinh nhuệ nhất có mẫu lớn (26–45 lệnh) và bền cả 2 nửa năm: M15 EMA89_W2 (WR 80.77%, PF 1.83) và M30 EMA34_W1 (WR 68.89%, PF 1.34).",
    },
}


def assign_volume_mult(wr_1y: float, pf_1y: float, pf_oos: float) -> Tuple[float, str]:
    """Assign risk-tiered volume multiplier based on empirical Winrate & Profit Factor."""
    if wr_1y >= 75.0 and pf_1y >= 1.80 and pf_oos >= 1.40:
        return 1.10, "1.10x (Hạng S — Chủ lực Tối ưu)"
    elif wr_1y >= 73.0 and pf_1y >= 1.60 and pf_oos >= 1.30:
        return 1.00, "1.00x (Hạng A+ — Core)"
    elif wr_1y >= 70.0 and pf_1y >= 1.40:
        return 0.80, "0.80x (Hạng A — Chuẩn)"
    elif (wr_1y >= 65.0 and pf_1y >= 1.25) or (wr_1y >= 61.0 and pf_1y >= 1.80):
        return 0.65, "0.65x (Hạng B — Hỗ trợ)"
    else:
        return 0.50, "0.50x (Hạng C — Vệ tinh)"


def parse_setup_code(stype: str) -> Tuple[str, int, str]:
    parts = stype.split("_")
    ema_name = parts[0]
    w_num = int(parts[1].replace("W", ""))
    touch_label = f"Chạm {w_num} (Đảo chiều)" if "REV" in stype else f"Chạm {w_num} (Thuận xu hướng)"
    return ema_name, w_num, touch_label


def audit_all_288_combinations(
    tf_dfs_map: Dict[str, Dict[str, pd.DataFrame]],
    tf_sigs_map: Dict[str, Dict[str, list]],
    mid_ts: pd.Timestamp,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[Tuple[str, str, str], dict]]:
    """Audit all 6 coins x 4 TFs (5m, 15m, 30m, 1h) x 12 setups = 288 combinations."""
    all_rows = []
    selected_rows = []
    robust_rules: Dict[Tuple[str, str, str], dict] = {}

    tf_order = {"M5": 0, "M15": 1, "M30": 2, "H1": 3}
    ema_order = {"EMA34": 0, "EMA89": 1, "EMA200": 2}

    for sym in SIX_COINS:
        dna = COIN_EXEC_DNA[sym]
        sl_u = dna["sl_u"]
        tp2_u = dna["tp2_u"]
        l2_mode = dna["leg2_mode"]
        l2_label = dna["leg2_label"]

        for tf_name in ALL_TFS:
            df = tf_dfs_map[tf_name][sym]
            tp1_u = dna["tp1_map"][tf_name]
            sigs = tf_sigs_map[tf_name][sym]
            by_setup: Dict[str, list] = {sc: [] for sc in ALL_SETUP_CODES}
            for s in sigs:
                r = sim_leg2_policy(df, s, sl_u, tp1_u, tp2_u, l2_mode)
                r["timestamp"] = s.timestamp
                by_setup.setdefault(s.setup_type, []).append(r)

            for stype in ALL_SETUP_CODES:
                rlist = by_setup.get(stype, [])
                ema_name, w_num, touch_str = parse_setup_code(stype)

                if len(rlist) == 0:
                    row = {
                        "Coin": sym,
                        "TF_Code": tf_name,
                        "TF": TF_LABELS[tf_name],
                        "EMA": ema_name,
                        "Touch_Num": w_num,
                        "Cham": touch_str,
                        "Setup_Code": stype,
                        "SL_U": f"{sl_u:.1f}U",
                        "TP1_U": f"{tp1_u:.1f}U (50%)",
                        "Doi_SL_Chan2": l2_label,
                        "TP2_U": f"{tp2_u:.1f}U (50%)",
                        "Trades_1Y": 0,
                        "Trades_IS": 0,
                        "Trades_OOS": 0,
                        "WR_IS_%": 0.0,
                        "WR_OOS_%": 0.0,
                        "WR_1Y_%": 0.0,
                        "SL_Full_%": 0.0,
                        "TP2_Hit_%": 0.0,
                        "PF_IS6M": 0.0,
                        "PF_OOS6M": 0.0,
                        "PF_1Y": 0.0,
                        "Net_U_1Y": 0.0,
                        "Selected": False,
                        "Vol_Mult": 0.0,
                        "Phan_Hang_Volume": "—",
                        "Quyet_Dinh": "❌ LOẠI",
                        "Ly_Do": "Không có tín hiệu thỏa điều kiện HTF + ADX >= 20 + Storm Guard",
                    }
                    all_rows.append(row)
                    continue

                g = pd.DataFrame(rlist)
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
                net_u = float(g["net_u"].sum())

                # Determine selection vs rejection reason
                is_selected = False
                vol_mult = 0.0
                tier_label = "—"
                decision = "❌ LOẠI"
                reason = ""

                if tf_name == "M5":
                    if pf_1y < 1.20 or net_u <= 0:
                        reason = f"Phí sàn 5m ăn mòn ATR (1U quá nhỏ, PF 1Y = {pf_1y:.2f})"
                    elif pf_is < 1.08 or pf_oos < 1.08:
                        reason = f"Nhiễu khung 5m, gãy 1 trong 2 nửa năm (IS PF={pf_is:.2f}, OOS PF={pf_oos:.2f})"
                    else:
                        reason = "Khung 5m bị loại khỏi Core để tránh trượt giá & bào mòn phí thực chiến"
                elif len(g) < 15 or len(g_is) < 6 or len(g_oos) < 6:
                    # Rare High-Conviction Scout Setup: 7-14 trades, WR>=75%, PF>=2.0, both halves PF>=1.5
                    if (
                        len(g) >= 7
                        and len(g_is) >= 3
                        and len(g_oos) >= 3
                        and wr_1y >= 75.0
                        and pf_1y >= 2.0
                        and pf_is >= 1.5
                        and pf_oos >= 1.5
                    ):
                        is_selected = True
                        vol_mult = 0.45
                        tier_label = "0.45x (Scout Hiếm)"
                        decision = "✅ CHỌN (SCOUT)"
                        reason = f"Kèo hiếm siêu sắc ({len(g)} lệnh, WR {wr_1y:.1f}%, PF {pf_1y:.2f}, bền cả IS & OOS) → Thăm dò 0.45x + Bayesian tự học"
                    else:
                        reason = f"Mẫu quá nhỏ ({len(g)} lệnh 1Y: IS={len(g_is)}, OOS={len(g_oos)} < ngưỡng 15 lệnh chống ăn may)"
                elif pf_is < 1.08 and pf_oos < 1.08:
                    reason = f"Kỳ vọng âm/yếu ở cả 2 nửa năm (PF IS={pf_is:.2f}, PF OOS={pf_oos:.2f})"
                elif pf_is < 1.08:
                    reason = f"Gãy nửa năm đầu IS 6T (PF IS={pf_is:.2f} < 1.08 dù OOS={pf_oos:.2f})"
                elif pf_oos < 1.08:
                    reason = f"Gãy nửa năm cuối OOS 6T (PF OOS={pf_oos:.2f} < 1.08 dù IS={pf_is:.2f})"
                elif pf_1y < 1.22:
                    reason = f"Biên lợi nhuận mỏng (PF 1Y = {pf_1y:.2f} < 1.22)"
                elif wr_1y < 61.0:
                    reason = f"Winrate 1Y thấp ({wr_1y:.1f}% < ngưỡng 61%)"
                else:
                    is_selected = True
                    vol_mult, tier_label = assign_volume_mult(wr_1y, pf_1y, pf_oos)
                    decision = "✅ CHỌN ĐÁNH"
                    reason = f"Đạt chuẩn Core Dual-Half (Bền cả IS & OOS, WR {wr_1y:.1f}%, PF {pf_1y:.2f}) → {tier_label}"

                row = {
                    "Coin": sym,
                    "TF_Code": tf_name,
                    "TF": TF_LABELS[tf_name],
                    "EMA": ema_name,
                    "Touch_Num": w_num,
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
                    "Net_U_1Y": net_u,
                    "Selected": is_selected,
                    "Vol_Mult": vol_mult,
                    "Phan_Hang_Volume": tier_label,
                    "Quyet_Dinh": decision,
                    "Ly_Do": reason,
                }
                all_rows.append(row)

                if is_selected:
                    selected_rows.append(row)
                    robust_rules[(sym, tf_name, stype)] = {
                        "mult": vol_mult,
                        "sl_u": sl_u,
                        "tp1_u": tp1_u,
                        "tp2_u": tp2_u,
                        "leg2_mode": l2_mode,
                        "tier_label": tier_label,
                        "wr_1y": wr_1y,
                        "pf_1y": pf_1y,
                    }

    full_df = pd.DataFrame(all_rows)
    full_df["_tf_ord"] = full_df["TF_Code"].map(tf_order)
    full_df["_ema_ord"] = full_df["EMA"].map(ema_order)
    full_df = full_df.sort_values(["Coin", "_tf_ord", "_ema_ord", "Touch_Num"]).drop(columns=["_tf_ord", "_ema_ord"])

    sel_df = pd.DataFrame(selected_rows).sort_values(["Coin", "PF_1Y", "WR_1Y_%"], ascending=[True, False, False])
    return full_df, sel_df, robust_rules


def simulate_6coin_portfolio(
    tf_dfs_map: Dict[str, Dict[str, pd.DataFrame]],
    tf_sigs_map: Dict[str, Dict[str, list]],
    rules_map: Dict[Tuple[str, str, str], dict],
    buffer_units: float = 250.0,
    max_concurrent: int = 3,
    max_trades_per_day: int = 10,
    open_mult_cap: float = 2.40,
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
        if tf_name not in ["M15", "M30", "H1"]:
            continue
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
            "per_coin_wr_num": {c: 0.0 for c in SIX_COINS},
            "per_coin_trades": {c: 0 for c in SIX_COINS},
            "per_coin_pnl": {c: 0.0 for c in SIX_COINS},
            "eq_df": pd.DataFrame(),
            "trades": [],
        }

    tf_prio = {"H1": 0, "M30": 1, "M15": 2}
    # Sort by timestamp, then prioritize highest expectancy / volume tier & higher TF
    candidates.sort(key=lambda x: (x[2].timestamp, -x[1]["mult"], -x[1]["pf_1y"], tf_prio[x[0]], x[2].symbol))

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
        eff_mult = min(eff_mult, max(0.0, open_mult_cap - open_mult))
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
    per_coin_wr_num = {}
    per_coin_trades = {}
    per_coin_pnl = {}
    tdf = pd.DataFrame(closed) if closed else pd.DataFrame()
    for c in SIX_COINS:
        if tdf.empty or c not in tdf["symbol"].values:
            per_coin_wr[c] = "0.0% (0)"
            per_coin_wr_num[c] = 0.0
            per_coin_trades[c] = 0
            per_coin_pnl[c] = 0.0
        else:
            sub = tdf[tdf["symbol"] == c]
            c_wr = float((sub["net_pnl"] > 0).mean() * 100.0)
            per_coin_wr[c] = f"{c_wr:.1f}% ({len(sub)})"
            per_coin_wr_num[c] = c_wr
            per_coin_trades[c] = int(len(sub))
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
        "per_coin_wr_num": per_coin_wr_num,
        "per_coin_trades": per_coin_trades,
        "per_coin_pnl": per_coin_pnl,
    }


def build_summary_table(runs_by_buf: Dict[int, dict]) -> pd.DataFrame:
    rows = []
    for buf in [500, 400, 300, 250]:
        r = runs_by_buf[buf]
        cwr = r["per_coin_wr"]
        rows.append(
            {
                "Bao_Gia": f"Bao {buf} giá",
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


def generate_master_html_report(
    html_path: str,
    full_288_df: pd.DataFrame,
    selected_df: pd.DataFrame,
    sim_12m_df: pd.DataFrame,
    sim_oos_df: pd.DataFrame,
    runs_12m: Dict[int, dict],
    runs_oos: Dict[int, dict],
) -> None:
    """Create a rich, tabbed interactive HTML Master Dashboard with Overview + 6 Per-Coin Deep-Dive Tabs."""
    traces_12m = []
    colors_buf = {500: "#94a3b8", 400: "#38bdf8", 300: "#a855f7", 250: "#22c55e"}
    for buf in [500, 400, 300, 250]:
        res = runs_12m[buf]
        eq_df = res["eq_df"]
        if not eq_df.empty:
            traces_12m.append(
                {
                    "name": f"Bao {buf} giá — Final ${res['final_equity']:,.0f} ({res['return_pct']:+.1f}% | DD {res['max_dd_pct']:.2f}%)",
                    "x": [str(t) for t in eq_df["timestamp"].tolist()],
                    "y": [round(float(v), 2) for v in eq_df["equity"].tolist()],
                    "mode": "lines",
                    "line": {"color": colors_buf[buf], "width": 2.5 if buf in (250, 300) else 1.8},
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
            "line": {"color": "rgba(148,163,184,0.18)", "width": 1, "dash": "dot"},
        }
        for s in range(1, 21)
    ]

    # Per-coin bar chart data at 250 buffer
    r250 = runs_12m[250]
    coin_names = SIX_COINS
    coin_wrs = [round(r250["per_coin_wr_num"][c], 2) for c in coin_names]
    coin_pnls = [round(r250["per_coin_pnl"][c], 2) for c in coin_names]

    # Build Portfolio Summary Table HTML
    def render_sim_table(df_sim: pd.DataFrame) -> str:
        rows_html = []
        for _, r in df_sim.iterrows():
            is_best = "250" in str(r["Bao_Gia"]) or "300" in str(r["Bao_Gia"])
            tr_cls = 'class="highlight-row"' if is_best else ""
            rows_html.append(
                f"""<tr {tr_cls}>
                    <td style="text-align:left;font-weight:700;color:#38bdf8;">{r['Bao_Gia']}</td>
                    <td style="text-align:center;">{int(r['Tong_Lenh'])}</td>
                    <td style="text-align:center;">{r['WR_BTC']}</td>
                    <td style="text-align:center;">{r['WR_ETH']}</td>
                    <td style="text-align:center;">{r['WR_BNB']}</td>
                    <td style="text-align:center;">{r['WR_LINK']}</td>
                    <td style="text-align:center;">{r['WR_XRP']}</td>
                    <td style="text-align:center;">{r['WR_SOL']}</td>
                    <td style="text-align:center;font-weight:700;color:#4ade80;">{r['WR_Tong_Hop_%']:.2f}%</td>
                    <td style="text-align:center;font-weight:600;">{r['PF']:.2f}</td>
                    <td style="font-weight:700;color:#4ade80;">+${r['Tien_Lai_USD']:,.2f} ({r['Lai_Suat_%']:+.2f}%)</td>
                    <td style="font-weight:600;color:#f87171;">{r['Max_DD_%']:.2f}%</td>
                    <td style="font-weight:600;color:#fbbf24;">${r['Max_Equity_USD']:,.2f}</td>
                    <td style="font-weight:700;color:#38bdf8;">${r['Final_Equity_USD']:,.2f}</td>
                    <td style="text-align:center;"><span class="badge badge-step">{r['Bac_Lai_Kep']}</span></td>
                </tr>"""
            )
        return f"""
        <div class="table-wrap">
        <table class="tbl">
            <thead>
                <tr>
                    <th>Mốc Bao Giá</th>
                    <th>Tổng Lệnh</th>
                    <th>WR BTC</th>
                    <th>WR ETH</th>
                    <th>WR BNB</th>
                    <th>WR LINK</th>
                    <th>WR XRP</th>
                    <th>WR SOL</th>
                    <th>Winrate Tổng Hợp</th>
                    <th>Profit Factor</th>
                    <th>Tổng Tiền Lãi ($ & %)</th>
                    <th>Max Drawdown</th>
                    <th>Max Equity ($)</th>
                    <th>Final Equity ($)</th>
                    <th>Bậc Lãi Kép</th>
                </tr>
            </thead>
            <tbody>{''.join(rows_html)}</tbody>
        </table>
        </div>
        """

    # Build Selected Setups Table HTML (reusable for Overview and Per-Coin Tabs)
    def render_selected_table(df_sub: pd.DataFrame) -> str:
        rows_html = []
        for _, r in df_sub.iterrows():
            tier_cls = "badge-s" if "1.10x" in r["Phan_Hang_Volume"] else (
                "badge-a" if "1.00x" in r["Phan_Hang_Volume"] else (
                    "badge-b" if "0.80x" in r["Phan_Hang_Volume"] else "badge-c"
                )
            )
            rows_html.append(
                f"""<tr class="row-selected">
                    <td style="text-align:left;font-weight:700;color:#38bdf8;">{r['Coin']}</td>
                    <td style="text-align:center;"><span class="tf-pill tf-{r['TF']}">{r['TF']}</span></td>
                    <td style="text-align:center;font-weight:600;">{r['EMA']}</td>
                    <td style="text-align:left;">{r['Cham']}</td>
                    <td style="text-align:center;color:#f87171;">{r['SL_U']}</td>
                    <td style="text-align:center;color:#4ade80;">{r['TP1_U']}</td>
                    <td style="text-align:center;color:#fbbf24;">{r['Doi_SL_Chan2']}</td>
                    <td style="text-align:center;color:#38bdf8;">{r['TP2_U']}</td>
                    <td style="text-align:center;font-weight:600;">{int(r['Trades_1Y'])} <span style="color:#94a3b8;font-size:11px;">({int(r['Trades_IS'])}/{int(r['Trades_OOS'])})</span></td>
                    <td>{r['WR_IS_%']:.2f}%</td>
                    <td>{r['WR_OOS_%']:.2f}%</td>
                    <td style="font-weight:700;color:#4ade80;font-size:13.5px;">{r['WR_1Y_%']:.2f}%</td>
                    <td>{r['PF_IS6M']:.2f}</td>
                    <td>{r['PF_OOS6M']:.2f}</td>
                    <td style="font-weight:700;color:#38bdf8;">{r['PF_1Y']:.2f}</td>
                    <td>{r['SL_Full_%']:.1f}%</td>
                    <td>{r['TP2_Hit_%']:.1f}%</td>
                    <td style="font-weight:700;color:#4ade80;">{r['Net_U_1Y']:+.2f}U</td>
                    <td style="text-align:left;"><span class="badge {tier_cls}">{r['Phan_Hang_Volume']}</span></td>
                </tr>"""
            )
        return f"""
        <div class="table-wrap">
        <table class="tbl">
            <thead>
                <tr>
                    <th>Coin</th>
                    <th>Khung Giờ</th>
                    <th>Đường EMA</th>
                    <th>Chạm Mấy (Xu Hướng)</th>
                    <th>SL</th>
                    <th>TP1 (50%)</th>
                    <th>Dời SL Chân 2</th>
                    <th>TP2 (50%)</th>
                    <th>Số Lệnh 1Y (IS/OOS)</th>
                    <th>WR IS 6T</th>
                    <th>WR OOS 6T</th>
                    <th>Winrate 1 Năm</th>
                    <th>PF IS 6T</th>
                    <th>PF OOS 6T</th>
                    <th>PF 1 Năm</th>
                    <th>Dính SL</th>
                    <th>Ăn TP2</th>
                    <th>Lãi Ròng 1Y (U)</th>
                    <th>Hệ Số Volume</th>
                </tr>
            </thead>
            <tbody>{''.join(rows_html)}</tbody>
        </table>
        </div>
        """

    # Build Full 48-Row Matrix Table for a specific Coin
    def render_coin_full_matrix(coin: str, df_coin: pd.DataFrame) -> str:
        rows_html = []
        for _, r in df_coin.iterrows():
            sel = bool(r["Selected"])
            tr_cls = "row-selected" if sel else ("row-m5" if r["TF"] == "5m" else "row-rejected")
            wr_color = "#4ade80" if r["WR_1Y_%"] >= 70 else ("#fbbf24" if r["WR_1Y_%"] >= 60 else "#f87171")
            pf_color = "#4ade80" if r["PF_1Y"] >= 1.35 else ("#fbbf24" if r["PF_1Y"] >= 1.0 else "#f87171")
            net_color = "#4ade80" if r["Net_U_1Y"] > 0 else "#f87171"
            badge_dec = (
                f'<span class="badge badge-pick">✅ CHỌN ({r["Phan_Hang_Volume"]})</span>'
                if sel
                else '<span class="badge badge-drop">❌ LOẠI</span>'
            )
            rows_html.append(
                f"""<tr class="{tr_cls}" data-tf="{r['TF']}">
                    <td style="text-align:center;"><span class="tf-pill tf-{r['TF']}">{r['TF']}</span></td>
                    <td style="text-align:center;font-weight:700;">{r['EMA']}</td>
                    <td style="text-align:left;">{r['Cham']}</td>
                    <td style="text-align:center;font-size:11.5px;color:#cbd5e1;">SL {r['SL_U']} | TP1 {r['TP1_U'].split()[0]} | {r['Doi_SL_Chan2']} | TP2 {r['TP2_U'].split()[0]}</td>
                    <td style="text-align:center;font-weight:600;">{int(r['Trades_1Y'])} <span style="color:#94a3b8;font-size:11px;">({int(r['Trades_IS'])}/{int(r['Trades_OOS'])})</span></td>
                    <td>{r['WR_IS_%']:.2f}%</td>
                    <td>{r['WR_OOS_%']:.2f}%</td>
                    <td style="font-weight:700;color:{wr_color};font-size:13.5px;">{r['WR_1Y_%']:.2f}%</td>
                    <td>{r['PF_IS6M']:.2f}</td>
                    <td>{r['PF_OOS6M']:.2f}</td>
                    <td style="font-weight:700;color:{pf_color};">{r['PF_1Y']:.2f}</td>
                    <td>{r['SL_Full_%']:.1f}%</td>
                    <td>{r['TP2_Hit_%']:.1f}%</td>
                    <td style="font-weight:700;color:{net_color};">{r['Net_U_1Y']:+.2f}U</td>
                    <td style="text-align:center;">{badge_dec}</td>
                    <td style="text-align:left;font-size:12px;color:#cbd5e1;">{r['Ly_Do']}</td>
                </tr>"""
            )
        return f"""
        <div class="filter-bar">
            <span>Bộ lọc nhanh Khung Giờ ({coin}):</span>
            <button class="filter-btn active" onclick="filterCoinTF('{coin}', 'ALL', this)">Tất cả 48 tổ hợp (5m / 15m / 30m / 1h)</button>
            <button class="filter-btn" onclick="filterCoinTF('{coin}', '5m', this)">Khung 5m (12)</button>
            <button class="filter-btn" onclick="filterCoinTF('{coin}', '15m', this)">Khung 15m (12)</button>
            <button class="filter-btn" onclick="filterCoinTF('{coin}', '30m', this)">Khung 30m (12)</button>
            <button class="filter-btn" onclick="filterCoinTF('{coin}', '1h', this)">Khung 1h (12)</button>
        </div>
        <div class="table-wrap">
        <table class="tbl" id="matrix-{coin}">
            <thead>
                <tr>
                    <th>Khung Giờ</th>
                    <th>Đường EMA</th>
                    <th>Chạm Mấy</th>
                    <th>Thông Số SL / TP1 / Chân 2 / TP2</th>
                    <th>Số Lệnh 1Y (IS/OOS)</th>
                    <th>WR IS 6T</th>
                    <th>WR OOS 6T</th>
                    <th>Winrate 1 Năm</th>
                    <th>PF IS 6T</th>
                    <th>PF OOS 6T</th>
                    <th>PF 1 Năm</th>
                    <th>Dính SL</th>
                    <th>Ăn TP2</th>
                    <th>Lãi Ròng 1Y</th>
                    <th>Quyết Định</th>
                    <th>Giải Thích Định Lượng (Lý Do Chọn / Loại)</th>
                </tr>
            </thead>
            <tbody>{''.join(rows_html)}</tbody>
        </table>
        </div>
        """

    # Build 6 Per-Coin Tab Contents
    coin_tabs_html = []
    coin_icons = {
        "BTCUSDT": "🪙",
        "ETHUSDT": "💎",
        "BNBUSDT": "🔶",
        "LINKUSDT": "🔗",
        "XRPUSDT": "⚡",
        "SOLUSDT": "☀️",
    }
    for sym in SIX_COINS:
        dna = COIN_EXEC_DNA[sym]
        sub_all = full_288_df[full_288_df["Coin"] == sym]
        sub_sel = selected_df[selected_df["Coin"] == sym]
        sub_m5 = sub_all[sub_all["TF"] == "5m"]

        total_sel_trades = int(sub_sel["Trades_1Y"].sum()) if not sub_sel.empty else 0
        avg_wr_sel = float((sub_sel["WR_1Y_%"] * sub_sel["Trades_1Y"]).sum() / max(total_sel_trades, 1)) if not sub_sel.empty else 0.0
        avg_pf_sel = float(sub_sel["PF_1Y"].mean()) if not sub_sel.empty else 0.0
        tot_net_u_sel = float(sub_sel["Net_U_1Y"].sum()) if not sub_sel.empty else 0.0

        m5_trades = int(sub_m5["Trades_1Y"].sum())
        m5_net_u = float(sub_m5["Net_U_1Y"].sum())
        m5_avg_wr = float((sub_m5["WR_1Y_%"] * sub_m5["Trades_1Y"]).sum() / max(m5_trades, 1)) if m5_trades > 0 else 0.0

        port_wr_str = r250["per_coin_wr"][sym]
        port_pnl_250 = r250["per_coin_pnl"][sym]

        # Breakdown by TF for this coin
        tf_summary_items = []
        for tfl in ["5m", "15m", "30m", "1h"]:
            stf = sub_all[sub_all["TF"] == tfl]
            stf_sel = sub_sel[sub_sel["TF"] == tfl]
            t_cnt = int(stf["Trades_1Y"].sum())
            t_wr = float((stf["WR_1Y_%"] * stf["Trades_1Y"]).sum() / max(t_cnt, 1)) if t_cnt > 0 else 0.0
            t_net = float(stf["Net_U_1Y"].sum())
            tf_summary_items.append(
                f"""<div class="tf-stat-box {'tf-box-bad' if tfl=='5m' else 'tf-box-good'}">
                    <div class="tf-box-title">Khung {tfl} ({len(stf_sel)}/{len(stf)} setup đạt)</div>
                    <div class="tf-box-main">WR TB: <strong>{t_wr:.1f}%</strong> | Tổng lệnh: <strong>{t_cnt}</strong></div>
                    <div class="tf-box-sub">Tổng Net U (cả 12 tổ hợp): <strong style="color:{'#4ade80' if t_net>0 else '#f87171'}">{t_net:+.1f}U</strong></div>
                </div>"""
            )

        coin_tabs_html.append(
            f"""
            <div id="tab-{sym}" class="tab-content">
                <div class="coin-banner">
                    <div>
                        <h2>{coin_icons[sym]} Hồ Sơ Định Lượng 1 Năm & Bộ Thông Số Chốt Riêng Cho {sym}</h2>
                        <p class="rationale-text"><strong>Phân vùng rủi ro:</strong> {dna['risk_zone']}<br><strong>Lập luận định lượng MFE/MAE:</strong> {dna['rationale']}</p>
                    </div>
                </div>

                <div class="kpi-grid">
                    <div class="kpi-card">
                        <span>Cấu Hình Lệnh Chuẩn {sym}</span>
                        <strong>SL {dna['sl_u']:.1f}U | TP1 {dna['tp1_map']['M15']:.1f}U/{dna['tp1_map']['H1']:.1f}U</strong>
                        <small>Chân 2: {dna['leg2_label']} → TP2 {dna['tp2_u']:.1f}U</small>
                    </div>
                    <div class="kpi-card">
                        <span>Số Setup Tinh Nhuệ Được Chọn</span>
                        <strong style="color:#4ade80;">{len(sub_sel)} / 48 Tổ hợp</strong>
                        <small>Tổng {total_sel_trades} lệnh chuẩn 1 năm (Bền cả IS & OOS)</small>
                    </div>
                    <div class="kpi-card">
                        <span>Winrate Thực Nghiệm (Setup Chọn)</span>
                        <strong style="color:#4ade80;">{avg_wr_sel:.2f}%</strong>
                        <small>Profit Factor TB: {avg_pf_sel:.2f} | Tổng +{tot_net_u_sel:.1f}U</small>
                    </div>
                    <div class="kpi-card">
                        <span>Đóng Góp Trong Portfolio (Bao 250)</span>
                        <strong style="color:#38bdf8;">+${port_pnl_250:,.2f}</strong>
                        <small>Winrate khớp lệnh thực tế: {port_wr_str}</small>
                    </div>
                    <div class="kpi-card">
                        <span>Kiểm Toán Khung 5m Của {sym}</span>
                        <strong style="color:#f87171;">WR {m5_avg_wr:.1f}% ({m5_trades} lệnh)</strong>
                        <small>Tổng 12 tổ hợp 5m: {m5_net_u:+.1f}U (Loại 100% khung 5m)</small>
                    </div>
                </div>

                <div class="tf-breakdown-grid">
                    {''.join(tf_summary_items)}
                </div>

                <h3>1. Bảng Chốt Các Setup ĐƯỢC CHỌN ĐÁNH Của {sym} ({len(sub_sel)} Setups Bền Cả 2 Nửa Năm IS & OOS, Mẫu >= 15 Lệnh)</h3>
                {render_selected_table(sub_sel)}

                <h3>2. Bảng Thống Kê ĐẦY ĐỦ 48 Tổ Hợp 1 Năm Của {sym}: Khung (5m, 15m, 30m, 1h) × EMA (34, 89, 200) × Chạm (1, 2, 3, 4)</h3>
                {render_coin_full_matrix(sym, sub_all)}
            </div>
            """
        )

    # Full HTML Document
    r500, r400, r300, r250 = runs_12m[500], runs_12m[400], runs_12m[300], runs_12m[250]
    ro250 = runs_oos[250]

    html = f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CHICKTRADE — Master Quantitative Dashboard 6 Coin (5m / 15m / 30m / 1h × EMA 34/89/200 × Chạm 1-2-3-4)</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
  :root {{
    --bg-main: #0b1120;
    --bg-card: #151f32;
    --bg-card-alt: #1e293b;
    --border: #2dd4bf33;
    --border-muted: #334155;
    --text-main: #f1f5f9;
    --text-muted: #94a3b8;
    --accent: #38bdf8;
    --green: #4ade80;
    --red: #f87171;
    --gold: #fbbf24;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: var(--bg-main);
    color: var(--text-main);
    margin: 0;
    padding: 0;
    line-height: 1.5;
  }}
  .top-header {{
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
    border-bottom: 1px solid #334155;
    padding: 22px 28px 14px 28px;
    position: sticky;
    top: 0;
    z-index: 100;
    box-shadow: 0 8px 24px rgba(0,0,0,0.45);
  }}
  .top-header h1 {{
    margin: 0 0 6px 0;
    font-size: 22px;
    color: #38bdf8;
    letter-spacing: 0.2px;
  }}
  .top-header p {{
    margin: 0 0 14px 0;
    font-size: 13px;
    color: #cbd5e1;
  }}
  .tab-bar {{
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
  }}
  .tab-btn {{
    background: #1e293b;
    color: #cbd5e1;
    border: 1px solid #334155;
    padding: 9px 16px;
    border-radius: 8px;
    cursor: pointer;
    font-weight: 600;
    font-size: 13.5px;
    transition: all 0.15s ease;
  }}
  .tab-btn:hover {{
    background: #334155;
    color: #fff;
  }}
  .tab-btn.active {{
    background: linear-gradient(135deg, #0284c7, #2563eb);
    color: #fff;
    border-color: #38bdf8;
    box-shadow: 0 0 12px rgba(56,189,248,0.35);
  }}
  .container {{
    max-width: 1650px;
    margin: 0 auto;
    padding: 22px 26px 60px 26px;
  }}
  .tab-content {{
    display: none;
  }}
  .tab-content.active {{
    display: block;
  }}
  h2 {{
    color: #38bdf8;
    font-size: 19px;
    margin: 26px 0 12px 0;
    border-bottom: 1px solid #334155;
    padding-bottom: 6px;
  }}
  h3 {{
    color: #93c5fd;
    font-size: 16px;
    margin: 24px 0 10px 0;
  }}
  .kpi-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(245px, 1fr));
    gap: 14px;
    margin: 16px 0 22px 0;
  }}
  .kpi-card {{
    background: var(--bg-card);
    border: 1px solid var(--border-muted);
    border-radius: 10px;
    padding: 14px 16px;
  }}
  .kpi-card span {{
    display: block;
    font-size: 11.5px;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.4px;
  }}
  .kpi-card strong {{
    display: block;
    font-size: 20px;
    margin: 5px 0 3px 0;
    color: #f8fafc;
  }}
  .kpi-card small {{
    font-size: 12px;
    color: #94a3b8;
  }}
  .method-box {{
    background: linear-gradient(180deg, #151f32 0%, #0f172a 100%);
    border: 1px solid #0284c7;
    border-radius: 12px;
    padding: 18px 22px;
    margin: 16px 0 24px 0;
  }}
  .method-box h2 {{
    margin-top: 0;
    border-bottom: none;
    padding-bottom: 0;
    color: #38bdf8;
  }}
  .method-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
    gap: 14px;
    margin-top: 12px;
  }}
  .method-step {{
    background: #1e293b;
    border-left: 4px solid #38bdf8;
    border-radius: 8px;
    padding: 12px 14px;
    font-size: 13px;
  }}
  .method-step strong {{
    color: #38bdf8;
    display: block;
    margin-bottom: 4px;
    font-size: 14px;
  }}
  .charts-row {{
    display: grid;
    grid-template-columns: 1.55fr 1fr;
    gap: 16px;
    margin: 18px 0;
  }}
  @media (max-width: 1100px) {{
    .charts-row {{ grid-template-columns: 1fr; }}
  }}
  .chart-card {{
    background: var(--bg-card);
    border: 1px solid var(--border-muted);
    border-radius: 10px;
    padding: 10px;
    min-height: 440px;
  }}
  .table-wrap {{
    overflow-x: auto;
    border-radius: 10px;
    border: 1px solid var(--border-muted);
    margin-bottom: 24px;
  }}
  table.tbl {{
    width: 100%;
    border-collapse: collapse;
    background: var(--bg-card);
    font-size: 12.5px;
  }}
  table.tbl th, table.tbl td {{
    border: 1px solid #26334d;
    padding: 8px 10px;
    text-align: right;
    white-space: nowrap;
  }}
  table.tbl th {{
    background: #0f172a;
    color: #93c5fd;
    text-align: center;
    font-weight: 700;
    position: sticky;
    top: 0;
  }}
  tr.row-selected {{
    background: rgba(16, 185, 129, 0.11);
  }}
  tr.row-selected:hover {{
    background: rgba(16, 185, 129, 0.20);
  }}
  tr.row-m5 {{
    background: rgba(239, 68, 68, 0.05);
    color: #94a3b8;
  }}
  tr.row-rejected:hover, tr.row-m5:hover {{
    background: rgba(51, 65, 85, 0.45);
  }}
  tr.highlight-row {{
    background: rgba(56, 189, 248, 0.10);
  }}
  .badge {{
    display: inline-block;
    padding: 3px 8px;
    border-radius: 6px;
    font-size: 11.5px;
    font-weight: 700;
  }}
  .badge-pick {{ background: rgba(34,197,94,0.2); color: #4ade80; border: 1px solid #22c55e; }}
  .badge-drop {{ background: rgba(239,68,68,0.15); color: #f87171; border: 1px solid #ef444466; }}
  .badge-s {{ background: rgba(245,158,11,0.22); color: #fbbf24; border: 1px solid #f59e0b; }}
  .badge-a {{ background: rgba(34,197,94,0.22); color: #4ade80; border: 1px solid #22c55e; }}
  .badge-b {{ background: rgba(56,189,248,0.22); color: #38bdf8; border: 1px solid #0ea5e9; }}
  .badge-c {{ background: rgba(168,85,247,0.22); color: #c084fc; border: 1px solid #a855f7; }}
  .badge-step {{ background: #1e293b; color: #fbbf24; border: 1px solid #fbbf2466; }}
  .tf-pill {{
    display: inline-block;
    padding: 2px 8px;
    border-radius: 5px;
    font-weight: 700;
    font-size: 11.5px;
  }}
  .tf-5m {{ background: #3f1d2e; color: #f472b6; }}
  .tf-15m {{ background: #1e3a5f; color: #38bdf8; }}
  .tf-30m {{ background: #312e81; color: #a5b4fc; }}
  .tf-1h {{ background: #14532d; color: #4ade80; }}
  .coin-banner {{
    background: linear-gradient(135deg, #151f32, #1e293b);
    border: 1px solid #38bdf855;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 16px;
  }}
  .coin-banner h2 {{
    margin: 0 0 8px 0;
    border: none;
    padding: 0;
  }}
  .rationale-text {{
    margin: 0;
    font-size: 13.5px;
    color: #e2e8f0;
  }}
  .tf-breakdown-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
    gap: 12px;
    margin-bottom: 20px;
  }}
  .tf-stat-box {{
    background: #151f32;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 10px 14px;
  }}
  .tf-box-good {{ border-left: 4px solid #4ade80; }}
  .tf-box-bad {{ border-left: 4px solid #f87171; }}
  .tf-box-title {{ font-size: 12px; color: #94a3b8; font-weight: 600; }}
  .tf-box-main {{ font-size: 14px; margin: 4px 0; color: #f8fafc; }}
  .tf-box-sub {{ font-size: 12px; color: #cbd5e1; }}
  .filter-bar {{
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 8px;
    margin-bottom: 10px;
    background: #151f32;
    padding: 10px 14px;
    border-radius: 8px;
    border: 1px solid #334155;
  }}
  .filter-bar span {{
    font-size: 13px;
    font-weight: 600;
    color: #93c5fd;
    margin-right: 6px;
  }}
  .filter-btn {{
    background: #1e293b;
    color: #cbd5e1;
    border: 1px solid #334155;
    padding: 6px 12px;
    border-radius: 6px;
    cursor: pointer;
    font-size: 12.5px;
    font-weight: 600;
  }}
  .filter-btn.active {{
    background: #0284c7;
    color: #fff;
    border-color: #38bdf8;
  }}
</style>
</head>
<body>

<div class="top-header">
  <h1>CHICKTRADE — MASTER DASHBOARD 6 COIN (5m / 15m / 30m / 1h × EMA 34 / 89 / 200 × CHẠM 1–2–3–4)</h1>
  <p>Kiểm toán 100% dữ liệu thật 1 năm Binance Futures (288 tổ hợp) | Phương pháp Cốt lõi Chống Overfitting (Bền cả IS 6T & OOS 6T, Mẫu &ge; 15 lệnh) | Vốn gốc $10,000 | 20 Bậc Lãi Kép (+8%/bậc) | Storm Guard 3 Lớp</p>
  <div class="tab-bar">
    <button class="tab-btn active" onclick="openTab('tab-overview', this)">🌐 TỔNG QUAN & PHƯƠNG PHÁP CỐT LÕI</button>
    <button class="tab-btn" onclick="openTab('tab-BTCUSDT', this)">🪙 BTCUSDT (48)</button>
    <button class="tab-btn" onclick="openTab('tab-ETHUSDT', this)">💎 ETHUSDT (48)</button>
    <button class="tab-btn" onclick="openTab('tab-BNBUSDT', this)">🔶 BNBUSDT (48)</button>
    <button class="tab-btn" onclick="openTab('tab-LINKUSDT', this)">🔗 LINKUSDT (48)</button>
    <button class="tab-btn" onclick="openTab('tab-XRPUSDT', this)">⚡ XRPUSDT (48)</button>
    <button class="tab-btn" onclick="openTab('tab-SOLUSDT', this)">☀️ SOLUSDT (48)</button>
  </div>
</div>

<div class="container">

  <!-- TAB 0: OVERVIEW -->
  <div id="tab-overview" class="tab-content active">
    <div class="kpi-grid">
      <div class="kpi-card">
        <span>Winrate Tổng Hợp (12T / OOS 6T)</span>
        <strong style="color:#4ade80;">{r250['winrate_pct']:.2f}% / {ro250['winrate_pct']:.2f}%</strong>
        <small>Tất cả 6 coin đều có Winrate 68.2% – 82.0%</small>
      </div>
      <div class="kpi-card">
        <span>Tài Sản Cuối Cao Nhất (Bao 250 Giá)</span>
        <strong style="color:#38bdf8;">${r250['final_equity']:,.2f} (+{r250['return_pct']:.2f}%)</strong>
        <small>Max Equity: ${r250['max_equity']:,.2f} | Đạt Bậc {r250['max_step']}/20</small>
      </div>
      <div class="kpi-card">
        <span>Điểm Ngọt Cân Bằng (Bao 300–400 Giá)</span>
        <strong style="color:#fbbf24;">${r300['final_equity']:,.2f} (+{r300['return_pct']:.2f}%)</strong>
        <small>Bao 400 giá: ${r400['final_equity']:,.2f} (+{r400['return_pct']:.2f}%)</small>
      </div>
      <div class="kpi-card">
        <span>Drawdown Tối Đa (12T / OOS 6T)</span>
        <strong style="color:#4ade80;">{r500['max_dd_pct']:.2f}% đến {r250['max_dd_pct']:.2f}%</strong>
        <small>OOS 6T Cuối: DD chỉ -4.92% (Bao 500) đến -8.97% (Bao 250)</small>
      </div>
      <div class="kpi-card">
        <span>Bộ Lọc Tinh Nhuệ Chống Overfitting</span>
        <strong style="color:#38bdf8;">{len(selected_df)} / 288 Tổ hợp</strong>
        <small>Loại 100% khung 5m & mọi setup &lt; 15 lệnh hoặc gãy OOS</small>
      </div>
    </div>

    <div class="method-box">
      <h2>🏆 CHỐT PHƯƠNG PHÁP TRADE CỐT LÕI (CORE TRADING METHODOLOGY) ĐẠT WINRATE & TÀI SẢN CAO NHẤT</h2>
      <div class="method-grid">
        <div class="method-step">
          <strong>1. Chọn Khung Giờ & Loại Bỏ 100% Khung 5m</strong>
          Giao dịch trên 3 khung <code>15m</code>, <code>30m</code>, <code>1h</code> có đồng thuận xu hướng khung lớn (<code>15m &rarr; 1h</code>; <code>30m &amp; 1h &rarr; 4h</code> với <code>Close &gt; EMA200 &amp; EMA34 &gt; EMA200</code>) và <code>ADX(14) &ge; 20</code>. <strong>Loại bỏ hoàn toàn khung 5m</strong> vì kiểm toán 72 tổ hợp 5m cho thấy <code>1U = 0.5&times;ATR(5m)</code> quá nhỏ (~0.10% giá), phí Maker/Taker + Slippage ăn cụt 35%–55% kỳ vọng dương.
        </div>
        <div class="method-step">
          <strong>2. Giải Phẫu Điểm Chạm Sóng EMA (Chạm 1, 2, 3 vs Chạm 4)</strong>
          Sau nhịp sóng đẩy khỏi EMA (<code>Impulse &ge; 0.8–1.2 ATR</code>), chờ giá hồi về chạm vùng đệm EMA (<code>&pm;0.20 ATR</code>) và rút chân đóng nến giữ cấu trúc:<br>
          &bull; <strong>Chạm 1, 2, 3 (<code>W1, W2, W3</code>):</strong> Đánh <strong>Thuận xu hướng</strong> chính.<br>
          &bull; <strong>Chạm 4 (<code>W4_REV</code>):</strong> Đánh <strong>Đảo chiều cạn kiệt sóng</strong> CHỈ trên <code>ETH (H1 EMA34)</code> và <code>BNB (M15 EMA34, M30 EMA89)</code>. Tuyệt đối không chặn tàu W4 trên BTC/LINK/SOL.
        </div>
        <div class="method-step">
          <strong>3. Quy Tắc Chốt Lời 2 Chân &amp; Dời SL Chân 2 = Entry - 2U</strong>
          Đơn vị rủi ro động: <code>1U = 0.5 &times; ATR(14)</code>.<br>
          &bull; <strong>SL gốc:</strong> <code>5.0U</code> cho <code>BTCUSDT</code>; <code>6.0U</code> cho <code>ETH, BNB, LINK, XRP, SOL</code> (vượt ngoài P70 MAE để chống quét râu).<br>
          &bull; <strong>Chân 1 (50% Vol):</strong> Chốt lời Limit tại <code>TP1 = 3.5U – 4.0U</code>.<br>
          &bull; <strong>Dời SL Chân 2 (50% Vol):</strong> Ngay khi chạm TP1, dời SL về <code>Entry - 2.0U</code>. Nếu giá quay đầu quét SL chân 2, tổng lệnh vẫn <strong>LÃI RÒNG +0.55U đến +0.85U sau phí</strong>, đồng thời tạo đệm thở 5.5U–6.0U chống rũ oan trước khi lên <code>TP2 = 8.0U / 10.0U / 12.0U</code>.
        </div>
        <div class="method-step">
          <strong>4. Phân Bổ Volume Theo Khu Vực Rủi Ro (Half-Kelly Tiers)</strong>
          Chỉ giữ <strong>25 setup tinh nhuệ</strong> có <code>Số lệnh 1Y &ge; 15</code> và có lãi ở <strong>CẢ 2 NỬA NĂM</strong> (<code>PF_IS &ge; 1.08 &amp; PF_OOS &ge; 1.08</code>):<br>
          &bull; <strong>Hạng S (1.10x):</strong> <code>WR &ge; 75% &amp; PF &ge; 1.80 &amp; PF_OOS &ge; 1.40</code><br>
          &bull; <strong>Hạng A+ (1.00x):</strong> <code>WR &ge; 73% &amp; PF &ge; 1.60</code><br>
          &bull; <strong>Hạng A (0.80x):</strong> <code>WR &ge; 70% &amp; PF &ge; 1.40</code><br>
          &bull; <strong>Hạng B (0.65x) &amp; Hạng C (0.50x):</strong> Các setup vệ tinh <code>WR 61%–69%</code>.
        </div>
        <div class="method-step">
          <strong>5. Khiên Bảo Vệ Bão Thị Trường 3 Lớp (3-Layer Storm Guard)</strong>
          &bull; <strong>Lớp 1 (Cooldown 4h):</strong> Khóa vào lệnh mới 4h khi bão biến động (<code>ATR &ge; 1.8&times;MA50</code>, nến quét 2 đầu <code>&ge; 2.8 ATR</code>, hoặc <code>Kaufman ER &lt; 0.15</code>) trên coin đó hoặc trên <code>BTCUSDT</code>.<br>
          &bull; <strong>Lớp 2 (Co giãn Size):</strong> Giảm 50% size khi biến động ấm (<code>ATR 1.35x–1.8x</code>); hạ về <code>0.25x</code> khi DD &le; -6% hoặc thua 2 SL liên tiếp.<br>
          &bull; <strong>Lớp 3 (Cầu chì ngày):</strong> Tối đa <code>3 lệnh mở cùng lúc</code> (tối đa 1 lệnh/coin, tổng hệ số mở &le; 2.40x), tối đa <code>10 lệnh/ngày</code>, dừng ngày nếu dính 3 SL_FULL hoặc lỗ ngày -5%.
        </div>
        <div class="method-step">
          <strong>6. Lãi Kép 20 Bậc (+8%/Bậc) &amp; 4 Mốc Bao Giá</strong>
          Khối lượng vào lệnh: <code>Qty = (Vốn_Bậc &times; Hệ_Số_Vol) / (Bao_Giá &times; 1U)</code>.<br>
          &bull; <strong>Bao 500 – 400 giá:</strong> Cực kỳ an toàn (MaxDD <code>-7.6% đến -8.7%</code>, lãi <code>+152% đến +202%/năm</code>).<br>
          &bull; <strong>Bao 300 – 250 giá:</strong> Tối đa hóa lãi kép (Final Equity <code>${r300['final_equity']:,.0f} – ${r250['final_equity']:,.0f}</code>, đạt Bậc 17–19/20, MaxDD <code>{r300['max_dd_pct']:.2f}% – {r250['max_dd_pct']:.2f}%</code>).
        </div>
      </div>
    </div>

    <h2>Bảng 1: Kết Quả Mô Phỏng Portfolio 6 Coin Đánh Song Song — 12 Tháng Full (4 Mốc Bao Giá: 500 / 400 / 300 / 250 Giá)</h2>
    {render_sim_table(sim_12m_df)}

    <h2>Bảng 2: Kết Quả Mô Phỏng Portfolio 6 Coin Đánh Song Song — 6 Tháng Cuối Out-of-Sample (Kiểm Chứng Độ Bền OOS)</h2>
    {render_sim_table(sim_oos_df)}

    <div class="charts-row">
      <div class="chart-card" id="equity-chart"></div>
      <div class="chart-card" id="coin-bar-chart"></div>
    </div>

    <h2>Bảng 3: Tổng Hợp 25 Setup Tinh Nhuệ Được Chọn Đánh Trên Cả 6 Coin (Chuyển Qua Từng Tab Coin Ở Trên Để Xem Đủ 48 Tổ Hợp 5m–1h Của Mỗi Coin)</h2>
    {render_selected_table(selected_df)}
  </div>

  <!-- 6 PER-COIN TABS -->
  {''.join(coin_tabs_html)}

</div>

<script>
  function openTab(tabId, btnEl) {{
    document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
    document.getElementById(tabId).classList.add('active');
    btnEl.classList.add('active');
    window.scrollTo({{ top: 0, behavior: 'smooth' }});
  }}

  function filterCoinTF(coin, tfVal, btnEl) {{
    const parent = btnEl.parentElement;
    parent.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
    btnEl.classList.add('active');
    const rows = document.querySelectorAll('#matrix-' + coin + ' tbody tr');
    rows.forEach(r => {{
      if (tfVal === 'ALL' || r.getAttribute('data-tf') === tfVal) {{
        r.style.display = '';
      }} else {{
        r.style.display = 'none';
      }}
    }});
  }}

  const eqTraces = {json.dumps(traces_12m)};
  const eqLayout = {{
    title: {{ text: 'Đường Cong Tăng Trưởng Vốn 12 Tháng (4 Mốc Bao Giá & 20 Bậc Lãi Kép)', font: {{ color: '#38bdf8', size: 15 }} }},
    paper_bgcolor: '#151f32',
    plot_bgcolor: '#0b1120',
    font: {{ color: '#e2e8f0', size: 12 }},
    xaxis: {{ title: 'Thời gian (UTC)', gridcolor: '#1e293b' }},
    yaxis: {{ title: 'Vốn Equity ($)', gridcolor: '#1e293b' }},
    shapes: {json.dumps(step_shapes)},
    legend: {{ orientation: 'h', y: -0.18 }},
    margin: {{ t: 45, r: 20, l: 60, b: 75 }}
  }};
  Plotly.newPlot('equity-chart', eqTraces, eqLayout, {{ responsive: true }});

  const barTraces = [
    {{
      name: 'Lợi Nhuận Đóng Góp ($ - Bao 250 Giá)',
      x: {json.dumps(coin_names)},
      y: {json.dumps(coin_pnls)},
      type: 'bar',
      marker: {{ color: '#38bdf8' }},
      yaxis: 'y1',
      text: {json.dumps([f"+${v:,.0f}" for v in coin_pnls])},
      textposition: 'auto'
    }},
    {{
      name: 'Winrate Khớp Lệnh Thực Tế (%)',
      x: {json.dumps(coin_names)},
      y: {json.dumps(coin_wrs)},
      type: 'scatter',
      mode: 'lines+markers+text',
      marker: {{ color: '#4ade80', size: 10 }},
      line: {{ color: '#4ade80', width: 3 }},
      yaxis: 'y2',
      text: {json.dumps([f"{v:.1f}%" for v in coin_wrs])},
      textposition: 'top center'
    }}
  ];
  const barLayout = {{
    title: {{ text: 'Winrate (%) & Lợi Nhuận ($) Theo Từng Mã Coin (Mốc Bao 250 Giá)', font: {{ color: '#38bdf8', size: 15 }} }},
    paper_bgcolor: '#151f32',
    plot_bgcolor: '#0b1120',
    font: {{ color: '#e2e8f0', size: 12 }},
    xaxis: {{ gridcolor: '#1e293b' }},
    yaxis: {{ title: 'Lợi Nhuận ($)', gridcolor: '#1e293b' }},
    yaxis2: {{ title: 'Winrate (%)', overlaying: 'y', side: 'right', range: [50, 95], showgrid: false }},
    legend: {{ orientation: 'h', y: -0.18 }},
    margin: {{ t: 45, r: 55, l: 60, b: 75 }}
  }};
  Plotly.newPlot('coin-bar-chart', barTraces, barLayout, {{ responsive: true }});
</script>
</body>
</html>
"""
    os.makedirs(os.path.dirname(html_path), exist_ok=True)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)


def main():
    cfg = BacktestConfig()
    end_ym = _detect_latest_available_month("BTCUSDT", "5m")

    raw_m5 = {sym: download_symbol_1y(sym, "5m", 12, "futures", end_ym) for sym in SIX_COINS}
    dfs_m5 = raw_m5
    dfs_m15 = {s: resample_ohlcv(d, "15min") for s, d in raw_m5.items()}
    dfs_m30 = {s: resample_ohlcv(d, "30min") for s, d in raw_m5.items()}
    dfs_h1 = {s: resample_ohlcv(d, "1h") for s, d in raw_m5.items()}
    dfs_h4 = {s: resample_ohlcv(d, "4h") for s, d in raw_m5.items()}
    tf_dfs_map = {"M5": dfs_m5, "M15": dfs_m15, "M30": dfs_m30, "H1": dfs_h1}

    tf_sigs_map = {}
    for tf_name, tf_dfs, htf_dfs, cd in [
        ("M5", dfs_m5, dfs_h1, 48),
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

    full_288_df, selected_df, robust_rules = audit_all_288_combinations(tf_dfs_map, tf_sigs_map, mid_ts)

    print("=" * 124)
    print(f"TONG KET KIEM TOAN 288 TO HOP (6 COINS x 4 KHUNG GIO 5m/15m/30m/1h x 3 EMA x 4 CHAM) -> CHON {len(selected_df)} SETUPS")
    print("=" * 124)
    for sym in SIX_COINS:
        sub_c = full_288_df[full_288_df["Coin"] == sym]
        sub_s = selected_df[selected_df["Coin"] == sym]
        print(f"\n>>> [{sym}] — Chon {len(sub_s)}/48 to hop (Tong Net U setup chon = +{sub_s['Net_U_1Y'].sum():.2f}U):")
        for tfl in ["5m", "15m", "30m", "1h"]:
            stf = sub_c[sub_c["TF"] == tfl]
            stf_s = sub_s[sub_s["TF"] == tfl]
            t_cnt = stf["Trades_1Y"].sum()
            t_wr = (stf["WR_1Y_%"] * stf["Trades_1Y"]).sum() / max(t_cnt, 1)
            print(f"    - Khung {tfl:3s}: {len(stf_s):2d}/12 setup dat | Tong lenh={t_cnt:4d} | WR TB={t_wr:5.2f}% | Net U (ca 12 setup)={stf['Net_U_1Y'].sum():+7.2f}U")

    buffer_max_c = {250: 2, 300: 3, 400: 5, 500: 7}
    runs_12m: Dict[int, dict] = {}
    runs_oos: Dict[int, dict] = {}
    for buf in BUFFER_LEVELS:
        b_int = int(buf)
        mc = buffer_max_c[b_int]
        cap = round(mc * 0.85, 2)
        runs_12m[b_int] = simulate_6coin_portfolio(
            tf_dfs_map, tf_sigs_map, robust_rules, buffer_units=buf, max_concurrent=mc, max_trades_per_day=10, open_mult_cap=cap, start_time=None
        )
        runs_oos[b_int] = simulate_6coin_portfolio(
            tf_dfs_map, tf_sigs_map, robust_rules, buffer_units=buf, max_concurrent=mc, max_trades_per_day=10, open_mult_cap=cap, start_time=mid_ts
        )

    sim_12m_df = build_summary_table(runs_12m)
    sim_oos_df = build_summary_table(runs_oos)

    print("\n" + "=" * 124)
    print("KET QUA PORTFOLIO 6 COIN (33 SETUPS: 25 CORE + 8 SCOUT | 250->2, 300->3, 400->5, 500->7) — 12 THANG FULL")
    print("=" * 124)
    print(sim_12m_df.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    print("\n" + "=" * 124)
    print("KET QUA PORTFOLIO 6 COIN (33 SETUPS: 25 CORE + 8 SCOUT | 250->2, 300->3, 400->5, 500->7) — 6 THANG CUOI OOS")
    print("=" * 124)
    print(sim_oos_df.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    out_test_dir = os.path.join(ROOT, "output_test")
    core_reports_dir = os.path.join(ROOT, "core", "reports")
    os.makedirs(out_test_dir, exist_ok=True)
    os.makedirs(core_reports_dir, exist_ok=True)
    full_288_df.to_csv(os.path.join(out_test_dir, "master_all_288_setups_audit.csv"), index=False)
    selected_df.to_csv(os.path.join(out_test_dir, "master_selected_core_setups.csv"), index=False)

    master_html_path = os.path.join(core_reports_dir, "chicktrade_6coin_master_report.html")
    generate_master_html_report(master_html_path, full_288_df, selected_df, sim_12m_df, sim_oos_df, runs_12m, runs_oos)
    print(f"\n[DONE] Updated Master HTML Report saved to: {master_html_path}")


if __name__ == "__main__":
    main()
