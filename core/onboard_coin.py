"""CHICKTRADE Per-Coin Auto-Onboarding & Isolated Coin Folder Pipeline.

Reusable CLI tool to onboard ANY Binance Futures USDT symbol into an isolated
directory `e:\\CHICKTRADE\\coins\\<SYMBOL>\\` containing:
  - data\\<SYMBOL>_5m_1y.csv.gz   (self-contained 1Y 5m OHLCV data copy)
  - rules.json                    (selected CORE + SCOUT setups, exit arch, Leg2 SL, Long/Short stats, Bayesian-Kelly prior)
  - audit_48.csv                  (full 48-setup audit across 5m/15m/30m/1h x EMA 34/89/200 x Touch 1..4)
  - report.html                   (standalone dark-themed HTML report with Long vs Short breakdown & TF filters)

Usage:
  python e:\\CHICKTRADE\\core\\onboard_coin.py --symbols BTCUSDT ETHUSDT BNBUSDT LINKUSDT XRPUSDT SOLUSDT
"""
import argparse
import glob
import html
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

CORE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CORE_DIR)
RESEARCH_DIR = os.path.join(ROOT, "research")
for _p in (ROOT, RESEARCH_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import BacktestConfig, RiskConfig
from core.adaptive_sizing import BayesianKellySizer, SetupPrior
from data.binance_loader import CACHE_DIR, _detect_latest_available_month, download_symbol_1y
from risk.compounding import CompoundingManager
from run_full_216_exit_sweetspot import sim_exit_arch
from run_full_3ema_storm_guard import build_cooldown_mask, compute_storm_flags, generate_all_4_touches
from run_research_mtf import resample_ohlcv

ALL_TFS = ["5m", "15m", "30m", "1h"]
ALL_EMAS = [34, 89, 200]
ALL_TOUCHES = [1, 2, 3, 4]

TF_SPEC = {
    "5m": {"rule": "5min", "htf": "15m", "cooldown_bars": 48},
    "15m": {"rule": "15min", "htf": "1h", "cooldown_bars": 16},
    "30m": {"rule": "30min", "htf": "4h", "cooldown_bars": 8},
    "1h": {"rule": "1h", "htf": "4h", "cooldown_bars": 4},
}

EMA_RISK_MULT = {"EMA34": 1.00, "EMA89": 0.95, "EMA200": 0.90}

CHAM_LABELS = {
    1: "Chạm 1 (Khởi sóng)",
    2: "Chạm 2 (Tiếp diễn)",
    3: "Chạm 3 (Chín muồi)",
    4: "Chạm 4 (Đảo chiều)",
}

EXIT_CANDIDATES = [
    ("ONLY_TP1", "ONLY_TP1", 0.0),
    ("ONLY_1R", "ONLY_1R", 0.0),
    ("PARTIAL_NO_MOVE", "PARTIAL_NO_MOVE", 0.0),
    ("PARTIAL_BE+Phi", "PARTIAL_MOVE_SL", 0.3),
    ("PARTIAL_Entry0U", "PARTIAL_MOVE_SL", 0.0),
    ("PARTIAL_Entry-1U", "PARTIAL_MOVE_SL", -1.0),
    ("PARTIAL_Entry-2U", "PARTIAL_MOVE_SL", -2.0),
]

EXIT_META = {
    "ONLY_TP1": ("ONLY_TP1", 0.0, 1.0, "N/A (100% TP1)"),
    "ONLY_1R": ("ONLY_1R", 0.0, 1.0, "N/A (100% 1R)"),
    "PARTIAL_NO_MOVE": ("PARTIAL_NO_MOVE", 0.0, 0.5, "Giữ SL gốc"),
    "PARTIAL_BE+Phi": ("PARTIAL_MOVE_SL", 0.3, 0.5, "BE + Phí (+0.3U)"),
    "PARTIAL_Entry0U": ("PARTIAL_MOVE_SL", 0.0, 0.5, "Entry (0U)"),
    "PARTIAL_Entry-1U": ("PARTIAL_MOVE_SL", -1.0, 0.5, "Entry - 1U"),
    "PARTIAL_Entry-2U": ("PARTIAL_MOVE_SL", -2.0, 0.5, "Entry - 2U"),
}

AUDIT_COLUMNS = [
    "Coin",
    "TF",
    "EMA",
    "Touch",
    "Setup",
    "Lenh_1Y",
    "Long_Lenh",
    "Long_WR_%",
    "Short_Lenh",
    "Short_WR_%",
    "Best_Exit",
    "SL_Full_%",
    "TP1_Hit_%",
    "Dinh_DoiSL_%",
    "TP2_Hit_%",
    "WR_1Y_%",
    "PF_IS",
    "PF_OOS",
    "PF_1Y",
    "Net_U_1Y",
    "Vol_Init",
    "Status",
]


def setup_code_for(ema_p: int, touch: int) -> str:
    return f"EMA{ema_p}_W4_REV" if touch == 4 else f"EMA{ema_p}_W{touch}"


def ensure_5m_data(
    symbol: str,
    months: int = 12,
    force_refresh: bool = False,
    end_ym: Optional[Tuple[int, int]] = None,
) -> Tuple[pd.DataFrame, str]:
    """Load cached 12m 5m Futures data for symbol, or download automatically if absent."""
    symbol = symbol.upper().strip()
    os.makedirs(CACHE_DIR, exist_ok=True)

    if not force_refresh:
        pattern = os.path.join(CACHE_DIR, f"{symbol}_futures_5m_*_{months}m.csv.gz")
        matches = sorted(glob.glob(pattern))
        if matches:
            cache_path = matches[-1]
            df = pd.read_csv(cache_path, parse_dates=["timestamp"])
            if df["timestamp"].dt.tz is None:
                df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
            return df, cache_path

    if end_ym is None:
        end_ym = _detect_latest_available_month("BTCUSDT", "5m")
    df = download_symbol_1y(
        symbol=symbol,
        interval="5m",
        months=months,
        market="futures",
        end_ym=end_ym,
        force_refresh=force_refresh,
    )
    end_y, end_m = end_ym
    cache_path = os.path.join(CACHE_DIR, f"{symbol}_futures_5m_{end_y}{end_m:02d}_{months}m.csv.gz")
    return df, cache_path


def build_tf_frames(raw_5m: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Resample 5m dataframe into 5m, 15m, 30m, 1h, and 4h frames."""
    df_5m = raw_5m.copy()
    if df_5m["timestamp"].dt.tz is None:
        df_5m["timestamp"] = df_5m["timestamp"].dt.tz_localize("UTC")
    return {
        "5m": df_5m,
        "15m": resample_ohlcv(df_5m, "15min"),
        "30m": resample_ohlcv(df_5m, "30min"),
        "1h": resample_ohlcv(df_5m, "1h"),
        "4h": resample_ohlcv(df_5m, "4h"),
    }


def build_btc_storm_context(btc_tfs: Dict[str, pd.DataFrame]) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Precompute BTC macro storm timestamps & cooldown masks for all 4 timeframes."""
    ctx: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for tfl in ALL_TFS:
        df_btc = btc_tfs[tfl]
        cd = TF_SPEC[tfl]["cooldown_bars"]
        st = compute_storm_flags(df_btc)
        mask = build_cooldown_mask(st["raw_storm"], cd)
        ctx[tfl] = (df_btc["timestamp"].values, mask)
    return ctx


def calibrate_coin_physics(
    tf_dfs: Dict[str, pd.DataFrame],
    tf_sigs: Dict[str, list],
) -> dict:
    """Data-driven physical SL_U (5.0 or 6.0), TP1_U (3.5 or 4.0), TP2_U (8.0/10.0/12.0) calibration.

    Uses:
      - MAE P70 across 15m/30m/1h bounces (12-bar window): <= 4.75U -> SL_U=5.0, else 6.0
      - Per-TF 12-bar MFE-before-SL median: >= 3.35U -> TP1_U=4.0, else 3.5
      - 36-bar MFE-before-SL P75 across 15m/30m/1h: >= 10.5U -> TP2_U=12.0, >= 8.5U -> 10.0, else 8.0
    """
    maes_12: List[float] = []
    mfes_36_before_sl: List[float] = []
    mfe12_by_tf: Dict[str, List[float]] = {tfl: [] for tfl in ALL_TFS}

    for tfl in ALL_TFS:
        df = tf_dfs[tfl]
        h_arr = df["high"].values
        l_arr = df["low"].values
        n = len(df)
        for s in tf_sigs[tfl]:
            u = max(0.5 * s.atr, s.entry_price * 1e-5)
            d = s.direction
            end12 = min(s.bar_idx + 13, n)
            end36 = min(s.bar_idx + 37, n)
            if end12 <= s.bar_idx + 1:
                continue

            wh12, wl12 = h_arr[s.bar_idx + 1 : end12], l_arr[s.bar_idx + 1 : end12]
            fav12 = (wh12 - s.entry_price) / u if d == 1 else (s.entry_price - wl12) / u
            adv12 = (s.entry_price - wl12) / u if d == 1 else (wh12 - s.entry_price) / u

            # MFE before hitting preliminary 6U stop in 12 bars
            best12 = 0.0
            for k in range(len(wh12)):
                if adv12[k] >= 6.0:
                    break
                if fav12[k] > best12:
                    best12 = float(fav12[k])
            mfe12_by_tf[tfl].append(best12)

            if tfl in ("15m", "30m", "1h"):
                maes_12.append(max(0.0, float(adv12.max())))
                wh36, wl36 = h_arr[s.bar_idx + 1 : end36], l_arr[s.bar_idx + 1 : end36]
                fav36 = (wh36 - s.entry_price) / u if d == 1 else (s.entry_price - wl36) / u
                adv36 = (s.entry_price - wl36) / u if d == 1 else (wh36 - s.entry_price) / u
                best36 = 0.0
                for k in range(len(wh36)):
                    if adv36[k] >= 6.0:
                        break
                    if fav36[k] > best36:
                        best36 = float(fav36[k])
                mfes_36_before_sl.append(best36)

    mae_p70 = float(np.quantile(maes_12, 0.70)) if maes_12 else 5.0
    mfe36_p75 = float(np.quantile(mfes_36_before_sl, 0.75)) if mfes_36_before_sl else 8.0

    # Physical SL_U decision based on MAE P70 (5.0U for cleaner bounces, 6.0U for deeper wicks)
    sl_u = 5.0 if mae_p70 <= 4.75 else 6.0

    # Physical TP2_U decision based on MFE P75
    if mfe36_p75 >= 10.5:
        tp2_u = 12.0
    elif mfe36_p75 >= 8.5:
        tp2_u = 10.0
    else:
        tp2_u = 8.0

    # Per-TF TP1_U decision (3.5U or 4.0U)
    tp1_by_tf: Dict[str, float] = {}
    mfe12_p50_by_tf: Dict[str, float] = {}
    for tfl in ALL_TFS:
        vals = mfe12_by_tf[tfl]
        med = float(np.quantile(vals, 0.50)) if vals else 3.0
        mfe12_p50_by_tf[tfl] = round(med, 2)
        tp1_by_tf[tfl] = 4.0 if med >= 3.35 else 3.5

    return {
        "sl_u": sl_u,
        "tp1_u_by_tf": tp1_by_tf,
        "tp2_u": tp2_u,
        "mae_p70_u": round(mae_p70, 2),
        "mfe36_p75_u": round(mfe36_p75, 2),
        "mfe12_p50_by_tf": mfe12_p50_by_tf,
    }


def evaluate_48_setups(
    symbol: str,
    tf_dfs: Dict[str, pd.DataFrame],
    tf_sigs: Dict[str, list],
    cal: dict,
) -> Tuple[pd.DataFrame, List[dict], dict]:
    """Scan ALL 48 setups (4 TFs x 3 EMAs x 4 Touches) across ALL 7 Exit Architectures.

    Returns (audit_df, selected_rules_list, scan_summary).
    """
    sizer = BayesianKellySizer()

    # Global midpoint timestamp for strict 6M IS / 6M OOS split
    all_ts = sorted(s.timestamp for tfl in ALL_TFS for s in tf_sigs[tfl])
    if all_ts:
        mid_ts = all_ts[len(all_ts) // 2]
    else:
        mid_ts = pd.Timestamp(tf_dfs["15m"]["timestamp"].iloc[len(tf_dfs["15m"]) // 2])

    audit_rows: List[dict] = []
    selected_rules: List[dict] = []

    total_long_all = 0
    total_long_wins_all = 0
    total_short_all = 0
    total_short_wins_all = 0

    sl_u = cal["sl_u"]
    tp2_u = cal["tp2_u"]

    for tfl in ALL_TFS:
        df = tf_dfs[tfl]
        h_arr = df["high"].values
        l_arr = df["low"].values
        c_arr = df["close"].values
        tp1_u = cal["tp1_u_by_tf"][tfl]

        by_stype: Dict[str, list] = {}
        for s in tf_sigs[tfl]:
            by_stype.setdefault(s.setup_type, []).append(s)

        for ema_p in ALL_EMAS:
            ema_label = f"EMA{ema_p}"
            ema_mult = EMA_RISK_MULT[ema_label]
            for touch in ALL_TOUCHES:
                stype = setup_code_for(ema_p, touch)
                sigs = by_stype.get(stype, [])
                n_all = len(sigs)
                n_is = sum(1 for s in sigs if s.timestamp < mid_ts)
                n_oo = n_all - n_is

                long_sigs_cnt = sum(1 for s in sigs if s.direction == 1)
                short_sigs_cnt = sum(1 for s in sigs if s.direction == -1)

                if n_all == 0:
                    audit_rows.append(
                        {
                            "Coin": symbol,
                            "TF": tfl,
                            "EMA": ema_label,
                            "Touch": f"Chạm {touch}",
                            "Setup": stype,
                            "Lenh_1Y": 0,
                            "Long_Lenh": 0,
                            "Long_WR_%": 0.0,
                            "Short_Lenh": 0,
                            "Short_WR_%": 0.0,
                            "Best_Exit": "SKIP",
                            "SL_Full_%": 0.0,
                            "TP1_Hit_%": 0.0,
                            "Dinh_DoiSL_%": 0.0,
                            "TP2_Hit_%": 0.0,
                            "WR_1Y_%": 0.0,
                            "PF_IS": 0.0,
                            "PF_OOS": 0.0,
                            "PF_1Y": 0.0,
                            "Net_U_1Y": 0.0,
                            "Vol_Init": 0.0,
                            "Status": "SKIP",
                        }
                    )
                    continue

                best_cand = None
                for label, arch, l2_u in EXIT_CANDIDATES:
                    outs, tp1s, l2s, tp2s = [], [], [], []
                    nus, is_nus, oo_nus = [], [], []
                    long_nus, short_nus = [], []
                    wins_u, losses_u = [], []

                    for s in sigs:
                        u = max(0.5 * s.atr, s.entry_price * 1e-5)
                        o, t1, l2h, t2, nu = sim_exit_arch(
                            h_arr, l_arr, c_arr, s.bar_idx, s.entry_price, s.direction, u,
                            sl_u, tp1_u, tp2_u, arch, l2_u,
                        )
                        outs.append(o)
                        tp1s.append(t1)
                        l2s.append(l2h)
                        tp2s.append(t2)
                        nus.append(nu)
                        (oo_nus if s.timestamp >= mid_ts else is_nus).append(nu)
                        if s.direction == 1:
                            long_nus.append(nu)
                        else:
                            short_nus.append(nu)
                        if nu > 0:
                            wins_u.append(nu)
                        else:
                            losses_u.append(-nu)

                    a_all = np.array(nus, dtype=float)
                    a_is = np.array(is_nus, dtype=float) if is_nus else np.array([0.0])
                    a_oo = np.array(oo_nus, dtype=float) if oo_nus else np.array([0.0])

                    pf_all = float(a_all[a_all > 0].sum() / max(-a_all[a_all < 0].sum(), 0.01))
                    pf_is = float(a_is[a_is > 0].sum() / max(-a_is[a_is < 0].sum(), 0.01)) if is_nus else 0.0
                    pf_oo = float(a_oo[a_oo > 0].sum() / max(-a_oo[a_oo < 0].sum(), 0.01)) if oo_nus else 0.0
                    wr_all = float((a_all > 0).mean() * 100.0)
                    sl_pct = float(np.mean([o == "SL" for o in outs]) * 100.0)
                    tp1_pct = float(np.mean(tp1s) * 100.0)
                    l2_pct = float(np.mean(l2s) * 100.0)
                    tp2_pct = float(np.mean(tp2s) * 100.0)
                    net_u = float(a_all.sum())

                    long_wins = sum(1 for x in long_nus if x > 0)
                    short_wins = sum(1 for x in short_nus if x > 0)
                    long_wr = (long_wins / len(long_nus) * 100.0) if long_nus else 0.0
                    short_wr = (short_wins / len(short_nus) * 100.0) if short_nus else 0.0

                    # Dual-half IS 6M / OOS 6M stability qualification
                    is_core = (
                        n_all >= 15
                        and n_is >= 6
                        and n_oo >= 6
                        and pf_is >= 1.10
                        and pf_oo >= 1.10
                        and pf_all >= 1.25
                        and net_u > 0.0
                    )
                    is_scout = (
                        7 <= n_all < 15
                        and n_is >= 3
                        and n_oo >= 3
                        and pf_is >= 1.50
                        and pf_oo >= 1.50
                        and pf_all >= 2.00
                        and wr_all >= 70.0
                        and net_u > 0.0
                    )
                    qualifies = is_core or is_scout

                    cand = {
                        "label": label,
                        "arch": arch,
                        "l2_u": l2_u,
                        "qualifies": qualifies,
                        "is_scout": is_scout and not is_core,
                        "wr": wr_all,
                        "sl_pct": sl_pct,
                        "tp1_pct": tp1_pct,
                        "l2_pct": l2_pct,
                        "tp2_pct": tp2_pct,
                        "pf_is": min(pf_is, 99.99),
                        "pf_oo": min(pf_oo, 99.99),
                        "pf_all": min(pf_all, 99.99),
                        "net_u": net_u,
                        "wins": len(wins_u),
                        "avg_win_u": float(np.mean(wins_u)) if wins_u else max(tp1_u * 0.8, 1.5),
                        "avg_loss_u": float(np.mean(losses_u)) if losses_u else (sl_u + 0.25),
                        "long_n": len(long_nus),
                        "long_wins": long_wins,
                        "long_wr": long_wr,
                        "short_n": len(short_nus),
                        "short_wins": short_wins,
                        "short_wr": short_wr,
                    }
                    if best_cand is None:
                        best_cand = cand
                    else:
                        if cand["qualifies"] and not best_cand["qualifies"]:
                            best_cand = cand
                        elif cand["qualifies"] == best_cand["qualifies"]:
                            if (cand["net_u"], cand["wr"]) > (best_cand["net_u"], best_cand["wr"]):
                                best_cand = cand

                total_long_all += best_cand["long_n"]
                total_long_wins_all += best_cand["long_wins"]
                total_short_all += best_cand["short_n"]
                total_short_wins_all += best_cand["short_wins"]

                status = ("SCOUT" if best_cand["is_scout"] else "CORE") if best_cand["qualifies"] else "SKIP"
                vol_init = 0.0

                if best_cand["qualifies"]:
                    prior = SetupPrior(
                        coin=symbol,
                        tf=tfl,
                        setup_code=stype,
                        prior_trades=n_all,
                        prior_wins=best_cand["wins"],
                        avg_win_u=best_cand["avg_win_u"],
                        avg_loss_u=best_cand["avg_loss_u"],
                        sl_full_pct=best_cand["sl_pct"],
                    )
                    dec_init = sizer.evaluate(prior, [])
                    dec_hot = sizer.evaluate(prior, [best_cand["avg_win_u"]] * 5)
                    dec_cold = sizer.evaluate(prior, [-best_cand["avg_loss_u"]] * 3)

                    raw_init = max(dec_init.vol_mult, 0.40) * ema_mult
                    raw_hot = max(dec_hot.vol_mult, 0.40) * ema_mult
                    raw_cold = dec_cold.vol_mult * ema_mult

                    if best_cand["is_scout"]:
                        vol_init = round(min(raw_init, 0.45), 2)
                        vol_hot = round(min(raw_hot, 0.70), 2)
                        vol_cold = round(min(raw_cold, 0.40), 2)
                    else:
                        vol_init = round(min(raw_init, 1.30), 2)
                        vol_hot = round(min(raw_hot, 1.30), 2)
                        vol_cold = round(min(raw_cold, 1.30), 2)

                    arch_norm, l2_norm, tp1_frac, l2_label = EXIT_META[best_cand["label"]]
                    if best_cand["label"] == "PARTIAL_NO_MOVE":
                        l2_norm = -sl_u
                        l2_label = f"Giữ SL gốc (-{int(sl_u)}U)"

                    selected_rules.append(
                        {
                            "coin": symbol,
                            "tf": tfl,
                            "ema": ema_label,
                            "touch": touch,
                            "cham": CHAM_LABELS[touch],
                            "setup_code": stype,
                            "status": status,
                            "is_scout": best_cand["is_scout"],
                            "exit_arch": arch_norm,
                            "best_exit_raw": best_cand["label"],
                            "tp1_frac": tp1_frac,
                            "sl_u": sl_u,
                            "tp1_u": tp1_u,
                            "tp2_u": tp2_u,
                            "leg2_sl_u": l2_norm,
                            "leg2_sl_label": l2_label,
                            "long_short_stats": {
                                "long_trades": best_cand["long_n"],
                                "long_wins": best_cand["long_wins"],
                                "long_wr_pct": round(best_cand["long_wr"], 2),
                                "short_trades": best_cand["short_n"],
                                "short_wins": best_cand["short_wins"],
                                "short_wr_pct": round(best_cand["short_wr"], 2),
                            },
                            "prior_1y": {
                                "trades": n_all,
                                "wins": best_cand["wins"],
                                "sl_full_pct": round(best_cand["sl_pct"], 2),
                                "tp1_hit_pct": round(best_cand["tp1_pct"], 2),
                                "leg2_sl_pct": round(best_cand["l2_pct"], 2),
                                "tp2_hit_pct": round(best_cand["tp2_pct"], 2),
                                "wr_pct": round(best_cand["wr"], 2),
                                "pf_is": round(best_cand["pf_is"], 2),
                                "pf_oos": round(best_cand["pf_oo"], 2),
                                "pf_1y": round(best_cand["pf_all"], 2),
                                "net_u": round(best_cand["net_u"], 2),
                                "avg_win_u": round(best_cand["avg_win_u"], 2),
                                "avg_loss_u": round(best_cand["avg_loss_u"], 2),
                            },
                            "adaptive_volume": {
                                "kelly_full_pct": round(dec_init.kelly_full_pct, 2),
                                "quarter_kelly_pct": round(dec_init.quarter_kelly_pct, 2),
                                "ema_risk_mult": ema_mult,
                                "init_vol_mult": vol_init,
                                "hot_streak_max_mult": vol_hot,
                                "cold_streak_3sl_mult": vol_cold,
                                "tier": "SCOUT (Capped)" if best_cand["is_scout"] else dec_init.tier_label,
                            },
                        }
                    )

                audit_rows.append(
                    {
                        "Coin": symbol,
                        "TF": tfl,
                        "EMA": ema_label,
                        "Touch": f"Chạm {touch}",
                        "Setup": stype,
                        "Lenh_1Y": n_all,
                        "Long_Lenh": best_cand["long_n"],
                        "Long_WR_%": round(best_cand["long_wr"], 2),
                        "Short_Lenh": best_cand["short_n"],
                        "Short_WR_%": round(best_cand["short_wr"], 2),
                        "Best_Exit": best_cand["label"] if best_cand["qualifies"] else f"SKIP ({best_cand['label']})",
                        "SL_Full_%": round(best_cand["sl_pct"], 2),
                        "TP1_Hit_%": round(best_cand["tp1_pct"], 2),
                        "Dinh_DoiSL_%": round(best_cand["l2_pct"], 2),
                        "TP2_Hit_%": round(best_cand["tp2_pct"], 2),
                        "WR_1Y_%": round(best_cand["wr"], 2),
                        "PF_IS": round(best_cand["pf_is"], 2),
                        "PF_OOS": round(best_cand["pf_oo"], 2),
                        "PF_1Y": round(best_cand["pf_all"], 2),
                        "Net_U_1Y": round(best_cand["net_u"], 2),
                        "Vol_Init": vol_init,
                        "Status": status,
                    }
                )

    audit_df = pd.DataFrame(audit_rows, columns=AUDIT_COLUMNS)
    selected_rules.sort(key=lambda r: (0 if r["status"] == "CORE" else 1, -r["prior_1y"]["pf_1y"], -r["prior_1y"]["net_u"]))

    sel_long = sum(r["long_short_stats"]["long_trades"] for r in selected_rules)
    sel_long_w = sum(r["long_short_stats"]["long_wins"] for r in selected_rules)
    sel_short = sum(r["long_short_stats"]["short_trades"] for r in selected_rules)
    sel_short_w = sum(r["long_short_stats"]["short_wins"] for r in selected_rules)

    summary = {
        "total_48_scanned": len(audit_df),
        "selected_total": len(selected_rules),
        "core_count": sum(1 for r in selected_rules if r["status"] == "CORE"),
        "scout_count": sum(1 for r in selected_rules if r["status"] == "SCOUT"),
        "skip_count": int((audit_df["Status"] == "SKIP").sum()),
        "all48_long_trades": total_long_all,
        "all48_long_wr_pct": round((total_long_wins_all / total_long_all * 100.0) if total_long_all else 0.0, 2),
        "all48_short_trades": total_short_all,
        "all48_short_wr_pct": round((total_short_wins_all / total_short_all * 100.0) if total_short_all else 0.0, 2),
        "selected_long_trades": sel_long,
        "selected_long_wr_pct": round((sel_long_w / sel_long * 100.0) if sel_long else 0.0, 2),
        "selected_short_trades": sel_short,
        "selected_short_wr_pct": round((sel_short_w / sel_short * 100.0) if sel_short else 0.0, 2),
    }
    return audit_df, selected_rules, summary


def sim_single_coin_portfolio(
    symbol: str,
    tf_dfs: Dict[str, pd.DataFrame],
    tf_sigs: Dict[str, list],
    selected_rules: List[dict],
    buffer_units: float = 400.0,
) -> dict:
    """Single-coin 2-way (LONG + SHORT) portfolio simulation strictly enforcing 1 direction at a time."""
    if not selected_rules:
        return {
            "buffer_units": buffer_units,
            "one_direction_at_a_time": True,
            "trades": 0,
            "long_trades": 0,
            "long_wr_pct": 0.0,
            "short_trades": 0,
            "short_wr_pct": 0.0,
            "wr_pct": 0.0,
            "pf": 0.0,
            "final_equity": 10_000.0,
            "return_pct": 0.0,
            "max_dd_pct": 0.0,
            "max_step": 1,
        }

    rules_map = {(r["tf"], r["setup_code"]): r for r in selected_rules}
    rcfg = RiskConfig(
        initial_capital=10_000.0,
        buffer_units=buffer_units,
        max_concurrent_trades=1,
        max_trades_per_day=10,
    )
    comp = CompoundingManager(rcfg)
    equity = 10_000.0
    peak_eq = equity

    cands = []
    for tfl in ALL_TFS:
        for s in tf_sigs[tfl]:
            key = (tfl, s.setup_type)
            if key in rules_map:
                cands.append((tfl, rules_map[key], s))

    tf_prio = {"1h": 0, "30m": 1, "15m": 2, "5m": 3}
    cands.sort(key=lambda x: (x[2].timestamp, tf_prio[x[0]], -x[1]["adaptive_volume"]["init_vol_mult"]))

    active_pos: Optional[Tuple[pd.Timestamp, dict]] = None
    closed: List[dict] = []
    daily_t: Dict[str, int] = {}
    daily_sl: Dict[str, int] = {}
    daily_pnl: Dict[str, float] = {}
    daily_eq: Dict[str, float] = {}
    consec_sl = 0
    eq_series = [equity]

    def settle(now_ts: pd.Timestamp):
        nonlocal active_pos, equity, peak_eq, consec_sl
        if active_pos is not None and active_pos[0] <= now_ts:
            ex_ts, tr = active_pos
            active_pos = None
            equity += tr["pnl"]
            peak_eq = max(peak_eq, equity)
            comp.update_equity(equity, ex_ts)
            dk = tr["dk"]
            daily_pnl[dk] = daily_pnl.get(dk, 0.0) + tr["pnl"]
            if tr["out"] == "SL":
                consec_sl += 1
                daily_sl[dk] = daily_sl.get(dk, 0) + 1
            elif tr["pnl"] > 0:
                consec_sl = 0
            closed.append(tr)
            eq_series.append(equity)

    for tfl, rule, sig in cands:
        settle(sig.timestamp)
        if equity <= 0:
            break
        # Strict 1-position / 1-direction at a time per coin
        if active_pos is not None:
            continue

        dk = sig.timestamp.strftime("%Y-%m-%d")
        if dk not in daily_eq:
            daily_eq[dk] = equity
        if daily_t.get(dk, 0) >= 10 or daily_sl.get(dk, 0) >= 3 or daily_pnl.get(dk, 0.0) <= -0.05 * daily_eq[dk]:
            continue

        eff_m = rule["adaptive_volume"]["init_vol_mult"]
        if sig.pa_vol_ok:
            eff_m *= 0.5
        if (equity - peak_eq) / peak_eq * 100.0 <= -6.0 or consec_sl >= 2:
            eff_m = min(eff_m, 0.25)
        if eff_m < 0.20:
            continue

        df = tf_dfs[tfl]
        u = max(0.5 * sig.atr, sig.entry_price * 1e-5)
        out, _, _, _, nu = sim_exit_arch(
            df["high"].values,
            df["low"].values,
            df["close"].values,
            sig.bar_idx,
            sig.entry_price,
            sig.direction,
            u,
            rule["sl_u"],
            rule["tp1_u"],
            rule["tp2_u"],
            rule["exit_arch"],
            rule["leg2_sl_u"],
        )
        sb = min(comp.tier_capital, equity)
        qty = (sb * eff_m) / (buffer_units * u)
        pnl = nu * u * qty

        ex_i = min(sig.bar_idx + (4 if out == "SL" else 8), len(df) - 1)
        ex_ts = pd.Timestamp(df["timestamp"].iloc[ex_i])
        if ex_ts.tzinfo is None:
            ex_ts = ex_ts.tz_localize("UTC")

        active_pos = (
            ex_ts,
            {
                "tf": tfl,
                "setup": sig.setup_type,
                "direction": "LONG" if sig.direction == 1 else "SHORT",
                "out": out,
                "pnl": pnl,
                "dk": dk,
            },
        )
        daily_t[dk] = daily_t.get(dk, 0) + 1

    if active_pos is not None:
        settle(pd.Timestamp("2099-01-01", tz="UTC"))

    ev = np.array(eq_series, dtype=float)
    pk = np.maximum.accumulate(ev)
    mdd = float(((ev - pk) / pk * 100.0).min())
    pnls = np.array([t["pnl"] for t in closed], dtype=float) if closed else np.array([0.0])
    wr = float((pnls > 0).mean() * 100.0) if closed else 0.0
    pos = float(pnls[pnls > 0].sum())
    neg = float(-pnls[pnls < 0].sum())
    pf = pos / max(neg, 0.01) if closed else 0.0
    ret = (float(ev[-1]) / 10_000.0 - 1.0) * 100.0

    longs = [t for t in closed if t["direction"] == "LONG"]
    shorts = [t for t in closed if t["direction"] == "SHORT"]
    l_wr = (sum(1 for t in longs if t["pnl"] > 0) / len(longs) * 100.0) if longs else 0.0
    s_wr = (sum(1 for t in shorts if t["pnl"] > 0) / len(shorts) * 100.0) if shorts else 0.0

    return {
        "buffer_units": buffer_units,
        "one_direction_at_a_time": True,
        "trades": len(closed),
        "long_trades": len(longs),
        "long_wr_pct": round(l_wr, 2),
        "short_trades": len(shorts),
        "short_wr_pct": round(s_wr, 2),
        "wr_pct": round(wr, 2),
        "pf": round(min(pf, 99.99), 2),
        "final_equity": round(float(ev[-1]), 2),
        "return_pct": round(ret, 2),
        "max_dd_pct": round(mdd, 2),
        "max_step": comp.max_step_reached,
    }


def _badge(text: str, cls: str) -> str:
    return f'<span class="badge badge-{cls}">{html.escape(text)}</span>'


def _status_badge(status: str) -> str:
    cls = {"CORE": "emerald", "SCOUT": "amber", "SKIP": "slate"}.get(status, "slate")
    return _badge(status, cls)


def _exit_badge(exit_label: str) -> str:
    if exit_label.startswith("SKIP"):
        return _badge(exit_label, "slate")
    if exit_label == "ONLY_TP1":
        return _badge(exit_label, "cyan")
    if exit_label == "ONLY_1R":
        return _badge(exit_label, "indigo")
    if exit_label == "PARTIAL_NO_MOVE":
        return _badge(exit_label, "amber")
    return _badge(exit_label, "emerald")


def generate_coin_html_report(
    symbol: str,
    date_range: str,
    cal: dict,
    summary: dict,
    port_sim: dict,
    selected_rules: List[dict],
    audit_df: pd.DataFrame,
    out_path: str,
) -> None:
    """Write standalone dark-themed HTML report for a single coin with Long/Short breakdown & TF filters."""
    # Selected setups rows
    sel_trs = []
    for r in selected_rules:
        ls = r["long_short_stats"]
        pr = r["prior_1y"]
        av = r["adaptive_volume"]
        sel_trs.append(
            f"<tr>"
            f"<td><b>{html.escape(r['tf'])}</b></td>"
            f"<td>{html.escape(r['ema'])}</td>"
            f"<td>{html.escape(r['cham'])}</td>"
            f"<td><code>{html.escape(r['setup_code'])}</code></td>"
            f"<td>{_status_badge(r['status'])}</td>"
            f"<td>{_exit_badge(r['best_exit_raw'])}</td>"
            f"<td>{html.escape(r['leg2_sl_label'])}</td>"
            f"<td class='num'>{pr['trades']}</td>"
            f"<td class='num long-txt'>{ls['long_trades']} ({ls['long_wr_pct']:.1f}%)</td>"
            f"<td class='num short-txt'>{ls['short_trades']} ({ls['short_wr_pct']:.1f}%)</td>"
            f"<td class='num'>{pr['wr_pct']:.1f}%</td>"
            f"<td class='num'>{pr['pf_is']:.2f} / {pr['pf_oos']:.2f} / <b>{pr['pf_1y']:.2f}</b></td>"
            f"<td class='num pos'>+{pr['net_u']:.1f}U</td>"
            f"<td class='num'><b>{av['init_vol_mult']:.2f}x</b></td>"
            f"<td><span class='muted'>{html.escape(av['tier'])}</span></td>"
            f"</tr>"
        )
    if not sel_trs:
        sel_trs.append("<tr><td colspan='15' style='text-align:center;color:#94a3b8;padding:24px;'>No setups passed Dual-Half IS/OOS qualification for this coin.</td></tr>")

    # Full 48-setup rows
    audit_trs = []
    for _, row in audit_df.iterrows():
        st = str(row["Status"])
        tf = str(row["TF"])
        net_u = float(row["Net_U_1Y"])
        net_cls = "pos" if net_u > 0 else ("neg" if net_u < 0 else "muted")
        audit_trs.append(
            f"<tr data-tf='{html.escape(tf)}' data-status='{html.escape(st)}'>"
            f"<td><b>{html.escape(tf)}</b></td>"
            f"<td>{html.escape(str(row['EMA']))}</td>"
            f"<td>{html.escape(str(row['Touch']))}</td>"
            f"<td><code>{html.escape(str(row['Setup']))}</code></td>"
            f"<td class='num'>{int(row['Lenh_1Y'])}</td>"
            f"<td class='num long-txt'>{int(row['Long_Lenh'])} ({float(row['Long_WR_%']):.1f}%)</td>"
            f"<td class='num short-txt'>{int(row['Short_Lenh'])} ({float(row['Short_WR_%']):.1f}%)</td>"
            f"<td>{_exit_badge(str(row['Best_Exit']))}</td>"
            f"<td class='num'>{float(row['SL_Full_%']):.1f}%</td>"
            f"<td class='num'>{float(row['TP1_Hit_%']):.1f}%</td>"
            f"<td class='num'>{float(row['Dinh_DoiSL_%']):.1f}%</td>"
            f"<td class='num'>{float(row['TP2_Hit_%']):.1f}%</td>"
            f"<td class='num'>{float(row['WR_1Y_%']):.1f}%</td>"
            f"<td class='num'>{float(row['PF_IS']):.2f}</td>"
            f"<td class='num'>{float(row['PF_OOS']):.2f}</td>"
            f"<td class='num'><b>{float(row['PF_1Y']):.2f}</b></td>"
            f"<td class='num {net_cls}'>{net_u:+.1f}U</td>"
            f"<td class='num'>{float(row['Vol_Init']):.2f}x</td>"
            f"<td>{_status_badge(st)}</td>"
            f"</tr>"
        )

    tot_ls_all = max(summary["all48_long_trades"] + summary["all48_short_trades"], 1)
    long_share_all = summary["all48_long_trades"] / tot_ls_all * 100.0
    short_share_all = summary["all48_short_trades"] / tot_ls_all * 100.0

    tot_ls_sel = max(summary["selected_long_trades"] + summary["selected_short_trades"], 1)
    long_share_sel = summary["selected_long_trades"] / tot_ls_sel * 100.0
    short_share_sel = summary["selected_short_trades"] / tot_ls_sel * 100.0

    tp1_str = " | ".join(f"{k}: {v:.1f}U" for k, v in cal["tp1_u_by_tf"].items())

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CHICKTRADE — {html.escape(symbol)} 48-Setup Isolated Audit Report</title>
<style>
  :root {{
    --bg: #0b1120;
    --card: #111827;
    --card2: #162033;
    --border: #1e293b;
    --text: #e2e8f0;
    --muted: #94a3b8;
    --emerald: #10b981;
    --rose: #f43f5e;
    --amber: #f59e0b;
    --cyan: #06b6d4;
    --indigo: #6366f1;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 24px;
    background: var(--bg); color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Inter, sans-serif;
    font-size: 13px; line-height: 1.5;
  }}
  .container {{ max-width: 1480px; margin: 0 auto; }}
  .header {{
    display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center;
    gap: 16px; padding: 20px 24px; background: var(--card);
    border: 1px solid var(--border); border-radius: 12px; margin-bottom: 20px;
  }}
  .title {{ font-size: 22px; font-weight: 700; margin: 0; letter-spacing: -0.01em; }}
  .subtitle {{ color: var(--muted); font-size: 12px; margin-top: 4px; }}
  .pill-row {{ display: flex; flex-wrap: wrap; gap: 8px; }}
  .pill {{
    padding: 5px 10px; border-radius: 999px; font-size: 11px; font-weight: 600;
    background: var(--card2); border: 1px solid var(--border); color: var(--text);
  }}
  .grid-kpi {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
    gap: 14px; margin-bottom: 20px;
  }}
  .kpi {{
    background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 16px;
  }}
  .kpi-label {{ color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; }}
  .kpi-val {{ font-size: 22px; font-weight: 700; margin-top: 6px; }}
  .kpi-sub {{ color: var(--muted); font-size: 11px; margin-top: 4px; }}
  .section {{
    background: var(--card); border: 1px solid var(--border); border-radius: 12px;
    padding: 18px 20px; margin-bottom: 20px;
  }}
  .section-head {{
    display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center;
    gap: 12px; margin-bottom: 14px;
  }}
  h2 {{ font-size: 16px; margin: 0; font-weight: 600; }}
  .ls-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 16px; }}
  .ls-box {{ background: var(--card2); border: 1px solid var(--border); border-radius: 10px; padding: 14px; }}
  .bar-track {{
    display: flex; height: 10px; border-radius: 999px; overflow: hidden;
    background: #1e293b; margin: 10px 0;
  }}
  .bar-long {{ background: var(--emerald); }}
  .bar-short {{ background: var(--rose); }}
  .ls-stats {{ display: flex; justify-content: space-between; font-size: 12px; }}
  .filter-bar {{ display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }}
  .filter-btn {{
    background: var(--card2); color: var(--text); border: 1px solid var(--border);
    padding: 5px 12px; border-radius: 6px; cursor: pointer; font-size: 12px; font-weight: 600;
    transition: all 0.15s ease;
  }}
  .filter-btn:hover {{ border-color: var(--cyan); }}
  .filter-btn.active {{ background: var(--cyan); color: #041016; border-color: var(--cyan); }}
  .table-wrap {{ overflow-x: auto; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
  th, td {{ padding: 8px 10px; border-bottom: 1px solid var(--border); white-space: nowrap; text-align: left; }}
  th {{ color: var(--muted); font-weight: 600; font-size: 11px; text-transform: uppercase; background: #0f172a; position: sticky; top: 0; }}
  tr:hover td {{ background: rgba(148, 163, 184, 0.05); }}
  .num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  .pos {{ color: var(--emerald); font-weight: 600; }}
  .neg {{ color: var(--rose); font-weight: 600; }}
  .long-txt {{ color: #34d399; }}
  .short-txt {{ color: #fb7185; }}
  .muted {{ color: var(--muted); }}
  code {{ background: #1e293b; padding: 2px 6px; border-radius: 4px; font-size: 11px; }}
  .badge {{
    display: inline-block; padding: 2px 7px; border-radius: 5px;
    font-size: 10px; font-weight: 700; letter-spacing: 0.03em;
  }}
  .badge-emerald {{ background: rgba(16, 185, 129, 0.16); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.35); }}
  .badge-amber {{ background: rgba(245, 158, 11, 0.16); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.35); }}
  .badge-cyan {{ background: rgba(6, 182, 212, 0.16); color: #22d3ee; border: 1px solid rgba(6, 182, 212, 0.35); }}
  .badge-indigo {{ background: rgba(99, 102, 241, 0.16); color: #a5b4fc; border: 1px solid rgba(99, 102, 241, 0.35); }}
  .badge-slate {{ background: rgba(148, 163, 184, 0.12); color: #94a3b8; border: 1px solid rgba(148, 163, 184, 0.25); }}
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <div>
      <h1 class="title">CHICKTRADE — {html.escape(symbol)} Isolated Onboarding Report</h1>
      <div class="subtitle">Data Range: {html.escape(date_range)} &bull; Full 48-Setup Auto-Scan (5m / 15m / 30m / 1h &times; EMA 34/89/200 &times; Touch 1..4)</div>
    </div>
    <div class="pill-row">
      <span class="pill">2-Way LONG + SHORT Verified</span>
      <span class="pill">1 Direction At A Time</span>
      <span class="pill">SL: {cal['sl_u']:.1f}U (MAE P70={cal['mae_p70_u']}U)</span>
      <span class="pill">TP2: {cal['tp2_u']:.1f}U (MFE P75={cal['mfe36_p75_u']}U)</span>
      <span class="pill">TP1: {html.escape(tp1_str)}</span>
    </div>
  </div>

  <div class="grid-kpi">
    <div class="kpi">
      <div class="kpi-label">Selected Setups (of 48)</div>
      <div class="kpi-val">{summary['selected_total']} <span style="font-size:13px;color:#94a3b8;">({summary['core_count']} CORE / {summary['scout_count']} SCOUT)</span></div>
      <div class="kpi-sub">Dual-Half IS 6M + OOS 6M verified</div>
    </div>
    <div class="kpi">
      <div class="kpi-label">1-Dir Portfolio Return (400G)</div>
      <div class="kpi-val {'pos' if port_sim['return_pct'] >= 0 else 'neg'}">{port_sim['return_pct']:+.2f}%</div>
      <div class="kpi-sub">Final Eq: ${port_sim['final_equity']:,.2f} &bull; Step {port_sim['max_step']}</div>
    </div>
    <div class="kpi">
      <div class="kpi-label">Portfolio Max Drawdown</div>
      <div class="kpi-val">{port_sim['max_dd_pct']:.2f}%</div>
      <div class="kpi-sub">PF: {port_sim['pf']:.2f} &bull; WR: {port_sim['wr_pct']:.1f}% ({port_sim['trades']} trades)</div>
    </div>
    <div class="kpi">
      <div class="kpi-label">Portfolio Long vs Short</div>
      <div class="kpi-val"><span class="long-txt">{port_sim['long_trades']}L</span> / <span class="short-txt">{port_sim['short_trades']}S</span></div>
      <div class="kpi-sub">L WR: {port_sim['long_wr_pct']:.1f}% &bull; S WR: {port_sim['short_wr_pct']:.1f}%</div>
    </div>
  </div>

  <div class="section">
    <div class="section-head">
      <h2>2-Way Directional Breakdown (LONG vs SHORT)</h2>
      <span class="muted">Confirms both Buy &amp; Sell signals are scanned and traded independently</span>
    </div>
    <div class="ls-grid">
      <div class="ls-box">
        <div><b>All 48 Setups Combined</b></div>
        <div class="bar-track">
          <div class="bar-long" style="width:{long_share_all:.1f}%"></div>
          <div class="bar-short" style="width:{short_share_all:.1f}%"></div>
        </div>
        <div class="ls-stats">
          <span class="long-txt">LONG: <b>{summary['all48_long_trades']}</b> trades ({long_share_all:.1f}%) &bull; WR {summary['all48_long_wr_pct']:.1f}%</span>
          <span class="short-txt">SHORT: <b>{summary['all48_short_trades']}</b> trades ({short_share_all:.1f}%) &bull; WR {summary['all48_short_wr_pct']:.1f}%</span>
        </div>
      </div>
      <div class="ls-box">
        <div><b>Selected Setups (CORE + SCOUT)</b></div>
        <div class="bar-track">
          <div class="bar-long" style="width:{long_share_sel:.1f}%"></div>
          <div class="bar-short" style="width:{short_share_sel:.1f}%"></div>
        </div>
        <div class="ls-stats">
          <span class="long-txt">LONG: <b>{summary['selected_long_trades']}</b> trades ({long_share_sel:.1f}%) &bull; WR {summary['selected_long_wr_pct']:.1f}%</span>
          <span class="short-txt">SHORT: <b>{summary['selected_short_trades']}</b> trades ({short_share_sel:.1f}%) &bull; WR {summary['selected_short_wr_pct']:.1f}%</span>
        </div>
      </div>
    </div>
  </div>

  <div class="section">
    <div class="section-head">
      <h2>Selected Setups ({len(selected_rules)} Active Rules — CORE + SCOUT)</h2>
      <span class="muted">Auto-selected optimal exit architecture &amp; Bayesian-Kelly initial volume</span>
    </div>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>TF</th><th>EMA</th><th>Touch</th><th>Setup</th><th>Status</th>
            <th>Best Exit</th><th>Leg2 SL</th><th class="num">Trades</th>
            <th class="num">Long (WR)</th><th class="num">Short (WR)</th>
            <th class="num">WR 1Y</th><th class="num">PF IS / OOS / 1Y</th>
            <th class="num">Net U</th><th class="num">Vol Init</th><th>Tier</th>
          </tr>
        </thead>
        <tbody>
          {''.join(sel_trs)}
        </tbody>
      </table>
    </div>
  </div>

  <div class="section">
    <div class="section-head">
      <h2>Full 48-Setup Audit Table</h2>
      <div class="filter-bar">
        <span class="muted" style="margin-right:4px;">TF:</span>
        <button class="filter-btn active" data-filter-tf="ALL" onclick="setTfFilter('ALL', this)">ALL (48)</button>
        <button class="filter-btn" data-filter-tf="5m" onclick="setTfFilter('5m', this)">5m (12)</button>
        <button class="filter-btn" data-filter-tf="15m" onclick="setTfFilter('15m', this)">15m (12)</button>
        <button class="filter-btn" data-filter-tf="30m" onclick="setTfFilter('30m', this)">30m (12)</button>
        <button class="filter-btn" data-filter-tf="1h" onclick="setTfFilter('1h', this)">1h (12)</button>
        <span class="muted" style="margin:0 4px 0 12px;">Status:</span>
        <button class="filter-btn active" data-filter-st="ALL" onclick="setStFilter('ALL', this)">ALL</button>
        <button class="filter-btn" data-filter-st="CORE" onclick="setStFilter('CORE', this)">CORE</button>
        <button class="filter-btn" data-filter-st="SCOUT" onclick="setStFilter('SCOUT', this)">SCOUT</button>
        <button class="filter-btn" data-filter-st="SKIP" onclick="setStFilter('SKIP', this)">SKIP</button>
      </div>
    </div>
    <div class="table-wrap">
      <table id="audit-table">
        <thead>
          <tr>
            <th>TF</th><th>EMA</th><th>Touch</th><th>Setup</th>
            <th class="num">Lenh 1Y</th><th class="num">Long (WR)</th><th class="num">Short (WR)</th>
            <th>Best Exit</th><th class="num">SL Full%</th><th class="num">TP1%</th>
            <th class="num">Dinh DoiSL%</th><th class="num">TP2%</th><th class="num">WR 1Y%</th>
            <th class="num">PF IS</th><th class="num">PF OOS</th><th class="num">PF 1Y</th>
            <th class="num">Net U 1Y</th><th class="num">Vol Init</th><th>Status</th>
          </tr>
        </thead>
        <tbody>
          {''.join(audit_trs)}
        </tbody>
      </table>
    </div>
  </div>
</div>

<script>
  let curTf = 'ALL';
  let curSt = 'ALL';
  function applyFilters() {{
    const rows = document.querySelectorAll('#audit-table tbody tr');
    rows.forEach(tr => {{
      const okTf = (curTf === 'ALL') || (tr.getAttribute('data-tf') === curTf);
      const okSt = (curSt === 'ALL') || (tr.getAttribute('data-status') === curSt);
      tr.style.display = (okTf && okSt) ? '' : 'none';
    }});
  }}
  function setTfFilter(tf, btn) {{
    curTf = tf;
    document.querySelectorAll('[data-filter-tf]').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    applyFilters();
  }}
  function setStFilter(st, btn) {{
    curSt = st;
    document.querySelectorAll('[data-filter-st]').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    applyFilters();
  }}
</script>
</body>
</html>
"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(doc)


def onboard_symbol(
    symbol: str,
    btc_storm_ctx: Dict[str, Tuple[np.ndarray, np.ndarray]],
    cfg: BacktestConfig,
    coins_root: str,
    months: int = 12,
    force_refresh: bool = False,
    end_ym: Optional[Tuple[int, int]] = None,
) -> dict:
    """Run full isolated onboarding pipeline for a single symbol."""
    symbol = symbol.upper().strip()
    coin_dir = os.path.join(coins_root, symbol)
    data_dir = os.path.join(coin_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    # 1. Load or download 1Y 5m data, then copy into coins/<SYMBOL>/data/<SYMBOL>_5m_1y.csv.gz
    raw_5m, cache_path = ensure_5m_data(symbol, months=months, force_refresh=force_refresh, end_ym=end_ym)
    isolated_data_path = os.path.join(data_dir, f"{symbol}_5m_1y.csv.gz")
    if os.path.exists(cache_path):
        shutil.copy2(cache_path, isolated_data_path)
    else:
        raw_5m.to_csv(isolated_data_path, index=False, compression="gzip")

    # 2. Resample & generate all 4-touch signals across 5m / 15m / 30m / 1h
    tf_dfs = build_tf_frames(raw_5m)
    tf_sigs: Dict[str, list] = {}
    for tfl in ALL_TFS:
        spec = TF_SPEC[tfl]
        btc_ts, btc_mask = btc_storm_ctx[tfl]
        sigs, _ = generate_all_4_touches(
            tf_dfs[tfl],
            tf_dfs[spec["htf"]],
            btc_ts,
            btc_mask,
            symbol,
            cfg.strategy,
            cooldown_bars=spec["cooldown_bars"],
        )
        tf_sigs[tfl] = sigs

    # 3. Auto-calibrate physical SL_U, TP1_U, TP2_U
    cal = calibrate_coin_physics(tf_dfs, tf_sigs)

    # 4. Scan all 48 setups x 7 exit architectures
    audit_df, selected_rules, summary = evaluate_48_setups(symbol, tf_dfs, tf_sigs, cal)

    # 5. Single-coin 2-way (1 direction at a time) portfolio simulation
    port_sim = sim_single_coin_portfolio(symbol, tf_dfs, tf_sigs, selected_rules, buffer_units=400.0)

    # 6. Write isolated outputs
    ts_min = pd.Timestamp(raw_5m["timestamp"].iloc[0]).strftime("%Y-%m-%d")
    ts_max = pd.Timestamp(raw_5m["timestamp"].iloc[-1]).strftime("%Y-%m-%d")
    date_range = f"{ts_min} to {ts_max} (UTC)"

    audit_path = os.path.join(coin_dir, "audit_48.csv")
    audit_df.to_csv(audit_path, index=False)

    rules_payload = {
        "schema_version": "chicktrade-per-coin-v1.0",
        "symbol": symbol,
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "data_file": os.path.join("data", f"{symbol}_5m_1y.csv.gz"),
        "data_bars_5m": int(len(raw_5m)),
        "date_range_utc": date_range,
        "two_way_verified": True,
        "one_direction_at_a_time": True,
        "calibration": cal,
        "summary": summary,
        "portfolio_sim_1y": port_sim,
        "rules": selected_rules,
    }
    rules_path = os.path.join(coin_dir, "rules.json")
    with open(rules_path, "w", encoding="utf-8") as f:
        json.dump(rules_payload, f, indent=2, ensure_ascii=False)

    report_path = os.path.join(coin_dir, "report.html")
    generate_coin_html_report(
        symbol=symbol,
        date_range=date_range,
        cal=cal,
        summary=summary,
        port_sim=port_sim,
        selected_rules=selected_rules,
        audit_df=audit_df,
        out_path=report_path,
    )

    return {
        "symbol": symbol,
        "coin_dir": coin_dir,
        "cal": cal,
        "summary": summary,
        "port_sim": port_sim,
        "files": {
            "data": isolated_data_path,
            "rules": rules_path,
            "audit": audit_path,
            "report": report_path,
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="CHICKTRADE Per-Coin Auto-Onboarding & Isolated Coin Folder Pipeline."
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        required=True,
        help="Binance Futures USDT symbols to onboard (e.g. BTCUSDT ETHUSDT BNBUSDT)",
    )
    parser.add_argument(
        "--months",
        type=int,
        default=12,
        help="Months of 5m history to load/download (default: 12)",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force re-download from Binance Data Vision even if cached",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(ROOT, "coins"),
        help="Root directory for isolated per-coin folders (default: e:\\CHICKTRADE\\coins)",
    )
    args = parser.parse_args()

    cfg = BacktestConfig()
    os.makedirs(args.output_dir, exist_ok=True)

    # Always prepare BTCUSDT storm guard context first
    print("[INIT] Loading BTCUSDT 5m reference data for macro Storm Guard...")
    btc_raw, _ = ensure_5m_data("BTCUSDT", months=args.months, force_refresh=args.force_refresh)
    btc_tfs = build_tf_frames(btc_raw)
    btc_storm_ctx = build_btc_storm_context(btc_tfs)

    results = []
    for sym in args.symbols:
        sym_u = sym.upper().strip()
        print(f"\n[ONBOARD] Processing {sym_u} ...")
        res = onboard_symbol(
            symbol=sym_u,
            btc_storm_ctx=btc_storm_ctx,
            cfg=cfg,
            coins_root=args.output_dir,
            months=args.months,
            force_refresh=args.force_refresh,
        )
        results.append(res)
        s = res["summary"]
        p = res["port_sim"]
        c = res["cal"]
        print(
            f"  [DONE] {sym_u}: SL={c['sl_u']:.1f}U TP2={c['tp2_u']:.1f}U | "
            f"Selected={s['selected_total']}/48 ({s['core_count']} CORE, {s['scout_count']} SCOUT) | "
            f"All48 L={s['all48_long_trades']} ({s['all48_long_wr_pct']:.1f}%) "
            f"S={s['all48_short_trades']} ({s['all48_short_wr_pct']:.1f}%) | "
            f"Port(400G,1-dir) Ret={p['return_pct']:+.2f}% DD={p['max_dd_pct']:.2f}% PF={p['pf']:.2f}"
        )

    print("\n" + "=" * 110)
    print("ONBOARDING SUMMARY (ISOLATED PER-COIN FOLDERS)")
    print("=" * 110)
    header = (
        f"{'Symbol':<10} {'SL/TP2':<10} {'Sel(C/S)':<12} "
        f"{'All48 Long(WR)':<18} {'All48 Short(WR)':<18} "
        f"{'Sel Long(WR)':<16} {'Sel Short(WR)':<16} {'Port Ret / DD'}"
    )
    print(header)
    print("-" * 110)
    for r in results:
        s = r["summary"]
        p = r["port_sim"]
        c = r["cal"]
        print(
            f"{r['symbol']:<10} {c['sl_u']:.0f}U/{c['tp2_u']:.0f}U   "
            f"{s['selected_total']:>2} ({s['core_count']}/{s['scout_count']})    "
            f"{s['all48_long_trades']:>4} ({s['all48_long_wr_pct']:>5.1f}%)    "
            f"{s['all48_short_trades']:>4} ({s['all48_short_wr_pct']:>5.1f}%)    "
            f"{s['selected_long_trades']:>4} ({s['selected_long_wr_pct']:>5.1f}%)  "
            f"{s['selected_short_trades']:>4} ({s['selected_short_wr_pct']:>5.1f}%)  "
            f"{p['return_pct']:>+6.2f}% / {p['max_dd_pct']:>6.2f}%"
        )


if __name__ == "__main__":
    main()
