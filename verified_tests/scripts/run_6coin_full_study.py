"""Comprehensive Per-Coin Quantitative Study (BTC, ETH, BNB, SOL, XRP, LINK):
1. Touches 1..6 of EMA 34 / 89 / 200: Max MFE, Mean MFE, Max MAE, Mean MAE, Median MFE/MAE
2. Mathematical SL / TP1 / TP2 / Leg2-SL Grid Optimization (IS 6M -> OOS 6M) + Recovery Trades per SL
3. Winrate Threshold Sweep (50%..75%) & Risk-Zone / Volume Matrix per Coin (Kelly & Calmar-based)
"""
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import BacktestConfig, StrategyConfig
from data.binance_loader import _detect_latest_available_month, download_symbol_1y
from run_full_3ema_storm_guard import build_cooldown_mask, compute_storm_flags
from run_research_mtf import compute_adx, resample_ohlcv
from strategy.indicators import add_indicators

SIX_COINS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "LINKUSDT", "XRPUSDT", "SOLUSDT"]
OUT_DIR = os.path.join(ROOT, "outputs", "per_coin_study")
os.makedirs(OUT_DIR, exist_ok=True)


def detect_touches_1_to_6(
    df: pd.DataFrame,
    df_htf: pd.DataFrame,
    btc_ts: np.ndarray,
    btc_storm_mask: np.ndarray,
    symbol: str,
    cfg: StrategyConfig,
    cooldown_bars: int,
    horizon_bars: int = 16,
) -> List[dict]:
    """Detect Touches 1..6 across EMA 34, 89, 200 with Storm Guard & HTF+ADX>=20, measuring full MAE/MFE."""
    data = add_indicators(df, cfg)
    adx = compute_adx(df, 14).values
    storm_df = compute_storm_flags(df)
    local_storm = build_cooldown_mask(storm_df["raw_storm"], cooldown_bars)

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
    highs = data["high"].values
    lows = data["low"].values
    ema34 = data["ema34"].values
    ema89 = data["ema89"].values
    ema200 = data["ema200"].values
    atr = data["atr"].values
    ts_arr = data["timestamp"].values
    n = len(data)

    events = []
    for ema_p, et_arr, imp_mult, min_gap in [(34, ema34, 0.8, 3), (89, ema89, 1.0, 5), (200, ema200, 1.2, 8)]:
        trend = 0
        wave_count = 0
        armed = False
        last_touch = -999

        for i in range(205, n - horizon_bars - 1):
            c, h, l, a = closes[i], highs[i], lows[i], atr[i]
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

            if trend == 1:
                if (h - et) >= imp_mult * a and l > et:
                    armed = True
            else:
                if (et - l) >= imp_mult * a and h < et:
                    armed = True

            if not armed or (i - last_touch) < min_gap:
                continue

            if trend == 1:
                touched = (l <= et + 0.20 * a) and (c >= et - 0.65 * a)
            else:
                touched = (h >= et - 0.20 * a) and (c <= et + 0.65 * a)
            if not touched:
                continue

            wave_count += 1
            armed = False
            last_touch = i
            if wave_count > 6:
                continue

            if adx[i] < 20.0:
                continue
            idx = int(np.searchsorted(htf_ts, ts_arr[i], side="right")) - 1
            if idx < 0:
                continue
            ht = int(htf_tr[idx])
            if wave_count <= 3 and ht != trend:
                continue

            b_idx = int(np.searchsorted(btc_ts, ts_arr[i], side="right")) - 1
            if local_storm[i] or (b_idx >= 0 and bool(btc_storm_mask[b_idx])):
                continue

            # Direction: W1-W3 trend-following, W4-W6 counter-trend reversal
            d = trend if wave_count <= 3 else -trend
            u = max(0.5 * a, c * 1e-5)

            wh = highs[i + 1 : i + 1 + horizon_bars]
            wl = lows[i + 1 : i + 1 + horizon_bars]
            fav = (wh - c) / u if d == 1 else (c - wl) / u
            adv = (c - wl) / u if d == 1 else (wh - c) / u

            ts = pd.Timestamp(ts_arr[i])
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")

            events.append(
                {
                    "symbol": symbol,
                    "ema": f"EMA{ema_p}",
                    "touch": wave_count,
                    "direction": d,
                    "bar_idx": i,
                    "timestamp": ts,
                    "entry": float(c),
                    "unit_size": float(u),
                    "unit_pct": float(u / c * 100.0),
                    "mfe_u": float(max(0.0, fav.max())),
                    "mae_u": float(max(0.0, adv.max())),
                }
            )
    return events


def fast_sim_policy(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    bar_idx: int,
    entry: float,
    d: int,
    u: float,
    sl_u: float,
    tp1_u: float,
    tp2_u: float,
    leg2_sl_u: float,   # e.g., -2.0 means Entry - 2U; 0.0 means Entry; +0.3 means Entry + Fee
    max_bars: int = 200,
) -> Tuple[str, float]:
    """Fast bar-by-bar simulation returning (outcome, net_u) including Binance Maker/Taker fees."""
    entry_fee, tp_fee, sl_fee, slip = 0.0002, 0.0002, 0.0005, 0.0001
    sl_orig = entry - d * sl_u * u
    tp1_p = entry + d * tp1_u * u
    tp2_p = entry + d * tp2_u * u
    if leg2_sl_u > 0:
        leg2_sl = entry + d * entry * (entry_fee + sl_fee + slip)
    else:
        leg2_sl = entry + d * leg2_sl_u * u  # negative moves SL below entry for long

    end_i = min(bar_idx + max_bars, len(highs) - 1)
    tp1_hit = False
    fee1_u = (entry * entry_fee + tp1_p * tp_fee) / u
    leg1_net = 0.0

    for j in range(bar_idx + 1, end_i + 1):
        h, l = highs[j], lows[j]
        if not tp1_hit:
            if (l <= sl_orig) if d == 1 else (h >= sl_orig):
                fill = sl_orig * (1.0 - d * slip)
                gross = d * (fill - entry) / u
                fee = (entry * entry_fee + fill * sl_fee) / u
                return "SL", gross - fee
            if (h >= tp1_p) if d == 1 else (l <= tp1_p):
                tp1_hit = True
                leg1_net = 0.5 * (tp1_u - fee1_u)
                if (h >= tp2_p) if d == 1 else (l <= tp2_p):
                    fee2 = (entry * entry_fee + tp2_p * tp_fee) / u
                    return "TP2", leg1_net + 0.5 * (tp2_u - fee2)
        else:
            if (l <= leg2_sl) if d == 1 else (h >= leg2_sl):
                fill = leg2_sl * (1.0 - d * slip)
                gross2 = d * (fill - entry) / u
                fee2 = (entry * entry_fee + fill * sl_fee) / u
                return "BE", leg1_net + 0.5 * (gross2 - fee2)
            if (h >= tp2_p) if d == 1 else (l <= tp2_p):
                fee2 = (entry * entry_fee + tp2_p * tp_fee) / u
                return "TP2", leg1_net + 0.5 * (tp2_u - fee2)

    fill = closes[end_i] * (1.0 - d * slip)
    gross = d * (fill - entry) / u
    fee = (entry * entry_fee + fill * sl_fee) / u
    if tp1_hit:
        return "EOD", leg1_net + 0.5 * (gross - fee)
    return "EOD", gross - fee


def main():
    cfg = BacktestConfig()
    end_ym = _detect_latest_available_month("BTCUSDT", "5m")

    raw_m5 = {sym: download_symbol_1y(sym, "5m", 12, "futures", end_ym) for sym in SIX_COINS}
    dfs_m15 = {s: resample_ohlcv(d, "15min") for s, d in raw_m5.items()}
    dfs_m30 = {s: resample_ohlcv(d, "30min") for s, d in raw_m5.items()}
    dfs_h1 = {s: resample_ohlcv(d, "1h") for s, d in raw_m5.items()}
    dfs_h4 = {s: resample_ohlcv(d, "4h") for s, d in raw_m5.items()}

    # Storm masks for BTC on M15 / M30 / H1
    btc_masks = {}
    for tf_n, tdf, cd in [("M15", dfs_m15["BTCUSDT"], 16), ("M30", dfs_m30["BTCUSDT"], 8), ("H1", dfs_h1["BTCUSDT"], 4)]:
        st = compute_storm_flags(tdf)
        btc_masks[tf_n] = (tdf["timestamp"].values, build_cooldown_mask(st["raw_storm"], cd))

    # =========================================================================
    # PART 1: PER-COIN TOUCHES 1..6 ANATOMY (EMA 34 / 89 / 200) on M30 (clean mid-TF) + M15 + H1
    # =========================================================================
    print("=" * 112)
    print("PHAN 1: THONG KE CHAM 1-2-3-4-5-6 CUA EMA 34/89/200 RIENG TUNG COIN (M30 + TONG HOP M15/M30/H1)")
    print("=" * 112)

    all_events_by_coin_tf: Dict[Tuple[str, str], List[dict]] = {}
    coin_summary_rows = []

    for sym in SIX_COINS:
        coin_ev_all = []
        for tf_n, tdfs, hdfs, cd in [
            ("M15", dfs_m15, dfs_h1, 16),
            ("M30", dfs_m30, dfs_h4, 8),
            ("H1", dfs_h1, dfs_h4, 4),
        ]:
            bts, bmask = btc_masks[tf_n]
            evs = detect_touches_1_to_6(tdfs[sym], hdfs[sym], bts, bmask, sym, cfg.strategy, cd, horizon_bars=16)
            all_events_by_coin_tf[(sym, tf_n)] = evs
            for e in evs:
                e["tf"] = tf_n
            coin_ev_all.extend(evs)

        edf = pd.DataFrame(coin_ev_all)
        # Save detailed Touch 1..6 table per coin
        t_rows = []
        for (ema, touch), g in edf.groupby(["ema", "touch"]):
            t_rows.append(
                {
                    "Coin": sym,
                    "EMA": ema,
                    "Cham": int(touch),
                    "So_Lenh": len(g),
                    "MFE_Max_U": g["mfe_u"].max(),
                    "MFE_TB_U": g["mfe_u"].mean(),
                    "MFE_Med_U": g["mfe_u"].median(),
                    "MFE_P75_U": g["mfe_u"].quantile(0.75),
                    "MAE_Max_U": g["mae_u"].max(),
                    "MAE_TB_U": g["mae_u"].mean(),
                    "MAE_Med_U": g["mae_u"].median(),
                    "MAE_P75_U": g["mae_u"].quantile(0.75),
                    "Ratio_MFE_MAE": g["mfe_u"].mean() / max(g["mae_u"].mean(), 0.01),
                }
            )
        tdf_out = pd.DataFrame(t_rows)
        tdf_out.to_csv(os.path.join(OUT_DIR, f"{sym}_touches_1_to_6.csv"), index=False)

        # Print compact touch 1..6 summary per coin (aggregated across EMA 34/89/200)
        print(f"\n>>> [{sym}] — PHAN BO CHAM 1..6 (Gom M15+M30+H1, Cua so 16 nen, 1U = {edf['unit_pct'].mean():.2f}% gia):")
        touch_agg = (
            edf.groupby("touch")
            .agg(
                So_Lenh=("mfe_u", "count"),
                MFE_Max=("mfe_u", "max"),
                MFE_TB=("mfe_u", "mean"),
                MFE_Med=("mfe_u", "median"),
                MAE_Max=("mae_u", "max"),
                MAE_TB=("mae_u", "mean"),
                MAE_Med=("mae_u", "median"),
            )
            .reset_index()
        )
        touch_agg["MFE/MAE"] = touch_agg["MFE_TB"] / touch_agg["MAE_TB"]
        print(touch_agg.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    # =========================================================================
    # PART 2: MATHEMATICAL SL / TP1 / TP2 / LEG2-SL OPTIMIZATION PER COIN (IS -> OOS)
    # =========================================================================
    print("\n" + "=" * 112)
    print("PHAN 2: TOI UU TOAN HOC SL / TP1 / TP2 / DOI SL CHAN 2 RIENG TUNG COIN (TICH HOP SO LENH GO 1 SL)")
    print("=" * 112)

    sl_grid = [4.0, 5.0, 6.0, 7.0]
    tp1_grid = [3.0, 3.5, 4.0]
    tp2_grid = [8.0, 10.0, 12.0]
    leg2_grid = [0.3, -1.0, -2.0, -3.0]  # 0.3 = Entry+Fee, -1/-2/-3 = Entry - kU

    per_coin_opt_rows = []
    all_trade_records = []  # for Part 3 winrate bucket & Kelly analysis

    for sym in SIX_COINS:
        # Use M15+M30+H1 events for this coin (W1, W2, W3, W4)
        evs_pack = []
        for tf_n, tdfs in [("M15", dfs_m15), ("M30", dfs_m30), ("H1", dfs_h1)]:
            df = tdfs[sym]
            h_arr, l_arr, c_arr = df["high"].values, df["low"].values, df["close"].values
            for e in all_events_by_coin_tf[(sym, tf_n)]:
                if e["touch"] <= 4:
                    evs_pack.append((e, h_arr, l_arr, c_arr))

        # Split IS (first 6m) vs OOS (last 6m)
        ts_list = sorted(x[0]["timestamp"] for x in evs_pack)
        mid_ts = ts_list[len(ts_list) // 2]

        # 2A: Compare SL levels directly (keeping TP1=3.5, TP2=10, Leg2=-2U) to answer "SL bao nhieu it bi quet & de go nhat?"
        sl_analysis = []
        for sl_u in sl_grid:
            nets, outs = [], []
            for e, ha, la, ca in evs_pack:
                o, nu = fast_sim_policy(ha, la, ca, e["bar_idx"], e["entry"], e["direction"], e["unit_size"], sl_u, 3.5, 10.0, -2.0)
                nets.append(nu)
                outs.append(o)
            nets_a = np.array(nets)
            sl_rate = np.mean([o == "SL" for o in outs]) * 100.0
            wr = (nets_a > 0).mean() * 100.0
            avg_win = nets_a[nets_a > 0].mean() if (nets_a > 0).any() else 0.01
            avg_loss = -nets_a[nets_a < 0].mean() if (nets_a < 0).any() else 0.01
            recov_trades = avg_loss / avg_win  # How many winning trades needed to recover 1 SL!
            pf = nets_a[nets_a > 0].sum() / max(-nets_a[nets_a < 0].sum(), 0.01)
            sl_analysis.append((sl_u, sl_rate, wr, avg_win, avg_loss, recov_trades, pf))

        # 2B: Find best (SL, TP1, TP2, Leg2_SL) on IS 6M, then check OOS 6M
        best_is_score = -1e9
        best_params = (5.0, 3.5, 10.0, -2.0)
        for sl_u in [4.0, 5.0, 6.0]:
            for tp1_u in tp1_grid:
                for tp2_u in tp2_grid:
                    for l2_u in leg2_grid:
                        if abs(l2_u) >= sl_u:
                            continue
                        is_nets = []
                        for e, ha, la, ca in evs_pack:
                            if e["timestamp"] >= mid_ts:
                                continue
                            _, nu = fast_sim_policy(ha, la, ca, e["bar_idx"], e["entry"], e["direction"], e["unit_size"], sl_u, tp1_u, tp2_u, l2_u)
                            is_nets.append(nu)
                        arr = np.array(is_nets)
                        pos = arr[arr > 0].sum()
                        neg = -arr[arr < 0].sum()
                        pf_is = pos / neg if neg > 0 else 0.0
                        if pf_is > best_is_score:
                            best_is_score = pf_is
                            best_params = (sl_u, tp1_u, tp2_u, l2_u)

        # Evaluate best_params on OOS 6M and Full 12M
        sl_b, tp1_b, tp2_b, l2_b = best_params
        full_nets, oos_nets, sl_hits = [], [], []
        setup_stats: Dict[Tuple[str, str, int], list] = {}
        for e, ha, la, ca in evs_pack:
            o, nu = fast_sim_policy(ha, la, ca, e["bar_idx"], e["entry"], e["direction"], e["unit_size"], sl_b, tp1_b, tp2_b, l2_b)
            full_nets.append(nu)
            sl_hits.append(o == "SL")
            if e["timestamp"] >= mid_ts:
                oos_nets.append(nu)
            key = (e["tf"], e["ema"], e["touch"])
            setup_stats.setdefault(key, []).append((e["timestamp"] >= mid_ts, nu, o == "SL"))

        f_arr = np.array(full_nets)
        o_arr = np.array(oos_nets)
        avg_w = f_arr[f_arr > 0].mean() if (f_arr > 0).any() else 0.01
        avg_l = -f_arr[f_arr < 0].mean() if (f_arr < 0).any() else 0.01
        recov = avg_l / avg_w
        pf_full = f_arr[f_arr > 0].sum() / max(-f_arr[f_arr < 0].sum(), 0.01)
        pf_oos = o_arr[o_arr > 0].sum() / max(-o_arr[o_arr < 0].sum(), 0.01)

        # Store per-setup WR & Expectancy for Part 3 Winrate Zone Analysis
        for (tf_n, ema_n, tch), items in setup_stats.items():
            if len(items) < 12:
                continue
            is_u = np.array([x[1] for x in items if not x[0]])
            oo_u = np.array([x[1] for x in items if x[0]])
            all_u = np.array([x[1] for x in items])
            if len(is_u) < 5 or len(oo_u) < 5:
                continue
            wr_is = (is_u > 0).mean() * 100.0
            wr_oo = (oo_u > 0).mean() * 100.0
            wr_all = (all_u > 0).mean() * 100.0
            aw = all_u[all_u > 0].mean() if (all_u > 0).any() else 0.01
            al = -all_u[all_u < 0].mean() if (all_u < 0).any() else 0.01
            b = aw / al
            p = wr_all / 100.0
            kelly = max(0.0, (p * b - (1 - p)) / b) if b > 0 else 0.0
            all_trade_records.append(
                {
                    "Coin": sym,
                    "TF": tf_n,
                    "Setup": f"{ema_n}_W{tch}",
                    "Trades": len(all_u),
                    "WR_IS_%": wr_is,
                    "WR_OOS_%": wr_oo,
                    "WR_1Y_%": wr_all,
                    "PF_1Y": all_u[all_u > 0].sum() / max(-all_u[all_u < 0].sum(), 0.01),
                    "PF_OOS": oo_u[oo_u > 0].sum() / max(-oo_u[oo_u < 0].sum(), 0.01),
                    "Avg_Net_U": all_u.mean(),
                    "Net_U_1Y": all_u.sum(),
                    "Kelly_Full_%": kelly * 100.0,
                    "Half_Kelly_%": kelly * 50.0,
                }
            )

        l2_str = "Entry+Phi" if l2_b > 0 else f"Entry{l2_b:+.0f}U"
        per_coin_opt_rows.append(
            {
                "Coin": sym,
                "Opt_SL": sl_b,
                "Opt_TP1": tp1_b,
                "Opt_TP2": tp2_b,
                "Opt_DoiSL": l2_str,
                "Dinh_SL_%": np.mean(sl_hits) * 100.0,
                "Lenh_Go_1SL": recov,
                "WR_1Y_%": (f_arr > 0).mean() * 100.0,
                "PF_1Y": pf_full,
                "PF_OOS6M": pf_oos,
            }
        )

        # Print SL trade-off for this coin
        print(f"\n[{sym}] Phan tich SL (TP1=3.5U, TP2=10U, DoiSL=Entry-2U):")
        sl_df = pd.DataFrame(sl_analysis, columns=["SL_U", "Dinh_SL_%", "WR_%", "AvgWin_U", "AvgLoss_U", "SoLenh_Go_1SL", "PF"])
        print(sl_df.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    opt_df = pd.DataFrame(per_coin_opt_rows)
    print("\n--- TONG HOP THONG SO TOI UU TOAN HOC RIENG TUNG COIN ---")
    print(opt_df.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
    opt_df.to_csv(os.path.join(OUT_DIR, "per_coin_optimal_params.csv"), index=False)

    # =========================================================================
    # PART 3: WINRATE ZONE ANALYSIS ("NEN DANH TU BAO NHIEU %?" + KHU VUC RUI RO + VOLUME)
    # =========================================================================
    print("\n" + "=" * 112)
    print("PHAN 3: PHAN VUNG TY LE THANG (%) -> EXPECTANCY, KELLY & KHU VUC RUI RO TUNG COIN")
    print("=" * 112)
    zdf = pd.DataFrame(all_trade_records)
    zdf.to_csv(os.path.join(OUT_DIR, "all_coin_setup_zones.csv"), index=False)

    bins = [0, 50, 55, 60, 65, 70, 75, 100]
    labels = ["<50%", "50-55%", "55-60%", "60-65%", "65-70%", "70-75%", ">75%"]
    zdf["WR_Zone"] = pd.cut(zdf["WR_1Y_%"], bins=bins, labels=labels, right=False)

    zone_summary = (
        zdf.groupby("WR_Zone", observed=False)
        .agg(
            So_Setup=("Setup", "count"),
            Tong_Lenh=("Trades", "sum"),
            PF_TB=("PF_1Y", "mean"),
            PF_OOS_TB=("PF_OOS", "mean"),
            Avg_Net_U=("Avg_Net_U", "mean"),
            Tong_Net_U=("Net_U_1Y", "sum"),
            Half_Kelly_Risk_Pct=("Half_Kelly_%", "mean"),
        )
        .reset_index()
    )
    print("\n--- HIEU SUAT THEO TUNG KHU VUC WINRATE (%) ---")
    print(zone_summary.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    # Per-coin risk tier summary
    print("\n--- XEP HANG KHU VUC RUI RO TUNG MA COIN (DUA TREN SETUP >= 60% WR) ---")
    c_good = zdf[zdf["WR_1Y_%"] >= 60.0]
    coin_risk = (
        c_good.groupby("Coin")
        .agg(
            So_Setup_Dat=("Setup", "count"),
            Tong_Lenh=("Trades", "sum"),
            WR_TB=("WR_1Y_%", "mean"),
            PF_1Y_TB=("PF_1Y", "mean"),
            PF_OOS_TB=("PF_OOS", "mean"),
            Tong_Net_U=("Net_U_1Y", "sum"),
            Half_Kelly_Pct=("Half_Kelly_%", "mean"),
        )
        .sort_values("PF_OOS_TB", ascending=False)
        .reset_index()
    )
    print(coin_risk.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
    zone_summary.to_csv(os.path.join(OUT_DIR, "winrate_zones_summary.csv"), index=False)
    coin_risk.to_csv(os.path.join(OUT_DIR, "coin_risk_ranking.csv"), index=False)
    print(f"\nAll detailed per-coin CSVs saved in: {OUT_DIR}")


if __name__ == "__main__":
    main()
