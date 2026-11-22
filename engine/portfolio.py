"""Stage 1 (Empirical Discovery) and Stage 2 (Portfolio 20-step Compounding) Backtest Engine."""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import RiskConfig
from risk.compounding import CompoundingManager
from strategy.ema_wave import Signal


@dataclass
class TradeRecord:
    symbol: str
    setup_type: str
    wave: int
    pa_vol_ok: bool
    ema_aligned: bool
    direction: int              # +1 LONG, -1 SHORT
    tier_label: str             # FULL / HALF / DISCOVERY
    step_at_entry: int
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    unit_size: float
    qty: float
    notional: float
    sl_price: float
    tp_price: float
    exit_reason: str            # "TP", "SL", "BE", "EOD"
    gross_units: float          # Price move in Units
    gross_pnl: float
    fee_usd: float
    net_pnl: float
    equity_after: float
    bars_held: int


def simulate_single_trade(
    df: pd.DataFrame,
    sig: Signal,
    sl_units: float,
    tp_units: float,
    be_trigger_units: float,
    enable_be: bool,
    atr_to_unit_mult: float,
    entry_fee_rate: float,
    exit_fee_rate_tp: float,
    exit_fee_rate_sl: float,
    slippage_bps: float,
    max_hold_bars: int = 288,   # 24 hours on M5
) -> Tuple[str, float, pd.Timestamp, int, float, float]:
    """Simulate a single signal forward bar-by-bar with conservative SL-first precedence.

    Returns: (exit_reason, exit_price, exit_time, bars_held, unit_size, gross_units)
    """
    unit_size = max(sig.atr * atr_to_unit_mult, sig.entry_price * 1e-5)
    entry = sig.entry_price
    d = sig.direction

    sl_price = entry - d * sl_units * unit_size
    tp_price = entry + d * tp_units * unit_size
    be_trigger_price = entry + d * be_trigger_units * unit_size
    # Real breakeven covers round-trip maker/taker fees
    be_stop_price = entry + d * entry * (entry_fee_rate + exit_fee_rate_sl)

    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    timestamps = df["timestamp"].values
    n = len(df)

    be_active = False
    slip = slippage_bps / 10_000.0
    end_idx = min(sig.bar_idx + max_hold_bars, n - 1)

    def _to_utc(val) -> pd.Timestamp:
        t = pd.Timestamp(val)
        return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")

    for j in range(sig.bar_idx + 1, end_idx + 1):
        h = highs[j]
        l = lows[j]

        if d == 1:
            # Conservative: check SL first before TP or BE activation on the same bar
            if l <= sl_price:
                reason = "BE" if be_active else "SL"
                fill = sl_price * (1.0 - slip)
                gross_u = (fill - entry) / unit_size
                ts = _to_utc(timestamps[j])
                return reason, float(fill), ts, j - sig.bar_idx, unit_size, float(gross_u)
            if h >= tp_price:
                fill = tp_price
                gross_u = (fill - entry) / unit_size
                ts = _to_utc(timestamps[j])
                return "TP", float(fill), ts, j - sig.bar_idx, unit_size, float(gross_u)
            if enable_be and (not be_active) and h >= be_trigger_price:
                be_active = True
                sl_price = max(sl_price, be_stop_price)
        else:
            if h >= sl_price:
                reason = "BE" if be_active else "SL"
                fill = sl_price * (1.0 + slip)
                gross_u = (entry - fill) / unit_size
                ts = _to_utc(timestamps[j])
                return reason, float(fill), ts, j - sig.bar_idx, unit_size, float(gross_u)
            if l <= tp_price:
                fill = tp_price
                gross_u = (entry - fill) / unit_size
                ts = _to_utc(timestamps[j])
                return "TP", float(fill), ts, j - sig.bar_idx, unit_size, float(gross_u)
            if enable_be and (not be_active) and l <= be_trigger_price:
                be_active = True
                sl_price = min(sl_price, be_stop_price)

    # Timeout / End of Data
    fill = closes[end_idx] * (1.0 - d * slip)
    gross_u = d * (fill - entry) / unit_size
    ts = _to_utc(timestamps[end_idx])
    return "EOD", float(fill), ts, end_idx - sig.bar_idx, unit_size, float(gross_u)


def run_discovery_stats(
    dfs: Dict[str, pd.DataFrame],
    signals_by_symbol: Dict[str, List[Signal]],
    risk_cfg: RiskConfig,
    sl_units: float,
    tp_units: float,
    be_trigger_units: float,
    enable_be: bool = True,
) -> pd.DataFrame:
    """Stage 1: Measure empirical winrate, net PnL in Units, and PF for every signal independently."""
    entry_fee = risk_cfg.maker_fee_rate if risk_cfg.use_maker_entry else risk_cfg.taker_fee_rate
    exit_fee_tp = risk_cfg.maker_fee_rate
    exit_fee_sl = risk_cfg.taker_fee_rate

    rows = []
    for sym, sigs in signals_by_symbol.items():
        df = dfs[sym]
        for sig in sigs:
            reason, exit_p, exit_ts, bars, u_size, gross_u = simulate_single_trade(
                df=df,
                sig=sig,
                sl_units=sl_units,
                tp_units=tp_units,
                be_trigger_units=be_trigger_units,
                enable_be=enable_be,
                atr_to_unit_mult=risk_cfg.atr_to_unit_mult,
                entry_fee_rate=entry_fee,
                exit_fee_rate_tp=exit_fee_tp,
                exit_fee_rate_sl=exit_fee_sl,
                slippage_bps=risk_cfg.slippage_bps,
            )
            exit_fee = exit_fee_tp if reason == "TP" else exit_fee_sl
            fee_in_units = (sig.entry_price * entry_fee + exit_p * exit_fee) / u_size
            net_u = gross_u - fee_in_units
            rows.append(
                {
                    "symbol": sym,
                    "timestamp": sig.timestamp,
                    "exit_time": exit_ts,
                    "wave": sig.wave,
                    "setup_type": sig.setup_type,
                    "pa_vol_ok": sig.pa_vol_ok,
                    "ema_aligned": sig.ema_aligned,
                    "direction": "LONG" if sig.direction == 1 else "SHORT",
                    "entry_price": sig.entry_price,
                    "exit_price": exit_p,
                    "unit_size": u_size,
                    "unit_pct": (u_size / sig.entry_price) * 100.0,
                    "exit_reason": reason,
                    "gross_units": gross_u,
                    "fee_units": fee_in_units,
                    "net_units": net_u,
                    "is_win": net_u > 0,
                    "bars_held": bars,
                }
            )
    return pd.DataFrame(rows)


def run_portfolio_backtest(
    dfs: Dict[str, pd.DataFrame],
    signals_by_symbol: Dict[str, List[Signal]],
    setup_winrates: Dict[str, float],
    risk_cfg: RiskConfig,
    start_time: Optional[pd.Timestamp] = None,
    end_time: Optional[pd.Timestamp] = None,
    require_pa_vol_for_w2: bool = True,
) -> Tuple[List[TradeRecord], pd.DataFrame, CompoundingManager]:
    """Stage 2: Event-driven portfolio simulation with 20-step +8% compounding, max 2 open, max 10/day."""
    comp = CompoundingManager(risk_cfg)
    equity = risk_cfg.initial_capital

    entry_fee = risk_cfg.maker_fee_rate if risk_cfg.use_maker_entry else risk_cfg.taker_fee_rate
    exit_fee_tp = risk_cfg.maker_fee_rate
    exit_fee_sl = risk_cfg.taker_fee_rate

    # Pre-resolve candidate trades chronologically and filter by start/end
    all_signals: List[Signal] = []
    for sym, sigs in signals_by_symbol.items():
        for s in sigs:
            if start_time is not None and s.timestamp < start_time:
                continue
            if end_time is not None and s.timestamp >= end_time:
                continue
            if require_pa_vol_for_w2 and s.setup_type == "W2_TREND" and not s.pa_vol_ok:
                continue
            all_signals.append(s)

    all_signals.sort(key=lambda s: (s.timestamp, s.symbol))

    # Active open trades: list of (exit_time, TradeRecord)
    active_trades: List[Tuple[pd.Timestamp, TradeRecord]] = []
    closed_trades: List[TradeRecord] = []
    daily_counts: Dict[str, int] = {}
    equity_curve = [{"timestamp": all_signals[0].timestamp if all_signals else pd.Timestamp.utcnow(), "equity": equity, "step": 1}]

    def settle_until(now_ts: pd.Timestamp):
        nonlocal equity
        still_open = []
        to_close = []
        for ex_ts, tr in active_trades:
            if ex_ts <= now_ts:
                to_close.append((ex_ts, tr))
            else:
                still_open.append((ex_ts, tr))
        to_close.sort(key=lambda x: x[0])
        for ex_ts, tr in to_close:
            equity += tr.net_pnl
            tr.equity_after = equity
            comp.update_equity(equity, ex_ts)
            closed_trades.append(tr)
            equity_curve.append({"timestamp": ex_ts, "equity": equity, "step": comp.step})
        active_trades[:] = still_open

    for sig in all_signals:
        settle_until(sig.timestamp)
        if equity <= 0:
            break

        # Enforce max 2 concurrent trades across portfolio (and no duplicate same-symbol open trade)
        if len(active_trades) >= risk_cfg.max_concurrent_trades:
            continue
        if any(tr.symbol == sig.symbol for _, tr in active_trades):
            continue

        # Enforce max 10 trades per UTC day
        day_key = sig.timestamp.strftime("%Y-%m-%d")
        if daily_counts.get(day_key, 0) >= risk_cfg.max_trades_per_day:
            continue

        # Lookup empirical winrate for this setup key (symbol+setup or fallback to setup)
        wr = setup_winrates.get(f"{sig.symbol}:{sig.setup_type}", setup_winrates.get(sig.setup_type, 0.0))
        plan = comp.compute_size(
            entry_price=sig.entry_price,
            atr=sig.atr,
            setup_winrate=wr,
            current_equity=equity,
        )
        if plan.tier_label == "SKIP" or plan.qty <= 0:
            continue

        # Simulate forward on this symbol's M5 bars
        reason, exit_p, exit_ts, bars, u_size, gross_u = simulate_single_trade(
            df=dfs[sig.symbol],
            sig=sig,
            sl_units=risk_cfg.sl_units,
            tp_units=risk_cfg.tp_units,
            be_trigger_units=risk_cfg.be_trigger_units,
            enable_be=risk_cfg.enable_be,
            atr_to_unit_mult=risk_cfg.atr_to_unit_mult,
            entry_fee_rate=entry_fee,
            exit_fee_rate_tp=exit_fee_tp,
            exit_fee_rate_sl=exit_fee_sl,
            slippage_bps=risk_cfg.slippage_bps,
        )
        exit_fee = exit_fee_tp if reason == "TP" else exit_fee_sl
        gross_pnl = sig.direction * (exit_p - sig.entry_price) * plan.qty
        fee_usd = (sig.entry_price * entry_fee + exit_p * exit_fee) * plan.qty
        net_pnl = gross_pnl - fee_usd

        tr = TradeRecord(
            symbol=sig.symbol,
            setup_type=sig.setup_type,
            wave=sig.wave,
            pa_vol_ok=sig.pa_vol_ok,
            ema_aligned=sig.ema_aligned,
            direction=sig.direction,
            tier_label=plan.tier_label,
            step_at_entry=comp.step,
            entry_time=sig.timestamp,
            exit_time=exit_ts,
            entry_price=sig.entry_price,
            exit_price=exit_p,
            unit_size=u_size,
            qty=plan.qty,
            notional=plan.notional,
            sl_price=sig.entry_price - sig.direction * plan.sl_distance,
            tp_price=sig.entry_price + sig.direction * plan.tp_distance,
            exit_reason=reason,
            gross_units=gross_u,
            gross_pnl=gross_pnl,
            fee_usd=fee_usd,
            net_pnl=net_pnl,
            equity_after=equity,
            bars_held=bars,
        )
        active_trades.append((exit_ts, tr))
        daily_counts[day_key] = daily_counts.get(day_key, 0) + 1

    # Settle remaining open trades at end of period
    if active_trades:
        far_future = pd.Timestamp("2099-01-01", tz="UTC")
        settle_until(far_future)

    eq_df = pd.DataFrame(equity_curve)
    return closed_trades, eq_df, comp
