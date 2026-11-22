"""Paper trading broker with full 2-leg execution across all 4 CHICKTRADE Exit Architectures.

Exit Architectures supported:
  1. ONLY_TP1        : 100% exit at TP1 (or SL)
  2. ONLY_1R         : 100% exit at 1R = SL_U (or SL)
  3. PARTIAL_NO_MOVE : tp1_frac (50%) exit at TP1; remaining keeps original SL and targets TP2
  4. PARTIAL_MOVE_SL : tp1_frac (50%) exit at TP1; moves Leg2 SL to leg2_sl_price; remaining targets TP2

Precedence & Friction:
  - Conservative SL-first precedence: if both SL and TP are touched in the same candle, SL ALWAYS triggers first.
  - Maker fee 0.02% on limit entry & TP1/TP2 fills; Taker fee 0.05% + 1 bps slippage on SL/market exits.
"""
from __future__ import annotations

from typing import Dict, List, Optional
import uuid

from bot.bot_config import BotConfig
from bot.execution.base import BaseBroker, OrderPlan, TradeEvent
from bot.state_store import ClosedTrade, OpenPosition, StateStore, utc_now_iso


class PaperBroker(BaseBroker):
    """Realistic USD-M Futures paper broker backed by SQLite StateStore."""

    def __init__(self, cfg: BotConfig, store: StateStore):
        self.cfg = cfg
        self.store = store
        self.maker_fee = float(cfg.maker_fee_rate)
        self.taker_fee = float(cfg.taker_fee_rate)
        self.slip_rate = float(cfg.slippage_bps) / 10_000.0

    def get_account_equity(self) -> float:
        return self.store.get_equity()

    def open_position(self, plan: OrderPlan) -> tuple[OpenPosition, TradeEvent]:
        pos_id = f"PAPER-{plan.symbol}-{uuid.uuid4().hex[:8].upper()}"
        entry_fee = plan.qty * plan.entry_price * self.maker_fee
        opened_at = utc_now_iso()

        pos = OpenPosition(
            position_id=pos_id,
            symbol=plan.symbol,
            tf=plan.tf,
            setup_code=plan.setup_code,
            ema=plan.ema,
            direction=plan.direction,
            entry_price=plan.entry_price,
            qty=plan.qty,
            remaining_qty=plan.qty,
            unit_size=plan.unit_size,
            sl_u=plan.sl_u,
            tp1_u=plan.tp1_u,
            tp2_u=plan.tp2_u,
            leg2_sl_u=plan.leg2_sl_u,
            exit_arch=plan.exit_arch,
            tp1_frac=plan.tp1_frac,
            tp1_filled=False,
            leg2_sl_moved=False,
            sl_price=plan.sl_price,
            tp1_price=plan.tp1_price,
            tp2_price=plan.tp2_price,
            leg2_sl_price=plan.leg2_sl_price,
            sl_risk_usd=plan.sl_risk_usd,
            vol_mult=plan.vol_mult,
            buffer_used=plan.buffer_used,
            leg1_realized_pnl_usd=0.0,
            leg1_realized_net_u=0.0,
            entry_fee_usd=entry_fee,
            opened_at=opened_at,
            signal_bar_ts=plan.signal_bar_ts,
        )
        self.store.save_open_position(pos)
        self.store.increment_daily_trades()

        side_str = "LONG" if plan.direction == 1 else "SHORT"
        evt = TradeEvent(
            event_type="OPEN",
            position_id=pos_id,
            symbol=plan.symbol,
            tf=plan.tf,
            setup_code=plan.setup_code,
            direction=plan.direction,
            exit_arch=plan.exit_arch,
            price=plan.entry_price,
            qty=plan.qty,
            outcome="OPEN",
            message=(
                f"OPEN {side_str} {plan.symbol} {plan.tf} {plan.setup_code} [{plan.exit_arch}] "
                f"@ {plan.entry_price:.4f} qty={plan.qty:.4f} SL={plan.sl_price:.4f} "
                f"TP1={plan.tp1_price:.4f} TP2={plan.tp2_price:.4f} buf={plan.buffer_used:.0f}G"
            ),
        )
        return pos, evt

    def sync_positions(
        self,
        market_prices: Dict[str, float],
        bar_highs_lows: Optional[Dict[str, tuple[float, float]]] = None,
    ) -> List[TradeEvent]:
        """Process all open positions against latest candle high/low (or live price fallback)."""
        events: List[TradeEvent] = []
        positions = self.store.get_open_positions()
        for pos in positions:
            price = market_prices.get(pos.symbol)
            if bar_highs_lows and pos.symbol in bar_highs_lows:
                bar_high, bar_low = bar_highs_lows[pos.symbol]
                if price is not None:
                    bar_high = max(bar_high, price)
                    bar_low = min(bar_low, price)
            elif price is not None:
                bar_high = price
                bar_low = price
            else:
                continue

            events.extend(self.evaluate_position_bar(pos, bar_high=bar_high, bar_low=bar_low))
        return events

    def evaluate_position_bar(self, pos: OpenPosition, bar_high: float, bar_low: float) -> List[TradeEvent]:
        """Evaluate a single open position on a candle's [bar_low, bar_high] with strict SL-first precedence."""
        events: List[TradeEvent] = []
        d = pos.direction
        arch = pos.exit_arch

        # --- Single-leg architectures: ONLY_TP1 and ONLY_1R ---
        if arch in ("ONLY_TP1", "ONLY_1R"):
            sl_hit = (bar_low <= pos.sl_price) if d == 1 else (bar_high >= pos.sl_price)
            if sl_hit:
                events.append(self._finalize_full_sl(pos))
                return events

            tp_hit = (bar_high >= pos.tp1_price) if d == 1 else (bar_low <= pos.tp1_price)
            if tp_hit:
                events.append(self._finalize_single_leg_tp(pos))
                return events
            return events

        # --- Partial 2-leg architectures: PARTIAL_NO_MOVE and PARTIAL_MOVE_SL ---
        if not pos.tp1_filled:
            # Conservative SL-first precedence BEFORE TP1
            sl_hit = (bar_low <= pos.sl_price) if d == 1 else (bar_high >= pos.sl_price)
            if sl_hit:
                events.append(self._finalize_full_sl(pos))
                return events

            tp1_hit = (bar_high >= pos.tp1_price) if d == 1 else (bar_low <= pos.tp1_price)
            if tp1_hit:
                tp1_evt = self._fill_leg1_tp1(pos)
                events.append(tp1_evt)
                # If TP2 is also reached on the same favorable bar, complete Leg 2 at TP2
                tp2_hit = (bar_high >= pos.tp2_price) if d == 1 else (bar_low <= pos.tp2_price)
                if tp2_hit:
                    events.append(self._finalize_leg2_tp2(pos))
                return events
            return events

        # --- Leg 2 already active (TP1 was previously filled) ---
        active_sl = pos.leg2_sl_price if (arch == "PARTIAL_MOVE_SL" and pos.leg2_sl_moved) else pos.sl_price
        # Conservative SL-first precedence for Leg 2
        leg2_sl_hit = (bar_low <= active_sl) if d == 1 else (bar_high >= active_sl)
        if leg2_sl_hit:
            events.append(self._finalize_leg2_sl(pos, active_sl))
            return events

        tp2_hit = (bar_high >= pos.tp2_price) if d == 1 else (bar_low <= pos.tp2_price)
        if tp2_hit:
            events.append(self._finalize_leg2_tp2(pos))
            return events

        return events

    def _fill_leg1_tp1(self, pos: OpenPosition) -> TradeEvent:
        d = pos.direction
        frac = pos.tp1_frac
        leg1_qty = pos.qty * frac
        rem_qty = max(pos.qty - leg1_qty, 0.0)

        fill_price = pos.tp1_price  # Limit TP1 -> Maker fee, zero slippage
        gross_usd = d * (fill_price - pos.entry_price) * leg1_qty
        entry_fee_leg1 = pos.entry_fee_usd * frac
        exit_fee_leg1 = leg1_qty * fill_price * self.maker_fee
        net_pnl_leg1 = gross_usd - (entry_fee_leg1 + exit_fee_leg1)

        fee1_u = (pos.entry_price * self.maker_fee + fill_price * self.maker_fee) / pos.unit_size
        gross1_u = d * (fill_price - pos.entry_price) / pos.unit_size
        leg1_net_u = frac * (gross1_u - fee1_u)

        pos.tp1_filled = True
        pos.remaining_qty = rem_qty
        pos.leg1_realized_pnl_usd = net_pnl_leg1
        pos.leg1_realized_net_u = leg1_net_u

        if pos.exit_arch == "PARTIAL_MOVE_SL":
            pos.sl_price = pos.leg2_sl_price
            pos.leg2_sl_moved = True
            adverse_dist = max(0.0, d * (pos.entry_price - pos.leg2_sl_price))
            pos.sl_risk_usd = rem_qty * adverse_dist
        else:
            # PARTIAL_NO_MOVE keeps original SL on remaining qty
            adverse_dist = max(0.0, d * (pos.entry_price - pos.sl_price))
            pos.sl_risk_usd = rem_qty * adverse_dist

        self.store.save_open_position(pos)

        # Credit Leg 1 realized PnL to equity & daily PnL
        new_eq = self.store.get_equity() + net_pnl_leg1
        self.store.update_equity(new_eq, self.cfg.compounding_steps, self.cfg.step_gain_pct)
        self.store.record_daily_pnl(net_pnl_leg1, is_full_sl=False)

        sl_note = f" | Leg2 SL -> {pos.sl_price:.4f}" if pos.leg2_sl_moved else " | Leg2 SL kept"
        return TradeEvent(
            event_type="TP1_PARTIAL",
            position_id=pos.position_id,
            symbol=pos.symbol,
            tf=pos.tf,
            setup_code=pos.setup_code,
            direction=pos.direction,
            exit_arch=pos.exit_arch,
            price=fill_price,
            qty=leg1_qty,
            outcome="TP1_PARTIAL",
            pnl_usd=net_pnl_leg1,
            net_u=leg1_net_u,
            message=f"TP1 ({int(frac*100)}%) {pos.symbol} {pos.setup_code} @ {fill_price:.4f} PnL=${net_pnl_leg1:+.2f}{sl_note}",
        )

    def _finalize_full_sl(self, pos: OpenPosition) -> TradeEvent:
        d = pos.direction
        fill_price = pos.sl_price * (1.0 - d * self.slip_rate)
        gross_usd = d * (fill_price - pos.entry_price) * pos.qty
        exit_fee = pos.qty * fill_price * self.taker_fee
        total_fees = pos.entry_fee_usd + exit_fee
        net_pnl = gross_usd - total_fees

        gross_u = d * (fill_price - pos.entry_price) / pos.unit_size
        fee_u = (pos.entry_price * self.maker_fee + fill_price * self.taker_fee) / pos.unit_size
        net_u = gross_u - fee_u

        return self._commit_close(
            pos=pos,
            exit_price=fill_price,
            outcome="SL",
            tp1_hit=False,
            leg2_sl_hit=False,
            tp2_hit=False,
            leg_pnl_usd=net_pnl,
            total_pnl_usd=net_pnl,
            total_fees_usd=total_fees,
            total_net_u=net_u,
            is_full_sl=True,
        )

    def _finalize_single_leg_tp(self, pos: OpenPosition) -> TradeEvent:
        d = pos.direction
        fill_price = pos.tp1_price  # Limit TP -> Maker fee
        gross_usd = d * (fill_price - pos.entry_price) * pos.qty
        exit_fee = pos.qty * fill_price * self.maker_fee
        total_fees = pos.entry_fee_usd + exit_fee
        net_pnl = gross_usd - total_fees

        gross_u = d * (fill_price - pos.entry_price) / pos.unit_size
        fee_u = (pos.entry_price * self.maker_fee + fill_price * self.maker_fee) / pos.unit_size
        net_u = gross_u - fee_u

        outcome = "TP1" if pos.exit_arch == "ONLY_TP1" else "TP_1R"
        return self._commit_close(
            pos=pos,
            exit_price=fill_price,
            outcome=outcome,
            tp1_hit=True,
            leg2_sl_hit=False,
            tp2_hit=False,
            leg_pnl_usd=net_pnl,
            total_pnl_usd=net_pnl,
            total_fees_usd=total_fees,
            total_net_u=net_u,
            is_full_sl=False,
        )

    def _finalize_leg2_sl(self, pos: OpenPosition, sl_trigger_price: float) -> TradeEvent:
        d = pos.direction
        rem_frac = max(1.0 - pos.tp1_frac, 0.0)
        rem_qty = pos.remaining_qty
        fill_price = sl_trigger_price * (1.0 - d * self.slip_rate)

        gross_usd = d * (fill_price - pos.entry_price) * rem_qty
        entry_fee_leg2 = pos.entry_fee_usd * rem_frac
        exit_fee_leg2 = rem_qty * fill_price * self.taker_fee
        leg2_pnl = gross_usd - (entry_fee_leg2 + exit_fee_leg2)

        gross2_u = d * (fill_price - pos.entry_price) / pos.unit_size
        fee2_u = (pos.entry_price * self.maker_fee + fill_price * self.taker_fee) / pos.unit_size
        leg2_net_u = rem_frac * (gross2_u - fee2_u)

        total_pnl = pos.leg1_realized_pnl_usd + leg2_pnl
        total_net_u = pos.leg1_realized_net_u + leg2_net_u
        leg1_exit_fee = (pos.qty * pos.tp1_frac) * pos.tp1_price * self.maker_fee
        total_fees = pos.entry_fee_usd + leg1_exit_fee + exit_fee_leg2

        outcome = "BE" if pos.exit_arch == "PARTIAL_MOVE_SL" else "LEG2_SL"
        return self._commit_close(
            pos=pos,
            exit_price=fill_price,
            outcome=outcome,
            tp1_hit=True,
            leg2_sl_hit=True,
            tp2_hit=False,
            leg_pnl_usd=leg2_pnl,
            total_pnl_usd=total_pnl,
            total_fees_usd=total_fees,
            total_net_u=total_net_u,
            is_full_sl=False,
        )

    def _finalize_leg2_tp2(self, pos: OpenPosition) -> TradeEvent:
        d = pos.direction
        rem_frac = max(1.0 - pos.tp1_frac, 0.0)
        rem_qty = pos.remaining_qty
        fill_price = pos.tp2_price  # Limit TP2 -> Maker fee

        gross_usd = d * (fill_price - pos.entry_price) * rem_qty
        entry_fee_leg2 = pos.entry_fee_usd * rem_frac
        exit_fee_leg2 = rem_qty * fill_price * self.maker_fee
        leg2_pnl = gross_usd - (entry_fee_leg2 + exit_fee_leg2)

        gross2_u = d * (fill_price - pos.entry_price) / pos.unit_size
        fee2_u = (pos.entry_price * self.maker_fee + fill_price * self.maker_fee) / pos.unit_size
        leg2_net_u = rem_frac * (gross2_u - fee2_u)

        total_pnl = pos.leg1_realized_pnl_usd + leg2_pnl
        total_net_u = pos.leg1_realized_net_u + leg2_net_u
        leg1_exit_fee = (pos.qty * pos.tp1_frac) * pos.tp1_price * self.maker_fee
        total_fees = pos.entry_fee_usd + leg1_exit_fee + exit_fee_leg2

        return self._commit_close(
            pos=pos,
            exit_price=fill_price,
            outcome="TP2",
            tp1_hit=True,
            leg2_sl_hit=False,
            tp2_hit=True,
            leg_pnl_usd=leg2_pnl,
            total_pnl_usd=total_pnl,
            total_fees_usd=total_fees,
            total_net_u=total_net_u,
            is_full_sl=False,
        )

    def close_position(self, pos: OpenPosition, exit_price: float, reason: str = "MANUAL") -> TradeEvent:
        d = pos.direction
        fill_price = exit_price * (1.0 - d * self.slip_rate)
        rem_qty = pos.remaining_qty
        rem_frac = rem_qty / pos.qty if pos.qty > 0 else 1.0

        gross_usd = d * (fill_price - pos.entry_price) * rem_qty
        entry_fee_rem = pos.entry_fee_usd * rem_frac
        exit_fee_rem = rem_qty * fill_price * self.taker_fee
        leg_pnl = gross_usd - (entry_fee_rem + exit_fee_rem)

        gross_u = d * (fill_price - pos.entry_price) / pos.unit_size
        fee_u = (pos.entry_price * self.maker_fee + fill_price * self.taker_fee) / pos.unit_size
        leg_net_u = rem_frac * (gross_u - fee_u)

        total_pnl = pos.leg1_realized_pnl_usd + leg_pnl
        total_net_u = pos.leg1_realized_net_u + leg_net_u
        leg1_exit_fee = (pos.qty * pos.tp1_frac * pos.tp1_price * self.maker_fee) if pos.tp1_filled else 0.0
        total_fees = pos.entry_fee_usd + leg1_exit_fee + exit_fee_rem

        return self._commit_close(
            pos=pos,
            exit_price=fill_price,
            outcome=reason,
            tp1_hit=pos.tp1_filled,
            leg2_sl_hit=False,
            tp2_hit=False,
            leg_pnl_usd=leg_pnl,
            total_pnl_usd=total_pnl,
            total_fees_usd=total_fees,
            total_net_u=total_net_u,
            is_full_sl=(not pos.tp1_filled and total_net_u < 0),
        )

    def _commit_close(
        self,
        pos: OpenPosition,
        exit_price: float,
        outcome: str,
        tp1_hit: bool,
        leg2_sl_hit: bool,
        tp2_hit: bool,
        leg_pnl_usd: float,
        total_pnl_usd: float,
        total_fees_usd: float,
        total_net_u: float,
        is_full_sl: bool,
    ) -> TradeEvent:
        closed_at = utc_now_iso()
        trade = ClosedTrade(
            position_id=pos.position_id,
            symbol=pos.symbol,
            tf=pos.tf,
            setup_code=pos.setup_code,
            ema=pos.ema,
            direction=pos.direction,
            exit_arch=pos.exit_arch,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            qty=pos.qty,
            unit_size=pos.unit_size,
            outcome=outcome,
            tp1_hit=tp1_hit,
            leg2_sl_hit=leg2_sl_hit,
            tp2_hit=tp2_hit,
            pnl_usd=total_pnl_usd,
            fees_usd=total_fees_usd,
            net_u=total_net_u,
            opened_at=pos.opened_at,
            closed_at=closed_at,
        )
        self.store.delete_open_position(pos.position_id)
        self.store.record_closed_trade(trade, record_bayesian=True)

        # Credit only the remaining leg PnL (Leg 1 PnL was already credited when TP1 filled)
        new_eq = self.store.get_equity() + leg_pnl_usd
        self.store.update_equity(new_eq, self.cfg.compounding_steps, self.cfg.step_gain_pct)
        self.store.record_daily_pnl(leg_pnl_usd, is_full_sl=is_full_sl)

        # Update consecutive SL counter
        consec_sl = int(self.store.get_state("consec_sl", 0))
        if is_full_sl:
            self.store.set_state("consec_sl", consec_sl + 1)
        elif total_net_u > 0:
            self.store.set_state("consec_sl", 0)

        return TradeEvent(
            event_type="CLOSED",
            position_id=pos.position_id,
            symbol=pos.symbol,
            tf=pos.tf,
            setup_code=pos.setup_code,
            direction=pos.direction,
            exit_arch=pos.exit_arch,
            price=exit_price,
            qty=pos.qty,
            outcome=outcome,
            pnl_usd=total_pnl_usd,
            net_u=total_net_u,
            message=(
                f"CLOSE [{outcome}] {pos.symbol} {pos.tf} {pos.setup_code} ({pos.exit_arch}) "
                f"@ {exit_price:.4f} | Total PnL=${total_pnl_usd:+.2f} ({total_net_u:+.2f}U)"
            ),
        )
