"""Base execution interfaces and order/event models for CHICKTRADE."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional

from bot.state_store import OpenPosition


@dataclass
class OrderPlan:
    symbol: str
    tf: str
    setup_code: str
    ema: str
    direction: int              # +1 LONG, -1 SHORT
    entry_price: float
    qty: float
    unit_size: float
    sl_u: float
    tp1_u: float
    tp2_u: float
    leg2_sl_u: float
    exit_arch: str              # ONLY_TP1 | ONLY_1R | PARTIAL_NO_MOVE | PARTIAL_MOVE_SL
    tp1_frac: float             # 1.0 for ONLY_TP1 / ONLY_1R, 0.5 for PARTIAL_*
    sl_price: float
    tp1_price: float
    tp2_price: float
    leg2_sl_price: float
    sl_risk_usd: float
    vol_mult: float
    buffer_used: float
    signal_bar_ts: str
    turbo_used: bool = False
    turbo_fallback_reason: str = ""
    tier_label: str = ""
    posterior_wr: float = 0.0


@dataclass
class TradeEvent:
    event_type: str             # OPEN | TP1_PARTIAL | CLOSED
    position_id: str
    symbol: str
    tf: str
    setup_code: str
    direction: int
    exit_arch: str
    price: float
    qty: float
    outcome: str = ""
    pnl_usd: float = 0.0
    net_u: float = 0.0
    message: str = ""


class BaseBroker(ABC):
    """Abstract execution broker implemented by PaperBroker and BinanceFuturesBroker."""

    @abstractmethod
    def open_position(self, plan: OrderPlan) -> tuple[OpenPosition, TradeEvent]:
        """Submit a new position according to `plan` and persist it."""

    @abstractmethod
    def sync_positions(
        self,
        market_prices: Dict[str, float],
        bar_highs_lows: Optional[Dict[str, tuple[float, float]]] = None,
    ) -> List[TradeEvent]:
        """Evaluate open positions against latest market highs/lows/prices and process exits."""

    @abstractmethod
    def close_position(self, pos: OpenPosition, exit_price: float, reason: str = "MANUAL") -> TradeEvent:
        """Force-close an open position immediately."""

    @abstractmethod
    def get_account_equity(self) -> float:
        """Return current account equity in USD."""
