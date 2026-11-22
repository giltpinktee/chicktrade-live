"""HMAC-SHA256 signed Binance USD-M Futures REST adapter (TESTNET & LIVE).

Endpoints:
  - TESTNET: https://testnet.binancefuture.com
  - LIVE   : https://fapi.binance.com (ONLY permitted when LIVE_CONFIRMED=true)

Includes `exchangeInfo` precision rounding (`stepSize`, `tickSize`) using Decimal
to guarantee Binance filter compliance.
"""
from __future__ import annotations

from decimal import Decimal, ROUND_DOWN, ROUND_HALF_UP
import hashlib
import hmac
import json
import time
from typing import Any, Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from bot.bot_config import BotConfig
from bot.execution.base import BaseBroker, OrderPlan, TradeEvent
from bot.market_feed import BinanceMarketFeed
from bot.state_store import OpenPosition, StateStore, utc_now_iso

TESTNET_BASE_URL = "https://testnet.binancefuture.com"
LIVE_BASE_URL = "https://fapi.binance.com"


def round_to_step(value: float, step: float, floor: bool = True) -> float:
    """Round a numeric value cleanly to Binance stepSize / tickSize using Decimal."""
    if step <= 0:
        return value
    d_val = Decimal(str(value))
    d_step = Decimal(str(step))
    rounding = ROUND_DOWN if floor else ROUND_HALF_UP
    units = (d_val / d_step).to_integral_value(rounding=rounding)
    return float(units * d_step)


def format_decimal(value: float, step: float, floor: bool = True) -> str:
    """Format a rounded float without scientific notation for Binance query params."""
    if step <= 0:
        return f"{value:.8f}".rstrip("0").rstrip(".")
    d_step = Decimal(str(step)).normalize()
    d_val = Decimal(str(value))
    rounding = ROUND_DOWN if floor else ROUND_HALF_UP
    units = (d_val / d_step).to_integral_value(rounding=rounding)
    quantized = (units * d_step).quantize(d_step)
    s = format(quantized, "f")
    return s


class BinanceFuturesBroker(BaseBroker):
    """Signed Binance USD-M Futures broker for TESTNET and LIVE modes."""

    def __init__(
        self,
        cfg: BotConfig,
        store: StateStore,
        market_feed: Optional[BinanceMarketFeed] = None,
        timeout: float = 10.0,
    ):
        self.cfg = cfg
        self.store = store
        self.timeout = float(timeout)

        mode = cfg.execution_mode.upper().strip()
        if mode == "LIVE":
            if not cfg.live_confirmed:
                raise PermissionError(
                    "Refusing to initialize BinanceFuturesBroker in LIVE mode without LIVE_CONFIRMED=true."
                )
            self.base_url = LIVE_BASE_URL
        elif mode == "TESTNET":
            self.base_url = TESTNET_BASE_URL
        else:
            raise ValueError(f"BinanceFuturesBroker requires EXECUTION_MODE=TESTNET or LIVE, got {mode!r}")

        if not cfg.binance_api_key or not cfg.binance_api_secret:
            raise ValueError("BINANCE_API_KEY and BINANCE_API_SECRET are required for TESTNET/LIVE mode.")

        self.api_key = cfg.binance_api_key
        self.api_secret = cfg.binance_api_secret
        self.feed = market_feed or BinanceMarketFeed(base_url=self.base_url, timeout=self.timeout)
        self._symbol_filters: Dict[str, dict] = {}

    def _sign_params(self, params: Dict[str, Any]) -> str:
        payload = dict(params)
        payload["timestamp"] = int(time.time() * 1000)
        payload.setdefault("recvWindow", 5000)
        query = urlencode(payload)
        sig = hmac.new(self.api_secret.encode("utf-8"), query.encode("utf-8"), hashlib.sha256).hexdigest()
        return f"{query}&signature={sig}"

    def _signed_request(self, method: str, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        signed_qs = self._sign_params(params or {})
        url = f"{self.base_url}{path}?{signed_qs}"
        req = Request(
            url,
            method=method.upper(),
            headers={
                "X-MBX-APIKEY": self.api_key,
                "User-Agent": "CHICKTRADE-Bot/2.0",
                "Accept": "application/json",
            },
        )
        try:
            with urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Binance API HTTP {exc.code} on {method} {path}: {body}") from exc
        except (URLError, TimeoutError) as exc:
            raise RuntimeError(f"Binance API network error on {method} {path}: {exc}") from exc

    def _ensure_symbol_filter(self, symbol: str) -> dict:
        sym = symbol.upper()
        if sym not in self._symbol_filters:
            info = self.feed.fetch_exchange_info([sym])
            if sym in info:
                self._symbol_filters[sym] = info[sym]
            else:
                self._symbol_filters[sym] = {
                    "stepSize": 0.001,
                    "tickSize": 0.01,
                    "minQty": 0.001,
                    "minNotional": 5.0,
                }
        return self._symbol_filters[sym]

    def round_qty(self, symbol: str, qty: float) -> tuple[float, str]:
        flt = self._ensure_symbol_filter(symbol)
        step = flt["stepSize"]
        val = round_to_step(qty, step, floor=True)
        return val, format_decimal(val, step, floor=True)

    def round_price(self, symbol: str, price: float) -> tuple[float, str]:
        flt = self._ensure_symbol_filter(symbol)
        tick = flt["tickSize"]
        val = round_to_step(price, tick, floor=False)
        return val, format_decimal(val, tick, floor=False)

    def get_account_equity(self) -> float:
        balances = self._signed_request("GET", "/fapi/v2/balance")
        for b in balances:
            if b.get("asset") == "USDT":
                eq = float(b.get("balance", 0.0)) + float(b.get("crossUnPnl", 0.0))
                self.store.update_equity(eq, self.cfg.compounding_steps, self.cfg.step_gain_pct)
                return eq
        return self.store.get_equity()

    def get_exchange_positions(self) -> List[dict]:
        return self._signed_request("GET", "/fapi/v2/positionRisk")

    def place_order(self, params: Dict[str, Any]) -> dict:
        return self._signed_request("POST", "/fapi/v1/order", params)

    def cancel_order(self, symbol: str, order_id: str) -> dict:
        return self._signed_request("DELETE", "/fapi/v1/order", {"symbol": symbol.upper(), "orderId": order_id})

    def open_position(self, plan: OrderPlan) -> tuple[OpenPosition, TradeEvent]:
        sym = plan.symbol.upper()
        entry_side = "BUY" if plan.direction == 1 else "SELL"
        exit_side = "SELL" if plan.direction == 1 else "BUY"

        qty_f, qty_s = self.round_qty(sym, plan.qty)
        if qty_f <= 0:
            raise ValueError(f"Rounded quantity for {sym} is 0 (raw={plan.qty})")

        sl_f, sl_s = self.round_price(sym, plan.sl_price)
        tp1_f, tp1_s = self.round_price(sym, plan.tp1_price)
        tp2_f, tp2_s = self.round_price(sym, plan.tp2_price)
        leg2_sl_f, _ = self.round_price(sym, plan.leg2_sl_price)

        # 1. Market entry order
        entry_resp = self.place_order(
            {
                "symbol": sym,
                "side": entry_side,
                "type": "MARKET",
                "quantity": qty_s,
            }
        )
        entry_oid = str(entry_resp.get("orderId", ""))
        fill_price = float(entry_resp.get("avgPrice") or plan.entry_price)
        if fill_price <= 0:
            fill_price = plan.entry_price

        # 2. Protective Stop-Loss (STOP_MARKET closePosition)
        sl_resp = self.place_order(
            {
                "symbol": sym,
                "side": exit_side,
                "type": "STOP_MARKET",
                "stopPrice": sl_s,
                "closePosition": "true",
                "workingType": "MARK_PRICE",
            }
        )
        sl_oid = str(sl_resp.get("orderId", ""))

        # 3. TP1 (and TP2 for partial architectures)
        tp1_oid = ""
        tp2_oid = ""
        if plan.exit_arch in ("ONLY_TP1", "ONLY_1R"):
            tp1_resp = self.place_order(
                {
                    "symbol": sym,
                    "side": exit_side,
                    "type": "LIMIT",
                    "timeInForce": "GTC",
                    "reduceOnly": "true",
                    "quantity": qty_s,
                    "price": tp1_s,
                }
            )
            tp1_oid = str(tp1_resp.get("orderId", ""))
        else:
            q1_f, q1_s = self.round_qty(sym, qty_f * plan.tp1_frac)
            q2_f, q2_s = self.round_qty(sym, qty_f - q1_f)
            if q1_f > 0:
                tp1_resp = self.place_order(
                    {
                        "symbol": sym,
                        "side": exit_side,
                        "type": "LIMIT",
                        "timeInForce": "GTC",
                        "reduceOnly": "true",
                        "quantity": q1_s,
                        "price": tp1_s,
                    }
                )
                tp1_oid = str(tp1_resp.get("orderId", ""))
            if q2_f > 0:
                tp2_resp = self.place_order(
                    {
                        "symbol": sym,
                        "side": exit_side,
                        "type": "LIMIT",
                        "timeInForce": "GTC",
                        "reduceOnly": "true",
                        "quantity": q2_s,
                        "price": tp2_s,
                    }
                )
                tp2_oid = str(tp2_resp.get("orderId", ""))

        pos_id = f"BINANCE-{sym}-{entry_oid or int(time.time())}"
        pos = OpenPosition(
            position_id=pos_id,
            symbol=sym,
            tf=plan.tf,
            setup_code=plan.setup_code,
            ema=plan.ema,
            direction=plan.direction,
            entry_price=fill_price,
            qty=qty_f,
            remaining_qty=qty_f,
            unit_size=plan.unit_size,
            sl_u=plan.sl_u,
            tp1_u=plan.tp1_u,
            tp2_u=plan.tp2_u,
            leg2_sl_u=plan.leg2_sl_u,
            exit_arch=plan.exit_arch,
            tp1_frac=plan.tp1_frac,
            tp1_filled=False,
            leg2_sl_moved=False,
            sl_price=sl_f,
            tp1_price=tp1_f,
            tp2_price=tp2_f,
            leg2_sl_price=leg2_sl_f,
            sl_risk_usd=plan.sl_risk_usd,
            vol_mult=plan.vol_mult,
            buffer_used=plan.buffer_used,
            opened_at=utc_now_iso(),
            signal_bar_ts=plan.signal_bar_ts,
            exchange_order_id=entry_oid,
            sl_order_id=sl_oid,
            tp1_order_id=tp1_oid,
            tp2_order_id=tp2_oid,
        )
        self.store.save_open_position(pos)
        self.store.increment_daily_trades()

        evt = TradeEvent(
            event_type="OPEN",
            position_id=pos_id,
            symbol=sym,
            tf=plan.tf,
            setup_code=plan.setup_code,
            direction=plan.direction,
            exit_arch=plan.exit_arch,
            price=fill_price,
            qty=qty_f,
            outcome="OPEN",
            message=f"[{self.cfg.execution_mode}] OPEN {entry_side} {sym} qty={qty_s} @ {fill_price:.4f} SL={sl_s}",
        )
        return pos, evt

    def move_leg2_sl(self, pos: OpenPosition) -> OpenPosition:
        """Cancel original SL order and place updated Leg 2 SL order on Binance."""
        sym = pos.symbol.upper()
        exit_side = "SELL" if pos.direction == 1 else "BUY"
        if pos.sl_order_id:
            try:
                self.cancel_order(sym, pos.sl_order_id)
            except RuntimeError:
                pass
        leg2_f, leg2_s = self.round_price(sym, pos.leg2_sl_price)
        resp = self.place_order(
            {
                "symbol": sym,
                "side": exit_side,
                "type": "STOP_MARKET",
                "stopPrice": leg2_s,
                "closePosition": "true",
                "workingType": "MARK_PRICE",
            }
        )
        pos.sl_price = leg2_f
        pos.sl_order_id = str(resp.get("orderId", ""))
        pos.leg2_sl_moved = True
        self.store.save_open_position(pos)
        return pos

    def sync_positions(
        self,
        market_prices: Dict[str, float],
        bar_highs_lows: Optional[Dict[str, tuple[float, float]]] = None,
    ) -> List[TradeEvent]:
        """Reconcile local open positions against Binance positionRisk & TP1 fills."""
        events: List[TradeEvent] = []
        open_positions = self.store.get_open_positions()
        if not open_positions:
            return events

        risk_list = self.get_exchange_positions()
        risk_map = {r["symbol"].upper(): abs(float(r.get("positionAmt", 0.0))) for r in risk_list}

        for pos in open_positions:
            live_qty = risk_map.get(pos.symbol.upper(), 0.0)
            if live_qty <= 0.0:
                self.store.delete_open_position(pos.position_id)
                events.append(
                    TradeEvent(
                        event_type="CLOSED",
                        position_id=pos.position_id,
                        symbol=pos.symbol,
                        tf=pos.tf,
                        setup_code=pos.setup_code,
                        direction=pos.direction,
                        exit_arch=pos.exit_arch,
                        price=market_prices.get(pos.symbol, pos.entry_price),
                        qty=pos.qty,
                        outcome="EXCHANGE_CLOSED",
                        message=f"[{self.cfg.execution_mode}] {pos.symbol} {pos.setup_code} closed on exchange.",
                    )
                )
            elif (
                not pos.tp1_filled
                and pos.exit_arch == "PARTIAL_MOVE_SL"
                and live_qty <= pos.qty * (1.0 - pos.tp1_frac + 0.05)
            ):
                pos.tp1_filled = True
                pos.remaining_qty = live_qty
                self.move_leg2_sl(pos)
                events.append(
                    TradeEvent(
                        event_type="TP1_PARTIAL",
                        position_id=pos.position_id,
                        symbol=pos.symbol,
                        tf=pos.tf,
                        setup_code=pos.setup_code,
                        direction=pos.direction,
                        exit_arch=pos.exit_arch,
                        price=pos.tp1_price,
                        qty=pos.qty - live_qty,
                        outcome="TP1_PARTIAL",
                        message=f"[{self.cfg.execution_mode}] TP1 filled on {pos.symbol}; Leg2 SL moved to {pos.sl_price:.4f}",
                    )
                )
        return events

    def close_position(self, pos: OpenPosition, exit_price: float, reason: str = "MANUAL") -> TradeEvent:
        sym = pos.symbol.upper()
        exit_side = "SELL" if pos.direction == 1 else "BUY"
        qty_f, qty_s = self.round_qty(sym, pos.remaining_qty)
        if qty_f > 0:
            self.place_order(
                {
                    "symbol": sym,
                    "side": exit_side,
                    "type": "MARKET",
                    "reduceOnly": "true",
                    "quantity": qty_s,
                }
            )
        self.store.delete_open_position(pos.position_id)
        return TradeEvent(
            event_type="CLOSED",
            position_id=pos.position_id,
            symbol=sym,
            tf=pos.tf,
            setup_code=pos.setup_code,
            direction=pos.direction,
            exit_arch=pos.exit_arch,
            price=exit_price,
            qty=qty_f,
            outcome=reason,
            message=f"[{self.cfg.execution_mode}] Force-closed {sym} ({reason})",
        )
