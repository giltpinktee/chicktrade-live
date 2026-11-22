"""Terminal status dashboard & optional Telegram push notifier for CHICKTRADE Bot."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from typing import Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from bot.bot_config import BotConfig
from bot.execution.base import TradeEvent
from bot.signal_engine import CoinScanSnapshot, StormStatus
from bot.state_store import DailyCounter, OpenPosition


class BotNotifier:
    """Renders structured terminal tables and optionally sends Telegram notifications."""

    def __init__(self, cfg: BotConfig, timeout: float = 8.0):
        self.cfg = cfg
        self.timeout = float(timeout)

    @property
    def telegram_enabled(self) -> bool:
        return bool(self.cfg.telegram_bot_token and self.cfg.telegram_chat_id)

    def send_telegram(self, text: str) -> bool:
        """Send a message to Telegram if configured; gracefully no-ops when token/chat_id empty."""
        if not self.telegram_enabled:
            return False
        url = f"https://api.telegram.org/bot{self.cfg.telegram_bot_token}/sendMessage"
        data = urlencode(
            {
                "chat_id": self.cfg.telegram_chat_id,
                "text": text,
                "disable_web_page_preview": "true",
            }
        ).encode("utf-8")
        req = Request(url, data=data, method="POST")
        try:
            with urlopen(req, timeout=self.timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
                return bool(payload.get("ok", False))
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError):
            return False

    def format_status_report(
        self,
        comp_info: Dict[str, object],
        daily: DailyCounter,
        btc_storm: StormStatus,
        snapshots: List[CoinScanSnapshot],
        open_positions: List[OpenPosition],
        market_prices: Dict[str, float],
        events: List[TradeEvent],
        skipped_notes: Optional[List[str]] = None,
    ) -> str:
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        turbo_flag = "ON (ACTIVE 230G)" if comp_info.get("turbo_active") else (
            "ON (STANDBY)" if self.cfg.auto_turbo_enabled else "OFF"
        )
        active_buf = self.cfg.auto_turbo_buffer_units if comp_info.get("turbo_active") else self.cfg.buffer_units
        max_conc = self.cfg.max_concurrent_for_buffer(active_buf)
        eq = float(comp_info.get("equity", self.cfg.initial_capital))
        step = int(comp_info.get("step", 1))
        target = float(comp_info.get("target_capital", self.cfg.initial_capital * 1.08))

        lines: List[str] = []
        lines.append("=" * 96)
        lines.append(
            f"CHICKTRADE BOT [{self.cfg.execution_mode}] | {now_str} | "
            f"Equity: ${eq:,.2f} (Step {step}, Target ${target:,.2f})"
        )
        lines.append(
            f"Buffer: {active_buf:.0f}G (Max {max_conc} pos) | Auto-Turbo: {turbo_flag} | "
            f"Floor: ${self.cfg.protected_floor_usd:,.0f} | "
            f"Daily: {daily.trades_count}/{self.cfg.max_trades_per_day} trades, "
            f"{daily.sl_count} SL, PnL ${daily.realized_pnl_usd:+.2f}"
        )
        lines.append(
            f"BTC Macro Storm Guard: {'BLOCKED - ' + btc_storm.reason if btc_storm.in_storm else 'CLEAR'} "
            f"(ATR={btc_storm.atr_ratio:.2f}x, ER={btc_storm.kaufman_er:.2f})"
        )
        lines.append("-" * 96)
        lines.append(
            f"{'COIN':<10} {'LAST PRICE':>12} {'15m ADX/HTF':>13} {'30m ADX/HTF':>13} "
            f"{'1h ADX/HTF':>13} {'STORM GUARD':<26}"
        )
        lines.append("-" * 96)

        def _fmt_tf(tf_map: Dict[str, Dict[str, object]], tf: str) -> str:
            m = tf_map.get(tf)
            if not m:
                return "-"
            ht = m.get("htf_trend", 0)
            arrow = "UP" if ht == 1 else ("DN" if ht == -1 else "--")
            return f"{float(m.get('adx', 0.0)):4.1f}/{arrow}"

        for snap in snapshots:
            sg_text = f"BLOCKED({snap.storm_reason})" if snap.storm_active else snap.storm_reason
            lines.append(
                f"{snap.symbol:<10} {snap.last_price:>12.4f} "
                f"{_fmt_tf(snap.tf_metrics, '15m'):>13} "
                f"{_fmt_tf(snap.tf_metrics, '30m'):>13} "
                f"{_fmt_tf(snap.tf_metrics, '1h'):>13} "
                f"{sg_text:<26}"
            )

        lines.append("-" * 96)
        lines.append(f"OPEN POSITIONS ({len(open_positions)}/{max_conc}):")
        if not open_positions:
            lines.append("  (none)")
        else:
            lines.append(
                f"  {'SYMBOL':<9} {'TF':<4} {'SETUP':<13} {'SIDE':<5} {'ARCH':<15} "
                f"{'ENTRY':>10} {'SL':>10} {'TP1':>10} {'TP1?':<4} {'uPnL$':>9}"
            )
            for p in open_positions:
                side = "LONG" if p.direction == 1 else "SHORT"
                cur = market_prices.get(p.symbol, p.entry_price)
                upnl = p.direction * (cur - p.entry_price) * p.remaining_qty + p.leg1_realized_pnl_usd
                lines.append(
                    f"  {p.symbol:<9} {p.tf:<4} {p.setup_code:<13} {side:<5} {p.exit_arch:<15} "
                    f"{p.entry_price:>10.4f} {p.sl_price:>10.4f} {p.tp1_price:>10.4f} "
                    f"{'YES' if p.tp1_filled else 'NO':<4} {upnl:>+9.2f}"
                )

        if events:
            lines.append("-" * 96)
            lines.append("TRADE EVENTS:")
            for ev in events:
                lines.append(f"  * {ev.message}")

        if skipped_notes:
            lines.append("-" * 96)
            lines.append("FILTERED / SKIPPED SIGNALS:")
            for note in skipped_notes:
                lines.append(f"  - {note}")

        lines.append("=" * 96)
        return "\n".join(lines)

    def notify_events(self, events: List[TradeEvent]) -> int:
        sent = 0
        for ev in events:
            if self.send_telegram(f"[CHICKTRADE {self.cfg.execution_mode}] {ev.message}"):
                sent += 1
        return sent
