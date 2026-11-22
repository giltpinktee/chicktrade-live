"""CHICKTRADE Live & Paper Trading Bot Runner (`--once` or 24/7 loop)."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bot.bot_config import BotConfig
from bot.execution.base import BaseBroker, TradeEvent
from bot.execution.binance_broker import BinanceFuturesBroker
from bot.execution.paper_broker import PaperBroker
from bot.market_feed import BinanceMarketFeed
from bot.notifier import BotNotifier
from bot.signal_engine import CoinScanSnapshot, HTF_MAP, SignalEngine, StormStatus
from bot.state_store import StateStore


class ChicktradeBot:
    """Production orchestrator connecting MarketFeed, SignalEngine, Broker, StateStore, and Notifier."""

    def __init__(
        self,
        cfg: BotConfig,
        store: Optional[StateStore] = None,
        feed: Optional[BinanceMarketFeed] = None,
        broker: Optional[BaseBroker] = None,
        notifier: Optional[BotNotifier] = None,
    ):
        self.cfg = cfg
        self.store = store or StateStore(
            cfg.db_path, initial_capital=cfg.initial_capital, step_gain_pct=cfg.step_gain_pct
        )
        self.feed = feed or BinanceMarketFeed()
        self.engine = SignalEngine(cfg, self.store)
        if broker is not None:
            self.broker = broker
        elif cfg.execution_mode == "PAPER":
            self.broker = PaperBroker(cfg, self.store)
        else:
            self.broker = BinanceFuturesBroker(cfg, self.store, market_feed=self.feed)
        self.notifier = notifier or BotNotifier(cfg)

    def run_once(self, print_report: bool = True) -> dict:
        """Execute a single full market scan, position sync, and signal execution cycle."""
        now_dt = datetime.now(timezone.utc)
        coins = self.engine.coins
        tfs = self.engine.tfs
        htfs = self.engine.required_htfs()

        # 1. Live prices for all tracked coins
        prices = self.feed.fetch_all_prices(coins)

        # 2. Fetch closed klines (300 bars) for all required timeframes + HTFs
        klines: Dict[tuple[str, str], object] = {}
        all_intervals = sorted(set(tfs) | set(htfs))
        for sym in coins:
            for iv in all_intervals:
                klines[(sym, iv)] = self.feed.fetch_closed_klines(sym, iv, limit=300)

        # 3. Sync open positions using latest closed 15m (or finest TF) bar highs/lows + live price
        finest_tf = tfs[0] if tfs else "15m"
        bar_hl: Dict[str, tuple[float, float]] = {}
        for sym in coins:
            df_finest = klines[(sym, finest_tf)]
            bar_hl[sym] = (float(df_finest["high"].iloc[-1]), float(df_finest["low"].iloc[-1]))

        events: List[TradeEvent] = []
        events.extend(self.broker.sync_positions(prices, bar_highs_lows=bar_hl))

        # 4. Evaluate BTC Macro Storm Guard (Layer 2 + 4h cooldown)
        btc_df = klines.get(("BTCUSDT", "15m"))
        if btc_df is None:
            btc_df = klines.get(("BTCUSDT", finest_tf))
        if btc_df is not None:
            btc_storm = self.engine.evaluate_storm(btc_df, scope="BTC_MACRO", now=now_dt)
        else:
            btc_storm = StormStatus(False, False, 1.0, 0.5, False, "NO_BTC_FEED")

        # 5. Scan every coin x timeframe for signals
        snapshots: List[CoinScanSnapshot] = []
        skipped_notes: List[str] = []

        for sym in coins:
            snap = CoinScanSnapshot(symbol=sym, last_price=prices.get(sym, 0.0))
            coin_storm_reasons: List[str] = []

            for tf in tfs:
                df = klines[(sym, tf)]
                htf = HTF_MAP.get(tf, "4h")
                df_htf = klines[(sym, htf)]

                local_storm = self.engine.evaluate_storm(df, scope=f"LOCAL:{sym}:{tf}", now=now_dt)
                if local_storm.in_storm:
                    snap.storm_active = True
                    coin_storm_reasons.append(f"{tf}:{local_storm.reason}")

                sigs, tf_meta = self.engine.detect_latest_signals(sym, tf, df, df_htf)
                tf_meta["atr_ratio"] = round(local_storm.atr_ratio, 2)
                tf_meta["er"] = round(local_storm.kaufman_er, 2)
                snap.tf_metrics[tf] = tf_meta

                for sig in sigs:
                    sig_key = f"{sig['symbol']}|{sig['tf']}|{sig['setup_code']}|{sig['bar_ts']}"
                    if self.store.is_signal_processed(sig_key):
                        continue

                    if btc_storm.in_storm:
                        skipped_notes.append(f"{sym} {tf} {sig['setup_code']} blocked by BTC_MACRO ({btc_storm.reason})")
                        self.store.mark_signal_processed(sig_key)
                        continue

                    if local_storm.in_storm:
                        skipped_notes.append(f"{sym} {tf} {sig['setup_code']} blocked by LOCAL_STORM ({local_storm.reason})")
                        self.store.mark_signal_processed(sig_key)
                        continue

                    rule = self.engine.rules[(sym, tf, sig["setup_code"])]
                    sizing = self.engine.evaluate_sizing_and_build_plan(
                        rule=rule,
                        direction=sig["direction"],
                        entry_price=prices.get(sym, sig["entry_price"]),
                        atr=sig["atr"],
                        signal_bar_ts=sig["bar_ts"],
                        elevated_vol=local_storm.elevated_vol,
                        now=now_dt,
                    )
                    self.store.mark_signal_processed(sig_key)

                    if not sizing.allowed or sizing.order_plan is None:
                        skipped_notes.append(f"{sym} {tf} {sig['setup_code']} skipped: {sizing.reason}")
                        continue

                    _, open_evt = self.broker.open_position(sizing.order_plan)
                    events.append(open_evt)

            snap.storm_reason = ", ".join(coin_storm_reasons) if coin_storm_reasons else "OK"
            snapshots.append(snap)

        # 6. Render status report & push Telegram events
        comp_info = self.store.get_compounding_info()
        daily = self.store.get_daily_counter()
        open_positions = self.store.get_open_positions()
        report = self.notifier.format_status_report(
            comp_info=comp_info,
            daily=daily,
            btc_storm=btc_storm,
            snapshots=snapshots,
            open_positions=open_positions,
            market_prices=prices,
            events=events,
            skipped_notes=skipped_notes,
        )
        if print_report:
            print(report, flush=True)
        if events:
            self.notifier.notify_events(events)

        return {
            "timestamp": now_dt.isoformat(),
            "mode": self.cfg.execution_mode,
            "equity": comp_info["equity"],
            "btc_storm": btc_storm,
            "snapshots": snapshots,
            "open_positions": open_positions,
            "events": events,
            "skipped": skipped_notes,
            "report": report,
        }

    def run_forever(self, interval_seconds: Optional[int] = None) -> None:
        sleep_sec = max(5, int(interval_seconds or self.cfg.poll_interval_seconds))
        print(f"Starting CHICKTRADE Bot in {self.cfg.execution_mode} mode (poll={sleep_sec}s)...", flush=True)
        try:
            while True:
                try:
                    self.run_once(print_report=True)
                except Exception as exc:
                    print(f"[ERROR] Scan cycle failed: {exc}", file=sys.stderr, flush=True)
                time.sleep(sleep_sec)
        except KeyboardInterrupt:
            print("\nCHICKTRADE Bot stopped cleanly by user.", flush=True)
        finally:
            self.store.close()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="CHICKTRADE Live & Paper Trading Bot")
    parser.add_argument("--once", action="store_true", help="Run a single live market scan and exit")
    parser.add_argument("--interval", type=int, default=None, help="Polling interval in seconds (24/7 mode)")
    parser.add_argument("--env", type=str, default=None, help="Path to .env configuration file")
    parser.add_argument("--db", type=str, default=None, help="Path to SQLite state database")
    args = parser.parse_args(argv)

    cfg = BotConfig.load(env_path=args.env, db_path=args.db)
    bot = ChicktradeBot(cfg)
    try:
        if args.once:
            bot.run_once(print_report=True)
        else:
            bot.run_forever(interval_seconds=args.interval)
    finally:
        if args.once:
            bot.store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
