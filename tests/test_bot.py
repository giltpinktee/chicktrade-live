"""Comprehensive unit tests for CHICKTRADE Live & Paper Trading Bot."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from bot.bot_config import BotConfig, resolve_max_concurrent
from bot.execution.base import OrderPlan
from bot.execution.binance_broker import BinanceFuturesBroker, format_decimal, round_to_step
from bot.execution.paper_broker import PaperBroker
from bot.notifier import BotNotifier
from bot.signal_engine import SignalEngine, StrategyRule
from bot.state_store import StateStore
from core.adaptive_buffer import AdaptiveBufferController, max_concurrent_for_buffer


def _make_sample_rule(exit_arch: str = "PARTIAL_MOVE_SL", leg2_sl_u: float = -1.0, sl_u: float = 6.0) -> StrategyRule:
    tp1_frac = 1.0 if exit_arch in ("ONLY_TP1", "ONLY_1R") else 0.5
    return StrategyRule(
        coin="BTCUSDT",
        tf="15m",
        ema="EMA34",
        setup_code="EMA34_W2",
        sl_u=sl_u,
        tp1_u=4.0,
        tp2_u=12.0,
        leg2_sl_u=leg2_sl_u,
        exit_arch=exit_arch,
        tp1_frac=tp1_frac,
        prior_trades=30,
        prior_wins=24,
        avg_win_u=3.8,
        avg_loss_u=6.25,
        sl_full_pct=20.0,
    )


def _make_plan(
    exit_arch: str,
    direction: int = 1,
    entry: float = 100.0,
    unit_size: float = 1.0,
    qty: float = 10.0,
    sl_u: float = 5.0,
    tp1_u: float = 4.0,
    tp2_u: float = 10.0,
    leg2_sl_u: float = -1.0,
) -> OrderPlan:
    d = direction
    sl_price = entry - d * sl_u * unit_size
    if exit_arch == "ONLY_1R":
        tp1_u_eff = sl_u
        tp1_frac = 1.0
    elif exit_arch == "ONLY_TP1":
        tp1_u_eff = tp1_u
        tp1_frac = 1.0
    else:
        tp1_u_eff = tp1_u
        tp1_frac = 0.5

    tp1_price = entry + d * tp1_u_eff * unit_size
    tp2_price = entry + d * tp2_u * unit_size
    leg2_sl_price = sl_price if exit_arch == "PARTIAL_NO_MOVE" else (entry + d * leg2_sl_u * unit_size)
    return OrderPlan(
        symbol="BTCUSDT",
        tf="15m",
        setup_code="EMA34_W2",
        ema="EMA34",
        direction=d,
        entry_price=entry,
        qty=qty,
        unit_size=unit_size,
        sl_u=sl_u,
        tp1_u=tp1_u_eff,
        tp2_u=tp2_u,
        leg2_sl_u=leg2_sl_u,
        exit_arch=exit_arch,
        tp1_frac=tp1_frac,
        sl_price=sl_price,
        tp1_price=tp1_price,
        tp2_price=tp2_price,
        leg2_sl_price=leg2_sl_price,
        sl_risk_usd=qty * sl_u * unit_size,
        vol_mult=1.0,
        buffer_used=400.0,
        signal_bar_ts="2026-01-01T00:00:00+00:00",
    )


class TestBotConfig(unittest.TestCase):
    def test_env_parser_and_buffer_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env"
            env_file.write_text(
                "\n".join(
                    [
                        "# Comment line",
                        "EXECUTION_MODE=PAPER",
                        "BUFFER_UNITS=300",
                        "AUTO_TURBO_ENABLED=true",
                        "INITIAL_CAPITAL=12000",
                        "MAX_TRADES_PER_DAY=8",
                    ]
                ),
                encoding="utf-8",
            )
            cfg = BotConfig.load(env_path=env_file, environ={})
            self.assertEqual(cfg.execution_mode, "PAPER")
            self.assertEqual(cfg.buffer_units, 300.0)
            self.assertEqual(cfg.max_concurrent_trades, 3)
            self.assertTrue(cfg.auto_turbo_enabled)
            self.assertEqual(cfg.initial_capital, 12000.0)
            self.assertEqual(cfg.max_trades_per_day, 8)

        self.assertEqual(resolve_max_concurrent(250), 2)
        self.assertEqual(resolve_max_concurrent(300), 3)
        self.assertEqual(resolve_max_concurrent(400), 5)
        self.assertEqual(resolve_max_concurrent(500), 7)
        self.assertEqual(resolve_max_concurrent(400, override=3), 3)

    def test_live_mode_requires_confirmation(self):
        with self.assertRaises(PermissionError):
            BotConfig.load(
                env_path=Path("nonexistent.env"),
                environ={"EXECUTION_MODE": "LIVE", "LIVE_CONFIRMED": "false"},
            )


class TestFourExitArchitectures(unittest.TestCase):
    def setUp(self):
        self.cfg = BotConfig(execution_mode="PAPER", initial_capital=10_000.0)
        self.store = StateStore(":memory:", initial_capital=10_000.0)
        self.broker = PaperBroker(self.cfg, self.store)

    def tearDown(self):
        self.store.close()

    def test_only_tp1_and_conservative_sl_first(self):
        # 1) Normal TP1 hit
        plan = _make_plan("ONLY_TP1", entry=100.0, unit_size=1.0, qty=10.0, sl_u=5.0, tp1_u=4.0)
        pos, _ = self.broker.open_position(plan)
        events = self.broker.evaluate_position_bar(pos, bar_high=104.5, bar_low=99.0)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].outcome, "TP1")
        self.assertGreater(events[0].pnl_usd, 35.0)
        self.assertEqual(len(self.store.get_open_positions()), 0)

        # 2) Same bar hits BOTH SL (95) and TP1 (104) -> MUST resolve as SL (conservative)
        pos2, _ = self.broker.open_position(plan)
        events2 = self.broker.evaluate_position_bar(pos2, bar_high=105.0, bar_low=94.0)
        self.assertEqual(len(events2), 1)
        self.assertEqual(events2[0].outcome, "SL")
        self.assertLess(events2[0].pnl_usd, -45.0)

    def test_only_1r_exit(self):
        # ONLY_1R targets sl_u (5.0U => price 105.0), ignoring tp1_u (4.0U)
        plan = _make_plan("ONLY_1R", entry=100.0, unit_size=1.0, qty=10.0, sl_u=5.0, tp1_u=4.0)
        self.assertAlmostEqual(plan.tp1_price, 105.0)
        pos, _ = self.broker.open_position(plan)

        # Bar reaches 104.2 -> should NOT close yet
        ev1 = self.broker.evaluate_position_bar(pos, bar_high=104.2, bar_low=99.5)
        self.assertEqual(len(ev1), 0)
        self.assertEqual(len(self.store.get_open_positions()), 1)

        # Bar reaches 105.1 -> closes 100% at TP_1R
        ev2 = self.broker.evaluate_position_bar(pos, bar_high=105.1, bar_low=101.0)
        self.assertEqual(len(ev2), 1)
        self.assertEqual(ev2[0].outcome, "TP_1R")
        self.assertGreater(ev2[0].pnl_usd, 45.0)

    def test_partial_no_move_keeps_original_sl(self):
        plan = _make_plan("PARTIAL_NO_MOVE", entry=100.0, unit_size=1.0, qty=10.0, sl_u=5.0, tp1_u=4.0, tp2_u=10.0)
        pos, _ = self.broker.open_position(plan)

        # Bar 1: TP1 (104) fills 50%
        ev1 = self.broker.evaluate_position_bar(pos, bar_high=104.5, bar_low=99.0)
        self.assertEqual(len(ev1), 1)
        self.assertEqual(ev1[0].event_type, "TP1_PARTIAL")

        reloaded = self.store.get_open_positions()[0]
        self.assertTrue(reloaded.tp1_filled)
        self.assertFalse(reloaded.leg2_sl_moved)
        self.assertAlmostEqual(reloaded.sl_price, 95.0)  # Original SL preserved!
        self.assertAlmostEqual(reloaded.remaining_qty, 5.0)

        # Bar 2: Pulls back to 97.0 (would stop out Entry-1U, but NOT PARTIAL_NO_MOVE at 95.0)
        ev2 = self.broker.evaluate_position_bar(reloaded, bar_high=102.0, bar_low=97.0)
        self.assertEqual(len(ev2), 0)

        # Bar 3: Hits TP2 (110.0)
        ev3 = self.broker.evaluate_position_bar(reloaded, bar_high=110.5, bar_low=101.0)
        self.assertEqual(len(ev3), 1)
        self.assertEqual(ev3[0].outcome, "TP2")
        self.assertEqual(len(self.store.get_open_positions()), 0)
        # Combined gross = 5*4 + 5*10 = $70 minus small maker fees
        self.assertGreater(ev3[0].pnl_usd, 68.0)

    def test_partial_move_sl_moves_leg2_sl_and_triggers_be_first(self):
        plan = _make_plan(
            "PARTIAL_MOVE_SL", entry=100.0, unit_size=1.0, qty=10.0, sl_u=5.0, tp1_u=4.0, tp2_u=10.0, leg2_sl_u=-1.0
        )
        pos, _ = self.broker.open_position(plan)

        # Bar 1: TP1 (104) fills 50% -> Leg 2 SL moves from 95.0 to 99.0 (Entry - 1U)
        ev1 = self.broker.evaluate_position_bar(pos, bar_high=104.2, bar_low=99.5)
        self.assertEqual(len(ev1), 1)
        self.assertEqual(ev1[0].event_type, "TP1_PARTIAL")

        reloaded = self.store.get_open_positions()[0]
        self.assertTrue(reloaded.tp1_filled)
        self.assertTrue(reloaded.leg2_sl_moved)
        self.assertAlmostEqual(reloaded.sl_price, 99.0)

        # Bar 2: Hits BOTH Leg2 SL (99.0) and TP2 (110.0) -> MUST resolve as BE (SL-first)
        ev2 = self.broker.evaluate_position_bar(reloaded, bar_high=111.0, bar_low=98.5)
        self.assertEqual(len(ev2), 1)
        self.assertEqual(ev2[0].outcome, "BE")
        self.assertEqual(len(self.store.get_open_positions()), 0)
        # Net PnL should still be positive: Leg1 (+20) + Leg2 (-5) - fees > +13
        self.assertGreater(ev2[0].pnl_usd, 13.0)


class TestAutoTurboFloorGuard(unittest.TestCase):
    def test_auto_turbo_engages_and_enforces_11000_hard_floor_fallback(self):
        cfg = BotConfig(
            execution_mode="PAPER",
            initial_capital=10_000.0,
            buffer_units=400.0,
            auto_turbo_enabled=True,
            auto_turbo_buffer_units=230.0,
            auto_turbo_trigger_ratio=1.25,
            auto_turbo_disengage_ratio=1.15,
            auto_turbo_floor_ratio=1.10,
        )
        store = StateStore(":memory:", initial_capital=10_000.0)
        engine = SignalEngine(cfg, store)
        broker = PaperBroker(cfg, store)
        rule = _make_sample_rule("PARTIAL_MOVE_SL", sl_u=6.0)

        # 1) Equity $13,000 (>= 1.25x = $12,500), 0 open risk -> Turbo 230G engages safely
        store.update_equity(13_000.0)
        d1 = engine.evaluate_sizing_and_build_plan(
            rule=rule, direction=1, entry_price=100.0, atr=2.0, signal_bar_ts="2026-01-01T00:00:00+00:00"
        )
        self.assertTrue(d1.allowed)
        self.assertTrue(d1.turbo_used)
        self.assertEqual(d1.active_buffer, 230.0)

        # 2) Simulate existing open SL risk of $1,800 at equity $13,000:
        #    Remaining cushion above $11,000 floor is only $200, while a 230G trade risks > $300 ->
        #    Current_Equity - (1800 + new_turbo_sl) < $11,000 => MUST fallback to 400G!
        heavy_plan = _make_plan("PARTIAL_MOVE_SL", entry=100.0, unit_size=1.0, qty=360.0, sl_u=5.0)
        heavy_plan.symbol = "ETHUSDT"
        heavy_plan.vol_mult = 0.5
        heavy_plan.sl_risk_usd = 1800.0
        broker.open_position(heavy_plan)

        d2 = engine.evaluate_sizing_and_build_plan(
            rule=rule, direction=1, entry_price=100.0, atr=2.0, signal_bar_ts="2026-01-01T01:00:00+00:00"
        )
        self.assertTrue(d2.allowed)
        self.assertFalse(d2.turbo_used)
        self.assertEqual(d2.active_buffer, 400.0)
        self.assertIn("FLOOR_GUARD", d2.turbo_fallback_reason)

        # 3) Equity drops below $11,500 (< 1.15x) -> Turbo disengages completely
        store.delete_open_position(store.get_open_positions()[0].position_id)
        store.update_equity(11_400.0)
        d3 = engine.evaluate_sizing_and_build_plan(
            rule=rule, direction=1, entry_price=100.0, atr=2.0, signal_bar_ts="2026-01-01T02:00:00+00:00"
        )
        self.assertTrue(d3.allowed)
        self.assertFalse(d3.turbo_used)
        self.assertEqual(d3.active_buffer, 400.0)
        self.assertFalse(store.get_state("turbo_active"))
        store.close()


class TestSqliteRestartStormAndKelly(unittest.TestCase):
    def test_sqlite_restart_recovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "test_state.db"
            cfg = BotConfig(execution_mode="PAPER", initial_capital=10_000.0, db_path=db_path)

            # Session 1: open position, fill TP1, advance equity, set storm cooldown
            store1 = StateStore(db_path, initial_capital=10_000.0)
            broker1 = PaperBroker(cfg, store1)
            plan = _make_plan("PARTIAL_MOVE_SL", entry=100.0, unit_size=1.0, qty=10.0, sl_u=5.0, tp1_u=4.0, leg2_sl_u=-1.0)
            pos, _ = broker1.open_position(plan)
            broker1.evaluate_position_bar(pos, bar_high=104.5, bar_low=99.5)
            store1.update_equity(10_900.0)
            store1.record_setup_outcome("BTCUSDT", "15m", "EMA34_W2", 3.5)
            cd_until = datetime.now(timezone.utc) + timedelta(hours=4)
            store1.set_storm_cooldown("BTC_MACRO", cd_until, "WHIPSAW")
            store1.close()

            # Session 2: fresh StateStore instance simulating process restart
            store2 = StateStore(db_path, initial_capital=10_000.0)
            positions = store2.get_open_positions()
            self.assertEqual(len(positions), 1)
            restored = positions[0]
            self.assertEqual(restored.exit_arch, "PARTIAL_MOVE_SL")
            self.assertTrue(restored.tp1_filled)
            self.assertTrue(restored.leg2_sl_moved)
            self.assertAlmostEqual(restored.sl_price, 99.0)
            self.assertAlmostEqual(restored.remaining_qty, 5.0)

            comp = store2.get_compounding_info()
            self.assertEqual(comp["step"], 2)
            self.assertAlmostEqual(comp["equity"], 10_900.0)
            self.assertEqual(store2.get_setup_outcomes("BTCUSDT", "15m", "EMA34_W2"), [3.5])
            in_cd, reason = store2.is_in_storm_cooldown("BTC_MACRO")
            self.assertTrue(in_cd)
            self.assertEqual(reason, "WHIPSAW")
            store2.close()

    def test_storm_guard_4h_cooldown(self):
        cfg = BotConfig(execution_mode="PAPER", initial_capital=10_000.0)
        store = StateStore(":memory:", initial_capital=10_000.0)
        engine = SignalEngine(cfg, store)

        # Build 70 calm bars, then 1 massive ATR spike bar
        ts = pd.date_range("2026-01-01", periods=71, freq="15min", tz="UTC")
        closes = np.linspace(100.0, 102.0, 71)
        highs = closes + 0.3
        lows = closes - 0.3
        # Spike on final bar
        highs[-1] = closes[-1] + 5.0
        lows[-1] = closes[-1] - 5.0
        df_spike = pd.DataFrame(
            {"timestamp": ts, "open": closes, "high": highs, "low": lows, "close": closes, "volume": np.full(71, 100.0)}
        )

        t0 = ts[-1].to_pydatetime()
        st1 = engine.evaluate_storm(df_spike, scope="BTC_MACRO", now=t0)
        self.assertTrue(st1.in_storm)

        # Calm candle 1 hour later -> still blocked by persisted 4h cooldown
        df_calm = df_spike.iloc[:-1].copy()
        st2 = engine.evaluate_storm(df_calm, scope="BTC_MACRO", now=t0 + timedelta(hours=1))
        self.assertTrue(st2.in_storm)
        self.assertIn("COOLDOWN", st2.reason)

        # 5 hours later -> cooldown expired, storm cleared
        st3 = engine.evaluate_storm(df_calm, scope="BTC_MACRO", now=t0 + timedelta(hours=5))
        self.assertFalse(st3.in_storm)
        store.close()

    def test_bayesian_kelly_live_update_throttles_and_pauses_cold_setup(self):
        cfg = BotConfig(execution_mode="PAPER", initial_capital=10_000.0)
        store = StateStore(":memory:", initial_capital=10_000.0)
        engine = SignalEngine(cfg, store)
        rule = _make_sample_rule("PARTIAL_MOVE_SL", sl_u=6.0)

        # Baseline (no live trades) -> active with high multiplier
        d_init = engine.evaluate_sizing_and_build_plan(
            rule=rule, direction=1, entry_price=100.0, atr=2.0, signal_bar_ts="2026-01-01T00:00:00+00:00"
        )
        self.assertTrue(d_init.allowed)
        init_mult = d_init.adaptive.vol_mult
        self.assertGreaterEqual(init_mult, 1.0)

        # Inject 4 consecutive live SL losses -> multiplier drops sharply
        for _ in range(4):
            store.record_setup_outcome(rule.coin, rule.tf, rule.setup_code, -6.25)
        d_cold = engine.evaluate_sizing_and_build_plan(
            rule=rule, direction=1, entry_price=100.0, atr=2.0, signal_bar_ts="2026-01-01T01:00:00+00:00"
        )
        self.assertLess(d_cold.adaptive.vol_mult, init_mult)

        # Inject 10 more live SL losses -> Bayesian-Kelly pauses setup automatically
        for _ in range(10):
            store.record_setup_outcome(rule.coin, rule.tf, rule.setup_code, -6.25)
        d_pause = engine.evaluate_sizing_and_build_plan(
            rule=rule, direction=1, entry_price=100.0, atr=2.0, signal_bar_ts="2026-01-01T02:00:00+00:00"
        )
        self.assertFalse(d_pause.allowed)
        self.assertIn("BAYESIAN_KELLY_PAUSE", d_pause.reason)
        self.assertEqual(d_pause.adaptive.tier_label, "PAUSE")
        store.close()

    def test_binance_broker_precision_and_safety_and_notifier_noop(self):
        self.assertAlmostEqual(round_to_step(1.2349, 0.001, floor=True), 1.234)
        self.assertEqual(format_decimal(1.2349, 0.001, floor=True), "1.234")
        self.assertEqual(format_decimal(65432.16, 0.1, floor=False), "65432.2")

        store = StateStore(":memory:", initial_capital=10_000.0)
        cfg_test = BotConfig(
            execution_mode="TESTNET",
            binance_api_key="test_key",
            binance_api_secret="test_secret",
        )
        broker = BinanceFuturesBroker(cfg_test, store)
        self.assertEqual(broker.base_url, "https://testnet.binancefuture.com")
        signed = broker._sign_params({"symbol": "BTCUSDT", "side": "BUY"})
        self.assertIn("signature=", signed)

        notifier = BotNotifier(BotConfig(execution_mode="PAPER", telegram_bot_token="", telegram_chat_id=""))
        self.assertFalse(notifier.telegram_enabled)
        self.assertFalse(notifier.send_telegram("hello"))
        store.close()


class TestAdaptiveBufferController(unittest.TestCase):
    def test_dd_proportional_scaling_snapping_and_cap(self):
        ctrl = AdaptiveBufferController()
        # Direct proportional scale example: 230 * 20 / 12 = 383.33 -> 380G
        self.assertEqual(ctrl.scale_buffer_for_dd(230.0, -20.0), 380.0)
        # Capped at max_buffer = 500G
        self.assertEqual(ctrl.scale_buffer_for_dd(400.0, -16.0), 500.0)

        # Moderate DD above target (12% < 14% < 15%) from 230G -> 230 * 14/12 = 268.3 -> 270G
        buf, reason = ctrl.compute_effective_buffer(
            current_buffer=230.0,
            current_dd_pct=-14.0,
            equity=12_000.0,
            peak_equity=13_953.0,
            initial_capital=10_000.0,
            turbo_enabled=True,
        )
        self.assertEqual(buf, 270.0)
        self.assertIn("DD_ADAPTIVE_SCALE", reason)

    def test_hard_dd_ceiling_and_severe_dd_halt(self):
        ctrl = AdaptiveBufferController()
        # At DD = -15% (>= hard_dd_ceiling_pct), immediately jump to at least 400G
        buf15, reason15 = ctrl.compute_effective_buffer(
            current_buffer=230.0,
            current_dd_pct=-15.0,
            equity=11_050.0,
            peak_equity=13_000.0,
            initial_capital=10_000.0,
            turbo_enabled=True,
        )
        self.assertEqual(buf15, 400.0)
        self.assertIn("DD_HARD_CEILING", reason15)

        # At DD = -18% (or -20%), immediately jump to 500G to halt bleed
        buf18, reason18 = ctrl.compute_effective_buffer(
            current_buffer=230.0,
            current_dd_pct=-18.0,
            equity=10_660.0,
            peak_equity=13_000.0,
            initial_capital=10_000.0,
            turbo_enabled=True,
        )
        self.assertEqual(buf18, 500.0)
        self.assertIn("DD_SEVERE_CEILING", reason18)

    def test_new_high_turbo_step_down_and_11k_floor_guard(self):
        ctrl = AdaptiveBufferController(turbo_buffer=230.0)
        # Smooth step-down toward 230G on new equity high (>= 1.25x = $12,500)
        buf_step, reason_step = ctrl.compute_effective_buffer(
            current_buffer=400.0,
            current_dd_pct=0.0,
            equity=13_000.0,
            peak_equity=13_000.0,
            initial_capital=10_000.0,
            turbo_enabled=True,
            existing_sl_risk_usd=200.0,
            candidate_sl_risk_usd=300.0,
            step_down_units=50.0,
        )
        self.assertEqual(buf_step, 350.0)
        self.assertIn("TURBO_ENGAGED", reason_step)

        # Worst-case SL sweep breaches $11,000 floor (13000 - (1800 + 300) = 10900 < 11000) -> fallback 400G
        buf_guard, reason_guard = ctrl.compute_effective_buffer(
            current_buffer=230.0,
            current_dd_pct=0.0,
            equity=13_000.0,
            peak_equity=13_000.0,
            initial_capital=10_000.0,
            turbo_enabled=True,
            existing_sl_risk_usd=1800.0,
            candidate_sl_risk_usd=300.0,
        )
        self.assertEqual(buf_guard, 400.0)
        self.assertIn("FLOOR_GUARD", reason_guard)

    def test_max_concurrent_mapping(self):
        ctrl = AdaptiveBufferController()
        for buf, expected in (
            (230.0, 2),
            (250.0, 2),
            (280.0, 3),
            (300.0, 3),
            (380.0, 5),
            (400.0, 5),
            (450.0, 7),
            (500.0, 7),
        ):
            self.assertEqual(ctrl.max_concurrent_for_buffer(buf), expected)
            self.assertEqual(max_concurrent_for_buffer(buf), expected)

    def test_signal_engine_adaptive_dd_scaling_and_one_direction_per_coin(self):
        cfg = BotConfig(
            execution_mode="PAPER",
            initial_capital=10_000.0,
            buffer_units=400.0,
            auto_turbo_enabled=True,
            auto_turbo_buffer_units=230.0,
        )
        store = StateStore(":memory:", initial_capital=10_000.0)
        engine = SignalEngine(cfg, store)
        broker = PaperBroker(cfg, store)
        rule_long = _make_sample_rule("PARTIAL_MOVE_SL", sl_u=6.0)

        # Push equity to peak $10,000 then drop -16% to $8,400 -> buffer scales to 500G
        store.update_equity(8_400.0)
        d_dd = engine.evaluate_sizing_and_build_plan(
            rule=rule_long, direction=1, entry_price=100.0, atr=2.0, signal_bar_ts="2026-01-01T00:00:00+00:00"
        )
        self.assertTrue(d_dd.allowed)
        self.assertEqual(d_dd.active_buffer, 500.0)
        self.assertIn("DD_HARD_CEILING", d_dd.buffer_reason)

        # Open BTCUSDT position -> any second trade on BTCUSDT (even different TF/setup/direction) MUST be blocked
        broker.open_position(d_dd.order_plan)
        rule_other_tf = StrategyRule(
            coin="BTCUSDT",
            tf="1h",
            ema="EMA89",
            setup_code="EMA89_W1",
            sl_u=6.0,
            tp1_u=4.0,
            tp2_u=12.0,
            leg2_sl_u=-1.0,
        )
        d_dup = engine.evaluate_sizing_and_build_plan(
            rule=rule_other_tf, direction=-1, entry_price=100.0, atr=2.0, signal_bar_ts="2026-01-01T01:00:00+00:00"
        )
        self.assertFalse(d_dup.allowed)
        self.assertIn("DUPLICATE_SYMBOL_OPEN", d_dup.reason)
        store.close()


if __name__ == "__main__":
    unittest.main()
