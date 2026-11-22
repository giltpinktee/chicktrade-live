"""Unit tests for CHICKTRADE core logic: compounding, 200-unit sizing, SL/BE/TP precedence."""
import unittest
import pandas as pd

from config import RiskConfig, StrategyConfig
from engine.portfolio import simulate_single_trade
from risk.compounding import CompoundingManager
from strategy.ema_wave import Signal


class TestChicktradeCore(unittest.TestCase):
    def test_compounding_20_steps_and_sizing(self):
        cfg = RiskConfig(initial_capital=10_000.0, buffer_units=250.0, atr_to_unit_mult=0.5)
        cm = CompoundingManager(cfg)
        self.assertEqual(cm.step, 1)
        self.assertAlmostEqual(cm.target_capital, 10_800.0, places=2)

        # Winrate < 51% => SKIP
        p_skip = cm.compute_size(entry_price=100_000.0, atr=300.0, setup_winrate=0.50, current_equity=10_000.0)
        self.assertEqual(p_skip.tier_label, "SKIP")
        self.assertEqual(p_skip.qty, 0.0)

        # Winrate 60% (51-70%) => HALF size (0.5x => 500-unit buffer per trade)
        p_half = cm.compute_size(entry_price=100_000.0, atr=300.0, setup_winrate=0.60, current_equity=10_000.0)
        self.assertEqual(p_half.tier_label, "HALF")
        self.assertAlmostEqual(p_half.qty, 10_000.0 / (500.0 * 150.0), places=6)

        # Winrate 75% (> 70%) => FULL size (1.0x => 250-unit buffer)
        p_full = cm.compute_size(entry_price=100_000.0, atr=300.0, setup_winrate=0.75, current_equity=10_000.0)
        self.assertEqual(p_full.tier_label, "FULL")
        self.assertAlmostEqual(p_full.qty, 10_000.0 / (250.0 * 150.0), places=6)
        self.assertAlmostEqual(p_full.qty, p_half.qty * 2.0, places=6)

        # Step up at +8% ($10,800)
        stepped = cm.update_equity(10_800.0)
        self.assertTrue(stepped)
        self.assertEqual(cm.step, 2)
        self.assertAlmostEqual(cm.tier_capital, 10_800.0, places=2)
        self.assertAlmostEqual(cm.target_capital, 10_000.0 * (1.08 ** 2), places=2)

    def test_conservative_sl_first_and_be_move(self):
        # Construct 4 bars: entry at bar 0 (price 100, ATR=2 => unit_size=1)
        # Bar 1 hits both SL (95) and TP (105) -> MUST resolve as SL (conservative)
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2025-01-01", periods=4, freq="5min", tz="UTC"),
                "open": [100.0, 100.0, 100.0, 100.0],
                "high": [101.0, 106.0, 106.0, 106.0],
                "low": [99.0, 94.0, 99.0, 99.0],
                "close": [100.0, 102.0, 102.0, 102.0],
                "volume": [100.0, 100.0, 100.0, 100.0],
            }
        )
        sig = Signal(
            bar_idx=0,
            timestamp=df["timestamp"].iloc[0],
            symbol="BTCUSDT",
            direction=1,
            wave=1,
            setup_type="W1_TREND",
            entry_price=100.0,
            atr=2.0,
            pa_vol_ok=True,
            ema_aligned=True,
        )
        reason, _, _, _, _, _ = simulate_single_trade(
            df, sig, sl_units=5.0, tp_units=5.0, be_trigger_units=3.0,
            enable_be=True, atr_to_unit_mult=0.5,
            entry_fee_rate=0.0002, exit_fee_rate_tp=0.0002, exit_fee_rate_sl=0.0005, slippage_bps=0.0,
        )
        self.assertEqual(reason, "SL")

        # Now test BE activation on bar 1 (+3.5 units) then reversal to entry on bar 2 -> BE
        df2 = pd.DataFrame(
            {
                "timestamp": pd.date_range("2025-01-01", periods=4, freq="5min", tz="UTC"),
                "open": [100.0, 100.0, 103.0, 100.0],
                "high": [101.0, 103.5, 103.5, 101.0],
                "low": [99.0, 99.0, 98.0, 99.0],
                "close": [100.0, 103.0, 99.0, 100.0],
                "volume": [100.0, 100.0, 100.0, 100.0],
            }
        )
        reason2, _, _, _, _, _ = simulate_single_trade(
            df2, sig, sl_units=5.0, tp_units=5.0, be_trigger_units=3.0,
            enable_be=True, atr_to_unit_mult=0.5,
            entry_fee_rate=0.0002, exit_fee_rate_tp=0.0002, exit_fee_rate_sl=0.0005, slippage_bps=0.0,
        )
        self.assertEqual(reason2, "BE")


if __name__ == "__main__":
    unittest.main()
