"""Global configuration for CHICKTRADE 1-year M5 backtest engine."""
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class StrategyConfig:
    ema_fast: int = 34
    ema_mid: int = 89
    ema_slow: int = 200
    atr_period: int = 14
    vol_ma_period: int = 20

    # Wave detection parameters (M5)
    min_impulse_atr: float = 0.8       # Price must move at least 0.8 * ATR away from EMA34 to register a swing
    touch_buffer_atr: float = 0.15     # Tolerance around EMA34 to count as a touch
    max_breach_atr: float = 0.60       # Close cannot breach too far past EMA34 on a valid trend bounce
    liquidity_void_atr: float = 3.0    # Skip candle if High-Low > 3.0 * ATR (abnormal spike / news)
    min_bars_between_touches: int = 3  # Minimum M5 bars before counting the next distinct wave touch

    # Price Action & Volume filter (used for Wave 2 & filter analysis)
    min_wick_ratio: float = 0.25       # Rejection wick >= 25% of candle range OR strong body in trend direction
    min_vol_ratio: float = 0.80        # Volume >= 0.8 * Volume MA(20)


@dataclass
class RiskConfig:
    initial_capital: float = 10_000.0
    compounding_steps: int = 20
    step_gain_pct: float = 0.08        # +8% per compounding step

    buffer_units: float = 400.0        # Pareto-optimal default "Bao 400 gia" (max 5 concurrent, Calmar 40.9)
    buffer_max_concurrent: Dict[int, int] = field(
        default_factory=lambda: {250: 2, 300: 3, 400: 5, 500: 7}
    )
    ema_risk_multipliers: Dict[str, float] = field(
        default_factory=lambda: {"EMA34": 1.00, "EMA89": 0.95, "EMA200": 0.90}
    )
    atr_to_unit_mult: float = 0.5      # 1 Unit = 0.5 * ATR(14) on M5/M15/M30/H1

    # Confidence-tiered sizing based on empirical winrate
    min_winrate_to_trade: float = 0.51     # Skip setups < 51%
    full_size_winrate: float = 0.70        # > 70% => Full Size (1.0x, bao 200 units)
    half_size_mult: float = 0.5            # 51% - 70% => Half Size (0.5x, chia doi bao 400 units/lenh)

    max_concurrent_trades: int = 2         # Max 2 open positions simultaneously across portfolio
    max_trades_per_day: int = 10           # Max 10 trades per UTC day

    # Trade management in Units
    sl_units: float = 5.0
    be_trigger_units: float = 3.0
    tp_units: float = 5.0
    enable_be: bool = True

    # Exchange fees & slippage (Binance USD-M Futures)
    maker_fee_rate: float = 0.0002         # 0.02% per side
    taker_fee_rate: float = 0.0005         # 0.05% per side
    use_maker_entry: bool = True           # Limit entry at EMA34 touch -> Maker fee; SL/TP -> configurable
    slippage_bps: float = 1.0              # 1 bps = 0.01% slippage on market/stop fills


@dataclass
class BacktestConfig:
    symbols: List[str] = field(default_factory=lambda: ["BTCUSDT", "ETHUSDT", "BNBUSDT"])
    reference_gold_symbol: str = "PAXGUSDT"
    timeframe: str = "5m"
    months: int = 12
    # Fixed reference date range: last 12 full months available on Binance Data Vision
    # Determined dynamically from latest available month if None
    end_year: int = 2026
    end_month: int = 10

    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
