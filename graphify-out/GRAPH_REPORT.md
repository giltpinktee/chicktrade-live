# Graph Report - CHICKTRADE  (2026-11-26)

## Corpus Check
- 49 files · ~233,291 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 4 file(s) not represented in the graph (top: (none) 2, .example 1, .csv 1)

## Summary
- 534 nodes · 1325 edges · 15 communities (14 shown, 1 thin omitted)
- Extraction: 94% EXTRACTED · 6% INFERRED · 0% AMBIGUOUS · INFERRED: 81 edges (avg confidence: 0.94)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Community 0
- Community 1
- Community 2
- Community 3
- Community 4
- Community 5
- Community 6
- Community 7
- Community 8
- Community 9
- Community 10
- Community 11
- Community 12
- Community 13
- Community 14

## God Nodes (most connected - your core abstractions)
1. `StateStore` - 54 edges
2. `BotConfig` - 35 edges
3. `PaperBroker` - 34 edges
4. `BinanceFuturesBroker` - 30 edges
5. `TradeEvent` - 28 edges
6. `OpenPosition` - 26 edges
7. `SignalEngine` - 25 edges
8. `BotNotifier` - 19 edges
9. `handleCallback()` - 18 edges
10. `ChicktradeBot` - 17 edges

## Surprising Connections (you probably didn't know these)
- `TestSqliteRestartStormAndKelly` --uses--> `BinanceFuturesBroker`  [INFERRED]
  tests/test_bot.py → bot/execution/binance_broker.py
- `TestAdaptiveBufferController` --uses--> `PaperBroker`  [INFERRED]
  tests/test_bot.py → bot/execution/paper_broker.py
- `TestAutoTurboFloorGuard` --uses--> `PaperBroker`  [INFERRED]
  tests/test_bot.py → bot/execution/paper_broker.py
- `TestFourExitArchitectures` --uses--> `PaperBroker`  [INFERRED]
  tests/test_bot.py → bot/execution/paper_broker.py
- `TestSqliteRestartStormAndKelly` --uses--> `PaperBroker`  [INFERRED]
  tests/test_bot.py → bot/execution/paper_broker.py

## Import Cycles
- None detected.

## Communities (15 total, 1 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.06
Nodes (67): base64, _apply_s2_step(), _brake_status_text(), build_chart_candles_payload(), build_quickchart_png(), build_recap_message(), _calc_leg_pnl(), _check_shadow_positions() (+59 more)

### Community 1 - "Community 1"
Cohesion: 0.05
Nodes (33): BotConfig, _parse_bool(), parse_env_file(), _parse_float(), _parse_int(), Path, Configuration loader for CHICKTRADE Live & Paper Trading Bot. Reads a `.env`…, Load configuration from `.env` file with OS environment overrides. (+25 more)

### Community 2 - "Community 2"
Cohesion: 0.06
Nodes (58): Build standalone Light-Mode HTML Report for CHICKTRADE v3.3 — XAUUSD 100x Quant…, get_effective_volume(), load_or_init_learning_state(), _new_setup_entry(), CHICKTRADE v5.1 — Slow Bayesian Online Volume Learner & Shadow Bench-Watch…, Load persisted live learning state, seeding active setups & initial BENCH_WATCH…, Return (vol_mult, lifecycle). Returns 0.0 when setup is in BENCH_WATCH., Record a hypothetical (non-executed) trade for a BENCH_WATCH setup and check… (+50 more)

### Community 3 - "Community 3"
Cohesion: 0.08
Nodes (27): ABC, BaseBroker, OrderPlan, Base execution interfaces and order/event models for CHICKTRADE., Abstract execution broker implemented by PaperBroker and BinanceFuturesBroker., Submit a new position according to `plan` and persist it., Evaluate open positions against latest market highs/lows/prices and process…, Force-close an open position immediately. (+19 more)

### Community 4 - "Community 4"
Cohesion: 0.07
Nodes (31): argparse, HMAC-SHA256 signed Binance USD-M Futures REST adapter (TESTNET & LIVE).…, ChicktradeBot, CHICKTRADE Live & Paper Trading Bot Runner (`--once` or 24/7 loop)., Production orchestrator connecting MarketFeed, SignalEngine, Broker,…, Execute a single full market scan, position sync, and signal execution cycle., BinanceMarketFeed, DataFrame (+23 more)

### Community 5 - "Community 5"
Cohesion: 0.12
Nodes (48): answerCallback(), base64ToUint8Array(), broadcastToSubscribers(), buildAdminUsersView(), buildChartCoinKeyboard(), buildChartTfKeyboard(), buildInlineDashboardKeyboard(), buildOpenAppButton() (+40 more)

### Community 6 - "Community 6"
Cohesion: 0.10
Nodes (38): calc_metrics(), compute_adx14(), compute_atr14(), compute_storm_cooldown(), _download_month_5m(), evaluate_signals(), extract_signals_for_tf(), forex_open_mask() (+30 more)

### Community 7 - "Community 7"
Cohesion: 0.09
Nodes (11): ClosedTrade, DailyCounter, Any, datetime, Path, Crash-resilient SQLite state store for CHICKTRADE bot., Persist updated equity, peak equity, and advance compounding step if target…, Total worst-case SL risk (USD) across all currently open positions. (+3 more)

### Community 8 - "Community 8"
Cohesion: 0.15
Nodes (14): datetime, CHICKTRADE Signal Engine: 3-EMA Wave Detection, HTF Trend, ADX, 3-Layer Storm…, Compute Bayesian-Kelly adaptive volume, Auto-Turbo buffer, and hard 1.10x…, SizingDecision, StrategyRule, Dynamic Drawdown-Targeted Adaptive Buffer Controller for CHICKTRADE.…, AdaptiveDecision, BayesianKellySizer (+6 more)

### Community 9 - "Community 9"
Cohesion: 0.20
Nodes (9): RiskConfig, CompoundingManager, 20-step (+8%/step) compounding ladder & ATR-normalized 200-unit position sizing., Tracks 20 compounding steps (+8%/step) and computes 200-unit buffer sizing., Advance step(s) whenever equity reaches current step's +8% target., Compute coin quantity using 200-unit safety buffer scaled by confidence tier., SizePlan, Unit tests for CHICKTRADE core logic: compounding, 200-unit sizing, SL/BE/TP… (+1 more)

### Community 10 - "Community 10"
Cohesion: 0.21
Nodes (12): BacktestConfig, Global configuration for CHICKTRADE 1-year M5 backtest engine., StrategyConfig, numpy, generate_signals(), DataFrame, EMA 34-89-200 Wave/Touch Counter & Signal Generator (Method 1 - M5)., Scan M5 OHLCV DataFrame sequentially (zero lookahead) and emit wave-touch… (+4 more)

### Community 11 - "Community 11"
Cohesion: 0.19
Nodes (11): compute_adx(), compute_htf_trend_series(), compute_storm_metrics(), DataFrame, ndarray, Series, Wilder ADX(14) matching CHICKTRADE research modules., Compute zero-lookahead volatility spike, sweep candle, and Kaufman ER(10) flags. (+3 more)

### Community 12 - "Community 12"
Cohesion: 0.26
Nodes (9): DataFrame, Stage 1: Measure empirical winrate, net PnL in Units, and PF for every signal…, Stage 2: Event-driven portfolio simulation with 20-step +8% compounding, max 2…, Simulate a single signal forward bar-by-bar with conservative SL-first…, run_discovery_stats(), run_portfolio_backtest(), simulate_single_trade(), Signal (+1 more)

### Community 13 - "Community 13"
Cohesion: 0.31
Nodes (7): Stage 1 (Empirical Discovery) and Stage 2 (Portfolio 20-step Compounding)…, TradeRecord, pandas, export_html_report(), DataFrame, Interactive HTML report generator (Plotly.js via CDN — zero extra Python deps)…, Write trades CSVs and a clean interactive HTML dashboard.

## Knowledge Gaps
- **2 isolated node(s):** `MEM_STORE`, `BacktestConfig`
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 165 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **1 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `StateStore` connect `Community 7` to `Community 8`, `Community 1`, `Community 3`, `Community 4`?**
  _High betweenness centrality (0.110) - this node is a cross-community bridge._
- **Why does `BinanceFuturesBroker` connect `Community 3` to `Community 1`, `Community 4`, `Community 7`?**
  _High betweenness centrality (0.047) - this node is a cross-community bridge._
- **Why does `AdaptiveBufferController` connect `Community 1` to `Community 8`?**
  _High betweenness centrality (0.044) - this node is a cross-community bridge._
- **Are the 8 inferred relationships involving `StateStore` (e.g. with `BinanceFuturesBroker` and `PaperBroker`) actually correct?**
  _`StateStore` has 8 INFERRED edges - model-reasoned connections that need verification._
- **Are the 11 inferred relationships involving `BotConfig` (e.g. with `BinanceFuturesBroker` and `PaperBroker`) actually correct?**
  _`BotConfig` has 11 INFERRED edges - model-reasoned connections that need verification._
- **Are the 11 inferred relationships involving `PaperBroker` (e.g. with `BotConfig` and `OrderPlan`) actually correct?**
  _`PaperBroker` has 11 INFERRED edges - model-reasoned connections that need verification._
- **Are the 8 inferred relationships involving `BinanceFuturesBroker` (e.g. with `BotConfig` and `OrderPlan`) actually correct?**
  _`BinanceFuturesBroker` has 8 INFERRED edges - model-reasoned connections that need verification._