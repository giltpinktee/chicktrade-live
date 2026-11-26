# Graph Report - CHICKTRADE  (2026-11-26)

## Corpus Check
- 47 files · ~289,049 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 4 file(s) not represented in the graph (top: (none) 2, .example 1, .csv 1)

## Summary
- 541 nodes · 1343 edges · 18 communities (14 shown, 4 thin omitted)
- Extraction: 94% EXTRACTED · 6% INFERRED · 0% AMBIGUOUS · INFERRED: 81 edges (avg confidence: 0.94)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- verified_tests Module (79)
- cloud_bot Module (69)
- bot Module (56)
- risk Module (52)
- bot Module (49)
- cloud_bot Module (49)
- coins Module (38)
- bot Module (27)
- bot Module (23)
- bot Module (22)
- bot Module (21)
- bot Module (18)
- core Module (17)
- tests Module (8)
- core Module (7)
- bot Module (2)
- cloud_bot Module (2)
- core Module (2)

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
- `TestAdaptiveBufferController` --uses--> `BotConfig`  [INFERRED]
  tests/test_bot.py → bot/bot_config.py
- `TestBotConfig` --uses--> `BotConfig`  [INFERRED]
  tests/test_bot.py → bot/bot_config.py
- `TestFourExitArchitectures` --uses--> `BotConfig`  [INFERRED]
  tests/test_bot.py → bot/bot_config.py
- `TestSqliteRestartStormAndKelly` --uses--> `BinanceFuturesBroker`  [INFERRED]
  tests/test_bot.py → bot/execution/binance_broker.py
- `TestAdaptiveBufferController` --uses--> `PaperBroker`  [INFERRED]
  tests/test_bot.py → bot/execution/paper_broker.py

## Import Cycles
- None detected.

## Communities (18 total, 4 thin omitted)

### Community 0 - "verified_tests Module (79)"
Cohesion: 0.06
Nodes (67): Build standalone Light-Mode HTML Report for CHICKTRADE v3.3 — XAUUSD 100x Quant…, collections, concurrent_futures, datetime, json, numpy, os, pandas (+59 more)

### Community 1 - "cloud_bot Module (69)"
Cohesion: 0.06
Nodes (67): base64, _apply_s2_step(), _brake_status_text(), build_chart_candles_payload(), build_quickchart_png(), build_recap_message(), _calc_leg_pnl(), _check_shadow_positions() (+59 more)

### Community 2 - "bot Module (56)"
Cohesion: 0.06
Nodes (33): argparse, HMAC-SHA256 signed Binance USD-M Futures REST adapter (TESTNET & LIVE).…, ChicktradeBot, main(), CHICKTRADE Live & Paper Trading Bot Runner (`--once` or 24/7 loop)., Production orchestrator connecting MarketFeed, SignalEngine, Broker,…, Execute a single full market scan, position sync, and signal execution cycle., BinanceMarketFeed (+25 more)

### Community 3 - "risk Module (52)"
Cohesion: 0.08
Nodes (36): BacktestConfig, Global configuration for CHICKTRADE 1-year M5 backtest engine., RiskConfig, StrategyConfig, dataclasses, DataFrame, Stage 1 (Empirical Discovery) and Stage 2 (Portfolio 20-step Compounding)…, Stage 1: Measure empirical winrate, net PnL in Units, and PF for every signal… (+28 more)

### Community 4 - "bot Module (49)"
Cohesion: 0.09
Nodes (20): ABC, BaseBroker, OrderPlan, Base execution interfaces and order/event models for CHICKTRADE., Abstract execution broker implemented by PaperBroker and BinanceFuturesBroker., Submit a new position according to `plan` and persist it., Evaluate open positions against latest market highs/lows/prices and process…, Force-close an open position immediately. (+12 more)

### Community 5 - "cloud_bot Module (49)"
Cohesion: 0.12
Nodes (48): answerCallback(), base64ToUint8Array(), broadcastToSubscribers(), buildAdminUsersView(), buildChartCoinKeyboard(), buildChartTfKeyboard(), buildInlineDashboardKeyboard(), buildOpenAppButton() (+40 more)

### Community 6 - "coins Module (38)"
Cohesion: 0.11
Nodes (35): calc_metrics(), compute_adx14(), compute_atr14(), compute_storm_cooldown(), _download_month_5m(), evaluate_signals(), extract_signals_for_tf(), forex_open_mask() (+27 more)

### Community 7 - "bot Module (27)"
Cohesion: 0.11
Nodes (20): compute_adx(), compute_htf_trend_series(), compute_storm_metrics(), DataFrame, datetime, ndarray, Series, CHICKTRADE Signal Engine: 3-EMA Wave Detection, HTF Trend, ADX, 3-Layer Storm… (+12 more)

### Community 8 - "bot Module (23)"
Cohesion: 0.15
Nodes (9): BotConfig, Loads CHICKTRADE strategy rules, scans closed klines, applies Storm Guard &…, SignalEngine, StrategyRule, BayesianKellySizer, Computes live-updated volume multipliers from 1Y prior + rolling live results., _make_sample_rule(), TestAutoTurboFloorGuard (+1 more)

### Community 9 - "bot Module (22)"
Cohesion: 0.14
Nodes (6): Any, Path, Crash-resilient SQLite state store for CHICKTRADE bot., Persist updated equity, peak equity, and advance compounding step if target…, Total worst-case SL risk (USD) across all currently open positions., StateStore

### Community 10 - "bot Module (21)"
Cohesion: 0.18
Nodes (9): BinanceFuturesBroker, format_decimal(), Any, Cancel original SL order and place updated Leg 2 SL order on Binance., Round a numeric value cleanly to Binance stepSize / tickSize using Decimal., Reconcile local open positions against Binance positionRisk & TP1 fills., Format a rounded float without scientific notation for Binance query params., Signed Binance USD-M Futures broker for TESTNET and LIVE modes. (+1 more)

### Community 11 - "bot Module (18)"
Cohesion: 0.15
Nodes (11): _parse_bool(), parse_env_file(), _parse_float(), _parse_int(), Path, Configuration loader for CHICKTRADE Live & Paper Trading Bot. Reads a `.env`…, Load configuration from `.env` file with OS environment overrides., Parse a standard KEY=VALUE .env file without external dependencies. (+3 more)

### Community 12 - "core Module (17)"
Cohesion: 0.18
Nodes (8): AdaptiveBufferController, Drawdown-targeted dynamic buffer controller with hard worst-case SL floor guard., Snap buffer to nearest `step_units` (10G) and clamp to `[min_buffer,…, Proportionally scale buffer up when |DD| > target_dd_pct, snapped to 10G and…, Map effective buffer units to maximum concurrent open positions., Verify that simultaneous SL sweep of all open + candidate trades stays >= 1.10x…, Compute `(effective_buffer, reason)` based on live drawdown, equity high, and…, TestAdaptiveBufferController

### Community 14 - "core Module (7)"
Cohesion: 0.33
Nodes (5): max_concurrent_for_buffer(), Dynamic Drawdown-Targeted Adaptive Buffer Controller for CHICKTRADE.…, Map effective buffer units `G` cleanly to maximum concurrent open positions., tempfile, Comprehensive unit tests for CHICKTRADE Live & Paper Trading Bot.

## Knowledge Gaps
- **6 isolated node(s):** `MEM_STORE`, `BacktestConfig`, `CHICKTRADE v5 6-Coin Report (53 Setups)`, `CHICKTRADE XAUUSD 100x Report`, `Free Cloud Setup Guide (Actions+Pages+CF Worker)` (+1 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 170 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **4 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `StateStore` connect `bot Module (22)` to `bot Module (56)`, `bot Module (49)`, `bot Module (27)`, `bot Module (23)`, `bot Module (21)`, `core Module (17)`, `tests Module (8)`, `core Module (7)`?**
  _High betweenness centrality (0.108) - this node is a cross-community bridge._
- **Why does `BinanceFuturesBroker` connect `bot Module (21)` to `bot Module (56)`, `bot Module (49)`, `bot Module (23)`, `bot Module (22)`, `core Module (7)`?**
  _High betweenness centrality (0.046) - this node is a cross-community bridge._
- **Why does `AdaptiveBufferController` connect `core Module (17)` to `bot Module (23)`, `core Module (7)`, `bot Module (27)`?**
  _High betweenness centrality (0.043) - this node is a cross-community bridge._
- **Are the 8 inferred relationships involving `StateStore` (e.g. with `BinanceFuturesBroker` and `PaperBroker`) actually correct?**
  _`StateStore` has 8 INFERRED edges - model-reasoned connections that need verification._
- **Are the 11 inferred relationships involving `BotConfig` (e.g. with `BinanceFuturesBroker` and `PaperBroker`) actually correct?**
  _`BotConfig` has 11 INFERRED edges - model-reasoned connections that need verification._
- **Are the 11 inferred relationships involving `PaperBroker` (e.g. with `BotConfig` and `OrderPlan`) actually correct?**
  _`PaperBroker` has 11 INFERRED edges - model-reasoned connections that need verification._
- **Are the 8 inferred relationships involving `BinanceFuturesBroker` (e.g. with `BotConfig` and `OrderPlan`) actually correct?**
  _`BinanceFuturesBroker` has 8 INFERRED edges - model-reasoned connections that need verification._