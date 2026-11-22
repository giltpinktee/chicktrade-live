# Graph Report - CHICKTRADE  (2026-11-22)

## Corpus Check
- 58 files · ~107,516 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 31 file(s) not represented in the graph (top: .csv 29, .example 1, (none) 1)

## Summary
- 508 nodes · 1580 edges · 17 communities (14 shown, 3 thin omitted)
- Extraction: 94% EXTRACTED · 6% INFERRED · 0% AMBIGUOUS · INFERRED: 91 edges (avg confidence: 0.94)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Module_0
- Module_1
- Module_2
- Module_3
- Module_4
- Module_5
- Module_6
- Module_7
- Module_8
- Module_9
- Module_10
- Module_11
- Module_12
- Module_13
- Module_14
- Module_15
- Module_16

## God Nodes (most connected - your core abstractions)
1. `StateStore` - 54 edges
2. `download_symbol_1y()` - 36 edges
3. `BotConfig` - 35 edges
4. `PaperBroker` - 34 edges
5. `BacktestConfig` - 34 edges
6. `CompoundingManager` - 33 edges
7. `_detect_latest_available_month()` - 33 edges
8. `generate_all_4_touches()` - 33 edges
9. `build_cooldown_mask()` - 31 edges
10. `resample_ohlcv()` - 31 edges

## Surprising Connections (you probably didn't know these)
- `SignalEngine` --uses--> `AdaptiveBufferController`  [INFERRED]
  bot/signal_engine.py → core/adaptive_buffer.py
- `StrategyRule` --uses--> `SetupPrior`  [INFERRED]
  bot/signal_engine.py → core/adaptive_sizing.py
- `SizingDecision` --uses--> `AdaptiveDecision`  [INFERRED]
  bot/signal_engine.py → core/adaptive_sizing.py
- `SignalEngine` --uses--> `BayesianKellySizer`  [INFERRED]
  bot/signal_engine.py → core/adaptive_sizing.py
- `compute_setup_payoff_stats()` --calls--> `sim_exit_arch()`  [INFERRED]
  core/build_v2_master.py → verified_tests/scripts/run_full_216_exit_sweetspot.py

## Import Cycles
- None detected.

## Communities (17 total, 3 thin omitted)

### Community 0 - "Module_0"
Cohesion: 0.05
Nodes (83): Global configuration for CHICKTRADE 1-year M5 backtest engine., RiskConfig, StrategyConfig, copy, Single-coin 2-way (LONG + SHORT) portfolio simulation strictly enforcing 1…, sim_single_coin_portfolio(), main(), Inject v1.4 Adaptive Volume (Bayesian-Kelly + EMA Risk) & Per-Setup Optimal… (+75 more)

### Community 1 - "Module_1"
Cohesion: 0.07
Nodes (46): ABC, Configuration loader for CHICKTRADE Live & Paper Trading Bot. Reads a `.env`…, BaseBroker, OrderPlan, Base execution interfaces and order/event models for CHICKTRADE., Abstract execution broker implemented by PaperBroker and BinanceFuturesBroker., Submit a new position according to `plan` and persist it., Evaluate open positions against latest market highs/lows/prices and process… (+38 more)

### Community 2 - "Module_2"
Cohesion: 0.06
Nodes (50): argparse, AdaptiveDecision, BayesianKellySizer, Bayesian + Fractional Kelly Adaptive Volume Engine for CHICKTRADE. Continuously…, Computes live-updated volume multipliers from 1Y prior + rolling live results., Compute updated posterior winrate, Kelly fraction, and optimal volume…, SetupPrior, main() (+42 more)

### Community 3 - "Module_3"
Cohesion: 0.11
Nodes (51): BacktestConfig, main(), _detect_latest_available_month(), download_symbol_1y(), _month_list(), _parse_klines_csv(), DataFrame, Probe data.binance.vision backwards from real current UTC time to find latest… (+43 more)

### Community 4 - "Module_4"
Cohesion: 0.09
Nodes (10): DailyCounter, Any, datetime, Path, Crash-resilient SQLite state store for CHICKTRADE bot., Persist updated equity, peak equity, and advance compounding step if target…, Total worst-case SL risk (USD) across all currently open positions., StateStore (+2 more)

### Community 5 - "Module_5"
Cohesion: 0.10
Nodes (14): BinanceFuturesBroker, format_decimal(), Any, Cancel original SL order and place updated Leg 2 SL order on Binance., Round a numeric value cleanly to Binance stepSize / tickSize using Decimal., Reconcile local open positions against Binance positionRisk & TP1 fills., Format a rounded float without scientific notation for Binance query params., Signed Binance USD-M Futures broker for TESTNET and LIVE modes. (+6 more)

### Community 6 - "Module_6"
Cohesion: 0.10
Nodes (16): DataFrame, sim_leg2_policy(), Timestamp, run_portfolio_with_storm_guard(), assign_volume_mult(), audit_all_288_combinations(), build_summary_table(), generate_master_html_report() (+8 more)

### Community 7 - "Module_7"
Cohesion: 0.21
Nodes (6): BotConfig, Loads CHICKTRADE strategy rules, scans closed klines, applies Storm Guard &…, SignalEngine, _make_sample_rule(), TestAutoTurboFloorGuard, TestSqliteRestartStormAndKelly

### Community 8 - "Module_8"
Cohesion: 0.18
Nodes (8): AdaptiveBufferController, Drawdown-targeted dynamic buffer controller with hard worst-case SL floor guard., Snap buffer to nearest `step_units` (10G) and clamp to `[min_buffer,…, Proportionally scale buffer up when |DD| > target_dd_pct, snapped to 10G and…, Map effective buffer units to maximum concurrent open positions., Verify that simultaneous SL sweep of all open + candidate trades stays >= 1.10x…, Compute `(effective_buffer, reason)` based on live drawdown, equity high, and…, TestAdaptiveBufferController

### Community 9 - "Module_9"
Cohesion: 0.18
Nodes (12): compute_adx(), compute_htf_trend_series(), compute_storm_metrics(), DataFrame, datetime, ndarray, Series, Wilder ADX(14) matching CHICKTRADE research modules. (+4 more)

### Community 10 - "Module_10"
Cohesion: 0.18
Nodes (9): _parse_bool(), parse_env_file(), _parse_float(), _parse_int(), Path, Load configuration from `.env` file with OS environment overrides., Parse a standard KEY=VALUE .env file without external dependencies., main() (+1 more)

### Community 11 - "Module_11"
Cohesion: 0.17
Nodes (10): apply_shadow_win_filter(), Apply 2-SL -> Shadow-Win filter sequentially (zero lookahead). Returns filtered…, audit_directional_asymmetry(), inspect_new_core_candidates(), DataFrame, Timestamp, Inspect LINKUSDT 15m EMA34_W2 and BNBUSDT 15m EMA89_W2 under all 7 exit…, Unified simulation engine based strictly on run_pause_resume_sweetspot.py… (+2 more)

### Community 13 - "Module_13"
Cohesion: 0.29
Nodes (6): max_concurrent_for_buffer(), Dynamic Drawdown-Targeted Adaptive Buffer Controller for CHICKTRADE.…, Map effective buffer units `G` cleanly to maximum concurrent open positions., tempfile, Comprehensive unit tests for CHICKTRADE Live & Paper Trading Bot., unittest

### Community 14 - "Module_14"
Cohesion: 0.40
Nodes (3): Compute Bayesian-Kelly adaptive volume, Auto-Turbo buffer, and hard 1.10x…, SizingDecision, StrategyRule

## Knowledge Gaps
- **3 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `StateStore` connect `Module_4` to `Module_1`, `Module_5`, `Module_7`, `Module_8`, `Module_12`, `Module_13`?**
  _High betweenness centrality (0.129) - this node is a cross-community bridge._
- **Why does `BinanceFuturesBroker` connect `Module_5` to `Module_1`, `Module_4`, `Module_13`, `Module_7`?**
  _High betweenness centrality (0.055) - this node is a cross-community bridge._
- **Why does `CompoundingManager` connect `Module_0` to `Module_2`, `Module_11`, `Module_6`?**
  _High betweenness centrality (0.052) - this node is a cross-community bridge._
- **Are the 8 inferred relationships involving `StateStore` (e.g. with `BinanceFuturesBroker` and `PaperBroker`) actually correct?**
  _`StateStore` has 8 INFERRED edges - model-reasoned connections that need verification._
- **Are the 11 inferred relationships involving `BotConfig` (e.g. with `BinanceFuturesBroker` and `PaperBroker`) actually correct?**
  _`BotConfig` has 11 INFERRED edges - model-reasoned connections that need verification._
- **Are the 11 inferred relationships involving `PaperBroker` (e.g. with `BotConfig` and `OrderPlan`) actually correct?**
  _`PaperBroker` has 11 INFERRED edges - model-reasoned connections that need verification._
- **Should `Module_0` be split into smaller, more focused modules?**
  _Cohesion score 0.051515151515151514 - nodes in this community are weakly interconnected._