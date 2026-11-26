# AGENTS.md — CHICKTRADE v6.1 Complete AI Context & Quantitative Specification

> **Purpose:** Read this file first if you are an AI coding agent (Antigravity, Claude Code, Codex, Cursor, Windsurf) or quant engineer cloning `giltpinktee/chicktrade-live`. It contains every mathematical formula, execution rule, historical decision, and directory boundary needed to understand and extend **CHICKTRADE v6.1** with zero guesswork.

---

## 1. Executive Summary (CHICKTRADE v6.1)
- **System:** Systematic Binance USD-M Futures EMA-pullback & wave-exhaustion strategy across **7 Coins** (`BTCUSDT`, `ETHUSDT`, `BNBUSDT`, `LINKUSDT`, `XRPUSDT`, `SOLUSDT`, `NEARUSDT`), **68 ACTIVE setups + 1 BENCH_WATCH setup** (`NEARUSDT 1h EMA34_W4_REV`), `15m / 30m / 1h` only (`5m` permanently excluded), with **3-Tier Storm Guard T1 (`4h / 12h / 24h`)**.
- **Verified 36-Month Performance (`2023-11-01 → 2026-10-31`, 2021–2023 warmup, `$10,000` initial capital, Argon-Audited Binance-Exact Engine):**
  - **Bao 400G (Safe · S2 Step-Locked):** `$10,000 → $233,697.36` (`+2,237.0%`) · MaxDD `-11.26%` · Calmar `198.6` · PF `1.63` · PWR `61.9%` · `1,062` trades (`S5 Pessimistic`: `$184,511`, `+1,745%`, DD `-11.65%`, PF `1.56`).
  - **Auto-Turbo 250G (ON ≥ +25%, 110% Armor → 400G):** `$10,000 → $551,868.99` (`+5,418.7%`) · MaxDD `-15.32%` · Calmar `353.7` · PF `1.54` (`S5 Pessimistic`: `$442,558`, `+4,326%`, DD `-16.67%`, PF `1.47`).
  - **Optional Mid-Risk Turbo Frontiers (same 68 setups):** `Turbo 350G` = `$316,652` (DD `-13.52%`, PF `1.61`) · `Turbo 320G` = `$375,967` (DD `-13.63%`, Calmar `268.5`, PF `1.59`).

---

## 2. Exact Mathematical Formulas (How Every Number Is Computed)

### 2.1 Dynamic Unit Size (`1U` — `U3_HTF_MACRO_ATR`)
All SL, TP1, TP2, and position sizes are normalized to **1 Unit (`1U`)** derived from **4H Macro ATR** (zero lookahead — strictly uses the last *closed* 4H bar):
$$\text{Macro4H} = \text{EMA}_{20}\!\left(\frac{0.5 \times \text{ATR}_{14}(\text{4H})}{\text{Close}_{\text{4H}}}\right)$$
$$u_{\text{frac}}(\text{TF}) = \max\!\big(0.0008,\; \text{Macro4H} \times \gamma_{\text{TF}}\big), \qquad \gamma = \{\text{15m}: 0.29,\; \text{30m}: 0.41,\; \text{1h}: 0.59\}$$
$$u_{\text{px}} = \text{EntryPrice} \times u_{\text{frac}}(\text{TF})$$

### 2.2 Signal Generation (EMA 34 / 89 / 200 × Touches W1–W3 & W4_REV)
- **HTF Trend Filter:** `1h` trend for `15m` signals; `4h` trend for `30m/1h` signals.
- **Wave Counting:** Continuous converged EMA 34/89/200 touches (`W1, W2, W3` with-trend; `W4_REV` counter-trend exhaustion).
- **3-Tier Storm Guard Cooldown (T1, applied on BOTH BTC & local symbol):**
  - **Light Storm (`4h`):** `ATR_ratio >= 1.8` or High-Vol Chop (`ER_10 < 0.15` & `ATR_ratio >= 1.20`).
  - **Medium Storm (`12h`):** `ATR_ratio >= 2.2` or Liquidity Sweep (`range >= 2.8 * ATR` or (`range >= 2.0 * ATR` & `wick >= 70%`)).
  - **Severe Storm (`24h`):** Flash Crash (`ATR_ratio >= 2.8` or `range >= 4.0 * ATR`).
- **Elevated Volatility Guard:** If local `ATR_ratio >= 1.35`, effective volume multiplier is halved (`× 0.5`).

### 2.3 Argon-Audited Binance-Exact Trade Execution (`build_v6_master.py`)
1. **Exact Timestamps:** Signal fires on close of bar `i` → `open_ts = ts[i+1]` (next bar open), `close_ts = ts[exit_idx] + TF_MS[tf]` (exit bar *close*), `tp1_ts = ts[j] + TF_MS[tf]` (TP1 bar *close*). Slots & margin never free early.
2. **True Binance LIMIT IOC (`±0.25U`):** LONG misses immediately if `opens[i+1] > sig_px + 0.25 * u_px`; SHORT misses if `opens[i+1] < sig_px - 0.25 * u_px`. No GTC intraday dip filling.
3. **Conservative Intrabar Ordering & Gap-SL:** SL is always evaluated strictly before TP1/TP2. Gap-through SL fills at `min(opens[j], active_sl)` (LONG) or `max(opens[j], active_sl)` (SHORT).
4. **No Same-Bar TP1 + TP2:** When TP1 hits on bar `j` (closing 50%), Leg 2 evaluation starts strictly at bar `j + 1`.
5. **Real Fees, Slippage & 8h Funding:**
   - Entry: Taker `0.05%` + slippage (`1 bp` majors `BTC/ETH/BNB/LINK/XRP/SOL`, `3 bps` `NEAR`) in S4; `+0.10U` extra slip on entry & SL in S5.
   - TP1 / TP2 Exits: Maker `0.02%`. SL / Timeout Exits: Taker `0.05%` + slippage.
   - Real 8h Binance Funding integrated at exact historical `calc_time_ms` in `(open_ts, close_ts]`, weighted `1.0` before `tp1_ts` and `0.5` strictly after `tp1_ts` (`pnl_u = leg1 + leg2 - fund_cost_u`).

### 2.4 Portfolio Concurrency, Margin & Causal 2-SL Brake
- **Hard Caps:** Max **5 concurrent open positions**, max **1 open position per symbol** (Binance One-Way Mode), max **3 same-direction** (`LONG` or `SHORT`).
- **20x Per-Symbol Notional Caps:** `BTC/ETH: $300k`, `BNB/SOL/XRP/LINK: $150k`, `NEAR: $100k`, and single-position notional `<= 5 × eq`.
- **Liquidation-Proof Risk & Margin Guards:**
  - Concurrent worst-case SL cap: `sum(open worst_sl_usd) + new_worst_sl_usd <= 12% eq` (realized peak in 36M: `7.24%`).
  - Available margin check: `avail_eq = eq - sum(open worst_sl_usd)`; `(used_margin + new_notional/20) <= 0.75 * avail_eq`.
- **100% Causal Shadow-Pause (2-SL Brake):** Two consecutive original SLs (on setups with `base_vol >= 0.75×`) trigger a 72h shadow pause. A winning shadow trade may only lift the pause at its actual **`close_ts`** (zero lookahead).

### 2.5 Setup Selection & V3 Barbell Alpha Volume (`0.25× → 1.75×`)
Forensic correlation across all 68 setups proved:
- **Strongest predictors of `Recovery Factor = Net_U / MaxDD_U`:** `EPR = avg_u / sl_u` (`+0.735`), `PWR` (`+0.649`), `Net_U` (`+0.613`).
- **Strongest drivers of `MaxDD_U`:** `Max Consec SL` (`+0.672`), Trade Count `N` (`+0.554`), `SL_U` (`+0.421`).

**V3 Barbell 6-Tier Formula (`core/live_learning.py:v3_barbell_tier`):**
| Tier | Multiplier | Hard Criteria |
|---|---:|---|
| `S++` | `1.75×` | `PWR ≥ 69.0%` & `EPR ≥ 0.28` & `N ≥ 25` |
| `S+` | `1.50×` | `PWR ≥ 65.5%` & `EPR ≥ 0.22` & `N ≥ 25` |
| `A` | `1.00×` | `PWR ≥ 61.5%` & `EPR ≥ 0.17` & `N ≥ 20` |
| `B` | `0.75×` | `PWR ≥ 58.5%` & `EPR ≥ 0.13` |
| `C` | `0.50×` | `EPR ≥ 0.10` |
| `D` | `0.25×` | `EPR < 0.10` or Cold-Only Cooling (`4 SL`) |

- **Cold-Only Cooling:** Per-setup streak `3 SL → cap 0.50×`, `4 SL → cap 0.25×` until next win. Never boosts above static cap.

### 2.6 Unlimited S2 Stepped Compounding & Auto-Turbo 250G
- **Step Ladder:** `tier_cap = E0 * 1.08^step` (`+8%` step up; `-10%` below `tier_cap` step down).
- **Step-Locked Volume:** Inside a given S2 step, `1U = (tier_cap / buffer) * vol_mult` stays **strictly fixed** (no equity-based micro-shrinking on losses so wins recover losses cleanly).
- **Auto-Turbo 250G (110% Armor):** Activates at `eq >= 1.25 * E0`; reverts to `400G` if `eq <= 1.10 * E0` or if `eq - 30 * (tier_cap / 250) < 1.10 * E0`.

### 2.7 Slow Bayesian Learning & Shadow Bench-Watch (`core/live_learning.py`)
- **Slow Bayesian Prior:** `prior_weight = 60` equivalent trades from 3Y history; requires `>= 15` live closed trades before allowing a V3 tier change.
- **3-State Lifecycle (`ACTIVE ↔ BENCH_WATCH ↔ PROBATION`):**
  - `ACTIVE → BENCH_WATCH`: Triggered on Y3 bleed (`≤ -15U, PF < 0.70`) or rolling 10 live trades (`≤ -12U, PF < 0.75`). Sets `vol = 0.0×` while continuing to log shadow trades (`record_shadow_trade`).
  - `BENCH_WATCH → PROBATION`: Triggered when last `6` shadow trades achieve `Net ≥ +6U`, `PWR ≥ 58%`, `PF ≥ 1.35`, and last trade `> 0`. Resumes real capital capped at `0.50×` for `5` trades.
  - `PROBATION → ACTIVE`: Graduates after `5` live trades with `Net > 0` (returns to `BENCH_WATCH` if `≤ -8U`).

---

## 3. Historical Research Decisions (Do Not Re-Test Blindly)
- **`5m` timeframe dropped:** Fee/microstructure churn hurt portfolio Calmar and triggered false 2-SL brakes.
- **`AVAXUSDT` excluded:** Only `+25.7%` standalone (PF `1.25`) diluted by high-frequency `15m EMA34_W1` (`175` trades) and high ETH/SOL correlation.
- **`ZECUSDT` excluded:** Strong standalone (`+63.8%`, PF `1.59`), but adding ZEC+NEAR (`79` setups) congested the Max-5-Open cap and raised DD (`-12.80%`) plus carries regional privacy-coin regulatory risk.
- **`NEARUSDT` included (15 ACTIVE + 1 BENCH_WATCH):** Standalone `+64.4%` (DD `-8.12%`, 3/3 years positive) and when added to the 6 core coins, **cut portfolio MaxDD while boosting ROI & PF**.

---

## 4. Smart Directory Structure (v6.0)
- `core/` — Production strategy rules (`core_strategy_rules.json`), adaptive sizing/buffer, Slow Bayesian + Bench-Watch engine (`live_learning.py`, `live_learning_state.json`), and official HTML reports (`core/reports/chicktrade_v6_7coin_master_report.html`, `chicktrade_xauusd_100x_report.html`).
- `cloud_bot/` — Zero-cost live Cloud Paper Bot (`paper_engine_v3.py`, 7 coins) & Cloudflare Worker (`cloudflare_worker.js`, zero secrets).
- `bot/` — **UNTOUCHED** live bot runtime (DO NOT modify unless explicitly instructed by the user).
- `docs/` — GitHub Pages live dashboard (`docs/index.html`, `docs/data/*`) & quant framework (`docs/CHICKTRADE_SETUP_AND_VOLUME_FRAMEWORK.md`).
- `verified_tests/` — Gold-Standard v6.0 data downloader, 6-year audit engine, single-coin discovery tool, and `build_v6_master.py` + locked JSON results.
- `coins/XAUUSD/` — Independent XAUUSD 100x quant study.
- `engine/`, `risk/`, `strategy/`, `config.py`, `tests/` — Core backtest primitives & 18 unit tests.
- `graphify-out/` — Rebuilt v6.0 Knowledge Graph (`GRAPH_REPORT.md`, `graph.html`, `graph.json`).
- `old/` — Local-only gitignored archive of v1–v5 legacy artifacts.

---

## 5. Non-Negotiable Engineering Rules & Verification Commands
1. **NEVER modify `bot/`** without explicit user approval.
2. **NEVER commit secrets, API keys, Telegram Bot Tokens, or personal Chat IDs.**
3. Always run after changes:
   ```powershell
   python -m unittest discover -s tests
   node --check cloud_bot/cloudflare_worker.js
   python cloud_bot/paper_engine_v3.py --once
   ```
4. To rebuild the entire v6.0 ruleset & Master HTML from scratch:
   ```powershell
   python -m verified_tests.scripts.build_v6_master
   ```
