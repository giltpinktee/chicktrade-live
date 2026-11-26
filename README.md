# 🏆 CHICKTRADE v6.0 — 7-Coin Systematic Futures Engine (Binance-Exact)

> **For AI Agents & Developers:** Start with [AGENTS.md](AGENTS.md) for the complete mathematical specification, execution rules, and design history, and [docs/CHICKTRADE_SETUP_AND_VOLUME_FRAMEWORK.md](docs/CHICKTRADE_SETUP_AND_VOLUME_FRAMEWORK.md) for the 5-Gate Setup Selection & V3 Barbell Volume playbook.

## Key Performance (36 Months · `2023-11 → 2026-10` · `$10,000` Start · Max 5 Open)
Evaluated on the Argon-audited **Binance USD-M Futures Exact** simulator (`7 Coins`: `BTC, ETH, BNB, LINK, XRP, SOL, NEAR` · `68 ACTIVE + 1 BENCH_WATCH`, zero lookahead, true LIMIT IOC `±0.25U`, Gap-Through SL, no same-bar TP1+TP2, real 8h Binance Funding, per-symbol 20x notional caps, 12% max concurrent SL risk):

| Mode | S4 Final Equity (ROI) | S4 MaxDD | S4 Calmar | S4 PF / PWR | S5 Pessimistic (DD) |
|---|---:|---:|---:|---|---:|
| **Bao 400G** *(Safe · S2 Step-Locked)* | **`$351,815` (`+3,418%`)** | **`-11.56%`** | `295.7` | **1.66 / 62.4%** | `$268,624` (`-12.05%`) |
| **Turbo 350G** *(Mid-Safe)* | `$441,897` (`+4,319%`) | `-13.34%` | `323.8` | 1.59 | `$353,161` (`-14.11%`) |
| **Turbo 320G** *(Peak Calmar < 15% DD)* | `$490,897` (`+4,809%`) | `-14.27%` | **`337.0`** | 1.54 | `$398,209` (`-15.19%`) |
| **Auto-Turbo 250G** *(ON ≥ +25% · 110% Armor)* | **`$618,457` (`+6,085%`)** | `-18.98%` | `320.6` | 1.46 / 59.4% | `$494,436` (`-19.43%`) |

## Core Architecture Highlights
- **V3 Barbell Alpha Volume (`0.25× → 1.75×`):** Allocates risk by `PWR` + `Edge-per-Risk (EPR = avg_u / sl_u)` (`corr(RecoveryFactor) = +0.735`), keeping static average volume at just `0.82×` while doubling profit.
- **Cold-Only Cooling:** Automatically scales down per-setup volume on `3 SL` (`0.50×`) and `4 SL` (`0.25×`); never boosts above static cap.
- **Slow Bayesian Learning + Shadow Bench-Watch (`core/live_learning.py`):** 3-state lifecycle (`ACTIVE ↔ BENCH_WATCH ↔ PROBATION`) tracks hypothetical future trades of paused setups and automatically reactivates them at `0.50×` probation once recovered.
- **Live Cloud Paper Bot & Dashboard:** Runs 24/7 on GitHub Actions (`cloud_bot/paper_engine_v3.py`, 7 coins) and publishes to GitHub Pages (`docs/`).

## Quick Links
- **Master Interactive HTML Report (v6.0):** `core/reports/chicktrade_v6_7coin_master_report.html`
- **Production Strategy Rules (68 setups / 7 coins):** `core/core_strategy_rules.json`
- **Live Learning & Bench-Watch State:** `core/live_learning_state.json`
- **Knowledge Graph:** `graphify-out/GRAPH_REPORT.md` & `graphify-out/graph.html`
- **Setup & Volume Playbook:** `docs/CHICKTRADE_SETUP_AND_VOLUME_FRAMEWORK.md`

## Rebuild & Verify
```powershell
python -m verified_tests.scripts.build_v6_master   # Full 36M v6.0 simulation + HTML + rules sync
python -m unittest discover -s tests               # 18 unit tests
python cloud_bot/paper_engine_v3.py --once         # 7-coin live Cloud Paper Bot smoke test
```
