# Phase 2: evaluator and strategy DSL

Status: built and tested on synthetic data (2026-09-24). Waiting on the real data build (TODO.md).

## What exists

| Component | Module | Notes |
| --- | --- | --- |
| DSL schema | `foundry/domains/nq/dsl_schema.json` | JSON Schema 2020-12. Every number is a bounded parameter. |
| Semantic validation and hashing | `foundry/domains/nq/dsl.py` | Units (`price` / `points` / `unitless`), no absolute price thresholds, depth limits, no unused params. The hash ignores name and rationale. |
| Features (15) | `foundry/domains/nq/features.py` | returns, ATR, VWAP distance, opening range, overnight high/low/gap, session volume ratio, realized vol, prior-day levels, close z-score, minutes into session |
| Market context | `features.build_context` | Session windows and flat bars from the calendar; volatility and trend regimes from earlier days; event days |
| Signal compiler | `foundry/domains/nq/signals.py` | Rule trees to boolean arrays; NaN never trades |
| Backtest kernel | `foundry/domains/nq/backtest.py` | Numba; semantics in ADR 0007 |
| Metrics | `foundry/core/metrics.py` | Net/gross PnL, trades, win rate, profit factor, daily Sharpe/Sortino, max drawdown, time in market |
| Evaluator | `foundry/domains/nq/evaluator.py` | Search sees search bars only; validation uses search bars as warm-up and trades only validation bars; NQ and MNQ; slippage multiplier for cost stress; holdout only through the vault |
| Results store | `foundry/core/results.py` | DuckDB tables: specs, evaluations, trades, daily PnL. Tracks the trial count for the deflated Sharpe ratio. Identical evaluations are served from the store. |
| Baseline strategy | `strategies/baseline_rth_orb.json` | RTH opening-range breakout |
| CLI | `foundry spec validate`, `foundry evaluate`, `foundry holdout open` | |

## Tests that carry the weight

- **Golden trades** (`tests/test_backtest.py`): market, stop gap-through, limit trade-through, stop-before-target,
  session flat, time stop, trailing stop, stop and limit entries, window gap, no pyramiding.
- **Lookahead**: kernel truncation property; feature prefix invariance for all 15 features; the design.md
  shuffle test (permute every bar after t and check that nothing up to t changes).
- **Costs**: net < gross, more slippage never helps, zero costs give reference fills, and MNQ equals NQ/10
  before commission.
- **Hand-computed feature values** for every feature.

## Sanity result

On a pure synthetic random walk the baseline loses about its costs (net −$2,530 against gross −$1,065 over
101 trades). This is expected: there's no edge, so costs dominate.
