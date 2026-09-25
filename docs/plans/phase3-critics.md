# Phase 3: critics (kill tests)

Status: built and tested on synthetic data (2026-09-24). Design choices are in ADR 0008.

## What exists

| Component | Module |
| --- | --- |
| Statistics: PSR/DSR, expected max Sharpe, PBO via CSCV, CPCV select-then-test, top-day share | `foundry/critics/stats.py` |
| Thresholds (fixed; the meta layer can't change them) | `foundry/critics/config.py`, `critics:` in `config/v0.yaml` |
| Runner: 8 tests, cheapest first, search then validation, first failure kills | `foundry/critics/runner.py` |
| Records: every test's numbers, verdicts, funnel | `critic_results`, `verdicts` tables in `foundry/core/results.py` |
| CLI | `foundry critique <spec>`, `foundry funnel` |

## Evidence the critics work (tests/test_critics.py, tests/test_critic_stats.py)

- **Noise dies:** the baseline on a random walk is killed at the first test (it loses money after costs).
- **A real edge lives:** with an imperfect planted edge (momentum on 70% of days), the baseline passes all 16
  checks (8 tests on each of search and validation).
- **Lucky winners die:** take the best of 300 random strategies on data with no edge. It made $7,161
  in-sample, survived double slippage, wasn't concentrated in a few days, and was profitable in all three
  volatility regimes. The deflated Sharpe then killed it: DSR 0.18 against the 0.95 threshold, because 300
  tries make a Sharpe that high expected by chance.
- **Statistics:** expected max Sharpe matches Monte Carlo within 3%. PBO averages about 0.5 on pure noise
  and falls below 0.05 when one configuration truly dominates.
- **Critic-internal evaluations aren't trials:** after a full critique, the trial count is still 1.

## Known limits

- PBO is marked "not assessed" when the trial pool has fewer than `pbo_min_trials` specs (true for
  hand-written specs). It becomes meaningful once phase 4 generates hundreds of trials.
- A single PBO estimate is noisy (standard deviation about 0.2 on noise). Which is why no test decides alone.
- White's Reality Check / Hansen's SPA are deferred (TODO.md).
