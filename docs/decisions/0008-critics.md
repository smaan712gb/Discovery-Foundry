# ADR 0008: Kill tests (critics)

- Status: accepted
- Date: 2026-09-24

## Context

design.md section 6 lists the kill tests but leaves several choices open:

- which set of configurations the Probability of Backtest Overfitting (PBO) is computed over
- what "stable out-of-fold performance" means for a strategy whose parameters are fixed
- what counts as a "trial" for the Deflated Sharpe Ratio (DSR)
- how kills are recorded for the funnel

## Decision

### Order and short-circuit

Tests run cheapest first, and the **first failure kills** the candidate. The funnel counts each candidate
under the test that killed it. Numbers for every test that ran are stored, pass or fail.

1. `min_trades_and_profit`: net PnL > 0 after costs, and trades ≥ `min_trades[split]` (200 on search,
   80 on validation).
2. `cost_stress`: net PnL > 0 with slippage × `cost_stress_multiplier` (2).
3. `concentration`: the top `top_day_fraction` (5%) of days contribute < `max_profit_share` (50%) of
   net profit.
4. `regime_stability`: net PnL > 0 in at least `min_profitable` (2) of the 3 volatility buckets. A spec
   whose own filter trades fewer than 3 buckets is labelled **regime-specific** and must be profitable in
   every bucket it trades.
5. `deflated_sharpe`: DSR > `dsr_threshold` (0.95).
6. `parameter_robustness`: every parameter is moved alone by each of ±10% and ±20% (for a parameter at 0,
   by that fraction of its range). Moves that leave the bounds, and duplicates, are dropped. The **median**
   neighbour's net PnL must be > 0.
7. `pbo`: PBO from CSCV below `pbo_threshold` (0.2).
8. `cpcv`: combinatorial purged cross-validation must be stable (below).

All eight run on the search split. A candidate that survives runs all eight again on validation. It is
alive only if it passes both.

### Trials (DSR)

- **N** is the number of distinct specs ever evaluated on that split (default contract) in the results
  store. That includes every dead candidate, since that's the point of the deflation. Critic-internal
  evaluations (cost stress, parameter neighbours) are **not** trials: they're never stored, because no
  generator chose them.
- The variance of Sharpe across trials comes from the same set, using daily (non-annualized) Sharpe.
- DSR = PSR(SR₀), where SR₀ = √V · ((1−γ)Φ⁻¹(1−1/N) + γΦ⁻¹(1−1/(Ne))) and γ is the Euler–Mascheroni
  constant (Bailey & López de Prado 2014). PSR uses the candidate's daily Sharpe, the skewness and raw
  kurtosis of its daily PnL, and T trading days. With N = 1, SR₀ = 0.

### PBO: the family is the search's trial pool

PBO needs several configurations, so the family is **every distinct spec evaluated on that split** (default
contract, base costs), read from the results store. Their daily PnL forms the T × N matrix, with the candidate
as one column. CSCV splits the days into `cscv_blocks` (16) contiguous blocks. For every choice of half the
blocks as in-sample, it picks the best column by in-sample Sharpe and ranks it out of sample. PBO is the share
of splits where that rank is at or below the median (logit ≤ 0, the conservative side).

So PBO measures the **selection process** that produced the candidate: "when this search picks its in-sample
winner, does the winner hold up out of sample?"

*Rejected alternative:* scoring PBO over the candidate's parameter neighbours. On a robust plateau the
neighbours all perform alike, so picking the in-sample best among them is a coin flip and PBO sits near 0.5.
That would kill exactly the robust strategies we want.

With fewer than `pbo_min_trials` trials in the pool, PBO can't discriminate. The test is then recorded as
**not assessed** (it passes, but the verdict carries the label `pbo_not_assessed`, and reports show it). That
happens with hand-written specs. Once the phase-4 search runs, the pool holds hundreds of trials. On its own,
a single family's PBO is noisy (standard deviation about 0.2 on pure noise, `tests/test_critic_stats.py`),
which is one more reason the critics must all agree.

### CPCV: select, then test

Days are split into `groups` (6) contiguous groups, and every choice of `test_groups` (2) of them is a
split (15 splits). For each split:

- Training days are the other groups, minus a **purge** of H days before each test group (H is the
  candidate's longest trade in trading days; 0 for session-flat strategies) and an **embargo** of
  `embargo_days` after each test group.
- The family (the candidate plus its parameter neighbours from test 6) member with the best training
  Sharpe is selected, and its PnL on the test days is recorded. A robust plateau passes here, because
  whichever neighbour gets picked still makes money on the test days.

The test is passed if the share of splits with positive test PnL ≥ `min_positive_fraction` (0.6) **and** the
mean annualized test Sharpe > 0. This measures the whole "tune, then trade" procedure, not one fixed
configuration.

### Records

Each test result is a row (spec, stage, test, passed, numbers as JSON). Each candidate gets one verdict
(alive, killed by, stage, label). The funnel counts: specs generated, evaluated on search, killed by each
test at each stage, alive.

### Thresholds are not part of an engine

Critic thresholds live in `config/v0.yaml` under `critics:`. The meta layer (phase 5) can't change them.

## Deferred

White's Reality Check / Hansen's SPA across the surviving set is optional in design.md, so it's deferred
(TODO.md).
