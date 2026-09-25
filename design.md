NQ Discovery Foundry (v0)

You are building NQ Discovery Foundry v0: a self-improving research engine that searches for NQ futures trading strategies and then tries hard to kill them. It is the first vertical of a general "autonomous discovery foundry", so keep the core domain-agnostic and put everything NQ-specific behind a domain plugin.

0. Ground rules (read first, never violate)
The goal is not a strategy that "always wins". No such strategy exists. A candidate that looks perfect on history is presumed overfit until it survives every kill test in section 6. Success for this project = an honest pipeline that reports which candidates survive, which die, and why.
No lookahead, ever. A signal computed on bar t can only trade at the open of bar t+1 or later. Write tests that prove it.
Costs are always on. No result may be reported without commissions and slippage.
The sealed holdout is sacred (section 3). Code may not read it except through one explicit, logged command.
Pipelines verify themselves. Every script ends by checking its own output (row counts, schema, no NaNs where forbidden, reproducibility hash) and exits non-zero on failure. No placeholders, no hard-coded magic numbers outside config/.
Deterministic code decides; models propose. LLMs may only emit strategy specs in the DSL (section 5), validated against a JSON schema. They never write executable code and never touch the evaluator, the kill tests, the holdout or the promotion rules.
This is research tooling, not a trading system. No broker connections, no order routing. Survivors go to the owner's separate shadow-trading workflow.
Record every significant design decision as a short ADR in docs/decisions/ and every deferred item in TODO.md with why and what "done" means.
1. Stack
Python 3.12, uv for environments, ruff + mypy --strict on foundry/, pytest + hypothesis.
Data: Polars and Parquet; results in DuckDB.
Speed: Numba for the backtest inner loop.
Search: own genetic algorithm + CMA-ES (via cma or Optuna's CMA sampler); LightGBM surrogate.
LLM proposer: provider-agnostic client, Deepseek 4.1 flash first first(Deepseek_API_KEY from env, never logged), with a hard per-run dollar and token budget.
CLI with typer. Config in config/*.yaml validated with Pydantic.
Every run is reproducible: seed, git commit, config hash, data hash and package versions stored with results.
2. Repository layout
foundry/
  core/          # domain-agnostic: candidate, evaluator protocol, tournament, lineage, budgets
  search/        # generators: llm_proposer, genetic, cma, surrogate
  critics/       # kill tests (section 6), generic where possible
  meta/          # engine configs, engine N vs N+1 tournaments, promotion rules
  domains/nq/    # data loader, sessions, rolls, costs, backtester, DSL, features
  reports/       # HTML/Markdown reports, equity curves, kill-test tables
config/
data/            # raw/, processed/, holdout_sealed/ (git-ignored)
docs/decisions/
tests/
TODO.md
3. Data (phase 1)
Input: Tradingview historical export of NQ, 1-minute bars (tick optional later). Support Tradingview's text export format (yyyyMMdd HHmmss;open;high;low;close;volume) and a generic CSV/Parquet format with explicit timestamp timezone in config. Ask the owner for the file path and exported timezone before assuming.
Continuous contract: build back-adjusted continuous series from individual contract files. Roll rule configurable (default: roll when next contract's daily volume exceeds current). Store roll dates and adjustments; tests must prove no price jumps at rolls in the adjusted series.
Timezone: convert everything to America/New_York with zoneinfo, DST-correct.
Sessions: RTH = 09:30 to 16:00 ET; ETH = 18:00 to 09:30 ET next day; the 17:00 to 18:00 maintenance break has no bars; handle early closes and holidays from a calendar file in config/.
Quality checks: duplicate timestamps, gaps, zero volume, bad ticks (price jumps beyond a configurable ATR multiple), bars outside sessions. Report them; never silently fill.
Splits (configurable, defaults):
Search: 2018-01-01 to 2022-12-31
Validation: 2023-01-01 to 2024-12-31
Sealed holdout: 2025-01-01 to latest. Written to data/holdout_sealed/ with a SHA-256 manifest. Only foundry holdout open --engine <id> --candidates <ids> may read it; each opening is appended to data/holdout_sealed/ACCESS_LOG.jsonl, and a candidate may be evaluated on the holdout once.

Acceptance for phase 1: foundry data build produces processed Parquet plus a data report; tests cover DST transitions, rolls, sessions and the holdout lock.

4. Evaluator (phase 2): the most important component
Event-driven bar backtester in Numba. Signals on bar close, fills at next bar open (market) or when price trades through (limit/stop), never on the signal bar.
Contract specs from config: NQ tick 0.25 = $5.00, point = $20; MNQ point = $2 (support both).
Costs (config, defaults): commission $4.50 round turn per NQ contract; slippage 1 tick per side on market and stop orders, 0 on limit fills but limit fills require price to trade through by 1 tick.
Position sizing: fixed contracts in v0 (default 1). Session-flat rules: optional forced exit at session end.
Outputs per candidate: trade list, daily PnL, net PnL, trade count, win rate, profit factor, Sharpe (daily), Sortino, max drawdown, time in market, PnL split by RTH / ETH / regime.
Tests: golden tests against hand-computed trades; a lookahead test that shuffles future bars and asserts no change in signals up to t; a cost test proving net < gross; property tests with hypothesis.
5. Strategy DSL (phase 2)

A JSON strategy spec, schema in domains/nq/dsl_schema.json:

session: RTH | ETH | both,
features: from a whitelisted library (returns, ATR, VWAP distance, opening range, overnight high/low and gap, session volume, realized volatility, prior-day levels, rolling z-scores). No feature may use future data; each feature has a unit test.
entry, exit: boolean rule trees over features with numeric parameters; stop, target, time stop, trailing stop.
filters: regime filters (volatility bucket, trend state, day of week, event-day exclusion list from config).
params: bounds for every numeric parameter, so search is constrained. Specs are hashed; identical specs are never re-evaluated.
6. Critics: kill tests (phase 3)

A candidate is alive only if all pass, on search and then on validation data:

Net profit > 0 after costs; at least N trades (config, default 200 on search, 80 on validation).
Deflated Sharpe Ratio (Bailey and Lopez de Prado) > threshold, using the true number of trials from the lineage DB.
Probability of Backtest Overfitting via combinatorially symmetric cross-validation (CSCV) below threshold (default 0.2).
Purged, embargoed walk-forward (combinatorial purged CV) with stable out-of-fold performance.
Parameter robustness: perturb each parameter +/-10 and 20 percent; median neighbour must stay profitable.
Concentration: top 5 percent of days contribute less than 50 percent of profit.
Regime stability: profitable in at least 2 of 3 volatility regimes, or explicitly labelled regime-specific.
Cost stress: survives 2x slippage.
Optional: White's Reality Check or Hansen SPA across the surviving set. Every kill writes the failed test and the numbers to the results DB. The report shows the full funnel: generated, evaluated, killed by each test, alive.
7. Search (phase 4)
LLM proposer: given the DSL schema, the feature library, summaries of what died and why (never raw prices, never validation or holdout numbers beyond aggregate pass/fail), proposes new hypotheses as DSL specs with a one-line rationale. Budget-capped.
Genetic algorithm and CMA-ES tune parameters and mutate rule trees within bounds.
Surrogate: LightGBM predicting search-set Sharpe from spec features, used to pre-screen before full backtests; retrained each generation; its accuracy logged.
Search only ever sees the search split. Validation is used by critics and tournaments.
8. Meta layer: engine N vs N+1 (phase 5)
An engine is a versioned config: generator mix, mutation rates, surrogate settings, LLM prompt template, budget allocation, critic thresholds are not part of it (critics are fixed).
foundry meta propose creates engine N+1 by changing the engine config (a model may propose changes; code validates them).
foundry meta tournament runs N and N+1 with equal compute budgets over K seeds on the search split and scores each engine by the number and quality of candidates that survive the critics on validation.
Promotion rule (deterministic): N+1 is promoted only if it beats N with a paired statistical test (default one-sided Wilcoxon, p < 0.05 across seeds) and breaks no constraint (budget, runtime, critic pass rates, reproducibility). Otherwise N stays. Every tournament is logged with full lineage.
To limit validation overfitting at the meta level, rotate validation folds across tournaments and keep a count of how many tournaments have touched validation; report it.
9. Governance and safety
Hard budgets per run: wall-clock, CPU, LLM dollars and tokens. Abort cleanly on breach.
Everything logged to DuckDB: candidates, specs, lineage (parent specs, generator, engine version), metrics, kills, tournaments, holdout openings.
No network access during backtests. LLM calls go through one client module only.
Secrets only from environment variables; never written to logs, reports or the DB.
10. Reports

foundry report <run> writes an HTML report: data summary, funnel, surviving candidates with equity curves (search vs validation), RTH vs ETH breakdown, kill reasons, engine tournament history, trial counts used in the deflated Sharpe. Label every chart with the split it comes from.

11. Build order and definition of done

Build and fully test one phase before starting the next:

Data pipeline (section 3)
Evaluator and DSL (sections 4 and 5), with one hand-written baseline strategy (e.g. RTH opening-range breakout) as a sanity check
Critics (section 6)
Search (section 7)
Meta layer (section 8)
Reports (section 10)

Done for v0 = foundry run --config config/v0.yaml goes end to end on the owner's data, produces the report, all tests pass, and a README explains how to run it and how to read the funnel. Then stop and summarise results honestly, including if nothing survived. Do not open the holdout without the owner's explicit instruction.

12. First action

Before writing code: ask the owner for (a) the export path and its timezone, (b) the date range available, (c) bar size, (d) whether to include MNQ. Then write ADR 0001 (architecture) and the phase-1 plan, and start phase 1.