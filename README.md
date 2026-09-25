# NQ Discovery Foundry (v0)

A research engine that searches for NQ futures strategies and then tries hard to kill them. Success
means an honest pipeline that reports which candidates survive, which die, and why. It is not a trading
system: there are no broker connections and no order routing.

**Status:** phases 1 (data pipeline), 2 (evaluator and strategy DSL), 3 (kill tests) and 4 (search) are
built and tested on synthetic data. The real data build is waiting on a NinjaTrader export covering 2025 onward (see
`TODO.md`). Phases 5–6 have not started.

## Setup

```powershell
copy .env.example .env                         # then put your DEEPSEEK_API_KEY in .env
uv sync                                        # Python 3.12 env with all dependencies
uv run pytest                                  # full test suite
uv run ruff check foundry tests; uv run mypy   # lint + strict types
```

## Commands

| Command | What it does |
| --- | --- |
| `foundry data build --config config/v0.yaml` | Loads raw files, labels sessions, runs quality checks, builds the continuous contract, writes `data/processed/` (search + validation), seals `data/holdout_sealed/`, writes `data_report.md`, then verifies its own outputs. Exits 1 if any check fails. |
| `foundry spec validate strategies/baseline_rth_orb.json` | Checks a strategy spec against the DSL schema and semantic rules, and prints its hash. |
| `foundry evaluate <spec> --split search [--contract MNQ]` | Backtests a spec with costs always on and records it in `results/foundry.duckdb`. `--split validation` uses search bars as warm-up and trades only validation bars. |
| `foundry critique <spec>` | Runs the 8 kill tests on search, then on validation, and records every number and the verdict (ADR 0008). |
| `foundry search [--no-llm]` | One search run on the search split: LLM, genetic, CMA-ES and random generators, surrogate pre-screen, hard budgets. The top candidates then go to the critics (ADR 0009). |
| `foundry funnel` | Generated → evaluated → killed by each test → alive, from the results store. |
| `foundry holdout status --config config/v0.yaml` | Verifies the sealed files against their SHA-256 manifest. Shows row counts and the access log, never data. |
| `foundry holdout open --engine <id> --candidates <hashes> --reason <text>` | The one way to evaluate on the holdout. Logged before any data is read, and at most once per candidate. **Don't run it without the owner's decision.** |

## Adding a data source

Every source in `config/v0.yaml` must state its `timezone` and whether timestamps mark bar `start` or
`end`. There are no defaults. The build fails if more than 0.1% of bars fall in the 17:00–18:00 ET
maintenance break, which catches a wrong timezone. The owner's NinjaTrader files are UTC with bar-end
stamps (ADR 0002 has the evidence).

## Reading the data report

- **Failed checks** comes first. Any failure means no processed series was written.
- **Splits** shows rows and trading days per split. For the holdout it shows counts and a hash only.
- **Rolls** lists every pre-holdout roll: the decision day (volume crossover), the day it takes effect
  (the next session), the gap in points, and the continuity check.
- **Quality flags** and **Gaps**: problem bars are flagged and missing minutes are counted. Nothing is
  filled.

## Writing a strategy spec

A spec is JSON (`foundry/domains/nq/dsl_schema.json`), never code. Every number is a parameter with
bounds, so the search stays inside a closed space. Operands have units (`price`, `points`,
`unitless`): you can compare `close` with the opening-range high, but never `close` with a number,
because absolute price levels don't survive back-adjustment and are an overfitting trap. Fill and
exit rules are in ADR 0007: signal on bar close, fill at the next bar's open, stop before target,
costs always on.

## Why the biggest backtest profit is not the answer

Search enough combinations and some will look spectacular by chance. In a test on synthetic data with **no
edge at all**, the best of 3,000 random strategies made +$7,309 in-sample with a Sharpe of 5.3. The top 20
in-sample winners **all lost money** on unseen validation data. So the Foundry ranks nothing by backtest
profit. A candidate counts only if it survives every kill test on both search and validation, with the
deflated Sharpe accounting for how many strategies were tried.

## Layout

| Path | Contents |
| --- | --- |
| `foundry/core/` | Domain-agnostic config, hashing, reproducibility, verification, holdout vault |
| `foundry/domains/nq/` | Data pipeline, plus DSL (`dsl.py`, `dsl_schema.json`), features, signal compiler, Numba backtester, evaluator |
| `foundry/search/` | Search loop, engine config, CMA-ES, surrogate, LLM client and proposer |
| `foundry/critics/` | Kill-test statistics and runner |
| `strategies/` | Hand-written specs (the baseline RTH opening-range breakout) |
| `config/` | `v0.yaml` and `calendar_cme_equity.yaml` |
| `data/` | `raw/`, `processed/`, `holdout_sealed/` (git-ignored) |
| `docs/decisions/` | ADRs 0001–0009 |
| `docs/plans/` | Phase plans |

## Ground rules enforced in code

- **The holdout is sealed.** Only `foundry/core/holdout.py` may touch it, and a test fails if any other
  module references it. Every opening is logged before data is returned, and each candidate can be
  opened once.
- **No lookahead.** A roll decided from day d's volume takes effect on day d+1. Bad-tick ATR uses only
  prior bars. Both are tested by truncating or perturbing future data.
- **Processed prices do not depend on holdout data.** The adjustment is anchored on the last pre-holdout
  contract (ADR 0003).
- **Reproducible.** Seed, git state, config/calendar/input/output hashes and package versions go into
  `build_manifest.json`. Two builds of the same inputs give byte-identical outputs.
