# ADR 0001: Architecture of the Discovery Foundry

- Status: accepted
- Date: 2026-09-24

## Context

The Foundry is a research engine that proposes trading strategies and then tries to kill them. NQ futures
are its first vertical, but the core has to stay domain-agnostic so other discovery domains can reuse the
search, critic, lineage and meta machinery. The failure mode this design guards against is self-deception:
lookahead, cost-free results, holdout leakage and multiple-testing blindness.

## Decision

### Layers and dependency direction

```
cli ──► meta ──► search ──► core ◄── critics
                   │          ▲
                   ▼          │
              domains/nq ─────┘      reports ──► core, domains (read-only)
```

- `foundry/core/` is domain-agnostic: config loading and hashing, reproducibility metadata, the
  `Domain` protocol, candidates, lineage, budgets, and the holdout vault. It imports nothing from
  `domains/`, `search/`, `critics/` or `meta/`.
- `foundry/domains/nq/` is the only place that knows about futures: raw file formats, sessions,
  calendars, rolls, contract specs, costs, the Numba backtester, the DSL schema and features. It
  implements the `core.domain.Domain` protocol.
- `foundry/critics/` holds the kill tests. They consume metric and PnL arrays through core types, so
  most of them (DSR, PBO/CSCV, CPCV, concentration, cost stress) are generic.
- `foundry/search/` holds the generators. They emit DSL specs only and see only the search split.
- `foundry/meta/` treats an engine as a versioned config and runs N vs N+1 tournaments. Critic
  thresholds are not part of an engine config and cannot be changed by the meta layer.
- `foundry/reports/` reads results and writes HTML and Markdown. It never computes metrics itself.

### Invariants enforced in code, not by convention

1. **No lookahead.** Signals on bar t trade at bar t+1 or later. Enforced in the backtester's fill
   logic and proven by shuffle tests. Data preparation follows the same rule: a roll decided from day
   d's volume takes effect from day d+1.
2. **Costs always on.** The evaluator has no code path that returns gross-only results.
3. **Sealed holdout.** `foundry.core.holdout.HoldoutVault` is the only module allowed to read or
   write `data/holdout_sealed/`. A static test fails the suite if any other module references it.
   Every opening is appended to `ACCESS_LOG.jsonl`, and a candidate can be evaluated on the holdout
   only once.
4. **Models propose, code decides.** LLM output is parsed as JSON and validated against
   `dsl_schema.json`. Nothing an LLM returns is executed.
5. **Self-verifying pipelines.** Every command ends with a verification step (schema, row counts,
   forbidden nulls, hashes) and exits non-zero on failure.
6. **Reproducibility.** Every artefact records seed, git commit and dirty flag, config hash, input
   data hashes, Python version and package versions.

### Stack

Python 3.12 via uv; Polars and Parquet for data; DuckDB for results and lineage; Numba for the
backtest inner loop; Pydantic v2 for config validation (`extra="forbid"` everywhere, so a typo in YAML
is an error, not a silent default); typer for the CLI; pytest and hypothesis for tests; ruff and
`mypy --strict` on `foundry/`.

### Configuration

All tunable numbers live in `config/*.yaml`. Each module receives a validated Pydantic model, and the
config hash is the SHA-256 of the canonical JSON dump of the validated model, so formatting changes
don't change the hash but semantic changes do.

## Consequences

- Adding a domain means implementing `Domain` under `domains/<name>/`. Core, critics, search and meta
  do not change.
- The holdout-access rule is testable, and the test breaks the build if it is violated.
- Some duplication is accepted in `domains/nq` (session logic in both the data build and the
  backtester's session-flat rule) in exchange for keeping core free of market concepts.
