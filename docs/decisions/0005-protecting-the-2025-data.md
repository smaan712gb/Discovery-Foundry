# ADR 0005: The existing 2025 files are future holdout; no strategy may be evaluated on them

- Status: superseded by ADR 0006 (the owner moved the history start to 2025; the holdout is now 2026-05-01 onward and the same protection rules apply to it)
- Date: 2026-09-24

## Context

Right now the only real data is 2025 (the owner's 5 NQ contract files). Under the default splits all of 2025
is the sealed holdout. Evaluating strategies on it during development, even "just to check the backtester",
would contaminate the holdout before the real search starts: every design choice made after seeing those
results would be fitted to it.

## Decision

- The 2025 files may be used for **data-pipeline engineering only**: parsing, timezone detection, session
  labels, rolls and quality checks. None of these produce strategy returns.
- Evaluator, DSL, critic and search development (phases 2–4) use synthetic data from
  `foundry/domains/nq/synthetic.py` until the owner's 2018–2024 export is in place.
- No development config shifts the holdout boundary to make 2025 visible to the evaluator.
- The data report shows only row counts, date ranges and hashes for the holdout split: no prices, no returns
  and no per-session statistics.

## Consequences

The pipeline can't show a real-data strategy result until the 2018–2024 export arrives. That is accepted.
