# ADR 0010: Meta layer: engine proposals, tournaments, promotion

- Status: accepted
- Date: 2026-09-24

## Context

design.md section 8: an engine is a versioned search config (ADR 0009). Engine N+1 is proposed by
changing it, the two engines compete on equal compute over K seeds, and N+1 replaces N only if a paired
one-sided Wilcoxon test says it is better and no constraint is broken. The critics are fixed and can't
be tuned by this loop.

## Decision

### Registry

`engines` table: engine id, label, config JSON, parent id, proposer (`code` / `llm` / `manual`) and
rationale. `champions` table: which engine was champion from which tournament. Engine v0 is registered as
the first champion. Every challenger's YAML is written to `config/engines/<label>.yaml`, so any engine can
be run by hand.

### Proposing N+1 (`foundry meta propose`)

- **Code proposer** (default, deterministic from its seed): Dirichlet re-draw of the generator mix around
  the current one, jitter of the genetic, CMA and surrogate numbers within their bounds.
- **LLM proposer** (`--llm`): the model gets the engine JSON schema, the champion's config and the
  tournament history (aggregate scores only). It returns `{"engine": {...}, "rationale": "..."}`, which
  is validated as an `EngineConfig`. A proposed prompt template must keep its `{{DSL}}`, `{{FEEDBACK}}`
  and `{{N}}` placeholders and stay under 6,000 characters. It is written to a new file, never over an
  existing one.
- Engines can't touch critic thresholds, budgets or data. Those aren't in `EngineConfig`.

### Tournament (`foundry meta tournament`)

- K seeds (`meta.seeds`, default 6). For each seed, champion and challenger each run one full search plus
  critics under **the same budget** (`meta.run_budget`).
- **Isolation**: each (engine, seed) run gets its own results store under the tournament workspace. Trial
  counts, de-duplication and deflated Sharpe then reflect that run alone, and neither engine is helped or
  hurt by the other's history.
- **Reproducibility**: LLM replies are cached by prompt hash in a tournament-wide cache, so re-running a
  run replays the same replies. After scoring, the first seed of the challenger is **re-run in a fresh
  store**, and its evaluated spec hashes must match exactly.
- **Validation fold rotation**: the validation split is cut into `meta.validation_folds` (3) contiguous
  folds. Tournament t uses fold `t mod F` as the critics' validation stage. The validation trade minimum
  is scaled by the fold's share of validation days. The count of tournaments that have touched validation
  (total and per fold) is stored and reported, because each one spends some of validation's
  independence.
- **Score per run** = Σ over survivors of (1 + clip(validation Sharpe, 0, 5) / 5). The survivor count
  dominates, and quality breaks ties. A run with no survivors scores 0.

### Promotion (deterministic)

The challenger is promoted only if **all** hold:

1. One-sided Wilcoxon signed-rank on the paired per-seed score differences (challenger − champion),
   p < `meta.wilcoxon_alpha` (0.05). If every difference is zero, p = 1. With 6 seeds, the smallest
   achievable p is 1/64 ≈ 0.016, so at least 5 wins with no big loss are needed.
2. Budget: no challenger run exceeded any budget limit (a clean `stopped:` on a budget is allowed; an
   error is not).
3. Runtime: challenger mean wall clock ≤ `meta.max_runtime_ratio` (1.5) × champion's.
4. Critic pass rates: every run used the same critic config hash.
5. Reproducibility: the re-run check above passed.

Otherwise the champion stays. Every tournament (both engines, per-seed scores and run ids, the p-value,
each constraint's result, the fold, the decision) goes into the `tournaments` and `tournament_runs`
tables.

## Consequences

- A tournament costs 2K full searches plus one re-run. Budgets are per run, so the total cost is known
  up front: (2K + 1) × `meta.run_budget`.
- On data where nothing survives, both engines score 0 and nothing is ever promoted. That is intended:
  the meta layer can't "improve" an engine on noise.
