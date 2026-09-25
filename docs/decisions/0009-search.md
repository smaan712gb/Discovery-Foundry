# ADR 0009: Search engine: generators, surrogate, budgets, LLM boundary

- Status: accepted
- Date: 2026-09-24

## Decision

### Layering

- `foundry/search/` holds generic machinery: the generation loop, the genetic algorithm, the CMA-ES tuner,
  the LightGBM surrogate, and the provider-agnostic LLM client. It knows nothing about futures.
- The domain supplies a **spec space** (`foundry/domains/nq/space.py`): random specs, mutation, crossover,
  a fixed-length featurization for the surrogate, the numeric-parameter view for CMA-ES, and the text that
  describes the DSL to an LLM.

### One generation

1. **Propose** a batch using the engine's generator mix: `llm` (LLM proposer), `genetic` (mutation and
   crossover of the best specs so far), `cma` (CMA-ES steps on the parameters of the current best
   structure), and `random` (fresh random specs, for exploration and the cold start).
2. **Drop duplicates**: any spec hash already in the results store is never evaluated again.
3. **Pre-screen**: once the surrogate is trained, it ranks the batch and only the top `prescreen_keep`
   share goes to a full backtest.
4. **Evaluate** on the **search split only**. Every evaluation is stored with its lineage (generator, parent
   hashes, engine id), so it counts as a trial for the deflated Sharpe.
5. **Retrain** the surrogate on everything evaluated so far. Its Spearman correlation on that generation's
   new evaluations, predicted before they ran, is logged.

After the last generation, or when a budget runs out, the `top_k` specs by search fitness go to the
critics (ADR 0008).

### Fitness

Fitness is search-split annualized Sharpe, with a penalty when the trade count is below the critics'
search minimum (`fitness = sharpe − 10`). The search can never see validation or holdout numbers.

### The LLM boundary

- Exactly one module, `foundry/search/llm_client.py`, touches the network. It reads the key from the
  `DEEPSEEK_API_KEY` environment variable and never writes it to logs, the database or reports. Error
  messages have the key redacted.
- The model returns JSON (`{"specs": [...]}`). Each spec goes through `validate_spec`. Invalid ones are
  counted and discarded, and nothing the LLM returns is ever executed.
- The prompt contains: the DSL schema, the feature table, the engine's prompt template, aggregate kill
  counts by test, and short summaries of dead specs from the **search stage only** (structure, kill reason,
  search metrics). Validation and holdout appear only as aggregate pass/fail counts. The prompt never
  contains prices.
- Default model: `deepseek-flash` (DeepSeek-V4.1-Flash). Every call is charged at the **peak** list price
  ($0.30 per million input tokens, $1.20 per million output), so the dollar budget can never be
  underestimated. Prices live in config.
- `deepseek-flash` reasons ("thinks") by default, and reasoning tokens are output tokens. The `thinking`
  setting (default `low`) keeps reasoning from eating the whole output budget.

### Budgets (hard, per run)

Wall clock, number of backtests, LLM dollars and LLM tokens. The limit is checked before every generator
call and every backtest. On a breach the loop stops cleanly: nothing half-written, the run is recorded as
`stopped: <budget>`, and the specs found so far still go to the critics. An LLM call that would go over the
remaining dollar budget, judged by its maximum output size, isn't made.

### Records

- `runs` table: run id, engine id, seed, config hash, budgets, counts per generator, surrogate accuracy per
  generation, status.
- `llm_calls` table: tokens, dollars, specs returned / valid / new, and a prompt hash. No prompt text and
  no key.
