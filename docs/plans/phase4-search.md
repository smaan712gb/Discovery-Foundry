# Phase 4: search

Status: built and tested on synthetic data (2026-09-24). Design choices are in ADR 0009.

## What exists

| Component | Module |
| --- | --- |
| Generation loop: propose, dedupe, pre-screen, evaluate (search split only), learn | `foundry/search/run.py` |
| Engine config (what the meta layer may change) and its id | `foundry/search/engine.py`, `config/engines/engine_v0.yaml` |
| Genetic operators, random specs, featurization, CMA view, LLM description | `foundry/domains/nq/space.py` |
| CMA-ES parameter tuner | `foundry/search/cma_tuner.py` |
| LightGBM surrogate, with Spearman accuracy logged each generation | `foundry/search/surrogate.py` |
| LLM client (the only network code), with hard budgets and key redaction | `foundry/search/llm_client.py` |
| LLM proposer: prompt from the DSL plus search-only feedback; every reply validated | `foundry/search/llm_proposer.py`, `config/prompts/proposer_v1.md` |
| Budgets and secrets | `foundry/core/budgets.py`, `foundry/core/secrets.py`, `.env.example` |
| CLI | `foundry search [--no-llm] [--no-critics] [--seed N]` |

## Evidence (tests/test_search.py, test_space.py, test_llm.py)

- Every evaluated spec is a distinct stored trial; nothing is evaluated twice. Lineage (generator and
  parents) is stored.
- Budgets stop the loop cleanly, and the top candidates still go to the critics.
- Without the LLM, the same seed produces the same run.
- LLM path, tested with a fake transport: request shape, charging at peak prices, calls refused before
  sending when they could break the budget, the key redacted from errors, invalid specs counted and
  discarded, duplicate proposals evaluated once.
- A static test enforces that only `llm_client.py` imports network code.
- Found and fixed along the way: spec hashes depended on JSON number formatting (`25` vs `25.0`), which
  would have let a re-tried spec escape de-duplication and the trial count. Numbers are now
  canonicalized by declared type.

## Live check (2026-09-24)

- First attempt: 402 Insufficient Balance. After funding, the old key had been revoked (401). The new
  key is in `.env`, but the Windows user environment still holds the revoked key, and the environment
  takes precedence (see TODO.md).
- `deepseek-flash` reasons ("thinks") by default. The first funded call spent all 6,000 output tokens
  on hidden reasoning and returned no answer. `thinking: low` is now in config, `max_output_tokens`
  is 16,000, and a reply that hits the limit before answering raises a clear error. Its tokens are
  still charged to the budget.
- With that setting, one call returned **5 of 5 valid specs** for **$0.012**: an opening-range
  breakout with volume confirmation, a VWAP stretch fade, a failed gap-up auction, a failed sweep of
  the prior-day high, and a breakout from a compressed opening range.
