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

One real call to DeepSeek (`deepseek-flash`) authenticated, but the account returned **402 Insufficient
Balance**. Nothing was charged and the key didn't appear in the error. The LLM generator is ready once
the account has credit. Until then, `foundry search --no-llm` runs the genetic, CMA-ES and random
generators.
