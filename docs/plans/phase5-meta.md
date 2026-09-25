# Phase 5: meta layer (engine N vs N+1)

Status: built and tested on synthetic data (2026-09-24). Design choices are in ADR 0010.

## What exists

| Component | Module |
| --- | --- |
| Settings (not part of any engine) | `foundry/meta/config.py`, `meta:` in `config/v0.yaml` |
| Engine proposers: code (deterministic) and LLM (validated; a new prompt only with placeholders, never overwriting) | `foundry/meta/propose.py` |
| Promotion: one-sided paired Wilcoxon plus budget, runtime, critic-config, reproducibility and pairing constraints | `foundry/meta/promotion.py` |
| Tournament: K seeds × 2 engines, each run in its own store, re-run check, fold rotation, full records | `foundry/meta/tournament.py` |
| NQ run function: fold-restricted validation, trade minimum scaled to the fold, run scoring | `foundry/domains/nq/meta_runner.py` |
| LLM reply cache (makes tournaments replayable) | `LLMCache` in `foundry/search/llm_client.py` |
| Registry: engines, champions, tournaments, runs, validation touches | `foundry/core/results.py` |
| CLI | `foundry meta propose [--llm]`, `foundry meta tournament --challenger <yaml>`, `foundry meta status` |

## Evidence (tests/test_meta.py)

- The Wilcoxon rule gives p = 1 on all ties and exactly 1/64 on six wins. A split record doesn't
  promote.
- Each constraint on its own blocks promotion: runtime, errored run, a different critic config, failed
  reproducibility, no significant gain.
- The code proposer is valid and deterministic, and keeps the LLM share at 0 when it isn't allowed.
  LLM proposals are validated, and bad prompts or overwrites are refused.
- A cache hit sends no request and costs nothing.
- Fake-run tournament: a dominant challenger is promoted, and the reverse match keeps it. Folds rotate
  0, 1, and validation touches are counted.
- Real tournament on synthetic data: isolated stores (2K + 1), equal budgets, the re-run evaluates
  exactly the same specs.

## Cost

A tournament is (2 × seeds + 1) runs at `meta.run_budget`. With the defaults (6 seeds, $0.25 LLM per
run) that's at most **$3.25 of LLM spend** and up to 13 × 15 minutes of wall clock.
