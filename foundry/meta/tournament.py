"""Champion vs challenger over K seeds, each run isolated; then the promotion decision (ADR 0010).

Domain-agnostic: the domain supplies `RunFn`, which performs one full search + critics for
(engine, seed) in its own store and returns a RunOutcome.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from foundry.core.results import ResultsStore
from foundry.meta.config import MetaConfig
from foundry.meta.promotion import Decision, RunOutcome, decide
from foundry.search.engine import EngineConfig

# (engine config, engine id, seed, isolated store path, fold index) -> outcome
RunFn = Callable[[EngineConfig, str, int, Path, int], RunOutcome]


@dataclass
class TournamentResult:
    tournament_id: str
    champion_id: str
    challenger_id: str
    fold: int
    champion_runs: list[RunOutcome]
    challenger_runs: list[RunOutcome]
    reproducible: bool
    decision: Decision

    def summary(self) -> dict[str, Any]:
        return {
            "tournament_id": self.tournament_id,
            "champion": self.champion_id,
            "challenger": self.challenger_id,
            "validation_fold": self.fold,
            "champion_scores": [r.score for r in self.champion_runs],
            "challenger_scores": [r.score for r in self.challenger_runs],
            "p_value": self.decision.p_value,
            "promoted": self.decision.promote,
            "reason": self.decision.reason,
            "constraints": self.decision.constraints,
        }


def run_tournament(
    meta: MetaConfig,
    champion: tuple[EngineConfig, str],
    challenger: tuple[EngineConfig, str],
    run_fn: RunFn,
    registry: ResultsStore,
    workspace: Path,
) -> TournamentResult:
    n_before = registry.tournament_count()
    fold = n_before % meta.validation_folds
    tid = f"t{n_before + 1:04d}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}"
    root = workspace / tid
    runs: dict[str, list[RunOutcome]] = {champion[1]: [], challenger[1]: []}
    for seed in range(meta.seeds):
        for cfg, eid in (champion, challenger):
            runs[eid].append(run_fn(cfg, eid, seed, root / f"{eid}_s{seed}.duckdb", fold))
    # Reproducibility: re-run the challenger's first seed in a fresh store; same specs, same order.
    first = runs[challenger[1]][0]
    again = run_fn(
        challenger[0],
        challenger[1],
        first.seed,
        root / f"{challenger[1]}_s{first.seed}_rerun.duckdb",
        fold,
    )
    reproducible = again.evaluated_hashes == first.evaluated_hashes and not again.status.startswith(
        "error"
    )
    decision = decide(
        runs[champion[1]],
        runs[challenger[1]],
        meta.wilcoxon_alpha,
        meta.max_runtime_ratio,
        reproducible,
    )
    result = TournamentResult(
        tid,
        champion[1],
        challenger[1],
        fold,
        runs[champion[1]],
        runs[challenger[1]],
        reproducible,
        decision,
    )
    registry.record_tournament(
        result.summary(),
        [
            (r.engine_id, r.seed, r.run_id, r.status, r.score, r.alive, r.evaluated, r.wall_clock)
            for r in runs[champion[1]] + runs[challenger[1]]
        ],
    )
    if decision.promote:
        registry.set_champion(challenger[1], tid)
    return result
