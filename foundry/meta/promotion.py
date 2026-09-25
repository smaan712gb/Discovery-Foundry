"""Deterministic promotion rule: paired one-sided Wilcoxon plus hard constraints (ADR 0010)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.stats import wilcoxon


@dataclass(frozen=True)
class RunOutcome:
    engine_id: str
    seed: int
    run_id: str
    status: str  # "completed", "stopped:<budget>", or "error:<message>"
    score: float
    alive: int
    evaluated: int
    wall_clock: float
    critic_config_hash: str
    evaluated_hashes: tuple[str, ...]


@dataclass
class Decision:
    promote: bool
    p_value: float
    constraints: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def reason(self) -> str:
        failed = [k for k, v in self.constraints.items() if not v["passed"]]
        if self.promote:
            return "challenger promoted"
        return "champion stays: " + (", ".join(failed) if failed else "no significant improvement")


def paired_wilcoxon_greater(challenger: list[float], champion: list[float]) -> float:
    """One-sided p-value that challenger scores exceed champion scores, paired by seed."""
    d = np.asarray(challenger, float) - np.asarray(champion, float)
    if d.size == 0 or np.all(d == 0):
        return 1.0
    res = wilcoxon(d, alternative="greater", zero_method="zsplit", method="auto")
    p = float(res.pvalue)
    return 1.0 if math.isnan(p) else p


def decide(
    champion: list[RunOutcome],
    challenger: list[RunOutcome],
    alpha: float,
    max_runtime_ratio: float,
    reproducible: bool,
) -> Decision:
    by_seed_a = {r.seed: r for r in champion}
    by_seed_b = {r.seed: r for r in challenger}
    seeds = sorted(set(by_seed_a) & set(by_seed_b))
    a = [by_seed_a[s].score for s in seeds]
    b = [by_seed_b[s].score for s in seeds]
    p = paired_wilcoxon_greater(b, a)
    rt_a = float(np.mean([by_seed_a[s].wall_clock for s in seeds])) if seeds else 0.0
    rt_b = float(np.mean([by_seed_b[s].wall_clock for s in seeds])) if seeds else 0.0
    critic_hashes = {r.critic_config_hash for r in champion + challenger}
    errors = [r.run_id for r in champion + challenger if r.status.startswith("error")]
    c: dict[str, dict[str, Any]] = {
        "significant": {
            "passed": p < alpha,
            "p_value": p,
            "alpha": alpha,
            "champion_scores": a,
            "challenger_scores": b,
        },
        "budget": {"passed": not errors, "errored_runs": errors},
        "runtime": {
            "passed": rt_a == 0 or rt_b <= max_runtime_ratio * rt_a,
            "champion_mean_s": rt_a,
            "challenger_mean_s": rt_b,
            "max_ratio": max_runtime_ratio,
        },
        "critic_config": {"passed": len(critic_hashes) == 1, "hashes": sorted(critic_hashes)},
        "reproducible": {"passed": reproducible},
        "paired_seeds": {
            "passed": len(seeds) == len(champion) == len(challenger) and len(seeds) >= 2,
            "seeds": seeds,
        },
    }
    return Decision(all(v["passed"] for v in c.values()), p, c)
