"""Hard per-run budgets: wall clock, backtests, LLM dollars and tokens (ADR 0009)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from pydantic import Field

from foundry.core.config import StrictModel


class BudgetConfig(StrictModel):
    wall_clock_seconds: float = Field(gt=0)
    max_evaluations: int = Field(ge=1)
    llm_dollars: float = Field(ge=0)
    llm_tokens: int = Field(ge=0)


class BudgetExceeded(Exception):  # noqa: N818 (a budget stop, not an error)
    def __init__(self, which: str, detail: str) -> None:
        super().__init__(f"budget exhausted: {which} ({detail})")
        self.which = which


@dataclass
class RunBudget:
    cfg: BudgetConfig
    started: float = field(default_factory=time.monotonic)
    evaluations: int = 0
    llm_dollars: float = 0.0
    llm_tokens: int = 0

    def elapsed(self) -> float:
        return time.monotonic() - self.started

    def check(self) -> None:
        if self.elapsed() >= self.cfg.wall_clock_seconds:
            raise BudgetExceeded(
                "wall_clock", f"{self.elapsed():.0f}s of {self.cfg.wall_clock_seconds:.0f}s"
            )
        if self.evaluations >= self.cfg.max_evaluations:
            raise BudgetExceeded("evaluations", f"{self.evaluations} of {self.cfg.max_evaluations}")

    def reserve_evaluation(self) -> None:
        self.check()
        self.evaluations += 1

    def can_afford_llm(self, worst_case_dollars: float, worst_case_tokens: int) -> bool:
        return (
            self.llm_dollars + worst_case_dollars <= self.cfg.llm_dollars
            and self.llm_tokens + worst_case_tokens <= self.cfg.llm_tokens
        )

    def charge_llm(self, dollars: float, tokens: int) -> None:
        self.llm_dollars += dollars
        self.llm_tokens += tokens

    def summary(self) -> dict[str, float]:
        return {
            "elapsed_seconds": round(self.elapsed(), 1),
            "evaluations": self.evaluations,
            "llm_dollars": round(self.llm_dollars, 6),
            "llm_tokens": self.llm_tokens,
        }
