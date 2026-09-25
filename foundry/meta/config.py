"""Meta-layer settings. Not part of any engine: an engine can't change how it is judged."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field

from foundry.core.budgets import BudgetConfig
from foundry.core.config import StrictModel


class MetaConfig(StrictModel):
    seeds: int = Field(ge=2)
    run_budget: BudgetConfig
    wilcoxon_alpha: float = Field(gt=0, lt=1)
    max_runtime_ratio: float = Field(gt=1)
    validation_folds: int = Field(ge=1)
    workspace: Path
    mix_concentration: float = Field(gt=0)
    jitter: float = Field(gt=0, le=1)
