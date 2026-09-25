"""Engine config: *how* the search runs. Critic thresholds are not in here (ADR 0008).

An engine is identified by the hash of its validated config, so two engines differ exactly when
their search behaviour can differ. The meta layer (phase 5) proposes engine N+1 by editing this.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, model_validator

from foundry.core.config import StrictModel, config_hash, load_model


class GeneratorMix(StrictModel):
    random: float = Field(ge=0)
    genetic: float = Field(ge=0)
    cma: float = Field(ge=0)
    llm: float = Field(ge=0)

    @model_validator(mode="after")
    def _sums(self) -> GeneratorMix:
        total = self.random + self.genetic + self.cma + self.llm
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"generator mix must sum to 1, got {total}")
        return self


class GeneticConfig(StrictModel):
    elite: int = Field(ge=2)
    tournament: int = Field(ge=2)
    crossover_rate: float = Field(ge=0, le=1)
    sigma: float = Field(gt=0, le=1)


class CMAConfig(StrictModel):
    sigma0: float = Field(gt=0, le=1)
    popsize: int = Field(ge=4)


class SurrogateConfig(StrictModel):
    enabled: bool
    min_train: int = Field(ge=10)
    oversample: float = Field(ge=1)
    num_leaves: int = Field(ge=4)
    n_estimators: int = Field(ge=10)


class LLMProposerConfig(StrictModel):
    prompt_template: Path
    specs_per_call: int = Field(ge=1, le=20)
    death_examples: int = Field(ge=0, le=50)
    top_examples: int = Field(ge=0, le=20)


class EngineConfig(StrictModel):
    label: str
    generations: int = Field(ge=1)
    batch_size: int = Field(ge=1)
    mix: GeneratorMix
    genetic: GeneticConfig
    cma: CMAConfig
    surrogate: SurrogateConfig
    llm: LLMProposerConfig
    top_k: int = Field(ge=1)
    trade_penalty: float = Field(ge=0)


def load_engine(path: Path) -> tuple[EngineConfig, str]:
    """(config, engine_id). The id is the label plus the first 12 hex chars of the config hash."""
    cfg = load_model(path, EngineConfig)
    return cfg, f"{cfg.label}-{config_hash(cfg)[:12]}"
