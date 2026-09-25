"""Strict Pydantic base model, YAML loading and canonical config hashing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict


class StrictModel(BaseModel):
    """Base for every config model: unknown keys are errors and instances are immutable."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ConfigError(Exception):
    """Raised when a config file is missing or invalid."""


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ConfigError(f"config file must contain a mapping at top level: {path}")
    return raw


def load_model[M: BaseModel](path: Path, model: type[M]) -> M:
    raw = load_yaml(path)
    try:
        return model.model_validate(raw)
    except ValueError as exc:
        raise ConfigError(f"invalid config {path}:\n{exc}") from exc


def canonical_json(model: BaseModel) -> str:
    """Stable JSON form of a validated model: formatting and key order in YAML do not matter."""
    return json.dumps(model.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))


def config_hash(model: BaseModel) -> str:
    return hashlib.sha256(canonical_json(model).encode("utf-8")).hexdigest()
