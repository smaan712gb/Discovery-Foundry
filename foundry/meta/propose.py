"""Propose engine N+1 from engine N: by code (deterministic) or by an LLM (validated). ADR 0010."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import ValidationError

from foundry.core.config import config_hash
from foundry.meta.config import MetaConfig
from foundry.search.engine import EngineConfig
from foundry.search.llm_client import LLMClient, LLMError

PLACEHOLDERS = ("{{DSL}}", "{{FEEDBACK}}", "{{N}}")
MAX_PROMPT_CHARS = 6000
GENERATORS = ("random", "genetic", "cma", "llm")


class ProposalError(Exception):
    pass


@dataclass(frozen=True)
class EngineProposal:
    cfg: EngineConfig
    engine_id: str
    proposer: str
    rationale: str
    prompt_text: str | None = None  # a new prompt template the proposal wants written


def engine_id(cfg: EngineConfig) -> str:
    return f"{cfg.label}-{config_hash(cfg)[:12]}"


def propose_by_code(
    parent: EngineConfig, label: str, meta: MetaConfig, seed: int, allow_llm: bool
) -> EngineProposal:
    rng = np.random.default_rng(seed)
    d = parent.model_dump(mode="json")
    mix = np.array([d["mix"][g] for g in GENERATORS], float)
    if not allow_llm:
        mix[GENERATORS.index("llm")] = 0.0
    alpha = mix / max(mix.sum(), 1e-9) * meta.mix_concentration + 0.5
    if not allow_llm:
        alpha[GENERATORS.index("llm")] = 1e-9
    new_mix = rng.dirichlet(alpha)
    new_mix = np.round(new_mix, 4)
    new_mix[int(np.argmax(new_mix))] += 1.0 - float(new_mix.sum())  # exact sum of 1
    d["mix"] = {g: float(max(v, 0.0)) for g, v in zip(GENERATORS, new_mix, strict=True)}

    def jitter(value: float, lo: float, hi: float, integer: bool) -> float | int:
        v = value * float(np.exp(rng.normal(0, meta.jitter)))
        v = min(max(v, lo), hi)
        return round(v) if integer else round(v, 4)

    g = d["genetic"]
    g["sigma"] = jitter(g["sigma"], 0.02, 1.0, False)
    g["crossover_rate"] = min(1.0, max(0.0, round(g["crossover_rate"] + rng.normal(0, 0.1), 3)))
    g["elite"] = jitter(g["elite"], 2, 200, True)
    d["cma"]["sigma0"] = jitter(d["cma"]["sigma0"], 0.02, 1.0, False)
    s = d["surrogate"]
    s["oversample"] = jitter(s["oversample"], 1.0, 10.0, False)
    s["min_train"] = jitter(s["min_train"], 10, 5000, True)
    d["label"] = label
    cfg = _validate(d)
    return EngineProposal(
        cfg, engine_id(cfg), "code", f"code perturbation of {parent.label} (seed {seed})"
    )


def _validate(d: dict[str, Any]) -> EngineConfig:
    try:
        return EngineConfig.model_validate(d)
    except ValidationError as exc:
        raise ProposalError(f"proposed engine is invalid:\n{exc}") from exc


LLM_SYSTEM = (
    "You improve the configuration of an automated strategy-search engine. You reply with "
    "one JSON object only and never write code."
)


def propose_by_llm(
    parent: EngineConfig,
    label: str,
    history: list[dict[str, Any]],
    client: LLMClient,
    prompts_dir: Path,
) -> EngineProposal:
    """The LLM sees the engine schema, the champion's config and aggregate tournament history."""
    schema = EngineConfig.model_json_schema()
    user = (
        "Propose the next engine configuration. It will compete against the current one on equal "
        "compute over several seeds; it wins only by producing more strategies that survive strict "
        "out-of-sample kill tests. Change a few things with a clear reason.\n\n"
        f"## Engine JSON schema\n{json.dumps(schema)}\n\n"
        f"## Current champion\n{parent.model_dump_json()}\n\n"
        f"## Tournament history (aggregate scores only)\n{json.dumps(history)}\n\n"
        "You may also replace the search prompt by giving `prompt_text`; it must contain the "
        f"placeholders {', '.join(PLACEHOLDERS)} and be under {MAX_PROMPT_CHARS} characters. "
        'Reply as JSON: {"engine": {...full config...}, "rationale": "...", "prompt_text": null}'
    )
    try:
        reply = client.complete_json(LLM_SYSTEM, user)
        data = json.loads(reply.content)
    except (LLMError, ValueError) as exc:
        raise ProposalError(f"LLM engine proposal failed: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("engine"), dict):
        raise ProposalError("LLM reply has no 'engine' object")
    d = copy.deepcopy(data["engine"])
    d["label"] = label
    prompt_text = data.get("prompt_text")
    if prompt_text:
        if (
            not isinstance(prompt_text, str)
            or len(prompt_text) > MAX_PROMPT_CHARS
            or not all(p in prompt_text for p in PLACEHOLDERS)
        ):
            raise ProposalError("proposed prompt_text is too long or misses a placeholder")
        target = prompts_dir / f"proposer_{label}.md"
        if target.exists():
            raise ProposalError(f"refusing to overwrite existing prompt {target}")
        d.setdefault("llm", {})["prompt_template"] = str(target)
    else:
        # Keep the champion's prompt unless a new one is supplied.
        d.setdefault("llm", {})["prompt_template"] = str(parent.llm.prompt_template)
    cfg = _validate(d)
    return EngineProposal(
        cfg,
        engine_id(cfg),
        "llm",
        str(data.get("rationale", ""))[:500],
        prompt_text if prompt_text else None,
    )


def write_engine(p: EngineProposal, engines_dir: Path) -> Path:
    """Write the challenger's YAML (and new prompt, if any). Never overwrites."""
    import yaml  # noqa: PLC0415

    engines_dir.mkdir(parents=True, exist_ok=True)
    path = engines_dir / f"engine_{p.cfg.label}.yaml"
    if path.exists():
        raise ProposalError(f"refusing to overwrite {path}")
    if p.prompt_text is not None:
        prompt_path = Path(p.cfg.llm.prompt_template)
        prompt_path.write_text(p.prompt_text, encoding="utf-8")
    header = f"# Engine {p.cfg.label} ({p.engine_id}), proposed by {p.proposer}: {p.rationale}\n"
    path.write_text(
        header + yaml.safe_dump(p.cfg.model_dump(mode="json"), sort_keys=False), encoding="utf-8"
    )
    return path
