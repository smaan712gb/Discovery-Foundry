"""LLM proposer: asks for new DSL specs given the DSL and what died. Output is data, not code."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from foundry.search.llm_client import LLMClient, LLMError


class SpecValidator(Protocol):
    def validate(self, raw: dict[str, Any]) -> Any: ...


@dataclass(frozen=True)
class Proposal:
    specs: list[Any]
    returned: int
    invalid: int
    dollars: float
    prompt_tokens: int
    completion_tokens: int
    error: str | None = None


SYSTEM = (
    "You design intraday NQ futures strategy hypotheses as JSON specs in a fixed DSL. "
    "You never write code. You reply with one JSON object only."
)


def build_prompt(template_path: Path, dsl_text: str, feedback: dict[str, Any], n: int) -> str:
    template = template_path.read_text(encoding="utf-8")
    return (
        template.replace("{{DSL}}", dsl_text)
        .replace("{{FEEDBACK}}", json.dumps(feedback, indent=1, sort_keys=True))
        .replace("{{N}}", str(n))
    )


def propose(client: LLMClient, validator: SpecValidator, prompt: str, name_prefix: str) -> Proposal:
    try:
        reply = client.complete_json(SYSTEM, prompt)
    except LLMError as exc:
        return Proposal([], 0, 0, 0.0, 0, 0, error=str(exc))
    try:
        data = json.loads(reply.content)
        items = data.get("specs", []) if isinstance(data, dict) else []
    except ValueError:
        items = []
    specs, invalid = [], 0
    for i, item in enumerate(items if isinstance(items, list) else []):
        if not isinstance(item, dict):
            invalid += 1
            continue
        named = {**item, "name": f"{name_prefix}_{i}", "dsl_version": 1}
        try:
            specs.append(validator.validate(named))
        except ValueError:
            invalid += 1
    return Proposal(
        specs,
        len(items) if isinstance(items, list) else 0,
        invalid,
        reply.dollars,
        reply.prompt_tokens,
        reply.completion_tokens,
    )
