"""The only module that talks to an LLM provider (ADR 0009).

Provider-agnostic: any OpenAI-compatible chat-completions endpoint (DeepSeek by default). The
API key comes from the environment, is sent only in the Authorization header, and is redacted
from every error.
Each call is charged against the run budget at the configured (peak) price *before* its reply is
used; a call whose worst case would break the budget is never sent.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import Field, HttpUrl

from foundry.core.budgets import RunBudget
from foundry.core.config import StrictModel
from foundry.core.secrets import redact, require


class LLMConfig(StrictModel):
    base_url: HttpUrl
    model: str
    api_key_env: str
    price_input_per_mtok: float = Field(ge=0)  # charge every input token at the cache-miss peak
    price_output_per_mtok: float = Field(ge=0)
    max_output_tokens: int = Field(ge=64)
    temperature: float = Field(ge=0, le=2)
    timeout_seconds: float = Field(gt=0)
    max_retries: int = Field(ge=0, le=5)


class LLMError(Exception):
    pass


@dataclass(frozen=True)
class LLMReply:
    content: str
    prompt_tokens: int
    completion_tokens: int
    dollars: float


# (url, headers, body bytes, timeout) -> (status, response bytes). Replaced by a fake in tests.
Transport = Callable[[str, dict[str, str], bytes, float], tuple[int, bytes]]


def _urllib_transport(
    url: str, headers: dict[str, str], body: bytes, timeout: float
) -> tuple[int, bytes]:
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return int(resp.status), resp.read()
    except urllib.error.HTTPError as exc:
        return int(exc.code), exc.read()


class LLMClient:
    def __init__(
        self, cfg: LLMConfig, budget: RunBudget, transport: Transport | None = None
    ) -> None:
        self.cfg = cfg
        self.budget = budget
        self._transport = transport or _urllib_transport

    def price(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (
            prompt_tokens * self.cfg.price_input_per_mtok
            + completion_tokens * self.cfg.price_output_per_mtok
        ) / 1e6

    def complete_json(self, system: str, user: str) -> LLMReply:
        """One JSON-mode chat completion. Raises LLMError; never leaks the key."""
        # Worst case: 3 characters per input token (real text averages about 4, so this
        # overestimates) plus a full-length reply.
        est_in = (len(system) + len(user)) // 3 + 50
        worst_tokens = est_in + self.cfg.max_output_tokens
        if not self.budget.can_afford_llm(
            self.price(est_in, self.cfg.max_output_tokens), worst_tokens
        ):
            raise LLMError("LLM budget would be exceeded by this call; not sent")
        key = require(self.cfg.api_key_env)
        body = json.dumps(
            {
                "model": self.cfg.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "response_format": {"type": "json_object"},
                "temperature": self.cfg.temperature,
                "max_tokens": self.cfg.max_output_tokens,
                "stream": False,
            }
        ).encode("utf-8")
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
        url = str(self.cfg.base_url).rstrip("/") + "/chat/completions"
        last = ""
        for attempt in range(self.cfg.max_retries + 1):
            try:
                status, raw = self._transport(url, headers, body, self.cfg.timeout_seconds)
            except (OSError, TimeoutError) as exc:
                status, raw = 0, redact(str(exc), key).encode()
            if status == 200:
                return self._parse(raw)
            last = redact(raw.decode("utf-8", "replace")[:300], key)
            if status in (400, 401, 402, 403, 422):
                break  # not retryable
            time.sleep(min(2**attempt, 10))
        raise LLMError(f"LLM request failed (status {status}): {last}")

    def _parse(self, raw: bytes) -> LLMReply:
        try:
            data: dict[str, Any] = json.loads(raw)
            content = str(data["choices"][0]["message"]["content"])
            usage = data.get("usage", {})
            pt, ct = int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"unexpected LLM response shape: {exc}") from exc
        dollars = self.price(pt, ct)
        self.budget.charge_llm(dollars, pt + ct)
        return LLMReply(content, pt, ct, dollars)
