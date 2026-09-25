"""The only module that talks to an LLM provider (ADR 0009).

Provider-agnostic: any OpenAI-compatible chat-completions endpoint (DeepSeek by default). The
API key comes from the environment, is sent only in the Authorization header, and is redacted
from every error.
Each call is charged against the run budget at the configured (peak) price *before* its reply is
used; a call whose worst case would break the budget is never sent.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import duckdb
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
    # Provider reasoning ("thinking") mode. None omits the field for providers without it.
    # Reasoning tokens count as output tokens and are charged like them.
    thinking: Literal["disabled", "low", "high", "max"] | None = None


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


class LLMCache:
    """Replies keyed by the exact request body, so a re-run replays identical LLM output (ADR 0010).

    A cache hit costs nothing and sends nothing. Stored: request hash and reply text only.
    """

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._con = duckdb.connect(str(path))
        self._con.execute(
            "CREATE TABLE IF NOT EXISTS llm_cache (key VARCHAR PRIMARY KEY, "
            "content VARCHAR NOT NULL,"
            " prompt_tokens INTEGER NOT NULL, completion_tokens INTEGER NOT NULL)"
        )

    def get(self, key: str) -> tuple[str, int, int] | None:
        row = self._con.execute(
            "SELECT content, prompt_tokens, completion_tokens FROM llm_cache WHERE key = ?", [key]
        ).fetchone()
        return (str(row[0]), int(row[1]), int(row[2])) if row else None

    def put(self, key: str, content: str, prompt_tokens: int, completion_tokens: int) -> None:
        self._con.execute(
            "INSERT INTO llm_cache VALUES (?, ?, ?, ?) ON CONFLICT DO NOTHING",
            [key, content, prompt_tokens, completion_tokens],
        )

    def close(self) -> None:
        self._con.close()


class LLMClient:
    def __init__(
        self,
        cfg: LLMConfig,
        budget: RunBudget,
        transport: Transport | None = None,
        cache: LLMCache | None = None,
    ) -> None:
        self.cfg = cfg
        self.budget = budget
        self._transport = transport or _urllib_transport
        self.cache = cache

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
        request: dict[str, Any] = {
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
        if self.cfg.thinking == "disabled":
            request["thinking"] = {"type": "disabled"}
        elif self.cfg.thinking is not None:
            request["thinking"] = {"type": "enabled", "reasoning_effort": self.cfg.thinking}
        body = json.dumps(request, sort_keys=True).encode("utf-8")
        cache_key = hashlib.sha256(body).hexdigest()
        if self.cache is not None and (hit := self.cache.get(cache_key)) is not None:
            return LLMReply(hit[0], hit[1], hit[2], 0.0)
        key = require(self.cfg.api_key_env)
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
        url = str(self.cfg.base_url).rstrip("/") + "/chat/completions"
        last = ""
        for attempt in range(self.cfg.max_retries + 1):
            try:
                status, raw = self._transport(url, headers, body, self.cfg.timeout_seconds)
            except (OSError, TimeoutError) as exc:
                status, raw = 0, redact(str(exc), key).encode()
            if status == 200:
                reply = self._parse(raw)
                if self.cache is not None:
                    self.cache.put(
                        cache_key, reply.content, reply.prompt_tokens, reply.completion_tokens
                    )
                return reply
            last = redact(raw.decode("utf-8", "replace")[:300], key)
            if status in (400, 401, 402, 403, 422):
                break  # not retryable
            time.sleep(min(2**attempt, 10))
        raise LLMError(f"LLM request failed (status {status}): {last}")

    def _parse(self, raw: bytes) -> LLMReply:
        try:
            data: dict[str, Any] = json.loads(raw)
            choice = data["choices"][0]
            content = str(choice["message"].get("content") or "")
            usage = data.get("usage", {})
            pt, ct = int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"unexpected LLM response shape: {exc}") from exc
        dollars = self.price(pt, ct)
        self.budget.charge_llm(dollars, pt + ct)  # charged even if the reply is unusable
        if choice.get("finish_reason") == "length" and not content.strip():
            raise LLMError(
                f"reply hit max_output_tokens ({self.cfg.max_output_tokens}) before any answer; "
                "reasoning used the whole budget (lower `thinking` or raise max_output_tokens)"
            )
        return LLMReply(content, pt, ct, dollars)
