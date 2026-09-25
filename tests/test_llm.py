"""LLM client and proposer against a fake transport: budgets, redaction, parsing. No network."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from foundry.core.budgets import BudgetConfig, RunBudget
from foundry.core.secrets import load_dotenv, redact
from foundry.domains.nq.space import NQSpecSpace
from foundry.search.llm_client import LLMClient, LLMConfig, LLMError
from foundry.search.llm_proposer import propose
from tests.conftest import REPO, SEARCH_CFG

KEY = "sk-test-SECRET-123"
BASELINE = json.loads((REPO / "strategies" / "baseline_rth_orb.json").read_text(encoding="utf-8"))


def cfg() -> LLMConfig:
    return LLMConfig.model_validate(SEARCH_CFG["llm"])


def budget(dollars: float = 1.0, tokens: int = 1_000_000) -> RunBudget:
    return RunBudget(
        BudgetConfig(
            wall_clock_seconds=60, max_evaluations=10, llm_dollars=dollars, llm_tokens=tokens
        )
    )


class Fake:
    def __init__(self, replies: list[tuple[int, dict[str, Any] | str]]) -> None:
        self.replies = replies
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self, url: str, headers: dict[str, str], body: bytes, timeout: float
    ) -> tuple[int, bytes]:
        self.calls.append({"url": url, "headers": headers, "body": json.loads(body)})
        status, payload = self.replies.pop(0)
        raw = payload if isinstance(payload, str) else json.dumps(payload)
        return status, raw.encode()


def ok(content: dict[str, Any], pt: int = 1000, ct: int = 500) -> tuple[int, dict[str, Any]]:
    return 200, {
        "choices": [{"message": {"content": json.dumps(content)}}],
        "usage": {"prompt_tokens": pt, "completion_tokens": ct},
    }


@pytest.fixture(autouse=True)
def key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FOUNDRY_TEST_LLM_KEY", KEY)


def test_request_shape_and_charging() -> None:
    fake = Fake([ok({"specs": []}, pt=10_000, ct=2_000)])
    b = budget()
    client = LLMClient(cfg(), b, fake)
    reply = client.complete_json("sys", "please reply in json")
    call = fake.calls[0]
    assert call["url"] == "https://llm.invalid/chat/completions"
    assert call["headers"]["Authorization"] == f"Bearer {KEY}"
    assert call["body"]["response_format"] == {"type": "json_object"}
    assert call["body"]["model"] == "test-model"
    expected = (10_000 * 0.30 + 2_000 * 1.20) / 1e6
    assert reply.dollars == pytest.approx(expected)
    assert b.llm_dollars == pytest.approx(expected) and b.llm_tokens == 12_000


def test_call_refused_when_it_could_break_the_budget() -> None:
    fake = Fake([])
    client = LLMClient(cfg(), budget(dollars=0.001), fake)
    with pytest.raises(LLMError, match="budget"):
        client.complete_json("s", "u" * 1000)
    assert fake.calls == []  # nothing was sent


def test_key_never_appears_in_errors() -> None:
    fake = Fake([(401, f"invalid key {KEY}")])
    client = LLMClient(cfg(), budget(), fake)
    with pytest.raises(LLMError) as exc:
        client.complete_json("s", "u")
    assert KEY not in str(exc.value)
    assert "[REDACTED]" in str(exc.value)


def test_retries_then_succeeds() -> None:
    c = cfg().model_copy(update={"max_retries": 2})
    fake = Fake([(503, "busy"), ok({"specs": []})])
    LLMClient(c, budget(), fake).complete_json("s", "u")
    assert len(fake.calls) == 2


def test_proposer_validates_every_spec() -> None:
    good = {k: v for k, v in BASELINE.items() if k != "name"}
    bad_units = json.loads(json.dumps(good))
    bad_units["params"]["lvl"] = {"type": "float", "low": 1, "high": 2, "value": 1.5}
    bad_units["entry"]["long"]["all"].append(
        {"op": ">", "left": {"price": "close"}, "right": {"param": "lvl"}}
    )
    fake = Fake([ok({"specs": [good, bad_units, "not a spec", {"code": "import os"}]})])
    prop = propose(LLMClient(cfg(), budget(), fake), NQSpecSpace(), "json please", "llm_t")
    assert prop.returned == 4 and prop.invalid == 3 and len(prop.specs) == 1
    assert prop.specs[0].name == "llm_t_0"


def test_proposer_survives_garbage_and_errors() -> None:
    fake = Fake([(200, {"choices": [{"message": {"content": "no json here"}}], "usage": {}})])
    prop = propose(LLMClient(cfg(), budget(), fake), NQSpecSpace(), "p", "x")
    assert prop.specs == [] and prop.error is None
    fake2 = Fake([(400, "bad request")])
    prop2 = propose(LLMClient(cfg(), budget(), fake2), NQSpecSpace(), "p", "x")
    assert prop2.error is not None


def test_dotenv_does_not_override_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = tmp_path / ".env"
    env.write_text(
        "FOUNDRY_TEST_LLM_KEY=from-file\nFOUNDRY_NEW_VAR='x y'\n# comment\n", encoding="utf-8"
    )
    monkeypatch.delenv("FOUNDRY_NEW_VAR", raising=False)
    names = load_dotenv(env)
    assert os.environ["FOUNDRY_TEST_LLM_KEY"] == KEY  # environment wins
    assert os.environ["FOUNDRY_NEW_VAR"] == "x y"
    assert names == ["FOUNDRY_NEW_VAR"]
    monkeypatch.delenv("FOUNDRY_NEW_VAR")
    assert redact(f"a {KEY} b", KEY) == "a [REDACTED] b"


def test_only_one_module_touches_the_network() -> None:
    offenders = []
    for path in (REPO / "foundry").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if any(
            tok in text
            for tok in (
                "urllib.request",
                "import requests",
                "import httpx",
                "http.client",
                "import socket",
            )
        ):
            offenders.append(path.name)
    assert offenders == ["llm_client.py"]


def test_thinking_field_and_reasoning_overrun() -> None:
    fake = Fake([ok({"specs": []})])
    LLMClient(cfg().model_copy(update={"thinking": "low"}), budget(), fake).complete_json("s", "u")
    assert fake.calls[0]["body"]["thinking"] == {"type": "enabled", "reasoning_effort": "low"}
    fake = Fake([ok({"specs": []})])
    LLMClient(cfg().model_copy(update={"thinking": "disabled"}), budget(), fake).complete_json(
        "s", "u"
    )
    assert fake.calls[0]["body"]["thinking"] == {"type": "disabled"}
    fake = Fake([ok({"specs": []})])
    LLMClient(cfg(), budget(), fake).complete_json("s", "u")
    assert "thinking" not in fake.calls[0]["body"]  # omitted for other providers
    # All output spent on reasoning: a clear error, and the tokens are still charged.
    overrun = (
        200,
        {
            "choices": [
                {"message": {"content": "", "reasoning_content": "..."}, "finish_reason": "length"}
            ],
            "usage": {"prompt_tokens": 100, "completion_tokens": 2000},
        },
    )
    b = budget()
    with pytest.raises(LLMError, match="max_output_tokens"):
        LLMClient(cfg(), b, Fake([overrun])).complete_json("s", "u")
    assert b.llm_tokens == 2100
