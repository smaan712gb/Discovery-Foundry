"""Meta layer: promotion rule, proposers, LLM cache, tournaments (fake and real runs)."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from foundry.cli import app
from foundry.core.budgets import BudgetConfig, RunBudget
from foundry.core.config import load_model
from foundry.core.results import ResultsStore
from foundry.domains.nq.build import run_build
from foundry.domains.nq.config import FoundryConfig
from foundry.domains.nq.meta_runner import make_run_fn, validation_folds
from foundry.meta.config import MetaConfig
from foundry.meta.promotion import RunOutcome, decide, paired_wilcoxon_greater
from foundry.meta.propose import (
    ProposalError,
    engine_id,
    propose_by_code,
    propose_by_llm,
    write_engine,
)
from foundry.meta.tournament import run_tournament
from foundry.search.engine import EngineConfig, load_engine
from foundry.search.llm_client import LLMCache, LLMClient, LLMConfig
from tests.conftest import META_CFG, REPO, SEARCH_CFG, write_config

ENGINE_V0 = REPO / "config" / "engines" / "engine_v0.yaml"


def meta_cfg(**over: Any) -> MetaConfig:
    return MetaConfig.model_validate(META_CFG | over)


def outcome(eid: str, seed: int, score: float, **over: Any) -> RunOutcome:
    base = RunOutcome(
        eid, seed, f"{eid}-{seed}", "completed", score, int(score), 10, 1.0, "c", ("h",)
    )
    return replace(base, **over)


# ---- promotion ---------------------------------------------------------------------------


def test_wilcoxon_edges() -> None:
    assert paired_wilcoxon_greater([0, 0, 0], [0, 0, 0]) == 1.0
    assert paired_wilcoxon_greater([2] * 6, [1] * 6) == pytest.approx(1 / 64)
    assert paired_wilcoxon_greater([2, 0, 2, 0, 2, 0], [1, 1, 1, 1, 1, 1]) > 0.05


def test_decide_requires_every_constraint() -> None:
    a = [outcome("A", s, 1.0) for s in range(6)]
    b = [outcome("B", s, 3.0) for s in range(6)]
    assert decide(a, b, 0.05, 1.5, reproducible=True).promote
    assert not decide(a, b, 0.05, 1.5, reproducible=False).promote
    slow = [replace(x, wall_clock=5.0) for x in b]
    assert "runtime" in decide(a, slow, 0.05, 1.5, True).reason
    err = [replace(b[0], status="error:boom"), *b[1:]]
    assert "budget" in decide(a, err, 0.05, 1.5, True).reason
    other_critics = [replace(b[0], critic_config_hash="z"), *b[1:]]
    assert "critic_config" in decide(a, other_critics, 0.05, 1.5, True).reason
    ties = [outcome("B", s, 1.0) for s in range(6)]
    d = decide(a, ties, 0.05, 1.5, True)
    assert not d.promote and d.p_value == 1.0 and "significant" in d.reason


# ---- proposers ---------------------------------------------------------------------------


def test_code_proposer_is_valid_and_deterministic() -> None:
    parent, pid = load_engine(ENGINE_V0)
    a = propose_by_code(parent, "v1", meta_cfg(), seed=4, allow_llm=True)
    b = propose_by_code(parent, "v1", meta_cfg(), seed=4, allow_llm=True)
    c = propose_by_code(parent, "v1", meta_cfg(), seed=5, allow_llm=True)
    assert a.engine_id == b.engine_id != c.engine_id != pid
    assert abs(sum(a.cfg.mix.model_dump().values()) - 1) < 1e-9
    no_llm = propose_by_code(parent, "v1", meta_cfg(), seed=4, allow_llm=False)
    assert no_llm.cfg.mix.llm == 0.0


def _llm(content: dict[str, Any]) -> LLMClient:
    def transport(
        url: str, headers: dict[str, str], body: bytes, timeout: float
    ) -> tuple[int, bytes]:
        return 200, json.dumps(
            {
                "choices": [{"message": {"content": json.dumps(content)}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 50},
            }
        ).encode()

    budget = RunBudget(
        BudgetConfig(wall_clock_seconds=60, max_evaluations=1, llm_dollars=1, llm_tokens=10**6)
    )
    return LLMClient(LLMConfig.model_validate(SEARCH_CFG["llm"]), budget, transport)


def test_llm_engine_proposals_are_validated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FOUNDRY_TEST_LLM_KEY", "k")
    parent, _ = load_engine(ENGINE_V0)
    good = parent.model_dump(mode="json") | {"batch_size": 80}
    p = propose_by_llm(
        parent, "v1", [], _llm({"engine": good, "rationale": "bigger batches"}), tmp_path
    )
    assert p.cfg.batch_size == 80 and p.proposer == "llm" and p.cfg.label == "v1"
    bad = parent.model_dump(mode="json") | {"batch_size": -3}
    with pytest.raises(ProposalError, match="invalid"):
        propose_by_llm(parent, "v1", [], _llm({"engine": bad}), tmp_path)
    with pytest.raises(ProposalError, match="placeholder"):
        propose_by_llm(
            parent, "v1", [], _llm({"engine": good, "prompt_text": "no slots"}), tmp_path
        )
    text = "Improved prompt {{DSL}} {{FEEDBACK}} {{N}}"
    p2 = propose_by_llm(parent, "v2", [], _llm({"engine": good, "prompt_text": text}), tmp_path)
    out = write_engine(p2, tmp_path / "engines")
    assert Path(p2.cfg.llm.prompt_template).read_text(encoding="utf-8") == text
    with pytest.raises(ProposalError, match="overwrite"):
        write_engine(p2, tmp_path / "engines")
    assert load_engine(out)[1] == p2.engine_id


def test_llm_cache_replays_without_cost(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FOUNDRY_TEST_LLM_KEY", "k")
    calls = []

    def transport(
        url: str, headers: dict[str, str], body: bytes, timeout: float
    ) -> tuple[int, bytes]:
        calls.append(1)
        return 200, json.dumps(
            {
                "choices": [{"message": {"content": '{"specs": []}'}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            }
        ).encode()

    cache = LLMCache(tmp_path / "c.duckdb")
    b = RunBudget(
        BudgetConfig(wall_clock_seconds=60, max_evaluations=1, llm_dollars=1, llm_tokens=10**6)
    )
    client = LLMClient(LLMConfig.model_validate(SEARCH_CFG["llm"]), b, transport, cache)
    first = client.complete_json("s", "u")
    second = client.complete_json("s", "u")
    cache.close()
    assert len(calls) == 1 and second.content == first.content and second.dollars == 0.0


# ---- tournaments -------------------------------------------------------------------------


def test_tournament_with_fake_runs_promotes_and_rotates_folds(tmp_path: Path) -> None:
    champ = load_engine(ENGINE_V0)
    chal_cfg = champ[0].model_copy(update={"label": "v1", "batch_size": 99})
    chal = (chal_cfg, engine_id(chal_cfg))
    folds_seen = []

    def run_fn(cfg: EngineConfig, eid: str, seed: int, path: Path, fold: int) -> RunOutcome:
        folds_seen.append(fold)
        return outcome(eid, seed, 3.0 if eid == chal[1] else 1.0)

    meta = meta_cfg(seeds=6, wilcoxon_alpha=0.05)
    with ResultsStore(tmp_path / "reg.duckdb") as reg:
        reg.set_champion(champ[1], None)
        r1 = run_tournament(meta, champ, chal, run_fn, reg, tmp_path / "ws")
        assert r1.decision.promote and reg.champion() == chal[1]
        assert r1.reproducible
        r2 = run_tournament(meta, chal, champ, run_fn, reg, tmp_path / "ws")
        assert not r2.decision.promote and reg.champion() == chal[1]
        assert reg.validation_touches() == {"fold_0": 1, "fold_1": 1, "total": 2}
        hist = reg.tournament_history()
        assert [h["promoted"] for h in hist] == [True, False]
    assert set(folds_seen) == {0, 1}


@pytest.fixture
def built(tmp_path: Path, synthetic_raw: Path) -> FoundryConfig:
    cfg_path = write_config(tmp_path, synthetic_raw)
    assert run_build(cfg_path, REPO).ok
    return load_model(cfg_path, FoundryConfig)


def test_real_tournament_mechanics(built: FoundryConfig, tmp_path: Path) -> None:
    """Real searches: isolated stores, equal budgets, the reproducibility re-run, full records."""
    cfg = built
    folds = validation_folds(tmp_path / "processed", cfg.meta.validation_folds)
    assert len(folds) == 2 and folds[0][1] < folds[1][0]
    e = yaml.safe_load(ENGINE_V0.read_text(encoding="utf-8"))
    e.update(
        {
            "generations": 3,
            "batch_size": 12,
            "top_k": 2,
            "mix": {"random": 0.5, "genetic": 0.5, "cma": 0.0, "llm": 0.0},
        }
    )
    e["llm"]["prompt_template"] = str(REPO / "config" / "prompts" / "proposer_v1.md")
    champ_cfg = EngineConfig.model_validate(e)
    chal_cfg = champ_cfg.model_copy(update={"label": "v1", "batch_size": 8})
    run_fn = make_run_fn(cfg, REPO, cfg.meta, use_llm=False, llm_cache=None)
    with ResultsStore(tmp_path / "reg.duckdb") as reg:
        reg.set_champion(engine_id(champ_cfg), None)
        res = run_tournament(
            cfg.meta,
            (champ_cfg, engine_id(champ_cfg)),
            (chal_cfg, engine_id(chal_cfg)),
            run_fn,
            reg,
            tmp_path / "ws",
        )
        assert res.reproducible, "re-running a seed must evaluate the same specs"
        assert len(res.champion_runs) == len(res.challenger_runs) == cfg.meta.seeds
        for r in res.champion_runs + res.challenger_runs:
            assert r.status == "completed" or r.status.startswith("stopped:")
            assert r.evaluated <= cfg.meta.run_budget.max_evaluations
        # isolated stores: one per run plus the re-run
        assert len(list((tmp_path / "ws").rglob("*.duckdb"))) == 2 * cfg.meta.seeds + 1
        assert reg.tournament_count() == 1
        assert res.decision.constraints["critic_config"]["passed"]


def test_cli_propose_and_status(
    built: FoundryConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Run from a scratch copy of the repo layout so no real engine or prompt files are written.
    work = tmp_path / "work"
    (work / "config" / "engines").mkdir(parents=True)
    (work / "config" / "prompts").mkdir(parents=True)
    (work / "config" / "engines" / "engine_v0.yaml").write_text(
        ENGINE_V0.read_text("utf-8"), "utf-8"
    )
    cfg_path = tmp_path / "config.yaml"
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["search"]["engine_file"] = str(work / "config" / "engines" / "engine_v0.yaml")
    cfg_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    monkeypatch.chdir(work)
    cli = CliRunner()
    res = cli.invoke(app, ["meta", "propose", "--config", str(cfg_path), "--seed", "3"])
    assert res.exit_code == 0, res.output
    assert (work / "config" / "engines" / "engine_v1.yaml").exists()
    status = json.loads(cli.invoke(app, ["meta", "status", "--config", str(cfg_path)]).output)
    assert status["champion"].startswith("v0-")
    assert status["validation_touches"] == {"total": 0}
