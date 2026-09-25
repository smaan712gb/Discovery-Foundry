"""The search loop on synthetic data: budgets, dedupe, lineage, determinism, LLM path, critics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import duckdb
import pytest
import yaml

from foundry.core.budgets import RunBudget
from foundry.core.config import load_model
from foundry.core.results import ResultsStore
from foundry.critics.runner import CriticRunner
from foundry.domains.nq.build import run_build
from foundry.domains.nq.config import FoundryConfig
from foundry.domains.nq.evaluator import NQEvaluator
from foundry.domains.nq.space import NQSpecSpace
from foundry.search.engine import load_engine
from foundry.search.llm_client import LLMClient
from foundry.search.run import SearchRun, allocate, run_search, structure_key
from tests.conftest import REPO, SEARCH_CFG, write_config

BASELINE = json.loads((REPO / "strategies" / "baseline_rth_orb.json").read_text(encoding="utf-8"))


def small_engine(tmp: Path, **over: Any) -> Path:
    e = yaml.safe_load((REPO / "config" / "engines" / "engine_v0.yaml").read_text(encoding="utf-8"))
    e.update({"generations": 4, "batch_size": 16, "top_k": 3})
    e["surrogate"].update({"min_train": 20})
    e["llm"]["prompt_template"] = str(REPO / "config" / "prompts" / "proposer_v1.md")
    e.update(over)
    path = tmp / "engine.yaml"
    path.write_text(yaml.safe_dump(e), encoding="utf-8")
    return path


@pytest.fixture
def setup(tmp_path: Path, synthetic_raw: Path) -> tuple[FoundryConfig, NQEvaluator]:
    cfg_path = write_config(tmp_path, synthetic_raw)
    assert run_build(cfg_path, REPO).ok
    cfg = load_model(cfg_path, FoundryConfig)
    return cfg, NQEvaluator(cfg, REPO)


def make_run(
    cfg: FoundryConfig,
    ev: NQEvaluator,
    store: ResultsStore,
    engine: Path,
    seed: int = 1,
    max_evals: int = 200,
    llm: LLMClient | None = None,
    budget: RunBudget | None = None,
) -> SearchRun:
    ecfg, eid = load_engine(engine)
    b = budget or RunBudget(cfg.search.budgets.model_copy(update={"max_evaluations": max_evals}))
    return SearchRun(
        NQSpecSpace(),
        ev,
        store,
        ecfg,
        eid,
        b,
        seed,
        cfg.critics.min_trades.search,
        "NQ",
        llm,
        "test-model",
    )


def test_allocate_and_structure_key() -> None:
    assert allocate({"random": 0.25, "genetic": 0.45, "cma": 0.15, "llm": 0.15}, 20) == {
        "random": 5,
        "genetic": 9,
        "cma": 3,
        "llm": 3,
    }
    assert sum(allocate({"a": 1 / 3, "b": 1 / 3, "c": 1 / 3}, 10).values()) == 10
    space = NQSpecSpace()
    s = space.validate({**BASELINE})
    moved = s.with_params({"stop_pts": 40})
    assert structure_key(s) == structure_key(moved) and s.hash != moved.hash


def test_search_run_records_everything(
    setup: tuple[FoundryConfig, NQEvaluator], tmp_path: Path
) -> None:
    cfg, ev = setup
    engine = small_engine(tmp_path, mix={"random": 0.4, "genetic": 0.4, "cma": 0.2, "llm": 0.0})
    with ResultsStore(tmp_path / "r.duckdb") as store:
        run = make_run(cfg, ev, store, engine)
        critics = CriticRunner(ev, cfg.critics, store, "NQ", 252)
        res = run_search(run, critics)
        assert res.status == "completed"
        n = len(res.evaluated)
        assert 0 < n <= 4 * 16
        # every evaluated spec is a distinct stored trial; nothing evaluated twice
        assert store.trial_count("search") == n
        assert len({e.spec.hash for e in res.evaluated}) == n
        assert run.counts["evaluated"]["genetic"] > 0 and run.counts["evaluated"]["random"] > 0
        assert len(res.verdicts) <= 3 and store.funnel()["critiqued"] == len(res.verdicts)
        # the surrogate trained and its accuracy was logged once it could predict
        assert any(g["surrogate_spearman"] is not None for g in run.log[1:])
    con = duckdb.connect(str(tmp_path / "r.duckdb"), read_only=True)
    gens = dict(con.execute("SELECT generator, COUNT(*) FROM specs GROUP BY 1").fetchall())
    assert gens.get("genetic", 0) > 0
    parents = con.execute(
        "SELECT COUNT(*) FROM specs WHERE generator = 'genetic' AND parent_hashes IS NOT NULL"
    ).fetchone()
    assert parents is not None and parents[0] == gens["genetic"]
    status = con.execute("SELECT status, engine_id FROM runs").fetchone()
    assert status is not None and status[0] == "completed" and status[1].startswith("v0-")
    con.close()


def test_budget_stops_cleanly_and_critics_still_run(
    setup: tuple[FoundryConfig, NQEvaluator], tmp_path: Path
) -> None:
    cfg, ev = setup
    engine = small_engine(tmp_path, mix={"random": 1.0, "genetic": 0.0, "cma": 0.0, "llm": 0.0})
    with ResultsStore(tmp_path / "r.duckdb") as store:
        run = make_run(cfg, ev, store, engine, max_evals=10)
        res = run_search(run, CriticRunner(ev, cfg.critics, store, "NQ", 252))
        assert res.status == "stopped:evaluations"
        assert len(res.evaluated) == 10 and store.trial_count("search") == 10


def test_search_is_deterministic_without_llm(
    setup: tuple[FoundryConfig, NQEvaluator], tmp_path: Path
) -> None:
    cfg, ev = setup
    engine = small_engine(
        tmp_path, generations=2, mix={"random": 0.5, "genetic": 0.5, "cma": 0.0, "llm": 0.0}
    )
    hashes = []
    for i in range(2):
        with ResultsStore(tmp_path / f"r{i}.duckdb") as store:
            _, done = make_run(cfg, ev, store, engine, seed=9).run()
            hashes.append(sorted(e.spec.hash for e in done))
    assert hashes[0] == hashes[1]


def test_llm_generator_path(
    setup: tuple[FoundryConfig, NQEvaluator], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg, ev = setup
    monkeypatch.setenv("FOUNDRY_TEST_LLM_KEY", "sk-test")
    spec = {k: v for k, v in BASELINE.items() if k != "name"}
    prompts: list[str] = []

    def transport(
        url: str, headers: dict[str, str], body: bytes, timeout: float
    ) -> tuple[int, bytes]:
        prompts.append(json.loads(body)["messages"][1]["content"])
        content = json.dumps({"specs": [spec]})
        return 200, json.dumps(
            {
                "choices": [{"message": {"content": content}}],
                "usage": {"prompt_tokens": 3000, "completion_tokens": 400},
            }
        ).encode()

    engine = small_engine(
        tmp_path, generations=2, mix={"random": 0.5, "genetic": 0.25, "cma": 0.0, "llm": 0.25}
    )
    budget = RunBudget(
        cfg.search.budgets.model_copy(update={"llm_dollars": 0.10, "llm_tokens": 100_000})
    )
    llm = LLMClient(cfg.search.llm, budget, transport)
    with ResultsStore(tmp_path / "r.duckdb") as store:
        run = make_run(cfg, ev, store, engine, llm=llm, budget=budget)
        run.run()
        assert run.counts["evaluated"]["llm"] == 1  # same spec twice: evaluated once
        assert run.counts["duplicates"] >= 1
        assert 0 < budget.llm_dollars <= 0.10
    # The prompt carries the DSL and search-only feedback. Validation may appear only as
    # aggregate kill counts ("validation:<test>" keys), never as metrics.
    assert '"dsl_version"' in prompts[0] and '"trials_so_far"' in prompts[-1]
    feedback = json.loads(prompts[-1].split("counts)")[1].split("## Task")[0])
    assert set(feedback) == {
        "trials_so_far",
        "kill_counts_all_runs",
        "recently_killed_on_search",
        "best_this_run_on_search",
    }
    assert all(isinstance(v, int) for v in feedback["kill_counts_all_runs"].values())
    con = duckdb.connect(str(tmp_path / "r.duckdb"), read_only=True)
    calls = con.execute("SELECT COUNT(*), SUM(valid) FROM llm_calls").fetchone()
    assert calls is not None and calls[0] >= 1 and calls[1] >= 1
    con.close()


def test_llm_budget_zero_means_no_calls(
    setup: tuple[FoundryConfig, NQEvaluator], tmp_path: Path
) -> None:
    cfg, ev = setup
    engine = small_engine(tmp_path, generations=1)

    def transport(*_: Any) -> tuple[int, bytes]:
        raise AssertionError("no request may be sent with a zero LLM budget")

    budget = RunBudget(cfg.search.budgets)  # SEARCH_CFG: llm_dollars 0
    assert SEARCH_CFG["budgets"]["llm_dollars"] == 0.0  # type: ignore[index]
    with ResultsStore(tmp_path / "r.duckdb") as store:
        run = make_run(
            cfg, ev, store, engine, llm=LLMClient(cfg.search.llm, budget, transport), budget=budget
        )
        run.run()
        assert run.counts["evaluated"]["llm"] == 0
