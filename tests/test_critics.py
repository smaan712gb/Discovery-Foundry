"""Critic runner end to end: noise dies, a planted edge survives, lucky winners die."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from typer.testing import CliRunner

from foundry.cli import app
from foundry.core.config import load_model
from foundry.core.results import ResultsStore
from foundry.critics.runner import TEST_ORDER, CriticRunner, neighbours
from foundry.domains.nq.build import run_build
from foundry.domains.nq.config import FoundryConfig
from foundry.domains.nq.dsl import Spec, load_spec, validate_spec
from foundry.domains.nq.evaluator import NQEvaluator
from foundry.domains.nq.synthetic import generate, write_ninjatrader
from tests.conftest import CRITICS_CFG, REPO, SYN_SPEC, write_config
from tests.random_specs import random_spec

BASELINE = REPO / "strategies" / "baseline_rth_orb.json"


def build(
    tmp: Path, calendar: Any, critics: dict[str, Any] | None = None, **spec_kw: Any
) -> tuple[Path, FoundryConfig]:
    raw = tmp / "raw"
    write_ninjatrader(generate(replace(SYN_SPEC, **spec_kw), calendar).frames, raw)
    cfg_path = write_config(tmp, raw, critics=critics)
    assert run_build(cfg_path, REPO).ok
    return cfg_path, load_model(cfg_path, FoundryConfig)


def runner(cfg: FoundryConfig, store: ResultsStore) -> tuple[CriticRunner, NQEvaluator]:
    ev = NQEvaluator(cfg, REPO)
    return CriticRunner(ev, cfg.critics, store, "NQ", cfg.evaluator.annualization_days), ev


def test_random_walk_baseline_is_killed_and_recorded(tmp_path: Path, calendar: Any) -> None:
    _, cfg = build(tmp_path, calendar)
    with ResultsStore(tmp_path / "r.duckdb") as store:
        cr, _ = runner(cfg, store)
        v = cr.critique(load_spec(BASELINE), generator="manual")
        assert not v.alive
        assert (v.killed_stage, v.killed_by) == ("search", "min_trades_and_profit")
        assert [r.test for r in v.results] == ["min_trades_and_profit"]  # short-circuit
        f = store.funnel()
        assert f["critiqued"] == 1 and f["alive"] == 0
        assert f["killed"] == [{"stage": "search", "test": "min_trades_and_profit", "count": 1}]


def test_planted_edge_survives_every_test(tmp_path: Path, calendar: Any) -> None:
    # A realistic, imperfect edge: the momentum shows up on 70% of days only.
    _, cfg = build(tmp_path, calendar, orb_drift_ticks=1, edge_day_share=0.7)
    with ResultsStore(tmp_path / "r.duckdb") as store:
        cr, _ = runner(cfg, store)
        v = cr.critique(load_spec(BASELINE), generator="manual")
        assert v.alive, [(r.stage, r.test, r.numbers) for r in v.results if not r.passed]
        assert [(r.stage, r.test) for r in v.results] == [
            (s, t) for s in ("search", "validation") for t in TEST_ORDER
        ]
        # one hand-written spec is not a trial pool: PBO is honestly marked as not assessed
        assert "pbo_not_assessed" in v.labels
        # critic-internal evaluations (cost stress, neighbours) are not trials
        assert store.trial_count("search") == 1
        assert store.trial_count("validation") == 1
        assert store.funnel()["alive"] == 1


def test_lucky_winner_from_many_noise_trials_is_killed(tmp_path: Path, calendar: Any) -> None:
    """The multiple-testing trap: the best of many unrelated strategies on data with no edge
    looks profitable in-sample. The critics must still kill it."""
    _, cfg = build(tmp_path, calendar)
    rng = np.random.default_rng(5)
    with ResultsStore(tmp_path / "r.duckdb") as store:
        cr, ev = runner(cfg, store)
        best: tuple[float, Spec] | None = None
        tried = 0
        while tried < 300:
            s = random_spec(rng, tried)
            if s is None:
                continue
            tried += 1
            m = ev.evaluate(s, "search", store=store, generator="random").metrics
            if m["trade_count"] >= 30 and (best is None or m["net_pnl"] > best[0]):
                best = (m["net_pnl"], s)
        assert best is not None and best[0] > 0, "the lucky winner should look profitable"
        v = cr.critique(best[1], generator="random")
        assert not v.alive
        assert v.killed_by != "min_trades_and_profit" or v.killed_stage == "validation"
        assert store.trial_count("search") == 300


def test_regime_specific_label(tmp_path: Path, calendar: Any) -> None:
    # The filtered spec trades less, so trade minimums are lowered for this short sample.
    critics = CRITICS_CFG | {"min_trades": {"search": 10, "validation": 5}}
    _, cfg = build(tmp_path, calendar, critics=critics, orb_drift_ticks=1, edge_day_share=0.7)
    raw = json.loads(BASELINE.read_text(encoding="utf-8"))
    raw["filters"] = {"volatility_buckets": ["mid", "high"]}
    with ResultsStore(tmp_path / "r.duckdb") as store:
        cr, _ = runner(cfg, store)
        v = cr.critique(validate_spec(raw))
        regime = [r for r in v.results if r.test == "regime_stability"]
        assert regime and regime[0].numbers["label"] == "regime_specific"
        assert "regime_specific" in v.labels


def test_neighbours() -> None:
    spec = load_spec(BASELINE)
    ns = neighbours(spec, [-0.2, -0.1, 0.1, 0.2])
    moved = [{k: n.param(k) for k in spec.params() if n.param(k) != spec.param(k)} for n in ns]
    assert all(len(m) == 1 for m in moved)  # one parameter at a time
    assert {"or_minutes": 24} in moved and {"or_minutes": 36} in moved
    assert {"stop_pts": 30.0} in moved
    assert len({n.hash for n in ns}) == len(ns)
    for n in ns:  # all inside bounds
        for p in n.params().values():
            assert p["low"] <= p["value"] <= p["high"]


def test_cli_critique_and_funnel(
    tmp_path: Path, calendar: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg_path, _ = build(tmp_path, calendar)
    monkeypatch.chdir(REPO)
    cli = CliRunner()
    res = cli.invoke(app, ["critique", str(BASELINE), "--config", str(cfg_path)])
    assert res.exit_code == 0, res.output
    assert "DEAD (search: min_trades_and_profit)" in res.output
    f = cli.invoke(app, ["funnel", "--config", str(cfg_path)])
    assert json.loads(f.output)["critiqued"] == 1
