"""Evaluator end to end on a synthetic data build: splits, costs, contracts, results store."""

from __future__ import annotations

import json
from dataclasses import replace as dc_replace
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from typer.testing import CliRunner

from foundry.cli import app
from foundry.core.config import load_model
from foundry.core.holdout import HoldoutError, HoldoutVault
from foundry.core.metrics import max_drawdown, sharpe, sortino
from foundry.core.results import ResultsStore
from foundry.domains.nq.build import run_build
from foundry.domains.nq.calendar import TradingCalendar
from foundry.domains.nq.config import EvaluatorConfig, FoundryConfig
from foundry.domains.nq.dsl import load_spec
from foundry.domains.nq.evaluator import EvaluationError, NQEvaluator, evaluate_context
from foundry.domains.nq.features import build_context
from foundry.domains.nq.loaders import TS_DTYPE
from foundry.domains.nq.quality import flag_bars
from foundry.domains.nq.rolls import build_continuous
from foundry.domains.nq.sessions import annotate_sessions
from foundry.domains.nq.synthetic import generate
from tests.conftest import (
    CALENDAR_FILE,
    EVALUATOR_CFG,
    QUALITY,
    REPO,
    SESSIONS,
    SYN_SPEC,
    write_config,
)

BASELINE = REPO / "strategies" / "baseline_rth_orb.json"


@pytest.fixture
def built(tmp_path: Path, synthetic_raw: Path) -> tuple[Path, FoundryConfig]:
    cfg_path = write_config(tmp_path, synthetic_raw)
    assert run_build(cfg_path, REPO).ok
    return cfg_path, load_model(cfg_path, FoundryConfig)


def test_search_evaluation(built: tuple[Path, FoundryConfig]) -> None:
    _, cfg = built
    ev = NQEvaluator(cfg, REPO)
    r = ev.evaluate(load_spec(BASELINE), "search")
    m = r.metrics
    assert m["trade_count"] > 20
    assert m["net_pnl"] < m["gross_pnl"]  # costs are always on
    assert r.trades["entry_ts"].dt.date().max() <= date(2024, 2, 15)
    assert set(r.trades["entry_session"].unique().to_list()) == {"RTH"}
    # every trade pays exactly one round-turn commission
    assert (r.trades["commission"] == 4.50).all()
    # daily PnL sums to net PnL and covers every search trading day
    assert r.daily["net_pnl"].sum() == pytest.approx(m["net_pnl"])
    assert r.daily.height == m["trading_days"]


def test_validation_trades_only_on_validation_days(built: tuple[Path, FoundryConfig]) -> None:
    _, cfg = built
    ev = NQEvaluator(cfg, REPO)
    r = ev.evaluate(load_spec(BASELINE), "validation")
    first = r.trades["signal_ts"].dt.date().min()
    assert first is not None and first >= date(2024, 2, 16)
    assert r.daily["trading_date"].min() >= date(2024, 2, 16)
    assert r.daily["trading_date"].max() <= date(2024, 3, 31)


def test_holdout_is_not_evaluable_and_untouched(
    built: tuple[Path, FoundryConfig], tmp_path: Path
) -> None:
    _, cfg = built
    ev = NQEvaluator(cfg, REPO)
    with pytest.raises(EvaluationError, match="cannot be evaluated"):
        ev.evaluate(load_spec(BASELINE), "holdout")
    ev.evaluate(load_spec(BASELINE), "search")
    ev.evaluate(load_spec(BASELINE), "validation")
    assert HoldoutVault(tmp_path / "sealed").access_log() == []


def test_mnq_is_one_tenth_of_nq_before_commission(built: tuple[Path, FoundryConfig]) -> None:
    _, cfg = built
    ev = NQEvaluator(cfg, REPO)
    spec = load_spec(BASELINE)
    nq = ev.evaluate(spec, "search", "NQ")
    mnq = ev.evaluate(spec, "search", "MNQ")
    assert nq.metrics["trade_count"] == mnq.metrics["trade_count"]
    assert mnq.metrics["gross_pnl"] == pytest.approx(nq.metrics["gross_pnl"] / 10)
    n = nq.metrics["trade_count"]
    slip_nq = nq.metrics["gross_pnl"] - nq.metrics["net_pnl"] - 4.50 * n
    slip_mnq = mnq.metrics["gross_pnl"] - mnq.metrics["net_pnl"] - 1.50 * n
    assert slip_mnq == pytest.approx(slip_nq / 10)


def test_cost_stress_lowers_pnl(built: tuple[Path, FoundryConfig]) -> None:
    _, cfg = built
    ev = NQEvaluator(cfg, REPO)
    spec = load_spec(BASELINE)
    base = ev.evaluate(spec, "search")
    stressed = ev.evaluate(spec, "search", slippage_multiplier=2.0)
    assert stressed.metrics["net_pnl"] < base.metrics["net_pnl"]
    assert stressed.metrics["gross_pnl"] == pytest.approx(base.metrics["gross_pnl"])


def test_results_store_caches_and_counts_trials(
    built: tuple[Path, FoundryConfig], tmp_path: Path
) -> None:
    _, cfg = built
    ev = NQEvaluator(cfg, REPO)
    spec = load_spec(BASELINE)
    with ResultsStore(tmp_path / "r.duckdb") as store:
        first = ev.evaluate(spec, "search", store=store)
        again = ev.evaluate(spec, "search", store=store)
        assert again.metrics == first.metrics
        assert again.trades.height == first.trades.height
        renamed = spec.with_params({"stop_pts": 30})
        ev.evaluate(renamed, "search", store=store)
        assert store.trial_count("search") == 2
        assert store.trial_count("validation") == 0


def test_evaluator_refuses_failed_or_tampered_builds(
    built: tuple[Path, FoundryConfig], tmp_path: Path
) -> None:
    _, cfg = built
    series = tmp_path / "processed" / "bars.parquet"
    pl.read_parquet(series).head(10).write_parquet(series)
    with pytest.raises(EvaluationError, match="does not match"):
        NQEvaluator(cfg, REPO)


def test_cli_spec_and_evaluate(
    built: tuple[Path, FoundryConfig], monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg_path, _ = built
    monkeypatch.chdir(REPO)
    runner = CliRunner()
    ok = runner.invoke(app, ["spec", "validate", str(BASELINE)])
    assert ok.exit_code == 0 and "valid" in ok.output
    res = runner.invoke(
        app, ["evaluate", str(BASELINE), "--split", "search", "--config", str(cfg_path)]
    )
    assert res.exit_code == 0, res.output
    out = json.loads(res.output)
    assert out["metrics"]["trade_count"] > 0


def test_metric_formulas() -> None:
    daily = np.array([100.0, -50.0, 200.0, -100.0, 50.0])
    mean, sd = daily.mean(), daily.std(ddof=1)
    assert sharpe(daily, 252) == pytest.approx(mean / sd * np.sqrt(252))
    dd = np.sqrt(np.mean(np.minimum(daily, 0) ** 2))
    assert sortino(daily, 252) == pytest.approx(mean / dd * np.sqrt(252))
    # equity 0,100,50,250,150,200 -> worst drawdown 250 -> 150
    assert max_drawdown(daily) == 100.0
    assert sharpe(np.array([5.0]), 252) == 0.0
    assert max_drawdown(np.array([-10.0, -5.0])) == 15.0


def test_holdout_open_is_logged_once_per_candidate(
    built: tuple[Path, FoundryConfig], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg_path, cfg = built
    ev = NQEvaluator(cfg, REPO)
    spec = load_spec(BASELINE)
    with ResultsStore(tmp_path / "results.duckdb") as store:
        ev.evaluate(spec, "search", store=store)
        (r,) = ev.evaluate_holdout([spec], "engine-0", "test opening", store)
        assert r.split == "holdout"
        assert r.trades["signal_ts"].dt.date().min() >= date(2024, 4, 1)
        assert r.daily["trading_date"].min() >= date(2024, 4, 1)
        vault = HoldoutVault(tmp_path / "sealed")
        assert len(vault.access_log()) == 1
        with pytest.raises(HoldoutError, match="already evaluated"):
            ev.evaluate_holdout([spec], "engine-1", "second try", store)
        assert len(vault.access_log()) == 1
    # CLI: an unknown candidate is refused before the vault is touched
    monkeypatch.chdir(REPO)
    res = CliRunner().invoke(
        app,
        [
            "holdout",
            "open",
            "--engine",
            "e",
            "--candidates",
            "deadbeef",
            "--reason",
            "r",
            "--config",
            str(cfg_path),
        ],
    )
    assert res.exit_code == 1
    assert "unknown candidate" in res.output
    assert len(HoldoutVault(tmp_path / "sealed").access_log()) == 1


def _synthetic_ctx(spec_kwargs: dict[str, int]) -> tuple[object, object]:

    cal = TradingCalendar.from_file(CALENDAR_FILE, SESSIONS)
    data = generate(dc_replace(SYN_SPEC, **spec_kwargs), cal)
    bars = pl.concat(
        [
            f.with_columns(pl.lit(c).alias("contract"), pl.lit("s").alias("source_file"))
            for c, f in data.frames.items()
        ]
    ).with_columns(pl.col("ts").cast(pl.Datetime("us", "America/New_York")))
    bars = bars.with_columns(pl.col("ts").cast(TS_DTYPE))
    series = build_continuous(
        flag_bars(annotate_sessions(bars, cal), 0.25, 1, QUALITY), SYN_SPEC.end
    ).series
    ev = EvaluatorConfig.model_validate(EVALUATOR_CFG)
    return build_context(series, cal, 1, ev.regimes, set()), ev


def test_evaluator_finds_a_planted_edge_and_no_edge_in_noise() -> None:
    """The backtester must not invent profit from a random walk, and must find a real edge."""
    spec = load_spec(BASELINE)
    noise_ctx, ev = _synthetic_ctx({})
    edge_ctx, _ = _synthetic_ctx({"orb_drift_ticks": 1})
    noise = evaluate_context(spec, noise_ctx, ev, "NQ", "search")  # type: ignore[arg-type]
    edge = evaluate_context(spec, edge_ctx, ev, "NQ", "search")  # type: ignore[arg-type]
    assert noise.metrics["net_pnl"] < 0  # no edge: costs win
    assert edge.metrics["net_pnl"] > 0  # planted opening-range momentum is found
    assert edge.metrics["sharpe"] > 2
    assert edge.metrics["win_rate"] > noise.metrics["win_rate"]
