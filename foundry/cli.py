"""The `foundry` command line."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from foundry.core.budgets import RunBudget
from foundry.core.config import ConfigError, load_model
from foundry.core.holdout import HoldoutError, HoldoutVault
from foundry.core.results import ResultsStore
from foundry.core.secrets import SecretError, load_dotenv
from foundry.critics.runner import CriticRunner, clean_numbers
from foundry.domains.nq.build import run_build
from foundry.domains.nq.config import FoundryConfig
from foundry.domains.nq.dsl import SpecError, load_spec, validate_spec
from foundry.domains.nq.evaluator import EvaluationError, NQEvaluator
from foundry.domains.nq.space import NQSpecSpace
from foundry.search.engine import load_engine
from foundry.search.llm_client import LLMClient
from foundry.search.run import SearchRun, run_search

app = typer.Typer(no_args_is_help=True, add_completion=False)
data_app = typer.Typer(no_args_is_help=True, help="Data pipeline (phase 1).")
holdout_app = typer.Typer(
    no_args_is_help=True, help="Sealed holdout: status and the one logged way in."
)
app.add_typer(data_app, name="data")
app.add_typer(holdout_app, name="holdout")

ConfigOpt = Annotated[Path, typer.Option("--config", "-c", exists=True, dir_okay=False)]


@data_app.command("build")
def data_build(config: ConfigOpt = Path("config/v0.yaml")) -> None:
    """Build processed Parquet, seal the holdout and write the data report."""
    try:
        outcome = run_build(config, Path.cwd())
    except ConfigError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc
    for c in outcome.checks:
        mark = "ok  " if c.passed else "FAIL"
        typer.echo(f"[{mark}] {c.name}: {c.detail}")
    typer.echo(f"report: {outcome.processed_dir / 'data_report.md'}")
    if not outcome.ok:
        typer.echo("data build FAILED verification", err=True)
    raise typer.Exit(outcome.exit_code)


spec_app = typer.Typer(no_args_is_help=True, help="Strategy specs (DSL v1).")
app.add_typer(spec_app, name="spec")
SpecArg = Annotated[Path, typer.Argument(exists=True, dir_okay=False)]


@spec_app.command("validate")
def spec_validate(spec: SpecArg) -> None:
    """Validate a spec against the DSL schema and semantic rules; print its hash."""
    try:
        s = load_spec(spec)
    except SpecError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"valid: {s.name} ({s.hash})")


@app.command("evaluate")
def evaluate(
    spec: SpecArg,
    split: Annotated[str, typer.Option(help="search or validation")] = "search",
    contract: Annotated[str | None, typer.Option(help="NQ or MNQ; default from config")] = None,
    config: ConfigOpt = Path("config/v0.yaml"),
) -> None:
    """Evaluate one spec on a split (costs always on) and record it in the results store."""
    try:
        cfg = load_model(config, FoundryConfig)
        s = load_spec(spec)
        ev = NQEvaluator(cfg, Path.cwd())
        with ResultsStore(ev.results_path) as store:
            r = ev.evaluate(s, split, contract, store=store, generator="manual")
    except (ConfigError, SpecError, EvaluationError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    typer.echo(
        json.dumps(
            {
                "spec": s.name,
                "hash": s.hash,
                "split": split,
                "contract": r.contract,
                "metrics": r.metrics,
                "breakdown": r.breakdown,
            },
            indent=2,
        )
    )


@app.command("critique")
def critique(
    spec: SpecArg,
    contract: Annotated[str | None, typer.Option(help="NQ or MNQ; default from config")] = None,
    config: ConfigOpt = Path("config/v0.yaml"),
) -> None:
    """Run the kill tests (search, then validation) on one spec and record the verdict."""
    try:
        cfg = load_model(config, FoundryConfig)
        s = load_spec(spec)
        ev = NQEvaluator(cfg, Path.cwd())
        with ResultsStore(ev.results_path) as store:
            runner = CriticRunner(
                ev,
                cfg.critics,
                store,
                contract or cfg.evaluator.default_contract,
                cfg.evaluator.annualization_days,
            )
            v = runner.critique(s, generator="manual")
    except (ConfigError, SpecError, EvaluationError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    for r in v.results:
        typer.echo(
            f"[{'pass' if r.passed else 'KILL'}] {r.stage:<10} {r.test:<22} "
            f"{json.dumps(clean_numbers(r.numbers), sort_keys=True)}"
        )
    status = "ALIVE" if v.alive else f"DEAD ({v.killed_stage}: {v.killed_by})"
    typer.echo(f"{s.name} {s.hash[:12]}: {status} labels={v.labels}")


@app.command("search")
def search(
    config: ConfigOpt = Path("config/v0.yaml"),
    engine: Annotated[Path | None, typer.Option(help="engine YAML; default from config")] = None,
    seed: Annotated[int | None, typer.Option(help="run seed; default from config")] = None,
    no_llm: Annotated[bool, typer.Option("--no-llm", help="skip the LLM proposer")] = False,
    no_critics: Annotated[bool, typer.Option("--no-critics", help="search only")] = False,
) -> None:
    """Run one search (search split only), then send the top candidates to the critics."""
    load_dotenv(Path.cwd() / ".env")
    try:
        cfg = load_model(config, FoundryConfig)
        engine_cfg, engine_id = load_engine(engine or cfg.search.engine_file)
        ev = NQEvaluator(cfg, Path.cwd())
        budget = RunBudget(cfg.search.budgets)
        use_llm = not no_llm and engine_cfg.mix.llm > 0 and cfg.search.budgets.llm_dollars > 0
        llm = LLMClient(cfg.search.llm, budget) if use_llm else None
        contract = cfg.evaluator.default_contract
        with ResultsStore(ev.results_path) as store:
            run = SearchRun(
                NQSpecSpace(),
                ev,
                store,
                engine_cfg,
                engine_id,
                budget,
                cfg.seed if seed is None else seed,
                cfg.critics.min_trades.search,
                contract,
                llm,
                cfg.search.llm.model,
            )
            critics = (
                None
                if no_critics
                else CriticRunner(
                    ev, cfg.critics, store, contract, cfg.evaluator.annualization_days
                )
            )
            result = run_search(run, critics)
    except (ConfigError, EvaluationError, SecretError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    typer.echo(
        json.dumps(
            {k: v for k, v in result.summary.items() if k != "generations"}, indent=2, default=str
        )
    )
    typer.echo(f"run {result.run_id}: {result.status}")


@app.command("funnel")
def funnel(config: ConfigOpt = Path("config/v0.yaml")) -> None:
    """Generated -> evaluated -> killed by each test -> alive, from the results store."""
    cfg = load_model(config, FoundryConfig)
    path = cfg.evaluator.results_db
    path = path if path.is_absolute() else Path.cwd() / path
    with ResultsStore(path) as store:
        typer.echo(json.dumps(store.funnel(), indent=2))


@holdout_app.command("open")
def holdout_open(
    engine: Annotated[str, typer.Option(help="engine id that produced the candidates")],
    candidates: Annotated[str, typer.Option(help="comma-separated spec hashes")],
    reason: Annotated[str, typer.Option(help="why the holdout is being opened")],
    config: ConfigOpt = Path("config/v0.yaml"),
) -> None:
    """Evaluate candidates on the sealed holdout. Logged, and at most once per candidate."""
    try:
        cfg = load_model(config, FoundryConfig)
        ev = NQEvaluator(cfg, Path.cwd())
        with ResultsStore(ev.results_path) as store:
            specs = []
            for h in [c.strip() for c in candidates.split(",") if c.strip()]:
                raw = store.spec_json(h)
                if raw is None:
                    raise EvaluationError(f"unknown candidate {h}: evaluate it on search first")
                specs.append(validate_spec(json.loads(raw)))
            results = ev.evaluate_holdout(specs, engine, reason, store)
    except (ConfigError, SpecError, EvaluationError, HoldoutError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    typer.echo(
        json.dumps(
            [{"spec": r.spec_name, "hash": r.spec_hash, "metrics": r.metrics} for r in results],
            indent=2,
        )
    )


@holdout_app.command("status")
def holdout_status(config: ConfigOpt = Path("config/v0.yaml")) -> None:
    """Verify the sealed holdout's hashes and show its manifest summary and access log."""
    cfg = load_model(config, FoundryConfig)
    directory = (
        cfg.holdout.directory
        if cfg.holdout.directory.is_absolute()
        else Path.cwd() / cfg.holdout.directory
    )
    vault = HoldoutVault(directory)
    try:
        m = vault.verify()
    except HoldoutError as exc:
        typer.echo(f"holdout: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(
        json.dumps(
            {
                "content_hash": m["content_hash"],
                "sealed_at": m["sealed_at"],
                "summary": m["summary"],
                "openings": len(vault.access_log()),
                "evaluated_candidates": sorted(vault.evaluated_candidates()),
            },
            indent=2,
        )
    )
