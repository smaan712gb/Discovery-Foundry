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
from foundry.domains.nq.meta_runner import make_run_fn
from foundry.domains.nq.space import NQSpecSpace
from foundry.meta.propose import ProposalError, propose_by_code, propose_by_llm, write_engine
from foundry.meta.tournament import run_tournament
from foundry.search.engine import EngineConfig, load_engine
from foundry.search.llm_client import LLMCache, LLMClient
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


meta_app = typer.Typer(no_args_is_help=True, help="Meta layer: engine N vs N+1 (ADR 0010).")
app.add_typer(meta_app, name="meta")


def _registry_with_champion(cfg: FoundryConfig, store: ResultsStore) -> tuple[EngineConfig, str]:
    """The current champion; registers the configured engine as the first one if none exists."""
    champ = store.champion()
    if champ is None:
        ecfg, eid = load_engine(cfg.search.engine_file)
        store.register_engine(
            eid, ecfg.label, ecfg.model_dump_json(), None, "manual", "initial engine from config"
        )
        store.set_champion(eid, None)
        return ecfg, eid
    raw = store.engine_config_json(champ)
    if raw is None:
        raise EvaluationError(f"champion {champ} is missing from the engine registry")
    return EngineConfig.model_validate_json(raw), champ


def _next_label(store: ResultsStore) -> str:
    n = 1
    labels = set(store.engine_labels())
    while f"v{n}" in labels:
        n += 1
    return f"v{n}"


@meta_app.command("propose")
def meta_propose(
    config: ConfigOpt = Path("config/v0.yaml"),
    llm: Annotated[bool, typer.Option("--llm", help="let the LLM propose (validated)")] = False,
    seed: Annotated[int, typer.Option(help="seed for the code proposer")] = 0,
) -> None:
    """Create engine N+1 from the current champion and write its YAML."""
    load_dotenv(Path.cwd() / ".env")
    try:
        cfg = load_model(config, FoundryConfig)
        path = cfg.evaluator.results_db
        with ResultsStore(path if path.is_absolute() else Path.cwd() / path) as store:
            parent, parent_id = _registry_with_champion(cfg, store)
            label = _next_label(store)
            if llm:
                budget = RunBudget(cfg.search.budgets)
                prop = propose_by_llm(
                    parent,
                    label,
                    store.tournament_history(),
                    LLMClient(cfg.search.llm, budget),
                    Path("config/prompts"),
                )
            else:
                prop = propose_by_code(parent, label, cfg.meta, seed, allow_llm=parent.mix.llm > 0)
            out = write_engine(prop, Path("config/engines"))
            store.register_engine(
                prop.engine_id,
                label,
                prop.cfg.model_dump_json(),
                parent_id,
                prop.proposer,
                prop.rationale,
            )
    except (ConfigError, EvaluationError, ProposalError, SecretError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"challenger {prop.engine_id} (parent {parent_id}) written to {out}")
    typer.echo(f"rationale: {prop.rationale}")


@meta_app.command("tournament")
def meta_tournament(
    challenger: Annotated[Path, typer.Option(exists=True, dir_okay=False, help="engine YAML")],
    config: ConfigOpt = Path("config/v0.yaml"),
    no_llm: Annotated[bool, typer.Option("--no-llm", help="disable the LLM generator")] = False,
) -> None:
    """Champion vs challenger over K seeds with equal budgets; promote only on a significant win."""
    load_dotenv(Path.cwd() / ".env")
    try:
        cfg = load_model(config, FoundryConfig)
        path = cfg.evaluator.results_db
        workspace = (
            cfg.meta.workspace
            if cfg.meta.workspace.is_absolute()
            else Path.cwd() / cfg.meta.workspace
        )
        with ResultsStore(path if path.is_absolute() else Path.cwd() / path) as store:
            champ = _registry_with_champion(cfg, store)
            ch_cfg, ch_id = load_engine(challenger)
            if ch_id == champ[1]:
                raise EvaluationError("the challenger is identical to the champion")
            store.register_engine(
                ch_id,
                ch_cfg.label,
                ch_cfg.model_dump_json(),
                champ[1],
                "manual",
                f"from {challenger}",
            )
            cache = LLMCache(workspace / "llm_cache.duckdb")
            try:
                run_fn = make_run_fn(cfg, Path.cwd(), cfg.meta, not no_llm, cache)
                result = run_tournament(cfg.meta, champ, (ch_cfg, ch_id), run_fn, store, workspace)
            finally:
                cache.close()
            touches = store.validation_touches()
    except (ConfigError, EvaluationError, SecretError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    typer.echo(
        json.dumps(result.summary() | {"validation_touches": touches}, indent=2, default=str)
    )


@meta_app.command("status")
def meta_status(config: ConfigOpt = Path("config/v0.yaml")) -> None:
    """Current champion, tournament history and how often validation has been touched."""
    cfg = load_model(config, FoundryConfig)
    path = cfg.evaluator.results_db
    with ResultsStore(path if path.is_absolute() else Path.cwd() / path) as store:
        typer.echo(
            json.dumps(
                {
                    "champion": store.champion(),
                    "tournaments": store.tournament_history(),
                    "validation_touches": store.validation_touches(),
                },
                indent=2,
                default=str,
            )
        )


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
