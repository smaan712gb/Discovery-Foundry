"""NQ tournament wiring: one isolated search + critics per (engine, seed) on a fold."""

from __future__ import annotations

import math
import time
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from foundry.core.budgets import RunBudget
from foundry.core.config import config_hash
from foundry.core.results import ResultsStore
from foundry.critics.config import MinTrades
from foundry.critics.runner import CriticRunner, Verdict
from foundry.domains.nq.config import FoundryConfig
from foundry.domains.nq.evaluator import NQEvaluator
from foundry.domains.nq.space import NQSpecSpace
from foundry.domains.nq.splits import VALIDATION
from foundry.meta.config import MetaConfig
from foundry.meta.promotion import RunOutcome
from foundry.meta.tournament import RunFn
from foundry.search.engine import EngineConfig
from foundry.search.llm_client import LLMCache, LLMClient
from foundry.search.run import SearchRun, run_search

SHARPE_CAP = 5.0


def validation_folds(processed_dir: Path, n_folds: int) -> list[tuple[date, date, int]]:
    days = (
        pl.read_parquet(processed_dir / "bars.parquet", columns=["trading_date", "split"])
        .filter(pl.col("split") == VALIDATION)["trading_date"]
        .unique()
        .sort()
        .to_list()
    )
    if len(days) < n_folds:
        raise ValueError(f"only {len(days)} validation days for {n_folds} folds")
    return [
        (chunk[0], chunk[-1], len(chunk))
        for chunk in (list(c) for c in np.array_split(np.array(days, dtype=object), n_folds))
    ]


def score_verdicts(verdicts: list[Verdict], annualization: int) -> tuple[float, int]:
    """Σ over survivors of 1 + clip(validation Sharpe, 0, 5)/5 (ADR 0010)."""
    score, alive = 0.0, 0
    for v in verdicts:
        if not v.alive:
            continue
        alive += 1
        dsr = next(
            (r for r in v.results if r.stage == VALIDATION and r.test == "deflated_sharpe"), None
        )
        daily = dsr.numbers.get("daily_sharpe") if dsr else None
        sharpe = float(daily) * math.sqrt(annualization) if isinstance(daily, float) else 0.0
        score += 1.0 + min(max(sharpe, 0.0), SHARPE_CAP) / SHARPE_CAP
    return score, alive


def make_run_fn(
    cfg: FoundryConfig, repo_root: Path, meta: MetaConfig, use_llm: bool, llm_cache: LLMCache | None
) -> RunFn:
    processed = (
        cfg.data.processed_dir
        if cfg.data.processed_dir.is_absolute()
        else repo_root / cfg.data.processed_dir
    )
    folds = validation_folds(processed, meta.validation_folds)
    total_days = sum(f[2] for f in folds)
    contract = cfg.evaluator.default_contract
    ann = cfg.evaluator.annualization_days

    def run_one(
        engine: EngineConfig, engine_id: str, seed: int, store_path: Path, fold: int
    ) -> RunOutcome:
        lo, hi, fold_days = folds[fold]
        # Fewer validation days in a fold: the validation trade minimum scales with them.
        critics_cfg = cfg.critics.model_copy(
            update={
                "min_trades": MinTrades(
                    search=cfg.critics.min_trades.search,
                    validation=max(
                        1, math.ceil(cfg.critics.min_trades.validation * fold_days / total_days)
                    ),
                )
            }
        )
        ev = NQEvaluator(cfg, repo_root, validation_window=(lo, hi))
        budget = RunBudget(meta.run_budget)
        llm = None
        if use_llm and engine.mix.llm > 0 and meta.run_budget.llm_dollars > 0:
            llm = LLMClient(cfg.search.llm, budget, cache=llm_cache)
        run_seed = cfg.seed * 1000 + seed
        started = time.monotonic()
        store_path.parent.mkdir(parents=True, exist_ok=True)
        with ResultsStore(store_path) as store:
            run = SearchRun(
                NQSpecSpace(),
                ev,
                store,
                engine,
                engine_id,
                budget,
                run_seed,
                critics_cfg.min_trades.search,
                contract,
                llm,
                cfg.search.llm.model,
            )
            try:
                res = run_search(run, CriticRunner(ev, critics_cfg, store, contract, ann))
                status, verdicts = res.status, res.verdicts
            except Exception as exc:
                status, verdicts = f"error:{type(exc).__name__}: {exc}", []
            run_id = run.run_id
            evaluated = tuple(run.done)
        score, alive = score_verdicts(verdicts, ann)
        return RunOutcome(
            engine_id,
            seed,
            run_id,
            status,
            score,
            alive,
            len(evaluated),
            time.monotonic() - started,
            config_hash(critics_cfg),
            evaluated,
        )

    return run_one
