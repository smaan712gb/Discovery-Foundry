"""The NQ evaluator: spec + split -> trades, daily PnL and metrics, with costs always on.

Search evaluations see only search bars. Validation evaluations see search + validation bars
(search bars are feature warm-up only) and may only open trades on validation bars. The holdout
is reachable only through `evaluate_holdout`, which goes through HoldoutVault.open (logged, and
at most once per candidate).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from foundry.core.config import config_hash, load_model
from foundry.core.hashing import combined_hash, sha256_file, sha256_text
from foundry.core.holdout import HoldoutVault
from foundry.core.metrics import trade_metrics
from foundry.core.repro import run_metadata
from foundry.core.results import ResultsStore
from foundry.domains.nq.backtest import EXIT_REASONS, run
from foundry.domains.nq.calendar import TradingCalendar
from foundry.domains.nq.config import EvaluatorConfig, EventsFile, FoundryConfig
from foundry.domains.nq.dsl import Spec
from foundry.domains.nq.features import MarketContext, build_context
from foundry.domains.nq.signals import compile_inputs
from foundry.domains.nq.splits import HOLDOUT, PROCESSED_SPLITS, SEARCH, VALIDATION

EVALUABLE_SPLITS = PROCESSED_SPLITS


class EvaluationError(Exception):
    pass


@dataclass(frozen=True)
class EvaluationResult:
    spec_hash: str
    spec_name: str
    split: str
    contract: str
    metrics: dict[str, Any]
    breakdown: dict[str, Any]
    trades: pl.DataFrame
    daily: pl.DataFrame


def evaluate_context(
    spec: Spec,
    ctx: MarketContext,
    ev: EvaluatorConfig,
    contract: str,
    split: str,
    trade_mask: np.ndarray | None = None,
    slippage_multiplier: float = 1.0,
) -> EvaluationResult:
    """Pure evaluation over a prepared context. Used by the evaluator, the critics and tests."""
    if contract not in ev.contracts:
        raise EvaluationError(f"unknown contract {contract!r}")
    cs = ev.contracts[contract]
    k = compile_inputs(spec, ctx, ev, cs, trade_mask, slippage_multiplier)
    t = run(k)
    qty = ev.contracts_per_trade
    mult = cs.point_value * qty
    commission = cs.commission_round_turn * qty
    gross = t.direction * (t.exit_ref - t.entry_ref) * mult
    net = t.direction * (t.exit_px - t.entry_px) * mult - commission

    ts = ctx.frame["ts"]
    tdates = ctx.frame["trading_date"]
    trades = pl.DataFrame(
        {
            "n": np.arange(t.count, dtype=np.int32),
            "signal_ts": ts.gather(t.signal_i),
            "entry_ts": ts.gather(t.entry_i),
            "exit_ts": ts.gather(t.exit_i),
            "direction": t.direction.astype(np.int32),
            "entry_price": t.entry_px,
            "exit_price": t.exit_px,
            "gross_pnl": gross,
            "commission": np.full(t.count, commission),
            "net_pnl": net,
            "reason": [EXIT_REASONS[int(r)] for r in t.reason],
            "entry_session": ctx.session[t.entry_i].astype(str) if t.count else np.array([], str),
            "vol_bucket": ctx.vol_bucket[t.entry_i].astype(str) if t.count else np.array([], str),
            "entry_trading_date": tdates.gather(t.entry_i),
            "exit_trading_date": tdates.gather(t.exit_i),
        }
    )

    window = ctx.windows[spec.session].window_id
    evaluated = window >= 0
    if trade_mask is not None:
        evaluated &= trade_mask
    days = pl.DataFrame({"trading_date": tdates.filter(pl.Series(evaluated)).unique().sort()})
    per_day = trades.group_by(pl.col("exit_trading_date").alias("trading_date")).agg(
        pl.col("net_pnl").sum()
    )
    daily = (
        days.join(per_day, on="trading_date", how="left")
        .with_columns(pl.col("net_pnl").fill_null(0.0))
        .sort("trading_date")
    )

    metrics = trade_metrics(
        net,
        gross,
        daily["net_pnl"].to_numpy(),
        t.bars_in_market,
        int(evaluated.sum()),
        ev.annualization_days,
    )
    breakdown = {
        "by_session": _group(trades, "entry_session"),
        "by_vol_regime": _group(trades, "vol_bucket"),
        "by_exit_reason": _group(trades, "reason"),
    }
    return EvaluationResult(
        spec.hash, spec.name, split, contract, metrics, breakdown, trades, daily
    )


def _group(trades: pl.DataFrame, key: str) -> dict[str, dict[str, float]]:
    g = trades.group_by(key).agg(pl.col("net_pnl").sum(), pl.len().alias("trades")).sort(key)
    return {
        str(r[key]): {"net_pnl": float(r["net_pnl"]), "trades": int(r["trades"])}
        for r in g.iter_rows(named=True)
    }


def load_events(path: Path) -> set[date]:
    return {e.date for e in load_model(path, EventsFile).entries}


class NQEvaluator:
    """Loads verified processed data once and evaluates specs on the search or validation split."""

    def __init__(
        self, cfg: FoundryConfig, repo_root: Path, config_path: Path | None = None
    ) -> None:
        self.cfg = cfg
        self.root = repo_root
        d = cfg.data

        def rp(p: Path) -> Path:
            return p if p.is_absolute() else repo_root / p

        self.processed = rp(d.processed_dir)
        self.calendar = TradingCalendar.from_file(rp(d.calendar_file), d.sessions)
        self.events = load_events(rp(cfg.evaluator.events_file))
        self.results_path = rp(cfg.evaluator.results_db)
        self._manifest = self._verified_manifest()
        self.data_hash: str = self._manifest["data_hash"]
        self._contexts: dict[str, tuple[MarketContext, np.ndarray | None]] = {}

    def _verified_manifest(self) -> dict[str, Any]:
        mpath = self.processed / "build_manifest.json"
        if not mpath.is_file():
            raise EvaluationError(f"no data build at {self.processed}; run `foundry data build`")
        m: dict[str, Any] = json.loads(mpath.read_text(encoding="utf-8"))
        if m.get("status") != "PASS":
            raise EvaluationError("the last data build did not pass; see data_report.md")
        series = self.processed / "bars.parquet"
        if sha256_file(series) != m["outputs"]["bars.parquet"]:
            raise EvaluationError("processed bars.parquet does not match its build manifest")
        return m

    def context(self, split: str) -> tuple[MarketContext, np.ndarray | None]:
        if split not in EVALUABLE_SPLITS:
            raise EvaluationError(
                f"split {split!r} cannot be evaluated here (only {EVALUABLE_SPLITS})"
            )
        if split not in self._contexts:
            bars = pl.read_parquet(self.processed / "bars.parquet")
            if split == SEARCH:
                # Search sees search bars only: nothing from validation can leak into features.
                bars = bars.filter(pl.col("split") == SEARCH)
            ctx = build_context(
                bars,
                self.calendar,
                self.cfg.data.instrument.bar_minutes,
                self.cfg.evaluator.regimes,
                self.events,
            )
            mask = None if split == SEARCH else (ctx.frame["split"] == VALIDATION).to_numpy()
            self._contexts[split] = (ctx, mask)
        return self._contexts[split]

    def evaluate_holdout(
        self,
        specs: list[Spec],
        engine_id: str,
        reason: str,
        store: ResultsStore,
        contract: str | None = None,
    ) -> list[EvaluationResult]:
        """The only holdout evaluation path. The vault logs the opening before returning data and
        refuses any candidate already evaluated there. Search and validation bars are warm-up only;
        trades may open only on holdout bars."""
        contract = contract or self.cfg.evaluator.default_contract
        vault = HoldoutVault(self._rp(self.cfg.holdout.directory))
        opened = vault.open(engine_id, [s.hash for s in specs], reason)
        warmup = pl.read_parquet(self.processed / "bars.parquet")
        bars = pl.concat([warmup, opened.tables["bars"].select(warmup.columns)])
        ctx = build_context(
            bars,
            self.calendar,
            self.cfg.data.instrument.bar_minutes,
            self.cfg.evaluator.regimes,
            self.events,
        )
        mask = (ctx.frame["split"] == HOLDOUT).to_numpy()
        holdout_hash = str(opened.manifest["content_hash"])
        results = []
        for spec in specs:
            r = evaluate_context(spec, ctx, self.cfg.evaluator, contract, HOLDOUT, mask)
            store.record_spec(spec.hash, spec.name, json.dumps(spec.raw, sort_keys=True))
            store.record_evaluation(
                combined_hash(
                    [
                        ("spec", spec.hash),
                        ("split", HOLDOUT),
                        ("contract", contract),
                        ("data", holdout_hash),
                        ("eval", self.eval_config_hash(contract, 1.0)),
                    ]
                ),
                spec.hash,
                HOLDOUT,
                contract,
                holdout_hash,
                self.eval_config_hash(contract, 1.0),
                r.metrics,
                r.breakdown,
                run_metadata(self.cfg.seed, config_hash(self.cfg), self.root)
                | {"engine_id": engine_id},
                r.trades,
                r.daily,
            )
            results.append(r)
        return results

    def _rp(self, p: Path) -> Path:
        return p if p.is_absolute() else self.root / p

    def eval_key(self, spec: Spec, split: str, contract: str, slippage_multiplier: float) -> str:
        cfg_hash = self.eval_config_hash(contract, slippage_multiplier)
        return combined_hash(
            [
                ("spec", spec.hash),
                ("split", split),
                ("contract", contract),
                ("data", self.data_hash),
                ("eval", cfg_hash),
            ]
        )

    def eval_config_hash(self, contract: str, slippage_multiplier: float) -> str:
        return sha256_text(f"{config_hash(self.cfg.evaluator)}|{contract}|{slippage_multiplier}")

    def evaluate(
        self,
        spec: Spec,
        split: str,
        contract: str | None = None,
        slippage_multiplier: float = 1.0,
        store: ResultsStore | None = None,
        generator: str | None = None,
        parents: list[str] | None = None,
        engine_id: str | None = None,
    ) -> EvaluationResult:
        contract = contract or self.cfg.evaluator.default_contract
        key = self.eval_key(spec, split, contract, slippage_multiplier)
        if store is not None and (cached := store.get(key)) is not None:
            # Identical spec, data and evaluator settings: never re-evaluated.
            return EvaluationResult(
                spec.hash,
                spec.name,
                split,
                contract,
                cached.metrics,
                cached.breakdown,
                store.trades(key),
                store.daily(key),
            )
        ctx, mask = self.context(split)
        result = evaluate_context(
            spec, ctx, self.cfg.evaluator, contract, split, mask, slippage_multiplier
        )
        if store is not None:
            store.record_spec(
                spec.hash,
                spec.name,
                json.dumps(spec.raw, sort_keys=True),
                generator,
                parents,
                engine_id,
            )
            store.record_evaluation(
                key,
                spec.hash,
                split,
                contract,
                self.data_hash,
                self.eval_config_hash(contract, slippage_multiplier),
                result.metrics,
                result.breakdown,
                run_metadata(self.cfg.seed, config_hash(self.cfg), self.root),
                result.trades,
                result.daily,
            )
        return result
