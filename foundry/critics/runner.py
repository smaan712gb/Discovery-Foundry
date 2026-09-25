"""Runs the kill tests on one candidate: search stage, then validation stage (ADR 0008).

Domain-agnostic: it needs an evaluator and a spec that expose the small protocols below. Every
test result and the final verdict go to the results store.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, Self

import numpy as np
import polars as pl

from foundry.core.results import ResultsStore
from foundry.critics.config import CriticConfig
from foundry.critics.stats import (
    cpcv_select_then_test,
    deflated_sharpe,
    pbo_cscv,
    top_day_share,
)

STAGES = ("search", "validation")
TEST_ORDER = (
    "min_trades_and_profit",
    "cost_stress",
    "concentration",
    "regime_stability",
    "deflated_sharpe",
    "parameter_robustness",
    "pbo",
    "cpcv",
)
VOL_BUCKETS = ("low", "mid", "high")


class CandidateSpec(Protocol):
    @property
    def hash(self) -> str: ...
    @property
    def name(self) -> str: ...
    def params(self) -> dict[str, dict[str, Any]]: ...
    def traded_regimes(self) -> list[str] | None: ...
    def with_params(self, values: dict[str, float]) -> Self: ...


class Result(Protocol):
    @property
    def metrics(self) -> dict[str, Any]: ...
    @property
    def breakdown(self) -> dict[str, Any]: ...
    @property
    def daily(self) -> pl.DataFrame: ...
    @property
    def trades(self) -> pl.DataFrame: ...


class Evaluator(Protocol):
    def evaluate(
        self,
        spec: Any,
        split: str,
        contract: str | None = ...,
        slippage_multiplier: float = ...,
        store: ResultsStore | None = ...,
        generator: str | None = ...,
    ) -> Any: ...
    def eval_config_hash(self, contract: str, slippage_multiplier: float) -> str: ...


@dataclass(frozen=True)
class TestResult:
    stage: str
    test: str
    passed: bool
    numbers: dict[str, Any]


@dataclass
class Verdict:
    spec_hash: str
    spec_name: str
    alive: bool = False
    killed_stage: str | None = None
    killed_by: str | None = None
    labels: list[str] = field(default_factory=list)
    results: list[TestResult] = field(default_factory=list)


def clean_numbers(x: Any) -> Any:
    """JSON-safe numbers: NaN and inf become None."""
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, dict):
        return {k: clean_numbers(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [clean_numbers(v) for v in x]
    if isinstance(x, np.floating | np.integer):
        return clean_numbers(x.item())
    return x


def neighbours(spec: CandidateSpec, perturbations: list[float]) -> list[Any]:
    """Each parameter moved alone by each relative perturbation; in-bounds and distinct only."""
    out: dict[str, Any] = {}
    for name, p in spec.params().items():
        v, lo, hi = float(p["value"]), float(p["low"]), float(p["high"])
        for d in perturbations:
            nv = v + d * (hi - lo) if v == 0 else v * (1 + d)
            if p["type"] == "int":
                nv = float(round(nv))
            if nv == v or not lo <= nv <= hi:
                continue
            cand = spec.with_params({name: nv})
            if cand.hash != spec.hash:
                out.setdefault(cand.hash, cand)
    return list(out.values())


class CriticRunner:
    def __init__(
        self,
        evaluator: Evaluator,
        cfg: CriticConfig,
        store: ResultsStore,
        contract: str,
        annualization: int,
    ) -> None:
        self.ev = evaluator
        self.cfg = cfg
        self.store = store
        self.contract = contract
        self.ann = annualization

    def critique(self, spec: CandidateSpec, generator: str | None = None) -> Verdict:
        verdict = Verdict(spec.hash, spec.name)
        for stage in STAGES:
            stage_run = _Stage(self, spec, stage, generator)
            for test in TEST_ORDER:
                res = stage_run.run(test)
                verdict.results.append(res)
                self.store.record_critic_result(
                    spec.hash, stage, test, res.passed, clean_numbers(res.numbers)
                )
                if res.numbers.get("label"):
                    verdict.labels.append(str(res.numbers["label"]))
                if not res.passed:
                    verdict.killed_stage, verdict.killed_by = stage, test
                    break
            if verdict.killed_by:
                break
        verdict.alive = verdict.killed_by is None
        verdict.labels = sorted(set(verdict.labels))
        self.store.record_verdict(
            spec.hash, verdict.alive, verdict.killed_stage, verdict.killed_by, verdict.labels
        )
        return verdict


class _Stage:
    """One split's worth of tests. Neighbour evaluations are shared between tests 6-8."""

    def __init__(
        self, runner: CriticRunner, spec: CandidateSpec, stage: str, generator: str | None
    ) -> None:
        self.r = runner
        self.spec = spec
        self.stage = stage
        # The candidate's own base evaluation is a trial: stored. Nothing else here is.
        self.base: Result = runner.ev.evaluate(
            spec, stage, runner.contract, store=runner.store, generator=generator
        )
        self._family: list[np.ndarray] | None = None
        self._family_net: list[float] | None = None

    def run(self, test: str) -> TestResult:
        fn: Callable[[], tuple[bool, dict[str, Any]]] = getattr(self, f"_t_{test}")
        passed, numbers = fn()
        return TestResult(self.stage, test, passed, numbers)

    # ---- helpers -----------------------------------------------------------------------

    def _daily(self, result: Result) -> np.ndarray:
        return np.asarray(result.daily["net_pnl"].to_numpy(), dtype=float)

    def _family_matrix(self) -> tuple[np.ndarray, list[float]]:
        if self._family is None or self._family_net is None:
            base_days = self.base.daily.select("trading_date")
            cols, nets = [self._daily(self.base)], [float(self.base.metrics["net_pnl"])]
            for nb in neighbours(self.spec, self.r.cfg.perturbations):
                res = self.r.ev.evaluate(nb, self.stage, self.r.contract)
                aligned = base_days.join(res.daily, on="trading_date", how="left").fill_null(0.0)
                cols.append(aligned["net_pnl"].to_numpy().astype(float))
                nets.append(float(res.metrics["net_pnl"]))
            self._family, self._family_net = cols, nets
        return np.column_stack(self._family), self._family_net

    # ---- the tests ---------------------------------------------------------------------

    def _t_min_trades_and_profit(self) -> tuple[bool, dict[str, Any]]:
        m = self.base.metrics
        need = getattr(self.r.cfg.min_trades, self.stage)
        ok = m["net_pnl"] > 0 and m["trade_count"] >= need
        return ok, {"net_pnl": m["net_pnl"], "trade_count": m["trade_count"], "min_trades": need}

    def _t_cost_stress(self) -> tuple[bool, dict[str, Any]]:
        mult = self.r.cfg.cost_stress_multiplier
        stressed = self.r.ev.evaluate(
            self.spec, self.stage, self.r.contract, slippage_multiplier=mult
        )
        net = float(stressed.metrics["net_pnl"])
        return net > 0, {"slippage_multiplier": mult, "net_pnl": net}

    def _t_concentration(self) -> tuple[bool, dict[str, Any]]:
        c = self.r.cfg.concentration
        share = top_day_share(self._daily(self.base), c.top_day_fraction)
        ok = not math.isnan(share) and share < c.max_profit_share
        return ok, {
            "top_day_fraction": c.top_day_fraction,
            "profit_share": share,
            "max_profit_share": c.max_profit_share,
        }

    def _t_regime_stability(self) -> tuple[bool, dict[str, Any]]:
        by = self.base.breakdown.get("by_vol_regime", {})
        pnl = {b: float(by.get(b, {}).get("net_pnl", 0.0)) for b in VOL_BUCKETS}
        traded = self.spec.traded_regimes()
        if traded is not None and len(traded) < len(VOL_BUCKETS):
            ok = all(pnl[b] > 0 for b in traded)
            return ok, {"pnl_by_regime": pnl, "traded": traded, "label": "regime_specific"}
        profitable = sum(v > 0 for v in pnl.values())
        return profitable >= self.r.cfg.min_profitable_regimes, {
            "pnl_by_regime": pnl,
            "profitable_regimes": profitable,
            "min_profitable": self.r.cfg.min_profitable_regimes,
        }

    def _t_deflated_sharpe(self) -> tuple[bool, dict[str, Any]]:
        base_cfg = self.r.ev.eval_config_hash(self.r.contract, 1.0)
        trials = self.r.store.trial_sharpes(self.stage, self.r.contract, base_cfg)
        sr = np.array(list(trials.values())) / math.sqrt(self.r.ann)
        d = deflated_sharpe(self._daily(self.base), max(len(trials), 1), sr)
        ok = not math.isnan(d.dsr) and d.dsr > self.r.cfg.dsr_threshold
        return ok, {
            "dsr": d.dsr,
            "threshold": self.r.cfg.dsr_threshold,
            "n_trials": d.n_trials,
            "daily_sharpe": d.sharpe,
            "sr0": d.sr0,
            "sr_variance": d.sr_variance,
            "days": d.t,
            "skew": d.skew,
            "kurtosis": d.kurt,
        }

    def _t_parameter_robustness(self) -> tuple[bool, dict[str, Any]]:
        _, nets = self._family_matrix()
        neigh = nets[1:]
        if not neigh:
            return False, {"reason": "no in-bounds parameter neighbours", "neighbours": 0}
        med = float(np.median(neigh))
        return med > 0, {
            "neighbours": len(neigh),
            "median_neighbour_net": med,
            "worst_neighbour_net": min(neigh),
            "profitable_share": float(np.mean(np.array(neigh) > 0)),
        }

    def _t_pbo(self) -> tuple[bool, dict[str, Any]]:
        base_cfg = self.r.ev.eval_config_hash(self.r.contract, 1.0)
        specs, wide = self.r.store.trial_daily_matrix(self.stage, self.r.contract, base_cfg)
        cfg = self.r.cfg
        if len(specs) < cfg.pbo_min_trials:
            return True, {
                "assessed": False,
                "n_trials": len(specs),
                "min_trials": cfg.pbo_min_trials,
                "label": "pbo_not_assessed",
            }
        res = pbo_cscv(wide.select(specs).to_numpy().astype(float), cfg.cscv_blocks)
        if math.isnan(res.pbo):
            return True, {
                "assessed": False,
                "n_trials": len(specs),
                "reason": "too few days",
                "label": "pbo_not_assessed",
            }
        return res.pbo < cfg.pbo_threshold, {
            "assessed": True,
            "pbo": res.pbo,
            "threshold": cfg.pbo_threshold,
            "n_trials": len(specs),
            "splits": res.n_splits,
        }

    def _t_cpcv(self) -> tuple[bool, dict[str, Any]]:
        matrix, _ = self._family_matrix()
        c = self.r.cfg.cpcv
        purge = self._max_holding_days()
        res = cpcv_select_then_test(
            matrix, c.groups, c.test_groups, purge, c.embargo_days, self.r.ann
        )
        ok = (
            not math.isnan(res.positive_fraction)
            and res.positive_fraction >= c.min_positive_fraction
            and res.mean_test_sharpe > 0
        )
        return ok, {
            "positive_fraction": res.positive_fraction,
            "min_positive_fraction": c.min_positive_fraction,
            "mean_test_sharpe": res.mean_test_sharpe,
            "splits": res.n_splits,
            "purge_days": purge,
            "embargo_days": c.embargo_days,
            "family_size": int(matrix.shape[1]),
        }

    def _max_holding_days(self) -> int:
        t = self.base.trades
        if t.is_empty() or "entry_trading_date" not in t.columns:
            return 0
        days = self.base.daily["trading_date"].to_numpy()
        e = np.searchsorted(days, t["entry_trading_date"].to_numpy())
        x = np.searchsorted(days, t["exit_trading_date"].to_numpy())
        return int(np.max(x - e)) if len(e) else 0
