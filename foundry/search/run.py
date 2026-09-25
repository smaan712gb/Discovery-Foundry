"""The search loop (ADR 0009): propose -> dedupe -> pre-screen -> evaluate on search -> learn.

Generic: the domain supplies a SpecSpace and an Evaluator. Only the search split is ever
evaluated here; survivors of the loop go to the critics afterwards.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol

import numpy as np

from foundry.core.budgets import BudgetExceeded, RunBudget
from foundry.core.results import ResultsStore
from foundry.critics.runner import CriticRunner, Verdict
from foundry.search.cma_tuner import CMATuner
from foundry.search.engine import EngineConfig
from foundry.search.llm_client import LLMClient
from foundry.search.llm_proposer import build_prompt, propose
from foundry.search.surrogate import Surrogate, spearman


class SpecSpace(Protocol):
    def validate(self, raw: dict[str, Any]) -> Any: ...
    def random(self, rng: np.random.Generator, name: str) -> Any: ...
    def mutate(self, spec: Any, rng: np.random.Generator, sigma: float, name: str) -> Any: ...
    def crossover(self, a: Any, b: Any, rng: np.random.Generator, name: str) -> Any: ...
    def numeric_params(self, spec: Any) -> list[tuple[str, str, float, float, float]]: ...
    def with_unit_vector(self, spec: Any, x: np.ndarray, name: str) -> Any: ...
    def featurize(self, spec: Any) -> np.ndarray: ...
    def describe_for_llm(self) -> str: ...
    def summarize(self, spec: Any) -> str: ...


class Evaluator(Protocol):
    def evaluate(
        self,
        spec: Any,
        split: str,
        contract: str | None = ...,
        slippage_multiplier: float = ...,
        store: ResultsStore | None = ...,
        generator: str | None = ...,
        parents: list[str] | None = ...,
        engine_id: str | None = ...,
    ) -> Any: ...


@dataclass
class Evaluated:
    spec: Any
    fitness: float
    metrics: dict[str, Any]
    generator: str
    parents: list[str]


@dataclass
class SearchResult:
    run_id: str
    status: str
    evaluated: list[Evaluated]
    verdicts: list[Verdict]
    summary: dict[str, Any] = field(default_factory=dict)


def structure_key(spec: Any) -> str:
    """Identity of a spec's structure: everything except parameter values."""
    raw = json.loads(json.dumps(spec.raw))
    raw.pop("name", None)
    raw.pop("rationale", None)
    for p in raw["params"].values():
        p.pop("value", None)
    return hashlib.sha256(json.dumps(raw, sort_keys=True).encode()).hexdigest()[:16]


def allocate(mix: dict[str, float], n: int) -> dict[str, int]:
    """Largest-remainder split of n slots by the generator mix."""
    raw = {k: v * n for k, v in mix.items()}
    out = {k: math.floor(v) for k, v in raw.items()}
    for k in sorted(raw, key=lambda k: raw[k] - out[k], reverse=True)[: n - sum(out.values())]:
        out[k] += 1
    return out


class SearchRun:
    def __init__(
        self,
        space: SpecSpace,
        evaluator: Evaluator,
        store: ResultsStore,
        engine: EngineConfig,
        engine_id: str,
        budget: RunBudget,
        seed: int,
        min_trades: int,
        contract: str,
        llm: LLMClient | None = None,
        llm_model: str = "",
    ) -> None:
        self.space, self.ev, self.store = space, evaluator, store
        self.engine, self.engine_id, self.budget = engine, engine_id, budget
        self.seed, self.min_trades, self.contract = seed, min_trades, contract
        self.llm, self.llm_model = llm, llm_model
        self.rng = np.random.default_rng(seed)
        self.run_id = f"{engine_id}-s{seed}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}"
        self.done: dict[str, Evaluated] = {}
        self.surrogate = Surrogate(engine.surrogate, seed)
        self.cma = CMATuner(engine.cma, seed)
        self.log: list[dict[str, Any]] = []
        self.counts: dict[str, Any] = {
            "proposed": dict.fromkeys(("random", "genetic", "cma", "llm"), 0),
            "evaluated": dict.fromkeys(("random", "genetic", "cma", "llm"), 0),
            "duplicates": 0,
            "llm_invalid": 0,
            "llm_calls": 0,
        }

    # ---- fitness ---------------------------------------------------------------------------

    def fitness(self, metrics: dict[str, Any]) -> float:
        f = float(metrics["sharpe"])
        return f - self.engine.trade_penalty if metrics["trade_count"] < self.min_trades else f

    def ranked(self) -> list[Evaluated]:
        return sorted(self.done.values(), key=lambda e: e.fitness, reverse=True)

    # ---- generators ------------------------------------------------------------------------

    def _tournament(self) -> Evaluated:
        pool = self.ranked()[: max(self.engine.genetic.elite, 2)]
        picks = self.rng.choice(
            len(pool), size=min(self.engine.genetic.tournament, len(pool)), replace=False
        )
        return pool[int(min(picks))]  # ranked list: lower index = fitter

    def _gen_random(self, g: int, k: int) -> list[tuple[Any, str, list[str]]]:
        return [(self.space.random(self.rng, f"rnd_g{g}_{i}"), "random", []) for i in range(k)]

    def _gen_genetic(self, g: int, k: int) -> list[tuple[Any, str, list[str]]]:
        if len(self.done) < 2:
            return self._gen_random(g, k)
        out = []
        for i in range(k):
            a = self._tournament()
            if self.rng.random() < self.engine.genetic.crossover_rate:
                b = self._tournament()
                child = self.space.crossover(a.spec, b.spec, self.rng, f"gx_g{g}_{i}")
                parents = [a.spec.hash, b.spec.hash]
            else:
                child = self.space.mutate(
                    a.spec, self.rng, self.engine.genetic.sigma, f"gm_g{g}_{i}"
                )
                parents = [a.spec.hash]
            if child is not None:
                out.append((child, "genetic", parents))
        return out

    def _gen_cma(self, g: int, k: int) -> list[tuple[Any, str, list[str]]]:
        if not self.done or k == 0:
            return []
        best = self.ranked()[0]
        key = structure_key(best.spec)
        if self.cma.structure != key:
            params = self.space.numeric_params(best.spec)
            x0 = np.array([(v - lo) / (hi - lo) for _, _, lo, hi, v in params])
            self.cma.reset(key, x0)
        if not self.cma.active:
            return self._gen_genetic(g, k)
        out = []
        for i, x in enumerate(self.cma.ask()[:k]):
            try:
                spec = self.space.with_unit_vector(best.spec, x, f"cma_g{g}_{i}")
            except ValueError:
                continue
            self.cma.bind(x, spec.hash)
            out.append((spec, "cma", [best.spec.hash]))
        return out

    def _feedback(self) -> dict[str, Any]:
        e = self.engine.llm
        dead = [
            {
                "structure": json.loads(
                    self.space.summarize(self.space.validate(json.loads(d["spec_json"])))
                ),
                "killed_by": d["killed_by"],
                "search_trades": d["search_trades"],
                "search_sharpe": round(d["search_sharpe"], 2),
            }
            for d in self.store.dead_on_search(e.death_examples)
        ]
        best = [
            {
                "structure": json.loads(self.space.summarize(x.spec)),
                "search_sharpe": round(float(x.metrics["sharpe"]), 2),
                "search_trades": x.metrics["trade_count"],
            }
            for x in self.ranked()[: e.top_examples]
        ]
        return {
            "trials_so_far": len(self.done),
            "kill_counts_all_runs": self.store.kill_counts(),
            "recently_killed_on_search": dead,
            "best_this_run_on_search": best,
        }

    def _gen_llm(self, g: int, k: int) -> list[tuple[Any, str, list[str]]]:
        if self.llm is None or k == 0:
            return []
        e = self.engine.llm
        out: list[tuple[Any, str, list[str]]] = []
        for c in range(math.ceil(k / e.specs_per_call)):
            prompt = build_prompt(
                e.prompt_template, self.space.describe_for_llm(), self._feedback(), e.specs_per_call
            )
            prop = propose(self.llm, self.space, prompt, f"llm_g{g}_{c}")
            self.counts["llm_calls"] += 1
            self.counts["llm_invalid"] += prop.invalid
            self.store.record_llm_call(
                self.run_id,
                g,
                self.llm_model,
                hashlib.sha256(prompt.encode()).hexdigest(),
                prop.prompt_tokens,
                prop.completion_tokens,
                prop.dollars,
                prop.returned,
                len(prop.specs),
                prop.error,
            )
            if prop.error:
                break  # budget or provider problem: stop asking
            out += [(s, "llm", []) for s in prop.specs]
        return out

    # ---- the loop --------------------------------------------------------------------------

    def run(self) -> tuple[str, list[Evaluated]]:
        e = self.engine
        self.store.start_run(
            self.run_id, self.engine_id, self.seed, {"engine": e.model_dump(mode="json")}
        )
        status = "completed"
        try:
            for g in range(e.generations):
                self.budget.check()
                n = e.batch_size
                pool_n = math.ceil(n * e.surrogate.oversample) if self.surrogate.ready else n
                slots = allocate(e.mix.model_dump(), pool_n)
                batch = (
                    self._gen_llm(g, slots["llm"])
                    + self._gen_cma(g, slots["cma"])
                    + self._gen_genetic(g, slots["genetic"])
                    + self._gen_random(g, slots["random"])
                )
                for _, gen, _ in batch:
                    self.counts["proposed"][gen] += 1
                seen: set[str] = set()
                fresh = []
                for spec, gen, parents in batch:
                    if (
                        spec.hash in seen
                        or spec.hash in self.done
                        or self.store.has_spec_evaluation(spec.hash, "search")
                    ):
                        self.counts["duplicates"] += 1
                        continue
                    seen.add(spec.hash)
                    fresh.append((spec, gen, parents))
                predicted: dict[str, float] = {}
                if self.surrogate.ready and fresh:
                    preds = self.surrogate.predict(
                        np.vstack([self.space.featurize(s) for s, _, _ in fresh])
                    )
                    predicted = {
                        s.hash: float(p) for (s, _, _), p in zip(fresh, preds, strict=True)
                    }
                    fresh = sorted(fresh, key=lambda t: predicted[t[0].hash], reverse=True)[:n]
                gen_fit: dict[str, float] = {}
                for spec, gen, parents in fresh:
                    self.budget.reserve_evaluation()
                    res = self.ev.evaluate(
                        spec,
                        "search",
                        self.contract,
                        store=self.store,
                        generator=gen,
                        parents=parents,
                        engine_id=self.engine_id,
                    )
                    fit = self.fitness(res.metrics)
                    self.done[spec.hash] = Evaluated(spec, fit, res.metrics, gen, parents)
                    self.counts["evaluated"][gen] += 1
                    self.surrogate.add(self.space.featurize(spec), fit)
                    gen_fit[spec.hash] = fit
                self.cma.tell(gen_fit)
                acc = math.nan
                if predicted and gen_fit:
                    hs = [h for h in gen_fit if h in predicted]
                    acc = spearman(
                        np.array([predicted[h] for h in hs]), np.array([gen_fit[h] for h in hs])
                    )
                self.surrogate.fit()
                best = self.ranked()[0] if self.done else None
                self.log.append(
                    {
                        "generation": g,
                        "evaluated": len(gen_fit),
                        "best_fitness": round(best.fitness, 4) if best else None,
                        "surrogate_spearman": None if math.isnan(acc) else round(acc, 3),
                        "budget": self.budget.summary(),
                    }
                )
        except BudgetExceeded as exc:
            status = f"stopped:{exc.which}"
        return status, self.ranked()

    def top_candidates(self) -> list[Evaluated]:
        """Distinct top_k by fitness among specs that met the trade minimum."""
        ok = [x for x in self.ranked() if x.metrics["trade_count"] >= self.min_trades]
        return ok[: self.engine.top_k]

    def summary(self, status: str) -> dict[str, Any]:
        return {
            "status": status,
            "counts": self.counts,
            "generations": self.log,
            "budget": self.budget.summary(),
            "evaluated": len(self.done),
        }


def run_search(run: SearchRun, critics: CriticRunner | None) -> SearchResult:
    status, evaluated = run.run()
    verdicts = []
    if critics is not None:
        for cand in run.top_candidates():
            verdicts.append(critics.critique(cand.spec, generator=cand.generator))
    summary = run.summary(status) | {
        "critiqued": len(verdicts),
        "alive": sum(v.alive for v in verdicts),
        "verdicts": [
            {
                "spec": v.spec_name,
                "hash": v.spec_hash,
                "alive": v.alive,
                "killed": f"{v.killed_stage}:{v.killed_by}" if v.killed_by else None,
                "labels": v.labels,
            }
            for v in verdicts
        ],
    }
    run.store.finish_run(run.run_id, status, summary)
    return SearchResult(run.run_id, status, evaluated, verdicts, summary)
