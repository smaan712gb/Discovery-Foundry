"""CMA-ES over the numeric parameters of one spec structure, in the unit cube."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

with warnings.catch_warnings():
    warnings.simplefilter("ignore")  # cma warns when matplotlib is absent; plotting isn't used
    import cma

from foundry.search.engine import CMAConfig


class CMATuner:
    """Tunes the current best structure. Restarts when the structure being tuned changes."""

    def __init__(self, cfg: CMAConfig, seed: int) -> None:
        self.cfg = cfg
        self.seed = seed
        self.structure: str | None = None
        self._es: Any = None
        self._pending: dict[str, np.ndarray] = {}
        self._asked: list[np.ndarray] = []

    def reset(self, structure: str, x0: np.ndarray) -> None:
        self.structure = structure
        self._pending.clear()
        self._asked = []
        if x0.size < 2:
            self._es = None  # CMA-ES needs at least 2 dimensions
            return
        self._es = cma.CMAEvolutionStrategy(
            np.clip(x0, 0, 1).tolist(),
            self.cfg.sigma0,
            {"bounds": [0, 1], "popsize": self.cfg.popsize, "seed": self.seed + 1, "verbose": -9},
        )

    @property
    def active(self) -> bool:
        return self._es is not None

    def ask(self) -> list[np.ndarray]:
        if self._es is None:
            return []
        self._asked = [np.asarray(x) for x in self._es.ask()]
        return self._asked

    def bind(self, x: np.ndarray, spec_hash: str) -> None:
        self._pending[spec_hash] = x

    def tell(self, fitness_by_hash: dict[str, float]) -> None:
        """Report fitness for asked points (unevaluated or duplicate points get the worst value)."""
        if self._es is None or not self._asked:
            return
        worst = min(fitness_by_hash.values(), default=0.0) - 1.0
        by_x = {
            tuple(np.round(x, 12)): f
            for h, x in self._pending.items()
            if (f := fitness_by_hash.get(h)) is not None
        }
        # cma minimizes; fitness is maximized.
        values = [-by_x.get(tuple(np.round(x, 12)), worst) for x in self._asked]
        self._es.tell(self._asked, values)
        self._pending.clear()
        self._asked = []
