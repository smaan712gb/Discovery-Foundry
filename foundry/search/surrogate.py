"""LightGBM surrogate: predicts search-split fitness from spec features, to pre-screen batches."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from foundry.search.engine import SurrogateConfig


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3 or np.all(a == a[0]) or np.all(b == b[0]):
        return math.nan
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


@dataclass
class Surrogate:
    cfg: SurrogateConfig
    seed: int
    X: list[np.ndarray] = field(default_factory=list)
    y: list[float] = field(default_factory=list)
    _model: object | None = None

    @property
    def ready(self) -> bool:
        return self.cfg.enabled and self._model is not None

    def add(self, x: np.ndarray, fitness: float) -> None:
        self.X.append(x)
        self.y.append(fitness)

    def fit(self) -> bool:
        if not self.cfg.enabled or len(self.y) < self.cfg.min_train:
            return False
        import lightgbm as lgb  # noqa: PLC0415 (heavy import, only when the surrogate trains)

        model = lgb.LGBMRegressor(
            n_estimators=self.cfg.n_estimators,
            num_leaves=self.cfg.num_leaves,
            learning_rate=0.05,
            min_child_samples=5,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            random_state=self.seed,
            verbose=-1,
            deterministic=True,
            force_row_wise=True,
        )
        model.fit(np.vstack(self.X), np.asarray(self.y, float))
        self._model = model
        return True

    def predict(self, X: np.ndarray) -> np.ndarray:  # noqa: N803
        if self._model is None:
            raise RuntimeError("surrogate not trained")
        pred: np.ndarray = self._model.predict(X)  # type: ignore[attr-defined]
        return np.asarray(pred, dtype=float)
