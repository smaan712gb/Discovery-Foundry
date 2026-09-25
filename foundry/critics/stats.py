"""Statistics for the kill tests: PSR/DSR, PBO via CSCV, CPCV, concentration.

References: Bailey & López de Prado, "The Deflated Sharpe Ratio" (2014); Bailey, Borwein,
López de Prado & Zhu, "The Probability of Backtest Overfitting" (2015); López de Prado,
"Advances in Financial Machine Learning" (2018), ch. 7 and 12.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from statistics import NormalDist

import numpy as np

EULER_GAMMA = 0.5772156649015329
_N = NormalDist()


def moments(x: np.ndarray) -> tuple[float, float, float]:
    """Per-period Sharpe (mean / sample std), skewness, and raw (Pearson) kurtosis."""
    if x.size < 2:
        return math.nan, math.nan, math.nan
    sd = float(np.std(x, ddof=1))
    if sd == 0:
        return math.nan, math.nan, math.nan
    sr = float(np.mean(x)) / sd
    z = (x - np.mean(x)) / np.std(x)
    return sr, float(np.mean(z**3)), float(np.mean(z**4))


def probabilistic_sharpe(sr: float, sr_benchmark: float, t: int, skew: float, kurt: float) -> float:
    """P(true Sharpe > benchmark), all in per-period units. NaN if undefined."""
    if t < 2 or any(math.isnan(v) for v in (sr, sr_benchmark, skew, kurt)):
        return math.nan
    var = 1 - skew * sr + (kurt - 1) / 4 * sr**2
    if var <= 0:
        return math.nan
    return _N.cdf((sr - sr_benchmark) * math.sqrt(t - 1) / math.sqrt(var))


def expected_max_sharpe(n_trials: int, sr_variance: float) -> float:
    """Expected maximum of N trial Sharpes under the null (true Sharpe 0), per-period units."""
    if n_trials <= 1:
        return 0.0
    if math.isnan(sr_variance) or sr_variance < 0:
        return math.nan
    a = _N.inv_cdf(1 - 1 / n_trials)
    b = _N.inv_cdf(1 - 1 / (n_trials * math.e))
    return math.sqrt(sr_variance) * ((1 - EULER_GAMMA) * a + EULER_GAMMA * b)


@dataclass(frozen=True)
class DeflatedSharpe:
    dsr: float
    sharpe: float  # per period
    sr0: float  # deflated benchmark, per period
    n_trials: int
    sr_variance: float
    t: int
    skew: float
    kurt: float


def deflated_sharpe(daily: np.ndarray, n_trials: int, trial_sharpes: np.ndarray) -> DeflatedSharpe:
    """`trial_sharpes`: per-period Sharpe of every trial on this split (including this one)."""
    sr, skew, kurt = moments(daily)
    var = float(np.var(trial_sharpes, ddof=1)) if trial_sharpes.size >= 2 else math.nan
    sr0 = expected_max_sharpe(n_trials, var if n_trials > 1 else 0.0)
    return DeflatedSharpe(
        probabilistic_sharpe(sr, sr0, int(daily.size), skew, kurt),
        sr,
        sr0,
        n_trials,
        var,
        int(daily.size),
        skew,
        kurt,
    )


@dataclass(frozen=True)
class PBOResult:
    pbo: float
    n_splits: int
    n_configs: int
    logits: np.ndarray


def pbo_cscv(matrix: np.ndarray, n_blocks: int) -> PBOResult:
    """Probability of Backtest Overfitting by combinatorially symmetric cross-validation.

    `matrix`: T periods x N configurations of PnL. Rows are cut into `n_blocks` contiguous blocks;
    every choice of half the blocks is in-sample. Performance is Sharpe (mean / std) per column.
    """
    t, n = matrix.shape
    if n < 2 or n_blocks < 2 or n_blocks % 2 or t < n_blocks * 2:
        return PBOResult(math.nan, 0, n, np.array([]))
    blocks = np.array_split(np.arange(t), n_blocks)
    s1 = np.stack([matrix[b].sum(axis=0) for b in blocks])  # blocks x N
    s2 = np.stack([(matrix[b] ** 2).sum(axis=0) for b in blocks])
    cnt = np.array([b.size for b in blocks], float)
    combos = np.array(
        [[i in c for i in range(n_blocks)] for c in combinations(range(n_blocks), n_blocks // 2)],
        float,
    )
    is_s1, is_s2, is_n = combos @ s1, combos @ s2, combos @ cnt
    oos = 1 - combos
    oos_s1, oos_s2, oos_n = oos @ s1, oos @ s2, oos @ cnt

    def sharpe(s: np.ndarray, ss: np.ndarray, k: np.ndarray) -> np.ndarray:
        kk = k[:, None]
        mean = s / kk
        var = (ss - s**2 / kk) / (kk - 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.where(var > 0, mean / np.sqrt(np.maximum(var, 0)), 0.0)
        return np.asarray(out)

    is_perf = sharpe(is_s1, is_s2, is_n)
    oos_perf = sharpe(oos_s1, oos_s2, oos_n)
    best = np.argmax(is_perf, axis=1)
    best_oos = oos_perf[np.arange(len(best)), best]
    # Relative rank in (0, 1): (number of configs beaten or tied) / (N + 1).
    rank = (oos_perf <= best_oos[:, None]).sum(axis=1)
    omega = rank / (n + 1)
    logits = np.log(omega / (1 - omega))
    return PBOResult(float(np.mean(logits <= 0)), len(best), n, logits)


@dataclass(frozen=True)
class CPCVResult:
    positive_fraction: float
    mean_test_sharpe: float  # annualized
    n_splits: int
    test_pnl: np.ndarray


def cpcv_select_then_test(
    matrix: np.ndarray, groups: int, test_groups: int, purge: int, embargo: int, annualization: int
) -> CPCVResult:
    """Combinatorial purged CV of "pick the best column in training, trade it on the test days".

    `matrix`: T days x N configurations of daily PnL. Purge removes `purge` training days before
    each test group; embargo removes `embargo` training days after each test group.
    """
    t, n = matrix.shape
    if groups < 2 or not 1 <= test_groups < groups or t < groups * 2 or n < 1:
        return CPCVResult(math.nan, math.nan, 0, np.array([]))
    bounds = np.array_split(np.arange(t), groups)
    pnls, sharpes = [], []
    for test in combinations(range(groups), test_groups):
        test_mask = np.zeros(t, bool)
        train_mask = np.ones(t, bool)
        for g in test:
            idx = bounds[g]
            test_mask[idx] = True
            lo, hi = int(idx[0]), int(idx[-1])
            train_mask[max(0, lo - purge) : min(t, hi + 1 + embargo)] = False
        train_mask &= ~test_mask
        tr = matrix[train_mask]
        if tr.shape[0] < 2:
            continue
        sd = tr.std(axis=0, ddof=1)
        perf = np.where(sd > 0, tr.mean(axis=0) / np.where(sd > 0, sd, 1), -np.inf)
        best = int(np.argmax(perf))
        test_pnl = matrix[test_mask, best]
        pnls.append(float(test_pnl.sum()))
        tsd = float(np.std(test_pnl, ddof=1)) if test_pnl.size > 1 else 0.0
        sharpes.append(
            float(np.mean(test_pnl)) / tsd * math.sqrt(annualization) if tsd > 0 else 0.0
        )
    if not pnls:
        return CPCVResult(math.nan, math.nan, 0, np.array([]))
    arr = np.array(pnls)
    return CPCVResult(float(np.mean(arr > 0)), float(np.mean(sharpes)), len(pnls), arr)


def top_day_share(daily: np.ndarray, fraction: float) -> float:
    """Share of net profit from the best `fraction` of days. NaN if not profitable."""
    total = float(daily.sum())
    if daily.size == 0 or total <= 0:
        return math.nan
    k = max(1, math.ceil(fraction * daily.size))
    return float(np.sort(daily)[::-1][:k].sum()) / total
