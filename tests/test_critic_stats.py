"""Kill-test statistics against formulas, Monte Carlo, and known cases."""

from __future__ import annotations

import math

import numpy as np
import pytest

from foundry.critics.stats import (
    cpcv_select_then_test,
    deflated_sharpe,
    expected_max_sharpe,
    moments,
    pbo_cscv,
    probabilistic_sharpe,
    top_day_share,
)


def test_expected_max_sharpe_matches_monte_carlo() -> None:
    rng = np.random.default_rng(0)
    for n in (10, 100, 1000):
        mc = rng.standard_normal((4000, n)).max(axis=1).mean()
        assert expected_max_sharpe(n, 1.0) == pytest.approx(mc, rel=0.03)
    assert expected_max_sharpe(1, 1.0) == 0.0
    # scales with the standard deviation of trial Sharpes
    assert expected_max_sharpe(100, 4.0) == pytest.approx(2 * expected_max_sharpe(100, 1.0))


def test_psr_formula_by_hand() -> None:
    # SR 0.1 per period, benchmark 0, T = 253, normal returns (skew 0, kurt 3)
    sr, t = 0.1, 253
    expected = 0.5 * (1 + math.erf(0.1 * math.sqrt(252) / math.sqrt(1 + 0.5 * 0.01) / math.sqrt(2)))
    assert probabilistic_sharpe(sr, 0.0, t, 0.0, 3.0) == pytest.approx(expected)
    # fat tails and negative skew lower confidence in the same Sharpe
    assert probabilistic_sharpe(sr, 0.0, t, -1.0, 8.0) < probabilistic_sharpe(sr, 0.0, t, 0.0, 3.0)


def test_moments() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0, 10.0])
    sr, skew, kurt = moments(x)
    assert sr == pytest.approx(x.mean() / x.std(ddof=1))
    assert skew > 0
    assert kurt > 1
    assert math.isnan(moments(np.array([1.0, 1.0, 1.0]))[0])


def test_dsr_falls_as_trials_grow() -> None:
    rng = np.random.default_rng(1)
    daily = rng.normal(0.08, 1.0, 250)
    trials = rng.normal(0.0, 0.06, 500)
    one = deflated_sharpe(daily, 1, trials[:1])
    many = deflated_sharpe(daily, 500, trials)
    assert one.sr0 == 0.0
    assert one.dsr == pytest.approx(probabilistic_sharpe(one.sharpe, 0.0, 250, one.skew, one.kurt))
    assert many.dsr < one.dsr
    assert many.sr0 > 0


def test_pbo_near_half_for_pure_noise_and_low_for_a_real_edge() -> None:
    # One noise dataset's PBO is itself noisy (sd ~0.22), so check the mean over datasets.
    vals = [pbo_cscv(np.random.default_rng(s).standard_normal((480, 12)), 16) for s in range(30)]
    assert vals[0].n_splits == 12870
    assert 0.35 < float(np.mean([v.pbo for v in vals])) < 0.65
    rng = np.random.default_rng(2)
    edge = rng.standard_normal((480, 12))
    edge[:, 3] += 0.4  # one configuration genuinely better in every period
    assert pbo_cscv(edge, 16).pbo < 0.05


def test_pbo_undefined_cases() -> None:
    assert math.isnan(pbo_cscv(np.zeros((100, 1)), 16).pbo)  # one config
    assert math.isnan(pbo_cscv(np.zeros((10, 5)), 16).pbo)  # too few periods
    assert math.isnan(pbo_cscv(np.zeros((100, 5)), 15).pbo)  # odd block count


def test_cpcv_rewards_a_consistent_edge_and_purges() -> None:
    rng = np.random.default_rng(3)
    good = rng.normal(0.3, 1.0, (300, 5))
    res = cpcv_select_then_test(good, 6, 2, purge=0, embargo=1, annualization=252)
    assert res.n_splits == 15
    assert res.positive_fraction >= 0.9
    assert res.mean_test_sharpe > 0
    bad = rng.normal(0.0, 1.0, (300, 5))
    bad[:150] += 0.5  # only works in the first half
    res_bad = cpcv_select_then_test(bad, 6, 2, purge=0, embargo=1, annualization=252)
    assert res_bad.positive_fraction < res.positive_fraction


def test_top_day_share() -> None:
    daily = np.array([100.0] + [1.0] * 99)
    assert top_day_share(daily, 0.05) == pytest.approx((100 + 4) / 199)
    assert math.isnan(top_day_share(np.array([-1.0, -2.0]), 0.05))
