import numpy as np
from scipy.stats import kstest

from ddms import GaussianShift


def test_lr_mean_under_h1_tends_to_one():
    model = GaussianShift(1.0)
    _, l = model.sample(1, 1_000_000, np.random.default_rng(1))
    assert abs(l.mean() - 1.0) < 0.01


def test_lr_cdf_matches_empirical_cdf():
    model = GaussianShift(1.2)
    for j in (1, 2):
        _, l = model.sample(j, 20_000, np.random.default_rng(j))
        res = kstest(l, lambda t: model.lr_cdf(t, j))
        assert res.pvalue > 0.01


def test_sample_consistent_with_likelihood_ratio():
    model = GaussianShift(0.7, sigma=1.3)
    y, l = model.sample(2, 1000, np.random.default_rng(2))
    np.testing.assert_array_equal(model.likelihood_ratio(y), l)


def test_zero_mu_is_degenerate():
    model = GaussianShift(0.0)
    _, l = model.sample(1, 100, np.random.default_rng(3))
    np.testing.assert_array_equal(l, 1.0)
    assert model.chernoff_information() == 0.0


def test_lr_cdf_monotone_with_limits():
    model = GaussianShift(1.0)
    t = np.concatenate(([0.0], np.logspace(-8, 8, 200)))
    for j in (1, 2):
        f = model.lr_cdf(t, j)
        assert np.all(np.diff(f) >= 0)
        assert f[0] == 0.0
        assert f[-1] > 1 - 1e-9
        assert model.lr_cdf(np.inf, j) == 1.0
