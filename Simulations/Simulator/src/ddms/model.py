"""Statistical models: ground truth H*, observations y, likelihood ratios l."""

from abc import ABC, abstractmethod

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp, ndtr


def chernoff_distance(q1, q2):
    """C(q1, q2) = -min_a log sum_u q2[u]^(1-a) * q1[u]^a for two pmfs."""
    with np.errstate(divide="ignore"):
        lq1 = np.log(np.asarray(q1, dtype=float))
        lq2 = np.log(np.asarray(q2, dtype=float))

    def objective(a):
        return logsumexp(a * lq1 + (1 - a) * lq2)

    res = minimize_scalar(objective, bounds=(1e-9, 1 - 1e-9), method="bounded")
    return -res.fun


class StatisticalModel(ABC):
    """Generates H* and, per sensor, the pair (y, l). Nothing downstream."""

    def __init__(self, p=0.5):
        self.p = p  # P(H* = H1)

    def sample_hypothesis(self, rng):
        return 1 if rng.random() < self.p else 2

    @abstractmethod
    def sample(self, j, n, rng):
        """n i.i.d. draws given H_j. Returns (y[n], l[n])."""

    @abstractmethod
    def likelihood_ratio(self, y):
        """Vectorized L(y) = f(y | H2) / f(y | H1). Pure function, no RNG."""

    @abstractmethod
    def lr_cdf(self, t, j):
        """P(l <= t | H_j), vectorized in t."""

    @abstractmethod
    def chernoff_information(self):
        """Unquantized ceiling, for benchmarking."""


class GaussianShift(StatisticalModel):
    """y ~ N(0, sigma^2) under H1, y ~ N(mu, sigma^2) under H2."""

    def __init__(self, mu, sigma=1.0, p=0.5):
        super().__init__(p)
        self.mu = float(mu)
        self.sigma = float(sigma)

    @classmethod
    def from_snr_db(cls, snr_db, sigma=1.0, p=0.5):
        # SNR = mu^2 / sigma^2
        return cls(sigma * 10 ** (snr_db / 20), sigma, p)

    def _mean(self, j):
        return 0.0 if j == 1 else self.mu

    def sample(self, j, n, rng):
        y = rng.normal(self._mean(j), self.sigma, n)
        return y, self.likelihood_ratio(y)

    def likelihood_ratio(self, y):
        y = np.asarray(y, dtype=float)
        return np.exp((self.mu * y - self.mu**2 / 2) / self.sigma**2)

    def lr_cdf(self, t, j):
        t = np.asarray(t, dtype=float)
        if self.mu == 0.0:
            return (t >= 1.0).astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            y_t = self.sigma**2 * np.log(t) / self.mu + self.mu / 2
        f = ndtr((y_t - self._mean(j)) / self.sigma)
        return np.where(t <= 0, 0.0, f)

    def chernoff_information(self):
        return self.mu**2 / (8 * self.sigma**2)


class DiscreteLR(StatisticalModel):
    """Finite observation alphabet y in {0, ..., K-1} given by two pmfs.

    Example 1 of the paper lives here."""

    def __init__(self, p1, p2, p=0.5):
        super().__init__(p)
        self.p1 = np.asarray(p1, dtype=float)
        self.p2 = np.asarray(p2, dtype=float)
        with np.errstate(divide="ignore"):
            self.l_values = np.where(self.p1 > 0, self.p2 / self.p1, np.inf)

    def _pmf(self, j):
        return self.p1 if j == 1 else self.p2

    def sample(self, j, n, rng):
        y = rng.choice(self.p1.size, size=n, p=self._pmf(j))
        return y, self.l_values[y]

    def likelihood_ratio(self, y):
        return self.l_values[np.asarray(y, dtype=int)]

    def lr_cdf(self, t, j):
        t = np.asarray(t, dtype=float)
        return (self.l_values <= t[..., None]) @ self._pmf(j)

    def chernoff_information(self):
        return chernoff_distance(self.p1, self.p2)
