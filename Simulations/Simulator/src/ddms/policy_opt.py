"""Optimal (Delta, delta) for the DDMS section 5 R-bit threshold policy.

The objective is the closed-form Chernoff bound C(R, Delta, delta)
= -log M(0.5; R, Delta, delta) (DDMS eq. general-closed-form). At an optimum
every free threshold t_j satisfies the stationarity condition

    L(t_j)^2 = (q_j q_{j+1}) / (p_j p_{j+1}),

the generalization of the paper's one-bit tau_2*^2 = P(u^2|H2)/P(u^2|H1)
(DDMS eq. opt-tau); under the two-parameter ladder, (Delta*, delta*) solve the
corresponding chain-rule sums. No closed form for R > 1, so the optimization
here is numeric: multistart Nelder-Mead (the surface has flat shoulders that
defeat a single start).
"""

import numpy as np
from scipy.optimize import brentq, minimize, minimize_scalar
from scipy.special import log_ndtr, logsumexp

from .encoder import RBitThresholdEncoder, ThresholdEncoder
from .fusion import _cgf, _hyp_atoms, _saddlepoint
from .model import chernoff_distance


def chernoff_C(model, R, Delta, delta=0.0):
    """C(R, Delta, delta): per-sensor Chernoff bound of gamma^R_{Delta,delta}."""
    enc = RBitThresholdEncoder(model, R, Delta, delta)
    return chernoff_distance(enc.cell_probs(model, 1), enc.cell_probs(model, 2))


def chernoff_alpha_star(model, encoder):
    """The minimizing alpha of M(alpha); DDMS section 5 argues it is 0.5."""
    with np.errstate(divide="ignore"):
        lq1 = np.log(encoder.cell_probs(model, 1))
        lq2 = np.log(encoder.cell_probs(model, 2))
    res = minimize_scalar(
        lambda a: logsumexp(a * lq1 + (1 - a) * lq2),
        bounds=(1e-9, 1 - 1e-9),
        method="bounded",
    )
    return float(res.x)


def optimal_params(model, R, silence=True):
    """Maximize C(R, Delta, delta) over the ladder parameters.

    Returns (Delta, delta, C). silence=False pins delta = 0 (Setup A);
    Setup A at R = 1 has no free parameter at all (the single threshold sits
    at mu/2), and Setup B at R = 1 has only delta free (Delta is inert).
    """
    if R == 1:
        if not silence:
            return 1.0, 0.0, chernoff_C(model, 1, 1.0, 0.0)
        res = minimize_scalar(
            lambda d: -chernoff_C(model, 1, 1.0, max(d, 0.0)),
            bounds=(0.0, 5.0 * model.sigma),
            method="bounded",
            options={"xatol": 1e-12},
        )
        d = float(res.x)
        return 1.0, d, chernoff_C(model, 1, 1.0, d)

    if not silence:
        res = minimize_scalar(
            lambda D: -chernoff_C(model, R, D, 0.0),
            bounds=(1e-4, 20.0 * model.sigma),
            method="bounded",
            options={"xatol": 1e-12},
        )
        D = float(res.x)
        return D, 0.0, chernoff_C(model, R, D, 0.0)

    def objective(v):
        return -chernoff_C(model, R, max(v[0], 1e-6), max(v[1], 0.0))

    best = None
    for D0 in (0.3, 0.6, 1.0, 2.0):
        for d0 in (0.05, 0.3, 0.8):
            res = minimize(
                objective,
                [D0 * model.sigma, d0 * model.sigma],
                method="Nelder-Mead",
                options={"xatol": 1e-10, "fatol": 1e-13, "maxiter": 5000},
            )
            if best is None or res.fun < best.fun:
                best = res
    D, d = max(float(best.x[0]), 1e-6), max(float(best.x[1]), 0.0)
    return D, d, -float(best.fun)


def optimal_delta_R1(model):
    """Closed-form delta* at R = 1 (DDMS eq. opt-tau): e^{mu d/s^2} = sqrt((1-q-)/q+).

    Solved in log domain -- 1 - q_- and q_+ are both normal tails, and the
    ratio degenerates to 0/0 in floating point for large delta."""
    mu, sigma = model.mu, model.sigma

    def g(d):
        # log(1 - q_-) = log Phi((mu/2 - d)/sigma), log q_+ = log Phi(-(mu/2 + d)/sigma)
        rhs = 0.5 * (log_ndtr((mu / 2 - d) / sigma) - log_ndtr(-(mu / 2 + d) / sigma))
        return mu * d / sigma**2 - rhs

    return brentq(g, 1e-12, 10.0 * sigma, xtol=1e-14)


def free_threshold_optimum(model, R):
    """Best C over ALL L unconstrained y-thresholds (no ladder), seeded from
    the ladder optimum. Gauges how little the two-parameter family gives up."""
    D, d, _ = optimal_params(model, R)
    seed = RBitThresholdEncoder(model, R, D, max(d, 1e-3)).y_thresholds

    def objective(y):
        t = model.likelihood_ratio(np.sort(y))
        enc = ThresholdEncoder(t)
        return -chernoff_distance(enc.cell_probs(model, 1), enc.cell_probs(model, 2))

    res = minimize(
        objective,
        seed,
        method="Nelder-Mead",
        options={"xatol": 1e-10, "fatol": 1e-13, "maxiter": 200000, "maxfev": 200000},
    )
    return np.sort(res.x), -float(res.fun)


def bahadur_rao_exponent(bank, model, nt=None):
    """Normalized exponent via the first-order Bahadur-Rao tail approximation.

    Same CGF as the tilted backend, no FFT and no grid: each tail is
    exp(Lambda(theta*) - theta* nt) / (|theta*| sqrt(2 pi Lambda''(theta*))).
    A large-N asymptotic -- agreement with the true exponent tightens with N.
    nt defaults to the MAP threshold log((1-p)/p) (0 at p = 1/2).
    """
    p = model.p
    if nt is None:
        nt = np.log((1 - p) / p)

    def log_tail(parts):
        theta = _saddlepoint(parts, nt)
        cgf, _, var = _cgf(theta, parts)
        pref = sum(n * log_f for _, _, log_f, n in parts)
        return pref + cgf - theta * nt - np.log(abs(theta) * np.sqrt(2 * np.pi * var))

    parts = {j: [(*_hyp_atoms(e, model, j), n) for e, n in bank.groups()] for j in (1, 2)}
    log_pe = logsumexp(
        [np.log(p) + log_tail(parts[1]), np.log1p(-p) + log_tail(parts[2])]
    )
    return -log_pe / len(bank)
