"""Fusion center: MAP decisions and the exact error probability J^N.

The FC knows gamma^{1:N} only through the channel matrices P(u | H_j).
All probability arithmetic is in log domain; the law of the scalar
S = N * Delta_N is built as an N-fold convolution of per-sensor atomic
distributions, carrying log-weights under both hypotheses at once.
"""

import math

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import gammaln, logsumexp

# Cap on count vectors the exact path may enumerate: the product over groups
# of C(n_k + M_k - 1, M_k - 1). 1e8 keeps every size ever run (N=200, M=5 is
# 7e7) while refusing the combinatorially infeasible ones instead of hanging.
MAX_EXACT_TERMS = 100_000_000


def _atoms(enc, model):
    """Per-sensor atoms lambda_u = log P(u|H1) - log P(u|H2), with log-weights.

    Cells with zero mass under both hypotheses (padding) are dropped."""
    q1 = np.asarray(enc.cell_probs(model, 1), dtype=float)
    q2 = np.asarray(enc.cell_probs(model, 2), dtype=float)
    keep = (q1 > 0) | (q2 > 0)
    with np.errstate(divide="ignore"):
        lq1 = np.log(q1[keep])
        lq2 = np.log(q2[keep])
    return lq1 - lq2, lq1, lq2


def _compositions(n, m):
    if m == 1:
        yield (n,)
        return
    for k in range(n + 1):
        for rest in _compositions(n - k, m - 1):
            yield (k, *rest)


def _agg_logsumexp(inv, lw, size):
    """Groupwise logsumexp of lw over groups given by inv."""
    hi = np.full(size, -np.inf)
    np.maximum.at(hi, inv, lw)
    with np.errstate(invalid="ignore"):
        shifted = lw - hi[inv]
    tot = np.zeros(size)
    np.add.at(tot, inv, np.where(np.isnan(shifted), 0.0, np.exp(shifted)))
    with np.errstate(divide="ignore"):
        return hi + np.log(tot)


def _merge(vals, lw1, lw2):
    """Drop zero-weight atoms and merge identical atom values."""
    keep = ~(np.isneginf(lw1) & np.isneginf(lw2))
    vals, lw1, lw2 = vals[keep], lw1[keep], lw2[keep]
    uvals, inv = np.unique(vals, return_inverse=True)
    if uvals.size == vals.size:
        out1 = np.empty(uvals.size)
        out2 = np.empty(uvals.size)
        out1[inv] = lw1
        out2[inv] = lw2
        return uvals, out1, out2
    return uvals, _agg_logsumexp(inv, lw1, uvals.size), _agg_logsumexp(inv, lw2, uvals.size)


def _group_dist(lam, lq1, lq2, n):
    """Law of the sum of n i.i.d. atoms: multinomial over symbol counts."""
    lgn = gammaln(n + 1)
    vals, w1, w2 = [], [], []
    for counts in _compositions(n, lam.size):
        k = np.asarray(counts, dtype=float)
        pos = k > 0
        vals.append((k[pos] * lam[pos]).sum())
        coeff = lgn - gammaln(k + 1).sum()
        w1.append(coeff + (k[pos] * lq1[pos]).sum())
        w2.append(coeff + (k[pos] * lq2[pos]).sum())
    return _merge(np.asarray(vals), np.asarray(w1), np.asarray(w2))


def _convolve(d1, d2):
    v1, a1, b1 = d1
    v2, a2, b2 = d2
    with np.errstate(invalid="ignore"):
        vals = (v1[:, None] + v2[None, :]).ravel()
    lw1 = (a1[:, None] + a2[None, :]).ravel()
    lw2 = (b1[:, None] + b2[None, :]).ravel()
    return _merge(vals, lw1, lw2)


class FusionCenter:
    """Fixed component. Consumes channel matrices plus the prior."""

    def __init__(self, p=0.5, t=None, N_max=16):
        self.p = p  # P(H* = H1), taken from the model
        self.t = t  # decision threshold; None means t = log((1-p)/p) / N
        self.N_max = N_max  # cap on distinct policies in the convolution path

    def _Nt(self, N):
        if self.t is not None:
            return N * self.t
        return np.log((1 - self.p) / self.p)

    def decide(self, u, P1, P2):
        """MAP decision for an action profile u[N] (or batch u[..., N])."""
        u = np.asarray(u)
        N = P1.shape[0]
        p1 = P1[np.arange(N), u]
        p2 = P2[np.arange(N), u]
        if np.any((p1 == 0) & (p2 == 0)):
            raise ValueError("action indexes a zero-mass (padded) cell")
        with np.errstate(divide="ignore"):
            s = np.sum(np.log(p1) - np.log(p2), axis=-1)
        return np.where(s >= self._Nt(N), 1, 2)[()]

    def _delta_dist(self, bank, model):
        groups = bank.groups()
        if len(groups) > self.N_max:
            raise ValueError(
                f"{len(groups)} distinct policies exceeds N_max={self.N_max}; "
                "use empirical_error_prob instead"
            )
        atoms = [(_atoms(enc, model), n) for enc, n in groups]
        terms = math.prod(math.comb(n + a[0].size - 1, a[0].size - 1) for a, n in atoms)
        if terms > MAX_EXACT_TERMS:
            raise ValueError(
                f"exact path would enumerate {float(terms):.2e} count vectors "
                f"(cap {MAX_EXACT_TERMS:.0e}); use method='tilted'"
            )
        dists = [_group_dist(*a, n) for a, n in atoms]
        dist = dists[0]
        for d in dists[1:]:
            dist = _convolve(dist, d)
        return dist

    def _log_tails(self, bank, model, nt):
        """(log P(S < nt | H1), log P(S >= nt | H2)) -- the two error tails."""
        vals, lw1, lw2 = self._delta_dist(bank, model)
        below, above = vals < nt, vals >= nt
        le1 = logsumexp(lw1[below]) if below.any() else -np.inf  # P(H2 hat | H1)
        le2 = logsumexp(lw2[above]) if above.any() else -np.inf  # P(H1 hat | H2)
        return le1, le2

    def log_error_prob(self, bank, model):
        """Exact log J^N."""
        le1, le2 = self._log_tails(bank, model, self._Nt(len(bank)))
        return logsumexp([np.log(self.p) + le1, np.log1p(-self.p) + le2])

    def total_exponent(self, bank, model):
        """-log J^N. Headline metric."""
        return -self.log_error_prob(bank, model)

    def normalized_exponent(self, bank, model):
        """J^N_EE = -(1/N) log J^N."""
        return self.total_exponent(bank, model) / len(bank)

    def chernoff_bound(self, bank, model):
        """-min_a sum_i log M(a, gamma^i), DDMS eq. 24 (unnormalized)."""
        groups = [(n, _atoms(enc, model)) for enc, n in bank.groups()]

        def objective(a):
            return sum(
                n * logsumexp(a * lq1 + (1 - a) * lq2) for n, (_, lq1, lq2) in groups
            )

        res = minimize_scalar(objective, bounds=(1e-9, 1 - 1e-9), method="bounded")
        return -res.fun

    def empirical_error_prob(self, bank, model, T, rng):
        """Monte Carlo J^N estimate. Validation only, never plotted."""
        P1 = bank.cell_probs_matrix(model, 1)
        P2 = bank.cell_probs_matrix(model, 2)
        h = np.where(rng.random(T) < model.p, 1, 2)
        errors = 0
        for j in (1, 2):
            tj = int((h == j).sum())
            if tj == 0:
                continue
            u = np.empty((tj, len(bank)), dtype=int)
            for i, enc in enumerate(bank.encoders):
                _, l = model.sample(j, tj, rng)
                u[:, i] = enc.encode(l)
            errors += int((self.decide(u, P1, P2) != j).sum())
        return errors / T
