"""Encoders: per-sensor quantization policies gamma^i, and banks of them.

An encoder is the swappable component of the simulator. Everything downstream
depends on it only through the induced 2 x M channel matrix P(u | H_j).
Actions are 0-indexed internally.
"""

from abc import ABC, abstractmethod

import numpy as np


class Encoder(ABC):
    """One sensor's policy. Carries its own alphabet size M."""

    def __init__(self, M):
        self.M = M

    @abstractmethod
    def encode(self, l):
        """Vectorized l -> u, integer actions in {0, ..., M-1}."""

    @abstractmethod
    def cell_probs(self, model, j):
        """Exact P(u | H_j) for u = 0, ..., M-1. Shape (M,), sums to 1."""

    def rate(self):
        return float(np.log2(self.M))

    def describe(self):
        return f"{type(self).__name__}(M={self.M})"


class VanillaEncoder(Encoder):
    """Empty template for the next encoder design. Fill in later.

    Contract:
      encode(l)            vectorized map from likelihood ratios to actions
                           in {0, ..., M-1}; used for Monte Carlo checks.
      cell_probs(model, j) exact P(u | H_j), shape (M,), nonnegative, sums
                           to 1; this is the real interface the FC consumes.
                           Use model.lr_cdf for exact cell masses.
    Optionally override describe() with a label for logs and legends.
    """

    def __init__(self, M):
        super().__init__(M)
        # TODO: parameters of the policy

    def encode(self, l):
        raise NotImplementedError("TODO: map likelihood ratios to actions")

    def cell_probs(self, model, j):
        raise NotImplementedError("TODO: exact P(u | H_j) via model.lr_cdf")


class ThresholdEncoder(Encoder):
    """The paper's threshold policy (Definition 1).

    Thresholds 0 < t_1 < ... < t_{m-1} define bins
    B_0 = [0, t_1], B_d = (t_d, t_{d+1}], B_{M-1} = (t_{m-1}, inf).
    """

    def __init__(self, thresholds):
        t = np.atleast_1d(np.asarray(thresholds, dtype=float))
        super().__init__(t.size + 1)
        self.thresholds = t

    def encode(self, l):
        return np.searchsorted(self.thresholds, np.asarray(l), side="left")

    def cell_probs(self, model, j):
        f = np.concatenate(([0.0], np.atleast_1d(model.lr_cdf(self.thresholds, j)), [1.0]))
        return np.diff(f)

    def describe(self):
        return f"ThresholdEncoder(t={self.thresholds.tolist()})"


class LRTEncoder(ThresholdEncoder):
    """M = 2 single-threshold likelihood ratio test."""

    def __init__(self, t=1.0):
        super().__init__([t])


class SilenceEncoder(ThresholdEncoder):
    """Free-silence variant of the threshold policy: only strong readings send.

    thresholds define the usual M = len(thresholds) + 1 bins; silent_indices
    names which of those bins are free to "occupy" (no transmission cost).
    The remaining active bins still need ceil(log2(#active)) bits whenever
    used. So the nominal rate() (log2(M)) overstates what's actually sent --
    use mean_rate() for the real, model-dependent cost.

    Example: SilenceEncoder([0.5, 2.0], [1]) is the M=3 case (silence
    between two thresholds, both active symbols need 1 bit).
    """

    def __init__(self, thresholds, silent_indices):
        super().__init__(thresholds)
        self.silent_indices = tuple(sorted(silent_indices))
        n_active = self.M - len(self.silent_indices)
        self._active_bits = float(np.ceil(np.log2(n_active)))

    def mean_rate(self, model):
        """Expected bits/sensor actually transmitted."""
        q1 = self.cell_probs(model, 1)
        q2 = self.cell_probs(model, 2)
        silent = list(self.silent_indices)
        p_silent = model.p * q1[silent].sum() + (1 - model.p) * q2[silent].sum()
        return self._active_bits * (1.0 - p_silent)

    def describe(self):
        return f"SilenceEncoder(t={self.thresholds.tolist()}, silent={list(self.silent_indices)})"


class RBitThresholdEncoder(SilenceEncoder):
    """The DDMS section 5 policy gamma^R_{Delta,delta} (eq. general-thresholds).

    L = 2^R regions of width Delta, L/2 per side of a silence bin of
    half-width delta centered on mu/2. Thresholds are built in y-space,

        t_j = mu/2 - delta - (L/2 - j) Delta,      j = 1, ..., L/2
        t_j = mu/2 + delta + (j - L/2 - 1) Delta,  j = L/2+1, ..., L

    then mapped to likelihood-ratio space (L(y) is increasing) for the
    ThresholdEncoder machinery. At delta = 0 the silence bin degenerates to a
    point: t_{L/2} = t_{L/2+1} = mu/2, only the L-1 distinct thresholds are
    kept, and no symbol is silent -- this is gamma^V. At delta > 0 the middle
    bin (index L/2) is silent -- gamma^S.
    """

    def __init__(self, model, R, Delta, delta=0.0):
        if delta < 0:
            raise ValueError("delta must be >= 0")
        L = 2**R
        if L >= 4 and Delta <= 0:
            raise ValueError("Delta must be > 0 for R >= 2")
        mu = model.mu
        j = np.arange(1, L + 1)
        y = np.where(
            j <= L // 2,
            mu / 2 - delta - (L // 2 - j) * Delta,
            mu / 2 + delta + (j - L // 2 - 1) * Delta,
        )
        if delta == 0.0:
            y = np.delete(y, L // 2)  # t_{L/2} == t_{L/2+1}: drop the duplicate
            silent = ()
        else:
            silent = (L // 2,)
        super().__init__(model.likelihood_ratio(y), silent)
        self.R = int(R)
        self.L = L
        self.Delta = float(Delta)
        self.delta = float(delta)
        self.y_thresholds = y

    def rate(self):
        # R bits identify the L active regions; the base log2(M) would
        # overstate this as log2(L+1) whenever the silence symbol exists.
        return float(self.R)

    def describe(self):
        return f"RBitThresholdEncoder(R={self.R}, Delta={self.Delta:g}, delta={self.delta:g})"


class EncoderBank:
    """The profile gamma^{1:N}: a list of encoders, possibly with different M."""

    def __init__(self, encoders):
        self.encoders = list(encoders)

    @classmethod
    def identical(cls, encoder, N):
        return cls([encoder] * N)

    @classmethod
    def from_fractions(cls, pairs, N):
        """pairs = [(encoder, c_k)]: K distinct policies used by fractions c_k.

        Counts are assigned by largest remainder so they sum to N."""
        fracs = np.array([c for _, c in pairs], dtype=float)
        counts = np.floor(fracs * N).astype(int)
        remainder = N - counts.sum()
        order = np.argsort(-(fracs * N - counts))
        counts[order[:remainder]] += 1
        encoders = []
        for (enc, _), n in zip(pairs, counts):
            encoders += [enc] * n
        return cls(encoders)

    def __len__(self):
        return len(self.encoders)

    def groups(self):
        """[(encoder, count)] grouped by object identity, first-seen order."""
        seen = {}
        for enc in self.encoders:
            key = id(enc)
            if key in seen:
                seen[key][1] += 1
            else:
                seen[key] = [enc, 1]
        return [(enc, n) for enc, n in seen.values()]

    def M_list(self):
        return [enc.M for enc in self.encoders]

    def total_rate(self):
        return float(sum(enc.rate() for enc in self.encoders))

    def uniform_alphabet(self):
        return len(set(self.M_list())) <= 1

    def cell_probs_matrix(self, model, j):
        """P[N, M_max], zero-padded for ragged alphabets."""
        m_max = max(self.M_list())
        P = np.zeros((len(self.encoders), m_max))
        for i, enc in enumerate(self.encoders):
            P[i, : enc.M] = enc.cell_probs(model, j)
        return P
