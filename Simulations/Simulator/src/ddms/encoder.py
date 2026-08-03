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
    """Free-silence variant of the LRT: only strong readings send.

    Thresholds t_lo < 1 < t_hi split l into three bins: below t_lo sends
    "0" (strong H1 evidence), above t_hi sends "1" (strong H2 evidence),
    the middle bin stays silent. Silence costs nothing to transmit; the
    two active symbols still need only 1 bit, same as LRTEncoder. So the
    nominal rate() (log2(3)) overstates what's actually sent -- use
    mean_rate() for the real, model-dependent cost.
    """

    def __init__(self, t_lo=0.5, t_hi=2.0):
        super().__init__([t_lo, t_hi])
        self.silent_index = 1

    def mean_rate(self, model):
        """Expected bits/sensor actually transmitted: 1 - P(silent | mixture)."""
        q1 = self.cell_probs(model, 1)
        q2 = self.cell_probs(model, 2)
        p_silent = model.p * q1[self.silent_index] + (1 - model.p) * q2[self.silent_index]
        return 1.0 - p_silent

    def describe(self):
        t_lo, t_hi = self.thresholds
        return f"SilenceEncoder(t_lo={t_lo}, t_hi={t_hi})"


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
