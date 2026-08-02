"""Two-group alphabet split v1: k sensors at M_hi, N-k at M_lo.

Spec: docs/experiments/two_group_split_v1.md. Self-contained -- no changes
to src/ddms. Total rate is not matched across k in v1; it is N + k bits.
"""

import matplotlib.pyplot as plt
import numpy as np

from ddms import EncoderBank, GaussianShift, ThresholdEncoder, save_run
from ddms.fusion import FusionCenter

N, M_hi, M_lo, snr_db = 20, 4, 2, 0.0
model = GaussianShift.from_snr_db(snr_db)
fc = FusionCenter(p=model.p)


def quantizer(M):
    """Uniform M-cell quantizer on y, centred at mu/2 and spanning +-2 sigma.

    Centring at mu/2 makes it symmetric in the two hypotheses, so M = 2 is
    exactly the LRT at l = 1."""
    y = model.mu / 2 + model.sigma * np.linspace(-2, 2, M + 1)[1:-1]
    return ThresholdEncoder(model.likelihood_ratio(y))


hi, lo = quantizer(M_hi), quantizer(M_lo)
ks = np.arange(N + 1)
banks = [EncoderBank([hi] * k + [lo] * (N - k)) for k in ks]

res = {
    "k": ks,
    "total_exponent": np.array([fc.total_exponent(b, model) for b in banks]),
    "chernoff_bound": np.array([fc.chernoff_bound(b, model) for b in banks]),
    "total_rate": np.array([b.total_rate() for b in banks]),
    "uniform_alphabet": np.array([b.uniform_alphabet() for b in banks]),
}

run_dir = save_run(
    res,
    label="two_group_split_v1",
    meta={
        "model": {"type": "GaussianShift", "mu": model.mu, "sigma": model.sigma, "p": model.p},
        "snr_db": snr_db,
        "N": N,
        "M_hi": M_hi,
        "M_lo": M_lo,
        "bank": f"k x {hi.describe()} + (N-k) x {lo.describe()}",
        "x": "k",
    },
)

fig, ax = plt.subplots()
ax.plot(res["k"], res["total_exponent"], "o-", ms=3, label=r"$-\log J^N$ (exact)")
ax.plot(res["k"], res["chernoff_bound"], "--", label="Chernoff bound (eq. 24)")
ax.set_xlabel(f"k (sensors at $M_{{hi}}={M_hi}$; total rate = {N}+k bits)")
ax.set_ylabel("total exponent")
ax.set_title(f"Two-group split, N={N}, $M_{{hi}}={M_hi}$ / $M_{{lo}}={M_lo}$, SNR {snr_db:g} dB")
ax.legend()
fig.tight_layout()
fig.savefig(run_dir / "fig_two_group_split_v1.pdf")
print(f"saved {run_dir}")
