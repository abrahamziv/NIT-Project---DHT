"""Total exponent vs N for the symmetric LRT bank, against the Chernoff bound.

Template for sweep figures: sweep, save_run, plot into the run directory.
"""

import matplotlib.pyplot as plt

from ddms import EncoderBank, GaussianShift, LRTEncoder, save_run
from ddms.sweep import sweep_over_N

model = GaussianShift.from_snr_db(0.0)
enc = LRTEncoder(1.0)
Ns = range(1, 41)
res = sweep_over_N(model, lambda N: EncoderBank.identical(enc, N), Ns)

run_dir = save_run(
    res,
    label="lrt_symmetric_exponent_vs_N",
    meta={
        "model": {"type": "GaussianShift", "mu": model.mu, "sigma": model.sigma, "p": model.p},
        "bank": f"identical {enc.describe()}",
        "x": "N",
    },
    session=f"M{enc.M}",
)

fig, ax = plt.subplots()
ax.plot(res["N"], res["total_exponent"], label=r"$-\log J^N$ (exact)")
ax.plot(res["N"], res["chernoff_bound"], "--", label="Chernoff bound (eq. 24)")
ax.set_xlabel("N")
ax.set_ylabel("total exponent")
ax.set_title(f"Symmetric LRT bank, {enc.describe()}, SNR 0 dB")
ax.legend()
fig.savefig(run_dir / "fig_exponent_vs_N.pdf")
print(f"saved {run_dir}")
