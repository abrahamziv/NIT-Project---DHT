"""Total exponent vs N, and vs R, for the symmetric LRT bank, against the
Chernoff bound.

Template for sweep figures: sweep, save_run, plot into the run directory.
Includes both mandatory sweeps (see claude.md, Default figures): over N and
over R (max affordable sensors per rate budget).
"""

import matplotlib.pyplot as plt
import numpy as np

from ddms import EncoderBank, GaussianShift, LRTEncoder, save_run
from ddms.sweep import sweep_over_N, sweep_over_R

model = GaussianShift.from_snr_db(0.0)
enc = LRTEncoder(1.0)
model_meta = {"type": "GaussianShift", "mu": model.mu, "sigma": model.sigma, "p": model.p}
Ns = range(1, 41)
res = sweep_over_N(model, lambda N: EncoderBank.identical(enc, N), Ns)

run_dir = save_run(
    res,
    label="lrt_symmetric_exponent_vs_N",
    meta={"model": model_meta, "bank": f"identical {enc.describe()}", "x": "N"},
    session=f"M{enc.M}",
)

Rs = np.array(list(Ns)) * enc.rate()
res_r = sweep_over_R(model, lambda N: EncoderBank.identical(enc, N), enc.rate(), Rs)
run_r_dir = save_run(
    res_r,
    label="lrt_symmetric_exponent_vs_R",
    meta={"model": model_meta, "bank": f"identical {enc.describe()}", "x": "rate_used"},
    session=f"M{enc.M}",
)

fig, (ax_n, ax_r) = plt.subplots(1, 2, figsize=(11, 4.5))

ax_n.plot(res["N"], res["total_exponent"], label=r"$-\log J^N$ (exact)")
ax_n.plot(res["N"], res["chernoff_bound"], "--", label="Chernoff bound (eq. 24)")
ax_n.set_xlabel("N")
ax_n.set_ylabel("total exponent")
ax_n.set_title("Sweep over N")
ax_n.legend()

ax_r.plot(res_r["rate_used"], res_r["total_exponent"], label=r"$-\log J^N$ (exact)")
ax_r.set_xlabel("rate used (bits)")
ax_r.set_ylabel("total exponent")
ax_r.set_title("Sweep over R (max N per budget)")
ax_r.legend()

fig.suptitle(f"Symmetric LRT bank, {enc.describe()}, SNR 0 dB")
fig.tight_layout()
fig.savefig(run_dir / "fig_exponent_vs_N.pdf")
fig.savefig(run_r_dir / "fig_exponent_vs_R.pdf")
print(f"saved {run_dir}")
print(f"saved {run_r_dir}")
