"""Compare Vanilla (LRT) vs. the free-silence encoder: rate used and exponent.

Both encoders observe the same GaussianShift environment and are swept over
the same N grid (capped at N_MAX -- the exact fusion path is O(N) for the
2-symbol Vanilla alphabet but O(N^2) for Silence's 3-symbol alphabet, so
N=10^6 is infeasible; see docs/PLAN.md).

Usage: python figs/fig_vanilla_vs_silence.py [snr_db]   (default 0.0)
Each invocation writes three fresh, timestamped folders under runs/,
tagged with the SNR used, so repeated runs never collide or overwrite.
"""

import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ddms import EncoderBank, GaussianShift, LRTEncoder, SilenceEncoder, save_run
from ddms.sweep import sweep_over_N

N_MAX = 3000
T_LO, T_HI = 0.5, 2.0
snr_db = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
snr_tag = f"snr{'m' if snr_db < 0 else 'p'}{abs(snr_db):g}".replace(".", "_")

Ns = np.unique(np.geomspace(10, N_MAX, 25).astype(int))

model = GaussianShift.from_snr_db(snr_db)
vanilla = LRTEncoder(1.0)
silence = SilenceEncoder([T_LO, T_HI], [1])

model_meta = {"type": "GaussianShift", "snr_db": snr_db, "mu": model.mu, "sigma": model.sigma, "p": model.p}

group = f"vanilla_vs_silence_{snr_tag}"
session = "M2"

res_vanilla = sweep_over_N(model, lambda N: EncoderBank.identical(vanilla, N), Ns)
vanilla_dir = save_run(
    res_vanilla,
    label=f"vanilla_exponent_vs_N_{snr_tag}",
    meta={"model": model_meta, "bank": f"identical {vanilla.describe()}", "x": "N"},
    group=group,
    session=session,
)

res_silence = sweep_over_N(model, lambda N: EncoderBank.identical(silence, N), Ns)
res_silence["mean_rate_used"] = Ns * silence.mean_rate(model)
silence_dir = save_run(
    res_silence,
    label=f"silence_exponent_vs_N_{snr_tag}",
    meta={"model": model_meta, "bank": f"identical {silence.describe()}", "x": "N"},
    group=group,
    session=session,
)

stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
compare_dir = Path(vanilla_dir).parent / f"{stamp}_vanilla_vs_silence_compare_{snr_tag}"
compare_dir.mkdir(parents=True)

fig, (ax_rate, ax_exp) = plt.subplots(1, 2, figsize=(11, 4.5))

ax_rate.plot(Ns, res_vanilla["total_rate"], label="Vanilla, R = N")
ax_rate.plot(Ns, res_silence["mean_rate_used"], label="Silence, mean rate used")
ax_rate.plot(Ns, res_silence["total_rate"], "--", label="Silence, nominal (N log2 3)")
ax_rate.set_xlabel("N")
ax_rate.set_ylabel("rate (bits)")
ax_rate.set_title("Rate")
ax_rate.legend()

ax_exp.plot(Ns, res_vanilla["total_exponent"], label="Vanilla")
ax_exp.plot(Ns, res_silence["total_exponent"], label="Silence")
ax_exp.set_xlabel("N")
ax_exp.set_ylabel(r"$-\log J^N$")
ax_exp.set_title("Error exponent")
ax_exp.legend()

fig.suptitle(f"Vanilla vs. silence-is-free, SNR {snr_db:g} dB")
fig.tight_layout()
fig.savefig(compare_dir / "fig_vanilla_vs_silence.pdf")

params = {
    "snr_db": snr_db,
    "model": model_meta,
    "vanilla": vanilla.describe(),
    "silence": silence.describe(),
    "silence_mean_rate_per_sensor": silence.mean_rate(model),
    "N_range": [int(Ns[0]), int(Ns[-1])],
    "N_points": len(Ns),
}
print(f"parameters: {params}")
print(f"saved {vanilla_dir}")
print(f"saved {silence_dir}")
print(f"saved {compare_dir}")
