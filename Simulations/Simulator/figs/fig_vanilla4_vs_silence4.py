"""Compare Vanilla (M=4 threshold quantizer) vs. its free-silence variant.

Both encoders share the same three thresholds {0.2, 1, 5} (M=4 bins).
Vanilla transmits all four bins; Silence stays silent on the two central
bins (B1 = (0.2, 1], B2 = (1, 5]) and only transmits on the two extreme
bins (B0 = "strongly H1", B3 = "strongly H2"). Both encoders are swept
over the same N grid, capped at N_MAX -- the exact fusion path is O(N^3)
for a 4-symbol identical-bank alphabet (compositions of N into 4 parts),
vs. O(N) for M=2 and O(N^2) for M=3. Measured: a single evaluation at
N=300 already takes ~47s, and a full 25-point sweep to N_MAX=300 would run
several minutes per encoder/SNR. N_MAX=150 keeps one 25-point sweep to
~20s (measured), ~2.7 minutes total across all 8 (encoder x SNR) sweeps.

Usage: python figs/fig_vanilla4_vs_silence4.py [snr_db]   (default 0.0)
Each invocation writes three fresh, timestamped folders under runs/,
tagged with the SNR used, so repeated runs never collide or overwrite.
"""

import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ddms import EncoderBank, GaussianShift, SilenceEncoder, ThresholdEncoder, save_run
from ddms.sweep import sweep_over_N

N_MAX = 150
THRESHOLDS = [0.2, 1, 5]
SILENT_INDICES = [1, 2]
snr_db = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
snr_tag = f"snr{'m' if snr_db < 0 else 'p'}{abs(snr_db):g}".replace(".", "_")

Ns = np.unique(np.geomspace(10, N_MAX, 25).astype(int))

model = GaussianShift.from_snr_db(snr_db)
vanilla4 = ThresholdEncoder(THRESHOLDS)
silence4 = SilenceEncoder(THRESHOLDS, SILENT_INDICES)

model_meta = {"type": "GaussianShift", "snr_db": snr_db, "mu": model.mu, "sigma": model.sigma, "p": model.p}
n_cap_note = (
    "N capped at 150 (vs 3000 for the M=3 silence run) because exact J^N is "
    "O(N^3) for this 4-symbol identical-bank alphabet (compositions of N "
    "into 4 parts): measured N=300 alone takes ~47s, N=150 ~6s; a full "
    "25-point sweep to N_MAX=150 takes ~20s."
)


def write_readme(run_dir, encoder, extra_lines=()):
    lines = [
        f"# {run_dir.name}",
        "",
        f"- Model: GaussianShift, SNR = {snr_db:g} dB (mu={model.mu:.4f}, sigma={model.sigma}), p={model.p}",
        f"- Encoder: {encoder.describe()}, |U|={encoder.M}",
        f"- N: {len(Ns)} points, geomspace({Ns[0]}, {Ns[-1]})",
        f"- Notes: {n_cap_note}",
    ]
    lines += list(extra_lines)
    (run_dir / "README.md").write_text("\n".join(lines) + "\n")


group = f"vanilla4_vs_silence4_{snr_tag}"
session = "M4"

res_vanilla = sweep_over_N(model, lambda N: EncoderBank.identical(vanilla4, N), Ns)
vanilla_dir = save_run(
    res_vanilla,
    label=f"vanilla4_exponent_vs_N_{snr_tag}",
    meta={"model": model_meta, "bank": f"identical {vanilla4.describe()}", "x": "N"},
    group=group,
    session=session,
)
write_readme(vanilla_dir, vanilla4)

res_silence = sweep_over_N(model, lambda N: EncoderBank.identical(silence4, N), Ns)
res_silence["mean_rate_used"] = Ns * silence4.mean_rate(model)
silence_dir = save_run(
    res_silence,
    label=f"silence4_exponent_vs_N_{snr_tag}",
    meta={"model": model_meta, "bank": f"identical {silence4.describe()}", "x": "N"},
    group=group,
    session=session,
)
write_readme(
    silence_dir,
    silence4,
    extra_lines=[f"- Mean rate used at this SNR: {silence4.mean_rate(model):.4f} bits/sensor"],
)

# SilenceEncoder never overrides cell_probs -- with the same thresholds as
# Vanilla4, its channel matrix (and therefore the FC's exact exponent) is
# identical to Vanilla4's, bit for bit. Plotting both as separate traces
# would just draw one exactly on top of the other; assert it and plot one.
assert np.allclose(res_vanilla["total_exponent"], res_silence["total_exponent"]), (
    "Vanilla4/Silence4 total_exponent diverged -- thresholds no longer match, "
    "the single-curve exponent/EE plots below are no longer valid"
)

SAME_EXPONENT_NOTE = (
    "Vanilla4 and Silence4 achieve the exact same total exponent " + r"$-\log J^N$" + " (one curve, not two overlapping ones).\n"
    "Why: both use thresholds {0.2, 1, 5} -> identical channel P(u|H); silence only zeroes the rate cost, never the FC's exact information."
)

stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
compare_dir = Path(vanilla_dir).parent / f"{stamp}_vanilla4_vs_silence4_compare_{snr_tag}"
compare_dir.mkdir(parents=True)

fig, (ax_rate, ax_exp, ax_ee) = plt.subplots(1, 3, figsize=(16, 4.5))

ax_rate.plot(Ns, res_vanilla["total_rate"], label="Vanilla4, R = 2N")
ax_rate.plot(Ns, res_silence["mean_rate_used"], label="Silence4, mean rate used")
ax_rate.plot(Ns, res_silence["total_rate"], "--", label="Silence4, nominal (N log2 4)")
ax_rate.set_xlabel("N")
ax_rate.set_ylabel("rate (bits)")
ax_rate.set_title("Rate")
ax_rate.legend()

ax_exp.plot(Ns, res_vanilla["total_exponent"], label="Vanilla4 = Silence4")
ax_exp.set_xlabel("N")
ax_exp.set_ylabel(r"$-\log J^N$")
ax_exp.set_title("Total exponent (identical)")
ax_exp.legend()

ax_ee.plot(Ns, res_vanilla["normalized_exponent"], label="Vanilla4 = Silence4")
ax_ee.set_xlabel("N")
ax_ee.set_ylabel(r"$J^N_{EE} = -\frac{1}{N}\log J^N$")
ax_ee.set_title("Normalized exponent (identical)")
ax_ee.legend()

fig.suptitle(f"Vanilla-M4 vs. silence-M4 (two central bins free), SNR {snr_db:g} dB")
fig.tight_layout(rect=(0, 0.06, 1, 1))
fig.text(0.5, 0.01, SAME_EXPONENT_NOTE, ha="center", va="bottom", fontsize=8)
fig.savefig(compare_dir / "fig_vanilla4_vs_silence4.pdf")

params = {
    "snr_db": snr_db,
    "model": model_meta,
    "vanilla4": vanilla4.describe(),
    "silence4": silence4.describe(),
    "silence4_mean_rate_per_sensor": silence4.mean_rate(model),
    "N_range": [int(Ns[0]), int(Ns[-1])],
    "N_points": len(Ns),
}
(compare_dir / "README.md").write_text(
    "\n".join(
        [
            f"# {compare_dir.name}",
            "",
            f"- Model: GaussianShift, SNR = {snr_db:g} dB (mu={model.mu:.4f}, sigma={model.sigma}), p={model.p}",
            f"- Vanilla4: {vanilla4.describe()}, |U|={vanilla4.M}",
            f"- Silence4: {silence4.describe()}, |U|={silence4.M}, mean rate = {silence4.mean_rate(model):.4f} bits/sensor",
            f"- N: {len(Ns)} points, geomspace({Ns[0]}, {Ns[-1]})",
            f"- Notes: {n_cap_note} Built from {vanilla_dir.name} and {silence_dir.name}.",
            "- Total/normalized exponent panels plot a single curve: Vanilla4 and "
            "Silence4 share thresholds {0.2, 1, 5}, hence identical cell_probs and "
            "identical exact exponent -- silence only zeroes the rate cost, never "
            "the FC's information. See the Rate panel for the actual difference.",
        ]
    )
    + "\n"
)

print(f"parameters: {params}")
print(f"saved {vanilla_dir}")
print(f"saved {silence_dir}")
print(f"saved {compare_dir}")
