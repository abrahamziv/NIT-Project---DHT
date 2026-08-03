"""Heterogeneous alphabet sizes at matched total rate: static, non-adaptive.

Vanilla4 gives every sensor the same M=4 quantizer (2.0 bits/sensor).
The mixed bank hands out three *different* fixed alphabet sizes -- M=4,
M=3, M=2 -- to a third of the sensors each, so it sits outside Assumption
1(iii) on purpose. All three encoders reuse vanilla4's own thresholds
{0.2, 1, 5}; dropping a threshold keeps the symmetry of the log-LR around
the LRT point t=1 (bin d under H1 = bin M-1-d under H2), so no new
thresholds are invented here.

The mixed bank averages 1.528 bits/sensor, not 2.0, so the two schemes are
compared at matched *total rate*, never at matched N: each rate target is
turned into the N that scheme needs to spend it. Both curves are plotted
against the rate the bank actually achieves, since from_fractions rounds
group counts and the achieved rate drifts slightly off the target.

This isolates the alphabet-size effect alone. It is NOT adaptive rate
allocation -- every sensor's alphabet is fixed in advance, independent of
its own observation. The adaptive comparison is a separate experiment.

Rate grid floor is 30 sensors, not 10: below N=30 the largest-remainder
rounding in from_fractions distorts the 1/3-1/3-1/3 split enough to matter
(N=10 gives counts (4,3,3), not (3.3,3.3,3.3)).

Timing (measured, SNR 0 dB): one log_error_prob at the top rate (300 bits)
costs 12.7s for the mixed bank at N=196 and 10.2s for vanilla4 at N=150.
The mixed bank is *not* faster despite its M=4 subgroup being only 66
sensors: profiling the atom support shows the per-group multinomials are
cheap (22859 + 797 + 66 merged atoms, ~1s total) and the cost is the
cross-group convolution. The M=2 group's log-LR atoms are incommensurate
with the M=4/M=3 lattice, so almost nothing merges -- the final convolution
takes 19.0M raw pairs to 13.7M distinct atoms (72% survive), against
585k -> 142k (76% merged away) for the single-group vanilla4 path.

Usage: python figs/fig_heterogeneous_M.py [snr_db]   (default 0.0)
"""

import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ddms import (
    EncoderBank,
    FusionCenter,
    GaussianShift,
    LRTEncoder,
    ThresholdEncoder,
    save_run,
)
from ddms.sweep import sweep_over_rate

THRESHOLDS = [0.2, 1, 5]
snr_db = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
snr_tag = f"snr{'m' if snr_db < 0 else 'p'}{abs(snr_db):g}".replace(".", "_")

model = GaussianShift.from_snr_db(snr_db)

enc4 = ThresholdEncoder(THRESHOLDS)  # vanilla4, M=4, 2.0 bits/sensor
enc3 = ThresholdEncoder([0.2, 5])  # M=3, log2(3)=1.585 bits/sensor
enc2 = LRTEncoder(1.0)  # M=2, 1.0 bits/sensor
fractions = [(enc4, 1 / 3), (enc3, 1 / 3), (enc2, 1 / 3)]
avg_rate_mixed = sum(c * enc.rate() for enc, c in fractions)

Ns_anchor = np.unique(np.geomspace(30, 150, 25).astype(int))
rates = enc4.rate() * Ns_anchor  # shared rate targets for both schemes


def n_vanilla(rate):
    return max(1, round(rate / enc4.rate()))


def n_mixed(rate):
    return max(1, round(rate / avg_rate_mixed))


def make_bank_vanilla(rate):
    return EncoderBank.identical(enc4, n_vanilla(rate))


def make_bank_mixed(rate):
    return EncoderBank.from_fractions(fractions, n_mixed(rate))


# Sanity: at fraction (1.0,) from_fractions must reduce exactly to identical,
# and the real mixed bank must actually be outside Assumption 1(iii).
_fc_check = FusionCenter(p=model.p)
_a = EncoderBank.from_fractions([(enc4, 1.0)], 30)
_b = EncoderBank.identical(enc4, 30)
assert np.isclose(_fc_check.log_error_prob(_a, model), _fc_check.log_error_prob(_b, model))
assert EncoderBank.from_fractions(fractions, 30).uniform_alphabet() is False

model_meta = {"type": "GaussianShift", "snr_db": snr_db, "mu": model.mu, "sigma": model.sigma, "p": model.p}
timing_note = (
    "One exact log_error_prob at the top rate (300 bits) takes 12.7s for the "
    "mixed bank (N=196) and 10.2s for vanilla4 (N=150), measured at SNR 0 dB. "
    "The mixed bank is not cheaper despite its smaller M=4 subgroup: the "
    "per-group multinomials cost ~1s, and the cross-group convolution "
    "dominates because the M=2 group's log-LR atoms are incommensurate with "
    "the M=4/M=3 lattice (19.0M raw pairs -> 13.7M distinct atoms)."
)
not_adaptive_note = (
    "Static heterogeneity only. Every sensor's alphabet size is fixed in "
    "advance and does not depend on its own observation, so this measures the "
    "alphabet-size effect alone, not adaptive rate allocation."
)

group = f"heterogeneous_M_{snr_tag}"
session = "Mmix"


def write_readme(run_dir, encoder_lines, res, extra_lines=()):
    achieved = res["total_rate"]
    lines = [
        f"# {run_dir.name}",
        "",
        f"- Model: GaussianShift, SNR = {snr_db:g} dB (mu={model.mu:.4f}, sigma={model.sigma}), p={model.p}",
        *encoder_lines,
        f"- Rate: {len(rates)} points, targets {rates[0]:g}..{rates[-1]:g} bits "
        f"(= 2.0 x geomspace({Ns_anchor[0]}, {Ns_anchor[-1]}) sensors of vanilla4)",
        f"- Achieved total_rate: {achieved[0]:.3f}..{achieved[-1]:.3f} bits",
        f"- uniform_alphabet(): {bool(res['uniform_alphabet'][0])} "
        f"({'inside' if res['uniform_alphabet'][0] else 'outside'} Assumption 1(iii))",
        f"- Notes: {timing_note}",
        f"- Caveat: {not_adaptive_note}",
    ]
    lines += list(extra_lines)
    (run_dir / "README.md").write_text("\n".join(lines) + "\n")


res_vanilla = sweep_over_rate(model, make_bank_vanilla, rates)
res_vanilla["N"] = np.array([n_vanilla(r) for r in rates])
vanilla_dir = save_run(
    res_vanilla,
    label=f"vanilla4_exponent_vs_rate_{snr_tag}",
    meta={"model": model_meta, "bank": f"identical {enc4.describe()}", "x": "total_rate"},
    group=group,
    session=session,
)
write_readme(
    vanilla_dir,
    [f"- Encoder: {enc4.describe()}, |U|={enc4.M}, {enc4.rate():g} bits/sensor (all sensors identical)"],
    res_vanilla,
    extra_lines=[f"- N range: {res_vanilla['N'][0]}..{res_vanilla['N'][-1]} sensors"],
)

res_mixed = sweep_over_rate(model, make_bank_mixed, rates)
res_mixed["N"] = np.array([n_mixed(r) for r in rates])
mixed_dir = save_run(
    res_mixed,
    label=f"mixedM432_exponent_vs_rate_{snr_tag}",
    meta={
        "model": model_meta,
        "bank": "from_fractions " + ", ".join(f"{enc.describe()} x {c:.4f}" for enc, c in fractions),
        "avg_rate_per_sensor": avg_rate_mixed,
        "x": "total_rate",
    },
    group=group,
    session=session,
)
write_readme(
    mixed_dir,
    [
        "- Encoders (fractions): "
        + "; ".join(f"{enc.describe()} |U|={enc.M} at c={c:.4f}" for enc, c in fractions),
        f"- Average rate: {avg_rate_mixed:.4f} bits/sensor (vs {enc4.rate():g} for vanilla4)",
    ],
    res_mixed,
    extra_lines=[
        f"- N range: {res_mixed['N'][0]}..{res_mixed['N'][-1]} sensors "
        f"(largest-remainder split into M=4/M=3/M=2 subgroups)",
        "- Rate floor is 30 sensors, not 10: below N=30 the largest-remainder "
        "rounding distorts the 1/3-1/3-1/3 split (N=10 gives counts (4,3,3)).",
    ],
)

# Gain factor at matched rate: interpolate both total_exponent curves onto
# common rate points inside the overlap of the two achieved-rate ranges.
r_van, r_mix = res_vanilla["total_rate"], res_mixed["total_rate"]
lo, hi = max(r_van[0], r_mix[0]), min(r_van[-1], r_mix[-1])
probe_rates = np.linspace(lo, hi, 3)
exp_van = np.interp(probe_rates, r_van, res_vanilla["total_exponent"])
exp_mix = np.interp(probe_rates, r_mix, res_mixed["total_exponent"])
gains = np.exp(exp_mix - exp_van)

# Chernoff tightness. At p = 1/2 the Chernoff bound (DDMS eqs. 37-40) upper
# bounds J^N for every N, so it is a LOWER bound on the exponent: the exact
# curve must sit above it, and the ratio says by how much.
tight_van = res_vanilla["total_exponent"] / res_vanilla["chernoff_bound"]
tight_mix = res_mixed["total_exponent"] / res_mixed["chernoff_bound"]
assert np.all(tight_van >= 1) and np.all(tight_mix >= 1), (
    "exact exponent fell below the Chernoff bound -- at p=1/2 it must not"
)

stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
compare_dir = Path(vanilla_dir).parent / f"{stamp}_heterogeneous_M_compare_{snr_tag}"
compare_dir.mkdir(parents=True)

fig, (ax_exp, ax_tight, ax_ee) = plt.subplots(1, 3, figsize=(16, 4.5))

label_van = f"Vanilla4: all M=4 ({enc4.rate():g} b/sensor)"
label_mix = f"Mixed M=4/3/2, 1/3 each ({avg_rate_mixed:.3f} b/sensor avg)"
c_van, c_mix = "tab:blue", "tab:orange"

ax_exp.plot(r_van, res_vanilla["total_exponent"], "-o", ms=3, color=c_van, label=label_van)
ax_exp.plot(r_mix, res_mixed["total_exponent"], "-s", ms=3, color=c_mix, label=label_mix)
ax_exp.plot(r_van, res_vanilla["chernoff_bound"], "--", color=c_van, label="Vanilla4, Chernoff bound")
ax_exp.plot(r_mix, res_mixed["chernoff_bound"], "--", color=c_mix, label="Mixed, Chernoff bound")
ax_exp.set_xlabel("achieved total rate (bits)")
ax_exp.set_ylabel(r"$-\log J^N$")
ax_exp.set_title("Total exponent vs Chernoff bound (headline)")
ax_exp.legend(fontsize=8)

ax_tight.plot(r_van, tight_van, "-o", ms=3, color=c_van, label=label_van)
ax_tight.plot(r_mix, tight_mix, "-s", ms=3, color=c_mix, label=label_mix)
ax_tight.axhline(1.0, color="k", lw=0.8, ls=":", label="bound attained (ratio = 1)")
ax_tight.set_xlabel("achieved total rate (bits)")
ax_tight.set_ylabel(r"$-\log J^N \;/\; $ Chernoff bound")
ax_tight.set_title("Tightness: how far above the bound")
ax_tight.legend(fontsize=8)

ax_ee.plot(r_van, res_vanilla["normalized_exponent"], "-o", ms=3, color=c_van, label=label_van)
ax_ee.plot(r_mix, res_mixed["normalized_exponent"], "-s", ms=3, color=c_mix, label=label_mix)
ax_ee.plot(
    r_van, res_vanilla["chernoff_bound"] / res_vanilla["N"], "--", color=c_van, label="Vanilla4, Chernoff/N"
)
ax_ee.plot(r_mix, res_mixed["chernoff_bound"] / res_mixed["N"], "--", color=c_mix, label="Mixed, Chernoff/N")
ax_ee.set_xlabel("achieved total rate (bits)")
ax_ee.set_ylabel(r"$J^N_{EE} = -\frac{1}{N}\log J^N$")
ax_ee.set_title("Normalized exponent vs rate (diagnostic)")
ax_ee.legend(fontsize=8)

fig.suptitle(f"Heterogeneous |U| at matched total rate (static, non-adaptive), SNR {snr_db:g} dB")
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.text(
    0.5,
    0.01,
    "Panel 1 is the comparison: the two schemes run at different N for the same rate, which is exactly the effect being measured. Dashed = Chernoff bound\n"
    "(DDMS eq. 24), a LOWER bound on the exponent at p=1/2, so the solid curves must sit above it. Panel 3 divides by N and hides the rate-vs-N effect.\n"
    "Static heterogeneity only -- alphabets are fixed in advance, not adapted to observations.",
    ha="center",
    va="bottom",
    fontsize=8,
)
fig.savefig(compare_dir / "fig_heterogeneous_M.pdf")

gain_lines = [
    f"  - rate {r:.1f} bits (vanilla4 N={round(r / enc4.rate())}, mixed N={round(r / avg_rate_mixed)}): "
    f"exponent {ev:.4f} -> {em:.4f}, gain factor exp(delta) = x{g:.3f}"
    for r, ev, em, g in zip(probe_rates, exp_van, exp_mix, gains)
]
(compare_dir / "README.md").write_text(
    "\n".join(
        [
            f"# {compare_dir.name}",
            "",
            f"- Model: GaussianShift, SNR = {snr_db:g} dB (mu={model.mu:.4f}, sigma={model.sigma}), p={model.p}",
            f"- Vanilla4: {enc4.describe()}, |U|={enc4.M}, {enc4.rate():g} bits/sensor, uniform_alphabet()=True",
            "- Mixed: "
            + "; ".join(f"{enc.describe()} |U|={enc.M} at c={c:.4f}" for enc, c in fractions)
            + f"; {avg_rate_mixed:.4f} bits/sensor average, uniform_alphabet()=False",
            f"- Rate: {len(rates)} points, targets {rates[0]:g}..{rates[-1]:g} bits; "
            f"compared on achieved total_rate, overlap {lo:.1f}..{hi:.1f} bits",
            f"- Built from {vanilla_dir.name} and {mixed_dir.name}.",
            "",
            "## Gain factor exp(total_exponent_mixed - total_exponent_vanilla), matched rate",
            "",
            *gain_lines,
            "",
            f"- Ballpark: the gain factor spans x{gains.min():.3f}..x{gains.max():.3f} across the rate "
            f"range, which {'IS' if 0.5 <= gains.max() / 1.19 <= 2.0 else 'is NOT'} in the same "
            "ballpark (within a factor of 2) as the x1.19 alphabet-size factor quoted for "
            "sim_JEE.py, and nowhere near the x15.85 quoted there for the adaptive rate model.",
            "",
            "## Chernoff tightness",
            "",
            "At p=1/2 the Chernoff bound (DDMS eq. 24, eqs. 37-40) upper bounds J^N for",
            "every N, so it is a lower bound on the exponent: the exact curve sits above",
            "it and the ratio below says by how much. Ratio 1.0 would mean the bound is",
            "attained.",
            "",
            f"  - Vanilla4: exponent/bound = {tight_van[0]:.4f} at {r_van[0]:.1f} bits -> "
            f"{tight_van[-1]:.4f} at {r_van[-1]:.1f} bits "
            f"(bound {res_vanilla['chernoff_bound'][-1]:.4f} vs exponent {res_vanilla['total_exponent'][-1]:.4f} at the top)",
            f"  - Mixed:    exponent/bound = {tight_mix[0]:.4f} at {r_mix[0]:.1f} bits -> "
            f"{tight_mix[-1]:.4f} at {r_mix[-1]:.1f} bits "
            f"(bound {res_mixed['chernoff_bound'][-1]:.4f} vs exponent {res_mixed['total_exponent'][-1]:.4f} at the top)",
            "",
            "Both schemes stay above the bound at every point, as they must. The bound is",
            "asymptotic in N, so the ratio is expected to drift toward 1 as rate grows;",
            "read the trend, not the absolute gap.",
            "",
            "## Caveat",
            "",
            "This experiment measures static heterogeneity: each sensor's alphabet",
            "size is fixed in advance and does not depend on its own observation. It",
            "is not adaptive rate allocation. The adaptive comparison (vanilla4 vs",
            "SilenceEncoder at matched rate) is a separate experiment and is not done",
            "here; nothing in this figure should be read as evidence about it.",
            "",
            "- Notes: " + timing_note,
        ]
    )
    + "\n"
)

print(f"avg_rate_mixed = {avg_rate_mixed:.6f} bits/sensor")
for line in gain_lines:
    print(line.strip())
print(f"saved {vanilla_dir}")
print(f"saved {mixed_dir}")
print(f"saved {compare_dir}")
