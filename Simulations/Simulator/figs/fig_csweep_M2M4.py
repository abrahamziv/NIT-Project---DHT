"""c-sweep between M=2 and M=4 alphabets at matched total rate.

A fraction c of the sensors uses the M=2 quantizer (threshold {1}) and the
rest use the M=4 quantizer (thresholds {0.2, 1, 5}); the M=2 thresholds are
a subset of the M=4 ones, so alphabet size is the only thing varying. The
endpoints c=0 (all M=4) and c=1 (all M=2) are homogeneous banks inside
Assumption 1(iii); every interior point is outside it.

All banks at one rate target R are compared at matched *total rate*: the
sensor count is N(c, R) = round(R / (2 - c)), never matched N. Each rate
target therefore turns c into a trade of many cheap sensors against fewer
expressive ones.

The dashed reference is the rate-weighted chord between the two homogeneous
endpoints: with per-bit exponents e_2 = E(c=1)/R and e_4 = E(c=0)/R and w2
the fraction of the total bits carried by the M=2 group,

    chord(c) = achieved_rate(c) * (w2 * e_2 + (1 - w2) * e_4).

If mixing alphabet sizes cost nothing beyond reallocating bits, the exact
curve would sit on the chord. The ratio exact/chord (panel 2) is therefore
the pure heterogeneity effect of breaking Assumption 1(iii): the shared
Chernoff-alpha coordination across groups, separated from the bit-efficiency
effect e_2 vs e_4 that already decides which endpoint wins.

Three rate targets per SNR show whether the effect grows or fades with the
total budget. c grid floor considerations: at the smallest bank (R=60,
c=0, N=30) largest-remainder rounding of the c split is still accurate to
one sensor; achieved c is recorded and plotted, not the target.

Usage: python figs/fig_csweep_M2M4.py    (runs all four SNRs + merged figure)
"""

import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from ddms import (
    EncoderBank,
    GaussianShift,
    LRTEncoder,
    ThresholdEncoder,
    save_run,
)
from ddms.sweep import sweep_over_rate

SNRS = [-5.0, 0.0, 5.0, 10.0]
CS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
RATES = [60.0, 120.0, 240.0]

enc4 = ThresholdEncoder([0.2, 1, 5])  # vanilla4, 2.0 bits/sensor
enc2 = LRTEncoder(1.0)  # M=2, 1.0 bit/sensor, threshold nested in enc4's

SESSION = "Mmix"
DATE = datetime.now().strftime("%Y-%m-%d")

chord_note = (
    "Chord reference: chord(c) = achieved_rate(c) * (w2*e2 + (1-w2)*e4) with "
    "e2, e4 the per-bit exponents of the homogeneous c=1 / c=0 endpoints at "
    "the same rate target and w2 the fraction of total bits carried by the "
    "M=2 group. exact/chord = 1 means breaking Assumption 1(iii) costs "
    "nothing beyond reallocating bits; deviation from 1 is the pure "
    "heterogeneity effect."
)


def n_for(c, rate):
    return max(1, round(rate / (c * enc2.rate() + (1 - c) * enc4.rate())))


def make_bank(c, rate):
    return EncoderBank.from_fractions([(enc2, c), (enc4, 1 - c)], n_for(c, rate))


def run_snr(snr_db):
    model = GaussianShift.from_snr_db(snr_db)
    rows, times = [], []
    for c in CS:
        t0 = time.perf_counter()
        res = sweep_over_rate(model, lambda r: make_bank(c, r), RATES)
        banks = [make_bank(c, r) for r in RATES]
        res["N"] = np.array([len(b) for b in banks])
        res["n2"] = np.array([sum(1 for e in b.encoders if e.M == 2) for b in banks])
        res["c_achieved"] = res["n2"] / res["N"]
        rows.append(res)
        times.append(time.perf_counter() - t0)
        print(f"SNR {snr_db:g} dB, c={c:.1f}: {times[-1]:.1f}s", flush=True)
    data = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    data["c_targets"] = np.array(CS)

    # Chord between the homogeneous endpoints, per rate target.
    i4, i2 = CS.index(0.0), CS.index(1.0)
    e4 = data["total_exponent"][i4] / data["total_rate"][i4]
    e2 = data["total_exponent"][i2] / data["total_rate"][i2]
    w2 = data["n2"] * enc2.rate() / data["total_rate"]
    data["chord"] = data["total_rate"] * (w2 * e2 + (1 - w2) * e4)
    data["chord_ratio"] = data["total_exponent"] / data["chord"]

    assert np.all(data["total_exponent"] >= data["chernoff_bound"] - 1e-9), (
        "exact exponent fell below the Chernoff bound -- at p=1/2 it must not"
    )
    return model, data, sum(times)


def plot_snr(ax_exp, ax_ratio, ax_ee, data, colors, ls="-", label_fmt="R = {r:g} bits"):
    for k, (r, color) in enumerate(zip(RATES, colors)):
        x = data["c_achieved"][:, k]
        ax_exp.plot(x, data["total_exponent"][:, k], ls, marker="o", ms=3, color=color,
                    label=label_fmt.format(r=r))
        ax_exp.plot(x, data["chord"][:, k], ":", color=color)
        ax_ratio.plot(x, data["chord_ratio"][:, k], ls, marker="o", ms=3, color=color)
        ax_ee.plot(x, data["normalized_exponent"][:, k], ls, marker="o", ms=3, color=color)


def write_snr_readme(run_dir, snr_db, model, data, elapsed):
    n_lo, n_hi = int(data["N"].min()), int(data["N"].max())
    ratio = data["chord_ratio"][1:-1]  # interior points only
    lines = [
        f"# {run_dir.name}",
        "",
        f"- Model: GaussianShift, SNR = {snr_db:g} dB (mu={model.mu:.4f}, sigma={model.sigma}), p={model.p}",
        f"- Encoders: {enc2.describe()} |U|=2 (fraction c) and {enc4.describe()} |U|=4 "
        "(fraction 1-c); M=2 threshold nested in the M=4 set",
        f"- c: {len(CS)} points, {CS[0]:g}..{CS[-1]:g} (largest-remainder split; achieved c recorded)",
        f"- Rate targets: {RATES[0]:g}, {RATES[1]:g}, {RATES[2]:g} bits total, matched across c "
        f"via N = round(R / (2 - c)); N spans {n_lo}..{n_hi}",
        "- uniform_alphabet(): True at c=0 and c=1 (inside Assumption 1(iii)), "
        "False at every interior c (outside)",
        f"- Interior exact/chord ratio spans {ratio.min():.4f}..{ratio.max():.4f}",
        f"- Notes: {chord_note}",
        f"- Timing: full c x rate grid at this SNR took {elapsed:.0f}s exact-path total.",
    ]
    (run_dir / "README.md").write_text("\n".join(lines) + "\n")


results = {}
for snr_db in SNRS:
    snr_tag = f"snr{'m' if snr_db < 0 else 'p'}{abs(snr_db):g}".replace(".", "_")
    model, data, elapsed = run_snr(snr_db)
    group = f"csweep_M2M4_{snr_tag}"
    run_dir = save_run(
        data,
        label=f"csweep_M2M4_{snr_tag}",
        meta={
            "model": {"type": "GaussianShift", "snr_db": snr_db, "mu": model.mu,
                      "sigma": model.sigma, "p": model.p},
            "bank": f"from_fractions [({enc2.describe()}, c), ({enc4.describe()}, 1-c)]",
            "axes": "rows = c_targets, cols = rate targets (data['rate'][0])",
        },
        group=group,
        session=SESSION,
    )
    write_snr_readme(run_dir, snr_db, model, data, elapsed)
    results[snr_db] = (model, data, run_dir)

    colors = ["tab:blue", "tab:orange", "tab:green"]
    fig, (ax_exp, ax_ratio, ax_ee) = plt.subplots(1, 3, figsize=(16, 4.5))
    plot_snr(ax_exp, ax_ratio, ax_ee, data, colors)
    ax_exp.set_xlabel("fraction of M=2 sensors, c (achieved)")
    ax_exp.set_ylabel(r"$-\log J^N$")
    ax_exp.set_title("Total exponent vs c (dotted = chord)")
    ax_exp.legend(fontsize=8)
    ax_ratio.axhline(1.0, color="k", lw=0.8, ls=":")
    ax_ratio.set_xlabel("fraction of M=2 sensors, c (achieved)")
    ax_ratio.set_ylabel("exact / chord")
    ax_ratio.set_title("Heterogeneity effect (1 = pure rate reallocation)")
    ax_ee.set_xlabel("fraction of M=2 sensors, c (achieved)")
    ax_ee.set_ylabel(r"$J^N_{EE} = -\frac{1}{N}\log J^N$")
    ax_ee.set_title("Normalized exponent (diagnostic)")
    fig.suptitle(f"c-sweep M=2/M=4 at matched total rate, SNR {snr_db:g} dB")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.text(
        0.5, 0.01,
        "Endpoints c=0 (all M=4) and c=1 (all M=2) are homogeneous, inside Assumption 1(iii); interior points are outside it. All banks at one rate\n"
        "target share the same total rate via N = round(R/(2-c)). Dotted chord = rate-weighted mix of the endpoint exponents; the exact/chord ratio\n"
        "in panel 2 isolates the heterogeneity effect from the M=2 vs M=4 bit-efficiency effect. Panel 3 divides by N, which varies with c.",
        ha="center", va="bottom", fontsize=8,
    )
    fig.savefig(run_dir / f"fig_csweep_M2M4_{snr_tag}.pdf")
    plt.close(fig)
    print(f"saved {run_dir}", flush=True)

# Merged figure: colour = SNR, line style = rate target, one legend.
stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
merged_dir = Path("runs") / SESSION / f"csweep_M2M4_all_snr_{DATE}" / f"{stamp}_csweep_M2M4_merged"
merged_dir.mkdir(parents=True)

snr_colors = dict(zip(SNRS, ["tab:blue", "tab:orange", "tab:green", "tab:red"]))
rate_styles = dict(zip(RATES, ["-", "--", ":"]))

fig, (ax_exp, ax_ratio, ax_ee) = plt.subplots(1, 3, figsize=(16, 4.5))
for snr_db in SNRS:
    _, data, _ = results[snr_db]
    for k, r in enumerate(RATES):
        x = data["c_achieved"][:, k]
        style = dict(ls=rate_styles[r], color=snr_colors[snr_db], marker="o", ms=2.5)
        ax_exp.plot(x, data["total_exponent"][:, k], **style)
        ax_ratio.plot(x, data["chord_ratio"][:, k], **style)
        ax_ee.plot(x, data["normalized_exponent"][:, k], **style)

handles = [Line2D([], [], color=snr_colors[s], ls="-", label=f"SNR {s:g} dB") for s in SNRS]
handles += [Line2D([], [], color="k", ls=rate_styles[r], label=f"R = {r:g} bits") for r in RATES]
ax_exp.set_yscale("log")
ax_exp.set_xlabel("fraction of M=2 sensors, c (achieved)")
ax_exp.set_ylabel(r"$-\log J^N$")
ax_exp.set_title("Total exponent vs c (log scale)")
ax_exp.legend(handles=handles, fontsize=8)
ax_ratio.axhline(1.0, color="k", lw=0.8, ls=":")
ax_ratio.set_xlabel("fraction of M=2 sensors, c (achieved)")
ax_ratio.set_ylabel("exact / chord")
ax_ratio.set_title("Heterogeneity effect (1 = pure rate reallocation)")
ax_ee.set_yscale("log")
ax_ee.set_xlabel("fraction of M=2 sensors, c (achieved)")
ax_ee.set_ylabel(r"$J^N_{EE} = -\frac{1}{N}\log J^N$")
ax_ee.set_title("Normalized exponent (log scale, diagnostic)")
fig.suptitle("c-sweep M=2/M=4 at matched total rate, all SNRs (colour = SNR, line style = rate)")
fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.text(
    0.5, 0.01,
    "Merged view of the per-SNR figures. Colour encodes SNR, line style encodes the total-rate target; chord overlays are omitted here, panel 2\n"
    "carries the exact/chord ratio instead. Endpoints are homogeneous (inside Assumption 1(iii)); interior c is outside it.",
    ha="center", va="bottom", fontsize=8,
)
fig.savefig(merged_dir / "fig_csweep_M2M4_merged.pdf")
plt.close(fig)

summary_lines = []
for snr_db in SNRS:
    _, data, run_dir = results[snr_db]
    ratio = data["chord_ratio"][1:-1]
    best = np.unravel_index(np.argmax(data["total_exponent"] / data["total_rate"]), data["total_exponent"].shape)
    summary_lines.append(
        f"  - SNR {snr_db:g} dB: interior exact/chord {ratio.min():.4f}..{ratio.max():.4f}; "
        f"best per-bit exponent at c={data['c_achieved'][best]:.2f}, R={RATES[best[1]]:g} bits "
        f"(from {run_dir.name})"
    )

(merged_dir / "README.md").write_text(
    "\n".join(
        [
            f"# {merged_dir.name}",
            "",
            f"- Model: GaussianShift at SNR in {{{', '.join(f'{s:g}' for s in SNRS)}}} dB, p=0.5",
            f"- Encoders: {enc2.describe()} |U|=2 (fraction c) vs {enc4.describe()} |U|=4 "
            "(fraction 1-c); nested thresholds",
            f"- c: {CS[0]:g}..{CS[-1]:g} in steps of 0.2; rate targets {RATES[0]:g}/{RATES[1]:g}/{RATES[2]:g} "
            "bits, matched across c via N = round(R / (2 - c))",
            "- Derived comparison, no data.json of its own; built from the per-SNR runs:",
            *[f"  - {results[s][2]}" for s in SNRS],
            "",
            "## Interior exact/chord ratio (pure Assumption 1(iii) effect)",
            "",
            *summary_lines,
            "",
            f"- Notes: {chord_note}",
        ]
    )
    + "\n"
)
print(f"saved {merged_dir}", flush=True)
