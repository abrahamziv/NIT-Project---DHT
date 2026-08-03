"""Combine every Vanilla4-vs-Silence4 run (across SNR) into two summary PDFs.

Reads back whatever vanilla4_exponent_vs_N*/silence4_exponent_vs_N* run
directories already exist under runs/ via load_run -- run
fig_vanilla4_vs_silence4.py [snr_db] first for each SNR to include.

1. fig_all_snr_pages_M4.pdf   -- one page per SNR, the same 3-panel
   (rate, total exponent, normalized exponent) layout as
   fig_vanilla4_vs_silence4.py, regenerated from the saved run data.
2. fig_all_snr_overlay_M4.pdf -- a single page overlaying every SNR on
   shared axes: rate on the left (color = SNR, solid = Vanilla4 R=2N,
   dashed = Silence4 mean rate used -- the two differ here), normalized
   exponent (J^N_EE = -(1/N) log J^N, "EE") on the right (color = SNR,
   one line per SNR -- Vanilla4 and Silence4 are identical on this axis,
   see the note printed on each figure).
"""

import glob
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from ddms import load_run


def snr_of(run):
    m = run["meta"]["model"]
    if "snr_db" in m:
        return float(m["snr_db"])
    return 20 * np.log10(m["mu"] / m["sigma"])


def find_pairs():
    vanilla_dirs = sorted(
        d for d in glob.glob("runs/**/*_vanilla4_exponent_vs_N*", recursive=True) if Path(d).is_dir()
    )
    silence_dirs = sorted(
        d for d in glob.glob("runs/**/*_silence4_exponent_vs_N*", recursive=True) if Path(d).is_dir()
    )
    vanilla = [(d, load_run(d)) for d in vanilla_dirs]
    silence = [(d, load_run(d)) for d in silence_dirs]
    pairs = []
    for vd, vr in vanilla:
        v_snr = round(snr_of(vr), 6)
        sd, sr = next((d, r) for d, r in silence if round(snr_of(r), 6) == v_snr)
        pairs.append((v_snr, vr, vd, sr, sd))
    pairs.sort(key=lambda p: p[0])
    return pairs


pairs = find_pairs()
print(f"combining {len(pairs)} SNR point(s): {[p[0] for p in pairs]} dB")

# SilenceEncoder never overrides cell_probs -- with the same thresholds as
# Vanilla4 its channel matrix, and therefore the FC's exact exponent, is
# identical to Vanilla4's at every SNR. Plotting both as separate traces
# would draw one exactly on top of the other; assert it and plot one.
for snr, vr, vd, sr, sd in pairs:
    assert np.allclose(vr["data"]["total_exponent"], sr["data"]["total_exponent"]), (
        f"Vanilla4/Silence4 total_exponent diverged at {snr:g} dB -- thresholds no "
        "longer match, the single-curve exponent/EE plots below are no longer valid"
    )

SAME_EXPONENT_NOTE = (
    "Vanilla4 and Silence4 achieve the exact same total exponent " + r"$-\log J^N$" + " at every SNR (one curve per SNR, not two overlapping ones).\n"
    "Why: both use thresholds {0.2, 1, 5} -> identical channel P(u|H); silence only zeroes the rate cost, never the FC's exact information."
)

stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
date = stamp[:10]
manifest = {
    "combined_from": [
        {"snr_db": snr, "vanilla4_run": vd, "silence4_run": sd} for snr, _, vd, _, sd in pairs
    ]
}

group_dir = Path("runs") / "M4" / f"all_snr_summary_M4_{date}"
pages_dir = group_dir / f"{stamp}_all_snr_combined_M4"
pages_dir.mkdir(parents=True)
with PdfPages(pages_dir / "fig_all_snr_pages_M4.pdf") as pdf:
    for snr, vr, vd, sr, sd in pairs:
        Ns = vr["data"]["N"]
        fig, (ax_rate, ax_exp, ax_ee) = plt.subplots(1, 3, figsize=(16, 4.5))
        ax_rate.plot(Ns, vr["data"]["total_rate"], label="Vanilla4, R = 2N")
        ax_rate.plot(Ns, sr["data"]["mean_rate_used"], label="Silence4, mean rate used")
        ax_rate.plot(Ns, sr["data"]["total_rate"], "--", label="Silence4, nominal (N log2 4)")
        ax_rate.set_xlabel("N")
        ax_rate.set_ylabel("rate (bits)")
        ax_rate.set_title("Rate")
        ax_rate.legend()

        ax_exp.plot(Ns, vr["data"]["total_exponent"], label="Vanilla4 = Silence4")
        ax_exp.set_xlabel("N")
        ax_exp.set_ylabel(r"$-\log J^N$")
        ax_exp.set_title("Total exponent (identical)")
        ax_exp.legend()

        ax_ee.plot(Ns, vr["data"]["normalized_exponent"], label="Vanilla4 = Silence4")
        ax_ee.set_xlabel("N")
        ax_ee.set_ylabel(r"$J^N_{EE} = -\frac{1}{N}\log J^N$")
        ax_ee.set_title("Normalized exponent (identical)")
        ax_ee.legend()

        fig.suptitle(f"Vanilla-M4 vs. silence-M4, SNR {snr:g} dB")
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        fig.text(0.5, 0.01, SAME_EXPONENT_NOTE, ha="center", va="bottom", fontsize=8)
        pdf.savefig(fig)
        plt.close(fig)
(pages_dir / "sources.json").write_text(json.dumps(manifest, indent=2))

overlay_dir = group_dir / f"{stamp}_all_snr_overlay_M4"
overlay_dir.mkdir(parents=True)
colors = plt.cm.viridis(np.linspace(0, 1, len(pairs)))

fig, (ax_rate, ax_ee) = plt.subplots(1, 2, figsize=(13, 5.5))
for (snr, vr, vd, sr, sd), color in zip(pairs, colors):
    Ns = vr["data"]["N"]
    ax_rate.plot(Ns, vr["data"]["total_rate"], color=color, ls="-", label=f"Vanilla4, {snr:g} dB")
    ax_rate.plot(
        Ns, sr["data"]["mean_rate_used"], color=color, ls="--", label=f"Silence4, {snr:g} dB"
    )
    # Vanilla4 and Silence4 normalized_exponent are identical at every SNR
    # (see assertion above) -- one line per SNR, not a solid/dashed pair.
    ax_ee.plot(Ns, vr["data"]["normalized_exponent"], color=color, ls="-", label=f"{snr:g} dB")

ax_rate.set_xlabel("N")
ax_rate.set_ylabel("rate (bits)")
ax_rate.set_title("Rate (solid = Vanilla4 R=2N, dashed = Silence4 mean rate used)")
ax_rate.legend(fontsize=7, ncol=2)

ax_ee.set_xlabel("N")
ax_ee.set_ylabel(r"$J^N_{EE} = -\frac{1}{N}\log J^N$")
ax_ee.set_title("Normalized exponent (identical for Vanilla4 and Silence4)")
ax_ee.legend(fontsize=7, ncol=2)

fig.suptitle("All SNRs: Vanilla-M4 vs. Silence-M4 -- rate and normalized exponent")
fig.tight_layout(rect=(0, 0.06, 1, 1))
fig.text(0.5, 0.01, SAME_EXPONENT_NOTE, ha="center", va="bottom", fontsize=8)
fig.savefig(overlay_dir / "fig_all_snr_overlay_M4.pdf")
(overlay_dir / "sources.json").write_text(json.dumps(manifest, indent=2))

print(f"saved {pages_dir}")
print(f"saved {overlay_dir}")
