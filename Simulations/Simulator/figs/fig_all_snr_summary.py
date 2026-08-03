"""Combine every Vanilla-vs-Silence run (across SNR) into two summary PDFs.

Reads back whatever vanilla_exponent_vs_N*/silence_exponent_vs_N* run
directories already exist under runs/ via load_run -- run
fig_vanilla_vs_silence.py [snr_db] first for each SNR to include.

1. fig_all_snr_pages.pdf   -- one page per SNR, the same 2-panel
   (rate, total exponent) layout as fig_vanilla_vs_silence.py, regenerated
   from the saved run data.
2. fig_all_snr_overlay.pdf -- a single page overlaying every SNR and both
   encoders on shared axes: rate on the left, normalized exponent
   (J^N_EE = -(1/N) log J^N, "EE") on the right. Color = SNR, solid =
   Vanilla, dashed = Silence.
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


def find_pairs(suffix):
    """Latest run per SNR wins -- reruns (e.g. backfilling R-sweep data onto
    an SNR only previously swept over N) supersede earlier ones instead of
    being combined as if they were distinct SNR points."""
    vanilla_dirs = sorted(
        d for d in glob.glob(f"runs/**/*_vanilla_exponent_vs_{suffix}*", recursive=True) if Path(d).is_dir()
    )
    silence_dirs = sorted(
        d for d in glob.glob(f"runs/**/*_silence_exponent_vs_{suffix}*", recursive=True) if Path(d).is_dir()
    )
    vanilla = [(d, load_run(d)) for d in vanilla_dirs]
    silence = [(d, load_run(d)) for d in silence_dirs]
    by_snr = {}
    for vd, vr in vanilla:
        v_snr = round(snr_of(vr), 6)
        sd, sr = next((d, r) for d, r in silence if round(snr_of(r), 6) == v_snr)
        by_snr[v_snr] = (v_snr, vr, vd, sr, sd)
    return sorted(by_snr.values(), key=lambda p: p[0])


pairs = find_pairs("N")
pairs_r = {snr: (vr, sr) for snr, vr, _, sr, _ in find_pairs("R")}
print(f"combining {len(pairs)} SNR point(s): {[p[0] for p in pairs]} dB")

stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
date = stamp[:10]
manifest = {
    "combined_from": [
        {"snr_db": snr, "vanilla_run": vd, "silence_run": sd} for snr, _, vd, _, sd in pairs
    ]
}

group_dir = Path("runs") / "M2" / f"all_snr_summary_M2_{date}"
pages_dir = group_dir / f"{stamp}_all_snr_combined"
pages_dir.mkdir(parents=True)
with PdfPages(pages_dir / "fig_all_snr_pages.pdf") as pdf:
    for snr, vr, vd, sr, sd in pairs:
        Ns = vr["data"]["N"]
        vr_r, sr_r = pairs_r[snr]
        fig, (ax_rate, ax_exp, ax_expR) = plt.subplots(1, 3, figsize=(16, 4.5))
        ax_rate.plot(Ns, vr["data"]["total_rate"], label="Vanilla, R = N")
        ax_rate.plot(Ns, sr["data"]["mean_rate_used"], label="Silence, mean rate used")
        ax_rate.plot(Ns, sr["data"]["total_rate"], "--", label="Silence, nominal (N log2 3)")
        ax_rate.set_xlabel("N")
        ax_rate.set_ylabel("rate (bits)")
        ax_rate.set_title("Rate")
        ax_rate.legend()

        ax_exp.plot(Ns, vr["data"]["total_exponent"], label="Vanilla")
        ax_exp.plot(Ns, sr["data"]["total_exponent"], label="Silence")
        ax_exp.set_xlabel("N")
        ax_exp.set_ylabel(r"$-\log J^N$")
        ax_exp.set_title("Error exponent")
        ax_exp.legend()

        ax_expR.plot(vr_r["data"]["rate_used"], vr_r["data"]["total_exponent"], label="Vanilla")
        ax_expR.plot(sr_r["data"]["rate_used"], sr_r["data"]["total_exponent"], label="Silence")
        ax_expR.set_xlabel("rate used (bits)")
        ax_expR.set_ylabel(r"$-\log J^N$")
        ax_expR.set_title("Error exponent vs. R (max N per budget)")
        ax_expR.legend()

        fig.suptitle(f"Vanilla vs. silence-is-free, SNR {snr:g} dB")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
(pages_dir / "sources.json").write_text(json.dumps(manifest, indent=2))

overlay_dir = group_dir / f"{stamp}_all_snr_overlay"
overlay_dir.mkdir(parents=True)
colors = plt.cm.viridis(np.linspace(0, 1, len(pairs)))

fig, (ax_rate, ax_ee, ax_expR) = plt.subplots(1, 3, figsize=(19, 5.5))
for (snr, vr, vd, sr, sd), color in zip(pairs, colors):
    Ns = vr["data"]["N"]
    vr_r, sr_r = pairs_r[snr]
    ax_rate.plot(Ns, vr["data"]["total_rate"], color=color, ls="-", label=f"Vanilla, {snr:g} dB")
    ax_rate.plot(
        Ns, sr["data"]["mean_rate_used"], color=color, ls="--", label=f"Silence, {snr:g} dB"
    )
    ax_ee.plot(
        Ns, vr["data"]["normalized_exponent"], color=color, ls="-", label=f"Vanilla, {snr:g} dB"
    )
    ax_ee.plot(
        Ns, sr["data"]["normalized_exponent"], color=color, ls="--", label=f"Silence, {snr:g} dB"
    )
    ax_expR.plot(
        vr_r["data"]["rate_used"], vr_r["data"]["total_exponent"], color=color, ls="-",
        label=f"Vanilla, {snr:g} dB",
    )
    ax_expR.plot(
        sr_r["data"]["rate_used"], sr_r["data"]["total_exponent"], color=color, ls="--",
        label=f"Silence, {snr:g} dB",
    )

ax_rate.set_xlabel("N")
ax_rate.set_ylabel("rate (bits)")
ax_rate.set_title("Rate (solid = Vanilla R=N, dashed = Silence mean rate used)")
ax_rate.legend(fontsize=7, ncol=2)

ax_ee.set_xlabel("N")
ax_ee.set_ylabel(r"$J^N_{EE} = -\frac{1}{N}\log J^N$")
ax_ee.set_title("Normalized exponent (EE)")
ax_ee.legend(fontsize=7, ncol=2)

ax_expR.set_xlabel("rate used (bits)")
ax_expR.set_ylabel(r"$-\log J^N$")
ax_expR.set_title("Error exponent vs. R (solid = Vanilla, dashed = Silence)")
ax_expR.legend(fontsize=7, ncol=2)

fig.suptitle("All SNRs: Vanilla vs. Silence -- rate, normalized exponent, and exponent vs. R")
fig.tight_layout()
fig.savefig(overlay_dir / "fig_all_snr_overlay.pdf")
(overlay_dir / "sources.json").write_text(json.dumps(manifest, indent=2))

print(f"saved {pages_dir}")
print(f"saved {overlay_dir}")
