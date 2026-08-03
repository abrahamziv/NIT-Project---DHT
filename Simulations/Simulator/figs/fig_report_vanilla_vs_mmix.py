"""Report figures: vanilla (uniform |U|) vs Mmix (heterogeneous |U|).

Reloads the committed c-sweep runs and re-renders them at report size --
fewer curves per axes, larger type -- for docs/report/. No new computation:
every number comes from the saved data.json files.

Usage: python figs/fig_report_vanilla_vs_mmix.py
"""

import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ddms import load_run

SNRS = [(-5, "snrm5"), (0, "snrp0"), (5, "snrp5"), (10, "snrp10")]
RATES = [60.0, 120.0, 240.0]
K = 2  # report figures use the largest budget, R = 240 bits
COLORS = {-5: "#2a78d6", 0: "#eb6834", 5: "#1baf7a", 10: "#a05fd0"}

out = Path("docs/report")
out.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "legend.frameon": False,
})

data = {}
for snr, tag in SNRS:
    d = glob.glob(f"runs/Mmix/csweep_M2M4_{tag}_*/*_csweep_M2M4_{tag}")[0]
    data[snr] = load_run(d)["data"]

# Figure 1: what heterogeneity buys, at R = 240 bits.
fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.0, 2.9))
for snr, _ in SNRS:
    p = data[snr]
    c, e = p["c_achieved"][:, K], p["total_exponent"][:, K]
    ax_a.plot(c, e, "-o", ms=3.5, lw=1.6, color=COLORS[snr])
    ax_a.annotate(f"{snr:g} dB", (c[-1], e[-1]), textcoords="offset points",
                  xytext=(4, 0), va="center", fontsize=8, color=COLORS[snr])
    perbit = e / p["total_rate"][:, K]
    ax_b.plot(c, perbit / perbit[0], "-o", ms=3.5, lw=1.6, color=COLORS[snr], label=f"{snr:g} dB")

ax_a.set_yscale("log")
ax_a.set_xlim(-0.04, 1.20)
ax_a.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
ax_a.set_xlabel("fraction of 1-bit sensors, $c$")
ax_a.set_ylabel(r"total exponent  $-\log J^N$")
ax_a.set_title("(a) Reliability at fixed budget", fontsize=9.5)

ax_b.axhline(1.0, color="0.4", lw=0.8, ls="--")
ax_b.set_xlabel("fraction of 1-bit sensors, $c$")
ax_b.set_ylabel("exponent per bit, rel. vanilla")
ax_b.set_title("(b) Value of a bit", fontsize=9.5)
ax_b.legend(title="SNR", fontsize=8, title_fontsize=8, loc="upper left", ncol=2,
            columnspacing=1.0, handlelength=1.4)
fig.tight_layout()
fig.savefig(out / "fig_report_gain.pdf")
plt.close(fig)

# Figure 2: the mixing penalty, all SNRs and all budgets.
fig, ax = plt.subplots(figsize=(7.0, 2.4))
for snr, _ in SNRS:
    p = data[snr]
    for k, r in enumerate(RATES):
        ax.plot(p["c_achieved"][:, k], p["chord_ratio"][:, k], "-o", ms=2.5, lw=1.1,
                color=COLORS[snr], alpha=0.85,
                label=f"{snr:g} dB" if k == K else None)
ax.axhspan(0.98, 1.02, color="0.5", alpha=0.10, lw=0)
ax.axhline(1.0, color="0.3", lw=0.9, ls="--")
ax.set_ylim(0.966, 1.032)
ax.set_xlabel("fraction of 1-bit sensors, $c$")
ax.set_ylabel("exact / chord")
ax.set_title("Mixing penalty: exact exponent vs. weighted-average prediction", fontsize=9.5)
ax.legend(fontsize=8, ncol=4, loc="lower center", handlelength=1.4, columnspacing=1.6)
fig.tight_layout()
fig.savefig(out / "fig_report_chord.pdf")
plt.close(fig)

for snr, _ in SNRS:
    p = data[snr]
    e = p["total_exponent"][:, K]
    r = p["chord_ratio"][1:-1]
    print(f"SNR {snr:>3} dB: exp {e[0]:.2f} -> {e[-1]:.2f} (x{np.exp(e[-1] - e[0]):.3g} lower error), "
          f"interior chord ratio {r.min():.4f}..{r.max():.4f}")
print(f"saved {out}/fig_report_gain.pdf and {out}/fig_report_chord.pdf")
