"""Export the three ICASSP figures as single-column paper-ready PNGs.

Re-plots from the committed run data rather than re-simulating, like
fig_export_paper_pngs.py does for DDMS.tex. The Chernoff bound
C(R, Delta*, delta*) and the mean transmitted rate are read straight off the
runs (chernoff_C, mean_rate), so the figures cannot drift from the numbers
quoted in the text.

One panel per operating point -- J^N_EE vs R at 0, -5 and +5 dB, each with
its C(R, Delta*, delta*) bound curves and its unquantized ceiling
C_inf = s^2/8. The three share a shape and differ only in vertical scale,
C_inf spanning 0.040 to 0.395; the y-axis is left free per panel so that
shape stays readable.

Figures carry no embedded title -- N, SNR and the policy description belong in
the LaTeX caption.

Usage: python figs/fig_export_icassp_pngs.py
"""

import glob
from pathlib import Path

import matplotlib.pyplot as plt

from ddms import load_run

N = 3000
SNRS = [(-5.0, "snrm5"), (0.0, "snrp0"), (5.0, "snrp5")]
DPI = 300
OUT_DIR = Path("../../Documents/ICASSP_Article/Figs")

# IEEEtran single column is ~3.5in wide; draw at that size so the type does not
# shrink when \includegraphics fits it to \linewidth.
FIGSIZE = (3.4, 2.55)
plt.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 9,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "lines.markersize": 4,
    "lines.linewidth": 1.3,
})


def find_run(tag, setup):
    hits = sorted(glob.glob(f"runs/Varying_R/R_sweep_optparams_N{N}_{tag}_*/*setup{setup}_vs_R"))
    if not hits:
        raise SystemExit(f"no saved run for N={N}, {tag}, setup {setup}")
    return hits[-1]  # newest


def load_pair(tag):
    """The always-transmit (setup A) and silent (setup B) runs at one point."""
    return load_run(find_run(tag, "A")), load_run(find_run(tag, "B"))


def ceiling(run):
    """C_inf = s^2/8, the unquantized Chernoff information."""
    model = run["meta"]["model"]
    return (model["mu"] / model["sigma"]) ** 2 / 8


def fig_jee_vs_R(tag="snrp0"):
    """J^N_EE against R at one SNR, against its bound and its ceiling."""
    run_a, run_s = load_pair(tag)
    R = run_a["data"]["R"]
    c_inf = ceiling(run_a)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.axhline(c_inf, color="0.35", ls=(0, (5, 3)), lw=1.1,
               label=r"$C_\infty$")
    ax.plot(R, run_s["data"]["normalized_exponent"], "s-", color="C1",
            label=r"$\gamma^S$, $J^N_{EE}$")
    ax.plot(R, run_s["data"]["chernoff_C"], ":", color="C1",
            label=r"$\gamma^S$, $C(R,\Delta^\star,\delta^\star)$")
    ax.plot(R, run_a["data"]["normalized_exponent"], "o-", color="C0",
            label=r"$\gamma^A$, $J^N_{EE}$")
    ax.plot(R, run_a["data"]["chernoff_C"], ":", color="C0",
            label=r"$\gamma^A$, $C(R,\Delta^\star,0)$")

    ax.set_xlabel(r"$R$ (bits/sensor)")
    ax.set_ylabel(r"$J^N_{EE}$")
    ax.set_xticks([int(r) for r in R])
    ax.legend(loc="lower right", framealpha=0.9)
    fig.tight_layout(pad=0.3)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"fig_icassp_JEE_vs_R_N{N}_{tag}.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"saved {path.resolve()}  (C_inf={c_inf:.4f})")
    return path


if __name__ == "__main__":
    for _, tag in SNRS:
        fig_jee_vs_R(tag)
