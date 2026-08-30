"""Export the J^N_EE vs R figures as paper-ready PNGs.

Re-plots from the committed run data rather than re-simulating, so every
panel gets identical styling regardless of which version of the sweep script
produced it. Writes into Latex_Critical_Summary/Figs/, which is what
DDMS.tex already points \\includegraphics at.

Figures carry no embedded title -- N, SNR and the policy description belong
in the LaTeX caption. mu/sigma are still read from the run metadata so the
caption text can be generated from the same source (printed on export).

Usage: python figs/fig_export_paper_pngs.py
"""

import glob
from pathlib import Path

import matplotlib.pyplot as plt

from ddms import load_run

N = 3000
SNRS = [(-5.0, "snrm5"), (0.0, "snrp0"), (5.0, "snrp5")]
DPI = 300
OUT_DIR = Path("../../Latex_Critical_Summary/Figs")


def find_run(tag, setup):
    hits = sorted(glob.glob(f"runs/Varying_R/R_sweep_optparams_N{N}_{tag}_*/*setup{setup}_vs_R"))
    if not hits:
        raise SystemExit(f"no saved run for N={N}, {tag}, setup {setup}")
    return hits[-1]  # newest


def export(snr_db, tag):
    run_a = load_run(find_run(tag, "A"))
    run_b = load_run(find_run(tag, "B"))
    mu = run_a["meta"]["model"]["mu"]
    sigma = run_a["meta"]["model"]["sigma"]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(run_a["data"]["R"], run_a["data"]["normalized_exponent"], "o-",
            label="Setup A (vanilla)")
    ax.plot(run_b["data"]["R"], run_b["data"]["normalized_exponent"], "s-",
            label="Setup B (silence)")
    ax.set_xlabel("R (bits/sensor)")
    ax.set_ylabel(r"$J^N_{EE} = -\frac{1}{N}\log J^N$")
    ax.set_xticks([int(r) for r in run_a["data"]["R"]])
    # No embedded title: the LaTeX caption carries N, SNR and the policy
    # description, and a duplicated title just eats vertical space.
    ax.legend()
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"fig_JEE_vs_R_N{N}_{tag}.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"saved {path.resolve()}  (SNR={snr_db:+g}dB, mu={mu:.4g}, sigma={sigma:g})")
    return path


if __name__ == "__main__":
    for snr_db, tag in SNRS:
        export(snr_db, tag)
