"""Normalized exponent vs. active senders, optimal-delta silence, normalized
by TOTAL N (not by active-sender count) -- same style/formulation as
fig_active_senders_chernoff.py, but sourced from the optimal-delta run group
(fig_active_senders_vanilla_vs_silence_optdelta.py) instead of the fixed
[0.5, 2.0] thresholds.

Both fields (normalized_exponent, normalized_exponent_active) are already
saved in that run's data.json; this script deliberately plots
normalized_exponent (divide by N), matching the convention established for
fig_active_senders_normalized_chernoff.png, not the active-sender-normalized
field the optdelta script's own default figure uses.

Writes two copies of the same figure:
  1. Into the optdelta run group's own compare_* directory (its proper home).
  2. Overwriting fig_active_senders_normalized_chernoff.png in the original
     fixed-threshold run group, so that "the" normalized-exponent figure
     reflects the optimal delta rather than the earlier fixed thresholds.

Usage: python figs/fig_active_senders_chernoff_optdelta.py [optdelta_group_dir] [old_group_dir]
Both default to the most recent matching group under runs/M2/.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt

from ddms import GaussianShift, LRTEncoder, SilenceEncoder, load_run
from ddms.model import chernoff_distance

optdelta_group = Path(sys.argv[1]) if len(sys.argv) > 1 else sorted(
    Path("runs/M2").glob("active_senders_vanilla_vs_silence_optdelta_*")
)[-1]
old_group = Path(sys.argv[2]) if len(sys.argv) > 2 else sorted(
    p for p in Path("runs/M2").glob("active_senders_vanilla_vs_silence_*") if "optdelta" not in p.name
)[-1]

vanilla_dir = sorted(optdelta_group.glob("*_vanilla_exponent_vs_activecount_*"))[-1]
silence_dir = sorted(optdelta_group.glob("*_silence_optdelta_exponent_vs_activecount_*"))[-1]
optdelta_compare_dir = sorted(optdelta_group.glob("*_active_senders_optdelta_compare_*"))[-1]
old_compare_dir = sorted(old_group.glob("*_active_senders_compare_*"))[-1]

run_vanilla = load_run(vanilla_dir)
run_silence = load_run(silence_dir)

snr_db = run_vanilla["meta"]["model"]["snr_db"]
tau1_star = run_silence["meta"]["tau1_star"]
tau2_star = run_silence["meta"]["tau2_star"]
delta_star = run_silence["meta"]["delta_star"]

model = GaussianShift.from_snr_db(snr_db)
vanilla = LRTEncoder(1.0)
silence = SilenceEncoder([tau1_star, tau2_star], [1])

C_vanilla = chernoff_distance(vanilla.cell_probs(model, 1), vanilla.cell_probs(model, 2))
C_silence = chernoff_distance(silence.cell_probs(model, 1), silence.cell_probs(model, 2))

fig, ax = plt.subplots(figsize=(7, 5))

ax.plot(run_vanilla["data"]["active_count"], run_vanilla["data"]["normalized_exponent"], label="Always-Transmit", color="C0")
ax.axhline(C_vanilla, linestyle="--", color="C0", alpha=0.6, label=f"Always-Transmit Chernoff info = {C_vanilla:.4f}")

ax.plot(run_silence["data"]["active_count"], run_silence["data"]["normalized_exponent"], label="Silence", color="C1")
ax.axhline(C_silence, linestyle="--", color="C1", alpha=0.6, label=f"Silence Chernoff info = {C_silence:.4f}")

ax.set_xlabel("sensors that sent data")
ax.set_ylabel(r"$J^N_{EE} = -\frac{1}{N}\log J^N$")
ax.set_title(f"Normalized exponent vs. active senders, optimal $\\delta$, SNR {snr_db:g} dB")
ax.legend()
fig.tight_layout(rect=(0, 0.1, 1, 1))
fig.text(
    0.5, 0.045,
    r"Note: $J^N_{EE}$ normalizes by total $N$, not by the active-sender count shown on the x-axis.",
    ha="center", va="bottom", fontsize=8,
)
fig.text(
    0.5, 0.01,
    f"delta* = {delta_star:.4f} (tau1*={tau1_star:.4f}, tau2*={tau2_star:.4f})",
    ha="center", va="bottom", fontsize=8,
)

out_optdelta = optdelta_compare_dir / "fig_active_senders_normalized_chernoff.png"
out_old = old_compare_dir / "fig_active_senders_normalized_chernoff.png"
fig.savefig(out_optdelta, dpi=200)
fig.savefig(out_old, dpi=200)

print(f"Always-Transmit Chernoff information: {C_vanilla}")
print(f"Silence (optimal delta) Chernoff information: {C_silence}")
print(f"delta* = {delta_star}, tau1* = {tau1_star}, tau2* = {tau2_star}")
print(f"saved {out_optdelta}")
print(f"saved {out_old}")
