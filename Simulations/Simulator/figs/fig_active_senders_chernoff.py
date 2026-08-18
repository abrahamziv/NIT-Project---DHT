"""Normalized exponent vs. active senders, with Chernoff-information asymptotes.

Loads the N-sweep runs already saved by fig_active_senders_vanilla_vs_silence.py
and overlays each scheme's per-sensor Chernoff information C(q1, q2) -- the
N -> inf limit of normalized_exponent for i.i.d. sensors (Chernoff-Stein
lemma) -- as a dashed horizontal line.

Usage: python figs/fig_active_senders_chernoff.py [run_group_dir]
Defaults to the most recent active_senders_vanilla_vs_silence_snrp0_* group
under runs/M2/. Writes a PNG into that same group's compare_* directory.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt

from ddms import GaussianShift, LRTEncoder, SilenceEncoder, load_run
from ddms.model import chernoff_distance

if len(sys.argv) > 1:
    group_dir = Path(sys.argv[1])
else:
    candidates = sorted(Path("runs/M2").glob("active_senders_vanilla_vs_silence_*"))
    if not candidates:
        raise SystemExit("no active_senders_vanilla_vs_silence_* run group found under runs/M2")
    group_dir = candidates[-1]

vanilla_dir = sorted(group_dir.glob("*_vanilla_exponent_vs_activecount_*"))[-1]
silence_dir = sorted(group_dir.glob("*_silence_exponent_vs_activecount_*"))[-1]
compare_dir = sorted(group_dir.glob("*_active_senders_compare_*"))[-1]

run_vanilla = load_run(vanilla_dir)
run_silence = load_run(silence_dir)

snr_db = run_vanilla["meta"]["model"]["snr_db"]
model = GaussianShift.from_snr_db(snr_db)
vanilla = LRTEncoder(1.0)
silence = SilenceEncoder([0.5, 2.0], [1])

C_vanilla = chernoff_distance(vanilla.cell_probs(model, 1), vanilla.cell_probs(model, 2))
C_silence = chernoff_distance(silence.cell_probs(model, 1), silence.cell_probs(model, 2))

fig, ax = plt.subplots(figsize=(7, 5))

ax.plot(run_vanilla["data"]["active_count"], run_vanilla["data"]["normalized_exponent"], label="Always-Transmit", color="C0")
ax.axhline(C_vanilla, linestyle="--", color="C0", alpha=0.6, label=f"Always-Transmit Chernoff info = {C_vanilla:.4f}")

ax.plot(run_silence["data"]["active_count"], run_silence["data"]["normalized_exponent"], label="Silence", color="C1")
ax.axhline(C_silence, linestyle="--", color="C1", alpha=0.6, label=f"Silence Chernoff info = {C_silence:.4f}")

ax.set_xlabel("sensors that sent data")
ax.set_ylabel(r"$J^N_{EE} = -\frac{1}{N}\log J^N$")
ax.set_title(f"Normalized exponent vs. active senders, SNR {snr_db:g} dB")
ax.legend()
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.text(
    0.5, 0.01,
    r"Note: $J^N_{EE}$ normalizes by total $N$, not by the active-sender count shown on the x-axis.",
    ha="center", va="bottom", fontsize=8,
)

out_path = compare_dir / "fig_active_senders_normalized_chernoff.png"
fig.savefig(out_path, dpi=200)
print(f"Always-Transmit Chernoff information: {C_vanilla}")
print(f"Silence Chernoff information: {C_silence}")
print(f"saved {out_path}")
