"""Same as fig_active_senders_chernoff.py, but normalized by active senders.

sweep.py's normalized_exponent divides total_exponent by N (total sensors,
including silent ones) -- see fig_active_senders_chernoff.py's plot. This
script instead divides by active_count (Vanilla: active_count == N, no
change; Silence: active_count = N * (1 - p_silent) < N), so both curves
answer "how many nats of exponent per sensor that actually transmitted."

The corresponding asymptote is C / (1 - p_silent) for Silence (C stays the
per-total-sensor Chernoff information; dividing by the smaller active count
inflates the constant by 1/(1-p_silent)) and C unchanged for Vanilla, since
total_exponent ~ N * C as N -> inf regardless of which count you divide by.

Usage: python figs/fig_active_senders_chernoff_normbyactive.py [run_group_dir]
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

q1_s, q2_s = silence.cell_probs(model, 1), silence.cell_probs(model, 2)
p_silent = float(model.p * q1_s[list(silence.silent_indices)].sum() + (1 - model.p) * q2_s[list(silence.silent_indices)].sum())

C_vanilla = chernoff_distance(vanilla.cell_probs(model, 1), vanilla.cell_probs(model, 2))
C_silence_total = chernoff_distance(q1_s, q2_s)
C_silence_active = C_silence_total / (1.0 - p_silent)

active_vanilla = run_vanilla["data"]["active_count"]
active_silence = run_silence["data"]["active_count"]
exp_per_active_vanilla = run_vanilla["data"]["total_exponent"] / active_vanilla
exp_per_active_silence = run_silence["data"]["total_exponent"] / active_silence

fig, ax = plt.subplots(figsize=(7, 5))

ax.plot(active_vanilla, exp_per_active_vanilla, label="Vanilla", color="C0")
ax.axhline(C_vanilla, linestyle="--", color="C0", alpha=0.6, label=f"Vanilla asymptote = {C_vanilla:.4f}")

ax.plot(active_silence, exp_per_active_silence, label="Silence", color="C1")
ax.axhline(C_silence_active, linestyle="--", color="C1", alpha=0.6, label=f"Silence asymptote = {C_silence_active:.4f}")

ax.set_xlabel("sensors that sent data")
ax.set_ylabel(r"$-\frac{1}{N_{active}}\log J^N$")
ax.set_title(f"Exponent normalized by active senders, SNR {snr_db:g} dB")
ax.legend()
fig.tight_layout()

out_path = compare_dir / "fig_active_senders_normbyactive_chernoff.png"
fig.savefig(out_path, dpi=200)
print(f"p_silent = {p_silent}")
print(f"Vanilla asymptote (unchanged, active==N): {C_vanilla}")
print(f"Silence asymptote per-total-N Chernoff info: {C_silence_total}, per-active: {C_silence_active}")
print(f"saved {out_path}")
