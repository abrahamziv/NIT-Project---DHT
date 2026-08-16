"""Compare Vanilla (LRT) vs. free-silence: exponent vs. *active* sender count.

Vanilla's N-sweep x-axis (N) already equals "sensors that sent data" -- every
sensor transmits. Silence's does not: sensors landing in the free middle
bucket never transmit, so plotting against raw N makes Silence look like it
uses more sensors than it actually communicates through. Here the x-axis is
the expected number of sensors that actually sent data:

  p_silent = p * P(u in silent | H1) + (1-p) * P(u in silent | H2)   (per sensor,
             independent of N)
  N_active(N) = N * (1 - p_silent)                                  (Vanilla: N_active = N)

This is a deterministic rescaling of N (p_silent depends only on the model
and thresholds), computed exactly the way SilenceEncoder.mean_rate() already
weighs silence -- there is no per-sensor Monte Carlo realization anywhere in
this simulator, so "sensors that sent data" is this expectation, not a count
drawn from a single run.

Panels:
  1. normalized_exponent vs. active-sender count (N-sweep)
  2. total_exponent       vs. active-sender count (N-sweep)
  3. normalized_exponent  vs. rate_used (R-sweep, max affordable N per budget)

Usage: python figs/fig_active_senders_vanilla_vs_silence.py [snr_db]  (default 0.0)
All runs and the comparison figure land in one new group directory under
runs/M2/.
"""

import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ddms import EncoderBank, GaussianShift, LRTEncoder, SilenceEncoder, save_run
from ddms.sweep import sweep_over_N, sweep_over_R

N_MAX = 3000
T_LO, T_HI = 0.5, 2.0
snr_db = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
snr_tag = f"snr{'m' if snr_db < 0 else 'p'}{abs(snr_db):g}".replace(".", "_")

Ns = np.unique(np.geomspace(10, N_MAX, 25).astype(int))

model = GaussianShift.from_snr_db(snr_db)
vanilla = LRTEncoder(1.0)
silence = SilenceEncoder([T_LO, T_HI], [1])

silent = list(silence.silent_indices)
q1, q2 = silence.cell_probs(model, 1), silence.cell_probs(model, 2)
p_silent = float(model.p * q1[silent].sum() + (1 - model.p) * q2[silent].sum())

model_meta = {"type": "GaussianShift", "snr_db": snr_db, "mu": model.mu, "sigma": model.sigma, "p": model.p}
group = f"active_senders_vanilla_vs_silence_{snr_tag}"
session = "M2"

# Mandatory-style N-sweep, reused for the active-sender x-axis.
res_vanilla = sweep_over_N(model, lambda N: EncoderBank.identical(vanilla, N), Ns)
res_vanilla["active_count"] = Ns.astype(float)
vanilla_dir = save_run(
    res_vanilla,
    label=f"vanilla_exponent_vs_activecount_{snr_tag}",
    meta={
        "model": model_meta,
        "bank": f"identical {vanilla.describe()}",
        "x": "active_count (== N, every sensor transmits)",
    },
    group=group,
    session=session,
)

res_silence = sweep_over_N(model, lambda N: EncoderBank.identical(silence, N), Ns)
res_silence["active_count"] = Ns * (1.0 - p_silent)
silence_dir = save_run(
    res_silence,
    label=f"silence_exponent_vs_activecount_{snr_tag}",
    meta={
        "model": model_meta,
        "bank": f"identical {silence.describe()}",
        "x": "active_count = N * (1 - p_silent)",
        "p_silent": p_silent,
    },
    group=group,
    session=session,
)

# Mandatory R-sweep: max affordable N per budget, matched across schemes.
silence_rate = silence.mean_rate(model)
Rs = Ns * vanilla.rate()

res_vanilla_r = sweep_over_R(model, lambda N: EncoderBank.identical(vanilla, N), vanilla.rate(), Rs, N_max=N_MAX)
res_vanilla_r["active_count"] = res_vanilla_r["N"].astype(float)
vanilla_r_dir = save_run(
    res_vanilla_r,
    label=f"vanilla_exponent_vs_R_{snr_tag}",
    meta={"model": model_meta, "bank": f"identical {vanilla.describe()}", "x": "rate_used"},
    group=group,
    session=session,
)

res_silence_r = sweep_over_R(model, lambda N: EncoderBank.identical(silence, N), silence_rate, Rs, N_max=N_MAX)
res_silence_r["active_count"] = res_silence_r["N"] * (1.0 - p_silent)
silence_r_dir = save_run(
    res_silence_r,
    label=f"silence_exponent_vs_R_{snr_tag}",
    meta={"model": model_meta, "bank": f"identical {silence.describe()}", "x": "rate_used", "p_silent": p_silent},
    group=group,
    session=session,
)

stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
compare_dir = Path(vanilla_dir).parent / f"{stamp}_active_senders_compare_{snr_tag}"
compare_dir.mkdir(parents=True)

fig, (ax_norm, ax_tot, ax_r) = plt.subplots(1, 3, figsize=(16, 4.5))

ax_norm.plot(res_vanilla["active_count"], res_vanilla["normalized_exponent"], label="Vanilla")
ax_norm.plot(res_silence["active_count"], res_silence["normalized_exponent"], label="Silence")
ax_norm.set_xlabel("sensors that sent data")
ax_norm.set_ylabel(r"$J^N_{EE} = -\frac{1}{N}\log J^N$")
ax_norm.set_title("Normalized exponent vs. active senders")
ax_norm.legend()

ax_tot.plot(res_vanilla["active_count"], res_vanilla["total_exponent"], label="Vanilla")
ax_tot.plot(res_silence["active_count"], res_silence["total_exponent"], label="Silence")
ax_tot.set_xlabel("sensors that sent data")
ax_tot.set_ylabel(r"$-\log J^N$")
ax_tot.set_title("Total exponent vs. active senders")
ax_tot.legend()

ax_r.plot(res_vanilla_r["rate_used"], res_vanilla_r["normalized_exponent"], label="Vanilla")
ax_r.plot(res_silence_r["rate_used"], res_silence_r["normalized_exponent"], label="Silence")
ax_r.set_xlabel("rate used (bits)")
ax_r.set_ylabel(r"$J^N_{EE} = -\frac{1}{N}\log J^N$")
ax_r.set_title("Normalized exponent vs. R (max N per budget)")
ax_r.legend()

fig.suptitle(f"Vanilla vs. silence-is-free, active senders, SNR {snr_db:g} dB")
fig.tight_layout()
fig.savefig(compare_dir / "fig_active_senders_vanilla_vs_silence.pdf")

r_table_lines = [
    "| R | N (vanilla) | N (silence) | active (silence) |",
    "|---|---|---|---|",
]
for R, Nv, Ns_, act in zip(Rs, res_vanilla_r["N"], res_silence_r["N"], res_silence_r["active_count"]):
    r_table_lines.append(f"| {R:.2f} | {int(Nv)} | {int(Ns_)} | {act:.2f} |")

readme = f"""# active_senders_vanilla_vs_silence_{snr_tag}

- Model: GaussianShift, SNR = {snr_db:g} dB (mu={model.mu:.4f}, sigma={model.sigma}), p={model.p}
- Vanilla: {vanilla.describe()}, |U|=2
- Silence: {silence.describe()}, |U|=3 (2 active + 1 free silent symbol),
  p_silent = {p_silent:.4f} (per-sensor probability of landing in the silent bucket)
- N: {len(Ns)} points, geomspace(10, {N_MAX})
- x-axis for panels 1-2 is *active senders*, not N: Vanilla active_count = N
  (every sensor transmits); Silence active_count = N * (1 - p_silent), the
  expected count of non-silent sensors -- this simulator has no per-sensor
  Monte Carlo realization, so "sensors that sent data" is this expectation.
- Panel 3 (R-sweep): N(R) = floor(R / per-sensor rate), matched budget across
  schemes; Vanilla per-sensor rate = {vanilla.rate():.4f} bits, Silence mean
  rate = {silence_rate:.4f} bits/sensor at this SNR. N capped at {N_MAX} --
  exact J^N is O(N^2) for Silence's 3-symbol alphabet.
- Sensors that actually sent data, per R (silence's active_count = N(R) * (1 - p_silent)):

{chr(10).join(r_table_lines)}

## Run directories

- {vanilla_dir}
- {silence_dir}
- {vanilla_r_dir}
- {silence_r_dir}
"""
(compare_dir / "README.md").write_text(readme)

params = {
    "snr_db": snr_db,
    "model": model_meta,
    "vanilla": vanilla.describe(),
    "silence": silence.describe(),
    "p_silent": p_silent,
    "silence_mean_rate_per_sensor": silence_rate,
    "N_range": [int(Ns[0]), int(Ns[-1])],
    "N_points": len(Ns),
    "R_range": [float(Rs[0]), float(Rs[-1])],
}
print(f"parameters: {params}")
print(f"saved {vanilla_dir}")
print(f"saved {silence_dir}")
print(f"saved {vanilla_r_dir}")
print(f"saved {silence_r_dir}")
print(f"saved {compare_dir}")
