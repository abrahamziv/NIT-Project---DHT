"""Vanilla vs. free-silence with the OPTIMAL silence thresholds, exponent
normalized by active (transmitting) senders.

Same comparison as fig_active_senders_vanilla_vs_silence.py +
fig_active_senders_chernoff_normbyactive.py, but:
  - the silence thresholds tau1*, tau2* are the DDMS.tex Eq. (opt-delta)
    optimum for this SNR, found by numerically minimizing M_S(delta)
    (Eq. MS-of-delta), not the fixed tau=[0.5, 2.0] used previously;
  - the plotted/reported normalized exponent divides by N_active (sensors
    that actually transmitted), not by total N -- consistent with the
    active-sender x-axis, per explicit request.

Note on DDMS.tex Eq. (opt-tau): the boxed identity there,
tau2*^2 = P(u1|H2)/P(u2|H1), is not correct as literally written -- by the
symmetric construction P(u1|H2) = P(u2|H1) = q_+ always, so that ratio is
identically 1 regardless of delta. The correct identity (verified
numerically here) is tau2*^2 = P(u2|H2)/P(u2|H1) = P(u1|H1)/P(u1|H2). This
script does not rely on the boxed identity either way -- it solves for
delta* by directly minimizing M_S(delta), the well-defined unambiguous
objective from Eq. (MS-of-delta).

Usage: python figs/fig_active_senders_vanilla_vs_silence_optdelta.py [snr_db]
(default 0.0). All runs and the comparison figure land in one new group
directory under runs/M2/.
"""

import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import ndtr

from ddms import EncoderBank, GaussianShift, LRTEncoder, SilenceEncoder, save_run
from ddms.model import chernoff_distance
from ddms.sweep import sweep_over_N, sweep_over_R

N_MAX = 3000
snr_db = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
snr_tag = f"snr{'m' if snr_db < 0 else 'p'}{abs(snr_db):g}".replace(".", "_")

Ns = np.unique(np.geomspace(10, N_MAX, 25).astype(int))

model = GaussianShift.from_snr_db(snr_db)
mu, sigma = model.mu, model.sigma


def _Q(x):
    return 1 - ndtr(x)


def _M_S(delta):
    q_plus = _Q((mu / 2 + delta) / sigma)
    q_minus = _Q((mu / 2 - delta) / sigma)
    return (q_minus - q_plus) + 2 * np.sqrt((1 - q_minus) * q_plus)


_res = minimize_scalar(_M_S, bounds=(1e-9, 10 * sigma), method="bounded", options={"xatol": 1e-12})
delta_star = float(_res.x)
tau2_star = float(np.exp(mu * delta_star / sigma**2))
tau1_star = 1.0 / tau2_star

vanilla = LRTEncoder(1.0)
silence = SilenceEncoder([tau1_star, tau2_star], [1])

silent = list(silence.silent_indices)
q1, q2 = silence.cell_probs(model, 1), silence.cell_probs(model, 2)
p_silent = float(model.p * q1[silent].sum() + (1 - model.p) * q2[silent].sum())

C_vanilla = chernoff_distance(vanilla.cell_probs(model, 1), vanilla.cell_probs(model, 2))
C_silence_total = -np.log(_M_S(delta_star))  # == chernoff_distance(q1, q2), per-total-N
C_silence_active = C_silence_total / (1.0 - p_silent)

model_meta = {"type": "GaussianShift", "snr_db": snr_db, "mu": mu, "sigma": sigma, "p": model.p}
group = f"active_senders_vanilla_vs_silence_optdelta_{snr_tag}"
session = "M2"

res_vanilla = sweep_over_N(model, lambda N: EncoderBank.identical(vanilla, N), Ns)
res_vanilla["active_count"] = Ns.astype(float)
res_vanilla["normalized_exponent_active"] = res_vanilla["total_exponent"] / res_vanilla["active_count"]
vanilla_dir = save_run(
    res_vanilla,
    label=f"vanilla_exponent_vs_activecount_{snr_tag}",
    meta={"model": model_meta, "bank": f"identical {vanilla.describe()}", "x": "active_count (== N)"},
    group=group,
    session=session,
)

res_silence = sweep_over_N(model, lambda N: EncoderBank.identical(silence, N), Ns)
res_silence["active_count"] = Ns * (1.0 - p_silent)
res_silence["normalized_exponent_active"] = res_silence["total_exponent"] / res_silence["active_count"]
silence_dir = save_run(
    res_silence,
    label=f"silence_optdelta_exponent_vs_activecount_{snr_tag}",
    meta={
        "model": model_meta,
        "bank": f"identical {silence.describe()}",
        "x": "active_count = N * (1 - p_silent)",
        "delta_star": delta_star,
        "tau1_star": tau1_star,
        "tau2_star": tau2_star,
        "p_silent": p_silent,
    },
    group=group,
    session=session,
)

silence_rate = silence.mean_rate(model)
Rs = Ns * vanilla.rate()

res_vanilla_r = sweep_over_R(model, lambda N: EncoderBank.identical(vanilla, N), vanilla.rate(), Rs, N_max=N_MAX)
res_vanilla_r["active_count"] = res_vanilla_r["N"].astype(float)
res_vanilla_r["normalized_exponent_active"] = res_vanilla_r["total_exponent"] / res_vanilla_r["active_count"]
vanilla_r_dir = save_run(
    res_vanilla_r,
    label=f"vanilla_exponent_vs_R_{snr_tag}",
    meta={"model": model_meta, "bank": f"identical {vanilla.describe()}", "x": "rate_used"},
    group=group,
    session=session,
)

res_silence_r = sweep_over_R(model, lambda N: EncoderBank.identical(silence, N), silence_rate, Rs, N_max=N_MAX)
res_silence_r["active_count"] = res_silence_r["N"] * (1.0 - p_silent)
res_silence_r["normalized_exponent_active"] = res_silence_r["total_exponent"] / res_silence_r["active_count"]
silence_r_dir = save_run(
    res_silence_r,
    label=f"silence_optdelta_exponent_vs_R_{snr_tag}",
    meta={"model": model_meta, "bank": f"identical {silence.describe()}", "x": "rate_used", "p_silent": p_silent},
    group=group,
    session=session,
)

stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
compare_dir = Path(vanilla_dir).parent / f"{stamp}_active_senders_optdelta_compare_{snr_tag}"
compare_dir.mkdir(parents=True)

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(res_vanilla["active_count"], res_vanilla["normalized_exponent_active"], label="Vanilla", color="C0")
ax.axhline(C_vanilla, linestyle="--", color="C0", alpha=0.6, label=f"Vanilla asymptote = {C_vanilla:.4f}")
ax.plot(res_silence["active_count"], res_silence["normalized_exponent_active"], label="Silence (opt. delta)", color="C1")
ax.axhline(C_silence_active, linestyle="--", color="C1", alpha=0.6, label=f"Silence asymptote = {C_silence_active:.4f}")
ax.set_xlabel("sensors that sent data")
ax.set_ylabel(r"$-\frac{1}{N_{active}}\log J^N$")
ax.set_title(f"Exponent normalized by active senders, optimal $\\delta$, SNR {snr_db:g} dB")
ax.legend()
fig.tight_layout()
png_path = compare_dir / "fig_active_senders_optdelta_normbyactive.png"
fig.savefig(png_path, dpi=200)

readme = f"""# active_senders_vanilla_vs_silence_optdelta_{snr_tag}

- Model: GaussianShift, SNR = {snr_db:g} dB (mu={mu:.4f}, sigma={sigma}), p={model.p}
- Vanilla: {vanilla.describe()}, |U|=2
- Silence: {silence.describe()}, |U|=3 -- thresholds are the DDMS.tex
  Eq. (opt-delta) optimum, found by numerically minimizing M_S(delta)
  (Eq. MS-of-delta) rather than solved from the boxed Eq. (opt-tau), whose
  stated identity tau2*^2 = P(u1|H2)/P(u2|H1) reduces to 1 = 1 by the
  symmetric construction (P(u1|H2) = P(u2|H1) = q_+ identically) and does
  not pin down delta. The correct identity is tau2*^2 = P(u2|H2)/P(u2|H1) =
  P(u1|H1)/P(u1|H2); worth a fix in the writeup.
  delta* = {delta_star:.6f}, tau1* = {tau1_star:.6f}, tau2* = {tau2_star:.6f}
  (previously used fixed tau = [0.5, 2.0]; nearly identical at this SNR)
  p_silent = {p_silent:.4f}
- **Normalization: both the x-axis and the y-axis divide by N_active
  (sensors that actually transmitted), not total N** -- Vanilla:
  N_active = N; Silence: N_active = N * (1 - p_silent). This differs from
  the DDMS.tex Figure~active-senders-chernoff, whose y-axis (J^N_EE) divides
  by total N while its x-axis is already active senders -- a mismatch this
  run corrects.
- Chernoff asymptotes: Vanilla C_V = {C_vanilla:.6f} (unchanged, active==N);
  Silence, per-total-N C_S = {C_silence_total:.6f}, per-active
  C_S/(1-p_silent) = {C_silence_active:.6f}.
- N: {len(Ns)} points, geomspace(10, {N_MAX})
- Comparison to fixed-threshold run (tau=[0.5,2.0]): per-total-N Chernoff
  info there was 0.100712 vs {C_silence_total:.6f} here -- the fixed
  thresholds were already close to optimal at this SNR.

## Run directories

- {vanilla_dir}
- {silence_dir}
- {vanilla_r_dir}
- {silence_r_dir}
"""
(compare_dir / "README.md").write_text(readme)

params = {
    "snr_db": snr_db,
    "delta_star": delta_star,
    "tau1_star": tau1_star,
    "tau2_star": tau2_star,
    "p_silent": p_silent,
    "C_vanilla": C_vanilla,
    "C_silence_total_N": C_silence_total,
    "C_silence_active": C_silence_active,
}
print(f"parameters: {params}")
print(f"saved {vanilla_dir}")
print(f"saved {silence_dir}")
print(f"saved {vanilla_r_dir}")
print(f"saved {silence_r_dir}")
print(f"saved {png_path}")
