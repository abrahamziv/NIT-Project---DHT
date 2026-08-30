"""Simulation 2 (Sim_plan.tex section 2) against the DDMS section 5 math:
N fixed, R = 1..5 swept, with the section-5 policy gamma^R_{Delta,delta} at
the *optimal* (Delta*, delta*) -- chosen by maximizing the closed-form
Chernoff bound C(R, Delta, delta) = -log M(0.5) (eq. general-closed-form).

One figure per SNR: the normalized error exponent J^N_EE vs R, one line per
setup (A = vanilla, delta = 0; B = silence, both parameters optimized).
Nothing else is plotted.

Recorded per point in data.json and in the run README, but deliberately kept
off the figure:
  - the section-5 analytical bound C(R, Delta*, delta*), valid at EVERY N
    here (p = 1/2 makes the Chernoff prefactor exactly 1 at alpha = 1/2), so
    simulation >= bound is asserted at every point,
  - the Bahadur-Rao first-order refinement, which restores the polynomial
    prefactor the Chernoff bound discards and lands on top of the simulation,
  - the finite-N re-optimization of (Delta, delta) against the true exponent,
    whose negligible drift is the evidence that the asymptotic design rule is
    already the right one at this N.

Usage: python figs/fig_R_sweep_opt_delta_Delta.py
All output lands under runs/Varying_R/.
"""

import math
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from ddms import EncoderBank, FusionCenter, GaussianShift, RBitThresholdEncoder, save_run
from ddms.policy_opt import bahadur_rao_exponent, chernoff_alpha_star, optimal_params

SESSION = "Varying_R"
N_FIXED = 3000
RS = [1, 2, 3, 4, 5]
SNR_DBS = [-5.0, 5.0]
EXACT_CHECK_LIMIT = 1e7  # cross-check against the exact path below this many count vectors

STAMP = datetime.now().strftime("%Y-%m-%d_%H%M%S")


def snr_tag(snr_db):
    return f"snr{'m' if snr_db < 0 else 'p'}{abs(snr_db):g}".replace(".", "p")


class RSweep:
    """One SNR point: model, both fusion backends, and the R sweep on them."""

    def __init__(self, snr_db):
        self.snr_db = snr_db
        self.model = GaussianShift.from_snr_db(snr_db)
        self.mu, self.sigma, self.p = self.model.mu, self.model.sigma, self.model.p
        self.fc = FusionCenter(p=self.p, method="tilted")
        self.fc_exact = FusionCenter(p=self.p)  # oracle, used only where feasible

    def _exact_check(self, bank, M):
        """Relative deviation of the tilted total_exponent from the exact one,
        or None when C(N+M-1, M-1) exceeds EXACT_CHECK_LIMIT."""
        if math.comb(len(bank) + M - 1, M - 1) > EXACT_CHECK_LIMIT:
            return None
        tilted = self.fc.total_exponent(bank, self.model)
        exact = self.fc_exact.total_exponent(bank, self.model)
        return abs(tilted - exact) / abs(exact)

    def _finite_N_reopt(self, R, D0, d0):
        """Re-optimize (Delta, delta) against the true finite-N exponent,
        seeded at the Chernoff optimum. Returns (Delta, delta, J^N_EE)."""

        def objective(v):
            enc = RBitThresholdEncoder(self.model, R, max(v[0], 1e-6), max(v[1], 1e-9))
            bank = EncoderBank.identical(enc, N_FIXED)
            return -self.fc.normalized_exponent(bank, self.model)

        res = minimize(
            objective,
            [D0, max(d0, 1e-6)],
            method="Nelder-Mead",
            options={"xatol": 1e-6, "fatol": 1e-12, "maxiter": 200},
        )
        return max(float(res.x[0]), 1e-6), max(float(res.x[1]), 0.0), -float(res.fun)

    def evaluate_setup(self, R, Delta, delta, C):
        """One (R, setup) point: simulate, bound-check, refine, cross-check."""
        enc = RBitThresholdEncoder(self.model, R, Delta, delta)
        bank = EncoderBank.identical(enc, N_FIXED)

        jee = self.fc.normalized_exponent(bank, self.model)
        assert jee >= C - 1e-9, f"simulation {jee} fell below the bound {C} at R={R}"

        br = bahadur_rao_exponent(bank, self.model)
        check = self._exact_check(bank, enc.M)
        p_silent = 1.0 - enc.mean_rate(self.model) / enc.rate()
        return {
            "R": float(R),
            "L": float(enc.L),
            "Delta": float(Delta),
            "delta": float(delta),
            "N": float(N_FIXED),
            "total_exponent": N_FIXED * jee,
            "normalized_exponent": jee,
            "chernoff_C": float(C),
            "bahadur_rao_exponent": br,
            "bound_gap": jee - C,
            "br_rel_error": abs(br - jee) / jee,
            "alpha_star": chernoff_alpha_star(self.model, enc),
            "P_silence": p_silent,
            "N_active": N_FIXED * (1.0 - p_silent),
            "mean_rate": enc.mean_rate(self.model),
            "exact_check_rel": np.nan if check is None else check,
        }

    def run(self):
        rows_a, rows_b = [], []
        for R in RS:
            DA, _, CA = optimal_params(self.model, R, silence=False)
            DB, dB, CB = optimal_params(self.model, R)
            row_a = self.evaluate_setup(R, DA, 0.0, CA)
            row_b = self.evaluate_setup(R, DB, dB, CB)

            Dn, dn, jee_n = self._finite_N_reopt(R, DB, dB)
            row_b["Delta_finiteN"] = Dn
            row_b["delta_finiteN"] = dn
            # At R = 1 (L = 2) Delta is inert -- the two thresholds are
            # mu/2 -+ delta -- so its drift is optimizer noise, not signal.
            drift_D = abs(Dn - DB) if R > 1 else 0.0
            row_b["finiteN_param_drift"] = max(drift_D, abs(dn - dB))
            row_b["finiteN_jee_gain"] = jee_n - row_b["normalized_exponent"]

            rows_a.append(row_a)
            rows_b.append(row_b)
            print(
                f"  [{self.snr_db:+g}dB] R={R}: "
                f"A (Delta*={DA:.4f}, JEE={row_a['normalized_exponent']:.6f}, C={CA:.6f})  "
                f"B (Delta*={DB:.4f}, delta*={dB:.4f}, "
                f"JEE={row_b['normalized_exponent']:.6f}, C={CB:.6f})  "
                f"drift={row_b['finiteN_param_drift']:.1e}"
            )
        collect = lambda rows: {k: np.array([r[k] for r in rows]) for k in rows[0]}
        return collect(rows_a), collect(rows_b)


def md_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def summary_table(res_a, res_b):
    headers = [
        "R", "Setup", "Delta*", "delta*", "J^N_EE (sim)", "C (bound)",
        "Bahadur-Rao", "gap", "P_silence", "mean_rate", "exact_check_rel",
    ]
    rows = []
    for i in range(len(res_a["R"])):
        for label, res in (("A (vanilla)", res_a), ("B (silence)", res_b)):
            check = res["exact_check_rel"][i]
            rows.append(
                [
                    f"{res['R'][i]:.0f}",
                    label,
                    f"{res['Delta'][i]:.4f}" if res["R"][i] > 1 else "--",
                    f"{res['delta'][i]:.4f}" if label.startswith("B") else "0",
                    f"{res['normalized_exponent'][i]:.6f}",
                    f"{res['chernoff_C'][i]:.6f}",
                    f"{res['bahadur_rao_exponent'][i]:.6f}",
                    f"{res['bound_gap'][i]:.6f}",
                    f"{res['P_silence'][i]:.4f}",
                    f"{res['mean_rate'][i]:.4f}",
                    "--" if np.isnan(check) else f"{check:.1e}",
                ]
            )
    return md_table(headers, rows)


def plot_sweep(sweep, res_a, res_b, out_dir):
    """The deliverable: J^N_EE vs R, one line per setup, nothing else."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(res_a["R"], res_a["normalized_exponent"], "o-", label="Setup A (vanilla)")
    ax.plot(res_b["R"], res_b["normalized_exponent"], "s-", label="Setup B (silence)")
    ax.set_xlabel("R (bits/sensor)")
    ax.set_ylabel(r"$J^N_{EE} = -\frac{1}{N}\log J^N$")
    ax.set_xticks(RS)
    ax.set_title(
        f"Normalized error exponent at optimal $(\\Delta^*, \\delta^*)$\n"
        f"N={N_FIXED}, SNR={sweep.snr_db:+g}dB "
        f"($\\mu$={sweep.mu:.4g}, $\\sigma$={sweep.sigma:g})"
    )
    ax.legend()
    fig.tight_layout()
    path = out_dir / "fig_R_sweep_optparams.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def run_and_save(snr_db):
    sweep = RSweep(snr_db)
    print(f"SNR = {snr_db:+g} dB (mu={sweep.mu:.6g}, sigma={sweep.sigma:g}, p={sweep.p})")
    res_a, res_b = sweep.run()

    tag = snr_tag(snr_db)
    group = f"R_sweep_optparams_N{N_FIXED}_{tag}"
    meta_common = {
        "model": {
            "type": "GaussianShift", "snr_db": snr_db,
            "mu": sweep.mu, "sigma": sweep.sigma, "p": sweep.p,
        },
        "fusion": sweep.fc.meta(),
        "x": "R",
        "N": N_FIXED,
        "params": "Delta, delta optimized per point by maximizing C(R, Delta, delta)",
    }
    dir_a = save_run(
        res_a,
        label=f"N{N_FIXED}_{tag}_optparams_setupA_vs_R",
        meta={**meta_common, "bank": "identical RBitThresholdEncoder(R, Delta*, 0)"},
        group=group,
        session=SESSION,
    )
    dir_b = save_run(
        res_b,
        label=f"N{N_FIXED}_{tag}_optparams_setupB_vs_R",
        meta={**meta_common, "bank": "identical RBitThresholdEncoder(R, Delta*, delta*)"},
        group=group,
        session=SESSION,
    )

    compare_dir = Path(dir_a).parent / f"{STAMP}_{tag}_optparams_compare"
    compare_dir.mkdir(parents=True)
    fig_path = plot_sweep(sweep, res_a, res_b, compare_dir)

    table_md = summary_table(res_a, res_b)
    print(f"\n  SNR {snr_db:+g} dB summary:\n{table_md}\n")

    ceiling = sweep.model.chernoff_information()
    readme = "\n".join(
        [
            f"# {compare_dir.name}",
            "",
            f"- Model: GaussianShift, SNR = {snr_db:+g} dB "
            f"(mu={sweep.mu:.6g}, sigma={sweep.sigma:g}), p={sweep.p}",
            f"- Unquantized ceiling mu^2/8sigma^2 = {ceiling:.6f}",
            f"- Fusion backend: method=tilted, G={sweep.fc.G}, nsig={sweep.fc.nsig}; "
            f"exact_check_rel cross-checks against method=exact wherever "
            f"C(N+M-1,M-1) <= {EXACT_CHECK_LIMIT:.0e}",
            f"- N (fixed): {N_FIXED}; R sweep: {RS} (L = {[2**r for r in RS]})",
            "- Policy: DDMS section 5 gamma^R_{Delta,delta} (RBitThresholdEncoder), with "
            "(Delta*, delta*) maximizing the closed-form bound C(R, Delta, delta) "
            "= -log M(0.5) per point (Setup A pins delta = 0)",
            "- Figure plots ONLY the normalized error exponent, one line per setup. "
            "The bound, the Bahadur-Rao refinement and the finite-N re-optimization "
            "are recorded here and in data.json, not plotted.",
            "- The bound is valid at every finite N here (p = 1/2 makes the Chernoff "
            "prefactor 1 at alpha = 1/2), so `normalized_exponent >= chernoff_C` is "
            "asserted at every point.",
            f"- Finite-N re-optimization (Setup B): max parameter drift "
            f"{res_b['finiteN_param_drift'].max():.2e}, max J^N_EE gain "
            f"{res_b['finiteN_jee_gain'].max():.2e}.",
            "",
            "## Results",
            "",
            table_md,
            "",
            f"Built from {dir_a.name}, {dir_b.name}.",
        ]
    )
    (compare_dir / "README.md").write_text(readme + "\n")

    for d in (dir_a, dir_b, fig_path, compare_dir / "README.md"):
        print(f"  saved {d}")
    return res_a, res_b


if __name__ == "__main__":
    for snr in SNR_DBS:
        run_and_save(snr)
        print()
