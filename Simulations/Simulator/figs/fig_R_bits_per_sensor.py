"""Simulation 2 (Sim_plan.tex section 2): fix N, sweep R = log2(L) bits per
sensor, for Setup A (always-transmit L-region threshold quantizer) and
Setup B (silence-bin variant: L/2 active regions per side of a fixed-width
silence bin around mu/2). SNR 0dB (mu = sigma = 1).

Runs on FusionCenter(method="tilted") (docs/SADDLEPOINT_PLAN.md): the exact
convolution path is combinatorially capped (C(N+M-1,M-1)), which previously
forced either fine-grained R at small N or coarse R at large N, never both.
The tilted backend is flat in N and near-flat in M, so this script now runs
the originally-wanted point directly:

  - R sweep: N=1000 (fixed), R in {1,2,3,4,5} (L up to 32).
  - N sweep: R in {1,2} (fixed), N from 10 to 5000 -- a convergence check
    against the exact-path values already committed for N=10 and N=200 in
    earlier runs under runs/Varying_R/, confirming the tilted backend lands
    on the old reliable numbers before trusting it further out.

Every point also gets an exact-path cross-check wherever the composition
count C(N+M-1,M-1) stays under EXACT_CHECK_LIMIT, printed and recorded.

Setup B's silence half-width delta is a fixed constant (not optimized, not
swept), annotated on every plot per explicit request. Threshold construction
(vanilla_thresholds/silence_thresholds) is local to this script -- it only
builds threshold arrays for the existing ThresholdEncoder/SilenceEncoder, no
new Encoder subclass.

Usage: python figs/fig_R_bits_per_sensor.py
All output lands under runs/Varying_R/.
"""

import math
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from ddms import EncoderBank, FusionCenter, GaussianShift, SilenceEncoder, ThresholdEncoder, save_run

SESSION = "Varying_R"
MARGIN_SIGMA = 3.0  # how far the outer thresholds extend beyond each mean
DELTA = 0.25  # fixed Setup B silence half-width; must be < mu/2
EXACT_CHECK_LIMIT = 1e7  # cross-check against the exact path below this many count vectors

model = GaussianShift.from_snr_db(0.0)
mu, sigma, p = model.mu, model.sigma, model.p
assert DELTA < mu / 2, "silence half-width must leave both sides non-degenerate"

fc = FusionCenter(p=p, method="tilted")
fc_exact = FusionCenter(p=p)  # cross-check oracle, used only where feasible

ROOT = Path("runs") / SESSION
STAMP = datetime.now().strftime("%Y-%m-%d_%H%M%S")
DATE = STAMP[:10]

# Exact-path values already committed for N in {10, 200}, R in {1, 2}, from
# runs/Varying_R/R_sweep_N{10,200}_snrp0_2026-08-17/. The N-sweep below
# re-derives these points with the tilted backend and asserts agreement --
# the convergence check that grounds the new backend in the old runs.
OLD_ANCHORS = {
    ("A", 10, 1): 2.210827715012699,
    ("A", 10, 2): 2.2177014938491477,
    ("A", 200, 1): 18.492812136999767,
    ("A", 200, 2): 18.61483388313994,
    ("B", 10, 1): 2.439678489591672,
    ("B", 10, 2): 2.4454296656227914,
    ("B", 200, 1): 21.306602007102377,
    ("B", 200, 2): 21.41222464982272,
}


def _vanilla_y(L, margin_sigma=MARGIN_SIGMA):
    """L-1 threshold positions in y-space, evenly spaced, symmetric about mu/2."""
    lo, hi = -margin_sigma * sigma, mu + margin_sigma * sigma
    k = L - 1
    return np.array([mu / 2]) if k == 1 else np.linspace(lo, hi, k)


def _silence_y_sides(L, delta=DELTA, margin_sigma=MARGIN_SIGMA):
    """(left thresholds, right thresholds) in y-space, L/2 each side of the silence bin."""
    lo, hi = -margin_sigma * sigma, mu + margin_sigma * sigma
    left_bound, right_bound = mu / 2 - delta, mu / 2 + delta
    k = L // 2
    y_left = np.array([left_bound]) if k == 1 else np.linspace(lo, left_bound, k)
    y_right = np.array([right_bound]) if k == 1 else np.linspace(right_bound, hi, k)
    return y_left, y_right


def vanilla_thresholds(L):
    return model.likelihood_ratio(_vanilla_y(L))


def silence_thresholds(L, delta=DELTA):
    y_left, y_right = _silence_y_sides(L, delta)
    thresholds = model.likelihood_ratio(np.concatenate([y_left, y_right]))
    return thresholds, L // 2


def make_setup_a(L):
    return ThresholdEncoder(vanilla_thresholds(L))


def make_setup_b(L):
    thresholds, silent_idx = silence_thresholds(L)
    return SilenceEncoder(thresholds, [silent_idx])


def collect(rows):
    return {k: np.array([r[k] for r in rows]) for k in rows[0]}


def _exact_check(bank, M):
    """Relative deviation of the tilted total_exponent from the exact one,
    or None when C(N+M-1, M-1) exceeds EXACT_CHECK_LIMIT."""
    N = len(bank)
    if math.comb(N + M - 1, M - 1) > EXACT_CHECK_LIMIT:
        return None
    tilted = fc.total_exponent(bank, model)
    exact = fc_exact.total_exponent(bank, model)
    return abs(tilted - exact) / abs(exact)


def _point_row(N, enc_a, enc_b):
    """One (N, encoder pair) evaluation, common to both sweeps."""
    bank_a = EncoderBank.identical(enc_a, N)
    bank_b = EncoderBank.identical(enc_b, N)

    q1, q2 = enc_b.cell_probs(model, 1), enc_b.cell_probs(model, 2)
    silent = list(enc_b.silent_indices)
    p_silent = float(p * q1[silent].sum() + (1 - p) * q2[silent].sum())

    check_a = _exact_check(bank_a, enc_a.M)
    check_b = _exact_check(bank_b, enc_b.M)
    row_a = {
        "N": float(N),
        "total_exponent": fc.total_exponent(bank_a, model),
        "normalized_exponent": fc.normalized_exponent(bank_a, model),
        "N_active": float(N),
        "mean_rate": enc_a.rate(),
        "exact_check_rel": np.nan if check_a is None else check_a,
    }
    row_b = {
        "N": float(N),
        "total_exponent": fc.total_exponent(bank_b, model),
        "normalized_exponent": fc.normalized_exponent(bank_b, model),
        "N_active": N * (1.0 - p_silent),
        "mean_rate": enc_b.mean_rate(model),
        "exact_check_rel": np.nan if check_b is None else check_b,
    }
    return row_a, row_b


def run_variant(N, Rs, tag):
    rows_a, rows_b = [], []
    for R in Rs:
        L = 2**R
        enc_a, enc_b = make_setup_a(L), make_setup_b(L)
        row_a, row_b = _point_row(N, enc_a, enc_b)
        row_a["R"], row_b["R"] = float(R), float(R)
        row_a["L"], row_b["L"] = float(L), float(L)
        rows_a.append(row_a)
        rows_b.append(row_b)
        check_a = "--" if np.isnan(row_a["exact_check_rel"]) else f"{row_a['exact_check_rel']:.1e}"
        check_b = "--" if np.isnan(row_b["exact_check_rel"]) else f"{row_b['exact_check_rel']:.1e}"
        print(
            f"[{tag}] R={R} L={L} done "
            f"(A: -logJ={row_a['total_exponent']:.4f} exact_rel={check_a}, "
            f"B: -logJ={row_b['total_exponent']:.4f} exact_rel={check_b}, "
            f"B mean_rate={row_b['mean_rate']:.4f})"
        )
    return collect(rows_a), collect(rows_b)


def run_n_sweep_at_R(R, Ns, tag):
    L = 2**R
    enc_a, enc_b = make_setup_a(L), make_setup_b(L)
    rows_a, rows_b = [], []
    for N in Ns:
        row_a, row_b = _point_row(N, enc_a, enc_b)
        rows_a.append(row_a)
        rows_b.append(row_b)
    res_a, res_b = collect(rows_a), collect(rows_b)

    for setup, res in (("A", res_a), ("B", res_b)):
        for N in (10, 200):
            key = (setup, N, R)
            if key not in OLD_ANCHORS or N not in Ns:
                continue
            fresh = float(res["total_exponent"][list(Ns).index(N)])
            old = OLD_ANCHORS[key]
            rel = abs(fresh - old) / abs(old)
            status = "OK" if rel < 1e-6 else "FAIL"
            print(
                f"[{tag}] convergence check Setup {setup} R={R} N={N}: "
                f"tilted={fresh:.10f} old_exact={old:.10f} rel={rel:.2e} [{status}]"
            )
            if status == "FAIL":
                raise AssertionError(
                    f"tilted backend diverged from committed exact run at "
                    f"Setup {setup}, N={N}, R={R}: rel={rel:.2e}"
                )
    return res_a, res_b


def md_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def rate_table(res_a, res_b):
    headers = ["R", "Setup", "N", "N_active", "mean_rate (bits/sensor)", "exact_check_rel"]
    rows = []
    for i in range(len(res_a["R"])):
        for label, res in (("A (vanilla)", res_a), ("B (silence)", res_b)):
            check = res["exact_check_rel"][i]
            rows.append(
                [
                    f"{res['R'][i]:.0f}",
                    label,
                    f"{res['N'][i]:.0f}",
                    f"{res['N_active'][i]:.1f}",
                    f"{res['mean_rate'][i]:.4f}",
                    "--" if np.isnan(check) else f"{check:.1e}",
                ]
            )
    return md_table(headers, rows)


def plot_variant(res_a, res_b, N, label, out_dir):
    fig, (ax_exp, ax_ee) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax_exp.plot(res_a["R"], res_a["total_exponent"], "o-", label="Setup A (vanilla)")
    ax_exp.plot(res_b["R"], res_b["total_exponent"], "o-", label="Setup B (silence)")
    ax_exp.set_xlabel("R (bits/sensor)")
    ax_exp.set_ylabel(r"$-\log J^N$")
    ax_exp.set_title("Total exponent")
    ax_exp.legend()

    ax_ee.plot(res_a["R"], res_a["normalized_exponent"], "o-", label="Setup A (vanilla)")
    ax_ee.plot(res_b["R"], res_b["normalized_exponent"], "o-", label="Setup B (silence)")
    ax_ee.set_xlabel("R (bits/sensor)")
    ax_ee.set_ylabel(r"$J^N_{EE} = -\frac{1}{N}\log J^N$")
    ax_ee.set_title("Normalized exponent")
    ax_ee.legend()

    fig.suptitle(
        f"{label}: N={N} (fixed), SNR=0dB (mu={mu:g}, sigma={sigma:g}), "
        f"delta={DELTA:g}, method=tilted"
    )
    fig.tight_layout()
    path = out_dir / f"fig_{label}.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_n_sweep(results_by_R, out_dir):
    """results_by_R: {R: (res_a, res_b)}. One figure, panels per metric,
    color by R, line style by setup; log-x axis over N."""
    fig, (ax_exp, ax_ee) = plt.subplots(1, 2, figsize=(12, 5))
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(results_by_R)))
    for color, (R, (res_a, res_b)) in zip(colors, results_by_R.items()):
        ax_exp.plot(res_a["N"], res_a["total_exponent"], "-o", color=color, label=f"A, R={R}")
        ax_exp.plot(res_b["N"], res_b["total_exponent"], "--s", color=color, label=f"B, R={R}")
        ax_ee.plot(res_a["N"], res_a["normalized_exponent"], "-o", color=color, label=f"A, R={R}")
        ax_ee.plot(res_b["N"], res_b["normalized_exponent"], "--s", color=color, label=f"B, R={R}")

    for ax, ylabel, title in (
        (ax_exp, r"$-\log J^N$", "Total exponent vs N"),
        (ax_ee, r"$J^N_{EE} = -\frac{1}{N}\log J^N$", "Normalized exponent vs N"),
    ):
        ax.set_xscale("log")
        ax.set_xlabel("N")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)

    fig.suptitle(
        f"N-sweep convergence check: SNR=0dB (mu={mu:g}, sigma={sigma:g}), "
        f"delta={DELTA:g}, method=tilted, solid=Setup A, dashed=Setup B"
    )
    fig.tight_layout()
    path = out_dir / "fig_n_sweep.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def run_and_save_variant(N, Rs, tag, label):
    res_a, res_b = run_variant(N, Rs, tag)

    group = f"R_sweep_{tag}_snrp0"
    meta_common = {
        "model": {"type": "GaussianShift", "snr_db": 0.0, "mu": mu, "sigma": sigma, "p": p},
        "delta": DELTA,
        "fusion": fc.meta(),
    }

    dir_a = save_run(
        res_a,
        label=f"{tag}_setupA_vs_R",
        meta={**meta_common, "bank": "identical ThresholdEncoder(L)", "x": "R", "N": N},
        group=group,
        session=SESSION,
    )
    dir_b = save_run(
        res_b,
        label=f"{tag}_setupB_vs_R",
        meta={**meta_common, "bank": f"identical SilenceEncoder(L, delta={DELTA})", "x": "R", "N": N},
        group=group,
        session=SESSION,
    )

    compare_dir = Path(dir_a).parent / f"{STAMP}_{tag}_compare"
    compare_dir.mkdir(parents=True)
    fig_path = plot_variant(res_a, res_b, N, label, compare_dir)

    table_md = rate_table(res_a, res_b)
    print(f"\n{label} rate-used table:\n{table_md}\n")

    readme = "\n".join(
        [
            f"# {compare_dir.name}",
            "",
            f"- Model: GaussianShift, SNR = 0 dB (mu={mu:g}, sigma={sigma:g}), p={p}",
            f"- Fusion backend: method=tilted, G={fc.G}, nsig={fc.nsig}; exact_check_rel "
            f"columns cross-check against method=exact wherever "
            f"C(N+M-1,M-1) <= {EXACT_CHECK_LIMIT:.0e}",
            f"- N (fixed): {N}",
            f"- R sweep: {[int(r) for r in Rs]} bits/sensor (L = {[2**r for r in Rs]})",
            f"- Setup A: ThresholdEncoder with L-1 thresholds evenly spaced in y over "
            f"[-{MARGIN_SIGMA:g}*sigma, mu+{MARGIN_SIGMA:g}*sigma], symmetric about mu/2",
            f"- Setup B: SilenceEncoder, L thresholds (L/2 per side), silence bin "
            f"[mu/2-delta, mu/2+delta] with **delta = {DELTA:g}** (fixed, not optimized)",
            "",
            "## Rate used",
            "",
            table_md,
            "",
            f"Built from {dir_a.name}, {dir_b.name}.",
        ]
    )
    (compare_dir / "README.md").write_text(readme + "\n")

    print(f"saved {dir_a}")
    print(f"saved {dir_b}")
    print(f"saved {fig_path}")
    print(f"saved {compare_dir / 'README.md'}")
    return res_a, res_b


def run_and_save_n_sweep(Rs, Ns, tag):
    results_by_R = {R: run_n_sweep_at_R(R, Ns, tag) for R in Rs}

    group = f"N_sweep_{tag}_snrp0"
    meta_common = {
        "model": {"type": "GaussianShift", "snr_db": 0.0, "mu": mu, "sigma": sigma, "p": p},
        "delta": DELTA,
        "fusion": fc.meta(),
        "x": "N",
    }
    saved_dirs = []
    for R, (res_a, res_b) in results_by_R.items():
        saved_dirs.append(
            save_run(
                res_a,
                label=f"{tag}_R{R}_setupA_vs_N",
                meta={**meta_common, "bank": "identical ThresholdEncoder(L)", "R": R},
                group=group,
                session=SESSION,
            )
        )
        saved_dirs.append(
            save_run(
                res_b,
                label=f"{tag}_R{R}_setupB_vs_N",
                meta={
                    **meta_common,
                    "bank": f"identical SilenceEncoder(L, delta={DELTA})",
                    "R": R,
                },
                group=group,
                session=SESSION,
            )
        )

    compare_dir = saved_dirs[0].parent / f"{STAMP}_{tag}_compare"
    compare_dir.mkdir(parents=True)
    fig_path = plot_n_sweep(results_by_R, compare_dir)

    anchor_lines = [
        f"  - Setup {s}, N={n}, R={r}: old exact = {v:.6f}"
        for (s, n, r), v in sorted(OLD_ANCHORS.items())
        if r in Rs
    ]
    readme = "\n".join(
        [
            f"# {compare_dir.name}",
            "",
            f"- Model: GaussianShift, SNR = 0 dB (mu={mu:g}, sigma={sigma:g}), p={p}",
            f"- Fusion backend: method=tilted, G={fc.G}, nsig={fc.nsig}",
            f"- R (fixed per curve): {Rs}",
            f"- N sweep: {list(Ns)}",
            "- Purpose: convergence check -- confirms the tilted backend reproduces the "
            "exact-path values already committed for N=10 and N=200 in the R-sweep runs "
            "(runs/Varying_R/R_sweep_N{10,200}_snrp0_2026-08-17/) before extending past "
            "where the exact path is feasible.",
            "- Anchor comparison (all passed at rtol=1e-6, see console log):",
            *anchor_lines,
            "",
            f"Built from {', '.join(d.name for d in saved_dirs)}.",
        ]
    )
    (compare_dir / "README.md").write_text(readme + "\n")

    for d in saved_dirs:
        print(f"saved {d}")
    print(f"saved {fig_path}")
    print(f"saved {compare_dir / 'README.md'}")
    return results_by_R


def plot_threshold_diagram(L, out_dir):
    lo, hi = -MARGIN_SIGMA * sigma, mu + MARGIN_SIGMA * sigma
    y = np.linspace(lo, hi, 400)
    f1 = norm.pdf(y, loc=0.0, scale=sigma)
    f2 = norm.pdf(y, loc=mu, scale=sigma)

    y_a = _vanilla_y(L)
    y_b_left, y_b_right = _silence_y_sides(L)

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, title in ((ax_a, "Setup A (vanilla)"), (ax_b, "Setup B (silence)")):
        ax.plot(y, f1, color="C0", label=r"$f_{y|H_1}$")
        ax.plot(y, f2, color="C1", label=r"$f_{y|H_2}$")
        ax.set_xlabel("y")
        ax.set_title(title)

    for t in y_a:
        ax_a.axvline(t, color="black", linestyle="--", linewidth=1)

    for t in np.concatenate([y_b_left, y_b_right]):
        ax_b.axvline(t, color="black", linestyle="--", linewidth=1)
    ax_b.axvspan(mu / 2 - DELTA, mu / 2 + DELTA, color="gray", alpha=0.25, label="silence bin")

    ax_a.legend()
    ax_b.legend()
    fig.suptitle(f"L={L} (R={int(np.log2(L))} bit/sensor), SNR=0dB (mu={mu:g}, sigma={sigma:g}), delta={DELTA:g}")
    fig.tight_layout()
    path = out_dir / f"thresholds_L{L}.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def run_threshold_diagrams(Ls):
    diagrams_dir = ROOT / f"threshold_diagrams_{DATE}" / f"{STAMP}_thresholds"
    diagrams_dir.mkdir(parents=True)
    paths = [plot_threshold_diagram(L, diagrams_dir) for L in Ls]
    readme = "\n".join(
        [
            f"# {diagrams_dir.name}",
            "",
            f"- Model: GaussianShift, SNR = 0 dB (mu={mu:g}, sigma={sigma:g}), p={p}",
            f"- delta = {DELTA:g} (fixed Setup B silence half-width)",
            f"- margin = {MARGIN_SIGMA:g} sigma beyond each mean",
            f"- L values: {Ls}",
            "- One figure per L: two panels (Setup A / Setup B) showing both Gaussian "
            "densities with the actual decision-boundary thresholds for that L; "
            "Setup B additionally shades the silence bin.",
        ]
    )
    (diagrams_dir / "README.md").write_text(readme + "\n")
    for path in paths:
        print(f"saved {path}")
    return diagrams_dir


if __name__ == "__main__":
    N_FIXED, RS = 1000, [1, 2, 3, 4, 5]
    RS_N_SWEEP = [1, 2]
    NS_SWEEP = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000]

    run_and_save_variant(N_FIXED, RS, tag=f"N{N_FIXED}", label=f"variant_N{N_FIXED}")
    run_and_save_n_sweep(RS_N_SWEEP, NS_SWEEP, tag="convergence")

    run_threshold_diagrams(sorted({2**r for r in RS} | {2**r for r in RS_N_SWEEP}))
