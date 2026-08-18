"""Simulation 2 (Sim_plan.tex section 2): fix N, sweep R = log2(L) bits per
sensor, for Setup A (always-transmit L-region threshold quantizer) and
Setup B (silence-bin variant: L/2 active regions per side of a fixed-width
silence bin around mu/2).

Two variants, both at SNR 0dB (mu = sigma = 1):
  - Variant A: N=10,  R in {1,2,3,4}  (L=2,4,8,16)  -- fine-grained R.
  - Variant B: N=200, R in {1,2}      (L=2,4)       -- large N, coarse R.
    Capped there by the exact fusion path's C(N+M-1,M-1) cost: Setup B at
    L=4, N=200 is already ~70M compositions (~12 min for that one point);
    L=8/16 at N=200 (let alone N=100) are combinatorially infeasible
    (C(116,16) ~= 1.9e19 at N=100, L=16).

Setup B's silence half-width delta is a fixed constant (not optimized, not
swept), annotated on every plot per explicit request. Threshold construction
(vanilla_thresholds/silence_thresholds) is local to this script -- it only
builds threshold arrays for the existing ThresholdEncoder/SilenceEncoder, no
new Encoder subclass.

Usage: python figs/fig_R_bits_per_sensor.py
All output lands under runs/Varying_R/.
"""

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from ddms import EncoderBank, FusionCenter, GaussianShift, SilenceEncoder, ThresholdEncoder, save_run

SESSION = "Varying_R"
MARGIN_SIGMA = 3.0  # how far the outer thresholds extend beyond each mean
DELTA = 0.25  # fixed Setup B silence half-width; must be < mu/2

model = GaussianShift.from_snr_db(0.0)
mu, sigma, p = model.mu, model.sigma, model.p
assert DELTA < mu / 2, "silence half-width must leave both sides non-degenerate"

ROOT = Path("runs") / SESSION
STAMP = datetime.now().strftime("%Y-%m-%d_%H%M%S")
DATE = STAMP[:10]


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


def run_variant(N, Rs, tag):
    fc = FusionCenter(p=p)
    rows_a, rows_b = [], []
    for R in Rs:
        L = 2**R
        enc_a, enc_b = make_setup_a(L), make_setup_b(L)
        bank_a = EncoderBank.identical(enc_a, N)
        bank_b = EncoderBank.identical(enc_b, N)

        q1, q2 = enc_b.cell_probs(model, 1), enc_b.cell_probs(model, 2)
        silent = list(enc_b.silent_indices)
        p_silent = float(p * q1[silent].sum() + (1 - p) * q2[silent].sum())

        rows_a.append(
            {
                "R": float(R),
                "L": float(L),
                "N": float(N),
                "total_exponent": fc.total_exponent(bank_a, model),
                "normalized_exponent": fc.normalized_exponent(bank_a, model),
                "N_active": float(N),
                "mean_rate": float(R),
            }
        )
        rows_b.append(
            {
                "R": float(R),
                "L": float(L),
                "N": float(N),
                "total_exponent": fc.total_exponent(bank_b, model),
                "normalized_exponent": fc.normalized_exponent(bank_b, model),
                "N_active": N * (1.0 - p_silent),
                "mean_rate": enc_b.mean_rate(model),
            }
        )
        print(f"[{tag}] R={R} L={L} done "
              f"(A: -logJ={rows_a[-1]['total_exponent']:.4f}, "
              f"B: -logJ={rows_b[-1]['total_exponent']:.4f}, "
              f"B mean_rate={rows_b[-1]['mean_rate']:.4f})")
    return collect(rows_a), collect(rows_b)


def md_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def rate_table(res_a, res_b):
    headers = ["R (bits)", "L", "Setup", "N", "N_active", "mean_rate (bits/sensor)"]
    rows = []
    for i in range(len(res_a["R"])):
        rows.append(
            [
                f"{res_a['R'][i]:.0f}",
                f"{res_a['L'][i]:.0f}",
                "A (vanilla)",
                f"{res_a['N'][i]:.0f}",
                f"{res_a['N_active'][i]:.1f}",
                f"{res_a['mean_rate'][i]:.4f}",
            ]
        )
        rows.append(
            [
                f"{res_b['R'][i]:.0f}",
                f"{res_b['L'][i]:.0f}",
                "B (silence)",
                f"{res_b['N'][i]:.0f}",
                f"{res_b['N_active'][i]:.1f}",
                f"{res_b['mean_rate'][i]:.4f}",
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

    fig.suptitle(f"{label}: N={N} (fixed), SNR=0dB (mu={mu:g}, sigma={sigma:g}), delta={DELTA:g}")
    fig.tight_layout()
    path = out_dir / f"fig_{label}.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def run_and_save_variant(N, Rs, tag, label):
    res_a, res_b = run_variant(N, Rs, tag)

    group = f"R_sweep_{tag}_snrp0"
    meta_common = {"model": {"type": "GaussianShift", "snr_db": 0.0, "mu": mu, "sigma": sigma, "p": p}, "delta": DELTA}

    dir_a = save_run(
        res_a,
        label=f"{tag}_setupA_vs_R",
        meta={**meta_common, "bank": f"identical ThresholdEncoder(L)", "x": "R", "N": N},
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
    N_FINE, RS_FINE = 10, [1, 2, 3, 4]
    N_LARGE, RS_LARGE = 200, [1, 2]

    run_and_save_variant(N_FINE, RS_FINE, tag=f"N{N_FINE}", label=f"variantA_N{N_FINE}")
    run_and_save_variant(N_LARGE, RS_LARGE, tag=f"N{N_LARGE}", label=f"variantB_N{N_LARGE}")

    all_Ls = sorted({2**r for r in RS_FINE} | {2**r for r in RS_LARGE})
    run_threshold_diagrams(all_Ls)
