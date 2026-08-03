# Execution Plan

Build the DDMS simulator specified in `claude.md`, with math taken from
DDMS.pdf and structure from Sim Plan.pdf. The encoder is the swappable
component; `VanillaEncoder` in `src/ddms/encoder.py` is the empty template
to be filled in later.

## Steps

1. **Scaffold** — `uv` project, src layout (`src/ddms/`), pytest, numpy/scipy/matplotlib only. Done.
2. **Statistical Model** (`model.py`) — abstract `StatisticalModel`;
   `GaussianShift` (with `from_snr_db`, closed-form `lr_cdf` and Chernoff
   information mu^2/8sigma^2); `DiscreteLR` (finite-alphabet model needed
   for the Example 1 regression anchor). Done.
3. **Encoder** (`encoder.py`) — abstract `Encoder` (encode / cell_probs /
   rate / describe); **`VanillaEncoder` empty template**; `ThresholdEncoder`
   (Definition 1); `LRTEncoder`; `EncoderBank` with `identical`,
   `from_fractions` (largest remainder), padded `cell_probs_matrix`. Done.
4. **Fusion Center** (`fusion.py`) — `decide` (eq. 16-17, batchable);
   exact `log_error_prob` via convolution of per-sensor log-likelihood-ratio
   atoms, multinomial compositions for grouped sensors, sequential
   convolution across distinct policies with atom merging, all log-domain;
   `chernoff_bound` (eq. 24); Monte-Carlo `empirical_error_prob`
   (validation only); `N_max` guard on the heterogeneous path. Done.
5. **Sweeps** (`sweep.py`) — `sweep_over_N`, `sweep_over_snr`,
   `sweep_over_rate`, returning dicts of arrays. Done.
6. **Figures** — `figs/fig_exponent_vs_N.py` as the first one-script figure. Done.
7. **Tests** — one file per section, per the lists in `claude.md`;
   Example 1 as the regression anchor. Done.
8. **Run persistence** (`runs.py`) — every sweep is saved via
   `save_run(results, label, meta, group=None, session=None)` to a fresh
   `runs/<session>/<group>_<date>/<timestamp>_<label>/` directory holding
   `data.json` (raw arrays + metadata) with figure PDFs saved alongside;
   `group` names the simulation's parameters and defaults to `label`,
   `date` is day-only so a simulation's runs from one day share a parent
   folder, `session` buckets by encoder family (e.g. `M2`, `M4`) so
   unrelated families never interleave alphabetically; `load_run`
   restores arrays and `figs/fig_compare_runs.py` overlays several runs.
   Run directories are committed so branches testing different encoders
   can be compared. Done.

## Deviations from claude.md (with reasons)

- **Example 1 values.** The exact errors are 19/90 = 0.2111 (asymmetric),
  2/9 = 0.2222 (both B), 53/225 = 0.23556 (both A). The paper's rounded
  0.21 / 0.22 / 0.23 are matched qualitatively (same ordering); the tests
  anchor to the exact fractions at rtol 1e-12. Note 53/225 rounds to 0.24,
  not 0.23 — the paper's 0.23 appears to be a truncation.
- **Chernoff test direction.** claude.md test 4 says `total_exponent` <=
  `chernoff_bound`. At p = 1/2 the Chernoff bound (DDMS eqs. 37-40) is an
  upper bound on J^N for every N, so the achieved exponent is *at least*
  the bound; the test asserts `total_exponent >= chernoff_bound`.
- **`DiscreteLR` model added.** Not in the spec, but required to run
  Example 1 (finite observation alphabet with a zero-probability cell).

## Example 1 (regression anchor)

N = 2, p = 1/2, Y = {1,2,3}, U = {1,2}.
P(y|H1) = (4/5, 1/5, 0), P(y|H2) = (1/3, 1/3, 1/3),
so l takes values 5/12, 5/3, inf.
Policy A: u=1 iff y=1 (threshold at l=1). Policy B: u=1 iff y in {1,2}
(threshold at l=2). Asymmetric (A,B) beats both symmetric profiles.

## Conventions

- Actions 0-indexed internally; 1-indexed only in printed output.
- Natural log internally; bits only in `rate()` / `total_rate()`.
- Zero-mass padding is -inf in log domain; indexing a padded cell raises.
- Every stochastic function takes an explicit `np.random.Generator`.
- The exact path produces every reported number; the empirical path is
  validation only and never plotted.
