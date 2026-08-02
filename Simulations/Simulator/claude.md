# The DDMS Simulator

## Requirements

This project is building a Simulator for a Remote hypothesis testing problem.
The itself is discussed in DDMS.pdf. Read it. The math is taken strictly from there, unless told otherwise.
The simulator should be implemented in python.

The simulator is devided into three main classes - 
1. Statistical Model
2. Encoder
3. Fusion Center (FC)

The idea is that the statistical model and FC will behave the same way, and the encoder should be "replaceable" in order to check different setups and measure them against benchmarks.
The structure of the system is described in Sim Plan.pdf


## Statistical Model Class
 
**Role.** Generates the ground truth $H^\star$ and, for each sensor, the pair $(y^i, l^i)$. Nothing downstream — no policies, no actions, no $N$-dependence beyond a sample count.
 
### Base class (abstract)
 
**Config**
 
- `p` — $P(H^\star = H_1)$, default $0.5$
**Methods**
 
| Method | Returns | Notes |
|---|---|---|
| `sample_hypothesis(rng)` | `j ∈ {1,2}` | Bernoulli(`p`) |
| `sample(j, N, rng)` | `(y[N], l[N])` | i.i.d. draws given $H_j$; both arrays returned |
| `likelihood_ratio(y)` | `l` | vectorized $L(y) = f(y \mid H_2) / f(y \mid H_1)$; pure function, no RNG |
| `lr_cdf(t, j)` | $P(l \le t \mid H_j)$ | vectorized in `t`; needed for exact cell masses |
| `chernoff_information()` | `float` | unquantized ceiling, for benchmarking |
 
`sample` returns both $y$ and $l$ so the encoder can be handed either one. Threshold encoders consume `l`; anything later that wants raw observations still has `y`.
 
`lr_cdf` is the only non-obvious requirement — it is what lets the FC compute
 
$$P^{\gamma^i}(u \mid H_j) = F_j(t_d) - F_j(t_{d-1})$$
 
exactly, rather than by counting samples. One method, and it makes the whole exact path possible.
 
### `GaussianShift(mu, sigma=1.0, p=0.5)`
 
$y \sim \mathcal{N}(0, \sigma^2)$ under $H_1$, $y \sim \mathcal{N}(\mu, \sigma^2)$ under $H_2$.
 
- **SNR knob.** Expose an alternate constructor `GaussianShift.from_snr_db(snr_db, sigma=1.0)` with $\mathrm{SNR} = \mu^2 / \sigma^2$, so that $\mu = \sigma \cdot 10^{\mathrm{SNR_{dB}}/20}$.
- **Likelihood ratio.** $l = \exp\!\big( (\mu y - \mu^2/2) / \sigma^2 \big)$ — strictly increasing in $y$, so thresholds in $l$ map to thresholds in $y$.
- **CDF.** Invert to $y_t = \dfrac{\sigma^2 \log t}{\mu} + \dfrac{\mu}{2}$, then
  $$F_j(t) = \Phi\!\left( \frac{y_t - m_j}{\sigma} \right), \qquad m_1 = 0,\; m_2 = \mu.$$
  Handle $t \le 0 \Rightarrow F_j(t) = 0$.
- **Chernoff information.** $C = \mu^2 / (8\sigma^2)$.
### `DiscreteLR(p1, p2, p=0.5)`
 
Finite observation alphabet given by two pmfs; $l$ takes the finitely many values $p_2(y)/p_1(y)$ (with $\infty$ where $p_1(y) = 0$). Needed to run Example 1 of the paper, which is the fusion-center regression anchor.
 
### Tests
 
1. $\hat{\mathbb{E}}_{H_1}[l] \to 1$ over many samples.
2. Empirical CDF of sampled `l` matches `lr_cdf` (KS test).
3. `likelihood_ratio(sample(...)[0])` equals the returned `l` exactly.
4. $\mu = 0 \Rightarrow l \equiv 1$, Chernoff information $= 0$.
5. `lr_cdf` is monotone, tending to $0$ and $1$ at the ends.
### Extensibility
 
Adding a model later means subclassing and implementing four methods; nothing else in the simulator changes.
 
---

## Encoder Structure
 
**Role.** One sensor's policy $\gamma^i$. This is the swappable component of the simulator; everything else is fixed. Each encoder carries its own alphabet size `M`, so heterogeneous alphabets are supported — this breaks Assumption 1(iii) on purpose.
 
### Two I/O paths
 
**Realization path** — `encode(l) -> u`. In: likelihood ratios. Out: integer actions in $\{1,\dots,M\}$. Used for Monte-Carlo checks and for inspecting realized behaviour.
 
**Exact path** — `cell_probs(model, j) -> P[M]`. In: the statistical model and a hypothesis index. Out: $P^{\gamma}(u \mid H_j)$. This is the real interface — the FC needs these numbers, not samples, since they are the terms inside $\Delta_N$.
 
An encoder is therefore best understood as reporting an induced $2 \times M$ channel matrix. Everything downstream depends on the encoder only through that matrix.
 
### Base class (abstract)
 
**Config**
 
- `M` — alphabet size, per-encoder
**Methods**
 
| Method | Returns | Notes |
|---|---|---|
| `encode(l)` | `u[·]` | vectorized $l \mapsto u$ |
| `cell_probs(model, j)` | `P[M]` | exact $P^{\gamma}(u \mid H_j)$ for $u = 1,\dots,M$ |
| `rate()` | `float` | $\log_2 M$ bits |
| `describe()` | `str` | label for logs and figure legends |
 
### `ThresholdEncoder(thresholds)`
 
The paper's $\Gamma^i_{\mathrm{TS}}$ (Definition 1). Thresholds $0 < t_1 < \dots < t_{m-1} < \infty$ define the bins
 
$$B_1 = [0, t_1], \qquad B_d = (t_{d-1}, t_d], \qquad B_m = (t_{m-1}, \infty],$$
 
with $M = $ `len(thresholds) + 1`.
 
- `encode(l)` = `searchsorted(thresholds, l) + 1`
- `cell_probs(model, j)` = successive differences of `model.lr_cdf(t, j)`, with $F_j(0^-) = 0$ and $F_j(\infty) = 1$ — one vectorized CDF call per hypothesis
**`LRTEncoder(t=1.0)`** — convenience subclass, the $M = 2$ single-threshold likelihood ratio test.
 
### `EncoderBank`
 
Holds the profile $\gamma^{1:N}$ as a list of encoders, possibly with different `M`. Asymmetric profiles are the point — Example 1 of the paper lives here.
 
**Constructors**
 
- `EncoderBank.identical(encoder, N)` — symmetric profile
- `EncoderBank.from_fractions([(enc, c_k)], N)` — $K$ distinct policies used by fractions $c_k$
- `EncoderBank(list_of_encoders)` — fully general
**Methods**
 
| Method | Returns | Notes |
|---|---|---|
| `cell_probs_matrix(model, j)` | `P[N, M_max]` | zero-padded for ragged alphabets |
| `M_list()` | `[M_1, ..., M_N]` | per-sensor alphabet sizes |
| `total_rate()` | `float` | $\sum_i \log_2 M_i$ bits |
| `uniform_alphabet()` | `bool` | whether Assumption 1(iii) holds |
 
Padding with zero mass under both hypotheses means unused symbols contribute nothing to $\Delta_N$, so the FC stays fully vectorized.
 
`uniform_alphabet()` is reported, not enforced — every run is labelled as inside or outside the paper's assumptions. Comparisons against a heterogeneous bank should be made at matched `total_rate()`, not matched $N$.
 
### Tests
 
1. `cell_probs` sums to 1 under each hypothesis; all entries $\ge 0$.
2. Histogram of `encode` on sampled $l$ matches `cell_probs`.
3. Change of measure: $P^{\gamma}(u \mid H_2) = \mathbb{E}_{H_1}\!\big[\, l \cdot \mathbb{1}\{\gamma(l) = u\} \,\big]$.
4. Trivial encoder ($M = 1$): both conditionals equal 1, exponent 0.
5. Adding a threshold never decreases the exponent.
6. Ragged bank: padded columns carry zero mass under both hypotheses.
### Deliberately out of scope
 
- **Permuted-label encoders** (Definition 1(2)) — a relabelling with identical cell masses and no effect on $J^N$ once the FC knows $\gamma^{1:N}$.
- **Randomized kernels** $\gamma(u \mid l)$ — only enter from Lemma 4 onward.
Both are straightforward to add later without changing the interface.

## Fusion Center

**Role.** Fixed component, never swapped. Consumes the encoder bank's channel matrices plus the prior, and produces (a) decisions and (b) the exact error probability $J^N$. It knows $\gamma^{1:N}$ only through the matrices $P[N, M_{\max}]$ — this is the content of the known-common-randomization assumption.

### Config

- `p` — $P(H^\star = H_1)$, taken from the model
- `t` — decision threshold, default $t = \dfrac{1}{N}\log\dfrac{1-p}{p}$ (DDMS eq. 21, $Nt = \log\frac{1-p}{p}$), evaluated per call since it depends on $N$; overridable for sweeping the operating point

Sign check: large $p$ gives $t < 0$, so $\Delta_N \ge t$ holds more easily and the FC leans toward $H_1$. At $p = 1/2$, $t = 0$.

### Decision path

`decide(u, P1, P2) -> Ĥ`

Given a realized action profile, look up $P1[i, u^i]$ and $P2[i, u^i]$, form

$$\Delta_N(u^{1:N}; \gamma^{1:N}) = \frac{1}{N}\sum_{i=1}^{N} \log \frac{P^{\gamma^i}(u^i \mid H_1)}{P^{\gamma^i}(u^i \mid H_2)},$$

and return $H_1$ if $\Delta_N \ge t$, else $H_2$.

### Exact error path

$$J^N = p \cdot P(\hat H = 2 \mid H_1) + (1-p) \cdot P(\hat H = 1 \mid H_2)$$

Each term is a sum over action profiles whose membership depends on $\Delta_N$, which couples all $N$ coordinates — so the sum does not factor, and naive enumeration costs $\prod_i M_i$ profiles. Unusable beyond tiny $N$.

**Convolution.** $\Delta_N$ is a *sum* of per-sensor terms

$$\lambda^i = \log \frac{P^{\gamma^i}(u^i \mid H_1)}{P^{\gamma^i}(u^i \mid H_2)},$$

independent given $H$. Only the law of the scalar $\sum_i \lambda^i$ is needed — the $N$-fold convolution of $N$ atomic distributions with $M_i$ atoms each, whose weights are row $i$ of `P1` or `P2`. Polynomial cost, exact result. The tail sum is then a one-dimensional accumulation over atoms falling below $t$.

Two regimes, selected automatically from the bank:

- **Identical or $K$-grouped.** Sensors sharing a policy are exchangeable, so their contribution depends only on symbol *counts* — multinomial over compositions rather than profiles, then convolve the $K$ groups.
- **Fully heterogeneous.** Sequential convolution of $N$ atomic distributions, merging identical atoms. Support can grow, so cap $N$ and fall back to Monte Carlo above the cap.

All arithmetic in log domain via `gammaln` / `logsumexp`; raw probabilities are never materialized.

### Empirical path (validation only)

Draw $H^\star$, draw $y^{1:N}$, encode, call `decide`, count mistakes over $T$ trials:

$$\hat J^N = \frac{1}{T}\sum_{k=1}^{T} \mathbb{1}\{\hat H_k \neq H_k\}$$

Since $J^N \sim e^{-\beta N}$, this is unusable in the regime of interest — at $N = 40$ it estimates a quantity near $10^{-6}$. It exists solely as a consistency harness, run at small $N$ and low SNR where $J^N \sim 0.1$, to confirm that `encode`, `decide`, and the convolution machinery agree.

**Role split:** the exact path produces every reported number and every figure; the empirical path is never plotted.

### Outputs

| Method | Returns | Notes |
|---|---|---|
| `log_error_prob(bank, model)` | $\log J^N$ | exact |
| `total_exponent(bank, model)` | $-\log J^N$ | **headline metric** |
| `normalized_exponent(bank, model)` | $-\frac{1}{N}\log J^N = J^N_{EE}$ | per-sensor |
| `chernoff_bound(bank, model)` | $-\min_{\alpha}\sum_i \log M(\alpha, \gamma^i)$ | DDMS eq. 24 |
| `empirical_error_prob(bank, model, T, rng)` | $\hat J^N$ | validation only |

Report $-\log J^N$ rather than $J^N_{EE}$ when comparing schemes: dividing by $N$ hides gains that come from changing how many sensors speak, which is exactly what a rate-reallocation scheme does.

### Tests

1. $N = 1$: $J^N$ matches direct two-symbol computation.
2. $N = 2$ with Example 1 parameters: reproduces the exact values $19/90$ (asymmetric), $2/9$ (both B), $53/225$ (both A) at `rtol=1e-12`, and the ordering asymmetric $<$ B $<$ A. These are the paper's rounded $0.21$, $0.22$, $0.23$ — note $53/225 = 0.2356$ actually rounds to $0.24$; the paper's $0.23$ is a truncation.
3. Empirical path converges to `log_error_prob` at small $N$ and low SNR.
4. `total_exponent` $\ge$ `chernoff_bound`. (At $p = 1/2$ the Chernoff bound, DDMS eqs. 37–40, upper-bounds $J^N$ for every $N$, so the achieved exponent is at least the bound.)
5. Trivial bank ($M = 1$): $J^N = \min(p, 1-p)$.
6. Grouped and heterogeneous code paths agree on the same identical bank.
7. `decide` and the exact path use the same $t$; changing `t` moves both consistently.

## Technical Decisions
 
### Stack
 
- Python 3.12+, with `numpy` and `scipy` only. `matplotlib` for figures, `pytest` for tests.
- `uv` for environment and dependency management.
- No config framework, no CLI framework, no server, no database. Plain dataclasses and function arguments.
### Layout
 
```
src/ddms/model.py       # Statistical Model
src/ddms/encoder.py     # Encoder, EncoderBank
src/ddms/fusion.py      # Fusion Center
src/ddms/sweep.py       # parameter sweeps, returns arrays
src/ddms/runs.py        # save_run / load_run: timestamped run directories
figs/                   # one script per figure
runs/                   # one directory per sweep run: data.json + figures, committed
tests/
docs/PLAN.md
```
 
One module per class, matching the sections of this document.
 
### Numerics
 
- All probability arithmetic in log domain, using `scipy.special.logsumexp` and `gammaln`. Raw probabilities are never materialized — $J^N$ reaches $10^{-15}$ and below in the regime of interest.
- float64 throughout. If a result needs more precision, the formulation is wrong.
- Natural log internally. Bits appear only in `rate()` and `total_rate()`.
- Actions are 0-indexed internally, 1-indexed only in printed output and documentation.
- Zero-mass padding is $-\infty$ in log domain. Indexing a padded entry is a bug and should surface loudly, not contribute silently.
### Reproducibility
 
- Every stochastic function takes an explicit `np.random.Generator`. No module-level RNG, no global seeding.
- The empirical path is validation-only; its seeds are fixed in the tests.
### Testing
 
- `pytest`. Each section's Tests list becomes one test file.
- Two tolerance classes: exact-path comparisons at `rtol=1e-12`, Monte-Carlo comparisons at a fixed seed with an explicit tolerance.
- Example 1 of the paper is the regression anchor.
### Sweeps and figures
 
The three classes are a pure computation pipeline and never touch the filesystem or matplotlib. Two layers sit above them:
 
- **`sweep.py`** — plain functions (`sweep_over_N`, `sweep_over_rate`, `sweep_over_snr`) returning arrays. This is where the nontrivial logic lives: building the right bank at each point, matching total rate across schemes. Tested. Never touches the filesystem.
- **`runs.py`** — `save_run(results, label, meta)` writes a sweep's raw arrays plus metadata to `runs/<timestamp>_<label>/data.json` and returns the run directory; figure scripts save their PDFs into that same directory. `load_run(dir)` restores the arrays. Every reported figure therefore sits next to the exact JSON that produced it, so runs made on different branches (different encoders) can be pulled together and overlaid later — `figs/fig_compare_runs.py` does exactly that. `runs/` is committed, not gitignored.
- **`figs/fig_*.py`** — one script per figure, roughly 30 lines each: call a sweep, `save_run`, plot, save a vector PDF into the run directory for direct LaTeX inclusion. Run with `python figs/fig_name.py`. Default matplotlib styling. Disposable; anything ad hoc (annotations, insets, closed-form overlays) lives here and never leaks into code that produces reported numbers.
No caching until a sweep is measurably slow.
 
### Guards
 
- `N_max` on the heterogeneous convolution path, raising an explicit error rather than silently falling back.
- `uniform_alphabet()` recorded in every run's metadata, so results are always labelled as inside or outside Assumption 1(iii).
### Coding standards
 
1. Keep it simple. No over-engineering, no unnecessary defensive programming, no extra features.
2. Be concise. Minimal README. No emojis.
3. When hitting issues, identify the root cause before fixing. Prove with evidence, then fix the cause.

## Coding standards

1. Use latest versions of libraries and idiomatic approaches as of today
2. Keep it simple - NEVER over-engineer, ALWAYS simplify, NO unnecessary defensive programming. No extra features - focus on simplicity.
3. Be concise. Keep README minimal. IMPORTANT: no emojis ever
4. When hitting issues, always identify root cause before trying a fix. Do not guess. Prove with evidence, then fix the root cause.

## Working documentation

All documents for planning and executing this project will be in the docs/ directory.
Please review the docs/PLAN.md document before proceeding.