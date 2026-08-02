# Simulation Architecture Plan
### Decentralized Detection with Many Sensors — Finite-$N$ Study

Scope: finite $N$, Assumption 1 holds (Ziv's track). Phase 3 covers the dropped-1(iii) variant (Oren's track) sharing the same base.

Notation follows the paper: $P^{\gamma^i}(u \mid H_j)$ denotes the action distribution induced by encoder $\gamma^i$ under hypothesis $H_j$.

---

## 0. Core Principle

- Under Assumption 1, observations $y^{1:N}$ are conditionally i.i.d. given $H$.
- Any encoder affects the fusion center **only** through the pair
  $$P^{\gamma^i}(u \mid H_1),\quad P^{\gamma^i}(u \mid H_2), \qquad u \in \mathbb{U}^i$$
- That pair is the **single interface**. Everything above it (observation law, thresholds, internal state) and below it (fusion, $J^N$, exponent, rate) is written once and never touched again.
- Threshold, silence-augmented, temporal-counter, and randomized-kernel encoders all satisfy the same contract.

**Definition to use** (the $l$-form, matching arXiv eq. 12):
$$P^{\gamma^i}(U^i = u \mid H_j) = \int \gamma^i(U^i = u \mid l^i)\, P(dl^i \mid H_j)$$

Use this integral form, *not* the "probability that $l^i$ lands in bin $B^i_d$" form. The partition form only holds for deterministic threshold policies; the integral form also covers randomized kernels, so the base needs no change when those are added later.

The paper's $g(H, u; \gamma^R)$ (eq. 27) is the same object written in $y$-space. Identify the two explicitly at first use in the write-up.

---

## 1. Module Layout

```
model/       GaussianShift, Atomic       -> bin_probs(thresholds) -> (pmf|H1, pmf|H2)
encoder/     Threshold, Silence, Counter -> action_pmf(...) -> (pmf|H1, pmf|H2) + alphabet spec
rate/        FixedAlphabet, PaidSilence, FreeSilence -> rate(sensors) -> total bits
fusion/      LLR-distribution interface; 3 backends
metrics/     J^N, -log J^N, J_EE, C (shared-s), r, eta = C/r
optimize/    Objective strategies, group-parameterized search
policy/      realized ensemble + full provenance record
experiment/  R-sweep driver, sweep configs, figure emitters
```

**Naming note.** `P_gamma` is awkward as an identifier and bare `P` collides with priors and observation measures. Suggest `p_act[i]` or `pmf[i]` in code, with a comment fixing it to $P^{\gamma^i}(\cdot \mid H_j)$. Decide once; keep consistent across modules.

---

## 2. Key Design Decisions

### Model
- Gaussian mean-shift primary: $\mathcal{N}(0,\sigma^2)$ under $H_1$, $\mathcal{N}(\mu,\sigma^2)$ under $H_2$. Single SNR knob $\mu/\sigma$.
- $l(y)$ monotone in $y$, so thresholds in $l$-space map bijectively to $y$-space and bin probabilities are **exact** via $\Phi$ — no discretization error.
- `Atomic` backend kept alongside: finite support with explicit per-atom probabilities. Needed for Example 1, cheap to add, and gives a sandbox for a sparse three-region observation law later.

### Alphabet
- $m = |\mathbb{U}|$ is a first-class parameter. Never hard-code $m=2$, including in fusion.
- Sweeping $m$ stays strictly inside Assumption 1(iii), which fixes $\mathbb{U}$ as common and finite but not *which* finite set.
- A threshold encoder with alphabet $m$ has $m-1$ free thresholds.

### Rate model — separate from the encoder
The single most important structural choice. The encoder must not decide what it costs.

- Signature: `rate(sensors) -> total_bits`, i.e. $R = \sum_{i=1}^N r_i$ — **not** $N \cdot r$. The identical case is computed, not assumed.

| Model | Per-sensor rate | Purpose |
|---|---|---|
| `FixedAlphabet` | $\log_2 m$ | Vanilla benchmark |
| `PaidSilence` (A′) | $\log_2(m{+}1)$ | Silence is just another symbol |
| `FreeSilence` (B) | $\bar p_{\text{send}} \cdot \log_2 m$ | Silence costs zero |

- $\bar p_{\text{send}} = \sum_j P(H_j)\big(1 - P^{\gamma}(\varnothing \mid H_j)\big)$ — prior-averaged across hypotheses. State this convention explicitly; a silent choice here later looks like a bug.
- Running the *same* silence encoder under `PaidSilence` vs. `FreeSilence` **is** the Scheme A′ ablation — one config flag, not a separate codepath. This turns "the gain is mostly a channel-model assumption" from an assertion into a measured decomposition.

### Fusion — an interface, not one algorithm
General object: the distribution of $\sum_i \lambda^i$ under each hypothesis, where
$$\lambda^i(u) = \log \frac{P^{\gamma^i}(u \mid H_1)}{P^{\gamma^i}(u \mid H_2)}, \qquad N\Delta_N = \sum_{i=1}^N \lambda^i$$
$$J^N = p\,\Pr\Big[\textstyle\sum_i \lambda^i < Nt \,\Big|\, H_1\Big] + (1-p)\,\Pr\Big[\textstyle\sum_i \lambda^i \ge Nt \,\Big|\, H_2\Big]$$

| Backend | When | Cost |
|---|---|---|
| Type enumeration | Identical sensors | $\binom{N+m-1}{m-1}$ |
| Grouped types | $G$ policy groups | $\prod_g \binom{N_g+m_g-1}{m_g-1}$ |
| Sequential LLR convolution | Fully heterogeneous | $O(N \cdot \lvert\text{support}\rvert)$ |

- The first two are special cases of the third — type enumeration *is* the collision-exploiting convolution when all sensors share a $\lambda$.
- Implement convolution as the general path; keep type enumeration as an optimization. **Cross-validate**: $G=1$ must agree to machine precision. Permanent test.
- For identical sensors the type vector $n = (n_1,\dots,n_m)$, $\sum_u n_u = N$, is sufficient at the FC:
  $$J^N = \sum_n \min\big\{p\,P(n \mid H_1),\ (1-p)\,P(n \mid H_2)\big\}$$
  with $P(n \mid H_j)$ multinomial. Exact — no Monte Carlo, no sampling noise.
- Log-domain throughout (`gammaln`, `logsumexp`). Never materialize raw probabilities.
- If support grows too large, quantize the LLR axis **upward and downward** to obtain a genuine two-sided bracket on $J^N$, not a point estimate.

### Metrics
- Chernoff information with a **single shared $s$** across sensors:
  $$C = -\min_{s \in [0,1]} \sum_{i=1}^N \log \sum_{u \in \mathbb{U}^i} P^{\gamma^i}(u \mid H_1)^s\, P^{\gamma^i}(u \mid H_2)^{1-s}$$
- **Do not expose a per-sensor $C_i$.** $\sum_i C_i$ (each at its own $s_i^\star$) is a loose upper bound, and a per-sensor helper invites exactly the wrong aggregation. Shared-$s$ optimization is the only public entry point.
- $-\log J^N \approx R \cdot \eta$ with $\eta = C/r$ (exponent per bit). Scheme comparison collapses to a slope ratio, and threshold optimization decouples from $N$ entirely.
- Emit $N\delta$ (expected active count) as a diagnostic column on **every** run.
- Report unnormalized $-\log J^N$, not $J_{EE} = \log(J^N)/N$, for cross-scheme comparison — dividing by $N$ masks the sensor-count gain.

---

## 3. Phase 1 — Validated Vanilla Benchmark

Build: `GaussianShift` + `Atomic` + type/grouped fusion + `Threshold` + `FixedAlphabet` + `Metrics`.

Add **grouped sensors** and **mixture-over-policy-profiles** now, not later — both are needed for Example 1 and both are painful to retrofit.

### Validation suite — run before any figure

1. **Example 1 exact reproduction.** `Atomic` model, $P(y \mid H_1) = (\tfrac45, \tfrac15, 0)$, $P(y \mid H_2) = (\tfrac13, \tfrac13, \tfrac13)$, $N=2$, $m=2$. Must reproduce to machine precision:
   - $19/90 \approx 0.21$ — asymmetric $(A, B)$
   - $53/225 \approx 0.23$ — both $A$
   - $2/9 \approx 0.22$ — both $B$
   - $0.22$ — symmetric-independent randomization
   The third atom has $l = \infty$: free stress test of log-domain $\log 0$ handling.
   Requires grouped fusion plus $J = \sum_{\text{profiles}} P(\text{profile}) J(\text{profile})$ with the FC knowing the realized profile — exactly the $L^N_{\text{EX}}$ vs. $L^N_{\text{PR,SYM}}$ distinction. Reproducing the $0.21$ vs. $0.22$ gap is a concrete illustration for the commentary section.
2. **Monotonicity.** $-\log J^N$ nondecreasing in $N$ at fixed policy; nondecreasing in $m$ at fixed model.
3. **Chernoff consistency.** $-\log J^N / N \to C$ as $N$ grows, fixed policy.
4. **Degenerate cases.** $P^{\gamma}(\cdot \mid H_1) = P^{\gamma}(\cdot \mid H_2)$ gives $J^N = \min(p, 1-p)$ and $C = 0$. Perfect sensor gives $J^N = 0$.
5. **Optimizer sanity.** $m=2$ Gaussian: numerical optimum matches a direct 1-D scan of $C(\tau)$.
6. **Rate-model equivalence.** `Silence` + `PaidSilence` with $P^{\gamma}(\varnothing) \to 0$ converges to `Threshold` + `FixedAlphabet` at $m{+}1$.

### Benchmark definition
- Vanilla at budget $R$ is the **upper envelope** $\max_m \eta_{\text{vanilla}}(m)$ over $m \in \{2,\dots,M\}$, with $N = \lfloor R/\log_2 m \rfloor$.
- Anything less is a strawman and will be the first thing questioned.
- Expectation: envelope attained at $m=2$, since Chernoff information saturates in $m$ faster than $\log_2 m$ grows. **Verify, do not assume.** If true it is a quotable finding on its own: *under a fixed-alphabet channel model, one bit per sensor is rate-optimal.*

---

## 4. Phase 2 — R-Sweep and Silence Encoder

### Protocol
- Sweep **integer $N$**; plot $R = Nr$ as a derived $x$-axis. Imposing $R$ and taking $N = \lfloor R/r \rfloor$ produces fake staircases.
- Since $-\log J^N \approx N C$ and $R = Nr$, the sweep is a family of near-straight lines with slope $\eta = C/r$. Gain = slope ratio.

### Figures
- **Primary.** Exact $-\log J^N$ vs. $R$, one curve per scheme, with the asymptotic line $R\eta^\star$ overlaid. **The gap between exact and line is the finite-$N$ story** — the interesting part.
- **Secondary.** $\eta^\star$ vs. $\delta$ (or SNR), with the closed-form $\approx 2/\delta$ overlay, plus the A′ ablation curve sitting between vanilla and B.
- Shade the region where $N\delta \lesssim 1$ as asymptotically invalid.

### Two mandatory framing decisions
- **Two budgets, not one.** Maximizing $\eta$ alone drives $p_{\text{send}} \to 0$ and $N \to \infty$ — a divergence, not a result. The well-posed problem constrains both rate $R$ **and** a physical population $N \le N_{\max}$. At small $\delta$, $N_{\max}$ binds and the $2/\delta$ gain saturates. Build `N_max` into the experiment driver from the start.
- **What $R$ means.** Expected rate vs. worst-case rate. Free silence under an expected-rate budget implicitly assumes something about the multiple-access channel. State the convention once, prominently — this is the crux of the channel-model critique.

### Finite-$N$ validity floor
With $p_{\text{send}} = \delta$, expected active count is $N\delta$. Below $\approx 1$ the asymptotic exponent is meaningless and exact $J^N$ will sit far from the line. Given the preference for finite-$N$ arguments over asymptotic ones, this diagnostic is probably the most defensible thing in the study.

---

## 5. Phase 3 — Heterogeneous Sensors (Assumption 1(iii) dropped)

The base is shared. Only two generalizations were needed, and both are already specified in Phases 1–2:
- Fusion as an LLR-distribution interface (convolution backend).
- Rate over ensembles, $R = \sum_i r_i$.

### What survives
- **Lemma 1 survives dropping 1(iii).** Its argument is a per-sensor best response: fix $\gamma^{-i}$, observe the subproblem is linear in $l^i$, conclude threshold. That derivation never uses $\mathbb{U}^i = \mathbb{U}$. Sensors remain threshold-type, so the encoder layer is genuinely shared with Oren's track.

### Unifying observation
Heterogeneous $|\mathbb{U}^i|$ and free-silence are **the same problem in different clothes** — both are rate reallocation across sensors. Oren does it by assigning different alphabet sizes ex ante; Scheme B does it by letting a sensor spend zero when uninformative. A shared rate layer puts both on one plot against the same vanilla benchmark: a stronger joint result than two separate comparisons.

### Caveat to state explicitly
Theorems 4 and 5 do not apply. Theorem 1's compactness argument is set up over a common $\mathbb{U}$, and exchangeability is not even meaningful when sensors have structurally different action spaces — permuting them is not a symmetry of the problem. Heterogeneous results are lower bounds from a restricted (threshold) class, with no existence or global-optimality guarantee. Unstated this looks like an oversight; stated it looks like rigor.

---

## 6. Phase 4 — Encoder Optimization

Optimizer chain: `params -> Encoder -> (P^{gamma^i}) -> Metrics -> scalar`. Every layer already parameterized.

### Three objectives, interchangeable via a strategy object

| Objective | $N$-dependent | Cost/eval | Use |
|---|---|---|---|
| $\eta = C/r$ | No | cheap | Design/screening; sets the $R$-sweep slope |
| $-\log J^N$ at fixed $N$ | Yes | fusion | Honest finite-$N$ answer |
| $-\log J^N$ s.t. $Nr \le R$, $N \le N_{\max}$ | $N$ is a variable | fusion × $N$-sweep | The real design problem |

- Handle integer $N$ as an **outer loop** over the small feasible set ($N \le \min(N_{\max}, R/r_{\min})$); optimize continuous thresholds inside. Sidesteps mixed-integer machinery entirely.
- **Open question worth measuring:** does $\arg\max_\tau$ drift with $N$? Impossible for the $\eta$-objective by construction; possible for the finite-$N$ objective. Measuring the drift is a genuine finite-$N$ result.

### Two parameter regimes
- **Symmetric search.** One threshold vector shared by all sensors. Dimension $m-1$ (plus silence edges). Used for the vanilla benchmark and Scheme B.
- **Asymmetric search.** Per-sensor vectors, dimension $N(m-1)$. Needed for the heterogeneous setting and for probing Example 1-style asymmetry.

Under Assumption 1 sensors are exchangeable, so the objective is **permutation-invariant**: every optimum has $N!$ relabelings and a naive optimizer will wander among them, reporting meaningless restart spread. Break it by canonicalizing (sort by threshold or by $r_i$), or better, parameterize by **group sizes + per-group thresholds** — collapses the dimension to $G(m{-}1) + G$ and maps directly onto the grouped-fusion backend already built.

### Search hazards and fixes
- **Non-convex.** $C(\tau)$ over multiple thresholds is not convex. Multi-start; log restart spread.
- **Flat regions.** Under free silence, once $p_{\text{send}}$ is small the objective barely moves with the silence edges and gradient-free methods stall. Reparameterize in $p_{\text{send}}$ / cumulative probability rather than raw threshold values — equalizes sensitivity, boxes the search to $[0,1]$.
- **Ordering constraints.** $\tau_1 < \tau_2 < \cdots$ enforced via softplus/cumsum increments, **not** penalties. Penalties introduce spurious local structure.
- **Caching.** Cache `Model.bin_probs`; the optimizer re-evaluates nearby thresholds constantly.
- **Warm-starting.** Initialize $N{+}1$ from the $N$-sensor solution — the optimum drifts slowly in $N$.

### Possible finding
Example 1 shows asymmetric beating symmetric at $N=2$; Theorem 5 says symmetric is optimal as $N \to \infty$. Numerically locating the crossover for the Gaussian model fills a gap the paper asserts at both endpoints but never explores in between.

### Provenance
`Policy` record carries: realized ensemble $(P^{\gamma^i})_{i=1}^N$, encoder, params, model, rate model, objective, optimizer seed. Once optimizing across $N$, across schemes, and across assumption sets, the output is a *table of optimized policies* and rows become unreconstructable without this.

Label results distinctly:
- **Global optimum within $\Gamma^N_{\text{TS}}$** — Assumption 1 holds, full alphabet; Theorem 1 guarantees the minimizer exists in this class, so the numerical result is genuinely global.
- **Lower bound** — assumptions dropped, or randomized kernels excluded.

---

## 7. Deferred

- **Scheme C.** Prof. Cohen's margin note: reallocate the whole budget to the single most-informative sensor. Variable-length / rate-reallocation with a further rate model. Will beat B. Scope as a third rate model later; keep out of the first build.
- **Counter encoder.** Sensor observes $T$ i.i.d. draws, maintains a counter, emits one action per block. Compute $P^{\gamma}(u \mid H_j)$ by **absorbing-Markov-chain analysis** over the $T$ slots, not by simulation. Sensors stay conditionally independent across $i$, so the core contract is untouched. **Normalization trap:** it consumes $T$ observation slots per action, so any comparison against vanilla must either give vanilla $T$ slots too, or report per-unit-*time* alongside per-unit-*rate*. Decide and state before generating counter figures.
- **Randomized kernels near thresholds.** Base needs no change — a randomized kernel still yields a $P^{\gamma^i}(\cdot \mid H_j)$ pair. Requires only that `Encoder` returns the pmf pair, never a hard partition.
- **Sparse three-region observation law.** `Atomic` backend already supports it if the tail-based informativeness story proves awkward for the $\delta$ narrative.
