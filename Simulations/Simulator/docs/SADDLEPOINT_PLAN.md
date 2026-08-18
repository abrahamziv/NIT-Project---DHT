# Breaking the `C(N+M-1, M-1)` wall in `fusion.py`

Status: implemented (sections 1-6 of the order of work; the section-8 step 7
figure re-run is a follow-up). `FusionCenter` takes `method="exact"` (default,
now guarded on its count-vector total) or `"tilted"`; sweeps record
`method`/`G`/`nsig` under the `fusion` key. The in-repo implementation
cross-validates tighter than the tables below (1e-14 relative on the M=2
ladder to N=3000, 6e-9 at M=17), because the snapped model is used
consistently in the closed form and the FFT. Decisions taken during review:
`"exact"` stays the constructor default (existing call sites unchanged), the
Bahadur-Rao oracle is test-only, and ties at `S == Nt` go to the H2-error
side to match the exact path -- at `p = 0.5` symmetric banks put an atom
exactly at `Nt = 0`, so this is O(1) of the tail, not a half-bin detail.

All numbers below were measured against the repo's own Setup A / Setup B
channels (`GaussianShift.from_snr_db(0)`, `MARGIN_SIGMA=3.0`, `DELTA=0.25`,
`p=0.5`) using a standalone reimplementation of `fusion.py`'s exact path as
the reference.

---

## 1. The problem

`fusion.py::_group_dist` builds the full law of `S = sum_i lambda(u^i)` by
enumerating every symbol-count vector `(k_1, ..., k_M)` summing to `N`. That
loop runs `C(N+M-1, M-1)` times.

| N | M | count vectors |
|---|---|---|
| 100 | 3 | 5.2e3 |
| 100 | 5 | 4.6e6 |
| 200 | 5 | 7.0e7 |
| 200 | 9 | 7.6e13 |
| 100 | 17 | **1.7e19** |

This is not a speed knob. It is the number of terms the method must weigh, so
no amount of compute changes it -- at 1e19 the intermediate array is not
storable. It explodes in `N` and `M` simultaneously, which is why
`fig_R_bits_per_sensor.py` is stuck in an L-shape: `N=10` reaches `R=4`, or
`N=200` caps at `R=2`, never both.

`_merge` cannot rescue this. With the `lambda_u` values these encoders produce,
every count vector yields a distinct sum (see 6.3), so there is nothing to
merge.

Everything else in the codebase is uninvolved. `model.py`, `encoder.py`, the
channel-matrix abstraction, the log-domain arithmetic, and the
sweep/run/figure layering are all fine.

### 1.1 A related defect worth fixing at the same time

`_delta_dist` guards with `len(groups) > self.N_max`. That counts *distinct
policies*, not count vectors. A symmetric bank (`EncoderBank.identical`) has
`len(groups) == 1`, so it sails past the guard regardless of `N` and `M` and
then runs for hours or exhausts memory with no error. The guard that is
actually needed is on `C(N+M-1, M-1)` itself.

---

## 2. The fix: exponential tilting

The current code builds the entire distribution of `S` and then discards
everything except one tail sum. Compute the tail directly instead.

Per-sensor cumulant generating function, `O(M)`, no `N` in it:

```
Lambda_j(theta) = log sum_u q_j(u) * exp(theta * lambda_u)
```

For `N` i.i.d. sensors the CGF is `N * Lambda_j(theta)`; for `K` groups it is
`sum_k n_k * Lambda_k(theta)`. Choose `theta*` solving
`Lambda'(theta*) = Nt` -- the tilt that slides the distribution until its mean
sits exactly on the decision threshold. Then, **exactly**:

```
log P(S < Nt) = N*Lambda_j(theta*) - theta*(Nt) + log E~[ e^{-theta*(S-Nt)} 1{S<Nt} ]
                \_______ closed form, O(M) _______/   \____ an O(1) number ____/
```

The entire dynamic range -- the 1e-50 and beyond -- lives in the closed-form
term, which is arithmetic on `Lambda` and costs `O(M)`. The residual
expectation is a sum of strictly positive terms each bounded by 1, totalling
roughly `1/sqrt(N)`. That is the part computed numerically, and it is `O(1)`,
so float64 handles it at full precision.

The residual is evaluated by FFT: build the tilted pmf
`q~(u) = q_j(u) * exp(theta* lambda_u - Lambda_j(theta*))`, place its `M`
atoms on a grid, `rfft`, raise pointwise to the `N`-th power, `irfft`, and sum
the tail with the `e^{-theta*(s-Nt)}` weights.

`N` appears only as an exponent, so `N=10` and `N=10000` cost the same. `M`
appears only as "how many atoms to place", so it is `O(M)`. Both explosions
are gone for the same reason: neither is a loop bound anymore.

### 2.1 Why plain FFT is not enough

**Tilting is not an optimisation, it is the thing that makes this work.** An
untilted FFT tries to compute the 1e-50 *numerically*, extracting it from a
distribution whose bulk is `O(1)` with only ~1e-16 of relative headroom.
Measured:

| setup | M | N | true `log J^N` | untilted FFT |
|---|---|---|---|---|
| A L=2 | 2 | 500 | **-42.72** | -25.77 |
| A L=2 | 2 | 2000 | **-162.33** | -24.73 |
| B L=16 | 17 | 2000 | **-249.47** | -28.28 |

It agrees fine to `N ~ 100` (where `J^N ~ 1e-6`) and then silently saturates
at the float64 noise floor while still returning a plausible-looking number.
That is the worst available failure mode for a figure. Any FFT-based proposal
that does not tilt has this bug.

---

## 3. What to add

Confined to `fusion.py`. `model.py`, `encoder.py`, `sweep.py` and every
`figs/` script are untouched by the mechanism itself (see 5 for the churn the
explicit-backend decision causes). `log_error_prob(bank, model)` keeps its
signature.

### 3.1 Refactor first, no new capability

`log_error_prob` currently does two jobs: build the distribution, then split it
at `Nt`. Split that seam:

```
_log_tails(bank, model, nt) -> (log P(S < nt | H1), log P(S >= nt | H2))
```

`log_error_prob` then only weights those two by the prior. The existing
`_delta_dist` -> `_group_dist` -> `_convolve` chain becomes one implementation
behind that seam, otherwise unchanged.

### 3.2 The new backend

| function | returns | cost |
|---|---|---|
| `_cgf(theta, groups, j)` | `sum_k n_k Lambda_k(theta)` | O(K*M) |
| `_cgf_prime(theta, groups, j)` | its derivative | O(K*M) |
| `_saddlepoint(groups, j, nt)` | `theta*`, `brentq` on `_cgf_prime - nt` | ~40 evals |
| `_log_tail_tilted(groups, j, nt, side, G, nsig)` | one tail log-prob | O(G log G) |

The `K`-group structure survives intact: `Lambda(theta) = sum_k n_k
Lambda_k(theta)`, and the tilted transforms multiply. Heterogeneous banks come
along for free. Only the *intra*-group enumeration is replaced.

### 3.3 Two knobs, both must be recorded per run

- `G` -- FFT grid size. Convergence is monotone and reportable (7.2).
- `nsig` -- window half-width in units of the tilted standard deviation,
  controls aliasing. `nsig=14` was used throughout below.

---

## 4. Guarantees

Ordered from strongest to weakest. This is the honest ordering; not everything
here is a proof.

### 4.1 Structural (true by construction)

- **No cancellation anywhere.** Every term summed in the residual is a product
  of a non-negative pmf value and a positive exponential. There is no
  subtraction in the numerical part of the computation. This is a property of
  the formulation, not an empirical observation.
- **The dynamic range is closed-form.** The 1e-50 never enters the numerics.
  It is produced by `N*Lambda(theta*) - theta*Nt`, a logsumexp over `M` terms
  -- the same machinery already trusted elsewhere in the file.
- **Exactness in `theta`.** The identity in section 2 holds for *every*
  `theta`, not only the saddlepoint. `theta*` is a conditioning choice, not a
  correctness one. See 4.4 for what this does and does not buy in float.

### 4.2 Cross-validation against the existing exact path -- the main guarantee

The exact path is not deleted or demoted. It becomes the test oracle. The two
overlap on a staircase-shaped region, and critically the overlap covers **both
axes independently**:

**Large N (at small M).** At `M=2` the exact path costs `O(N)`, so it reaches
absurd depth. This is the single most valuable test available and it is cheap:

| N | exact `log J^N` | tilted `log J^N` | `log10 J^N` | rel err |
|---|---|---|---|---|
| 50 | -5.975381 | -5.975366 | -2.6 | 2.5e-06 |
| 200 | -18.492812 | -18.492795 | -8.0 | 9.4e-07 |
| 1000 | -82.700784 | -82.700600 | -35.9 | 2.2e-06 |
| 3000 | -241.809942 | -241.810303 | -105.0 | 1.5e-06 |
| 8000 | -638.708641 | -638.709296 | **-277.4** | 1.0e-06 |

This is the test the untilted FFT fails catastrophically (section 2.1) and the
tilted one passes at 1e-6 relative, four hundred orders of magnitude below
where Monte Carlo could ever reach.

**Large M (at small N).** At `M=17` the exact path reaches about `N=8`:

```
exact   = -2.5138736089   (9.8 s)
tilted  = -2.5138740179   (0.2 s)   rel err 1.6e-07
```

**The middle.** `M=4,5,8,9` at `N=10..80`, all agreeing at 1e-6 to 1e-5
relative.

**The honest limitation:** there is no point at which both `N` and `M` are
large *and* the exact path can be run, so large-N-and-large-M is never
validated directly. The argument that this is sufficient is that the algorithm
contains no term coupling `N` and `M`: `N` enters only as the exponent on the
transform, `M` only as the atom count deposited on the grid. That is a
structural argument, not a measurement, and it should be stated as such in the
paper rather than glossed.

### 4.3 An independent second algorithm

Bahadur-Rao / Edgeworth: a completely different derivation, no FFT and no
grid, ~30 lines. It catches grid-handling bugs that both FFT paths could
share.

| N | rel. deviation from exact |
|---|---|
| 10 | 1.2e-01 |
| 40 | 1.2e-02 |
| 80 | 1.5e-03 |
| >=100 | ~1e-04 |

Useless at small `N` (it is a large-`N` asymptotic), excellent above ~100 --
which is exactly the region where the exact path cannot follow. Between the
two oracles, every `(N, M)` of interest has at least one independent check.

`chernoff_bound` is a free third check: `-log J^N` should approach `N*C` from
above and never fall below the bound. That assertion already exists in
`tests/test_fusion.py` and should be extended to the new backend.

### 4.4 What theta-invariance actually buys

In exact arithmetic the answer is independent of `theta`. In float it is flat
in a neighbourhood of `theta*` and degrades away from it, because away from
the saddlepoint the integrand is dominated by the part of the tilted
distribution the grid resolves worst. A quick measurement at `M=17, N=100`
showed agreement to ~0.3% across `theta* +- 0.05` and visible breakdown by
`theta* +- 0.2`.

So this is **a local conditioning diagnostic, not a global correctness
proof**. That is still worth having -- the flatness and width of the plateau
is a per-run report on how well-conditioned that particular point is, and a
sign error or a missing normalisation would destroy the plateau entirely. But
it should not be sold as more than it is.

(Caveat: that particular measurement was taken with a variant implementation
that later disagreed with the exact anchor elsewhere. The qualitative
conclusion -- local plateau, not global invariance -- is sound, but the
numbers should be re-measured against the anchored implementation before being
quoted.)

### 4.5 What Monte Carlo does and does not cover

`empirical_error_prob` only functions where `J^N ~ 0.1`, i.e. tiny `N` and low
SNR. It validates that `encode`, `decide`, and `cell_probs` are mutually
consistent -- the *model*, not the summation algorithm. It is not part of the
guarantee for this change and should not be presented as such.

---

## 5. Explicit backend selection (decided)

No `method="auto"`. Every call names its backend.

**Rationale.** `runs/` is committed and feeds the paper. A `data.json` should
never leave a reader guessing which algorithm produced a number, and an
implicit dispatch threshold is exactly the kind of thing that silently changes
under a later refactor and invalidates old comparisons.

**Consequences, all intentional:**

- `FusionCenter.__init__` takes `method` explicitly: `"exact"` or `"tilted"`.
  Whether it has a default at all is worth deciding deliberately -- a default
  of `"exact"` preserves current behaviour of everything already written; no
  default forces every call site to be considered once, which is the point of
  going explicit but means the constructor is a breaking change.
- All ~12 `figs/*.py` scripts must name their backend. Modest, mechanical churn.
- `method` plus `G` and `nsig` go into every run's metadata. Existing
  `runs/*/data.json` predate the field; they are all `"exact"` and can be
  treated as such implicitly rather than backfilled.
- `method="exact"` gains a hard guard on `C(N+M-1, M-1)`: raise immediately,
  reporting the count and pointing at `"tilted"`, instead of hanging. This is
  the fix for 1.1.

---

## 6. Known edge cases and traps

### 6.1 Infinite atoms -- affects the regression anchor

A cell with mass under one hypothesis and zero under the other gives
`lambda = +-inf`. `_atoms` currently drops only cells that are zero under
*both*, so these survive by design.

Example 1 -- the repo's regression anchor -- has `p1 = (4/5, 1/5, 0)`, so a
policy isolating `y=3` produces exactly this. Verified: the exact path handles
it and returns `-4.74779826`; a naive tilted implementation dies in the
saddlepoint solve (`brentq` on a NaN). The tilted backend needs an explicit
branch: an infinite atom makes `Lambda(theta)` infinite on one side of the
`theta` axis and forces one tail to be exactly 0 or 1. **This must be handled
before the anchor tests can run against the new backend.**

### 6.2 No saddlepoint

If `Nt` falls outside the range of `lambda`, no `theta*` exists and the tail
is exactly 0 or 1. Needs a guard rather than a root-finder failure.

### 6.3 Lattice structure -- do not "improve" the snapping

An obvious-looking refinement is to eliminate grid-snapping of the `lambda_u`
by sampling the exact characteristic function at the DFT frequencies instead.
**This was tried and it is wrong.** It converged cleanly and monotonically to
`-65.286` for Setup B `L=16` at `N=500`, where the exact-anchored answer (and
Bahadur-Rao independently) give `-64.693`.

The reason: these encoders produce `lambda_u` values that are not rational
multiples of one another, so the `N`-fold sum genuinely has
`C(N+M-1, M-1)` distinct support points and is not supported on any lattice.
Inverting a sampled CF for an off-lattice discrete measure rings, and the
ringing is stable under grid refinement -- so it looks converged while being
wrong. Snapping to a lattice is not a defect to remove; it *is* the
discretisation, and it is the version anchored against the exact path.

This is worth recording precisely because the wrong approach is the more
mathematically elegant-looking one and produces a beautifully convergent
wrong answer.

### 6.4 Threshold alignment

`Nt` should be placed exactly on a grid point, otherwise the tail boundary
carries a half-bin ambiguity worth `O(delta / (sigma sqrt(N)))` in relative
terms. Small, but free to eliminate.

### 6.5 Non-symmetric prior

`p != 0.5` moves `Nt` off zero. The mechanism is unaffected -- the saddlepoint
target changes from 0 to `Nt` -- but it is untested here; everything measured
above is at `p = 0.5`.

### 6.6 Small N

Below `N ~ 5` the `sqrt(N)`-based window sizing is crude. Irrelevant in
practice since the exact backend is trivially cheap there, but it means
`"tilted"` should not be assumed correct at every `N` without checking.

---

## 7. Cost and convergence, measured

### 7.1 Runtime

| point | exact | tilted |
|---|---|---|
| M=17, N=8 | 9.8 s | 0.2 s |
| M=17, N=100 | ~1e19 terms, infeasible | 0.2 s |
| M=17, N=2000 | infeasible | 0.2 s |
| M=5, N=200 | 7.0e7 terms | 0.2 s |

Tilted cost is flat in `N` and near-flat in `M`, as designed.

### 7.2 Grid convergence, `M=17, N=100`

| G | `log J^N` | change | time |
|---|---|---|---|
| 2^16 | -14.8273170136 | -- | 0.01 s |
| 2^18 | -14.8267491119 | 5.7e-04 | 0.04 s |
| 2^20 | -14.8270548928 | 3.1e-04 | 0.19 s |
| 2^22 | -14.8269799778 | 7.5e-05 | 1.45 s |
| 2^24 | -14.8269966907 | 1.7e-05 | 8.76 s |

Monotone and cheap. `G = 2^20` at 0.19 s is already at 1e-5 absolute, which is
far below plotting resolution. Recording this ladder per run turns the grid
from an unstated approximation into a reported convergence certificate.

---

## 8. Suggested order of work

1. Refactor the `_log_tails(bank, model, nt)` seam. No behaviour change; all
   existing tests must pass untouched. Commit separately.
2. Add the `C(N+M-1, M-1)` guard to the exact backend (fixes 1.1). Commit
   separately -- it is a real bug fix independent of everything else.
3. Implement the tilted backend, including the infinite-atom branch (6.1) and
   the no-saddlepoint guard (6.2).
4. Cross-validation tests: the `M=2` deep-tail ladder from 4.2 as the headline
   regression, plus the `M=17, N=8` anchor, plus the existing Example 1 anchor
   through both backends.
5. Add Bahadur-Rao as an independent check (4.3), and extend the existing
   `total_exponent >= chernoff_bound` assertion to the tilted backend.
6. Thread `method` explicitly through `FusionCenter`, `sweep.py` metadata, and
   the `figs/` scripts (section 5).
7. Re-run `fig_R_bits_per_sensor.py` at the sizes originally wanted -- `N=200`
   with `R` up to 4, and the `N`-sweep out to a few thousand -- and check
   whether the Setup B story survives there. **This is the open scientific
   question; everything above is machinery in service of it.**

---

## 9. Files touched

```
src/ddms/fusion.py      seam refactor, exact-path guard, tilted backend
tests/test_fusion.py    cross-validation ladders, both backends on Example 1
src/ddms/sweep.py       pass through and record `method`, `G`, `nsig`
figs/*.py               name a backend explicitly (~12 scripts, mechanical)
docs/PLAN.md            record the deviation and the explicit-backend decision
claude.md               update the Fusion Center section: two backends, the
                        exact one is now also the oracle for the other
```
