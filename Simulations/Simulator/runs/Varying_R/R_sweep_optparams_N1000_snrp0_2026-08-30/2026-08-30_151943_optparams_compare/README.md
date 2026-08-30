# 2026-08-30_151943_optparams_compare

- Model: GaussianShift, SNR = 0 dB (mu=1, sigma=1), p=0.5
- Fusion backend: method=tilted, G=1048576, nsig=14.0; exact_check_rel cross-checks against method=exact wherever C(N+M-1,M-1) <= 1e+07
- N (fixed): 1000; R sweep: [1, 2, 3, 4, 5] (L = [2, 4, 8, 16, 32])
- Policy: DDMS section 5 gamma^R_{Delta,delta} (RBitThresholdEncoder), with (Delta*, delta*) maximizing the closed-form bound C(R, Delta, delta) = -log M(0.5) per point (Setup A pins delta = 0)
- The bound is valid at every finite N here (p = 1/2 makes the Chernoff prefactor 1 at alpha = 1/2), so `normalized_exponent >= chernoff_C` is asserted at every point; `bound_gap` is the polynomial prefactor the bound discards, and the Bahadur-Rao column recovers it (`br_rel_error` in data.json).
- Finite-N re-optimization (Setup B): max parameter drift 8.22e-04, max J^N_EE gain 3.83e-08 -- the Chernoff-optimal design rule is already the N=1000 optimum to numerical precision.

- `compare_vs_heuristic_baseline.pdf` overlays these runs on the committed
  2026-08-18 heuristic baseline (evenly spaced +-3 sigma thresholds,
  delta = 0.25 fixed): the optimization is worth ~31 nats of total exponent
  for Setup A at R = 2 and ~22 for Setup B at R = 2, shrinking as R grows.

## Results

| R | Setup | Delta* | delta* | C (bound) | J^N_EE (sim) | Bahadur-Rao | gap | P_silence | mean_rate | exact_check_rel |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | A (vanilla) | -- | 0 | 0.079282 | 0.082701 | 0.082747 | 0.003419 | 0.0000 | 1.0000 | 6.5e-15 |
| 1 | B (silence) | -- | 0.6170 | 0.101076 | 0.104633 | 0.104659 | 0.003557 | 0.4146 | 0.5854 | 9.5e-16 |
| 2 | A (vanilla) | 0.9878 | 0 | 0.110232 | 0.113860 | 0.113856 | 0.003628 | 0.0000 | 2.0000 | -- |
| 2 | B (silence) | 0.8668 | 0.3840 | 0.114958 | 0.118606 | 0.118602 | 0.003647 | 0.2655 | 1.4690 | -- |
| 3 | A (vanilla) | 0.5665 | 0 | 0.120506 | 0.124174 | 0.124171 | 0.003668 | 0.0000 | 3.0000 | -- |
| 3 | B (silence) | 0.5291 | 0.2244 | 0.121365 | 0.125036 | 0.125032 | 0.003671 | 0.1570 | 2.5289 | -- |
| 4 | A (vanilla) | 0.3205 | 0 | 0.123647 | 0.127327 | 0.127323 | 0.003679 | 0.0000 | 4.0000 | -- |
| 4 | B (silence) | 0.3096 | 0.1265 | 0.123792 | 0.127472 | 0.127468 | 0.003680 | 0.0889 | 3.6443 | -- |
| 5 | A (vanilla) | 0.1794 | 0 | 0.124595 | 0.128277 | 0.128273 | 0.003682 | 0.0000 | 5.0000 | -- |
| 5 | B (silence) | 0.1763 | 0.0699 | 0.124618 | 0.128301 | 0.128297 | 0.003682 | 0.0492 | 4.7541 | -- |

Built from 2026-08-30_152218_N1000_optparams_setupA_vs_R, 2026-08-30_152218_N1000_optparams_setupB_vs_R.
