# 2026-08-30_160357_optparams_compare

- Model: GaussianShift, SNR = 0 dB (mu=1, sigma=1), p=0.5
- Fusion backend: method=tilted, G=1048576, nsig=14.0; exact_check_rel cross-checks against method=exact wherever C(N+M-1,M-1) <= 1e+07
- N (fixed): 3000; R sweep: [1, 2, 3, 4, 5] (L = [2, 4, 8, 16, 32])
- Policy: DDMS section 5 gamma^R_{Delta,delta} (RBitThresholdEncoder), with (Delta*, delta*) maximizing the closed-form bound C(R, Delta, delta) = -log M(0.5) per point (Setup A pins delta = 0)
- The bound is valid at every finite N here (p = 1/2 makes the Chernoff prefactor 1 at alpha = 1/2), so `normalized_exponent >= chernoff_C` is asserted at every point; `bound_gap` is the polynomial prefactor the bound discards, and the Bahadur-Rao column recovers it (`br_rel_error` in data.json).
- Finite-N re-optimization (Setup B): max parameter drift 7.11e-04, max J^N_EE gain 1.62e-08 -- the Chernoff-optimal design rule is already the N=3000 optimum to numerical precision.

## Results

| R | Setup | Delta* | delta* | C (bound) | J^N_EE (sim) | Bahadur-Rao | gap | P_silence | mean_rate | exact_check_rel |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | A (vanilla) | -- | 0 | 0.079282 | 0.080603 | 0.080620 | 0.001321 | 0.0000 | 1.0000 | 1.4e-14 |
| 1 | B (silence) | -- | 0.6170 | 0.101076 | 0.102444 | 0.102454 | 0.001368 | 0.4146 | 0.5854 | 9.2e-15 |
| 2 | A (vanilla) | 0.9878 | 0 | 0.110232 | 0.111624 | 0.111623 | 0.001392 | 0.0000 | 2.0000 | -- |
| 2 | B (silence) | 0.8668 | 0.3840 | 0.114958 | 0.116356 | 0.116356 | 0.001398 | 0.2655 | 1.4690 | -- |
| 3 | A (vanilla) | 0.5665 | 0 | 0.120506 | 0.121911 | 0.121911 | 0.001405 | 0.0000 | 3.0000 | -- |
| 3 | B (silence) | 0.5291 | 0.2244 | 0.121365 | 0.122771 | 0.122770 | 0.001406 | 0.1570 | 2.5289 | -- |
| 4 | A (vanilla) | 0.3205 | 0 | 0.123647 | 0.125056 | 0.125056 | 0.001409 | 0.0000 | 4.0000 | -- |
| 4 | B (silence) | 0.3096 | 0.1265 | 0.123792 | 0.125201 | 0.125201 | 0.001409 | 0.0889 | 3.6443 | -- |
| 5 | A (vanilla) | 0.1794 | 0 | 0.124595 | 0.126005 | 0.126004 | 0.001410 | 0.0000 | 5.0000 | -- |
| 5 | B (silence) | 0.1763 | 0.0699 | 0.124618 | 0.126028 | 0.126028 | 0.001410 | 0.0492 | 4.7541 | -- |

Built from 2026-08-30_160702_N3000_optparams_setupA_vs_R, 2026-08-30_160702_N3000_optparams_setupB_vs_R.
