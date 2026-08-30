# 2026-08-30_161625_snrp5_optparams_compare

- Model: GaussianShift, SNR = +5 dB (mu=1.77828, sigma=1), p=0.5
- Unquantized ceiling mu^2/8sigma^2 = 0.395285
- Fusion backend: method=tilted, G=1048576, nsig=14.0; exact_check_rel cross-checks against method=exact wherever C(N+M-1,M-1) <= 1e+07
- N (fixed): 3000; R sweep: [1, 2, 3, 4, 5] (L = [2, 4, 8, 16, 32])
- Policy: DDMS section 5 gamma^R_{Delta,delta} (RBitThresholdEncoder), with (Delta*, delta*) maximizing the closed-form bound C(R, Delta, delta) = -log M(0.5) per point (Setup A pins delta = 0)
- Figure plots ONLY the normalized error exponent, one line per setup. The bound, the Bahadur-Rao refinement and the finite-N re-optimization are recorded here and in data.json, not plotted.
- The bound is valid at every finite N here (p = 1/2 makes the Chernoff prefactor 1 at alpha = 1/2), so `normalized_exponent >= chernoff_C` is asserted at every point.
- Finite-N re-optimization (Setup B): max parameter drift 2.70e-04, max J^N_EE gain 1.19e-08.

## Results

| R | Setup | Delta* | delta* | J^N_EE (sim) | C (bound) | Bahadur-Rao | gap | P_silence | mean_rate | exact_check_rel |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | A (vanilla) | -- | 0 | 0.250247 | 0.248762 | 0.250300 | 0.001485 | 0.0000 | 1.0000 | 2.9e-15 |
| 1 | B (silence) | -- | 0.6280 | 0.320150 | 0.318607 | 0.320182 | 0.001544 | 0.3324 | 0.6676 | 1.5e-15 |
| 2 | A (vanilla) | 1.0014 | 0 | 0.349609 | 0.348021 | 0.349609 | 0.001588 | 0.0000 | 2.0000 | -- |
| 2 | B (silence) | 0.8772 | 0.3878 | 0.364788 | 0.363195 | 0.364788 | 0.001593 | 0.2073 | 1.5854 | -- |
| 3 | A (vanilla) | 0.5706 | 0 | 0.382553 | 0.380955 | 0.382554 | 0.001598 | 0.0000 | 3.0000 | -- |
| 3 | B (silence) | 0.5327 | 0.2256 | 0.385300 | 0.383701 | 0.385300 | 0.001599 | 0.1210 | 2.6369 | -- |
| 4 | A (vanilla) | 0.3219 | 0 | 0.392583 | 0.390983 | 0.392583 | 0.001601 | 0.0000 | 4.0000 | -- |
| 4 | B (silence) | 0.3109 | 0.1270 | 0.393045 | 0.391445 | 0.393045 | 0.001601 | 0.0682 | 3.7273 | -- |
| 5 | A (vanilla) | 0.1799 | 0 | 0.395600 | 0.393998 | 0.395600 | 0.001601 | 0.0000 | 5.0000 | -- |
| 5 | B (silence) | 0.1768 | 0.0700 | 0.395675 | 0.394073 | 0.395675 | 0.001601 | 0.0376 | 4.8118 | -- |

Built from 2026-08-30_162323_N3000_snrp5_optparams_setupA_vs_R, 2026-08-30_162323_N3000_snrp5_optparams_setupB_vs_R.
