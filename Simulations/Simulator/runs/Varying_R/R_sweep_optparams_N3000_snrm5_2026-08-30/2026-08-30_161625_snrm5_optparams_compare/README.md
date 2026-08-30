# 2026-08-30_161625_snrm5_optparams_compare

- Model: GaussianShift, SNR = -5 dB (mu=0.562341, sigma=1), p=0.5
- Unquantized ceiling mu^2/8sigma^2 = 0.039528
- Fusion backend: method=tilted, G=1048576, nsig=14.0; exact_check_rel cross-checks against method=exact wherever C(N+M-1,M-1) <= 1e+07
- N (fixed): 3000; R sweep: [1, 2, 3, 4, 5] (L = [2, 4, 8, 16, 32])
- Policy: DDMS section 5 gamma^R_{Delta,delta} (RBitThresholdEncoder), with (Delta*, delta*) maximizing the closed-form bound C(R, Delta, delta) = -log M(0.5) per point (Setup A pins delta = 0)
- Figure plots ONLY the normalized error exponent, one line per setup. The bound, the Bahadur-Rao refinement and the finite-N re-optimization are recorded here and in data.json, not plotted.
- The bound is valid at every finite N here (p = 1/2 makes the Chernoff prefactor 1 at alpha = 1/2), so `normalized_exponent >= chernoff_C` is asserted at every point.
- Finite-N re-optimization (Setup B): max parameter drift 4.04e-04, max J^N_EE gain 2.57e-09.

## Results

| R | Setup | Delta* | delta* | J^N_EE (sim) | C (bound) | Bahadur-Rao | gap | P_silence | mean_rate | exact_check_rel |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | A (vanilla) | -- | 0 | 0.026275 | 0.025135 | 0.026279 | 0.001140 | 0.0000 | 1.0000 | 4.7e-14 |
| 1 | B (silence) | -- | 0.6136 | 0.033178 | 0.031996 | 0.033180 | 0.001182 | 0.4448 | 0.5552 | 2.2e-14 |
| 2 | A (vanilla) | 0.9836 | 0 | 0.036076 | 0.034876 | 0.036074 | 0.001199 | 0.0000 | 2.0000 | -- |
| 2 | B (silence) | 0.8636 | 0.3828 | 0.037570 | 0.036364 | 0.037568 | 0.001206 | 0.2871 | 1.4257 | -- |
| 3 | A (vanilla) | 0.5652 | 0 | 0.039325 | 0.038111 | 0.039323 | 0.001213 | 0.0000 | 3.0000 | -- |
| 3 | B (silence) | 0.5280 | 0.2240 | 0.039596 | 0.038382 | 0.039595 | 0.001215 | 0.1705 | 2.4885 | -- |
| 4 | A (vanilla) | 0.3201 | 0 | 0.040319 | 0.039102 | 0.040318 | 0.001217 | 0.0000 | 4.0000 | -- |
| 4 | B (silence) | 0.3092 | 0.1264 | 0.040365 | 0.039147 | 0.040363 | 0.001218 | 0.0967 | 3.6132 | -- |
| 5 | A (vanilla) | 0.1793 | 0 | 0.040619 | 0.039401 | 0.040618 | 0.001219 | 0.0000 | 5.0000 | -- |
| 5 | B (silence) | 0.1762 | 0.0698 | 0.040627 | 0.039408 | 0.040625 | 0.001219 | 0.0535 | 4.7324 | -- |

Built from 2026-08-30_161947_N3000_snrm5_optparams_setupA_vs_R, 2026-08-30_161947_N3000_snrm5_optparams_setupB_vs_R.
