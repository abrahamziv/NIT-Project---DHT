# 2026-08-18_194225_N1000_compare

- Model: GaussianShift, SNR = 0 dB (mu=1, sigma=1), p=0.5
- Fusion backend: method=tilted, G=1048576, nsig=14.0; exact_check_rel columns cross-check against method=exact wherever C(N+M-1,M-1) <= 1e+07
- N (fixed): 1000
- R sweep: [1, 2, 3, 4, 5] bits/sensor (L = [2, 4, 8, 16, 32])
- Setup A: ThresholdEncoder with L-1 thresholds evenly spaced in y over [-3*sigma, mu+3*sigma], symmetric about mu/2
- Setup B: SilenceEncoder, L thresholds (L/2 per side), silence bin [mu/2-delta, mu/2+delta] with **delta = 0.25** (fixed, not optimized)

## Rate used

| R | Setup | N | N_active | mean_rate (bits/sensor) | exact_check_rel |
|---|---|---|---|---|---|
| 1 | A (vanilla) | 1000 | 1000.0 | 1.0000 | 6.5e-15 |
| 1 | B (silence) | 1000 | 825.3 | 0.8253 | 2.1e-15 |
| 2 | A (vanilla) | 1000 | 1000.0 | 2.0000 | -- |
| 2 | B (silence) | 1000 | 825.3 | 1.6507 | -- |
| 3 | A (vanilla) | 1000 | 1000.0 | 3.0000 | -- |
| 3 | B (silence) | 1000 | 825.3 | 2.4760 | -- |
| 4 | A (vanilla) | 1000 | 1000.0 | 4.0000 | -- |
| 4 | B (silence) | 1000 | 825.3 | 3.3013 | -- |
| 5 | A (vanilla) | 1000 | 1000.0 | 5.0000 | -- |
| 5 | B (silence) | 1000 | 825.3 | 4.1267 | -- |

Built from 2026-08-18_194238_N1000_setupA_vs_R, 2026-08-18_194238_N1000_setupB_vs_R.
