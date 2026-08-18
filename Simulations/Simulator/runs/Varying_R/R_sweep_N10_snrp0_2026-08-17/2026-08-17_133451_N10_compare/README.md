# 2026-08-17_133451_N10_compare

- Model: GaussianShift, SNR = 0 dB (mu=1, sigma=1), p=0.5
- N (fixed): 10
- R sweep: [1, 2, 3, 4] bits/sensor (L = [2, 4, 8, 16])
- Setup A: ThresholdEncoder with L-1 thresholds evenly spaced in y over [-3*sigma, mu+3*sigma], symmetric about mu/2
- Setup B: SilenceEncoder, L thresholds (L/2 per side), silence bin [mu/2-delta, mu/2+delta] with **delta = 0.25** (fixed, not optimized)

## Rate used

| R (bits) | L | Setup | N | N_active | mean_rate (bits/sensor) |
|---|---|---|---|---|---|
| 1 | 2 | A (vanilla) | 10 | 10.0 | 1.0000 |
| 1 | 2 | B (silence) | 10 | 8.3 | 0.8253 |
| 2 | 4 | A (vanilla) | 10 | 10.0 | 2.0000 |
| 2 | 4 | B (silence) | 10 | 8.3 | 1.6507 |
| 3 | 8 | A (vanilla) | 10 | 10.0 | 3.0000 |
| 3 | 8 | B (silence) | 10 | 8.3 | 2.4760 |
| 4 | 16 | A (vanilla) | 10 | 10.0 | 4.0000 |
| 4 | 16 | B (silence) | 10 | 8.3 | 3.3013 |

Built from 2026-08-17_133839_N10_setupA_vs_R, 2026-08-17_133839_N10_setupB_vs_R.
