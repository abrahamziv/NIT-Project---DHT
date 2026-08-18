# 2026-08-17_133451_N200_compare

- Model: GaussianShift, SNR = 0 dB (mu=1, sigma=1), p=0.5
- N (fixed): 200
- R sweep: [1, 2] bits/sensor (L = [2, 4])
- Setup A: ThresholdEncoder with L-1 thresholds evenly spaced in y over [-3*sigma, mu+3*sigma], symmetric about mu/2
- Setup B: SilenceEncoder, L thresholds (L/2 per side), silence bin [mu/2-delta, mu/2+delta] with **delta = 0.25** (fixed, not optimized)

## Rate used

| R (bits) | L | Setup | N | N_active | mean_rate (bits/sensor) |
|---|---|---|---|---|---|
| 1 | 2 | A (vanilla) | 200 | 200.0 | 1.0000 |
| 1 | 2 | B (silence) | 200 | 165.1 | 0.8253 |
| 2 | 4 | A (vanilla) | 200 | 200.0 | 2.0000 |
| 2 | 4 | B (silence) | 200 | 165.1 | 1.6507 |

Built from 2026-08-17_140129_N200_setupA_vs_R, 2026-08-17_140129_N200_setupB_vs_R.
