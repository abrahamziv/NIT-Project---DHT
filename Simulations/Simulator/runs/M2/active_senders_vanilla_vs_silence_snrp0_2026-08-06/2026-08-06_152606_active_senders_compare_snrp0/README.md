# active_senders_vanilla_vs_silence_snrp0

- Model: GaussianShift, SNR = 0 dB (mu=1.0000, sigma=1.0), p=0.5
- Vanilla: ThresholdEncoder(t=[1.0]), |U|=2
- Silence: SilenceEncoder(t=[0.5, 2.0], silent=[1]), |U|=3 (2 active + 1 free silent symbol),
  p_silent = 0.4602 (per-sensor probability of landing in the silent bucket)
- N: 25 points, geomspace(10, 3000)
- x-axis for panels 1-2 is *active senders*, not N: Vanilla active_count = N
  (every sensor transmits); Silence active_count = N * (1 - p_silent), the
  expected count of non-silent sensors -- this simulator has no per-sensor
  Monte Carlo realization, so "sensors that sent data" is this expectation.
- Panel 3 (R-sweep): N(R) = floor(R / per-sensor rate), matched budget across
  schemes; Vanilla per-sensor rate = 1.0000 bits, Silence mean
  rate = 0.5398 bits/sensor at this SNR. N capped at 3000 --
  exact J^N is O(N^2) for Silence's 3-symbol alphabet.
- Sensors that actually sent data, per R (silence's active_count = N(R) * (1 - p_silent)):

| R | N (vanilla) | N (silence) | active (silence) |
|---|---|---|---|
| 10.00 | 10 | 18 | 9.72 |
| 12.00 | 12 | 22 | 11.88 |
| 16.00 | 16 | 29 | 15.66 |
| 20.00 | 20 | 37 | 19.97 |
| 25.00 | 25 | 46 | 24.83 |
| 32.00 | 32 | 59 | 31.85 |
| 41.00 | 41 | 75 | 40.49 |
| 52.00 | 52 | 96 | 51.82 |
| 66.00 | 66 | 122 | 65.86 |
| 84.00 | 84 | 155 | 83.67 |
| 107.00 | 107 | 198 | 106.89 |
| 136.00 | 136 | 251 | 135.50 |
| 173.00 | 173 | 320 | 172.74 |
| 219.00 | 219 | 405 | 218.63 |
| 278.00 | 278 | 514 | 277.47 |
| 353.00 | 353 | 653 | 352.51 |
| 448.00 | 448 | 829 | 447.52 |
| 568.00 | 568 | 1052 | 567.90 |
| 720.00 | 720 | 1333 | 719.59 |
| 914.00 | 914 | 1693 | 913.93 |
| 1159.00 | 1159 | 2146 | 1158.47 |
| 1470.00 | 1470 | 2723 | 1469.95 |
| 1865.00 | 1865 | 3000 | 1619.48 |
| 2365.00 | 2365 | 3000 | 1619.48 |
| 3000.00 | 3000 | 3000 | 1619.48 |

## Run directories

- runs/M2/active_senders_vanilla_vs_silence_snrp0_2026-08-06/2026-08-06_152013_vanilla_exponent_vs_activecount_snrp0
- runs/M2/active_senders_vanilla_vs_silence_snrp0_2026-08-06/2026-08-06_152220_silence_exponent_vs_activecount_snrp0
- runs/M2/active_senders_vanilla_vs_silence_snrp0_2026-08-06/2026-08-06_152220_vanilla_exponent_vs_R_snrp0
- runs/M2/active_senders_vanilla_vs_silence_snrp0_2026-08-06/2026-08-06_152606_silence_exponent_vs_R_snrp0
