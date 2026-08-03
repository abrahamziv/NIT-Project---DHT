# 2026-08-03_132234_vanilla4_vs_silence4_compare_snrp5

- Model: GaussianShift, SNR = 5 dB (mu=1.7783, sigma=1.0), p=0.5
- Vanilla4: ThresholdEncoder(t=[0.2, 1.0, 5.0]), |U|=4
- Silence4: SilenceEncoder(t=[0.2, 1.0, 5.0], silent=[1, 2]), |U|=4, mean rate = 0.5300 bits/sensor
- N: 25 points, geomspace(10, 150)
- Notes: N capped at 150 (vs 3000 for the M=3 silence run) because exact J^N is O(N^3) for this 4-symbol identical-bank alphabet (compositions of N into 4 parts): measured N=300 alone takes ~47s, N=150 ~6s; a full 25-point sweep to N_MAX=150 takes ~20s. Built from 2026-08-03_132215_vanilla4_exponent_vs_N_snrp5 and 2026-08-03_132234_silence4_exponent_vs_N_snrp5.
- Total/normalized exponent panels plot a single curve: Vanilla4 and Silence4 share thresholds {0.2, 1, 5}, hence identical cell_probs and identical exact exponent -- silence only zeroes the rate cost, never the FC's information. See the Rate panel for the actual difference.
