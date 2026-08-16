# 2026-08-03_193835_vanilla4_vs_silence4_compare_snrp5

- Model: GaussianShift, SNR = 5 dB (mu=1.7783, sigma=1.0), p=0.5
- Vanilla4: ThresholdEncoder(t=[0.2, 1.0, 5.0]), |U|=4
- Silence4: SilenceEncoder(t=[0.2, 1.0, 5.0], silent=[1, 2]), |U|=4, mean rate = 0.5300 bits/sensor
- N: 25 points, geomspace(10, 150)
- R: 25 points, [20.0000, 300.0000] bits; N(R) range Vanilla4 [10, 150], Silence4 [37, 150]
- Notes: N capped at 150 (vs 3000 for the M=3 silence run) because exact J^N is O(N^3) for this 4-symbol identical-bank alphabet (compositions of N into 4 parts): measured N=300 alone takes ~47s, N=150 ~6s; a full 25-point sweep to N_MAX=150 takes ~20s. Built from 2026-08-03_193637_vanilla4_exponent_vs_N_snrp5, 2026-08-03_193656_silence4_exponent_vs_N_snrp5, 2026-08-03_193714_vanilla4_exponent_vs_R_snrp5, 2026-08-03_193835_silence4_exponent_vs_R_snrp5.
- Total/normalized exponent panels (vs N) plot a single curve: Vanilla4 and Silence4 share thresholds {0.2, 1, 5}, hence identical cell_probs and identical exact exponent at matched N -- silence only zeroes the rate cost, never the FC's information. See the Rate panel for the actual difference.
- The vs-R panel plots two separate curves: at matched R, Silence4's lower real per-sensor cost affords more N than Vanilla4, so the two diverge here even though they agree at matched N.
