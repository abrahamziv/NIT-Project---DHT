# 2026-08-03_193926_vanilla4_exponent_vs_R_snrp10

- Model: GaussianShift, SNR = 10 dB (mu=3.1623, sigma=1.0), p=0.5
- Encoder: ThresholdEncoder(t=[0.2, 1.0, 5.0]), |U|=4
- N: 25 points, geomspace(10, 150)
- Notes: N capped at 150 (vs 3000 for the M=3 silence run) because exact J^N is O(N^3) for this 4-symbol identical-bank alphabet (compositions of N into 4 parts): measured N=300 alone takes ~47s, N=150 ~6s; a full 25-point sweep to N_MAX=150 takes ~20s.
- R: 25 points, [20.0000, 300.0000] bits
- N(R) range: [10, 150]
