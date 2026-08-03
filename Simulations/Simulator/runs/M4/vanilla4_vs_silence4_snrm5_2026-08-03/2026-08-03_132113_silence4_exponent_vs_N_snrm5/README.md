# 2026-08-03_132113_silence4_exponent_vs_N_snrm5

- Model: GaussianShift, SNR = -5 dB (mu=0.5623, sigma=1.0), p=0.5
- Encoder: SilenceEncoder(t=[0.2, 1.0, 5.0], silent=[1, 2]), |U|=4
- N: 25 points, geomspace(10, 150)
- Notes: N capped at 150 (vs 3000 for the M=3 silence run) because exact J^N is O(N^3) for this 4-symbol identical-bank alphabet (compositions of N into 4 parts): measured N=300 alone takes ~47s, N=150 ~6s; a full 25-point sweep to N_MAX=150 takes ~20s.
- Mean rate used at this SNR: 0.0058 bits/sensor
