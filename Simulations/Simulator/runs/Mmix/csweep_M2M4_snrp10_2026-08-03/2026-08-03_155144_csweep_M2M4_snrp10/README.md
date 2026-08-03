# 2026-08-03_155144_csweep_M2M4_snrp10

- Model: GaussianShift, SNR = 10 dB (mu=3.1623, sigma=1.0), p=0.5
- Encoders: ThresholdEncoder(t=[1.0]) |U|=2 (fraction c) and ThresholdEncoder(t=[0.2, 1.0, 5.0]) |U|=4 (fraction 1-c); M=2 threshold nested in the M=4 set
- c: 6 points, 0..1 (largest-remainder split; achieved c recorded)
- Rate targets: 60, 120, 240 bits total, matched across c via N = round(R / (2 - c)); N spans 30..240
- uniform_alphabet(): True at c=0 and c=1 (inside Assumption 1(iii)), False at every interior c (outside)
- Interior exact/chord ratio spans 1.0008..1.0074
- Notes: Chord reference: chord(c) = achieved_rate(c) * (w2*e2 + (1-w2)*e4) with e2, e4 the per-bit exponents of the homogeneous c=1 / c=0 endpoints at the same rate target and w2 the fraction of total bits carried by the M=2 group. exact/chord = 1 means breaking Assumption 1(iii) costs nothing beyond reallocating bits; deviation from 1 is the pure heterogeneity effect.
- Timing: full c x rate grid at this SNR took 14s exact-path total.
