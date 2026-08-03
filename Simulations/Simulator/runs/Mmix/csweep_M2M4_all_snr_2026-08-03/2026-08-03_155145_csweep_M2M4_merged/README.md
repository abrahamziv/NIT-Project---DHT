# 2026-08-03_155145_csweep_M2M4_merged

- Model: GaussianShift at SNR in {-5, 0, 5, 10} dB, p=0.5
- Encoders: ThresholdEncoder(t=[1.0]) |U|=2 (fraction c) vs ThresholdEncoder(t=[0.2, 1.0, 5.0]) |U|=4 (fraction 1-c); nested thresholds
- c: 0..1 in steps of 0.2; rate targets 60/120/240 bits, matched across c via N = round(R / (2 - c))
- Derived comparison, no data.json of its own; built from the per-SNR runs:
  - runs\Mmix\csweep_M2M4_snrm5_2026-08-03\2026-08-03_155058_csweep_M2M4_snrm5
  - runs\Mmix\csweep_M2M4_snrp0_2026-08-03\2026-08-03_155114_csweep_M2M4_snrp0
  - runs\Mmix\csweep_M2M4_snrp5_2026-08-03\2026-08-03_155129_csweep_M2M4_snrp5
  - runs\Mmix\csweep_M2M4_snrp10_2026-08-03\2026-08-03_155144_csweep_M2M4_snrp10

## Interior exact/chord ratio (pure Assumption 1(iii) effect)

  - SNR -5 dB: interior exact/chord 0.9974..1.0236; best per-bit exponent at c=1.00, R=60 bits (from 2026-08-03_155058_csweep_M2M4_snrm5)
  - SNR 0 dB: interior exact/chord 1.0019..1.0132; best per-bit exponent at c=1.00, R=60 bits (from 2026-08-03_155114_csweep_M2M4_snrp0)
  - SNR 5 dB: interior exact/chord 1.0002..1.0075; best per-bit exponent at c=1.00, R=60 bits (from 2026-08-03_155129_csweep_M2M4_snrp5)
  - SNR 10 dB: interior exact/chord 1.0008..1.0074; best per-bit exponent at c=1.00, R=60 bits (from 2026-08-03_155144_csweep_M2M4_snrp10)

- Notes: Chord reference: chord(c) = achieved_rate(c) * (w2*e2 + (1-w2)*e4) with e2, e4 the per-bit exponents of the homogeneous c=1 / c=0 endpoints at the same rate target and w2 the fraction of total bits carried by the M=2 group. exact/chord = 1 means breaking Assumption 1(iii) costs nothing beyond reallocating bits; deviation from 1 is the pure heterogeneity effect.
