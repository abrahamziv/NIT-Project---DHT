# active_senders_vanilla_vs_silence_optdelta_snrp0

- Model: GaussianShift, SNR = 0 dB (mu=1.0000, sigma=1.0), p=0.5
- Vanilla: ThresholdEncoder(t=[1.0]), |U|=2
- Silence: SilenceEncoder(t=[0.5395346311328639, 1.8534491435708107], silent=[1]), |U|=3 -- thresholds are the DDMS.tex
  Eq. (opt-delta) optimum, found by numerically minimizing M_S(delta)
  (Eq. MS-of-delta) rather than solved from the boxed Eq. (opt-tau), whose
  stated identity tau2*^2 = P(u1|H2)/P(u2|H1) reduces to 1 = 1 by the
  symmetric construction (P(u1|H2) = P(u2|H1) = q_+ identically) and does
  not pin down delta. The correct identity is tau2*^2 = P(u2|H2)/P(u2|H1) =
  P(u1|H1)/P(u1|H2); worth a fix in the writeup.
  delta* = 0.617048, tau1* = 0.539535, tau2* = 1.853449
  (previously used fixed tau = [0.5, 2.0]; nearly identical at this SNR)
  p_silent = 0.4146
- **Normalization: both the x-axis and the y-axis divide by N_active
  (sensors that actually transmitted), not total N** -- Vanilla:
  N_active = N; Silence: N_active = N * (1 - p_silent). This differs from
  the DDMS.tex Figure~active-senders-chernoff, whose y-axis (J^N_EE) divides
  by total N while its x-axis is already active senders -- a mismatch this
  run corrects.
- Chernoff asymptotes: Vanilla C_V = 0.079282 (unchanged, active==N);
  Silence, per-total-N C_S = 0.101076, per-active
  C_S/(1-p_silent) = 0.172663.
- N: 25 points, geomspace(10, 3000)
- Comparison to fixed-threshold run (tau=[0.5,2.0]): per-total-N Chernoff
  info there was 0.100712 vs 0.101076 here -- the fixed
  thresholds were already close to optimal at this SNR.

## Run directories

- runs/M2/active_senders_vanilla_vs_silence_optdelta_snrp0_2026-08-17/2026-08-17_120010_vanilla_exponent_vs_activecount_snrp0
- runs/M2/active_senders_vanilla_vs_silence_optdelta_snrp0_2026-08-17/2026-08-17_120204_silence_optdelta_exponent_vs_activecount_snrp0
- runs/M2/active_senders_vanilla_vs_silence_optdelta_snrp0_2026-08-17/2026-08-17_120204_vanilla_exponent_vs_R_snrp0
- runs/M2/active_senders_vanilla_vs_silence_optdelta_snrp0_2026-08-17/2026-08-17_120538_silence_optdelta_exponent_vs_R_snrp0
