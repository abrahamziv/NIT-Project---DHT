# 2026-08-18_194225_convergence_compare

- Model: GaussianShift, SNR = 0 dB (mu=1, sigma=1), p=0.5
- Fusion backend: method=tilted, G=1048576, nsig=14.0
- R (fixed per curve): [1, 2]
- N sweep: [10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
- Purpose: convergence check -- confirms the tilted backend reproduces the exact-path values already committed for N=10 and N=200 in the R-sweep runs (runs/Varying_R/R_sweep_N{10,200}_snrp0_2026-08-17/) before extending past where the exact path is feasible.
- Anchor comparison (all passed at rtol=1e-6, see console log):
  - Setup A, N=10, R=1: old exact = 2.210828
  - Setup A, N=10, R=2: old exact = 2.217701
  - Setup A, N=200, R=1: old exact = 18.492812
  - Setup A, N=200, R=2: old exact = 18.614834
  - Setup B, N=10, R=1: old exact = 2.439678
  - Setup B, N=10, R=2: old exact = 2.445430
  - Setup B, N=200, R=1: old exact = 21.306602
  - Setup B, N=200, R=2: old exact = 21.412225

Built from 2026-08-18_194442_convergence_R1_setupA_vs_N, 2026-08-18_194442_convergence_R1_setupB_vs_N, 2026-08-18_194442_convergence_R2_setupA_vs_N, 2026-08-18_194442_convergence_R2_setupB_vs_N.
