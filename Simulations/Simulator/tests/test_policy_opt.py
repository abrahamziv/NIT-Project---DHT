import numpy as np
import pytest

from ddms import EncoderBank, FusionCenter, GaussianShift, RBitThresholdEncoder
from ddms.policy_opt import (
    bahadur_rao_exponent,
    chernoff_C,
    chernoff_alpha_star,
    free_threshold_optimum,
    optimal_delta_R1,
    optimal_params,
)

MODEL = GaussianShift.from_snr_db(0.0)  # mu = sigma = 1, p = 1/2


def test_r1_delta_matches_closed_form():
    # The numeric optimizer must land on the appendix's closed-form root
    # e^{mu d / sigma^2} = sqrt((1 - q_-)/q_+)  (DDMS eq. opt-tau).
    d_closed = optimal_delta_R1(MODEL)
    np.testing.assert_allclose(d_closed, 0.6170482653718768, rtol=1e-12)
    _, d_opt, _ = optimal_params(MODEL, 1)
    np.testing.assert_allclose(d_opt, d_closed, atol=1e-6)


def test_chernoff_minimizer_at_half():
    # DDMS section 5: M(alpha) is symmetric about 0.5 and convex, so the
    # minimizer is exactly 0.5 for every R, with and without silence.
    for R in (1, 2, 3, 4, 5):
        D, d, _ = optimal_params(MODEL, R)
        enc = RBitThresholdEncoder(MODEL, R, D, d)
        assert abs(chernoff_alpha_star(MODEL, enc) - 0.5) < 1e-6
        enc_v = RBitThresholdEncoder(MODEL, R, max(D, 0.5), 0.0)
        assert abs(chernoff_alpha_star(MODEL, enc_v) - 0.5) < 1e-6


def test_chernoff_C_matches_fusion_center_bound():
    # Per-sensor C times N must equal the FC's unnormalized eq.-24 bound.
    fc = FusionCenter(p=MODEL.p)
    for R, D, d in ((1, 1.0, 0.6), (3, 0.5291, 0.2244)):
        bank = EncoderBank.identical(RBitThresholdEncoder(MODEL, R, D, d), 7)
        np.testing.assert_allclose(
            7 * chernoff_C(MODEL, R, D, d), fc.chernoff_bound(bank, MODEL), rtol=1e-9
        )


@pytest.mark.parametrize("R", [2, 3, 4])
def test_ladder_close_to_free_threshold_optimum(R):
    # The uniform-Delta ladder gives up at most 0.2% of C against fully free
    # thresholds -- the justification for the two-parameter parametrization.
    _, _, C_ladder = optimal_params(MODEL, R)
    _, C_free = free_threshold_optimum(MODEL, R)
    assert C_ladder <= C_free + 1e-9
    assert C_ladder >= C_free * (1 - 2e-3)


def test_silence_helps_and_R_converges_to_ceiling():
    ceiling = MODEL.chernoff_information()
    prev = 0.0
    for R in (1, 2, 3, 4, 5):
        _, _, C_A = optimal_params(MODEL, R, silence=False)
        _, _, C_B = optimal_params(MODEL, R)
        assert C_B >= C_A - 1e-12
        assert C_A > prev  # monotone in R
        assert C_A < ceiling  # quantization can't beat the unquantized bound
        prev = C_A
    assert ceiling - C_A < 1e-3  # R = 5 is already within 1e-3 of mu^2/8sigma^2


def test_bound_holds_at_finite_N():
    # At p = 1/2 the Chernoff prefactor 2 p^a (1-p)^{1-a} equals 1 at
    # a = 1/2, so J^N_EE >= C at EVERY N, not just asymptotically.
    for N, fc in ((10, FusionCenter(p=MODEL.p)), (1000, FusionCenter(p=MODEL.p, method="tilted"))):
        for R in (1, 2, 3, 4, 5):
            D, d, C = optimal_params(MODEL, R)
            bank = EncoderBank.identical(RBitThresholdEncoder(MODEL, R, D, d), N)
            if fc.method == "exact" and R > 2:
                continue  # composition count infeasible for the exact path
            assert fc.normalized_exponent(bank, MODEL) >= C - 1e-9


def test_bahadur_rao_matches_tilted_at_N1000():
    fc = FusionCenter(p=MODEL.p, method="tilted")
    for R, atol in ((1, 1e-3), (2, 1e-4), (3, 1e-4), (4, 1e-4), (5, 1e-4)):
        D, d, _ = optimal_params(MODEL, R)
        bank = EncoderBank.identical(RBitThresholdEncoder(MODEL, R, D, d), 1000)
        br = bahadur_rao_exponent(bank, MODEL)
        assert abs(br - fc.normalized_exponent(bank, MODEL)) < atol
