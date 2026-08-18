import numpy as np

from ddms import (
    DiscreteLR,
    EncoderBank,
    FusionCenter,
    GaussianShift,
    LRTEncoder,
    ThresholdEncoder,
)


def example1():
    """Example 1 of the paper (arXiv:2509.21724, Section 5)."""
    model = DiscreteLR([4 / 5, 1 / 5, 0.0], [1 / 3, 1 / 3, 1 / 3])
    # l takes values 5/12 (y=0), 5/3 (y=1), inf (y=2)
    policy_a = LRTEncoder(1.0)  # u=0 iff y=0
    policy_b = LRTEncoder(2.0)  # u=0 iff y in {0, 1}
    return model, policy_a, policy_b


def test_n1_matches_direct_computation():
    model = GaussianShift(1.0, p=0.6)
    enc = LRTEncoder(1.0)
    fc = FusionCenter(p=0.6)
    q1, q2 = enc.cell_probs(model, 1), enc.cell_probs(model, 2)
    lam = np.log(q1) - np.log(q2)
    nt = np.log(0.4 / 0.6)
    direct = 0.6 * q1[lam < nt].sum() + 0.4 * q2[lam >= nt].sum()
    exact = np.exp(fc.log_error_prob(EncoderBank.identical(enc, 1), model))
    np.testing.assert_allclose(exact, direct, rtol=1e-12)


def test_example1_regression_anchor():
    # Exact values: asymmetric 19/90 = 0.211, both-B 2/9 = 0.222,
    # both-A 53/225 = 0.2356 (the paper reports 0.21, 0.22, 0.23).
    model, a, b = example1()
    fc = FusionCenter()
    j_aa = np.exp(fc.log_error_prob(EncoderBank.identical(a, 2), model))
    j_bb = np.exp(fc.log_error_prob(EncoderBank.identical(b, 2), model))
    j_ab = np.exp(fc.log_error_prob(EncoderBank([a, b]), model))
    np.testing.assert_allclose(j_aa, 53 / 225, rtol=1e-12)
    np.testing.assert_allclose(j_bb, 2 / 9, rtol=1e-12)
    np.testing.assert_allclose(j_ab, 19 / 90, rtol=1e-12)
    assert j_ab < j_bb < j_aa  # the asymmetric profile wins


def test_empirical_path_converges_to_exact():
    model = GaussianShift(0.8)
    bank = EncoderBank.identical(LRTEncoder(1.0), 4)
    fc = FusionCenter()
    exact = np.exp(fc.log_error_prob(bank, model))
    emp = fc.empirical_error_prob(bank, model, 200_000, np.random.default_rng(7))
    assert abs(emp - exact) < 0.005


def test_exponent_dominates_chernoff_bound():
    # At p = 1/2 the Chernoff bound (eqs. 37-40) upper-bounds J^N for
    # every N, so the achieved total exponent is at least the bound.
    model = GaussianShift(1.0)
    bank = EncoderBank.identical(LRTEncoder(1.0), 10)
    fc = FusionCenter()
    assert fc.total_exponent(bank, model) >= fc.chernoff_bound(bank, model) - 1e-9


def test_trivial_bank_error_is_min_prior():
    for p in (0.3, 0.5, 0.7):
        model = GaussianShift(1.0, p=p)
        bank = EncoderBank.identical(ThresholdEncoder([]), 5)
        fc = FusionCenter(p=p)
        j = np.exp(fc.log_error_prob(bank, model))
        np.testing.assert_allclose(j, min(p, 1 - p), rtol=1e-12)


def test_grouped_and_heterogeneous_paths_agree():
    model = GaussianShift(1.0)
    fc = FusionCenter()
    grouped = EncoderBank.identical(LRTEncoder(1.0), 6)
    hetero = EncoderBank([LRTEncoder(1.0) for _ in range(6)])
    np.testing.assert_allclose(
        fc.log_error_prob(grouped, model), fc.log_error_prob(hetero, model), rtol=1e-12
    )


def test_decide_and_exact_path_share_t():
    model = GaussianShift(1.0)
    bank = EncoderBank.identical(LRTEncoder(1.0), 4)
    P1 = bank.cell_probs_matrix(model, 1)
    P2 = bank.cell_probs_matrix(model, 2)
    u = np.array([[0, 0, 0, 0], [1, 1, 1, 1], [0, 1, 0, 1]])
    # t large: always decide H2, so J = p exactly
    fc_hi = FusionCenter(t=10.0)
    np.testing.assert_allclose(np.exp(fc_hi.log_error_prob(bank, model)), 0.5, rtol=1e-12)
    assert np.all(fc_hi.decide(u, P1, P2) == 2)
    # t small: always decide H1, so J = 1 - p exactly
    fc_lo = FusionCenter(t=-10.0)
    np.testing.assert_allclose(np.exp(fc_lo.log_error_prob(bank, model)), 0.5, rtol=1e-12)
    assert np.all(fc_lo.decide(u, P1, P2) == 1)


def test_n_max_guard_raises():
    model = GaussianShift(1.0)
    bank = EncoderBank([LRTEncoder(1.0 + 0.01 * i) for i in range(20)])
    fc = FusionCenter(N_max=16)
    try:
        fc.log_error_prob(bank, model)
    except ValueError:
        return
    raise AssertionError("expected ValueError from N_max guard")


def test_exact_composition_guard_raises():
    # A single identical group bypasses the N_max (distinct policies) guard;
    # the guard on C(N+M-1, M-1) itself must refuse instead of hanging.
    model = GaussianShift(1.0)
    enc = ThresholdEncoder(np.exp(np.linspace(-1.5, 1.5, 16)))  # M = 17
    bank = EncoderBank.identical(enc, 100)  # C(116, 16) ~ 1.9e19 count vectors
    try:
        FusionCenter().log_error_prob(bank, model)
    except ValueError as e:
        assert "tilted" in str(e)
        return
    raise AssertionError("expected ValueError from composition-count guard")


def test_decide_raises_on_padded_cell():
    model = GaussianShift(1.0)
    bank = EncoderBank([LRTEncoder(1.0), ThresholdEncoder([0.5, 1.0, 2.0])])
    P1 = bank.cell_probs_matrix(model, 1)
    P2 = bank.cell_probs_matrix(model, 2)
    try:
        FusionCenter().decide(np.array([3, 3]), P1, P2)
    except ValueError:
        return
    raise AssertionError("expected ValueError on padded cell")
