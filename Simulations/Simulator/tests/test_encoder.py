import numpy as np

from ddms import (
    EncoderBank,
    FusionCenter,
    GaussianShift,
    LRTEncoder,
    RBitThresholdEncoder,
    SilenceEncoder,
    ThresholdEncoder,
)


def test_cell_probs_are_a_pmf():
    model = GaussianShift(1.0)
    enc = ThresholdEncoder([0.5, 1.0, 2.0])
    for j in (1, 2):
        q = enc.cell_probs(model, j)
        assert q.shape == (enc.M,)
        assert np.all(q >= 0)
        np.testing.assert_allclose(q.sum(), 1.0, rtol=1e-12)


def test_encode_histogram_matches_cell_probs():
    model = GaussianShift(1.0)
    enc = ThresholdEncoder([0.5, 1.0, 2.0])
    rng = np.random.default_rng(4)
    for j in (1, 2):
        _, l = model.sample(j, 200_000, rng)
        freq = np.bincount(enc.encode(l), minlength=enc.M) / l.size
        np.testing.assert_allclose(freq, enc.cell_probs(model, j), atol=0.01)


def test_change_of_measure():
    # P(u | H2) = E_H1[ l * 1{gamma(l) = u} ]
    model = GaussianShift(1.0)
    enc = ThresholdEncoder([0.5, 1.0, 2.0])
    _, l = model.sample(1, 1_000_000, np.random.default_rng(5))
    u = enc.encode(l)
    q2 = enc.cell_probs(model, 2)
    for a in range(enc.M):
        np.testing.assert_allclose((l * (u == a)).mean(), q2[a], atol=0.01)


def test_trivial_encoder():
    model = GaussianShift(1.0)
    enc = ThresholdEncoder([])
    assert enc.M == 1
    for j in (1, 2):
        np.testing.assert_array_equal(enc.cell_probs(model, j), [1.0])
    fc = FusionCenter()
    assert abs(fc.chernoff_bound(EncoderBank.identical(enc, 4), model)) < 1e-9


def test_refinement_never_decreases_exponent():
    model = GaussianShift(1.0)
    fc = FusionCenter()
    coarse = EncoderBank.identical(ThresholdEncoder([1.0]), 3)
    fine = EncoderBank.identical(ThresholdEncoder([0.5, 1.0]), 3)
    assert fc.total_exponent(fine, model) >= fc.total_exponent(coarse, model) - 1e-12


def test_ragged_bank_padding_and_metadata():
    model = GaussianShift(1.0)
    bank = EncoderBank([LRTEncoder(1.0), ThresholdEncoder([0.5, 1.0, 2.0])])
    assert bank.M_list() == [2, 4]
    assert not bank.uniform_alphabet()
    np.testing.assert_allclose(bank.total_rate(), 3.0, rtol=1e-12)
    for j in (1, 2):
        P = bank.cell_probs_matrix(model, j)
        assert P.shape == (2, 4)
        np.testing.assert_array_equal(P[0, 2:], 0.0)


def test_silence_encoder_mean_rate_matches_direct_computation():
    model = GaussianShift.from_snr_db(0.0)
    enc = SilenceEncoder([0.5, 2.0], [1])
    q1 = enc.cell_probs(model, 1)
    q2 = enc.cell_probs(model, 2)
    expected = 1.0 - (model.p * q1[1] + (1 - model.p) * q2[1])
    np.testing.assert_allclose(enc.mean_rate(model), expected, rtol=1e-12)


def test_silence_encoder_wider_silence_uses_less_rate():
    model = GaussianShift.from_snr_db(0.0)
    narrow = SilenceEncoder([0.9, 1.1], [1])
    wide = SilenceEncoder([0.2, 5.0], [1])
    assert wide.mean_rate(model) < narrow.mean_rate(model)


def test_silence_encoder_M4_mean_rate_matches_direct_computation():
    model = GaussianShift.from_snr_db(0.0)
    enc = SilenceEncoder([0.2, 1, 5], [1, 2])
    q1 = enc.cell_probs(model, 1)
    q2 = enc.cell_probs(model, 2)
    p_silent = model.p * (q1[1] + q1[2]) + (1 - model.p) * (q2[1] + q2[2])
    expected = 1.0 * (1.0 - p_silent)  # 2 active bins -> 1 bit when active
    np.testing.assert_allclose(enc.mean_rate(model), expected, rtol=1e-12)


def test_silence_encoder_M4_refines_M3_at_equal_rate():
    # Same outer bounds (0.2, 5.0): M3 merges the middle into one silent bin,
    # M4 keeps the two halves as separate (still free) symbols. Both have
    # the same active-bit count and the same total silent mass, so mean
    # rate matches -- but M4 gives the FC strictly more resolution for it.
    model = GaussianShift.from_snr_db(0.0)
    merged = SilenceEncoder([0.2, 5.0], [1])
    split = SilenceEncoder([0.2, 1, 5], [1, 2])
    np.testing.assert_allclose(split.mean_rate(model), merged.mean_rate(model), rtol=1e-12)

    fc = FusionCenter()
    N = 20
    bank_merged = EncoderBank.identical(merged, N)
    bank_split = EncoderBank.identical(split, N)
    assert fc.total_exponent(bank_split, model) >= fc.total_exponent(bank_merged, model) - 1e-9


def test_from_fractions_counts_sum_to_n():
    a, b = LRTEncoder(1.0), LRTEncoder(2.0)
    bank = EncoderBank.from_fractions([(a, 1 / 3), (b, 2 / 3)], 10)
    assert len(bank) == 10
    counts = {id(a): 0, id(b): 0}
    for enc in bank.encoders:
        counts[id(enc)] += 1
    assert counts[id(a)] + counts[id(b)] == 10
    assert counts[id(a)] in (3, 4)


def test_rbit_encoder_cell_probs_are_a_pmf():
    model = GaussianShift.from_snr_db(0.0)
    for R, delta in ((1, 0.0), (1, 0.6), (3, 0.0), (3, 0.2)):
        enc = RBitThresholdEncoder(model, R, 0.5, delta)
        for j in (1, 2):
            q = enc.cell_probs(model, j)
            assert q.shape == (enc.M,)
            assert np.all(q >= 0)
            np.testing.assert_allclose(q.sum(), 1.0, rtol=1e-12)


def test_rbit_encoder_mirror_symmetry():
    # DDMS eq. mirror-probs: P(u^m | H1) = P(u^{L+1-m} | H2), silence in the
    # middle hypothesis-independent (eq. silence-prob).
    model = GaussianShift.from_snr_db(0.0)
    for R in (1, 2, 3):
        enc = RBitThresholdEncoder(model, R, 0.4, 0.3)
        q1 = enc.cell_probs(model, 1)
        q2 = enc.cell_probs(model, 2)
        np.testing.assert_allclose(q1, q2[::-1], rtol=1e-12)


def test_rbit_encoder_silence_prob_closed_form():
    # DDMS eq. silence-prob: P(u^S) = Q(s/2 - delta/sigma) - Q(s/2 + delta/sigma)
    from scipy.stats import norm

    model = GaussianShift.from_snr_db(0.0)
    s = model.mu / model.sigma
    delta = 0.3
    for R in (1, 2, 4):
        enc = RBitThresholdEncoder(model, R, 0.4, delta)
        expected = norm.sf(s / 2 - delta / model.sigma) - norm.sf(s / 2 + delta / model.sigma)
        for j in (1, 2):
            q = enc.cell_probs(model, j)
            np.testing.assert_allclose(q[enc.L // 2], expected, rtol=1e-12)


def test_rbit_encoder_delta_zero_is_vanilla():
    # delta = 0: t_{L/2} = t_{L/2+1} = mu/2 degenerate, so M = L, nothing is
    # silent, and the encoder equals a plain ThresholdEncoder on the same
    # thresholds. rate() = R in both branches.
    model = GaussianShift.from_snr_db(0.0)
    for R in (1, 2, 3):
        enc = RBitThresholdEncoder(model, R, 0.5, 0.0)
        assert enc.M == enc.L == 2**R
        assert enc.silent_indices == ()
        assert enc.rate() == float(R)
        assert enc.mean_rate(model) == float(R)
        plain = ThresholdEncoder(enc.thresholds)
        for j in (1, 2):
            np.testing.assert_allclose(
                enc.cell_probs(model, j), plain.cell_probs(model, j), rtol=1e-12
            )
    silent = RBitThresholdEncoder(model, 2, 0.5, 0.3)
    assert silent.M == silent.L + 1
    assert silent.rate() == 2.0
    assert silent.mean_rate(model) < 2.0


def test_rbit_encoder_histogram_matches_cell_probs():
    model = GaussianShift.from_snr_db(0.0)
    enc = RBitThresholdEncoder(model, 2, 0.6, 0.25)
    rng = np.random.default_rng(11)
    for j in (1, 2):
        _, l = model.sample(j, 200_000, rng)
        freq = np.bincount(enc.encode(l), minlength=enc.M) / l.size
        np.testing.assert_allclose(freq, enc.cell_probs(model, j), atol=0.01)
