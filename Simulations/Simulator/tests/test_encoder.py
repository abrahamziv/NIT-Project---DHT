import numpy as np

from ddms import EncoderBank, FusionCenter, GaussianShift, LRTEncoder, ThresholdEncoder


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


def test_from_fractions_counts_sum_to_n():
    a, b = LRTEncoder(1.0), LRTEncoder(2.0)
    bank = EncoderBank.from_fractions([(a, 1 / 3), (b, 2 / 3)], 10)
    assert len(bank) == 10
    counts = {id(a): 0, id(b): 0}
    for enc in bank.encoders:
        counts[id(enc)] += 1
    assert counts[id(a)] + counts[id(b)] == 10
    assert counts[id(a)] in (3, 4)
