import numpy as np

from ddms import EncoderBank, FusionCenter, GaussianShift, LRTEncoder, ThresholdEncoder
from ddms.sweep import sweep_over_N, sweep_over_R, sweep_over_rate, sweep_over_snr


def test_sweep_over_n_shapes_and_monotonicity():
    model = GaussianShift(1.0)
    enc = LRTEncoder(1.0)
    ns = [1, 2, 4, 8]
    res = sweep_over_N(model, lambda n: EncoderBank.identical(enc, n), ns)
    assert res["total_exponent"].shape == (4,)
    assert np.all(np.diff(res["total_exponent"]) > 0)
    np.testing.assert_array_equal(res["total_rate"], ns)


def test_sweep_records_fusion_backend():
    # Every sweep result names the backend that produced it, so no saved
    # data.json leaves the reader guessing.
    model = GaussianShift(1.0)
    make_bank = lambda n: EncoderBank.identical(LRTEncoder(1.0), n)
    assert sweep_over_N(model, make_bank, [2])["fusion"] == {"method": "exact"}
    res = sweep_over_N(
        model, make_bank, [2], fc=FusionCenter(p=model.p, method="tilted", G=2**16)
    )
    assert res["fusion"] == {"method": "tilted", "G": 2**16, "nsig": 14.0}


def test_sweep_over_snr_empty_raises():
    try:
        sweep_over_snr([], GaussianShift.from_snr_db, lambda model, n: None, N=6)
    except ValueError:
        return
    raise AssertionError("expected ValueError for empty snr_dbs")


def test_sweep_over_snr_monotone():
    snrs = [-5.0, 0.0, 5.0]
    res = sweep_over_snr(
        snrs,
        GaussianShift.from_snr_db,
        lambda model, n: EncoderBank.identical(LRTEncoder(1.0), n),
        N=6,
    )
    assert np.all(np.diff(res["total_exponent"]) > 0)
    assert np.all(np.diff(res["chernoff_information"]) > 0)


def test_sweep_over_rate():
    model = GaussianShift(1.0)

    def make_bank(bits):
        thresholds = np.exp(np.linspace(-1.0, 1.0, 2**bits - 1))
        return EncoderBank.identical(ThresholdEncoder(thresholds), 4)

    res = sweep_over_rate(model, make_bank, [1, 2, 3])
    np.testing.assert_array_equal(res["total_rate"], [4.0, 8.0, 12.0])
    assert np.all(np.diff(res["total_exponent"]) > 0)


def test_sweep_over_R():
    model = GaussianShift(1.0)
    enc = LRTEncoder(1.0)

    res = sweep_over_R(
        model, lambda n: EncoderBank.identical(enc, n), enc.rate(), [3.5, 5.0, 10.0]
    )
    np.testing.assert_array_equal(res["N"], [3, 5, 10])
    np.testing.assert_array_equal(res["rate_used"], [3.0, 5.0, 10.0])
    assert np.all(np.diff(res["total_exponent"]) > 0)


def test_sweep_over_R_caps_N():
    model = GaussianShift(1.0)
    enc = LRTEncoder(1.0)

    res = sweep_over_R(
        model, lambda n: EncoderBank.identical(enc, n), enc.rate(), [3.0, 10.0], N_max=5
    )
    np.testing.assert_array_equal(res["N"], [3, 5])
    np.testing.assert_array_equal(res["rate_used"], [3.0, 5.0])
