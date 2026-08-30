"""Parameter sweeps. Plain functions returning dicts of numpy arrays.

The nontrivial logic lives here: building the right bank at each point.
Comparisons across schemes should be made at matched total_rate, not
matched N.
"""

import numpy as np

from .fusion import FusionCenter


def _evaluate(fc, bank, model):
    log_j = fc.log_error_prob(bank, model)
    return {
        "log_error": log_j,
        "total_exponent": -log_j,
        "normalized_exponent": -log_j / len(bank),
        "chernoff_bound": fc.chernoff_bound(bank, model),
        "total_rate": bank.total_rate(),
        "uniform_alphabet": bank.uniform_alphabet(),
    }


def _collect(rows, extra):
    out = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    out.update(extra)
    return out


def sweep_over_N(model, make_bank, Ns, fc=None):
    """make_bank(N) -> EncoderBank. Sweeps the number of sensors."""
    fc = FusionCenter(p=model.p) if fc is None else fc
    rows = [_evaluate(fc, make_bank(N), model) for N in Ns]
    return _collect(rows, {"N": np.asarray(list(Ns)), "fusion": fc.meta()})


def sweep_over_snr(snr_dbs, make_model, make_bank, N, fc=None):
    """make_model(snr_db) -> model, make_bank(model, N) -> EncoderBank."""
    rows = []
    for snr_db in snr_dbs:
        model = make_model(snr_db)
        point_fc = FusionCenter(p=model.p) if fc is None else fc
        row = _evaluate(point_fc, make_bank(model, N), model)
        row["chernoff_information"] = model.chernoff_information()
        rows.append(row)
    return _collect(rows, {"snr_db": np.asarray(list(snr_dbs)), "fusion": point_fc.meta()})


def sweep_over_R(model, make_bank, per_sensor_rate, Rs, fc=None, N_max=None):
    """make_bank(N) -> EncoderBank, as in sweep_over_N.

    Sweeps a grid of total-rate budgets R. At each R, uses the maximum
    affordable N = floor(R / per_sensor_rate) sensors, so schemes are
    compared at matched rate rather than matched N. per_sensor_rate is the
    scheme's real per-sensor cost: rate() for fixed-length encoders,
    mean_rate(model) for anything with a free/zero-cost symbol.

    N_max, if given, caps N independent of the budget -- a lower-rate scheme
    can otherwise demand far more sensors than a fusion path capped for
    compute reasons (see each fig script's own N_max) can afford to evaluate.
    N and rate_used both reflect the capped N, so they stay self-consistent.
    """
    fc = FusionCenter(p=model.p) if fc is None else fc
    Ns = np.array([max(1, int(R // per_sensor_rate)) for R in Rs])
    if N_max is not None:
        Ns = np.minimum(Ns, N_max)
    rows = [_evaluate(fc, make_bank(int(N)), model) for N in Ns]
    extra = {
        "R": np.asarray(list(Rs)),
        "N": Ns,
        "rate_used": Ns * per_sensor_rate,
        "fusion": fc.meta(),
    }
    return _collect(rows, extra)


def sweep_over_rate(model, make_bank, rates, fc=None):
    """make_bank(rate) -> EncoderBank. Sweeps total rate at a fixed scheme."""
    fc = FusionCenter(p=model.p) if fc is None else fc
    rows = [_evaluate(fc, make_bank(r), model) for r in rates]
    return _collect(rows, {"rate": np.asarray(list(rates)), "fusion": fc.meta()})
