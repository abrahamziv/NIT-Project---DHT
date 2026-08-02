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
    return _collect(rows, {"N": np.asarray(list(Ns))})


def sweep_over_snr(snr_dbs, make_model, make_bank, N, fc=None):
    """make_model(snr_db) -> model, make_bank(model, N) -> EncoderBank."""
    rows = []
    for snr_db in snr_dbs:
        model = make_model(snr_db)
        point_fc = FusionCenter(p=model.p) if fc is None else fc
        row = _evaluate(point_fc, make_bank(model, N), model)
        row["chernoff_information"] = model.chernoff_information()
        rows.append(row)
    return _collect(rows, {"snr_db": np.asarray(list(snr_dbs))})


def sweep_over_rate(model, make_bank, rates, fc=None):
    """make_bank(rate) -> EncoderBank. Sweeps total rate at a fixed scheme."""
    fc = FusionCenter(p=model.p) if fc is None else fc
    rows = [_evaluate(fc, make_bank(r), model) for r in rates]
    return _collect(rows, {"rate": np.asarray(list(rates))})
