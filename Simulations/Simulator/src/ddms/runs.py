"""Run persistence: each sweep is saved to a fresh timestamped directory.

A run directory holds data.json (the raw sweep arrays plus metadata) and
any figures plotted from it. Runs made on different branches, with
different encoders, can then be loaded back and graphed together.
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np


def save_run(results, label, meta=None, root="runs"):
    """Write a sweep's results to runs/<timestamp>_<label>/data.json.

    results: dict of arrays/scalars as returned by the sweep functions.
    meta: free-form JSON-serializable dict (model params, bank description).
    Returns the run directory Path; save figures into it."""
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = Path(root) / f"{stamp}_{label}"
    run_dir.mkdir(parents=True)
    payload = {
        "label": label,
        "timestamp": stamp,
        "meta": meta or {},
        "data": {k: np.asarray(v).tolist() for k, v in results.items()},
    }
    (run_dir / "data.json").write_text(json.dumps(payload, indent=2))
    return run_dir


def load_run(run_dir):
    """Load a run back; data values are restored as numpy arrays."""
    payload = json.loads((Path(run_dir) / "data.json").read_text())
    payload["data"] = {k: np.asarray(v) for k, v in payload["data"].items()}
    return payload
