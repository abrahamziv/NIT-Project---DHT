"""Run persistence: each sweep is saved to a fresh timestamped directory.

A run directory holds data.json (the raw sweep arrays plus metadata) and
any figures plotted from it. Runs made on different branches, with
different encoders, can then be loaded back and graphed together.
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np


def save_run(results, label, meta=None, root="runs", group=None, session=None):
    """Write a sweep's results to
    runs/<session>/<group>_<date>/<timestamp>_<label>/data.json.

    results: dict of arrays/scalars as returned by the sweep functions.
    meta: free-form JSON-serializable dict (model params, bank description).
    group: name for the dedicated parameter+date directory this run and its
    siblings (e.g. a paired comparison run, a compare figure) share; defaults
    to label so every run always gets its own directory.
    session: top-level bucket (e.g. "M2", "M4") for the encoder family this
    run belongs to, so unrelated families never interleave alphabetically;
    omit for runs that don't belong to a family.
    Returns the run directory Path; save figures into it."""
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    date = stamp[:10]
    root_path = Path(root) / session if session else Path(root)
    run_dir = root_path / f"{group or label}_{date}" / f"{stamp}_{label}"
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
