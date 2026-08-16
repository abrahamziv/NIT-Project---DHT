# DDMS Simulator

Simulator for decentralized detection with many sensors. Spec in
`claude.md`, math in `DDMS.pdf`, plan in `docs/PLAN.md`.

## Usage

    uv sync
    uv run pytest
    uv run python figs/fig_exponent_vs_N.py
    uv run python figs/fig_compare_runs.py runs/<dir1> runs/<dir2>

Each figure script runs a sweep and saves everything to a fresh
`runs/<session>/<group>_<date>/<timestamp>_<label>/` directory: `data.json`
with the raw arrays and metadata, plus the figure PDF. `session` buckets
by encoder family (e.g. `M2`, `M4`) so unrelated families never interleave
alphabetically; `group` names the simulation's parameters (e.g.
`vanilla_vs_silence_snrp5`) so related runs land together; `date` has no
hour, so everything run the same day shares one parent folder. Commit run
directories so results from different branches (different encoders) can
be compared and overlaid with `fig_compare_runs.py`.

## Structure

    src/ddms/model.py    Statistical Model (GaussianShift, DiscreteLR)
    src/ddms/encoder.py  Encoder, EncoderBank; VanillaEncoder is the
                         empty template to fill in
    src/ddms/fusion.py   Fusion Center (exact J^N, Chernoff bound)
    src/ddms/sweep.py    parameter sweeps
    src/ddms/runs.py     save_run / load_run for grouped, timestamped run dirs
    figs/                one script per figure, output into runs/
    runs/                saved sweep runs, grouped by encoder family + parameters + date, committed
