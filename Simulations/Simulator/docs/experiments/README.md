# Experiments

One experiment per branch, so parallel encoder studies never collide.
Per slug: branch `exp/<slug>`, spec `docs/experiments/<slug>.md` (results
appended after the run), script `figs/fig_<slug>.py`, output
`runs/<timestamp>_<slug>/`. Experiment scripts are self-contained; changes
to `src/ddms` are a last resort and must be additive and tested.

| Slug | Branch | Run |
|---|---|---|
| `two_group_split_v1` | `exp/two-group-split-v1` | `runs/2026-08-02_163129_two_group_split_v1` |
