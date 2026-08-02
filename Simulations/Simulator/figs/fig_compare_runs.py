"""Overlay total-exponent curves from several saved runs.

Usage: python figs/fig_compare_runs.py runs/<dir1> runs/<dir2> ...
Writes compare.pdf to the current directory. All runs must share the same
x variable (meta["x"], falling back to whichever of N/snr_db/rate exists).
"""

import sys

import matplotlib.pyplot as plt

from ddms import load_run


def x_key(run):
    if "x" in run["meta"]:
        return run["meta"]["x"]
    return next(k for k in ("N", "snr_db", "rate") if k in run["data"])

runs = [load_run(arg) for arg in sys.argv[1:]]

fig, ax = plt.subplots()
for run in runs:
    key = x_key(run)
    ax.plot(run["data"][key], run["data"]["total_exponent"], label=run["label"])
ax.set_xlabel(x_key(runs[0]))
ax.set_ylabel("total exponent")
ax.legend()
fig.savefig("compare.pdf")
print("saved compare.pdf")
