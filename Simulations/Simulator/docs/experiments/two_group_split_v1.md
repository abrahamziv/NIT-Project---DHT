# Two-Group Alphabet Split — v1

## Idea

Break Assumption 1(iii) in the simplest possible way: split the $N$ sensors
into two groups with different alphabet sizes.

- $k$ sensors use $M_{hi}$ symbols
- $N-k$ sensors use $M_{lo}$ symbols

$k=N$ recovers the uniform baseline. Sweep $k$ from $0$ to $N$.

## Construction

Uses the existing `EncoderBank` as-is, no new classes:

```python
from ddms import EncoderBank, ThresholdEncoder

def make_bank(k, N, M_hi, M_lo):
    hi = ThresholdEncoder(...)   # M_hi symbols
    lo = ThresholdEncoder(...)   # M_lo symbols, or [] for M_lo = 1 (silent)
    return EncoderBank([hi] * k + [lo] * (N - k))
```

## Fixed for v1

- $N$ — fixed sensor count
- $M_{hi}$, $M_{lo}$ — fixed alphabet sizes for the two groups
- model / SNR — fixed

## Swept

- $k$ — number of high-alphabet sensors, $k = 0, 1, \dots, N$

## Measured (y-axis, per $k$)

- `total_exponent` ($-\log J^N$) — headline
- `chernoff_bound` — cheap upper reference, if `total_rate` gets large
- report `total_rate` alongside each point (it changes with $k$ in v1 —
  not matched yet, that's v2)

## Output

One plot: x = $k$, y = `total_exponent` (and `chernoff_bound` as a dashed
reference line). $k=N$ point should match the existing uniform-baseline run.

## Explicitly out of scope for v1

- Matching total rate across $k$ (that's the next version, once this one
  runs and looks sane)
- More than two groups
- Any realization-dependent behavior

---

## Run

`figs/fig_two_group_split_v1.py` -> `runs/2026-08-02_163129_two_group_split_v1`.
Nothing in `src/ddms` changed; the script is self-contained.

Settings: $N=20$, SNR $0$ dB ($\mu=\sigma=1$), $p=1/2$, $M_{hi}=4$,
$M_{lo}=2$, $k=0,\dots,20$.

Thresholds: uniform $M$-cell quantizer on $y$, centred at $\mu/2$ and
spanning $\pm 2\sigma$. Centring at $\mu/2$ makes it symmetric in the two
hypotheses, so $M=2$ is exactly the LRT at $l=1$ and $k=0$ is comparable to
the committed baseline. Here it gives $l \in \{e^{-1}, 1, e\}$ for $M=4$.

Validation: $k=0$ reproduces
`runs/2026-08-02_154327_lrt_symmetric_exponent_vs_N` at $N=20$
($-\log J^N = 3.236801001873414$) to machine precision; `total_exponent`
$\ge$ `chernoff_bound` at every $k$; `uniform_alphabet` true only at
$k \in \{0, 20\}$.

## Results

| $k$ | rate (bits) | $-\log J^N$ | exponent/bit |
|---|---|---|---|
| 0 | 20 | 3.2368 | 0.1618 |
| 5 | 25 | 3.4866 | 0.1395 |
| 10 | 30 | 3.6688 | 0.1223 |
| 15 | 35 | 3.8524 | 0.1101 |
| 20 | 40 | 4.0225 | 0.1006 |

Monotone increasing in $k$, as it must be — in v1, $k$ buys rate. Doubling
the total rate ($20 \to 40$ bits) buys $+24.3\%$ exponent, while the same 40
bits spent as $N=40$ binary sensors gives $5.090$, i.e. $+57.3\%$. At 0 dB,
more sensors beats richer alphabets by a wide margin, and the value per bit
falls monotonically across the sweep. Establishing that properly is v2.

### Parity artifact: the $\Delta_N = 0$ atom

Per-$k$ gains oscillate with parity ($+0.112$ at $k=1$, $+0.019$ at $k=2$).
Cause: atoms of the law of $\Delta_N$ carry log-weights differing by exactly
the atom value, so an atom at $0$ has equal mass under both hypotheses and
contributes exactly half of it to

$$J^N = P(\Delta_N < 0 \mid H_1) + \tfrac12 P(\Delta_N = 0 \mid H_1)
\qquad (p = 1/2).$$

The $M=2$ atoms are $\pm 0.807$, the $M=4$ atoms $\pm 1.530, \pm 0.460$; a
zero sum needs $N-k$ even, so with $N$ even the tie atom exists only at even
$k$. Its half-mass is 46% of $J^N$ at $k=0$ and 9-15% at even $k>0$. The
tie-breaking convention is irrelevant: ties to $H_1$ and to $H_2$ give the
same $J^N$ at $p=1/2$. Within each parity class the curve is smooth; only
across parity is it contaminated.

For v2: compare at fixed parity, offset $t$ from zero, or use quantizers not
symmetric about $l=1$.
