# Draft 1 — Asaf's Line Feedback

**Reviewed doc:** `draft1.tex` / `Draft_1_-_30_7.pdf` (submitted July 30, 2026)
**Course:** Network Information Theory 371-21814 · Prof. Asaf Cohen
**Team:** Oren Ram, Ziv Abraham

This file catalogs every note Asaf left on Draft 1, tags each by type/priority, and gives
the specific fix needed. Work through it in the order in **§3**, not top-to-bottom —
some items share a root cause and should be fixed together.

---

## 1. Legend

| Type | Meaning |
|---|---|
| 🔧 **Wording/Notation** | Copy-edit or symbol fix. No new math. |
| 🧩 **Structural** | Reframe or restructure a passage. Requires understanding, not just editing. |
| 🧮 **Math gap** | A derivation/justification is missing and must be written out. |

| Priority | Meaning |
|---|---|
| 🔴 High | Blocks correctness or was flagged as unclear/wrong |
| 🟡 Medium | Needed for quality but doesn't block anything else |
| ⚪ Low | Pure polish |

---

## 2. Triage Table

| ID | Section | Asaf's Note (paraphrased) | Type | Priority | Owner | Status |
|----|---------|---------------------------|------|----------|-------|--------|
| A1 | §1.1 | Remove — irrelevant | 🔧 | ⚪ | — | ⬜ |
| A2 | §1.2 | "Finite-N setting" framing is wrong emphasis — we care about $N\to\infty$ / error exponent | 🧩 | 🔴 | — | ⬜ |
| A3 | §2 | "Random" circled — name the variable | 🔧 | ⚪ | — | ⬜ |
| A4 | §2 | $y^i$ notation — vector or scalar? | 🔧 | ⚪ | — | ⬜ |
| A5 | §2, Eq. (2) | Expectation in cost function not explained — over what? | 🔧 | 🟡 | — | ⬜ |
| A6 | §2.1 (ii) | "Common reference measure" too vague — give closed form | 🔧 | 🟡 | — | ⬜ |
| A7 | §3.1 | Prop. 5.1 → Eq. (4) discussion unclear | 🧩 | 🔴 | — | ⬜ |
| A8 | §3.1 | $\gamma^{0:N\prime}$ notation ambiguous (looks like a different $N$) | 🔧 | 🔴 | — | ⬜ |
| A9 | §3.1 | $J^N_{EE}$ sign convention — usually has a minus | 🔧 | 🟡 | — | ⬜ |
| A10 | §3.3, Lemma 1 (after Eq. 7) | Reduction to per-sensor optimization asserted, not justified | 🧮 | 🔴 | — | ⬜ |
| A11 | §3.3, Lemma 1 (after Eq. 8) | "Depends on $y^i$ only through $l^i$" — not shown | 🧮 | 🔴 | — | ⬜ |
| A12 | §3.3, Lemma 1 (Eq. 8→9) | Missing derivation steps | 🧮 | 🔴 | — | ⬜ |
| A13 | §3.3, Lemma 2 | "...is coming from" — weak language, proof exists in paper | 🔧 | ⚪ | — | ⬜ |
| A14 | §3.4 | "Where to look" / "something to find" — too informal | 🔧 | ⚪ | — | ⬜ |

**14 items total: 8 wording/notation, 2 structural, 3 math gaps.** (A9 counted once, tied to A2 — see §3.)

---

## 3. Suggested Order of Attack

Don't go section-by-section. Group by what's actually the same fix:

1. **Batch 1 — pure copy-edits (30 min, do together):** A1, A3, A4, A5, A13, A14
2. **Batch 2 — the exponent-sign fix (A2 + A9 together):** these are the *same* underlying comment — see §4.2. Fixing the sign convention once resolves both.
3. **Batch 3 — Assumption (ii) closed form (A6):** quick, but needs the exact paper statement — given below, don't re-derive.
4. **Batch 4 — the Prop. 5.1 paragraph (A7 + A8 together):** same paragraph, same root cause (unclear logical order + bad notation). Rewrite once.
5. **Batch 5 — Lemma 1 math (A10, A11, A12), in this order:** these are three sequential derivation gaps in the *same* proof. Don't split between Oren/Ziv — do as one continuous derivation, since A10 sets up what A11–A12 need.

---

## 4. Detailed Notes

### A1 · §1.1 Background — remove
**Verdict:** Cut the whole subsection. It states the generic HT setup with no connection to this paper's contribution. Section 1.2 already does the real framing.

---

### A2 · §1.2 — wrong framing 🧩 🔴
> Circled: *"in our work we focus on the finite-N setting."*
> Asaf: we care about the $N\to\infty$ limit / error exponent, i.e. $P_e(N) \sim e^{-E\cdot N}$.

**The problem:** the sentence as written implies you're *avoiding* the asymptotic/exponent story in favor of finite-$N$ bookkeeping. That's backwards — the whole point of studying finite $N$ here (Lemmas 1–3, Theorem 1) is to characterize $J^N$ precisely enough to then take the limit and get the error exponent. The finite-$N$ analysis is the *means*, not competing content.

**Fix:** rewrite to state the goal as: characterize the finite-$N$ optimal cost $J^{N\star}$, in order to study its exponential decay rate $E := -\lim_{N\to\infty} \frac{1}{N}\log J^{N\star}$. This also directly resolves **A9** — see §4.2.

**Cross-reference:** this is *not* a new point — it's the same thing Asaf raised as **Q1** in `ASAF_NOTES.md` ("is the limit in Eq. (3)/(4) proven to be a lower bound / is optimality shown?"). Treat A2 as reinforcing that this framing has to be front-loaded in the introduction, not buried in §3.1.

---

### A3 · §2 — "random" circled ⚪
Likely referring to: *"the true one can be regarded as $H^\star$, a random variable..."* — **but verify against the annotated PDF**, the text extraction doesn't let me confirm exactly which instance was circled.

**Most likely actual issue:** the draft defines $H^\star$ here but then uses bare $H$ everywhere else (Assumption (i): "conditioned on $H$", Eq. (3): $f_{y^i|H}$, etc.). Pick one symbol and use it consistently through the whole document — either $H^\star$ throughout, or drop the star and just call it $H$ from the start (simpler, and matches the paper's own later usage).

---

### A4 · §2 — $y^i$ notation ⚪
> Comment: if $y^i$ is not a vector, write $y_i$ — unless the paper itself uses that notation.

**Resolution:** the paper uses superscript $y^i$ throughout (confirmed in both ICASSP and arXiv versions) — and it is *not* purely scalar: $\mathcal{Y}^i$ is a general (possibly multi-dimensional) Borel space, which is also why Assumption 1(iv) invokes a **Jacobian matrix** for $L(y^i)$ (Jacobians are only meaningful for vector-valued maps). So the superscript is correct and matches the source, but the notation genuinely is ambiguous on first read.

**Fix:** keep $y^i$, add one footnote at first use: *"the superscript indexes the sensor, not an exponent; $y^i$ may itself be vector-valued."* No further change needed.

---

### A5 · §2, Eq. (2) — unexplained expectation 🟡
**Fix:** one sentence after Eq. (2) stating the expectation is taken over $H^\star$ (or $H$, per A3) and $y^{1:N}$ jointly — i.e. $J^N(\gamma^{0:N}) = \sum_{j=1,2} P(H_j)\, \mathbb{E}[c(H_j, u^0) \mid H_j]$, expanding what randomness the expectation averages over before it collapses to $P_e$ in §3.1.

---

### A6 · §2.1, Assumption (ii) — needs closed form 🟡
**Fix:** replace "measured against one common reference measure" with the paper's own precise statement (Assumption 1(ii) in the arXiv version):

$$
\text{There exists } Q \in \mathcal{P}(\mathcal{Y}^i) \text{ and } f: \mathcal{Y}^i \times \mathcal{H} \to \mathbb{R} \text{ s.t. for any Borel } A_i \subseteq \mathcal{Y}^i,\quad
\Pr(y^i \in A_i \mid H) = \int_{A_i} f(y^i, H)\, Q(dy^i) \quad \forall H \in \mathcal{H}.
$$

In words: $f(\cdot, H)$ is the Radon–Nikodym density of $y^i$'s conditional law w.r.t. a *fixed* measure $Q$ that doesn't depend on $H$ — that's the "common dominating measure" precisely. State it this way instead of the vague prose version; it also sets up Eq. (3)'s likelihood ratio cleanly since $l^i = f(y^i,H_2)/f(y^i,H_1)$ is then just a ratio of two densities against the same $Q$.

---

### A7 · §3.1 — Prop. 5.1 → Eq. (4) discussion unclear 🧩 🔴
The current paragraph runs the logic in a confusing order and restates its own conclusion mid-sentence. Rebuild as two clean steps:

**Step 1.** By [Prop. 5.1, 2], since $y^1,\dots,y^N$ are i.i.d. given $H$ and $P_{Y|H_1}\neq P_{Y|H_2}$: there **exists** *some* sensor policy $\gamma'$ (used identically by all sensors) and constants $\alpha,\beta>0$, independent of $N$, such that using $\gamma'$ at every sensor and MAP at the FC gives $P_e < \alpha e^{-N\beta}$.

**Step 2.** $J^{N\star} := \inf_{\gamma^{0:N}} J^N(\gamma^{0:N})$ is an infimum over *all* policy profiles — and the profile from Step 1 is just one candidate in that infimum. So:
$$
J^{N\star} \le J^N(\gamma^0_{\text{MAP}}, \gamma', \dots, \gamma') < \alpha e^{-N\beta}.
$$

That's the whole argument: one good policy exists $\Rightarrow$ the optimum is at least as good. Write it in that order — existence first, then "therefore the infimum inherits the bound" — not the other way around.

---

### A8 · §3.1 — $\gamma^{0:N\prime}$ notation 🔴
> Asaf: "the tag $'$ on $N$ — is that a different sensor count?"

**No** — and that's exactly the bug. The prime does **not** mean a different $N$. Per Step 1 above, $\gamma'$ is a *single sensor policy*, reused identically by all $N$ sensors: the achieving profile is $(\gamma^0_{\text{MAP}}, \gamma', \gamma', \dots, \gamma')$, still with $N$ sensors, same $N$ as everywhere else. Writing it as $\gamma^{0:N\prime}$ visually reads like "prime the whole $(0{:}N)$ index," which is why it looks like a count change.

**Fix:** don't prime the vector. Write the achieving profile out explicitly, e.g. $\big(\gamma^0_{\text{MAP}},\underbrace{\gamma',\dots,\gamma'}_{N}\big)$, or give it its own symbol distinct from anything indexed by $N$.

---

### A9 · §3.1 — $J^N_{EE}$ sign convention 🟡
> Asaf: usually there's a minus sign, and then you don't need the explanatory sentence below.

**Current:** $J^N_{EE} = \log(J^N)/N$, negative since $J^N\le 1$, so the draft has to explain "we want to minimize a negative quantity."

**Fix — do this together with A2:** define the (positive) error exponent
$$
E^N := -J^N_{EE}(\gamma^{0:N}) = -\frac{1}{N}\log J^N(\gamma^{0:N}) \ge 0,
$$
so $P_e \sim e^{-E^N \cdot N}$ — matching standard error-exponent convention (Chernoff/Stein) **and** matching exactly the form Asaf wrote in his A2 comment. Then the goal becomes "maximize $E^N$," no sign caveat needed, and it's the *same* fix as reframing §1.2 around the exponent. Just be consistent about which sign convention is used from here through Lemma 1–3 and Theorem 1 — don't flip back.

---

### A10 · Lemma 1, after Eq. (7) — reduction not justified 🧮 🔴
The draft states the per-sensor optimization (Eq. 8) as if it directly proves Eq. (7), without explaining why fixing everyone else and optimizing one sensor at a time establishes the *joint* infimum equality. This needs an actual argument, not just the restated formula.

**The argument (write this up, don't skip it):**
- $\Gamma^N_{TS} \subseteq \Gamma^N$ trivially gives $\inf_{\Gamma^N} J^N \le \inf_{\Gamma^N_{TS}} J^N$ — one direction is free.
- For the other direction: fix sensors $j\neq i$ to **arbitrary** (not necessarily optimal) policies $\gamma^{j}$. Sensor $i$'s best response, shown via Eq. (8)–(9), is always achievable by a threshold policy — because the objective is *linear* in $l^i$, so the argmin is always an endpoint/threshold rule, **regardless of what the other sensors are doing**.
- Since this dominance holds for *any* fixed configuration of the other sensors, you can replace each sensor's policy with its threshold-type best response **one at a time**, in any order, without ever increasing the cost.
- After replacing all $N$ sensors this way, you land on a fully-threshold profile whose cost is $\le$ the cost of the original (arbitrary) profile. So $\inf_{\Gamma^N_{TS}} J^N \le \inf_{\Gamma^N} J^N$.
- Combined with the trivial direction: equality.

This is the standard "person-by-person optimization dominance" argument — state it as such.

---

### A11 · Lemma 1, after Eq. (8) — "depends on $y^i$ only through $l^i$" 🧮 🔴
Underlined, unclear. This needs the actual Bayes' rule computation, not an assertion.

**Derivation to include:**
$$
P(H_j \mid y^i) = \frac{f_{y^i|H}(y^i \mid H_j)\, P(H_j)}{f_{y^i}(y^i)}, \qquad j = 1,2.
$$
Take the ratio (the normalizer $f_{y^i}(y^i)$ cancels):
$$
\frac{P(H_2\mid y^i)}{P(H_1\mid y^i)} = \frac{f_{y^i|H}(y^i\mid H_2)}{f_{y^i|H}(y^i\mid H_1)}\cdot\frac{P(H_2)}{P(H_1)} = l^i \cdot \frac{P(H_2)}{P(H_1)}.
$$
So the *ratio* of posteriors — which is all that matters for comparing the two terms inside the Eq. (8) infimum — depends on $y^i$ **only** through $l^i$; the individual posteriors do too, up to the shared normalizer that drops out of the comparison. That's the sufficiency statement showing up concretely, not just abstractly (§3.2).

---

### A12 · Lemma 1, Eq. (8) → (9) — missing steps 🧮 🔴
**Derivation to include**, using A11's result: factor $P(H_1\mid y^i)>0$ out of the sum in Eq. (8) (it's a positive constant w.r.t. $u^i$, so it doesn't affect the argmin):
$$
\sum_{j=1,2} P(H_j\mid y^i)\, \mathbb{E}_{\gamma^{-i\star}}[c(H_j,\gamma^0(u^{1:N}))\mid H_j]
= P(H_1\mid y^i)\left[g^i(H_1,u^i) + \frac{P(H_2\mid y^i)}{P(H_1\mid y^i)} g^i(H_2,u^i)\right].
$$
Drop the positive prefactor $P(H_1\mid y^i)$ (doesn't change the $\arg\inf_{u^i}$), then substitute the ratio from A11:
$$
\inf_{u^i}\Big[g^i(H_1,u^i) + g^i(H_2,u^i)\, l^i\, \tfrac{P(H_2)}{P(H_1)}\Big],
$$
which is exactly Eq. (9). Two lines, both worth showing explicitly.

---

### A13 · Lemma 2 — "...is coming from" ⚪
> Asaf: bad language, there's an actual proof ahead in the paper.

**Fix:** you already derive $\Delta_N$ explicitly in Eqs. (12)–(17) right after this sentence — the phrasing just undersells it. Replace with something like: *"We now derive the explicit form of the fusion center's optimal rule, following the MAP-optimality result of [2]."* Then Eqs. (12)–(17) are the derivation, not "where it's coming from."

---

### A14 · §3.4 — informal language ⚪
> "tell us where to look" / "something is actually there to find" flagged as too informal for a report.

**Fix:** *"Lemmas 1 and 2 characterize the necessary structure of an optimal policy — threshold at the sensors, MAP at the FC. Theorem 1, using the compactness result of Lemma 3, establishes that a minimizer with this structure exists."*

---

## 5. Open Items to Flag Back to Asaf (not fixable from our side alone)

- **A3** — confirm exact word/location circled (verify against the physical annotated PDF, text extraction is ambiguous here).
