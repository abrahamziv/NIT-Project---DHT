# 📋 Asaf Meeting Notes — NIT Project
**Paper:** *Decentralized Detection with Many Sensors: Optimality of Exchangeable and Identical Encoding Policies*
**Authors:** Sanjari, Saldi, Gezici, Yüksel (ICASSP 2026)
**Course:** Network Information Theory 371-21814 · BGU · Prof. Asaf Cohen · Spring 2026
**Team:** Oren Ram, Ziv Abraham

---

## 🗂️ Table of Contents
1. [Open Questions — Q\&A with Asaf](#open-questions--qa-with-asaf)
2. [Asaf's Discussion Points](#assafs-discussion-points)
3. [Action Items](#action-items)
4. [Key Implications for the Critical Summary](#key-implications-for-the-critical-summary)

---

## Open Questions — Q&A with Asaf

### Q1 · The Limit in Equation (3)

> **Oren & Ziv:**
> We understood that Equation (3) is the really interesting performance metric — it captures performance over an "infinite" number of sensors. But what is the limit itself?

> **Asaf:**
> Do they prove *optimality*? That the resulting value is the **best possible**?

**Status:** Open. The paper establishes existence and structure of optimal policies, but the gap between what Eq. (3) achieves and a provable lower bound is not addressed.

---

### Q2 · Effect of Action Space Size on Performance and Theorem 1

> **Oren & Ziv:**
> How does the size of the action space $\mathcal{U}$ affect performance, and to what extent is the proof of Theorem 1 (existence of a deterministic threshold policy) based on the fact that $|\mathcal{U}|$ is **fixed** for each sensor?

> **Asaf:**
> **(1)** The value of $|\mathcal{U}|$ does not affect performance — at least not in the way it is expressed in Equation (3).
>
> **(2)** How important is it that $|\mathcal{U}|$ is fixed *per sensor*?
> **Conjecture:** Changing $|\mathcal{U}|$ per sensor while maintaining a fixed **average** may allow more room for maneuver and potentially yield **better performance**.

**Status:** Open research direction. This is essentially a variable-rate / adaptive quantization question framed in the sensor policy context.

---

### Q3 · Explicit Statement of Threshold Policy + MAP Rule at FC

> **Oren & Ziv:**
> The paper proves the *existence* of a threshold policy at sensors and a MAP rule at the fusion center, but we did not see it actually *state* the result explicitly. We will check the paper and, if not there, try to reach it analytically or numerically.

> **Asaf:**
> I did not understand the comment.

**Status:** Clarification needed. Re-read Theorems 1–2 and check whether the MAP structure at the FC is stated as a named result or only implicitly assumed/used.

---

### Q4 · Dynamic Rate Allocation — The Central Open Question ⭐

> **Oren & Ziv:**
> When **dynamic (non-uniform) allocation of rates** is allowed for each sensor, will we get better performance for the same total rate $R$?
> *(Think about directions for simulation.)*

> **Asaf:**
> **Yes.**

**Status:** Confirmed direction. This is the primary open question for the critical analysis (Part 2) and the simulation component.

> **Asaf's elaboration (separate annotation):**
> *"If you are allowed to allocate rate between sensors — not just one bit per sensor (or $\log |\mathcal{U}|$ bits like here) — you can do much better.*
> *The sensors will (distributively) see if their measurement is in a place allowing to distinguish between $H_1$ and $H_2$ or not.*
> *If not — they will not send (or send less). If so — they will send at high rate.*
> *At the limit: only one sensor who got a measurement at a place where $f_1$ and $f_2$ differ significantly will send the whole $N$ bits.*
> *You will get a better error exponent at the same rate!"*

---

## Asaf's Discussion Points

These are standalone points Asaf wants the critical summary to address:

### Point A · LRT as a Sufficient Statistic

> Talk about the fact that the **Likelihood Ratio Test (LRT) is a Sufficient Statistic** and does not lose "important" information.

**What to develop:**
- The LRT $\Lambda(x) = f_1(x)/f_0(x)$ compresses the raw observation $x$ without losing any information relevant to the hypothesis test.
- This is the information-theoretic justification for why threshold policies on the LRT are not just convenient but *optimal* — no deterministic encoding can do better given the same rate constraint.
- Connection to the Data Processing Inequality and Neyman-Pearson.

---

### Point B · The FC Knowing Each Sensor's Policy

> Talk about the real meaning of the fact that the **fusion center knows the policy of each sensor**.

**What to develop:**
- This is a *common knowledge* / *shared information structure* assumption.
- Without this, the FC's MAP rule cannot be implemented correctly — it needs to invert $\gamma_i(x_i)$ to form a belief about the underlying $x_i$.
- This connects to distributed source coding: if sensor policies were unknown to the FC, it would require a fundamentally different (and harder) decoding problem.
- Ask: what breaks if sensors use *private* randomization unknown to the FC? (Theorem 2 actually addresses this by requiring the FC to have access to the randomness of encoders' policies.)

---

## Action Items

| # | Task | Owner | Priority | Status |
|---|------|-------|----------|--------|
| 1 | Re-read Theorems 1–2 and verify whether MAP at FC is stated explicitly | Both | High | 🔲 Open |
| 2 | Formalize Q1: Is Eq. (3) proven to be a lower bound? Check if there is a converse | Both | High | 🔲 Open |
| 3 | Simulation: implement fixed uniform rate allocation vs. dynamic allocation, compare error exponent | Both | High | 🔲 Open |
| 4 | Write discussion of LRT as sufficient statistic for Part 2 | Oren | Medium | 🔲 Open |
| 5 | Write discussion of FC knowledge of sensor policies for Part 2 | Ziv | Medium | 🔲 Open |
| 6 | Clarify Q3 with Asaf in next meeting | Both | Low | 🔲 Open |

---

## Key Implications for the Critical Summary

### Part 2 — Critical Analysis Centerpiece
The **rate allocation question (Q4)** is the central direction for the critical analysis. The paper assumes a fixed, uniform $\log|\mathcal{U}|$ bits per sensor. Asaf explicitly confirmed that adaptive allocation yields a strictly better error exponent at the same total rate $R$.

**Argument skeleton:**
1. State what the paper assumes: fixed action space $\mathcal{U}$, same for all sensors.
2. Identify the constraint: every sensor uses the same rate $r = \log|\mathcal{U}|$ bits.
3. Asaf's conjecture: allow each sensor $i$ to use rate $r_i$ subject to $\sum_i r_i \leq R$.
4. Sensors with observations in informative regions ($f_1/f_0$ far from 1) get high rate; others send nothing.
5. This breaks the exchangeability of policies — sensors are no longer identical.
6. Claim: the error exponent improves. Show this numerically in the simulation.

### Connection to Course Syllabus (Topic 5 — Distributed Hypothesis Testing)
- The paper sits at the intersection of decentralized detection and information-constrained inference.
- Rate allocation connects directly to the Ahlswede-Csiszár framework for hypothesis testing under communication constraints.
- The LRT sufficient statistic point connects to Stein's Lemma and error exponents in hypothesis testing.

---

*Last updated: June 2026 · File maintained in `/notes/ASAF_NOTES.md`*
