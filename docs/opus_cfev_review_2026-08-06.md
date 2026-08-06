## VERDICT: **DEAD**

**Earliest failed gate:** §1.6 (*The extrapolated Recall@1 risk*), enabled by §1.5 (*σ̂ = softplus(σ_raw)*). Everything downstream — §2's degeneracy proofs, §4's controls, §5's scaling law — is scored against a term that does not exist as specified.

**Decisive mechanism-level reason:** Cosine similarity is bounded, so the negative tail is in the Weibull domain (ξ<0) — the proposal asserts this itself in §6.3. A GPD with ξ<0 has a **finite right endpoint** `u + σ/|ξ|`, and its survival is **exactly zero** beyond it. CFEV's estimand `q_i = ζ̂·(1+ξ̂w̃_i)^{-1/ξ̂}` is that survival evaluated at the hardest *positive*. So the objective carries gradient only while `s⁺_i < u + σ̂/|ξ̂|` ≈ the estimated **maximum in-batch negative** — i.e. only for anchors that are already failing — and is identically 0 (times any M) the moment a positive beats the negative population, which is the state the method exists to produce. `M · 0 = 0`: the advertised "extrapolate to M = 59,551" is vacuous in precisely the regime that motivates it. This is not a repair; it follows from bounded support plus the choice of estimand.

---

## The verification

**PWM is correct** (validated against exact GPD draws, n=10⁵, both signs):

| true (ξ, σ) | recovered ξ_raw | recovered σ_raw |
|---|---|---|
| (0.00, 1.0) | +0.0023 | 0.9968 |
| (+0.30, 1.0) | +0.3010 | 0.9983 |
| (−0.30, 1.0) | −0.3009 | 1.0017 |
| (−0.45, 0.07) | −0.4505 | 0.06998 |

Hosking–Wallis is transcribed correctly, including the ξ = −k convention flip (`ξ_raw = 2 − â₀/D`) and the r=1 plotting weight `(k−j)/(k−1)`. §1.5's exponential check (`a_r = σ/(r+1)²`) is right.

**§1.5's positivity map destroys it.** `σ_raw` is the mean-exceedance scale of cosine similarities above their 90th percentile — structurally ≪ 1. `softplus` is near-identity only for arguments ≳ 3, which cosine can never reach:

| σ_raw | 0.01 | 0.05 | 0.08 | 0.15 | 0.30 |
|---|---|---|---|---|---|
| σ̂ = softplus+1e−4 | 0.698 | 0.719 | 0.734 | 0.771 | 0.855 |
| inflation | ×70 | ×14 | ×9.2 | ×5.1 | ×2.8 |

σ̂ is pinned to ≈0.70–0.77 regardless of the data. The fitted tail scale is discarded.

**The two horns are coupled.** Simulating a 23-class × 4-image half in 512-D with low-rank class structure (u = p90 of ~4048 negatives, k ≈ 404 exceedances, ζ̂ = 0.100–0.104 — §2.3's "ζ̂ pinned near p₀" claim is correct):

| regime | ξ̂ | σ_raw | σ̂ as specified | anchors NaN with **specified** σ̂ | anchors NaN with **corrected** σ̂ |
|---|---|---|---|---|---|
| loose / early | −0.089 | 0.041 | 0.714 | 0/92 | 0/92 |
| typical trained | −0.374 | 0.087 | 0.737 | 0/92 | **92/92** |
| tight / late | −0.392 | 0.141 | 0.766 | 0/92 | **92/92** |

Fix the softplus and `log(1 + ξ̂w̃)` takes the log of a negative number for **every anchor** once the encoder separates classes — a NaN that appears *after* the λ_t ramp completes (epoch 25), not at init. Keep the softplus and σ̂ is a constant ≈0.73 carrying no tail information, so `q_i` is not a tail probability and the novelty claim in §0/§3 is void. Clamp the log argument instead and `q_i ≡ 0` with zero gradient. All three exits fail.

Separately, ξ̂ = −0.37/−0.39 against a ±0.45 clamp in an *untrained* simulation: falsification test #7 is near-firing before training starts.

---

## Second independent kill: M is gradient-inert (§1.7, §4, §5.1)

`L_i = log(1 + M q_i)` has only two asymptotic regimes:

- **M q ≫ 1:** `L_i = log M + log ζ̂ + ℓ_i + O(1/Mq)` — M is an **additive constant, zero gradient**. My simulation gives median `M q ≈ 2.5–4.5 × 10³` for **100% of anchors** under the specified formulas. This is the actual operating regime, not the `q ≪ 1` regime §1.6 assumes.
- **M q ≪ 1:** `L ≈ M·mean(q)`, so `‖∂L_CFEV/∂Z‖ ∝ M`, so §1.7's rule `λ* = r‖∂L_PA‖/‖∂L_CFEV‖ ∝ 1/M`. The product **λ*·M cancels exactly**.

M therefore has no effect on the update in either regime; only the log1p knee's position among anchors depends on it. Consequences:

- **C2 (M := 180) is not the decisive control** it is billed as. At M=180 the median `M q ≈ 9.9` — still the log regime. C2 and CFEV differ by a ~10% per-anchor reweighting, not by presence/absence of extrapolation. Falsification test #2 (`CFEV − C2 < +0.6` ⇒ rejected) is near-certain to fire and would be uninformative.
- **C2′ (M-sweep) cannot identify gallery extrapolation.** With λ* re-derived per arm it is exactly null; with λ* held fixed it is a pure loss-scale sweep, confounded with C1.
- **§5.1's `Δ ≈ γ·log(M/m)` law has no derivation** from the objective — the objective's M-dependence is `log M`, an additive constant. Falsification test #4 tests a law the mechanism does not imply.

## Third: the loss contains no negative-pair information per anchor (§1.6)

`L_{B|A}` is separable — one batch-global scalar plus a monotone function of `s⁺_i` alone. **B's negatives never enter it.** With ξ̂=−0.374, σ̂=0.737, ζ̂=0.103: `log(Mζ̂) = 8.72` constant, `ℓ_i ∈ ≈[−1.5, 0]` variable. CFEV is a bounded soft-margin pull on the hardest positive with a ~1.5-nat range, plus batch statistics.

This is not an estimator of Recall@1 risk. R@1 error is governed by the **query-conditional** negative tail (each query's own confusers); `(ξ̂, σ̂, ζ̂)` are fitted on the **pooled** negative population, so `q_i` is a deterministic monotone map of `s⁺_i`, identical for a hub query and an isolated one. Consequently **C5 (stopgrad on ξ̂, σ̂, ζ̂) is the entire loss up to global scalars** — and §4 concedes that if C5 reproduces the gain "the mechanism claim collapses."

## Cross-fitting is variance injection, not a constraint (§1.3, §2.2)

- The nuisance is three scalars from a **closed-form** estimator on ~405 exceedances. Own-observation bias is O(1/k). Chernozhukov et al. (2018) cross-fitting exists to remove overfitting bias from a *high-capacity* nuisance learner; that bias is not present here. The device solves a problem the estimator does not have.
- The halves are **uniformly random partitions of the same 45 batch classes**, redrawn each step. `(ξ̂_A,σ̂_A,ζ̂_A)` and `(ξ̂_B,σ̂_B,ζ̂_B)` are i.i.d. estimates of one batch-level quantity. Averaging `L_{B|A}` and `L_{A|B}` equals the same-batch loss computed with a noisier half-sample estimate. It does not block pair memorization (the memorizable content lives in `s⁺` and in a negative population unchanged by the split) and it does not enforce tail transfer (no cross-half *negative* term exists).
- The O(C^{−1/2}) U-statistic argument is correct but **non-discriminating**: the mean negative similarity is also a second-order U-statistic over classes with the same rate — so §2.2 does not separate CFEV from C6, the control it must beat. With C=23 per half, C^{−1/2} = 0.21 per step. §2.2's own "honest limit" is the crux and is unresolved.
- Effective sample size: 4048 pair similarities are functions of 92 vectors — d.o.f. ≈ O(92), not O(4048). The 405 "exceedances" are dominated by a few hub images.
- POT condition: Pickands/Balkema–de Haan is asymptotic in u → right endpoint. A fixed 10% fraction of a **mixture over class pairs** does not license the GPD form, and the extrapolation target is the *unseen-class* law, about which the fitted tail says nothing.

## Controls, protocol, cost (§4–§6)

- **C1 does not match gradient norm.** `‖∇L_PA + λ∇L_CFEV‖ = (1+r)‖∇L_PA‖` requires the gradients to be *parallel*. Generic near-orthogonality in 92,160 dimensions gives `√(1+r²) = 1.118`, not 1.5 — C1 over-scales by **34%**. Worse, AdamW is first-order invariant to global loss scale except through decoupled WD and ε, so a Frobenius match is the wrong calibration target for the confound §1.8 names.
- **C3 is a fair control in principle** (empirical survival, same wrapper, same cross-fit) and §8 correctly names it the largest worry. But under the corrected σ̂ the parametric arm is NaN and under the specified σ̂ it is not parametric, so C3 has nothing to contrast against.
- **Threshold identifiability:** seven tests × four datasets with no multiplicity correction; `CFEV − max(C5,C6,C7,C9)` is a max-statistic over four arms; the pass thresholds (0.8/0.6/0.4/0.4) were set after the §5.2 forecasts (Δ≈2.9 on SOP) were written and are not independent of them. No commitment to raw *and* selection-corrected reporting.
- **Admissibility of the forecast:** the best case is two point-estimate crossings at P≈0.55 (SOP) and P≈0.62 (In-Shop, where §6.3 concedes no significance test is possible), with explicit forecast failure on CUB and Cars. Even with a working mechanism, that is a coin flip on half the benchmarks with no available significance claim on either — not an admissible expected-value case for a frontier objective.
- **Cost is credible.** ~1.00× epoch time, 0 deployed parameters, +260 kB. (Two 92×92×512 Gram blocks are ~17 MFLOP, not 8.7 — immaterial at 4×10⁻⁶ of a 2.2 TFLOP step.)

## Novelty

Moot given the earlier gate, and I flag a limit on my own evidence: I did **not** run primary-source retrieval for §3 in this pass, so I neither confirm nor contest the survey. From domain knowledge the mechanism distinctions drawn against EVM/OpenMax/Meta-Recognition (post-hoc vs in-graph), Histogram loss (mean functional vs order statistic) and Recall@k-Surrogate (empirical vs parametric) are the right axes. But the "new supervision object" reduces, as executed, to a bounded monotone pull on the hardest positive similarity normalized by a batch-global scalar — which is occupied territory. There is no new supervision *action* to credit.

## Preserved correct subcomponents

1. **PWM/Hosking–Wallis transcription (§1.5)** — validated to ±0.01 in ξ and <1% in σ across both signs. Reusable as-is with `exp(·)` or a scale-matched positivity map.
2. **PWM-over-MLE justification** — correct: GPD-MLE is ill-behaved for ξ < −1/2, which is the relevant regime here.
3. **ζ̂ self-normalization (§2.3 D2)** — correct; simulation gives 0.100–0.104 for p₀=0.10. But this makes gradient path 3 a near-null direction, contradicting §1.7's "all four are load-bearing."
4. **All arithmetic checks pass:** |N_A|=4048, |N_B|=3696, k≈405, m⁻=176, log(59551/176)=5.82, the four log(M/m) values (3.48/3.80/4.97/5.80), the γ=0.5 products, and §5.2's C0→CFEV deltas. Dataset cardinalities (5,864/8,054/25,882/59,551; C=100/98/3,997/11,318) are right, as is the ProxyAnchor recipe including the `1/|P|` normalization of the negative term.
5. **§6.3's Weibull-domain admission** — correct, and it is the fact that kills the method.
6. **M = |D_train|, refusing the test gallery** (§1.9) — the right call.
7. **Falsification test #7** (clamp saturation) is well-chosen; my simulation shows it near-firing at initialization.
8. The §5.4/§7/§8 disclosure structure and the refusal to inherit PFML's frontier are good practice.

## What revival would require

Not a patch. A different estimand: for bounded support the object with non-trivial M-dependence is the **distance to the tail's right endpoint** `(u + σ/|ξ|) − s⁺` under a reverse-Weibull/GEV parameterization, or a query-conditional tail fitted per anchor. Either is a new method, needing its own degeneracy analysis and its own M-identification argument — and it would still need a control that can distinguish gallery extrapolation from loss scaling, which §4 currently does not contain.
