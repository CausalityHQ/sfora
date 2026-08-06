# ADJUDICATION — NSRC (Pass 41), frozen 2026-08-06

## VERDICT: **DEAD**

**Earliest failed gate: Gate 1 (mathematics / mechanism), at the first audited object — the projector. No GPU work is warranted.**

**Decisive mechanism-level reason.** `Π` is defined as the **95%-energy eigenprojector** of `M = (1/K)Σ p_c p_cᵀ`, not the projector onto the algebraic proxy span. Because `tr(M) = 1`, the threshold `τ = 0.95` *directly sets* the retained proxy mass in the complement:

```
(1/K) Σ_c ‖Π⊥ p_c‖² = Σ_{i>r} λ_i ≈ 0.05     (measured: 0.0491 isotropic, 0.0493 clustered proxies)
                                              rms ‖Π⊥ p_c‖ ≈ 0.222 per proxy
```

So `Π⊥ p_c ≠ 0` for essentially every class, by construction, with a magnitude the threshold itself guarantees. The correct null-block task gradient is therefore

```
Π⊥ ∇_h L_PA = (1/‖h‖)·Π⊥(Σ_c w_c p_c)  −  (Σ_c w_c s_c)·b/‖h‖²
              └── tangential, omitted ──┘   └── the only term §2 keeps ──┘
```

The frozen §2 drops the first term. It is not negligible: at the energy floor (`‖b‖²/‖h‖² = 0.15`) with one positive proxy and five hard negatives, I measure **‖tangential‖ / ‖radial‖ = 2.23** — the omitted term is *larger* than the retained one.

Consequences, both fatal, both at Gate 1:

1. **Collapse theorem is false.** The task gradient in `Π⊥` is not "exactly radial"; it rotates `b` toward/away from the proxy-tail directions. The complement is *not* unsupervised, so the premise that it is "monotonically destroyed" for lack of supervision does not hold as frozen.
2. **Pointwise-orthogonality theorem is false.** `∇L_null` lies in `range(Π⊥)` and is `⊥ĉ` (that part is correct), but the tangential task term is a generic vector in that block and is not `⊥ ∇L_null`. Measured `cos(∇L_PA, ∇L_null) ≈ 9.2×10⁻³` for an admissible null direction — **four orders of magnitude above the proposal's own F5 threshold of 10⁻⁶.** The frozen algebra guarantees F5 fires.

This destroys the *sole stated distinction* from the nearest prior work. §3's Anti-Collapse bullet rests entirely on "*zero* interference rather than a tuned truce." There is no zero interference. What remains is Anti-Collapse Loss's coding-rate maximization applied to a subspace, traded off by scalar weights `λ_i, λ_r, λ_f` — a tuned truce.

The repair (set `τ = 1`, project on the exact span) is a **new proposal** and cannot upgrade this one. It also does not survive: see item 6, where it is self-contradictory.

---

## Independently sufficient kills (each stands alone)

### Item 4 — the anti-degeneracy defense is refuted by direct evaluation

§2(b) claims: *"`L_rate` is `−log det`, unbounded above as any eigenvalue → 0; a rank-deficient code has infinite loss."* This confuses `−log det(CCᵀ)` with the frozen `−log det(I + αCCᵀ)`. The `+I` makes every eigenvalue `≥ 1`, so `log det ≥ 0` and `L_rate ≤ 0` — **bounded**. Evaluating the frozen expression at `B=128, d−r=424, κ=1`:

| code | `L_rate` |
|---|---|
| constant (rank 1) | **−0.0143** = `−log(1+424)/424`, closed form |
| rank 4 | −0.0440 |
| rank 64 | −0.2713 |
| full-rank isotropic | −0.4134 |

Total collapse costs **0.399** at `λ_r = 1`, against `L_PA` magnitudes of order 1–10. This is the MCR² functional (Yu, Chan, You, Song, Ma, NeurIPS 2020, arXiv 2006.08558), bounded by construction. Since `L_inv` is *globally minimized* (exactly 0) by a constant code, `(L_inv, L_rate)` has a cheap joint near-optimum at a near-constant, low-rank code. This is precisely the collapse mode VICReg and Barlow Twins guard against with an explicit variance/decorrelation term; NSRC claims the guard and does not have it.

Compounding, all unaddressed in the frozen text:
- **Energy floor is stationary at collapse.** `∂L_f/∂‖b‖ = −4·max(0, ε−u)·‖b‖` → **exactly 0 at `b = 0`.** The squared hinge has vanishing restoring force at the one point it must repel. §2(a)'s "excluded by `L_f`" is false; `b = 0` is an equilibrium of the floor.
- **`c = 0` has no epsilon.** `∇L_null ∝ 1/‖c‖`. The `λ`-ramp starts at epoch 5, after 5 epochs of `L_PA` shrinking `‖b‖` — the null loss switches on exactly where it is most singular.
- **Rank ceiling.** `B = 128` image passes = 64 distinct images × 2 views, and `L_inv` drives each view pair together ⇒ `rank(ĈĈᵀ) ≤ 64` of 424 available directions. The rate term can certify at most 15% of the block it claims to fill, per step.
- **The rate term is anti-metric by construction.** The sampler is 4 images/class; `L_rate` maximizes spread *within the batch*, i.e. it explicitly rewards **separating same-identity images** in the deployed block.

### Item 5 — the deployment argument is the wrong sign, and D1 has no favorable branch

At `γ = 1`, `Π + γΠ⊥ = I`, so `W̃ = W` exactly — the deployed model is plain `normalize(Wg)` and the projector is training-only (internally consistent, but §1's "folded once" framing is decorative). The floor forces `β = ‖Π⊥z‖² ≥ 0.15` into the deployed descriptor. Paired simulation (common random numbers, calibrated to the proposal's own 0.69 baseline, `d−r = 424`, gallery 5000):

| deployed null block | ΔR@1 vs C4 (block dropped) |
|---|---|
| class-exchangeable, uniform β = 0.15 | **−0.4** |
| + per-image β spread sd 0.05 | **−2.2** |
| + per-image β spread sd 0.10 | **−6.5** |
| carries unseen-ID signal 0.02 | +0.7 |
| carries unseen-ID signal 0.05 | +2.3 |

Per-image β spread is *guaranteed*, not hypothetical: `L_f` is a one-sided hinge, so `β_i ≥ 0.15` with heterogeneity, and `√((1−β_q)(1−β_g))` then injects an identity-independent multiplicative bias per gallery image. Break-even needs null-block identity correlation ≥ ~0.02–0.03 — and `L_rate` under a 4-images/class sampler optimizes *toward* the harmful branch. **The expected deployment tax is the same order as, or larger than, the entire +1.6/+1.3 forecast gain.**

D1 as frozen has **no favorable branch**: near-chance ⇒ 15% of the descriptor is identity-irrelevant (the harmful column above); near-task ⇒ redundant, no gain. The proposal reads near-chance as reassuring ("nuisance-only"); it is the warning. D1 also probes *seen*-class accuracy, which cannot separate useful unseen-identity content from stable nuisance — the exact question. C4's falsifier F3 is written one-sided ("costs < 0.3 ⇒ dead") and cannot register the predicted outcome, that dropping the block **helps**.

### Item 6 — the only frontier-crossing arm is the arm where the theorem is definitionally void

15 proxies × 100 classes = 1500 unit vectors in `d = 512`. Measured: `rank(span) = 512/512` ⇒ **true null space dimension = 0**. The 95%-energy projector gives `r = 426`, so the "null" block is 86 dimensions **all of which lie inside the supervised proxy span** — it is exactly the discarded proxy tail. Installing a rate code there fights supervision head-on, and the item-1 tangential term becomes the entire story.

Three claims cannot hold simultaneously:
- **(a)** orthogonality theorem — requires `Π⊥p_c = 0`, i.e. `Π` = full-span projector;
- **(b)** nonempty complement under multi-proxy — requires `Π` ≠ full-span projector, since the span is all of `ℝ⁵¹²`;
- **(c)** the combined forecast 0.742/0.930, the *only* arm claimed to cross PFML.

(a) ∧ (b) is a contradiction. The standalone arm — the only one where the theorem is even arguable — is conceded in §5 to **not cross the frontier**.

There is also a §2↔§5 internal contradiction: §2 cites PFML's 15 proxies/class as *evidence for* the mechanism ("raises rank(Π) toward `d`"). But 1500 ≫ 512 saturates the span at `M ≥ 6`; PFML has, on this story, already fully solved the alleged problem, so NSRC-on-top has nothing to supply. The proposal's own explanation predicts the combined arm gains ≈ 0.

---

## Remaining items, in brief

**2 — descriptor-orthogonality does not transfer to parameters.** Even granting exact pointwise orthogonality at `h`: `⟨Jᵀu, Jᵀv⟩ = uᵀJJᵀv ≠ 0` unless `JJᵀ ∝ I`. At the head, single-sample gradients are `ug ᵀ` and *do* preserve orthogonality (`⟨ugᵀ, vgᵀ⟩_F = (uᵀv)‖g‖²`) — but batch-summed, cross terms `(u_iᵀv_j)(g_iᵀg_j)`, `i≠j`, are nonzero. One layer back, `⟨Wᵀu, Wᵀv⟩ = uᵀWWᵀv ≠ 0` for unconstrained trained `W`. AdamW's per-coordinate `m̂/(√v̂+ε)` is a nonlinear map that does not preserve orthogonality, and decoupled weight decay acts on the shared `W`. "Training decomposes into three mutually orthogonal channels" is false at the level that governs learning, independently of item 1.

**3 — not uniquely executable.** Stop-gradient semantics through `Π(p)` are never stated: include the dependence and orthogonality dies outright; exclude it and the update is a non-conservative field that is not the gradient of any scalar, voiding "the objective." The hard 95% threshold makes `r` an integer that jumps between recomputations — `d−r` changes, so `1/(d−r)` and `(d−r)/(Bκ)` shift discontinuously, `c`'s dimension changes, and content encoded in a direction inside `Π⊥` at step *t* can be inside `Π` at *t+50*, silently deleting the learned code. Near-degenerate eigenvalues at the boundary make `r` oscillate. No transport or rescaling is specified. *Correct subcomponent:* both null losses are invariant to eigenvector sign and to rotation within degenerate blocks (`L_inv` via `c → Rᵀc` with a shared basis across views; `L_rate` via the Gram) — verified, and a genuine soundness property.

**7 — novelty is a wrapper, not a supervision object.** Judged by object and action: coding-rate maximization in DML is **Anti-Collapse Loss (arXiv 2407.03106)**; the functional is **MCR²**; invariance + spread on two views is **VICReg / Barlow Twins / W-MSE**; allocating class-discriminative and SSL signal to different blocks of a *deployed* descriptor is **DiVA / MIC / Sharing Matters** (Milbich, Roth et al.); null-space gradient partitioning is **OGD / GPM / Adam-NSCL**. The prompt's "shared/private or orthogonally partitioned representations" is not confronted at all: **Domain Separation Networks** (Bousmalis et al., NeurIPS 2016) imposes an explicit orthogonality constraint between a shared task subspace and a private subspace carrying non-task content with a diversity term — the same supervision object, a decade earlier. NSRC's difference is that the split is a live eigen-decomposition rather than fixed coordinates: a wrapper. With the "zero interference" distinction gone (items 1–2), no mechanism-level novelty survives.

**8 — causal provenance is absent.** "Annihilates ~80% of directions" is a dimension count (`1 − r/d`), not an energy measurement and not an error measurement. "Explains PFML's hyperparameters" is a literature diagnosis, and per item 6 an incorrect one. **F6 is a *future* premise falsifier — an explicit admission that the founding premise (`E‖Π⊥z‖²` collapsing below 0.15 in the local corrected baseline) is unmeasured.** To establish causation one would need, on corrected checkpoints: (i) per-epoch `E‖Π⊥z‖²` and `r` in the corrected local baseline; (ii) error-linked evidence that for official queries the corrected baseline gets *wrong*, the correct gallery neighbour is recoverable from complement content and not from a random equal-dimension subspace; (iii) a class-disjoint transfer probe showing complement content predicts matches on *held-out training identities*, above both the `Π`-block and random-subspace controls; (iv) a same-seed corrected In-Shop paired control. None exist.

**9 — the control set cannot separate the hypotheses.** Missing: (a) **paired-view / no-null control** — NSRC uses 2 views of 64 images; the reported 0.690 baseline's sampler is asserted matched but never evidenced, so the comparison confounds the null loss with a unique-image-exposure change (64 vs 128 distinct images per step ⇒ either 2× the optimizer steps per epoch or half the data coverage, and altered `|P⁺|`/hard-negative pools in `L_PA`); (b) **no-floor control** — `λ_f = 10` is the largest weight and forces 15% complement energy *independently* of the rate/invariance terms; PA + `L_f` alone is unrun, and if it reproduces the gain the entire rate-coding story is irrelevant; (c) **full-span vs. truncated-span control** (`τ = 1` vs `τ = 0.95`) — the one axis that would expose item 1, and the one never varied; C3's random 412-dim subspace does not substitute, since it still overlaps the proxy span. F2 (C1 ≈ NSRC ⇒ mechanism dead) and §5's headline (C1-based combination wins) are mutually exclusive. No control distinguishes useful unseen-identity content from stable nuisance — and §2(e) concedes the background/context shortcut is "the top failure mode and is *not* fully defended," offering only D1, which cannot detect it.

**10 — protocol non-compliance.**
- **The mandated corrected paired In-Shop screen is absent.** There is no In-Shop arm; F4 pre-declares In-Shop/SOP gains ≤ ⅓. Worse, the frozen algorithm still builds a nonempty `Π⊥` there (bottom-5% eigenspace of a full-rank `M`, entirely supervised), contradicting §2's own "SOP has no null space." The required first screen lands in the regime the mechanism explicitly does not cover.
- **Baseline substitution.** §1 replaces the audited Lane-A reference with a self-declared reproduction (0.690/0.880), 4.4/4.7 points below PFML. Every "+1.6/+1.3" is a delta on a weak local recipe, not a frontier claim. BatchNorm trainability is never stated; sampler and LR schedule ("no decay") are changed. Repository-recipe matching is unverifiable as frozen.
- **No raw-best vs. independently-selected/final split.** Single numbers per arm. §1 pre-fixes `ε, τ, κ, λ_i, λ_r, λ_f, γ`, ramp bounds, recompute period — ~10 new knobs — while §6(b) *promises* class-disjoint selection that is neither run nor costed. At ±0.4 seed noise, a 10-knob search has ample capacity to manufacture +1.3–1.6.
- **Uncertainty arithmetic.** `σ_comb ≈ 0.007` is asserted; quadrature is not the paired-seed statistic. 1.2σ (CUB) / 0.4σ (Cars) against a reference with `σ = 0.003` and no local reproduction is not a crossing, as §5 half-concedes at 30–35%.
- **Envelope horizons unaddressed.** A general SOTA claim must confront VAPNet (CUB 0.762, Cars 0.948) and AdvRF (0.766, 0.949). The combined forecast 0.742/0.930 is below both; neither is mentioned.
- **The combined arm is undefined.** No multi-proxy loss is written anywhere in §1 — no variant, no `M`-assignment rule, no regularizer. Its unattributed +3.6/+3.7 contribution carries the entire frontier claim. Per protocol, an undefined composition cannot be credited.

**Cost recomputation — correct, and preserved.** `512³` eigendecomposition every 50 iters ≈ 1.3×10⁸ flops vs. ~1.5×10¹² flops/step for ResNet-50 at `B=128`; `B×B` Cholesky ≈ 7×10⁵; forming `ĈĈᵀ` = `B²(d−r)` ≈ 6.7×10⁶. **~1.00× train time and memory is right**, and deployment at `γ=1` is exactly the baseline. This is the proposal's most reliable claim. It does, however, conceal a *data-exposure* change (unique images per step halved), which is not a compute matter.

---

## Preserved correct subcomponents (independent of the verdict)

1. `∇_h cos(h,p) = (1/‖h‖)(p − s·ĥ)` — correct.
2. `∇_h L_null ∈ range(Π⊥)` and `⊥ ĉ`, hence `⊥ b` — correct; the *radial* half of the orthogonality claim holds.
3. Both null losses are invariant to eigenvector sign and to rotation within degenerate eigen-blocks — verified, a real soundness property and a non-trivial design choice.
4. `Π + γΠ⊥ = I` at `γ=1`, so deployment is exactly the baseline encoder: honest, no post-hoc knob.
5. FLOPs/memory ≈ 1.00× — recomputed and confirmed.
6. §5's own frontier arithmetic (+4.4/+4.7 needed; standalone does not cross; ~1.2σ/0.4σ; 30–35%) is arithmetically honest and self-limiting.
7. §6(a)'s insistence on paired 5-seed testing against ±0.4 seed noise, and §6(b)'s class-disjoint selection requirement, are correct protocol statements — they are simply not executed.
8. §2(e)'s concession that the background/context shortcut is undefended is accurate.

## Unresolved uncertainty (stated, not resolved)

- Item 1's tangential/radial ratio (2.23) is from my own construction with synthetic proxies and a plausible `w`-profile. The **existence and ≈5% scale** of the leak is exact — it is forced by `τ=0.95` and `tr(M)=1`, independent of proxy geometry. The precise ratio at a trained checkpoint is not measured here and is not needed: any nonzero tangential term voids both theorems as frozen and fires F5.
- Item 5's R@1 table is a stylized paired simulation, **not a repository measurement**. It establishes sign and order of magnitude, not the exact deficit.
- Prior-art citations (DSN, DiVA, MIC, MCR², VICReg, Barlow Twins, OGD/GPM/Adam-NSCL, neural collapse, Kornblith et al.) are from my own knowledge; I did not fetch PDFs, per the read-exactly constraint. The Anti-Collapse and MCR² identifications are the load-bearing ones and are corroborated by the frozen document's own §3 and source list.
- I did not attempt to determine whether `τ=1` plus a redesigned complement objective could be made to work. That would be a **new proposal**, and per instruction cannot upgrade this frozen object.

---

**Bottom line.** The proposal fails at Gate 1 on its own algebra: `τ = 0.95` guarantees ~5% proxy mass inside the block both theorems assume is proxy-free, the omitted tangential gradient term exceeds the retained radial one, and the frozen F5 falsifier is fired by the frozen mathematics. Three independent kills follow — an anti-collapse guarantee that evaluates to a bounded 0.399 penalty, a deployment argument whose sign is negative in the branch D1 is designed to confirm, and a frontier arm in which the true null space is provably empty. No implementation or GPU work is warranted.
