# Independent cold review of EFML — 2026-08-06

- Durable consultation: `c5c587dd0af944a6`
- Frozen review prompt: `docs/fable_efml_review_prompt_2026-08-06.txt`
- Frozen prompt SHA-256: `89afee6baf4121c00adcd46522af3dc9c0e0c0cb361643337f57800c2f88c02d`
- Caller attribution: `sfora/emafactorial`
- Provider path: Fable returned an out-of-credits message; the configured same-job Claude Opus fallback produced the complete review below.
- Durable job status: completed / exit 0.
- Result truncation: false.
- Preservation: exact durable `result` field; no local edits to reviewer text.

You're out of usage credits. Run /usage-credits to keep using Fable 5 or /model to switch models.
I reviewed this cold against primary sources. Verdict below.

---

# Independent adversarial review — EFML (frozen Pass 24)

## Verdict: **DEAD**

---

## 1. The single strongest decisive reason

**EFML's two operative terms jointly fix every class's fiber radius at a constant frozen at epoch 5 and forbid its reduction — and by the proposal's *own* error model in §2, that converts a per-class failure into a universal one.**

This is internal; no experiment is needed.

§2(C) states the failure condition explicitly: class B with fiber radius Δ and a neighbour at center distance γ fails when `2Δ > γ − Δ`, i.e. **Δ > γ/3**. §2(B) states exactness needs **γ > 4R**. So the proposal itself identifies `max_c Δ_c / γ_min` as the causal quantity.

Now apply EFML to it:

- `L_hom` at its minimum makes `F_c ≡ F` for all c. A single common law has a single support radius `Δ*`, with `Δ*² ≥ E‖ξ‖² = m*`.
- `L_sc = (m̂/m* − 1)²` is a **two-sided** squared penalty on the batch mean of `‖ξ_i‖²`, with `m*` frozen from the last warm-up epoch — an epoch in which, per §1.1, only the head and proxies train and BN is frozen. So `m*` is a property of a linear head on a frozen ImageNet backbone.
- The pin binds. Once PA's margins are met its gradient decays as `e^{−α(cos φ − δ)}` with α=32, δ=0.1: at φ = 0.1 rad that factor is ≈4×10⁻¹³, while `∂L_sc/∂m̂ = 2λ_sc(m̂/m*−1)/m*` is O(1). The pin's gradient dominates by ~12 orders of magnitude, so `m̂` is held at `m*` for the remaining ~55 epochs.

Consequence: equalization is a **levelling-up**. Classes that were tight are raised to `Δ* ≥ √m*`. Before EFML, only loose classes satisfy `Δ > γ/3`. After EFML, *every* class with a neighbour closer than `3Δ*` satisfies it — including classes that were previously safe. Nothing in the construction ties `m*` to `γ_min`, and `γ_min` is smallest exactly on SOP (11,318 classes) and In-Shop (3,997 classes) on S⁵¹¹ — the two datasets where the crossing is claimed.

The dilemma is closed on the other branch too. If one argues λ_sc = 0.5 is too soft to bind: `L_sc → 1` as `m̂ → 0`, so the *entire* penalty for total per-class collapse onto proxies is bounded by **0.5 nats**, while `L_hom = 0` exactly at collapse (all ξ = 0 ⟹ all projected laws identical). A bounded barrier cannot exclude a degeneracy. §2(i)'s claim — "*Collapse* (F=δ₀ satisfies equality): excluded by L_sc with m*>0" — is therefore **false as written**; and collapse onto proxies is precisely the argmin of L_PA, so it is the direction the base loss is already travelling. (At that point σ = median pairwise distance in Q → 0 and the kernel is 0/0 — see §11.4.)

Either the pin binds (fiber radius frozen at a warm-up value, which the proposal's own inequality says makes zero-shot NN worse), or it does not (the stated degeneracy defense is void). There is no third branch.

---

## 2. The gauge is not pole-invariant, and its invariant content is only isotropy (question 2)

The minimal rotation `p̂_c → o` is `R = I − (a+b)(a+b)ᵀ/(1+⟨a,b⟩) + 2baᵀ`, `a = p̂_c`, `b = o`. It is smooth on S⁵¹¹∖{−o} (a punctured sphere is contractible, so a coherent frame field *does* exist there — the proposal is right that this is executable) and it correctly maps `T_{p̂_c}S → T_oS`. Two defects follow.

**(a) The singular branch is unhandled.** `1 + ⟨a,b⟩ → 0` at `p̂_c = −o`. The only stated numerical guard is a clamp on *the log map's* cosines; nothing guards `R_c`. The gauge frame varies arbitrarily fast for any proxy near the antipode.

**(b) The constraint is pole-dependent — decisively.** Let `o' = Qo`. Then `R^{o'}_a ≠ Q R^{o}_a`: `R^{o}_a` is supported on `span{a, o}`, and `Q R^o_a` on `span{Qa, Qo}`, whereas the minimal rotation to `o'` is supported on `span{a, Qo}`. Define `H_a = (R^{o'}_a)^{-1} Q R^{o}_a`. It maps `a → o → o' → a`, so it is a rotation fixing `a` whose angle is the holonomy of the geodesic triangle `(a, o, o')` — by Gauss–Bonnet, the enclosed spherical area. It is **class-dependent**. Hence

> equality of `{R^o_c ξ_c}` across c ⟺ equality of `{R^{o'}_c ξ_c}` across c **only if** the common law is invariant under every `H_c`.

Varying `o'` over the sphere generates single-plane rotations of `T_a` in all 2-planes with generic angles; that group is dense in SO(511). Therefore:

**The pole-invariant content of EFML's constraint is exactly "every class's transported fiber law is isotropic with a common radial profile."** Everything else it enforces is an artifact of one frozen random draw of `o` — a demand that each class's covariance *orientation* be aligned to a random frame in a way that depends on where its proxy happens to sit. That has no semantic content, is unlearnable at unseen base points (an unseen class sits at a position whose `R_q` was never exercised), and is the direct route to proxy/gauge gaming: `R_c` is a function of `p̂_c` alone, so moving the proxy rotates the gauge. §2(iv)'s defense ("proxies simultaneously serve L_PA") does not rule this out; it only says the gaming is constrained, not blocked.

The advertised object — "the transported within-class displacement distribution must be identical across classes" — is thus either (i) not well-defined as a geometric statement, or (ii) equal to class-independent *isotropic* within-class noise with a common scale. Reading (ii) is the DVML premise (see §6).

---

## 3. The enforced object is not the advertised object (questions 1, 2)

`ξ ∈ T_oS ≅ ℝ⁵¹¹`. `P_J ∈ ℝ^{64×512}` is **frozen at init**. Restricted to `T_oS`, its kernel has dimension **447**.

`MMD²(P_J#F_c, P_J#F_pool) = 0` implies equality of the 64-D pushforwards. It does **not** imply `F_c = F_pool`. Cramér–Wold requires all directions; a single fixed 64-dim subspace does not suffice, and Johnson–Lindenstrauss is a statement about approximate preservation of pairwise distances for a **finite point set chosen independently of the projection** — an assumption violated by construction here, since the encoder is trained against a fixed `P_J` for 55 epochs. The cheapest descent direction on `L_hom` is to push class-specific structure into `ker(P_J)`.

The only other constraint on those 447 directions is `L_sc`, which is a **global batch mean** over all classes in the batch, computed in full 512-D. So:

- per-class heterogeneity in 447 of 511 tangent directions is unconstrained;
- classwise collapse and classwise inflation trade freely against each other under a global mean;
- `L_sc` and `L_hom` live in **different spaces** (512-D radii vs 64-D projections), which makes radius trading between the visible and invisible subspaces free and unpenalized.

The claim in §1.2/§2 that the method constrains "the transported within-class displacement distribution" in the 511-D fiber is therefore not supported by the objective as specified. E\* is the one diagnostic that could expose this — and F5's threshold is only 15%, energy distance is itself a distance-based statistic subject to the same high-dimensional power loss ([Ramdas et al., AAAI 2015](https://arxiv.org/abs/1406.2083)), and E\*'s definition conflicts with the training protocol (§9).

---

## 4. Estimation: the statistic cannot see what it claims to see (question 3)

Per-class sample sizes, from the official splits:

| Dataset | train classes | train images | **images/class** | `A_c` = live + buffer |
|---|---|---|---|---|
| CUB | 100 | 5,864 | 58.6 | 4 + 16 = 20 |
| Cars196 | 98 | 8,054 | 82.2 | 4 + 16 = 20 |
| SOP | 11,318 | 59,551 | **5.3** | 2 + 8 = 10 |
| In-Shop | 3,997 | 25,882 | **6.5** | 2 + 8 = 10 |

Three consequences, all concentrated on the two datasets carrying the crossing claim:

1. **The buffer exceeds the class.** `B_c` holds 8 entries for classes that own ~5–6 images *in total*. `A_c` is therefore not 10 samples from a distribution; it is ≤6 distinct images, several present as stale duplicates from earlier epochs. There is no "law" to match. The proposal's Risk (6) names M=2 as making MMD "noisy" and claims buffers mitigate it; it never notices that the buffer cannot exceed the class size.

2. **The term is repulsive among same-class samples.** `MMD²(A_c, Q̃)` contains `(1/m²)Σ_{i,j} k(u_i,u_j)`; minimizing it spreads `A_c`. The live samples carry gradient, the buffer entries are detached, so the cross terms push each live sample *away from stale embeddings of its own class's images*. On SOP/In-Shop that is a repulsion between the ~5 images retrieval must rank first, and a penalty on augmentation-invariance and on temporal stability of the encoder. This holds for both the biased V-statistic and the unbiased U-statistic (removing the i=j diagonal does not remove the live↔buffer cross terms).

3. **The LSE selects on bias, not deviance.** §1.2 claims the log-sum-exp "focuses on the most deviant classes." With |A_c| ≈ 10–20 in 64 dimensions, the estimator's O(1/|A_c|) bias/variance is ≈0.05–0.1, plausibly larger than the true between-class spread of MMD². The ranking of `MMD²_c` is then noise-dominated, and if the biased estimator is used the bias is a monotone function of |A_c| — so the LSE preferentially weights whichever class contributed *fewest live samples*. The estimator is never specified (§11.5).

Separately, MMD power against "fair" alternatives drops polynomially with dimension, and the median heuristic is specifically implicated ([Ramdas et al. 2015](https://arxiv.org/abs/1406.2083)). At 64 dimensions with a 10-point sample, "equality of the fiber law" is not identified by this statistic under any reading.

---

## 5. The causal step does not close (question 4)

The exchangeability lemma §2(A) establishes identity-*independence* of ranking statistics. That is not accuracy. Bound (B) and failure condition (C) both depend on `max_c Δ_c` versus `γ_min`. Equalization is silent on `max_c Δ_c` — it can only raise the minimum, and with the pin fixing the common level it *does* raise it (§1). So cross-class law equality is not merely insufficient for repairing unseen-class NN errors; in the proposal's own model it is the wrong sign.

Two further gaps:

- **Nothing transfers the constraint to unseen classes.** `L_hom` is imposed on the 100 (or 11,318) *training* classes. An encoder with sufficient capacity can equalize fiber laws conditioned on training identity while behaving arbitrarily at novel base points — and §2's gauge makes this easier, because the required frame `R_q` at an unseen position was never exercised. No argument or measurement addresses this; §2(iii) explicitly acknowledges the copula leak but the transfer gap is broader than the copula.
- **§2(ii)'s "cosmetic scatter" defense is an assertion.** "The only image factors available with that property are generic nuisance factors" is unargued, and the specification actively rewards cosmetic scatter: PA saturates exponentially once margins are met, so injecting nuisance-driven, identity-irrelevant displacement into the 64 visible directions is nearly free on `L_PA` and directly reduces `L_hom` and `L_sc`.

Direct contrary evidence exists in the primary literature. [IAA (Zhu et al., IEEE T-MM 2022)](https://arxiv.org/abs/2211.16264) introduces *neighbor correction* precisely because, empirically, "similar classes generally have similar variation distributions" — i.e. intra-class variation is class-dependent and best estimated **locally**, not from a global pool. EFML's pooled-target equalization would erase exactly the structure a published method found useful. The proposal cites neither IAA nor this finding.

---

## 6. Prior art: novelty is thin, and the closest neighbour is mischaracterized (question 5)

**NIR is described falsely.** §3 says NIR "never compares residual laws *across* classes... and uses no transport gauge." [NIR (Roth et al., CVPR 2022)](https://openaccess.thecvf.com/content/CVPR2022/html/Roth_Non-Isotropy_Regularization_for_Proxy-Based_Deep_Metric_Learning_CVPR_2022_paper.html) optimizes `L_NIR = (1/|B|)Σ‖τ⁻¹(ψ(x)|ρ_{y_x})‖²₂ − log|det J_{τ⁻¹}|`: a **proxy-conditioned invertible transport** `τ⁻¹` carrying every class's sample-to-proxy residual into a **single common base law** N(0,I), with gradient flowing into the embeddings. That is a transport gauge, and matching all classes to one common reference *is* a cross-class comparison of residual laws in a common frame. EFML is structurally NIR with the learned proxy-conditioned bijection replaced by a fixed proxy-conditioned *rotation*, and MLE-against-N(0,I) replaced by MMD-against-a-pooled-empirical-sample. The honest residual distinction — NIR's conditioning can absorb class-dependent structure, EFML's rotation cannot — is real but far narrower than the one claimed, and it is not the distinction the proposal states.

**SFT already occupies the geometry.** [SFT (Zhu, Bai, Wei, ECCV 2020)](https://link.springer.com/chapter/10.1007/978-3-030-58529-7_25) relaxes identical-covariance-between-classes to *similar covariances on a hypersphere, similarity measured by equivalence of the covariance eigenvalues*, and performs the transform by **a rotation that respects the spherical distribution**. Hypersphere + rotation transport between class positions + cross-class covariance equality-up-to-rotation is SFT's assumption set. §3's "enforce rather than assume" is a legitimate distinction, but it is incremental.

**DVML is the invariant content.** [DVML (Lin et al., ECCV 2018)](https://link.springer.com/chapter/10.1007/978-3-030-01267-0_42) rests on the observation that "the distribution of variance within classes is independent on classes," and forces the conditional intra-class variance to be **isotropic multivariate Gaussian**. Per §2 above, the pole-invariant content of EFML's constraint is precisely *class-independent isotropic intra-class variation with a common scale*. That is the same object, enforced by a penalty instead of assumed by a prior — a smaller delta than §3 claims ("generation-free, spherical, and enforces rather than assumes").

I did not find a primary source enforcing cross-class equality of transported within-class distributions as a train-time DML penalty. On my search the prior-art gate is *thin but survivable*. It is not what kills this proposal.

---

## 7. Controls do not isolate the claimed mechanism (question 6)

C1–C7 are better than typical, and C6/C7 are well chosen. Five gaps, each fatal to a specific reading:

- **No gauge-only ablation.** C4 deletes the log map, the rotation, and the sphere simultaneously. The rotation `R_c` is the novel geometric ingredient; F4 ("Euclidean C4 within 0.2 → geometric gauge unnecessary") would misattribute a log-map effect to the gauge, or vice versa.
- **No pole-resampling control.** The direct test of §2's objection — rerun with a different frozen `o` — is absent. If results move with `o`, the constraint is an artifact of a random draw.
- **No fixed-arbitrary-reference control.** C5's own-EMA-buffer placebo is a *temporal-consistency* regularizer, not a matched-strength stand-in. The sharp control is matching every class to a fixed isotropic reference — that isolates "equal to *each other*" from "equal to *some* common target," which is the actual exchangeability claim. F3 can pass for the wrong reason.
- **No projection control.** Neither `P_J = I` (full-dimensional MMD) nor per-step resampled projections appear, so the 447-dim null-space evasion of §3 is untestable within the control set.
- **No floor-vs-pin control.** `L_sc` conflates "block collapse" (one-sided) with "hold dispersion at the warm-up value" (two-sided). C2 tests the composite only.

**P0 is near-vacuous.** A ≥1.5× P90/P10 ratio in per-class scatter is essentially guaranteed for any real embedding, so it cannot falsify. And it measures **scalar** heterogeneity — exactly what C6 already addresses — so passing P0 licenses the scalar variant, not the full-law claim.

---

## 8. Forecast and frontier arithmetic (question 7)

**No forecast derives from a measured premise or a reproduced base.** P0 is unrun. The PA-repro rows are themselves forecasts. The Δ column (+2.4, +1.9, +2.6, +1.5) has no stated derivation from anything. Criterion 7 fails on its face.

**The probabilities contradict the stated uncertainties by orders of magnitude.**

| | forecast | reference | gap | pooled sd (per-seed) | z | P(single) | P(5-seed mean) | **stated** |
|---|---|---|---|---|---|---|---|---|
| CUB | 71.8±0.5 | 73.4±0.3 | 1.6 | 0.58 | 2.75 | ~0.003 | ~3×10⁻⁵ | **0.10** |
| Cars | 89.2±0.5 | 92.7±0.3 | 3.5 | 0.58 | 6.0 | ~10⁻⁹ | ~10⁻³⁶ | **0.02** |
| In-Shop | 93.2±0.3 | 93.0 | −0.2 | 0.3 | −0.67 | 0.75 | 0.93 | **0.60** |

Either the ± are meaningless or the P's are; they cannot both be the stated quantities.

**SOP is internally inconsistent.** The table gives Δ = 82.2 − 79.6 = **+2.6** and a requirement of +3.3, so the deficit is **−0.7**. The prose asserts "Δ = +2.6–3.0, hence deficit −0.3," quoting a +3.0 that appears nowhere else and taking the optimistic end.

**The load-bearing In-Shop margin is a fabricated constant.** The "best ≈" column adds exactly **+0.4** to all four EFML forecasts (71.8→72.2, 89.2→89.6, 82.2→82.6, 93.2→93.6), while the PA-repro best-vs-final offsets are +0.5/+0.4/+0.2/+0.4 — including +0.2 for SOP against +0.4 for EFML on the same dataset. A uniform best-minus-final gap across four datasets differing by two orders of magnitude in class count is an invention, and it is exactly what produces the quoted "margin +0.6" (93.6 − 93.0). At the final-epoch convention the margin is +0.2.

**Scope.** By the proposal's own table, 3 of 4 datasets are forecast sub-frontier by −1.6 (CUB), −3.5 (Cars), −0.7 (SOP) — I confirmed the reference row: PFML reports R50 R@1 of 73.4 / 92.7 / 82.9 ([Bhatnagar & Ahuja, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Bhatnagar_Potential_Field_Based_Deep_Metric_Learning_CVPR_2025_paper.html)). The sole claimed crossing is In-Shop by +0.2 (final-epoch), on top of an **invented** PA-R50/512 In-Shop anchor (92.1 ± 0.5, by the proposal's own admission), against [PA+DADA's 93.0](https://arxiv.org/html/2401.00617v1) — which I confirmed is a single run with no seeds or variance reported. A ±0.5 invented anchor swamps a +0.2 margin.

**The pre-registered falsifiers do not test the legal task.** F1 lets the method survive by beating its own baseline by +1.0 on two datasets while crossing zero frontiers. F2 is a conjunction, so In-Shop ≥ 93.0 alone preserves the frontier claim regardless of SOP; and its SOP threshold (82.4) sits below both the proposal's own best-epoch forecast (82.6) and the frontier (82.9), so a SOP result of 82.5 is simultaneously "no crossing" and "not falsified."

---

## 9. Protocol, tuning, capacity (question 8)

The disclosure here is genuinely good and I want to record it: §1.1 correctly reports that the official repo gives ResNet-50/512 recipes only for CUB (`--warm 5 --bn-freeze 1 --lr-decay-step 5`, batch 120, lr 1e-4) and Cars (same, decay 10), with R@1 69.9 / 87.7, and documents **Inception-BN only** for SOP and In-Shop. I verified all of this against the repository. Matching the class-balanced sampler in PA-repro correctly pre-empts the sampler confound. Cost accounting is right and, if anything, conservative: the A↔Q̃ kernel work is ≈20M MACs/step against ≈246 GMACs for the ResNet forward at batch 120 — ~0.008%, and the SOP buffer footprint is ≈12 MB in fp16.

Two protocol defects remain:

- **The validation protocol contradicts the mechanism metric.** §1.2 tunes λ on a 15% class-disjoint split carved from training classes, "then retrain on full train." §4 defines E\* — the pre-registered mechanism metric and F5's falsifier — on "held-out training-split classes." After retraining on the full training set, no held-out training classes exist. E\* is therefore either measured on a different model than the one reported (so F5 does not test the reported model) or measured on classes the final model trained on (contaminated). Underdefined, and it is the one measurement that could detect the null-space evasion of §3.
- **`m*` is set under a different capacity regime than it is enforced in.** Warm-up trains head+proxies with BN frozen; `m*` is thus a statistic of a frozen ImageNet backbone, then imposed on a fully fine-tuned model for the remaining ~55 epochs. No argument ties that value to anything.

---

## 10. Complete list of false, inconsistent, or underdefined operations

1. **False** — §2(i): "Collapse... excluded by `L_sc` with m*>0." `λ_sc·L_sc → 0.5` as `m̂ → 0`; a bounded penalty excludes nothing.
2. **False** — §3: NIR "never compares residual laws *across* classes... uses no transport gauge." NIR's loss transports every class's residual through a proxy-conditioned bijection into one shared N(0,I).
3. **False/overstated** — §1.2: the LSE "focuses on the most deviant classes." At |A_c| = 10–20 in 64-D the ranking of `MMD²_c` is estimator-noise-dominated; under the biased estimator it tracks |A_c|.
4. **Underdefined** — MMD estimator: biased V-statistic vs unbiased U-statistic never specified. Biased ⟹ an |A_c|-dependent O(1/m) bias drives the LSE selection; unbiased ⟹ the estimator can go negative and minimizing it fits its own noise.
5. **Underdefined** — kernel bandwidth: `k_σ` is never written out; "median pairwise distance" leaves σ vs σ² vs 2σ² and median-of-distances vs median-of-squared-distances open (a factor of 2 in the exponent).
6. **Unguarded** — σ refresh every 200 steps, with no bound on drift. Too-large σ ⟹ `k ≈ 1` and MMD ≈ 0, silently switching the constraint off; too-small σ ⟹ `k ≈ 0` off-diagonal and zero gradient. Both are most likely right after epoch 6, when the backbone unfreezes. `σ → 0` under compaction is a 0/0.
7. **Unguarded** — `R_c` singularity at `p̂_c = −o`: `1/(1+⟨p̂_c,o⟩)` diverges; the stated clamp applies only to the log map's cosines.
8. **Wrong guard** — Log map: the branch is stated as "0 if φ=0," an exact-equality test on φ, while the numerically singular quantity is `‖u‖ = sin φ` in `u/‖u‖`. The analytic `φ·O(1/φ)` cancellation is exact but is not realized in floating point, and no Taylor/ε branch is specified. The antipodal case `‖u‖ = 0` at φ=π is not handled at all.
9. **Gradient claim unmet** — clamping cosines to ±(1−10⁻⁷) zeroes the gradient through φ at saturation rather than bounding it.
10. **Gauge-dependent** — the constraint's solution set changes with the frozen random pole `o` by a class-dependent holonomy `H_c`; its pole-invariant content is only isotropy plus a common radial profile (§2).
11. **Space mismatch** — `L_sc` uses 512-D `‖ξ‖²`, `L_hom` uses 64-D `P_Jξ`. Radius trading between the visible and invisible subspaces is free.
12. **Scope mismatch** — `L_sc` pins a **global batch mean**, so classwise collapse and classwise inflation cancel; per-class dispersion is constrained only through the 64-D projection.
13. **Stale gauge mixing** — `B_c` and `Q` store `u = P_J R_c Log_{p̂_c}(ẑ)` computed under *old* proxies, hence old frames. The "common tangent gauge" premise is violated by the estimator itself, and no re-gauging is specified. On SOP a class is visited ≈2.6×/epoch, so an 8-entry `B_c` spans ~3 epochs.
14. **Buffer exceeds class** — `B_c` = 8 for SOP (5.3 images/class) and In-Shop (6.5), so `A_c` is duplicates of ≤6 images, not a sample from a law.
15. **Wrong-sign gradient** — the within-`A_c` kernel term repels live samples from detached same-class entries: an explicit anti-augmentation-invariance, anti-temporal-stability force on the exact image pairs SOP/In-Shop retrieval must rank together.
16. **Protocol contradiction** — E\* is defined on held-out training classes; the final model retrains on all training classes (§9).
17. **Arithmetic inconsistency** — SOP Δ is +2.6 in the table and "+2.6–3.0" in the prose; the stated deficit (−0.3) uses only the unsupported upper end; from the table it is −0.7.
18. **Fabricated constant** — a uniform +0.4 best-vs-final offset applied to all four EFML forecasts, inconsistent with the same paragraph's PA-repro offsets (+0.5/+0.4/+0.2/+0.4), and load-bearing for the only claimed crossing.
19. **Invalid probability arithmetic** — stated crossing probabilities are inconsistent with the stated ± by 1–34 orders of magnitude (§8 table); the ± are per-seed while crossings are judged on 5-seed means.
20. **Near-vacuous premise check** — P0's ≥1.5× P90/P10 scatter ratio will pass for essentially any embedding, and tests scalar heterogeneity, which control C6 already covers.
21. **Misaligned falsifiers** — F1 and F2 permit survival with zero frontier crossings; F2's SOP threshold (82.4) lies below both the proposal's own best-epoch forecast (82.6) and the frontier (82.9).

---

## 11. What survives

The base-recipe audit, the sampler-confound handling, the cost accounting, the final-epoch convention, the refusal to inflate CUB/Cars forecasts, and the C6/C7 controls are all sound work. A run of C1 vs C6 vs C7 would be informative regardless of this verdict.

But the headline mechanism does not survive its own §2. The constraint EFML enforces is not the one it advertises (447 invisible tangent directions, a global-mean pin, ≤6-image "distributions" on the crossing datasets); its geometric content is pole-dependent and reduces on the invariant part to DVML's class-independent isotropic intra-class variance; the closest prior art is mischaracterized in the direction that inflates novelty; the causal step runs backwards through the proposal's own inequality `Δ > γ/3`; and the sole claimed frontier crossing is +0.2 over a single-run reference, built on an invented baseline anchor with ±0.5 uncertainty and a fabricated +0.4 convention offset.

Any of the repairs implied above — a one-sided floor instead of a two-sided pin, per-step resampled projections or a sliced estimator, per-class rather than global dispersion control, a pole-invariant (isotropy-only) formulation, dropping SOP/In-Shop where classes own 5 images — would be a **new proposal**, and I make none of them here.

Sources: [Proxy-Anchor (Kim et al., CVPR 2020)](https://openaccess.thecvf.com/content_CVPR_2020/html/Kim_Proxy_Anchor_Loss_for_Deep_Metric_Learning_CVPR_2020_paper.html) · [Proxy-Anchor official repo](https://github.com/tjddus9597/Proxy-Anchor-CVPR2020) · [NIR (Roth et al., CVPR 2022)](https://openaccess.thecvf.com/content/CVPR2022/html/Roth_Non-Isotropy_Regularization_for_Proxy-Based_Deep_Metric_Learning_CVPR_2022_paper.html) · [SFT (Zhu et al., ECCV 2020)](https://link.springer.com/chapter/10.1007/978-3-030-58529-7_25) · [DVML (Lin et al., ECCV 2018)](https://link.springer.com/chapter/10.1007/978-3-030-01267-0_42) · [IAA (Zhu et al., IEEE T-MM 2022)](https://arxiv.org/abs/2211.16264) · [PFML (Bhatnagar & Ahuja, CVPR 2025)](https://openaccess.thecvf.com/content/CVPR2025/html/Bhatnagar_Potential_Field_Based_Deep_Metric_Learning_CVPR_2025_paper.html) · [DADA (Ren et al., AAAI 2024)](https://arxiv.org/html/2401.00617v1) · [Ramdas et al., AAAI 2015](https://arxiv.org/abs/1406.2083)

