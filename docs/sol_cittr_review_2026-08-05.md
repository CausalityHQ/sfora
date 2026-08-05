# Independent review of frozen Pass-18 CITTR proposal

- Date: 2026-08-05
- Durable consultation: `ef371910a1fb47d9`
- Caller: `sfora/emafactorial`
- Provider path: Fable began the consultation; the configured same-job fallback completed it with Claude Opus after Fable failed.
- Frozen proposal: `docs/sol_cittr_proposal_pass18_2026-08-05.md`
- Frozen review prompt: `docs/sol_cittr_review_prompt_2026-08-05.txt`

The text below is the completed independent result, preserved before local adjudication.

I'll review this frozen proposal cold, using only web search and primary literature. Let me start by loading the search tools, then verify the math by hand, the PFML baseline fidelity, and the prior-art landscape.
# Verdict: **DEAD**

Reviewed cold against primary sources. The spherical geometry is correct; the proposal fails on causal identifiability, occupied mechanism, and unreachable/vacuous frozen operations.

---

## Decisive reason

**At the frozen constants, `L_CITTR` is dominated by a plain γ = 0.05 angular proxy margin on the receiver's *real* descriptor, and no control in the frozen set removes that term — so a positive result is not attributable to cross-identity tangent transport even if every forecast lands.**

Substituting the exp map into the hinge (v ⊥ z, ‖v‖ = 1):

$$h(q,y)=\Big[\gamma-\cos\varepsilon\,\Delta(z)+\sin\varepsilon\,\big(v^\top p_n-v^\top p_y\big)\Big]_+$$

with ε = 0.20 (see §1 below — ε is saturated, not measured): cos ε = 0.9801, sin ε = 0.1987. For a random unit tangent in 511 dims, `v·p ~ N(0, 1/511)`, so `v·(p_n − p_y)` has σ ≈ 0.063 and `max` over K = 4 gives ≈ 0.064. The transport term contributes **≈ 0.013** against a base of **γ = 0.05**: a ~25 % inflation of an effective cosine margin of 0.051. For the transported direction to be *first-order* rather than a perturbation, it needs `|v·(p_n − p_y)| ≳ 0.25`, i.e. alignment ≳ 0.18 with the receiver's own hardest proxy-difference direction — **≈ 4× chance (1/√511 = 0.044)**, and never measured.

The frozen control set has no ε = 0 control, no γ-matched control, and no γ sweep. Control 1 deletes `L_CITTR` *including* the margin term; controls 2–6 all retain it. So:

- if the gain is pure margin, CITTR ties random-tangent → declared falsified (conservative, fine);
- **if CITTR beats random-tangent because its effective margin is 0.10 rather than 0.064, the proposal declares the mechanism confirmed when the only difference is margin magnitude.** That false-positive path is uncontrolled.

`L_CITTR` also backpropagates into proxies ("updates … selected positive/negative proxies"), which is precisely the "extra proxy logits" confound the rubric names. Adding a γ-matched or ε = 0 arm is a substantive repair and cannot be credited.

---

## 1. Equations, stop-gradients, sampler, cost

**Correct as written:**

- Log map. `Log_a(b) = ϑ/√(1−(aᵀb)²)·(b−(aᵀb)a)`; since `‖b−(aᵀb)a‖ = sin ϑ`, this is standard and `‖u‖ = ϑ`. ✓
- Parallel transport. `P(û) = û − (ûᵀb)/(1+aᵀb)(a+b)`. Verified: `bᵀP = 0` (tangent at b) and `‖P‖ = ‖û‖` (isometry, using `‖a+b‖² = 2+2aᵀb`). ✓ This is exactly the rotation in span{a,b} carrying a→b, identity on the orthogonal complement — see §3.
- Exp map `cos ε·z + sin ε·v`. ✓ Both "renormalize" steps are no-ops (transport is an isometry; v ⊥ z exactly since `b_j = sg(z_j)`).
- Gradient paths are unambiguous and the stop-gradient set is complete.

**Broken or vacuous:**

**(a) The isotropic-fallback branch is unreachable — dead code.** ϑ is `arccos` of a value clipped at `1−10⁻⁵`, so `ϑ ≥ arccos(1−10⁻⁵) ≈ 4.47×10⁻³`. Since `‖u_i‖ = ϑ`, the guard `‖u_i‖ < 10⁻³` **can never fire**. The degeneracy defence "a zero measured tangent triggers an isotropic tangent intervention" rests on a branch that cannot execute. (The ε ≥ 0.05 floor independently blocks shrinkage, so the *conclusion* survives — the stated *mechanism* does not.)

**(b) Clipping asymmetry is a NaN hazard the guard cannot catch.** The numerator clips `aᵀb`; the denominator `√(1−(aᵀb)²)` does not. At `aᵀb = 1` (or fp32 `aᵀb > 1−5×10⁻⁹`, where `1−c²` underflows), the denominator is 0 → NaN. `NaN < 10⁻³` is false, so the fallback still does not fire, and NaN propagates into q → h → every gradient. The guard is written as a magnitude test; the actual failure mode is non-finite.

**(c) ε is not measured — it is the constant 0.20.** `ε = clip(ϑ, 0.05, 0.20)` leaves the ceiling only when the two augmented views satisfy `cos ϑ ≥ 0.980`. Two independent RandomResizedCrop(scale 0.16–1) + jitter + grayscale views of one image, through a supervised proxy-trained embedding never optimized for two-view invariance, land nowhere near 0.980 (typical two-view cosine ≈ 0.5–0.9, ϑ ≈ 0.45–1.05 rad). **Both clip endpoints therefore bind on one side, always: ε ≡ 0.20.** The radius is a fixed hyperparameter, not augmentation-grounded; only the direction is measured. The listed risk "the fixed minimum radius can over-regularize" describes the wrong endpoint.

**(d) SOP proxy count implies test identities.** SOP training split is 11,318 classes; at M = 2 that is **22,636** proxies. The frozen text states "about 45,000 proxies" = 22,634 × 2 = *all* SOP classes, train **and** test. Either a 2× cost error or an implied proxy table over held-out identities. Either way the cost section is wrong; on the second reading it is a protocol violation.

**(e) The `aᵀb ≤ −0.95` resample guard is inert** in 511 dims for a non-degenerate embedding. Harmless.

**Cost forecast otherwise plausible:** 256 synthetic × 22,636 × 512 ≈ 6 GFLOP fwd vs ≈ 790 GFLOP for 64 ResNet-50 fwd+bwd. 1.04–1.08× is defensible. Compute matching at 200 N views is internally consistent.

**Not verified:** the CVF PDF returned HTTP 403 and the arXiv abstract carries no method detail, so I could **not** independently confirm PFML's `(δ/d)^α` form, δ = 0.2 / α = 4, M = 15 / M = 2 proxy counts, or the 0.734 / 0.927 / 0.829 ± figures against the primary text. Those are taken as given from the prompt and flagged as unaudited here.

---

## 2. Is the donor tangent class-exogenous and valid after transport?

**Not established, and there is a specific reason to doubt it.** RandomResizedCrop at scale ∈ [0.16, 1] on CUB/Cars routinely crops to a *part* — a beak, a wing, a wheel arch. The resulting displacement encodes a part-vs-whole content change, and on fine-grained benchmarks *which parts are visible and how they look is the identity signal*. Such a direction is class-**endogenous**, not nuisance. Transported onto identity j, `q` can be a mislabelled point, and the hinge asserts `y_j` at it — label-noise injection.

This creates a squeeze the proposal does not escape: small ε ⇒ the intervention is second-order (the decisive reason above); large ε ⇒ label corruption. The frozen ε ≡ 0.20 (cos ε = 0.98) sits firmly in the negligible half.

The proposal's own text concedes the premise: *"The proof does not establish coverage of real test nuisances."* **No measurement of any kind is offered** — no donor/receiver tangent-alignment statistic, no held-out check that `q` retains identity `y_j`, no estimate of the alignment scalar the whole mechanism depends on. Rubric item 6's first clause fails outright.

The "sampled-direction certificate" in §2 is stated correctly (max of non-negative hinges ⇒ every sampled direction, not the average) and is honestly labelled as not covering test nuisances. It is sound and load-bearing on nothing.

---

## 3. Mechanism equivalence

**Yes — to spherical cross-class variation transfer, in DML, on these benchmarks.**

[Spherical Feature Transform for Deep Metric Learning (Zhu, Bai, Wei, ECCV 2020)](https://arxiv.org/abs/2008.01469) performs feature augmentation in which *"feature transform between different classes is performed via a rotation that respects the spherical data distributions,"* explicitly *"relax[ing] the assumption of identical covariance between classes to an assumption of similar covariances of different classes on a hypersphere,"* evaluated on DML benchmarks including Cars196.

CITTR's transport `û − (ûᵀb)/(1+aᵀb)(a+b)` **is** that rotation: it is the Givens rotation in span{a, b} carrying a→b, acting as the identity on span{a,b}^⊥ (immediate from the formula when `ûᵀb = 0`). SFT transfers a covariance; CITTR transfers one paired displacement. That is a parameterization difference within an identical mechanism — exactly the distinction class the rubric warns against.

[Deep Variational Metric Learning (Lin, Duan, Dong, Lu, Zhou, ECCV 2018)](https://link.springer.com/chapter/10.1007/978-3-030-01267-0_42) states CITTR's causal premise verbatim: *"the distribution of variance within classes is actually independent on classes,"* and exploits that class-independence to synthesize discriminative samples for other classes. Same premise, same field, same benchmarks, seven years earlier.

[Delta-encoder (Schwartz et al., NeurIPS 2018)](https://proceedings.neurips.cc/paper/2018/hash/1714726c817af50457d810aae9d27a2e-Abstract.html) *"extract[s] transferable intra-class deformations, or 'deltas', between same-class pairs of training examples, and … appl[ies] those deltas to the few provided examples of a novel class."* That is paired-sample offset transplantation across identities — the rubric's named comparator, and the closest analogue to CITTR's paired-augmentation displacement.

[Yin et al., CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Yin_Feature_Transfer_Learning_for_Face_Recognition_With_Under-Represented_Data_CVPR_2019_paper.html) transfers intra-class variance from regular to long-tail identities under a center-based framework — the face/re-ID instance of the same mechanism.

**None of these four appear in the frozen novelty table**, despite the rubric naming augmentation-offset transplantation, Delta-Encoder, feature-space augmentation, and cross-class variation transfer as the axes to search. The table's actual entries (PFML, DADA, IAA, MVT, EE, HFA, Manifold Mixup, CCFA, SAM) are correctly distinguished, but they are not the nearest works. The residual conjunction — *paired augmentation tangent + exact spherical transport + worst-donor proxy margin* — may be narrowly unoccupied; the mechanism is not, and the search that was frozen does not support the novelty claim it makes.

---

## 4–5. Controls

Controls 3 (raw Euclidean copy), 5 (same-identity donors), 6 (mean vs. max) and 7 (wall-clock match) are well-chosen and correctly targeted. Control 5 in particular cleanly separates cross-identity transport from tangent propagation / consistency regularization.

They still do not isolate the mechanism:

- **No plain-margin arm.** Covered above — the decisive defect.
- **Control 2 is confounded with effective margin magnitude, not equalized by it.** "Identical radii" is trivially satisfied (all ε = 0.20) but radius is not the operative quantity; `sin ε · v·(p_n − p_y)` is, and it differs between arms by construction. Matching radii matches the wrong scalar.
- **Control 3 is magnitude-mismatched.** `ε‖z⁺ − a‖` uses the chord `2 sin(ϑ/2)` ≈ 0.3–0.6 times ε, versus `sin(0.2) = 0.199` for the geodesic arm. Any difference confounds transport with perturbation size.
- **No ε sweep**, so "exact transport matters" cannot be separated from "this radius happens to work."

The falsification rule (beat random-tangent and raw-copy by ≥ 0.002 mean R@1) is admirably predeclared and blocks one error direction, but as shown it admits the other.

---

## 6. Frontier crossing

By the proposal's own arithmetic: **1.2σ (CUB), 1.2σ (Cars), 1.8σ (SOP)**. A 1.2σ predicted crossing is a coin flip; the proposal states this plainly and also notes that PFML reproduction variance "of a few tenths of a point [is] large relative to the forecast frontier gaps." Forecast gains of +0.006 / +0.005 / +0.005 are smaller than the margin-vs-transport confound quantified above (~25 % of an already small effective margin). The forecasts do not cross the frontier in any decision-relevant sense.

Declining an In-Shop forecast on capacity/reference-mismatch grounds is the right call and is credited.

---

## 7. Data, contamination, protocol

The contamination protocol is mechanical and correct: official-split hashes, no test instantiation during development, identity-disjoint tuning split from training identities only, freeze before final retraining, single test evaluation, deploy epoch 200 with no checkpoint selection, ImageNet-1K as the sole external source. Test operation is compliant — one model, one view, 512-D, cosine NN, no TTA/reranking/gallery processing.

**One exception:** the "about 45,000 proxies" figure for SOP (§1d). Under the reading that reconciles the arithmetic, proxies are instantiated for all 22,634 SOP classes including the 11,316 test identities. I read this as an arithmetic slip rather than intent, but as frozen it is the one protocol-relevant inconsistency in an otherwise clean section.

---

## Summary of frozen defects

| # | Defect | Type |
|---|---|---|
| 1 | No ε = 0 / γ-matched control; `L_CITTR` dominated by a plain 0.05 angular margin that no arm removes | **decisive — causal claim unidentifiable** |
| 2 | Mechanism occupied by SFT (ECCV 2020, rotation transfer on the hypersphere) and DVML (ECCV 2018, identical class-independence premise); neither cited | novelty |
| 3 | Zero measurement of donor→receiver tangent alignment; premise conceded unestablished | rubric 6 failure |
| 4 | `‖u‖ < 10⁻³` fallback unreachable — clip floor forces `‖u‖ ≥ 4.47×10⁻³` | dead operation |
| 5 | Numerator clipped, denominator not → NaN at `aᵀb → 1`, undetectable by the magnitude guard | numerical |
| 6 | ε saturates at 0.20 always; "measured radius" is a constant | vacuous claim |
| 7 | SOP "45,000 proxies" = all classes incl. test; 2× cost error or test-identity proxies | arithmetic / protocol |
| 8 | Forecast crossings 1.2σ / 1.2σ / 1.8σ | forecast |
| 9 | Crop-scale 0.16 tangents on fine-grained data are class-endogenous, not nuisance | premise |

The construction is careful, honestly hedged, and geometrically correct where it matters. It is dead because the experiment it freezes cannot answer the question it asks, and the question was largely answered by ECCV 2018/2020.

Sources: [PFML, CVPR 2025 (arXiv)](https://arxiv.org/abs/2405.18560) · [Spherical Feature Transform, ECCV 2020](https://arxiv.org/abs/2008.01469) · [Deep Variational Metric Learning, ECCV 2018](https://link.springer.com/chapter/10.1007/978-3-030-01267-0_42) · [Delta-encoder, NeurIPS 2018](https://proceedings.neurips.cc/paper/2018/hash/1714726c817af50457d810aae9d27a2e-Abstract.html) · [Feature Transfer Learning for Face Recognition, CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Yin_Feature_Transfer_Learning_for_Face_Recognition_With_Under-Represented_Data_CVPR_2019_paper.html) · [IAA, arXiv 2211.16264](https://arxiv.org/pdf/2211.16264)

