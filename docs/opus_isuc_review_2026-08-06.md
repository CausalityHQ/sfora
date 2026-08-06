## Verdict: **DEAD**

**Earliest failed gate: Gate 1 — measurement provenance.**

**Decisive mechanism-level reason:** Proposition 1 is the proposal's *only* bridge from "test identities are disjoint" to "maximize A," and it conflates **class-disjointness with statistical independence**. §1.2 step 1 draws *one uniformly random balanced partition* of the batch's P classes, so S₂ is the complement of S₁ — maximally negatively dependent, not independent. The claimed identity 𝔼⟨G(S₁),G(S₂)⟩ = ‖Ḡ‖² requires 𝔼⟨E(S₁),E(S₂)⟩ = 0, which complementary sampling without replacement forbids. Writing T for the batch field and V = 𝔼‖E‖², the additive case gives exactly

```
E⟨G₁,G₂⟩ = ⟨Ḡ,T⟩ − E‖G₁‖² = ‖Ḡ‖² − V   (not ‖Ḡ‖²)
⇒ E[A] ≈ (‖Ḡ‖² − V)/(‖Ḡ‖² + V)
```

so the "fraction of each SGD step a fresh identity sample would also produce" is **(1+A)/2, not A**. The proposal's own §6.3 guess of A ≈ 0.05 would mean a transferable fraction of ≈0.525, not 0.05 — the diagnosis inverts. The optimization is monotone-equivalent, so this does not break the penalty; it breaks the *theorem that is standing in for measurement*.

That is why the failure lands at Gate 1 rather than later: the repository has measured no G(S), A, or E(S), and the proposal measures none either (F8 is post-hoc, and §6.3 *guesses* the central quantity). The supplied persistence statistics (0.7675 / 0.8085 / 0.6476) are never referenced anywhere in the frozen document. The causal error mode is inferred from the disjoint-identity protocol alone, via a proposition that is false at its first step.

---

## Independent confirmations (each would also be fatal)

**Item 5 — the expansion.** The prompt's *second* form is correct. Since Σ_{i∈S_k} d̃_i = 0, any constant is annihilated on the h side:

```
G_k = Σ(d_i − d̄_k)(h_i − h̄)ᵀ = Σ d_i h_iᵀ − n_k d̄_k h̄_kᵀ
```

The **global h̄ drops out identically — the feature centering is a provable no-op.** §2.2's "Centering is therefore load-bearing, not cosmetic" is half false: d-centering delivers *both* shift invariances; h-centering delivers nothing. The removed common mode is **rank ≤ 1**, not "rank ≤ 2," and it uses the *half* mean h̄_k, not the batch mean the definition specifies. G_k has rank ≤ 89 inside a 512×2048 space, which is the real source of the §6.3 noise worry.

**Item 4, second half — restricted energies are not subdivisions.** U₁ + U₂ ≠ U(B): every cross-half sample–sample, sample–proxy, and proxy–proxy repulsion is deleted, and U(B)'s proxy term runs over all C=100 classes while U_k's runs over 15. So **∇_W U₁ + ∇_W U₂ ≠ ∇_W U(B)**. G(S) is not a piece of the step; it is the gradient of a *different, 15-class problem*. No correction to the independence algebra recovers the claimed estimand, because the estimand is not the step. Compounding this, the expectation in Prop 1 is over the batch's own 30 classes — a within-batch statistic — and 𝔼[ratio] ≠ ratio of 𝔼 is used with only an "≈" to cover a strongly dependent numerator and denominator.

**Item 6 — last-layer alignment does not identify backbone transfer.** ⟨G₁,G₂⟩ uses the feature Gram h_i·h_j; the actual parameter update uses ∇_θU_k = Σ J_iᵀWᵀd_i, whose alignment is Σ g_iᵀ J_iJ_jᵀ g_j. Counterexamples both ways:
- *A = 1, backbone alignment exactly 0.* One hidden ReLU layer, h = ReLU(Vx), J_i = diag(1[Vx_i>0]) ⊗ x_iᵀ ⇒ J_iJ_jᵀ = (x_i·x_j)·diag(1_i)diag(1_j). Take cross-half inputs orthogonal (x_i·x_j = 0) with identical post-ReLU h and identical d. Then A = 1 and ⟨∇_θU₁,∇_θU₂⟩ = 0 exactly.
- *A = 0, backbone alignment > 0.* Let cross-half d_i·d_j = 0 (orthogonal 256-D halves of ℝ⁵¹²) ⇒ A = 0. With a linear backbone, ⟨∇_VU₁,∇_VU₂⟩ = Σ(x_i·x_j)(d_iᵀWWᵀd_j), which is strictly positive whenever WWᵀ couples the two subspaces.

**Item 7 — P2 is refuted as a proof.** Two separate defects:

1. *Bounded value ≠ bounded gradient.* ∂A/∂G₁ = G₂/(‖G₁‖‖G₂‖) − A·G₁/‖G₁‖² scales as **1/‖G₁‖**, diverging exactly as a half's centered field vanishes (i.e. as PFML converges). ε = 1e−8 does not regularize this at any realistic scale. "Total influence ≤ 2λ" bounds the *value*; SGD follows gradients.

2. *The 0.1998 bound is void on the operative region.* I verified the clamp structure against the primary source: repulsion is **constant beyond δ** and attraction constant within δ ([arXiv 2405.18560](https://arxiv.org/html/2405.18560v2), Eqs. 1–2). On the ℓ2 sphere, δ = 0.2 means cross-class repulsion exerts **exactly zero force on any pair with cosine below ≈0.98**. So PFML's base energy is *flat* — zero excess, zero gradient — across the entire manifold containing every retrieval-relevant configuration. P2's premise ("any configuration whose base energy exceeds the optimum by more than 2λ can never be preferred") is true and inert: the configurations at issue exceed the optimum by **zero**. Packing all classes at mutual cosine ≈ 0.97 has *identical* repulsive energy and *lower* attractive energy than a well-spread embedding, satisfies η ≥ 0.1998, maximizes coherence, and destroys R@1. P2 proves a floor at the clamp radius, not no-collapse. (The K² ordered-pair count should be 2K² given §1.1's doubled sums; that error is conservative and the 0.1998 arithmetic is otherwise correct.)

**P3 contradicts §2.2's own residual-risk paragraph.** "d = Mh̃ is the target, not a shortcut" is false as stated: take M = ab ᵀ with b a nuisance direction (illumination, background); then d̃_i ∝ a·(nuisance_i), A → 1, and the penalty is fully satisfied by a field carrying zero identity information. Image content ≠ identity-discriminative content. P3 and the residual-risk paragraph cannot both be right about the same object. **P5 inherits P2's exact hole** — proxy–proxy divergence at zero distance is a barrier *inside δ only*. **P4** is understated: on CUB, taxonomically adjacent species are the typical case, so A is preferentially raised by *not* separating confusable classes — a concrete route by which ISUC could reduce fine-grained R@1.

---

## Item 2 & 3 — novelty: estimator/wrapper recurrence

The reduction that killed Candidate 228 survives with only cosmetic change. ISUC's object is ⟨∇_W L_A, ∇_W L_B⟩ normalized — precisely the first-order term of L_B(θ − η∇L_A). ISUC computes it directly instead of via virtual update, but **MLDG already does exactly that analytically**; Reptile/Fish are the ones that *approximate* it. Each remaining differentiator is prior art on its own axis:

- **Cosine of gradients as the operative optimization signal** — Du et al., [arXiv 1812.02224](https://arxiv.org/abs/1812.02224) (cosine similarity between task gradients as an adaptive weight).
- **Gradient agreement as an explicit meta-learning objective** — Eshratifar, Eigen & Pedram, [arXiv 1810.08178](https://arxiv.org/abs/1810.08178).
- **Inter-task gradient alignment as a training target** — Sequential Reptile, Lee et al., ICLR 2022, [arXiv 2110.02600](https://arxiv.org/abs/2110.02600).
- **Episodes over disjoint label subsets in DML** — Zheng, Lu & Zhou, *Deep Metric Learning with Dynamic Constraints*, TPAMI 2023 (per supplied evidence); Deep Meta Metric Learning, ICCV 2019; MASF, NeurIPS 2019; M3L, CVPR 2021.
- **The ⟨d_i,d_j⟩⟨h_i,h_j⟩ last-layer device** — TracIn / stiffness, credited by the proposal itself.

**Narrowest defensible novelty:** *a closed-form, cosine-normalized, half-mean-centered, last-layer-only gradient-alignment penalty between two class-disjoint halves of a single mini-batch, on a proxy-based potential-field loss.* On the three axes the prompt names: swapping domains→random identity subsets is **relabeling the free grouping variable** in Fish/IGA/Fishr, not substantive; **disjoint proxies are not substantive and are affirmatively harmful** (they are what breaks U₁+U₂ = U(B)); **cosine normalization is not substantive** given Du et al. and Eshratifar et al. Restricting to the last layer is a known approximation, not an invention.

## Item 8 — executable fidelity

Two transcription findings against the primary source, both against the proposal's self-assessment:
- §6.2 lists "whether the proxy–proxy term ranges over all C classes" as undisclosed. **It is disclosed** (Eq. 5 ranges γ = 1…N, γ ≠ j). §1.1 transcribes it correctly, but §6.1's cost model does not price it: 1500² proxy pairs × 512 dims ≈ **5.8 TFLOP/step, larger than the 2.2 TFLOP ResNet-50 fwd+bwd** the overhead is quoted against. The denominator in "<0.2%" is wrong (in the proposal's favor).
- §6.2 assumes the published ± is SD because "it is not disclosed." **It is disclosed** — the Table 1 caption states standard deviations over 5 runs. The assumption is correct, so the SEM escalation worry is moot and the bar stays at CUB ≥ 0.739 / Cars ≥ 0.932.

The <0.2% FLOP claim is arithmetically plausible *for the restricted energies as scoped*. The **<2% memory claim is doubtful**: `create_graph=True` over four restricted energies retains pairwise intermediates (~315×315×512 per energy ≈ 0.2 GB fp32 each unless chunked). F6 binds it, which is honest. Separately: because ψ_att is constant within δ and ψ_rep constant beyond δ, **most of the double-backward is identically zero** — the second-order signal is carried almost entirely by same-class attraction, which materially narrows what the penalty can shape. The "graph never touches the backbone" claim is correct as scoped, though the d̃ path does reach θ via v = Wh + b; one backbone backward remains achievable.

## Item 9 — controls and falsifiers

- **F3 has no discriminative power.** "No generic regularizer predicts [monotone gain as C shrinks]" is simply false — weight decay, dropout, label smoothing, mixup and SAM all gain more as training data shrinks, and shrinking C shrinks images too. At 3 seeds against effects of 0.008–0.012, it is also badly underpowered.
- **F8 is not a mechanism test.** Spearman ρ > 0.5 at n = 5 has p ≈ 0.2 (critical value is 0.90). It is tracking, not causation — and since ISUC *directly optimizes* A, "A rises under ISUC" is tautological.
- **F2/F4/F5 are underpowered by an order of magnitude.** Resolving "75% vs 70% of a 0.010 gain" means resolving 0.0005 against SEM ≈ 0.002.
- **No falsifier attached to Control 8 (Fish/Reptile inner loop) or Control 7 (proxy-side alignment).** Given the Item-2 finding, Control 8 is *the* decisive novelty test, and its omission from the pre-registered list is the single most consequential protocol gap.
- **No control addresses the conceded principal failure mode.** §2.2 names shared nuisance as the main risk and delegates it to F5/F8; neither measures nuisance. A direct nuisance-decodability probe is absent.

Control 2 (class-shared split at identical FLOPs) is correctly designed and is the one genuinely decisive control here.

## Item 10 — frontier arithmetic and protocol

Verified correct: SE_diff = √(0.00134² + 0.00179²) = 0.002236; 1.96× = 0.004383; bars 0.739 / 0.932. Errors found:

- **The two "80% CI"s are mislabeled and mutually inconsistent.** With σ = 0.004, n = 5, SEM = 0.00179, an 80% CI is ±0.0023. CUB is stated as ±0.008 (≈4.5 SEM, ~99% coverage); Cars as ±0.006 (≈3.4 SEM, ~97%). Neither is 80%, and they do not agree with each other (2σ vs 1.5σ).
- **The n = 10 remark is wrong.** 0.0031 assumes the *published* PFML baseline also gets 10 seeds; it is fixed at n = 5. Correct value: √(0.00134² + (0.004/√10)²) × 1.96 = **0.0036**.
- **Cars fails the proposal's own bar at the forecast mean** (0.931 < 0.932) — F7 fires by pre-registration. CUB clears by 0.001, one-half of its own SEM.
- **The frontier-inheritance condition is forecast to be at its own boundary**: the PFML reproduction is forecast at CUB 0.730, exactly −0.004 from 0.734, with ten source hyperparameters chosen by the author.
- **No F1-analogue for Cars.** If Cars shows zero gain, no falsifier fires.
- **The mandated first screen is absent.** The envelope requires paired corrected In-Shop *first*; §5 states "In-Shop: no forecast," there is no In-Shop arm, threshold, or falsifier, and In-Shop appears only in §6.4 to be excluded. Out-of-sample confirmation has no arm or threshold (SOP is "directional only"); replication exists across CUB/Cars for baseline-relative gain but is forecast **not** to occur for the frontier claim. Raw vs independently-selected/final reporting is not distinguished — one figure per arm. Train-only tuning is correctly specified in principle (§6.4).

---

## Correct subcomponents, preserved separately from the verdict

These survive independent of the verdict and should not be discarded with it:

1. **The Frobenius factorization is exact.** ⟨G₁,G₂⟩_F = Σᵢ Σⱼ (d̃ᵢ·d̃ⱼ)(h̃ᵢ·h̃ⱼ) is correct, and the 90×90 Gram formulation is the right O(n²(D+p)) computation.
2. **G_k = Σ dᵢhᵢᵀ − n_k d̄_k h̄_kᵀ** is a valid, cheap object — correct algebra, wrong description.
3. **P1's invariances are true** (shift in h, per-half shift in d, positive rescaling of either), delivered entirely by d-centering.
4. **The frontier-bar derivation is arithmetically correct**, and its SD-vs-SEM reading is confirmed correct against the source.
5. **Stating plainly that Cars fails at the forecast mean** is a real and rare piece of intellectual honesty; the frontier-inheritance void condition (±0.004) is a well-designed binding device.
6. **The exchangeability caveat** about CUB/Cars' deterministic taxonomic class split (§6.4) is a correctly identified threat.
7. **Control 2** (class-shared split, identical FLOPs) is the right null for class-disjointness.
8. **Deployment invariance** — byte-identical test path, one encoder, one 512-D descriptor, cosine NN — is correct and within the legal envelope.

## Uncertainty

- The **rank-89 / SNR concern** for A is my inference from G_k's structure, not measured; §6.3's "A ≈ 0.05" is the proposal's own unmeasured guess.
- The **collapse argument refutes P2 as a proof, not as an empirical prediction.** Whether packing actually occurs depends on optimization dynamics neither the proposal nor I have measured — 200 epochs plausibly never reaches PFML's degenerate energy minimum. My claim is narrower and firmer: P2's stated premise does not hold on the operative region, so the no-collapse claim reverts to unverified hope.
- **Prior-art absence cannot be certified**, and I share the proposal's §3 limitation. I verified four load-bearing citations against primary sources; I did not verify M3L, PADS, MASF, IGA, or Fishr venue details beyond recall, and I did not read the Zheng et al. TPAMI 2023 full text (title and authorship confirmed; its disjoint-label episode construction I take from the supplied evidence).
- I did not edit any repository file, inspect git history, prior reviews, the ledger, or any local audit, and launched no consultation or agent.

**Sources:** [PFML, arXiv 2405.18560](https://arxiv.org/html/2405.18560v2) · [Adapting Auxiliary Losses Using Gradient Similarity, arXiv 1812.02224](https://arxiv.org/abs/1812.02224) · [Gradient Agreement as an Optimization Objective for Meta-Learning, arXiv 1810.08178](https://arxiv.org/abs/1810.08178) · [Sequential Reptile, ICLR 2022, arXiv 2110.02600](https://arxiv.org/abs/2110.02600) · [Deep Metric Learning with Dynamic Constraints, TPAMI 2023 (Zheng, Lu, Zhou)](https://ivg.au.tsinghua.edu.cn/Jiwen_Lu/publications.html) · [Deep Meta Metric Learning, ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/papers/Chen_Deep_Meta_Metric_Learning_ICCV_2019_paper.pdf) · [Fish / Gradient Matching for Domain Generalization, arXiv 2104.09937](https://arxiv.org/abs/2104.09937)
