# Independent frozen-proposal review — CFK

- Durable consultation: `e35e30324c1c4ade`
- Caller attribution: `sfora/emafactorial`
- Provider path: Fable → same-job Claude Opus fallback
- Final status: completed
- Frozen review prompt: `docs/fable_cfk_review_prompt_2026-08-05.txt`
- Prompt SHA-256: `29e082ecf37b1f7ecae41dc1e3c9971d1785c48211b3390fb51f827d227c44a1`

The text below is the complete provider result, preserved verbatim before local reconciliation.

---

I'm the cold adversarial reviewer for this frozen CFK proposal. I'll verify the surviving math offline, then check the two decisive external facts — the Kirchhof et al. ECCV 2022 mechanism and the PFML reference — against primary literature via web search, without touching the local repository.
## Verdict: **DEAD**

### The single decisive reason

The mechanism is fully anticipated, at matched capacity, on the same benchmarks, with released code — and already measured, negatively.

**Kirchhof, Roth, Akata, Kasneci, ECCV 2022** defines, verbatim, `K_p = diag(κ_p) = diag(κ_{p,1},…,κ_{p,M})` with `κ_{p,m} > 0` — a **per-class, per-embedding-dimension concentration vector, learned, diagonal in the shared global embedding basis**, with no per-class rotation. It reduces to the isotropic case exactly when `K_p = cI_M` ("the nivMF simplifies to a vMF"). The released implementation is `criteria/proxyvmf_panc.py` — nivMF **on top of Proxy Anchor** ("PANC+EL-nivMF"), with α/δ. Test-time retrieval is plain cosine.

That is CFK's mechanism claim item-for-item: per-class anisotropic scale, diagonal in one basis shared by all classes, inserted into the Proxy Anchor objective, exactly reducing to PA in the isotropic limit, absent from the deployed metric.

The proposal pre-registered its own off-ramp and this closes it. It also fails empirically before it starts: Kirchhof's R50/512 headline results are **CUB 69.3, Cars 86.2, SOP 79.4** — at or *below* CFK's own frozen PA baseline prior (69.2 / 87.9 / 79.7). The only existing benchmark-matched realization of per-class diagonal anisotropy in a proxy-anchor loss delivered ≈0 on CUB and −1.7 on Cars against exactly the baseline CFK forecasts +3.1 / +2.4 against. CFK's own **F1 fires on the published prior art**.

The escape clause is additionally malformed. It requires prior art to already have "a total-precision normalization," then concludes that if so, "CFK's remaining novelty is the gauge + shrinkage + control set." The gauge *is* the total-precision normalization — the clause demands the prior art contain the very thing it then counts as remaining novelty. It cannot be satisfied in either direction, so it is not a usable live off-ramp. What actually survives the comparison is a gauge choice, an ∞-norm clip, an `n_c` shrinkage, and a re-derived Bingham/Euclidean score form in place of a vMF likelihood — a parameterization variant of a published, benchmark-matched method.

### Independently confirmed defects in the frozen text

**§5.4 contains a negative probability.** Marginals 0.45 + 0.33 + 0.30 = 1.08 = E[N]. Since E[N] = P(N≥1)+P(N≥2)+P(N≥3), the frozen P(≥1)=0.62 and P(≥2)=0.22 force P(≥3) = 0.24 > P(≥2) = 0.22, i.e. **P(exactly two clean crosses) = −0.02**. At the extreme corner of every rounding it can be nudged to ≈0, which then asserts that two crosses is impossible while all three has probability ~0.22 — a structure the text never states and which contradicts its own "the most likely single outcome is one clean cross on CUB." The decision-relevant arithmetic is incoherent.

**C5's null is false and F8 is a false halt.** With class-independent `ω`, σ = 1 − ½(‖z‖²_Ω + ‖p̂‖²_Ω) + ⟨z,p̂⟩_Ω. The term ‖z‖²_Ω = Σⱼωⱼzⱼ² is *not* constant on the sphere — it ranges over [e^{−2τ}, e^{2τ}] = [0.33, 3.0] at τ=0.55. PA's loss is per-proxy LogSumExp-with-1, **not** a softmax over classes, so a per-sample additive score offset does not cancel; it shifts the exponent by up to ±42 at α=32. The absorbability argument is applied to the wrong object: Ω acting *after* L2 normalization is not a reparameterization of W acting *before* it — ν(Ω^{1/2}Wx) is parallel to Ω^{1/2}ν(Wx) but differs by the sample-dependent scalar ‖Ω^{1/2}Wx‖/‖Wx‖. So C5 ≠ C0, and F8 would halt a correct implementation. The identical error voids the C2 parenthetical ("a fixed random Q … is absorbed into W"): QᵀΩQ post-normalization is not absorbable either.

**C1, "the single most decisive control," is mis-specified.** CFK's gauge ∏ⱼω_{c,j}=1 is *within-class over j*; isotropy inside CFK means ω_c^d = 1 ⇒ ω_c = 1. A per-class scalar temperature is therefore **unreachable inside CFK's own parameter set**. C1 uses a different, *across-class* gauge (⅟C Σ_c log κ_c = 0) and a different score form. The claim "same parameter type and same gauge" is false, C1 is not nested in CFK, and F2 does not cleanly isolate anisotropy.

**§2.4 guard (iii) is a non-sequitur.** "ũ is discarded at test, so memorization … can only waste capacity" — discarding a parameter at test does not neutralize its training-time effect. ω_c scales the gradient into the backbone per axis; class-discriminative structure absorbed into ω_c is structure the backbone is no longer required to encode. That is memorization *relocated into the discarded metric*, which is strictly worse than wasted capacity: it degrades the deployed descriptor while training loss looks healthy, and it does not show up as "CFK ≤ PA" in any guaranteed way. This was the one guard claiming to structurally rule out memorization; it does not. The fallback, C4, is itself mislabeled — "SoftTriple K=2, i.e. PA with 2 proxies/class" is not an identity (SoftTriple uses a class-softmax over intra-class softmax-weighted similarity plus a small-cluster regularizer), so F4's parameter-matched null is unmatched in objective.

**§1.6's gauge justification is false as written.** "Using the additive Bingham gauge here would leave an unfixed global precision scale" — an additive normalization Σⱼω_{c,j} = const *does* pin the multiplicative scale (Σγωⱼ = γΣωⱼ). The stated reason for preferring ∏ω=1 does not hold. Relatedly, §2.3's claim that the hard gauge makes collapse "exactly unreachable" closes only the *uniform*-scale direction, which fixing α already closes; the degeneracy that matters is §2.2's GM shortcut, which the gauge does not touch.

**§2.1's causal claim is derived from the positive term alone.** ω_{c,j} multiplies the same σ_c in both the positive and negative terms, so lowering it attenuates class c's repulsion on axis j exactly as much as its attraction. The mechanism is gradient *attenuation*, not a reward for representing within-class variation — nothing in the complete objective pays to retain sⱼ. The proposal's own Risk-2 fallback ("retards catastrophic forgetting of pretrained factors") is the accurate description, and it is the weaker claim.

**No discriminating control for the load-bearing common-frame claim.** C2 re-randomizes the frame every step, which simultaneously destroys class-specific axis identity, makes ω_c effectively unlearnable (the accumulated gradient over random rotations is isotropic in expectation), and injects noise. The proper contrast — a **per-class learned rotation** — is absent, so the CPC/Flury structural claim has no control that isolates it.

**§5.3 is not executable.** §7.2 concedes PFML's objective, field parameterization, proxy schedule, optimizer, LR schedule, batch composition and pooling are all unknown. Every forecast row that touches the frontier sits inside that conditional; §5.2 alone crosses nowhere by the proposal's own admission. So no frozen forecast crosses the proper frontier via a specified route, independent of novelty.

**Unavailable, not repaired.** Everything before mid-§1.6 is gone: L_CFK, the map ũ_c → u_c, η_c, λ_u, α, δ, and the joint projection onto {Σⱼu_{c,j}=0} ∩ {‖u_c‖_∞ ≤ 2τ} — clip-then-center and center-then-clip each violate the other constraint, and the composite operator appears nowhere in surviving text. I did not reconstruct it. What survives is mutually consistent: τ=0 ⇒ σ_c = 1 − ½‖z−p̂‖² = ⟨z,p̂⟩ ✓; e^{−1.1} = 0.333 ✓; α=32 ⇒ α_eff=10.7 ✓; 32e^{−2.2}=3.55 ✓; the GM Lagrangian and the two-sided bound are correct as derived.

**Minor.** The §7.2 PFML venue ambiguity resolves — CVPR 2025, arXiv:2405.18560 is the 2024 preprint; the prompt's dating was right. The proposal cites a liner.com AI-summary page as its PFML source rather than the primary paper, in a frozen document whose entire frontier arithmetic keys off that reference.

### On protocol legality
No hidden-data, contamination, transduction, or capacity violation found. The §6.3 pre-commitment to a class-disjoint validation split carved from training identities is correct practice, and the deployment (one R50, one view, 512-D, cosine) is lane-legal — identically so for nivMF, which is why the deployment argument cannot carry novelty.

**Sources:** [Kirchhof et al., ECCV 2022 (ECVA)](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136860423.pdf) · [arXiv:2207.03784](https://arxiv.org/pdf/2207.03784) · [released code, `criteria/proxyvmf_panc.py`](https://github.com/ExplainableML/Probabilistic_Deep_Metric_Learning) · [Bhatnagar & Ahuja, PFML, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Bhatnagar_Potential_Field_Based_Deep_Metric_Learning_CVPR_2025_paper.html) · [arXiv:2405.18560](https://arxiv.org/abs/2405.18560)

