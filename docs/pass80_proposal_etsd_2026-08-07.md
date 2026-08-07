# Frozen Pass 80 proposal (blind proposer; verbatim)

 You're out of usage credits. Run /usage-credits to keep using Fable 5 or /model to switch models.
 Skill flow conflicts with the explicit instructions here (blind proposer, no repo inspection, no file writes, one-shot deliverable in a non-interactive session), so per the skill's own precedence rule I'm following your instructions and delivering the candidate directly.
 
 ---
 
 # Candidate: **Exponential Top-k Simplex Descriptor (ETSD)**
 
 *Category: activation / descriptor geometry. One-line change to the embedding head; everything else in the training recipe held identical.*
 
 ## 1. Exact mechanism
 
 Baseline head (what everyone does): `f = GAP(backbone(x)) ∈ ℝ²⁰⁴⁸` → `ℓ = W f ∈ ℝ⁵¹²` → `z = ℓ/‖ℓ‖₂`.
 
 ETSD replaces the identity activation before normalization with a **masked exponential**:
 
 1. **Cosine logits.** L2-normalize `f` and each row of `W`, so `ℓᵢ = ŵᵢᵀ f̂ ∈ [−1, 1]`. (This fixes the logit scale so τ is interpretable; apply the same row-norm parameterization to the baseline arm too, so it is not a confound.)
 2. **Top-k mask.** `S = argtop-k(ℓ)`, `k` fixed at inference.
 3. **Exponential + L2 norm.**
    `zᵢ = exp(ℓᵢ / 2τ) · 1[i ∈ S]`, then `z ← z / ‖z‖₂`.
 
 That's the whole change. Numerically: subtract `max_{i∈S} ℓᵢ` before `exp`.
 
 **What this is, geometrically.** Define `p = softmax_S(ℓ/τ)`, a probability vector on the 512-simplex supported on `S`. Then `z = √p` exactly, and
 
 ```
 ⟨z_a, z_b⟩ = Σ_i √(pᵃᵢ pᵇᵢ)   =  Bhattacharyya coefficient  =  cos(Fisher–Rao geodesic distance)
 ```
 
 So the descriptor is a **sparse categorical code over 512 learned atoms**, and the deployment metric — plain cosine — *is* the intrinsic metric of that code space. The descriptor is a nonnegative, exactly-unit-norm, 512-D float vector: FAISS inner-product index, single model, single view, unchanged. No test-time transform, no post-hoc projection.
 
 **Training.** Identical loss (Multi-Similarity), optimizer, epochs, batch composition, augmentation, and backbone as the baseline. Two ETSD-specific settings, both preregistered from preflight:
 - **τ ∈ {0.05, 0.1, 0.2}** — controls within-support mass concentration.
 - **k annealed 512 → k_final = 32**, geometrically over the first 1/3 of epochs. This is the only piece of machinery beyond the activation, and it exists to prevent dead atoms: early training is a dense softmax so every atom receives gradient, then support tightens.
 - ε-floor is unnecessary in this parameterization (no `√` in the graph — the exponential *is* the square root), which is why I write it as `exp(ℓ/2τ)` rather than `sqrt(softmax(ℓ/τ))`.
 
 Similarities now live in [0, 1] rather than [−1, 1], so the MS loss scale must be re-searched. **Both arms get the identical 3-point grid over the loss scale**, same seeds, same budget — otherwise the comparison is just retuning.
 
 **No** auxiliary loss, **no** extra views, **no** teacher/EMA, **no** proxies or sub-centers, **no** whitening, **no** margin change, **no** post-hoc projection. Added compute: one top-k over 512 per sample (~nil).
 
 ## 2. Why this should improve unseen-class retrieval
 
 The failure mode ETSD targets is not spectral collapse — it's that a dense signed head lets the model separate 100 seen classes with a **private direction per class**, which is worthless on class 101. Four distinct pressures push against that, each with a measurable signature:
 
 1. **Fixed mass budget makes shortcuts expensive.** `Σpᵢ = 1`. A class signature concentrated in one or two coordinates consumes the *entire* budget and drives that image's similarity to everything else toward 0. In a dense head, one large coordinate dominating cosine is the *cheapest* solution; here it is the most expensive. This is a mechanism neither margins nor whitening supplies — both operate on the arrangement of codes, not the cost of concentrating them.
 2. **Atoms are shared, so gradients are shared.** Every class that activates atom *i* trains atom *i*. With k=32 of 512, capacity is `C(512,32) ≈ 10⁵³` — no capacity pressure — but the *only* way to reuse atoms is to make them correspond to sub-class-level visual evidence (parts, colorations, textures) rather than class identity. Unseen classes are then new subsets of known atoms, and similarity degrades gracefully instead of falling off a learned-directions cliff.
 3. **Metric/parameterization match.** Training shapes a simplex code; retrieval uses cosine; Bhattacharyya makes these the *same* geometry. In the standard head, the code lives in ℝ⁵¹² and the sphere is imposed afterward — a mismatch normally patched by whitening or re-centering at test.
 4. **Reduced heterogeneity of "spreadness."** All descriptors have identical L1 mass and identical support size, removing one degree of freedom that drives hubness under distribution shift. (Weaker claim than 1–3 — mass *concentration* within the support still varies with τ. Treated as a secondary prediction below, and it may go the other way.)
 
 ## 3. Cheap preflight (GPU ~5 min, CPU ~40 min)
 
 **Frozen-feature head probe** — tests the geometry claim without any backbone finetuning.
 
 1. Extract ImageNet-init ResNet-50 GAP features once for CUB-200-2011, all 11,788 images (~48 MB per split at fp32). GPU: 1–2 min. CPU: ~30 min.
 2. Train **only the 512-D head** on the 100 seen classes with the MS loss — arm A = baseline (row-normed linear + L2), arm B = ETSD — 3 seeds × 3 τ/scale points each, few hundred steps. Seconds per run.
 3. Evaluate R@1 on the 100 **unseen** classes.
 
 **Go / no-go, all three required:**
 
 | Check | Threshold | Purpose |
 |---|---|---|
 | Probe sensitivity | baseline arm beats untrained frozen-feature cosine by ≥ +3 R@1 | confirms the probe can detect anything at all |
 | Primary | ETSD mean unseen R@1 ≥ baseline mean (3 seeds) | the geometry claim |
 | Atom usage | usage entropy over 512 atoms ≥ log(256) ≈ 5.55 nats on the seen-class train set | not silently collapsed to a few atoms |
 | Non-underfit | seen-class train R@1 within 2 pts of baseline | gain isn't just "trained less well" |
 
 Also log, for free: N₁₀-count skewness (Radovanović hubness) on unseen classes, and mean support overlap between same-class vs different-class pairs. If ETSD wins R@1 but same-class support overlap is *not* above different-class overlap, the compositionality story is wrong even if the number is up.
 
 If atom usage entropy fails, the single permitted fallback is a load-balancing term on batch-mean atom usage (coefficient 0.01, preregistered) — reported as a separate arm, not folded into the main result.
 
 ## 4. Preregistered prediction and falsifier
 
 **Main run.** CUB (100/100), Cars196 (98/98), In-Shop (standard). ResNet-50 ImageNet init, frozen BN, 512-D, batch 32 classes × 4, AdamW (1e-5 backbone / 1e-4 head), fixed epoch budget, RandomResizedCrop(224)+hflip train / resize-256-center-crop-224 test. **5 seeds per arm.** Matched wall-clock and matched tuning grid size.
 
 **Prediction (primary):** on CUB, ETSD improves **mean Recall@1 on unseen classes by ≥ +1.5 points absolute**, with the 95% bootstrap CI of the 5-seed mean difference excluding 0, evaluated at the **fixed final epoch** (no test-set epoch selection).
 
 **Secondary:** ΔmAP@R ≥ +1.0 on CUB; hubness skewness reduced ≥ 20% relative.
 
 **Falsifiers — any one rejects the candidate:**
 - CUB mean ΔR@1 ≤ **+0.5** points, or 95% CI includes 0.
 - Fails to replicate: ΔR@1 ≥ +1.0 on **at most one** of {Cars196, In-Shop}. (Require ≥ 2 of 3 datasets.)
 - The gain requires k or τ retuned per dataset — i.e., the CUB-preregistered (k, τ) loses on Cars196/In-Shop while a per-dataset-tuned setting wins. That's hyperparameter search, not a method.
 - **Mechanism falsifier (does not reject the number, but rejects the explanation):** R@1 improves while atom-usage entropy, same-class support overlap, and hubness all fail to move in the predicted direction. Report as "effect without mechanism" and do not claim the compositional account.
 
 **Reporting honesty.** Report (a) raw 5-seed mean ± std at fixed final epoch as the headline; (b) separately, best-epoch-selected-on-test numbers *labelled as inflated*, only for literature comparability; (c) if any configuration selection happened after seeing test numbers, report the max-over-configs statistic with a selection correction rather than the naked max.
 
 ## 5. Likely closest prior art (and the delta)
 
 - **Hellinger kernel / RootSIFT** (Arandjelović & Zisserman, CVPR 2012) — `√` of an L1-normalized histogram then L2, so cosine = Bhattacharyya. *Closest in geometry.* Delta: post-hoc on hand-crafted histograms; here it is the trained descriptor parameterization with learned atoms and enforced sparse support.
 - **BoW / NetVLAD soft-assignment** (Sivic & Zisserman 2003; Arandjelović et al. 2016) — learned codebook assignment. Delta: VLAD aggregates local *residuals* into a high-dim vector; ETSD is a global 512-D simplex code with no local aggregation and no residuals.
 - **Sparsemax / entmax** (Martins & Astudillo 2016; Peters et al. 2019) and **top-k gating in MoE** (Shazeer et al. 2017) — the activation family and the load-balancing concern. Delta: used as attention/routing outputs, never as a retrieval descriptor.
 - **SPLADE** (Formal et al. 2021) — learned sparse nonneg retrieval vectors. Delta: text, vocabulary-tied dimensions, inverted-index efficiency motive, ReLU+log saturation with a FLOPS regularizer; not simplex-constrained, not Fisher–Rao, not zero-shot visual DML.
 - **Matryoshka Representation Learning** (Kusupati et al. 2022), embedding dropout — subset robustness. Delta: nested prefixes over dense codes; no mass constraint, no compositional atom claim.
 - **Deep hashing** (HashNet et al.) — sparse/discrete codes, but motivated by compression and evaluated in Hamming space.
 - **Roth et al., ICML 2020 (ρ-regularization / spectral decay)** — closest *diagnosis* in ZS-DML, opposite intervention: they flatten the spectrum by perturbing negatives; ETSD changes the code space and leaves sampling untouched.
 
 I have not run a literature search for this; treat the list as the nearest work I can name, and add a targeted search on "simplex embedding metric learning", "sparse embedding deep metric learning", and "Bhattacharyya retrieval descriptor" to the preflight before committing GPU time.
 
 ## 6. Known risks
 
 - Sparse embeddings have historically underperformed dense ones in DML; a clean loss here is a real possibility, which is why the preflight is designed to kill it in minutes rather than GPU-days.
 - Top-k zeroes gradient to unselected atoms — the k-annealing is load-bearing, not cosmetic. If preflight shows dead atoms even with annealing, the method is fragile regardless of R@1.
 - τ conflates two things (support sparsity and within-support concentration). If the winning τ sits at a grid edge, the honest read is that the grid was wrong, not that the method works.
