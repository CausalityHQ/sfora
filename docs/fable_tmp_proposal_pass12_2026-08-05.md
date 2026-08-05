# Proposal: Tied-Mode Proxies (TMP)

**Lane: A** (matched 512-D CNN lane: ResNet-50, 512-D normalized descriptor, ~224 px input, single-view cosine retrieval, ≤200-epoch budget). All forecasts and comparisons below are in Lane A only.

**One-paragraph summary.** Multi-proxy methods hold the Lane-A frontier (PFML uses 15 free proxies per class), but every existing method lets each class's intra-class modes sit anywhere: the proxies of class *c* are free parameters unrelated to the proxies of class *c′*. TMP replaces free multi-proxies with an additively factorized, cross-class-tied parameterization: proxy *k* of class *c* is `normalize(q_c + r_k + ε_ck)`, where the mode offsets `r_k` are **shared by every class** and norm-budgeted, `q_c` is the unit identity anchor, and `ε_ck` is a small elastic per-class residual. The tied offsets force the encoder toward a class-independent, bounded-magnitude response to identity-preserving mode changes (viewpoint, catalog shot type, pose family). Because that response is class-independent, it extrapolates to unseen identities — which is exactly what free per-class proxies cannot do, since proxies are train-only machinery whose class-specific structure is discarded at test. An elastic-budget ladder on `ε` interpolates continuously from TMP to free multi-proxies, giving an unusually decisive control axis.

---

## 1. Executable mathematics

**Learned objects.**
- Backbone `φ_θ`: ResNet-50, ImageNet-1K init, BatchNorm frozen (lane-standard), global average pool → `h ∈ R^2048`.
- Head `W ∈ R^{512×2048}` (no bias); embedding `z = Wh / ||Wh||₂ ∈ S^511`.
- Identity anchors `q_c ∈ R^512`, constrained `||q_c||₂ = 1`, for train classes `c = 1..C`.
- Shared mode offsets `r_k ∈ R^512`, `k = 1..K`, constrained `||r_k||₂ ≤ ρ`. Shared across **all** classes.
- Elastic residuals `ε_ck ∈ R^512`, constrained `||ε_ck||₂ ≤ ρ_ε = γρ`.

**Proxy construction.** `p_ck = (q_c + r_k + ε_ck) / ||q_c + r_k + ε_ck||₂`.

**Mode-pooled class similarity** (log-sum-exp soft-max over modes, mean-normalized so `S_c ≤ max_k⟨z,p_ck⟩` with equality as β→∞):

`S_c(z) = (1/β) · log[ (1/K) Σ_{k=1..K} exp(β ⟨z, p_ck⟩) ]`

**Main loss** (Proxy-Anchor generalized to pooled multi-proxies). For batch `B` with per-class sample sets `X_c`, `C⁺` = classes present in the batch, and negatives taken against **all** C classes (PA convention):

`L_PA = (1/|C⁺|) Σ_{c∈C⁺} log[1 + Σ_{i:y_i=c} exp(−α(S_c(z_i) − δ))] + (1/C) Σ_{c=1..C} log[1 + Σ_{i:y_i≠c} exp(α(S_c(z_i) + δ))]`

**Mode-usage balance.** Soft assignment `a_k(i) = softmax_k(β⟨z_i, p_{y_i,k}⟩)`, batch mean `ā_k`; `Ω_u = Σ_k ā_k log(K·ā_k)` (KL to uniform). Total objective: `L = L_PA + λ_u Ω_u`.

**Gradient paths.** Full gradients everywhere, no stop-gradients. `∂L/∂z_i` flows through the softmax pooling (concentrating on the nearest modes) into head and backbone. `∂L/∂p_ck` splits by the chain rule of the normalization into `q_c`, `r_k`, `ε_ck`. The mechanism carrier is the `r_k` path: **each shared offset accumulates positive-pull gradients from samples of every class assigned to mode k and negative-push gradients from samples near mode k of wrong classes**, so `r_k` is estimated from ~C× more data than any per-class proxy; `ε_ck` receives only class-c gradient and is norm-clipped.

**Constraints via projection after each optimizer step:** `q_c ← q_c/||q_c||`; `r_k ← r_k·min(1, ρ/||r_k||)`; `ε_ck ← ε_ck·min(1, ρ_ε/||ε_ck||)`. The `||q_c||=1` projection is load-bearing (see D5).

**Initialization.** One forward pass over the training set with init weights: `q_c` = normalized class-mean embedding; `r_k` = top-K principal directions of the within-class-centered embedding matrix, rescaled to norm ρ/2 (train images only — legal); `ε_ck = 0`.

**Frozen hyperparameters.** K = 8 (CUB, Cars196, In-Shop), K = 2 (SOP, where classes have 2–5 images); ρ = 0.3; γ = 0.1 (so ρ_ε = 0.03); β = 20; α = 32; δ = 0.1; λ_u = 0.01. AdamW: lr 1e-4 (backbone + head), 1e-2 (q, r, ε); weight decay 1e-4 on backbone/head, 0 on constrained proxy variables. Batch 120: 30 classes × 4 images (CUB/Cars), 60 × 2 (In-Shop, SOP). Augmentation: random-resized-crop (scale 0.16–1.0) to 224, horizontal flip p=0.5 — ordinary lane augmentation only. Schedule: 1 warm-up epoch training proxies with frozen embedding, then 150 epochs cosine decay, fixed length, no early stopping (within the 200-epoch budget). All development-time selection (K, ρ, γ, β sensitivity) done on an 80/20 split of **training classes** (held-out train classes as pseudo-unseen); final runs retrain on all training classes with the frozen recipe, 5 seeds. No test data, gallery statistics, or transduction anywhere.

**Deployment.** Encode one 256→224 center-cropped view; descriptor = the 512-D normalized `z`; cosine nearest-neighbour retrieval. `q, r, ε` are train-only and discarded; the deployed artifact is byte-shape-identical to a Proxy-Anchor lane model.

Pseudo-code (per step): `z = normalize(head(backbone(x)))`; `P = normalize(Q[:,None,:] + R[None,:,:] + E)`; `sims = z @ P.reshape(C*K,512).T`; pool over K with LSE(β) → `S ∈ R^{B×C}`; compute `L_PA + λ_u Ω_u`; Adam step; project `Q, R, E`.

---

## 2. Causal zero-shot error mode and degeneracy attacks

**The error mode: class-conditional mode-response extrapolation.** Write the generative structure as `x = g(ι, v, n)` (identity, mode/view factor, nuisance). Zero-shot R@1 fails on a query when `||f(x(ι,v)) − f(x(ι,v′))|| > min_{ι′≠ι} ||f(x(ι,v)) − f(x(ι′,v))||` — the cross-mode positive is farther than the nearest same-mode lookalike. On Cars196 and In-Shop this is the canonical failure: catalog and car photography have a handful of near-discrete shot types, and fine-grained negatives in the *same* shot type are very close. Free multi-proxy training (SoftTriple, PFML) fits each class's mode layout independently, so the encoder's response to `v` is learned *per class*; for an unseen identity the mode response is an uncontrolled extrapolation of class-specific behaviors — nothing in training bounds its direction or magnitude. TMP makes the mode response a **global, bounded, shared structure**: training pressure factors the encoder as `f(x(ι,v)) ≈ normalize(q_ι + r_{k(v)})` uniformly over classes, so the unseen-class extrapolation defaults to the same additive response with chordal magnitude ≤ ~ρ, keeping cross-mode positives inside the same-mode-lookalike margin. Measurable causal signature: TMP should specifically shrink cross-mode positive distances on held-out classes relative to free multi-proxies, without shrinking same-mode negative margins (control C7).

**Proof-level attacks on the cheap degeneracies.**
- **D1 — identity cannot hide in ε.** With `||q||=1, ||r||≤ρ, ||ε||≤ρ_ε`, the denominators `||q+r+ε||` lie in `[1−ρ−ρ_ε, 1+ρ+ρ_ε]`, and the ε-attributable part of any between-class similarity margin at a fixed mode is bounded by `2ρ_ε/(1−ρ−ρ_ε)`. At ρ=0.3, ρ_ε=0.03 this is ≈ 0.09 cosine units — well under typical trained inter-class proxy margins (~0.25–0.45). So class discrimination must be carried by `Q`; residuals can only locally deform mode geometry.
- **D2 — identity cannot hide in mode usage.** If `q_c = q_{c′}` and ε is negligible, then `p_ck = p_{c′k}` for all k, hence `S_c ≡ S_{c′}`; the positive term for c and the negative term for c′ then act on the *same* scalar in opposite directions, lower-bounding the loss away from optimum and producing a direct gradient separating `q_c` from `q_{c′}`. With K=8 ≪ C, pigeonhole makes usage-pattern coding of identity impossible anyway.
- **D3 — mode collapse `r_k → 0`** reduces TMP to plain Proxy-Anchor: vacuous, not harmful, and detectable (we report `||r_k||` trajectories and usage entropy). λ_u prevents dead modes; if a dataset truly lacks mode structure, collapse is the correct adaptive behavior — the mechanism claim is conditional on mode-structured data (Cars/In-Shop), which is why those are the primary forecasts.
- **D4 — modes routing class-correlated nuisance** (e.g., background style correlated with brand): a direction helps `r_k` only if it reduces loss *summed over all classes*; class-correlated directions cost other classes and are pushed into `ε`, where D1 bounds them. Residual risk: genuinely class-independent photography nuisance can occupy some modes — acceptable (it is real shared structure) but dilutive; bounded by K and ρ.
- **D5 — anchor-norm inflation:** without `||q_c||=1`, the optimizer nullifies modes by inflating `||q_c||` (shrinking the angular effect of r). The projection kills this exactly; this is why q is constrained rather than free.
- **D6 — additivity is only locally valid on the sphere:** large-angle view changes are not additive; ρ = 0.3 (~17° mode angles) deliberately restricts the claim to the local-linear regime.

---

## 3. Adversarial novelty search (primary sources)

Searched August 2026; no occupied instance of cross-class-tied, norm-budgeted additive proxy factorization for zero-shot DML was found. Nearest works and one-sentence mechanism distinctions:

**Inside DML / retrieval:**
- **PFML** ([arXiv 2405.18560](https://arxiv.org/pdf/2405.18560)) — models attraction/repulsion via a continuous potential field with many free proxies per class; TMP changes the proxy *parameterization* (tied additive offsets), not the interaction law, and its proxies share 512·K mode parameters across all classes where PFML's 15C proxies are free.
- **DADA** ([AAAI 2024](https://ojs.aaai.org/index.php/AAAI/article/view/29400), [arXiv 2401.00617](https://arxiv.org/pdf/2401.00617)) — domain-adapts the sample distribution to the proxy distribution; no multi-mode structure and no cross-class tying.
- **SoftTriple (ICCV 2019)** and **[Multi-proxy DML](https://www.sciencedirect.com/science/article/abs/pii/S0020025523007053)** / **[Hierarchical multiple proxy loss](https://www.sciencedirect.com/science/article/abs/pii/S1051200422004432)** — multiple *free* proxies (or main+sub proxies) per class with per-class merging regularizers; no parameter is shared across classes, which is precisely the transfer mechanism TMP adds.
- **Non-isotropic probabilistic proxy DML** ([ECCV 2022](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136860423.pdf)) and NIR/DVML/HORDE (memory-cited) — preserve or model intra-class spread per class, direction-free or per-class; none ties the *directions* of intra-class structure across classes.
- **Camera-aware proxies** ([AAAI 2021, arXiv 2012.10674](https://arxiv.org/abs/2012.10674)) — splits clusters into per-camera proxies using **camera labels** (metadata unavailable/illegal here) with free proxies; TMP discovers modes label-free and ties them additively under a norm budget.
- **Sub-center ArcFace (ECCV 2020)**, **[sub-center speaker embeddings](https://arxiv.org/html/2407.04291)** — free sub-centers for noise/diversity, no cross-class structure. **HIER (CVPR 2023)** — hierarchical ancestors, not shared mode offsets. **Proxy Synthesis (AAAI 2021)** / **MemVir (ICCV 2021)** — virtual *classes*, a different object entirely.

**Outside DML:**
- **Few-shot variance transfer** — hallucination ([Low-Shot Learning from Imaginary Data](https://arxiv.org/pdf/1801.05401)), [intra-class knowledge transfer](https://arxiv.org/pdf/2008.09892), [variational feature disentangling](https://arxiv.org/pdf/2010.03255), Distribution Calibration (ICLR 2021), Yin et al. CVPR 2019, Liu et al. CVPR 2020 long-tail angular-variance transfer (memory-cited) — all *generate features or calibrate statistics at adaptation time using novel-class labels*; TMP imposes the shared-variation prior as train-only proxy geometry with zero test-time machinery.
- **Common principal components** (Flury 1984), **PLDA** (Prince & Elder 2007) — shared within-class covariance as an *estimation/backend* model; TMP builds the shared structure into the supervision geometry to shape a neural encoder.
- **Style/content bilinear models** (Tenenbaum & Freeman 2000) and **Attributes-as-Operators / CZSL** (ECCV 2018) — additive/operator composition of *annotated* factors; TMP discovers latent modes without factor labels inside standard DML and deploys pure cosine retrieval. (Memory-cited items flagged as such.)

The shared-modes *prior* is classic; the claimed novelty is the **object**: budgeted additive tying of multi-proxy geometry inside a proxy-anchor loss, with an elastic residual ladder, as a train-only mechanism for zero-shot retrieval.

---

## 4. Decisive matched-compute controls

All controls share the identical recipe, data pipeline, schedule, and 5 seeds; only the proxy parameterization changes (compute is matched to <3%):

- **C1** PA baseline (K=1).
- **C2** **Free multi-proxy at same K** (γ=∞, ε unconstrained ⇒ SoftTriple/PFML-family endpoint). *The decisive comparison:* TMP − C2 isolates tying at identical proxy count and cost.
- **C3** Elastic ladder γ ∈ {0, 0.1, 0.5, 2, ∞}: dose–response of tying; mechanism predicts an interior or left-end optimum for zero-shot R@1.
- **C4** Shared offsets without norm budget (ρ=∞): tests budget necessity.
- **C5** Frozen random offsets (r fixed, not learned): tests learned shared structure vs mere multi-anchor jitter.
- **C6** Per-class scalar gates `p = normalize(q_c + s_ck·r_k)`: orientation tied, magnitude free — locates where tying matters.
- **C7** Mechanism audit on held-out train classes (never in the loss): assign each image to `k̂ = argmax_k⟨z − q̂_c, r_k⟩`; TMP must reduce cross-mode positive distances vs C2 while same-mode negative margins stay flat.

Decision rule: mechanism confirmed iff TMP > C2 and TMP > C5 on Cars and In-Shop, with a coherent C3 ladder and a positive C7 signature.

---

## 5. Frozen R@1 forecasts, frontier arithmetic, falsification

In-recipe reproduction commitments (5-seed means): PA — CUB 0.715, Cars 0.912, SOP 0.805, In-Shop 0.925 (±≈0.004); gate: if PA-Cars repro < 0.906, the recipe is debugged before any mechanism claim. C2 (free-K) — CUB 0.722, Cars 0.920, SOP 0.807, In-Shop 0.929.

| Dataset | TMP frozen forecast (90% int.) | Reference (Lane A) | Crossing arithmetic |
|---|---|---|---|
| **Cars196** (primary) | **0.931** [0.923, 0.938] | PFML 0.927 ± 0.003 | 0.912 (PA) + 0.8 (free-K, occupied) + **1.1 (tying, novel)** = 0.931 → +0.4 pt over reference mean; P(mean > 0.927) ≈ 0.6; decisive crossing declared only at ≥ 0.933 (≥2σ_ref) |
| **In-Shop** (primary) | **0.936** [0.929, 0.941] | PA+DADA 0.930 (seedless) | 0.925 + 0.4 + **0.7** = 0.936 → +0.6 pt; because the reference has no reported variance, crossing declared only at 5-seed mean ≥ 0.935 |
| CUB (secondary) | 0.734 [0.725, 0.742] | PFML 0.734 ± 0.003 | Parity forecast; **no crossing claimed** (deformable birds weaken the discrete-mode fit) |
| SOP (tertiary) | 0.812 [0.805, 0.818] | PFML 0.829 ± 0.002 | **No claim**: 2–5 images/class makes mode structure barely estimable (K=2); listed for completeness and no-harm check (TMP ≥ PA − 0.002) |

**Falsification thresholds (frozen).** F1: TMP − C2 < +0.3 pt on both Cars and In-Shop ⇒ mechanism refuted. F2: C3 ladder shows free (γ=∞) ≥ tied everywhere ⇒ refuted. F3: Cars 5-seed mean < 0.925 or In-Shop < 0.928 ⇒ frontier claim failed even if the mechanism effect is real. F4: no C7 cross-mode positive-distance reduction vs C2 on held-out classes ⇒ causal story refuted regardless of aggregate numbers. Honest joint odds: P(at least one decisive frontier crossing) ≈ 0.4–0.45; P(mechanism validated by controls) ≈ 0.55–0.6.

---

## 6. Costs, risks

**Cost.** Parameters: +512·(C + K + C·K) floats (CUB: ≈0.46 M vs PFML's ≈0.77 M free proxies — TMP is *smaller*). Compute: proxy similarities scale ×K (Cars: 784 pooled proxies; SOP: 22.6 k with K=2, same order as PA's 11.3 k) — epoch time ≈ 1.01–1.03× PA, memory ≈ 1.00×, inside DADA's 1.06×/1.01× envelope. Deployment identical to any Lane-A model: one ResNet-50, one view, one 512-D vector, cosine NN.

**Risks, stated plainly.** (1) The tying gain (+0.7–1.1 pt over free-K) is the speculative number; the elastic ladder exists because it may be smaller. (2) Sphere-additivity of mode response is a local-linear approximation; if binding, C6 (tied orientation, free magnitude) is the measured fallback and would weaken the claim. (3) In-Shop's reference is seedless, so its crossing threshold is necessarily conservative. (4) Recipe mismatch vs published references (batch, BN freezing, epochs) is mitigated by in-recipe controls and full recipe publication, but external comparison risk remains. (5) Contamination: known CUB test-class overlap with ImageNet-1K affects all lane methods equally (ImageNet init is lane-legal); Cars196 hosting/label-noise drift is a shared benchmark risk; TMP itself uses no external data, text, attributes, camera metadata, transduction, or test-side fitting — modes are discovered label-free from official training images only. (6) Shared modes may partially absorb class-independent photography nuisance (D4) — bounded but dilutive.

I sent the operator a completion ping via devbox-tell. Note also: several claude.ai MCP connectors (Gmail, Calendar, Drive, IBKR) are unauthenticated in this session and would need authorization via claude.ai connector settings to be usable — not needed for this task.

Sources: [PFML (arXiv 2405.18560)](https://arxiv.org/pdf/2405.18560), [DADA (AAAI 2024)](https://ojs.aaai.org/index.php/AAAI/article/view/29400), [DADA (arXiv 2401.00617)](https://arxiv.org/pdf/2401.00617), [Multi-proxy DML (Inf. Sciences 2023)](https://www.sciencedirect.com/science/article/abs/pii/S0020025523007053), [Hierarchical multiple proxy loss](https://www.sciencedirect.com/science/article/abs/pii/S1051200422004432), [Non-isotropic probabilistic proxy DML (ECCV 2022)](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136860423.pdf), [Camera-aware proxies (arXiv 2012.10674)](https://arxiv.org/abs/2012.10674), [Sub-center speaker embeddings (arXiv 2407.04291)](https://arxiv.org/html/2407.04291), [Low-shot imaginary data (arXiv 1801.05401)](https://arxiv.org/pdf/1801.05401), [Intra-class knowledge transfer (arXiv 2008.09892)](https://arxiv.org/pdf/2008.09892), [Variational feature disentangling (arXiv 2010.03255)](https://arxiv.org/pdf/2010.03255), [Proxy-Anchor loss](https://www.semanticscholar.org/paper/f7646cccbd6edbc148d08fea37e31bcd0592c992).
