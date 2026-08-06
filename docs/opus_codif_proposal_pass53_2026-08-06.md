<!-- Recovered verbatim from durable terminal partial of failed consultation job e368f4caf5454264; provider completed the answer before process exit 1. -->

I'll work as the blind proposer. Loading only public web tools, per the stated tool policy.I could run only two of my planned searches before the session budget was nearly exhausted, so I am disclosing verification status per claim below rather than pretending to a complete primary-source sweep.

---

# Proposal: **CoDiF** — Correlation-Dimension Flattening for zero-shot deep metric learning

**Lane chosen: Lane A** (ResNet-50, 512-D normalized descriptor, ~224 px, single-view cosine, 200 epochs). All forecasts and comparisons below are in Lane A only.

---

## 0. Verification status and unresolved source ambiguities (stated up front)

I ran this blind and could not verify the following. None of it is assumed away:

- **PFML (CVPR 2025):** I could not retrieve its loss, proxy update rule, optimizer, LR schedule, batch size, or pooling. I therefore **do not build on PFML and do not inherit 0.734 / 0.927 / 0.829.** My base is a separately specified multi-proxy anchor loss (MPA, §1.2) that I define completely and must reproduce myself.
- **Proxy Anchor (CVPR 2020) recipe:** recalled from memory as AdamW, lr 1e-4, weight decay 1e-4, proxy LR ×100, α=32, δ=0.1, batch 180, 512-D, 1-epoch embedding warmup with frozen backbone, RandomResizedCrop(224)+flip, and an official implementation that **sums global average and global max pooling** rather than using GAP alone. I specify GAP-only for lane cleanliness; this is a **deviation** and requires a matched reproduction before any comparison — I do not inherit any published Proxy-Anchor number.
- **DADA / PA+DADA:** seed count and In-Shop protocol unreported (as stated in the brief); augmentation/crop parity with my runs unknown.
- **AdvRF, VAPNet, CRT:** Lane B / transformer lane; not used.

---

## 1. The method

### 1.1 Deployed system (unchanged from baseline)

Backbone $\phi_\theta$: ResNet-50, ImageNet-1K init. Global average pool → $z\in\mathbb R^{2048}$. Head $W\in\mathbb R^{512\times 2048}$ (no bias). $f=Wz$, deployed descriptor $u=f/\lVert f\rVert_2\in S^{511}$. Test time: one model, one view, one 512-D descriptor, cosine kNN. **CoDiF adds zero deployed parameters and zero deployed FLOPs.**

### 1.2 Base loss (MPA — fully specified, reproduced by me, not inherited)

$K$ proxies per class, $\hat p_{c,k}=p_{c,k}/\lVert p_{c,k}\rVert$. Class score by smooth-max over that class's proxies:

$$s_c(u)=\tfrac1\gamma\log\sum_{k=1}^{K}\exp\!\big(\gamma\,\langle u,\hat p_{c,k}\rangle\big),\qquad \gamma=10$$

Proxy-Anchor form on class scores:

$$\mathcal L_{\mathrm{MPA}}=\frac{1}{|\mathcal C^+|}\sum_{c\in\mathcal C^+}\log\Big(1+\!\!\sum_{u\in U_c^+}\!\!e^{-\alpha(s_c(u)-\delta)}\Big)\;+\;\frac1C\sum_{c=1}^{C}\log\Big(1+\!\!\sum_{u\in U_c^-}\!\!e^{\alpha(s_c(u)+\delta)}\Big)$$

$\alpha=32$, $\delta=0.1$. $K=15$ (CUB, Cars), $K=2$ (SOP, In-Shop) — matching the proxy counts disclosed for PFML so the *proxy-count* axis is held fixed and cannot explain any delta. AdamW, lr $10^{-4}$, wd $10^{-4}$, proxy LR ×100, cosine decay, 200 epochs, $P{=}30$ classes × $K_s{=}6$ images = batch 180, 1-epoch head warmup with frozen backbone.

### 1.3 The new objective

Two independent augmentations per image: $u_i^{(1)},u_i^{(2)}$, $i=1..B$.

**(a) Nuisance floor.** $\varepsilon_i=\lVert u_i^{(1)}-u_i^{(2)}\rVert_2$, $\ \varepsilon=\mathrm{median}_i\,\varepsilon_i$.

**(b) Cross-view pair distances.** $d_{ij}=\lVert u_i^{(1)}-u_j^{(2)}\rVert_2$ for all $i\neq j$ ($B(B-1)$ ordered pairs). Cross-view is essential: every pair already carries one nuisance displacement, so nuisance-driven "structure" cannot masquerade as semantic fine structure (see D1).

**(c) Log-spaced scale grid, anchored at the floor.** $\theta_m=\varepsilon\rho^m$, $m=0..M$, $M=8$, $\rho=(\theta_{\max}/\varepsilon)^{1/M}$, with $\theta_{\max}=\mathrm{Quantile}_{0.60}\{d_{ij}\}$. **$\varepsilon$ and $\theta_{\max}$ are `stop_grad` for grid placement** (see D3).

**(d) Smoothed correlation integral, with scale-proportional bandwidth.**

$$C_m=\frac{1}{B(B-1)}\sum_{i\neq j}\sigma\!\Big(\frac{\theta_m-d_{ij}}{\beta\,\theta_m}\Big),\quad \beta=0.15,\qquad g_m=\log(C_m+\eta),\ \ \eta=\tfrac{1}{2B(B-1)}$$

The bandwidth $\beta\theta_m$ (not a constant $h$) is what makes the *estimator itself* scale-free; a fixed bandwidth would inject a characteristic scale and defeat the objective.

**(e) Scale-freeness (curvature) penalty — the core term.**

$$\boxed{\ \mathcal L_{\mathrm{flat}}=\frac{1}{M-1}\sum_{m=1}^{M-1}\Big(\frac{g_{m+1}-2g_m+g_{m-1}}{(\log\rho)^2}\Big)^{2}\ }$$

This is the discrete second derivative of $\log C$ with respect to $\log\theta$. It is zero iff $C(\theta)\propto\theta^{D}$ over $[\varepsilon,\theta_{\max}]$ — i.e. iff the embedded data has **no characteristic scale** between the nuisance floor and the class scale. Equivalently (Hill / Levina–Bickel): $\log d$ is memoryless over that range.

**(f) Dimension floor (anti-fractal-thread).** $\bar D=(g_M-g_0)/(M\log\rho)$; $\ \mathcal L_{\dim}=\max(0,\ D^\*-\bar D)^2$, with $D^\*=\kappa D_0$, $\kappa=1$, and $D_0$ = the same correlation-dimension estimator applied **once, before training, to the frozen ImageNet-1K GAP features of the training images only** (no test data, no external data). Interpretation: *fine-tuning may not lose dimensional resolution relative to the permitted pretrained prior.*

**(g) Intra-class closure (anti-heterogeneity).** Same construction restricted to same-class cross-view pairs (pooled over the batch's 30 classes, $30\cdot6\cdot5=900$ pairs), grid $M'=4$ from $\varepsilon$ to $\mathrm{Quantile}_{0.90}$ of those distances → $\mathcal L_{\mathrm{flat}}^{\mathrm{intra}}$. Prevents "some classes collapsed, some spread" from faking a smooth pooled curve (D7).

**(h) Floor-lowering term.** $\mathcal L_\varepsilon=\frac1B\sum_i\varepsilon_i^2$. This is an ordinary positive-pair alignment term; I state that plainly, and it gets its own isolated control (C2) precisely because it is *not* novel. Its role is to extend the scale-free range downward in decades.

**Total.**

$$\mathcal L=\tfrac12\big(\mathcal L_{\mathrm{MPA}}^{(1)}+\mathcal L_{\mathrm{MPA}}^{(2)}\big)+\lambda_f\big(\mathcal L_{\mathrm{flat}}+\tfrac12\mathcal L_{\mathrm{flat}}^{\mathrm{intra}}\big)+\lambda_d\mathcal L_{\dim}+\lambda_\varepsilon\mathcal L_\varepsilon$$

$\lambda_f=0.5$, $\lambda_d=1.0$, $\lambda_\varepsilon=0.5$; $\lambda_f,\lambda_d$ ramp linearly $0\to$ full over epochs 5→25 (class structure must exist before its scale spectrum is meaningful), constant after.

**Gradient paths.** $\partial\mathcal L_{\mathrm{flat}}/\partial d_{ij}=\sum_m w_m\cdot\big(-\sigma'(\cdot)/(\beta\theta_m)\big)$, so each grid term is driven almost entirely by the pairs sitting *at* scale $\theta_m$ — the penalty is a genuinely multiscale, per-band force, not a global rescaling. Gradients flow into $u^{(1)},u^{(2)}$, hence $W$ and $\theta$. $\bar D$ carries gradient only through $g_0,g_M$. Grid placement carries none. Proxies receive no CoDiF gradient.

**Scale is operational, not cosmetic.** $\lambda_f,\lambda_d,\lambda_\varepsilon$ interact with AdamW weight decay on $W$ and with the fixed $\alpha,\delta$ of the base loss; I do not treat any normalization as free. Every ablation re-tunes nothing else, and the $\lambda_f$ sweep is a required deliverable, not an appendix.

---

## 2. The causal zero-shot error mode, and a proof-level attack on the degeneracies

### 2.1 Error mode: **the scale gap**

Let $a=\mathrm{median}\{d(u_i,u_j):y_i=y_j\}$ and $b=\mathrm{median}\{d:y_i\ne y_j\}$ on training classes. Proxy losses drive $a/b\to0$ (neural collapse NC1) because that is the unique direction of monotone loss decrease once classes are separated.

Now take two unseen classes $t,t'$. Nothing in training ever required a direction separating them; the model represents both as variation *within* the region it learned for whichever training classes they resemble. Their separation is therefore realized at scale $\sim a$ — exactly the scale the loss has been actively driving to zero, and whatever survives there is dominated by nuisance (it is the only variation with no restoring force). Retrieval rank for $t$ is then decided inside a cluster of width $a$ whose internal structure has been made unstructured. R@1 saturates *independently of how well training classes are separated* — which is why pushing training-class separation harder yields diminishing returns on this benchmark family.

CoDiF forbids the gap directly: a gap between $a$ and $b$ *is* a plateau in $C(\theta)$, and a plateau *is* curvature in $\log C$ vs $\log\theta$.

### 2.2 Why existing anti-collapse regularizers cannot see this

**Lemma (second-moment blindness).** Let the batch be in exact NC1/NC2: zero within-class variance, $C$ centroids forming a simplex ETF spanning $\mathbb R^{C-1}\subset\mathbb R^{512}$. Then the batch second-moment matrix equals the centroid second-moment matrix $\frac1C\sum_c p_cp_c^\top=\frac{1}{C-1}\Pi$, whose nonzero spectrum is **exactly flat with multiplicity $C-1$**. Consequently: effective rank $\exp H(\tilde\sigma)=C-1$ (its maximum for $C$ classes); soft-effective-rank regularizers are at their optimum; covariance-decorrelation terms are exactly satisfied; spectral-decay $\rho$ is at its best attainable value; total coding rate of the set is maximal for $C$ atoms. **Every regularizer that is a function of the batch second-moment matrix or its singular spectrum attains its optimum on the configuration with exactly zero zero-shot resolution below the class scale.** $\square$

Under the same configuration the pooled pair-distance law is $w\,\delta_0+(1-w)\,\delta_{\sqrt2 s}$ with $w$ fixed by the sampler ($\approx0.028$ at $P{=}30,K_s{=}6$). Its $\log C$ vs $\log\theta$ curve is a two-step staircase, so $\lVert\Delta^2 g\rVert=\Omega(1)$ on any grid spanning $[\varepsilon,\theta_{\max}]$. CoDiF is maximally violated exactly where the spectral family is maximally satisfied. This is the mechanism claim, and C5 is its empirical test.

Single-scale *pairwise-potential* regularizers (Gaussian-kernel uniformity) are not literally blind — the atom at 0 contributes — but they carry a characteristic scale $t^{-1/2}$ by construction, weight the atom only by $w$, and have the **class-agnostic uniform measure as their optimum**, so they oppose the proxy loss globally instead of forbidding a plateau locally. C4 tests this.

### 2.3 Degeneracy attacks

**D1 — Nuisance inflation** (retain augmentation info to spread same-class points). Model the cheat as adding independent isotropic noise of scale $\varepsilon$ in $D_n$ effective dimensions: $d_{ij}^2\approx\tilde d_{ij}^2+\varepsilon^2$. Collapsed pairs then concentrate at $\varepsilon\sqrt2$ with relative width $O(D_n^{-1/2})$ — the step is **relocated, not removed**, and $\lVert\Delta^2g\rVert$ stays $\Omega(1)$. A step is a step at any location. Additionally the grid is anchored at $\varepsilon$, so inflating $\varepsilon$ only truncates the range from below while $\mathcal L_\varepsilon$ pushes back. *Nuisance inflation is strictly penalized, not rewarded.*

**D2 — Fractal thread** (put everything on a 1-D geometrically-spaced curve: zero curvature, $D\approx1$). Blocked by the $\mathcal L_{\dim}$ floor at $D^\*\approx D_0$ (ImageNet R50 GAP features: empirically $\mathcal O(20\text{–}40)$). C6a removes $\mathcal L_{\dim}$ and must reproduce this failure; if it does not, $\mathcal L_{\dim}$ is dead weight and should be dropped.

**D3 — Grid gaming** (move $\varepsilon$ or $\theta_{\max}$ instead of fixing geometry). Blocked by `stop_grad` on both. C6b re-enables the gradient and must degrade; if it does not, the constraint is not binding and the whole construction is suspect.

**D4 — Batch-composition gaming.** $w$ is fixed by the class-balanced $P\times K_s$ sampler, so the model can change the *geometry* but not the *mixture weights*.

**D5 — Uniformity takeover** (satisfy flatness by mapping to a uniform $D^\*$-dimensional measure — exact power law, zero curvature, no class structure). This is *not* blocked by construction; it is blocked by $\mathcal L_{\mathrm{MPA}}$, and the existence of a useful interior equilibrium is the substantive empirical claim. Falsifier F2 makes it decidable: a monotone $\lambda_f$ sweep with no interior optimum refutes the method.

**D6 — Junk subspace** (produce scale-free structure in directions orthogonal to the discriminative subspace). Not free: if a fraction $\omega$ of squared norm goes to junk, all class cosine similarities shrink by $\sqrt{1-\omega}$, and $\mathcal L_{\mathrm{MPA}}$ at fixed $\alpha,\delta$ pays $\approx\alpha\delta\,\omega/2$ per pair — a first-order cost. But partial junk is still possible, so this needs a **direct semantic test**, not an argument: **F3, the held-out-training-class probe** — train on 50 of CUB's 100 training classes, evaluate R@1 on the *other 50 training classes* (real zero-shot conditions, zero test contamination). If the preserved fine structure is semantic, this must improve; if it is junk, it will not.

**D7 — Radius heterogeneity.** Closed by $\mathcal L_{\mathrm{flat}}^{\mathrm{intra}}$ (§1.3g); C6c ablates it.

---

## 3. Adversarial novelty search and one-sentence distinctions

**Verified in this session:**

| Nearest work | Distinction |
|---|---|
| **IDRR — Intrinsic Dimension Regularization** (soft effective rank = exp-entropy of normalized singular values, two-sided) | Spectral and single-scale; the Lemma shows it is at its optimum on exact neural collapse, whereas CoDiF constrains the *shape of the log–log pair-correlation curve*, which no second-moment functional can see. |
| **VICReg / variance–covariance regularization** | Per-coordinate variance floor plus off-diagonal decorrelation at one scale; CoDiF has no per-coordinate term, no target covariance, and no expander. |
| **HORDE (ICCV 2019), high-order moment regularizer** | Matches/repels high-order moments of *class-conditional* distributions; CoDiF says nothing about class-conditional distributions and instead forbids a plateau in the *pooled* distance spectrum. |
| **Uniform Priors for Data-Efficient Transfer / Wang–Isola uniformity** | Single-scale Gaussian-potential energy whose optimum is the class-agnostic uniform measure; CoDiF's optimum set contains strongly clustered measures, so it composes with the proxy loss rather than opposing it. |
| **Confusion-based metric learning (energy + diversity confusion)** | Blurs class boundaries with adversarial/energy terms; CoDiF never blurs a boundary, it removes the *empty band* between boundaries. |

**From memory, unverified in this consultation — flagged as such:**

| Nearest work | Distinction |
|---|---|
| ρ-spectral-decay regularization (Roth et al., ICML 2020) | Its intervention is a *tuple-sampling* change (negatives swapped to positives) and its statistic is a linear spectrum; CoDiF changes no sampling and uses a non-spectral multiscale statistic. |
| SoftTriple / multi-proxy / PFML | They raise the rank of the *learned class-parameter span* to model intra-class multimodality; CoDiF adds zero parameters and constrains the geometry of the *data*, and the two compose (K is held fixed across all my arms). |
| S2SD / self-distillation on batch manifolds | Requires an auxiliary higher-dimensional space and a teacher target; CoDiF has neither. |
| MMCR (NeurIPS 2023, manifold capacity via nuclear norm) | Self-supervised, spectral, and *compresses* augmentation manifolds to raise capacity; CoDiF is supervised, non-spectral, and deliberately *preserves* within-class extent. |
| Grassberger–Procaccia correlation dimension; Levina–Bickel / Hill MLE; LID methods (adversarial detection, D2L noisy-label stopping) | All use these estimators as *measurements or stopping criteria*; CoDiF makes the scale-derivative of the correlation integral a differentiable training objective and — the distinctive part — penalizes its **curvature** rather than its value. |
| Renormalization-group self-similarity (physics) | The import is the objective's functional form (absence of a characteristic scale = RG fixed-point condition), not a metaphor. |

**Honest adjacency statement:** "preserve intra-class variance" is a crowded neighbourhood. CoDiF's claim to distinctness rests entirely on the Lemma — that the occupied methods are single-scale/spectral and are *satisfied by* the failure configuration — and control **C3** is designed to kill CoDiF if that claim is wrong.

---

## 4. Decisive matched-compute controls

All at 5 seeds, identical schedule, identical $K$, identical augmentation, identical epoch count.

- **C0** MPA, 1 view, 200 ep (published-style reference point).
- **C1** MPA, **2 views**, no regularizer — *the* compute control. CoDiF's delta is measured against C1, never C0.
- **C2** C1 + $\mathcal L_\varepsilon$ only (isolates the ordinary alignment term).
- **C3** C1 + a **single-scale intra-class variance floor**, tuned so the final $a/b$ ratio *matches CoDiF's measured $a/b$*. If this recovers ≥70% of the gain, the multiscale claim is dead and CoDiF is just variance preservation (**F4**).
- **C4** C1 + uniformity loss ($t=2$), tuned.
- **C5** C1 + soft-effective-rank (IDRR-style) regularizer, tuned — the empirical form of the Lemma.
- **C6** CoDiF minus $\mathcal L_{\dim}$ (a) / with grid gradients enabled (b) / minus $\mathcal L^{\mathrm{intra}}_{\mathrm{flat}}$ (c).
- **C7** CoDiF with 2× batch instead of 2 views (equal FLOPs, different gradient variance).
- **C8 — scale-shuffle placebo.** Apply the identical curvature penalty to a **random permutation of the grid indices**, destroying the scale ordering while preserving magnitude, gradient count, and gradient noise. If this recovers ≥50% of the gain, the effect is generic gradient noise, not scale structure (**F5**). This is the cheapest and most dangerous control; it runs first.

Reported diagnostics (never optimized): $\bar D$, $\lVert\Delta^2 g\rVert$, $\varepsilon$, $a/b$, NC1 scatter ratio, effective rank.

---

## 5. Frozen forecasts, falsifiers, frontier arithmetic (Lane A, 5 seeds, R@1)

**Frozen forecasts — ResNet-50, 512-D, 224 px, single view, cosine, 200 epochs:**

| | CUB | Cars196 | SOP | In-Shop |
|---|---|---|---|---|
| MPA, 1 view (C0) | 0.702 ± 0.005 | 0.893 ± 0.005 | 0.807 ± 0.003 | 0.913 ± 0.004 |
| **MPA, 2 views (C1)** | **0.706 ± 0.005** | **0.897 ± 0.005** | **0.809 ± 0.003** | **0.916 ± 0.004** |
| **MPA + CoDiF** | **0.727 ± 0.005** | **0.913 ± 0.005** | **0.817 ± 0.003** | **0.924 ± 0.004** |
| Δ vs C1 | **+2.1** | **+1.6** | **+0.8** | **+0.8** |

SOP's small delta is predicted by the mechanism, not excused after the fact: with 11 318 training classes at ~5 images each there is little room for a scale gap to form.

**Frontier arithmetic (explicit, and it does not cross):**

- CUB: 0.727 − 0.734 (PFML) = **−0.007**. Does not cross. Vs DADA's matched-cost row 0.729: **−0.002**, inside noise.
- Cars: 0.913 − 0.927 = **−0.014**. Does not cross. Vs DADA 0.921: −0.008.
- SOP: 0.817 − 0.829 = **−0.012**. Does not cross. Vs DADA 0.810: **+0.007**.
- In-Shop: 0.924 − 0.930 (PA+DADA) = **−0.006**. Does not cross (and that reference's seed count is unreported).

**Conditional composition, explicitly not claimed.** CoDiF is additive to any proxy loss and adds no parameters, so the natural composition is CoDiF ⊕ PFML. Assuming attenuation to ~0.4× because PFML's 15 proxies/class already partially populate the intra-class scale: CUB 0.734+0.008 = **0.742**, Cars 0.927+0.006 = **0.933**, SOP 0.829+0.003 = **0.832**. **This is conditional on a PFML recipe I could not verify and did not reproduce; I am not entitled to it and do not assert it.** The scientifically decisive numbers are the C1 deltas above.

**Falsification thresholds (pre-registered):**

- **F1** Δ(CoDiF − C1) < +1.0 on CUB **or** < +0.7 on Cars ⇒ refuted at claimed strength.
- **F2** $\lambda_f\in\{0,0.1,0.25,0.5,1,2\}$ shows no interior optimum ⇒ refuted (D5 won).
- **F3** Held-out-training-class probe (train 50 CUB train classes → eval R@1 on the other 50) improves by < +1.0 ⇒ the preserved fine structure is not semantic ⇒ refuted (D6 won).
- **F4** C3 recovers ≥70% of the gain ⇒ multiscale claim refuted; the method reduces to a variance floor.
- **F5** C8 scale-shuffle placebo recovers ≥50% ⇒ effect is gradient noise; abandon.
- **F6** Diagnostics show $\lVert\Delta^2g\rVert$ falls as intended but $a/b$ and R@1 do not move ⇒ the regularizer is being satisfied in a junk subspace; abandon.

---

## 6. Cost, and benchmark / contamination risk

**Training cost.** Two views ⇒ ~1.9–2.0× forward+backward per step at fixed steps; ~1.9× activation memory at fixed batch (or hold memory flat via gradient accumulation across views — this changes BatchNorm statistics and must then be applied to C1 identically; disclosed). The regularizer itself reuses the $B\times B$ distance matrix and costs $M{+}M'$ sigmoid passes over it (≈3×10⁵ ops/step at $B{=}180$), **< 0.5% overhead**. Zero extra learned parameters. This is materially more expensive than PA+DADA's reported ~1.06× epoch time and must be reported as such — CoDiF is not a cheap add-on, and C1/C7 exist so the cost is never mistaken for the mechanism.

**Deployment cost.** Identical to baseline: one ResNet-50, one view, one 512-D descriptor, ordinary cosine kNN. No reranking, no transduction, no test-gallery fitting.

**Contamination and benchmark risks.**
- $D_0$ is measured on **training images only** with ImageNet-1K-pretrained features. That is permitted initialization, but it makes one hyperparameter inherit the pretraining prior more explicitly than baselines do. Mitigation: also report $D^\*$ fixed at a constant (20) and $D^\*$ from a leave-classes-out estimate on training data; the gain must survive both, or $\mathcal L_{\dim}$ is doing hidden work.
- ImageNet-1K contains bird and car categories; this contaminates CUB/Cars zero-shot claims for **every** method in this lane, including all references. It is a shared floor, not a differential advantage, but it should not go unstated.
- CUB seed noise is ~±0.3–0.5 R@1; single-seed comparisons in this lane are not interpretable. All arms are 5 seeds; the ±0.005 bands above are forecasts of seed SD, not confidence intervals.
- SOP and In-Shop contain near-duplicate images; standard splits are used unmodified and no dedup is applied.
- Grid quantiles are computed per training batch only; no test statistic ever enters training.

**Where I most expect this to fail.** D5 (the uniformity takeover) and D6 (the junk subspace) are the two live risks; F2 and F3 are their decision procedures. If CoDiF fails, my prediction is that it fails by F4 — the multiscale statistic turns out to be a roundabout intra-class variance floor — and in that case the honest outcome is a negative result about the *sufficiency* of second-moment anti-collapse regularizers, not a repair.

Sources: [Learning Compact Representations via Intrinsic Dimension Regularization](https://openreview.net/forum?id=1Knlr5MCCw), [Variance-Covariance Regularization Improves Representation Learning](https://arxiv.org/html/2306.13292v2), [Metric Learning With HORDE: High-Order Regularizer for Deep Embeddings](https://openaccess.thecvf.com/content_ICCV_2019/papers/Jacob_Metric_Learning_With_HORDE_High-Order_Regularizer_for_Deep_Embeddings_ICCV_2019_paper.pdf), [Uniform Priors for Data-Efficient Transfer](https://arxiv.org/pdf/2006.16524), [Confusion-Based Metric Learning for Regularizing Zero-Shot Image Retrieval and Clustering](https://pubmed.ncbi.nlm.nih.gov/35834457/), [Self-distillation with Online Diffusion on Batch Manifolds Improves Deep Metric Learning](https://arxiv.org/pdf/2211.07566)
