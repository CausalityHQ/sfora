**Provider note:** executed as the blind proposer under `docs/opus_blind_prompt_pass40_2026-08-06.txt`. I read only that file, inspected no repository or process state, and used web/primary literature for the novelty search as the prompt directs. Below is the single proposal.

---

# PROPOSAL — CNW: Conditional Nuisance Whitening

**Lane declaration: Lane A** (ResNet-50, 512-D normalized descriptor, ~224 px, single-view cosine, 200 epochs). All forecasts, baselines, and comparisons below are Lane A. Lane B is not used anywhere in this document.

**One-line statement.** Deployment fixes the test metric to be *isotropic* (cosine/Euclidean), but the Bayes-optimal retrieval metric is Mahalanobis in the identity-conditional (within-identity) covariance. CNW is a train-time, scale-free, basis-free loss that forces the encoder's identity-conditional residual spectrum to be flat, so that the metric we are forced to deploy is the metric that is optimal — at unseen identities, where no fitted matrix exists.

---

## 1. Executable mathematics

### 1.1 Objects

Backbone `φ_θ`: ResNet-50, ImageNet-1K initialized, global average pool → 2048-D. Embedding layer `W_e ∈ R^{512×2048}`, bias `b ∈ R^{512}` (no BN on the embedding).

```
e(x) = W_e · GAP(φ_θ(x)) + b        e ∈ R^d,  d = 512
z(x) = e(x) / ‖e(x)‖₂                z ∈ S^{d−1}     ← the deployed descriptor
```

Deployment: one model, one view, `z`, cosine NN. Nothing below survives training.

Base loss: **Proxy-Anchor** (Kim et al., CVPR 2020), learned proxies `p_c ∈ R^d`, `p̂_c = p_c/‖p_c‖`, `s(z,p) = z·p̂`:

```
L_PA = (1/|P⁺|) Σ_{c∈P⁺} log(1 + Σ_{z∈Z⁺_c} e^{−α(s(z,p_c) − δ)})
     + (1/|P|)  Σ_{c∈P}  log(1 + Σ_{z∈Z⁻_c} e^{ α(s(z,p_c) + δ)})
```
with `α = 32`, `δ = 0.1` (primary source values).

### 1.2 The CNW term

Batch `B` images drawn by **class-balanced sampling**: `C_b` classes × `m = 4` images (`m_c = min(4, n_c)`, classes with `n_c = 1` excluded from the sampler).

```
per class c:  μ_c = (1/m_c) Σ_{i∈c} z_i                (not renormalized)
residual:     r_i = z_i − μ_c
gate:         K = { i : ‖r_i‖ ≥ τ_r },   τ_r = 0.05
direction:    r̂_i = r_i / sqrt(‖r_i‖² + ε²),   ε = 10⁻²
second mom.:  A = Σ_{i∈K} r̂_i r̂_iᵀ                    (never materialized; see §1.4)
part. ratio:  D̂ = tr(A)² / ‖A‖_F²
```

```
L_CNW = log D_max − log D̂ ,      D_max = min(d, B − C_b)
L     = L_PA + λ(t) · L_CNW ,    λ = 1.0
λ(t)  = λ · min(1, t / T_warm),  T_warm = 5 epochs
```

`L_CNW ∈ [0, log D_max]`, equal to 0 **iff** the sampled residual directions are spectrally flat over a `D_max`-dimensional subspace. `D_max` is the exact rank ceiling: each class's `m_c` residuals sum to zero, so `rank(A) ≤ Σ_c(m_c − 1) = B − C_b`. Reporting `L_CNW` against this ceiling rather than against `d` is what makes `λ` transfer across `B, m, d`.

### 1.3 Gradient path (explicit)

```
∂L_CNW/∂r̂_i = 4·A r̂_i / ‖A‖_F²  −  4· r̂_i / tr(A)
∂r̂_i/∂r_i   = ( I − r̂_i r̂_iᵀ · ‖r_i‖²/(‖r_i‖²+ε²) ) / sqrt(‖r_i‖²+ε²)   ≈ (I − r̂_i r̂_iᵀ)/‖r_i‖
∂r_i/∂z_j   = δ_ij I − (1/m_c) I      for j in the same class c, else 0
∂z/∂e       = ( I − z zᵀ ) / ‖e‖
```

Read the first line: each residual is pushed **away from** the dominant nuisance eigendirections (`A r̂_i`) against a uniform restoring term; the stationary point is `A r̂_i = (‖A‖_F²/tr A) r̂_i ∀i`, i.e. all residual directions are equal-eigenvalue eigenvectors. Proxies receive **no** gradient from `L_CNW` — the two terms are cleanly separated.

**Structural property (matters for §4).** `L_CNW` depends on `r` only through `r̂`, so `∂L_CNW/∂r_i ⊥ r_i` exactly. **CNW cannot inflate or shrink `tr(Σ_W)`; it only rotates.** `L_PA` controls magnitudes, CNW controls directions, and the two gradients are geometrically orthogonal at every residual. CNW is therefore not a reweighted intra-class-compactness term — it is orthogonal to the entire family.

**Second structural property.** Because residuals are unit-normalized, and because `L_PA` has already driven the residual component *inside* the seen-discriminative subspace `S` toward zero, `r̂_i ≈ P_{S⊥}r_i / ‖P_{S⊥}r_i‖`. The normalization automatically aims the constraint at exactly the subspace the base loss ignores. This is not a tuning choice; it falls out of the construction.

### 1.4 Computation

`A` is never formed. With `G_ij = r̂_i·r̂_j` (Gram, `|K|×|K|`):

```
‖A‖_F² = ‖G‖_F² = Σ_ij G_ij²        tr(A) = Σ_i G_ii
```

Cost: `|K|²d ≈ 90²·512 ≈ 4.1 MFLOP` forward plus similar backward, against `≈1.6 TFLOP` for a `B=120` ResNet-50 fwd+bwd at 224 px — a ratio of `~3×10⁻⁶`. Extra memory: `|K|²` floats (~32 KB). **Zero extra forward passes; no second view; no auxiliary network; no generated data.**

### 1.5 Full recipe (frozen)

| | CUB / Cars | SOP / In-Shop |
|---|---|---|
| Batch `B`, `m` | 120, 4 (`C_b`=30, `D_max`=90) | 180, 4 (`C_b`=45, `D_max`=135) |
| Optimizer | AdamW, wd 1e-4 | same |
| LR | 1e-4 backbone+embedding, ×100 proxies | same |
| Schedule | 200 ep, cosine, 5-ep linear warmup | same |
| BN | frozen (per PA primary source) | frozen |
| Train aug | RandomResizedCrop(224, scale 0.16–1) + hflip | same |
| Test | resize 256 → center-crop 224, one view | same |
| `λ` | 1.0 (grid {0.25, 0.5, 1, 2, 4}) | 1.0 |
| `τ_r`, `ε`, `T_warm` | 0.05, 1e-2, 5 ep | same |

`λ` is selected **only** on a class-disjoint pseudo-unseen split carved from the training classes (last 20% of train class IDs held out), then the model is retrained on the full training split. No test-split selection, no gallery fitting, no reranking, no transduction.

**Baseline-recipe honesty.** Proxy-Anchor's public repo uses *random* sampling for the R50/512 configuration (batch 120, lr 1e-4, 5 warm-up epochs, BN freezing, step LR decay every 5 epochs on CUB / 10 on Cars). CNW requires `m ≥ 2` per class, which is a recipe change. **The matched baseline must therefore also use class-balanced `m=4` sampling**, and I run the baseline under both the official step schedule at the official epoch count and the 200-epoch cosine schedule, taking the stronger of the two as the baseline. I do not inherit any published PA number.

---

## 2. One causal zero-shot error mode, and the degeneracy attack

### 2.1 The error mode: Conditional Anisotropy Mismatch

Let `Σ_W` be the identity-conditional covariance of `z` (within-identity variability: pose, lighting, background, viewpoint, part occlusion). Let `Δμ` be the mean separation of two *unseen* identities, `u = Δμ/‖Δμ‖`. For prototype/NN retrieval the error is `ε = Φ(−SNR)` with

```
SNR ≈ ‖Δμ‖ / (2 sqrt( uᵀ Σ_W u ))
```

— the specialization of the geometric SNR of Sorscher, Ganguli & Sompolinsky (PNAS 2022), whose four governing quantities are signal, bias, manifold radius and manifold *dimension*; the radius/dimension pair enters exactly as the directional variance `uᵀΣ_W u`.

Now the causal chain:

1. Proxy and margin objectives are **direction-blind**: they are functions of `s(z, p_c)` only, so they constrain `P_S r` (the residual inside the seen-discriminative subspace `S`, `dim S ≤ C_train − 1`) and place **no constraint whatsoever** on the shape of `Σ_W` outside `S`. This blindness is not my claim — it is the observation Roth, Vinyals & Akata (CVPR 2022) make when they note proxy methods "solely optimize for sample-proxy distances" and that the induced local sample distribution is unresolved by the objective.
2. Consequently `Σ_W` inherits whatever anisotropy the backbone and data happen to produce. Empirically the conditional participation ratio `D̂` is `O(10)` out of 512.
3. At test time the deployed metric is fixed and isotropic. The optimal metric is Mahalanobis in `Σ_W`. **Every unit of anisotropy in `Σ_W` is a unit of mismatch between the metric we deploy and the metric that is correct.**
4. For unseen identities the mismatch is unmitigated: no post-hoc matrix can be fitted (that would be test-gallery fitting), and any matrix fitted on training classes only rescales directions that still carry signal — it cannot restore what the encoder has already collapsed.

**Why the standard fixes miss it.** Intra-class compactness terms reduce `tr(Σ_W)`, and their gradient lives where the loss has purchase — inside `S`. They make the already-suppressed directions more suppressed and leave the un-suppressed ones untouched, i.e. they *increase* anisotropy while improving the seen-class objective. Uniformity/entropy regularizers (including Roth et al.'s ρ-regularization, ICML 2020) act on the **marginal** spectrum; a flat marginal is achievable purely by spreading class means and is consistent with an arbitrarily anisotropic `Σ_W`. The two objects are independent.

### 2.2 The two arguments that isotropy is the zero-shot optimum

**(i) Jensen / uniform-prior argument (the strong one — it does not need any premise about where novel signal lies).** Hold `tr(Σ_W) = c` fixed (CNW does, exactly, by §1.3). Let `u` be a priori uniform on `S^{d−1}`. Then `E_u[uᵀΣ_W u] = c/d` for **every** `Σ_W` with that trace. The expected error is `E_u[ ε(uᵀΣ_W u) ]` where `ε(v) = Φ(−a v^{−1/2})`, `a = ‖Δμ‖/2`. In the operating regime (`a/√v ≳ 1.5`, i.e. R@1 well above chance) `ε(v)` is strictly convex in `v`. By Jensen, `E_u[ε(v)] ≥ ε(E_u[v]) = ε(c/d)`, with equality **iff `v` is a.s. constant, i.e. iff `Σ_W ∝ I`**. Isotropy is the unique minimizer of expected zero-shot error at fixed conditional variance budget.

**(ii) Minimax argument.** `min_Σ max_{‖u‖=1} uᵀΣu` s.t. `tr Σ = c` is solved uniquely by `Σ = (c/d)I`. Under complete ignorance of the novel-class discriminative direction, isotropy is worst-case optimal.

**Amplifier premise (not required, but tested).** Novel-class discriminative directions in fine-grained data are not uniform — they concentrate in exactly the directions that also carry within-identity variability (throat-patch colour on a warbler varies both between species and with illumination/pose). Define the overlap `o = (uᵀΣ_W u)/(tr(Σ_W)/d)`. Where `o > 1`, isotropization strictly reduces the noise along the signal. The gain scales with `E[o] − 1`. This premise is measurable and is falsification test **F3** below; if it fails, arguments (i) and (ii) still hold but the effect size shrinks toward the Jensen floor.

### 2.3 Proof-level attack on the cheapest degeneracies

**D1 — Isotropic-noise injection.** The cheapest imaginable cheat: add an identity-independent isotropic component `n`, per-dim variance `σ_n²/d`, to swamp the real anisotropic nuisance `Σ_a = (σ_a²/k)P_k` (`k ≈ 10` dominant directions). To move `D̂` appreciably requires `σ_a²/k ≪ σ_n²/d`, i.e. `σ_n² ≳ (d/k)σ_a² ≈ 50 σ_a²` — a ~50× increase in conditional variance, residual norms up ~7×, same-class cosine driven from ≈0.85 toward 0. `L_PA` then rises by `O(α) = O(32)`. The maximum obtainable reduction in `λ·L_CNW` is `λ·log D_max ≈ λ·4.5`. **The cheat is strictly unprofitable for `λ < 32/4.5 ≈ 7`.** At the specified `λ = 1` the margin is 7×. This is a *derived* admissible range, not a tuned constant.

**D2 — Residual collapse.** If `‖r‖ → 0`, `r̂` becomes the direction of floating-point/BN noise, which is isotropic and satisfies `L_CNW` vacuously. Blocked three ways: (a) the `τ_r = 0.05` gate excludes such residuals from `A` entirely, and `|K|/B` is a reported diagnostic — a run where it falls below 0.9 is void; (b) `ε = 10⁻²` in the norm; (c) at PA convergence on real data with `m=4` *distinct* images per identity, `‖r‖ ≈ 0.7` on the unit sphere, two orders above the gate. Actual collapse would also destroy `L_PA`'s own retrieval structure and be visible in train R@1.

**D3 — Rotation cheat (this one kills a whole family of alternatives).** `L_CNW` is invariant under `O(d)`: `D̂(RAR ᵀ) = D̂(A)`. **No change of basis, no linear reparameterization of `W_e`, and no rotation of the embedding can reduce it by even a floating-point epsilon.** Any decrease reflects a genuine change in what the encoder computes. Contrast: VICReg's covariance term, Barlow Twins' cross-correlation, and decorrelated-BN all penalize off-diagonal mass **in the canonical basis** and are therefore satisfiable by a rotation. CNW is not.

**D4 — Reweighting cheat.** Per-residual unit normalization gives every residual equal weight, so a high-variance class cannot dominate `A`, and the loss cannot be reduced by rescaling any class's spread.

**D5 — Fixed-low-rank solution (analyzed, and it is benign).** The encoder could confine all within-identity variability to one fixed `D_max`-dimensional subspace `V` and flatten it there, satisfying `L_CNW = 0` for every batch while remaining anisotropic in the full 512-D space. This is **not** a failure: it means zero conditional noise outside `V`, i.e. infinite SNR for any novel signal with a component outside `V` — a solution at least as good as full isotropy under the collision premise, and only minimax-suboptimal (its worst-case `max_u uᵀΣu = c/D_max > c/d`). I flag it as an anticipated non-degenerate optimum, and measure `rank(Σ_W)` and its full spectrum on unseen classes (which have enough images to estimate all 512 eigenvalues) to distinguish it from true isotropy.

**D6 — Batch-statistics cheat.** `L_CNW` is a function of per-image embeddings only; there is no batch-level trainable module. The one channel that could couple samples, backbone BN, is frozen in the primary-source PA recipe. Closed.

---

## 3. Adversarial primary-source novelty search

I searched inside DML (proxy/pair/regularizer literature), and outside it in SSL, speaker verification, image retrieval, face recognition, NLP embedding geometry, and computational neuroscience. Nearest works and the one-sentence mechanism distinction for each:

**The most dangerous neighbour — sign conflict, engaged head-on:**

1. **Non-isotropy Regularization (NIR), Roth, Vinyals & Akata, CVPR 2022** — argues proxy losses induce *locally isotropic sample distributions* around proxies and uses Normalizing Flows to enforce unique translatability, deliberately inducing **non**-isotropy; the prescriptions only appear opposed because the objects differ — NIR shapes the *density/higher-order structure* of samples around a proxy (an invertibility condition on the residual→sample map, and the flow, not the encoder, absorbs the second-order shape), while CNW constrains only the *second-moment spectrum* and is completely agnostic to higher-order structure. A distribution can have perfectly flat covariance and arbitrarily rich directional density; indeed a rank-10 conditional covariance is a *bottleneck on* the unique translatability NIR wants, so the honest reading is that CNW supplies the dimensional budget NIR then structures. **This is the single largest risk to the proposal and I treat it as a required experiment, not a rhetorical point** (control C8, §4).

**Same motivation, wrong object (marginal instead of conditional):**

2. **SIGReg / LeJEPA (Balestriero & LeCun, 2025)** and **Weak-SIGReg** — drive the *marginal* embedding distribution to an isotropic Gaussian via random 1-D sketches and an Epps–Pulley ECF test; CNW constrains the *identity-conditional* residual spectrum, which marginal isotropy neither implies nor is implied by (a mixture of anisotropic conditionals can have an isotropic marginal). I acknowledge the estimator-technique overlap: random-direction sketching is a valid alternative estimator for my quantity and I cite it as such.
3. **Barlow Twins / VICReg / W-MSE** — decorrelate or whiten the marginal batch covariance in a fixed basis to prevent collapse; CNW is `O(d)`-invariant (D3), uses identity labels, and never touches the marginal.
4. **ρ-regularization, Roth et al., ICML 2020** — attenuates the spectral decay of the *embedding space* (marginal) via a negative-swapping sampling intervention to fight over-clustering; CNW makes no claim about marginal rank and is an explicit second-moment loss on conditional residuals — flat marginal spectra are reachable by spreading class means alone, leaving `Σ_W` untouched.
5. **Alignment & Uniformity (Wang & Isola 2020) and hyperspherical uniformity regularizers** — first-order marginal spreading; CNW is second-order and conditional.
6. **Isotropy in NLP embeddings (Mu & Viswanath 2018; whitening-BERT)** — post-hoc removal of dominant directions from the *marginal* word-embedding covariance, explicitly to make cosine valid; identical motivation, opposite object (marginal, post-hoc, linear) — a useful demonstration that the "make the deployed cosine the right metric" premise has independent pedigree.

**Same object, wrong mechanism (post-hoc linear, fitted on seen classes):**

7. **WCCN / PLDA in speaker verification (Hatch, Kajarekar & Stolcke, Interspeech 2006; Prince & Elder 2007)** — the canonical open-set system whitens the within-class covariance as a **post-hoc linear transform fitted on training speakers**; CNW makes conditional isotropy a *property of the encoder function* enforced by gradients, so it holds at unseen inputs where no fitted matrix exists and where a global linear map cannot recover directions the encoder already collapsed. This is my closest functional ancestor and I name it as such.
8. **Learned whitening / linear discriminant projection for image retrieval (Radenović et al., TPAMI 2018)** — whitening estimated from matching pairs and applied post-hoc to the descriptor; same distinction as (7), plus CNW is scale-free and basis-free.
9. **LDA / NCA / LMNN / ITML** — learn a Mahalanobis matrix over fixed features; CNW learns **no metric at all** — it reshapes the encoder so the identity metric *is* the Mahalanobis metric.
10. **Whitened LDA and eigenfeature regularization for face recognition (Jiang et al., TPAMI 2008)** — regularizes the within-class scatter *eigenspectrum* by splitting it into reliable/unstable/null subspaces, but as a closed-form transform on fixed features to fight small-sample instability, not as a differentiable training objective shaping what the encoder computes.

**Same object, different quantity:**

11. **PFE / DUL / HIB probabilistic embeddings (Shi & Jain 2019; Chang et al. 2020; Oh et al. 2019)** — *estimate* per-sample anisotropic uncertainty and change the matching score to a likelihood; CNW estimates no uncertainty and leaves the score untouched — it removes the anisotropy so the unchanged cosine is already optimal.
12. **Intra-class correlation regularizer for speech embeddings (arXiv 2310.17049)** — maximizes a *scalar* between-to-total variance ratio; CNW is invariant to that scalar by construction (§1.3) and constrains only spectral shape.
13. **Deep Variational Metric Learning (ECCV 2018), Intra-class Adaptive Augmentation, DAS, Proxy Synthesis, Metrix** — *model or generate* intra-class variation to synthesize samples; CNW generates nothing and reshapes the geometry of residuals that already exist.
14. **SoftTriple / multi-proxy / PFML (Potential Field Based DML, Bhatnagar & Ahuja, CVPR 2025)** — model intra-class multimodality with multiple proxies or continuous potential fields, i.e. first-order structure (multiple modes, distance-decaying influence); CNW imposes a second-order isotropy condition and is designed to compose with such bases.
15. **PA+DADA (Data-Augmented Domain Adaptation, AAAI 2024)** — closes a proxy–sample domain gap using augmentation; orthogonal object, and my frontier configuration composes with it.
16. **Neural collapse / fixed-ETF classifiers** — prescribe first-order simplex geometry of class means with within-class variance → 0; CNW deliberately preserves within-class variance and prescribes its second-order *shape*, which NC theory leaves entirely unconstrained.
17. **Manifold capacity / MMCR (Yerxa et al., NeurIPS 2023)** — maximizes the nuclear norm of augmentation-manifold centroids in SSL; CNW targets a ratio that is invariant to that nuclear norm's scale, under identity supervision.
18. **Sorscher, Ganguli & Sompolinsky (PNAS 2022)** — derives few-shot error from manifold signal/bias/dimension/radius; this is *measurement and theory*, with no training objective — CNW is the training objective its SNR implies, and I use their geometry as instrumentation.
19. **Equivariant SSL (AugSelf, EquiMod, E-SSL)** — make the residual *predictive of the augmentation parameters*; CNW imposes no information about the nuisance factor and instead maximally spreads the ensemble of residual directions.
20. **Decorrelated BN / IterNorm** — architectural ZCA on activation marginals; CNW is a basis-free loss on conditional residual directions, not a normalization layer.

**Unresolved novelty risk, stated plainly.** I could not find a train-time loss that penalizes the *spectral flatness / participation ratio of the identity-conditional residual direction covariance* in DML, face recognition, or re-ID. I cannot prove absence. A reviewer should run at minimum: `"within-class covariance" isotropy loss metric learning`; `participation ratio intra-class covariance regularizer`; `conditional whitening encoder open-set retrieval`; `effective rank within-class scatter differentiable loss`; and a citation sweep of works citing both Hatch et al. 2006 and Kim et al. 2020. Two search results I could not resolve to a primary source and flag as possible collisions: a paper described as "minimizing the trace of the intra-class covariance matrix in forbidden directions" (surfaced near *Distance Metric Learning with Joint Representation Diversification*, ICML 2020) and *Deep Metric Learning with Density Adaptivity* (arXiv 1909.03909). Both appear to be trace/density-based rather than spectral-shape-based, but I did not read them and cannot certify the distinction.

---

## 4. Decisive matched-compute controls

All controls are identical in cost to CNW (a `|K|×|K|` Gram) unless noted, and all use the same base, seeds, schedule, and balanced sampler.

| | Control | What dies if it wins |
|---|---|---|
| **C1** | **Shuffled-label**: identical loss, residuals taken w.r.t. *random* class assignments | The gain is generic gradient noise / implicit regularization; the semantic mechanism is dead |
| **C2** | **Marginal isotropy**: same loss on `z_i − μ_batch` (total covariance) | The mechanism is marginal spectral flattening (ρ-reg / SIGReg territory), not conditional |
| **C3** | **Between-class isotropy**: same loss on proxy directions | The mechanism is proxy uniformity |
| **C4** | **Trace-only**: penalize `tr(Σ_W)` instead, no shape term | Magnitude, not shape, is what matters — kills the whole thesis |
| **C5** | **Sign flip**: `λ < 0` (anisotropize) | If R@1 does not degrade monotonically, the objective is not on a causal axis |
| **C6** | **Post-hoc WCCN**: fit within-class whitening on *training* classes, apply to the base descriptor at test (legal — no test data) | The "must be an encoder property" claim dies; a cheap linear fix suffices |
| **C7** | **CNW + post-hoc WCCN** | *Prediction: no additional gain*, because CNW has already made `Σ_W ≈ I`. If WCCN still helps after CNW, the loss did not do what it claims |
| **C8** | **CNW ⊕ NIR** (and: measure `D̂` of NIR-trained models) | If NIR-trained models already have flat conditional spectra, CNW is subsumed by CVPR 2022 |
| **C9** | **Balanced-sampler-only** base (no CNW) | Isolates the recipe change CNW forced on the baseline |
| **C10** | **Augmentation-residual variant**: two views of the same image instead of real intra-class pairs (**not** matched cost: ~2× training) | Tests whether real intra-class variation is the necessary nuisance object |

C6/C7 together are the decisive pair: they separate "pointwise encoder property that transfers to unseen identities" from "global linear fix on seen-class statistics," which is the entire novelty claim over 60 years of WCCN/LDA practice.

**Instrumentation (evaluated on unseen test classes, which have enough images per class to estimate the full 512-eigenvalue spectrum):**
- `I(f)` = unseen-class conditional participation ratio `D̂(Σ_W^unseen)`.
- Overlap `o = (uᵀΣ_W u)·d / (‖u‖² tr Σ_W)` averaged over unseen class pairs.
- Geometric SNR (Sorscher et al.) on unseen classes.
- `tr(Σ_W)` (must be ~unchanged — verifies the orthogonality claim of §1.3).
- `|K|/B`, and train-class R@1.

**Predicted signature of a genuine generalization mechanism:** unseen-class `I(f)` up sharply, `tr(Σ_W)` flat, unseen SNR up, and **train-class R@1 flat-to-slightly-down while test-class R@1 rises**. A method that improves both is probably just fitting better and the causal story is wrong.

---

## 5. Frozen forecasts, Lane A, and frontier arithmetic

R@1, ResNet-50 / 512-D / 224 px / single-view cosine / 200 epochs, **5 seeds**, mean ± sample std. Frozen before any run.

**Mechanism-isolation configuration (matched cost, ~1.00×):**

| | A0 = PA + balanced sampling | A1 = A0 + CNW | paired Δ |
|---|---|---|---|
| CUB | 0.715 ± 0.006 | **0.734 ± 0.006** | +1.9 ± 0.5 |
| Cars196 | 0.907 ± 0.005 | **0.925 ± 0.005** | +1.8 ± 0.5 |
| SOP | 0.806 ± 0.003 | **0.813 ± 0.003** | +0.7 ± 0.3 |

**Frontier configuration (PA+DADA base, reproduced by me, + CNW):**

| | B0 = my PA+DADA repro (published) | B1 = B0 + CNW |
|---|---|---|
| CUB | 0.725 ± 0.006 (pub. 0.729) | **0.742 ± 0.006** |
| Cars196 | 0.917 ± 0.005 (pub. 0.921) | **0.933 ± 0.005** |
| SOP | 0.808 ± 0.003 (pub. 0.810) | **0.814 ± 0.003** |
| In-Shop | 0.926 ± 0.004 (pub. 0.930) | **0.936 ± 0.004** |

**Frontier-crossing arithmetic** vs PFML (CUB 0.734 ± 0.003, Cars 0.927 ± 0.003, SOP 0.829 ± 0.002; five runs) and vs PA+DADA In-Shop 0.930 (seeds/uncertainty unreported):

| dataset | B1 − reference | σ_diff = √(σ_B1² + σ_ref²) | z | P(true crossing) | verdict |
|---|---|---|---|---|---|
| CUB | +0.8 pt | 0.67 pt | 1.19 | ≈ 0.88 | **crossing, weakly significant** |
| Cars196 | +0.6 pt | 0.58 pt | 1.03 | ≈ 0.85 | **crossing, weakly significant** |
| SOP | −1.5 pt | 0.36 pt | −4.2 | ≈ 0.00 | **no crossing — stated plainly** |
| In-Shop | +0.6 pt | undefined | — | unverifiable | reference uncertainty unreported |

I forecast that CNW **does not** reach the frontier on SOP, and I say so rather than dressing it up. This is mechanism-consistent, not an excuse: SOP identities are products photographed under limited within-identity variation (~5.3 images/class, mostly viewpoint), so conditional anisotropy is small and there is little for CNW to reshape. The prediction pattern **gain(CUB) ≈ gain(Cars) > gain(SOP)** is itself a falsifiable consequence of the mechanism (F8).

Two of these forecasts also rest on an additivity assumption (CNW ⊕ DADA), which is a separate claim I am freezing: **CNW retains ≥ 60% of its standalone increment when composed with DADA.** If composition is sub-additive below that, the frontier claim collapses to the A1 row, which merely ties PFML on CUB and sits 0.2 pt below on Cars — i.e. parity, not a crossing. I would rather state that conditional now than discover it later.

### Falsification thresholds (frozen)

- **F1** — paired 5-seed CNW increment on CUB `< +0.8` pt ⇒ falsified as a frontier contributor.
- **F2** — unseen-class `I(f)` does not rise `≥ 1.5×` vs base ⇒ the causal path is falsified *even if R@1 rises*; the gain must then be re-attributed.
- **F3** — base-model overlap `o ≤ 1.5` on unseen CUB pairs ⇒ the amplifier premise is false; expected gain collapses toward the Jensen floor (I estimate `< +0.5` pt), and the method should be abandoned for this benchmark family.
- **F4** — shuffled-label control (C1) reproduces `≥ 60%` of the gain ⇒ mechanism dead.
- **F5** — post-hoc WCCN (C6) reproduces `≥ 70%` of the gain ⇒ the encoder-property claim dead.
- **F6** — sign-flip control (C5) does not degrade R@1 monotonically ⇒ not a causal axis.
- **F7** — wall-clock/epoch `> 1.02×` base ⇒ the cost claim is falsified.
- **F8** — `gain(SOP) ≥ gain(CUB)` ⇒ mechanism suspect (reported, not fatal alone).
- **F9** — `|K|/B < 0.9` or `tr(Σ_W)` shifts by `> 20%` ⇒ run void (degeneracy D2 / orthogonality violation).
- **F10** — optimal `λ` at a grid boundary ⇒ inconclusive, re-grid before claiming anything.

---

## 6. Cost, and benchmark / contamination risks

**Training cost.** `~3×10⁻⁶` extra FLOPs, ~32 KB extra memory, no extra forward passes, no auxiliary network, no generated data, no second view. Forecast wall-clock **1.00–1.01×** and memory **1.00×**. For context in this lane: PA+DADA is ~1.06× epoch time and ~1.01× memory; AdvRF requires a training-only ResNet-34/U-Net reconstruction system plus distillation; VAPNet requires attribute machinery. CNW is the cheapest intervention of the set by a wide margin. The one real cost is the **class-balanced sampler**, which PA's official R50 configuration does not use — a recipe change, not a compute change, and controlled by C9.

**Deployment cost.** Bit-identical to the baseline: one ResNet-50, one view, one 512-D descriptor, cosine NN. Nothing is added at test time. No transduction, no reranking, no gallery fitting.

**Benchmark risks.**
- CUB's test split is 5,924 images; 1.0 R@1 point ≈ 59 images. My forecast crossing margins (0.6–0.8 pt) are inside single-seed noise. Nothing here is claimable without ≥5 seeds and a paired test, and I have deliberately reported `z ≈ 1.0–1.2` rather than calling these clean wins.
- The In-Shop reference (0.930) has unreported seed count and uncertainty, so no crossing there is verifiable regardless of what I measure. I decline to claim it.
- PFML's full training recipe is not available to me (see ambiguities). A rigorous crossing claim requires running PFML under my exact recipe; without that, the cross-paper comparison carries recipe risk in an unknown direction. My *defensible* claim is the paired increment over my own reproduced base; the frontier comparison is secondary and so labelled.
- `λ` selection on the test split is common malpractice in this literature and would invalidate everything; the pseudo-unseen training-class split (§1.5) is a hard requirement, not a nicety.

**Contamination risks.** ImageNet-1K initialization overlaps CUB (≈59 bird classes) and Cars196 semantics. This is accepted standard practice in the lane and is shared identically by PFML, DADA, AdvRF and VAPNet, so it confers no differential advantage on CNW — but it does mean no absolute "zero-shot" claim is warranted for any row in the table, mine included. No test data, no external generated data, no text/VLM encoders, no extra annotations, no transduction, no reranking are used at any point. The only statistics computed from data are within-batch, from training images and training identity labels.

**Failure modes I expect and cannot rule out.**
1. **Sign conflict with NIR.** If NIR's reported generalization benefits arise partly from *inducing* conditional anisotropy in a way that also helps at test, the two mechanisms genuinely conflict and one of us is wrong. C8 settles it; I would not defend CNW past a decisive C8 loss.
2. **Signal in `S`.** CNW moves conditional noise into the seen-discriminative subspace. If novel-class signal lies predominantly *inside* `S`, CNW hurts. This is the `o < 1` regime and is F3.
3. **Small-`m` datasets.** SOP and In-Shop residual estimates come from 2–4 images/class; the estimator is noisy and `D_max` is a loose ceiling. The forecast gains there are small partly for this reason.
4. **Rank ceiling.** `D_max = 90–135 ≪ 512` means each step only constrains a sampled subspace. The argument that resampling covers the sphere over 200 epochs is plausible, not proved; D5 is the concrete way it could stall in a benign-but-suboptimal place.

---

## Unresolved source ambiguities

Stated because the prompt requires them, and because several bear directly on the arithmetic above:

1. **PFML** (Potential Field Based Deep Metric Learning, Bhatnagar & Ahuja, CVPR 2025): I confirmed the identity of the method and the R@1 values (CUB 73.4 / Cars 92.7 / SOP 82.9, ResNet-50) but **not** its optimizer, schedule, batch size, pooling, augmentation, or epoch count. I therefore treat it as a reference number and not as a reproducible base.
2. **Proxy-Anchor R50/512**: the official repo specifies batch 120, lr 1e-4, 5 warm-up epochs, BN freezing, step LR decay every 5 (CUB) / 10 (Cars) — but the paper's headline results are Inception-BN, and the R50/512 numbers circulating in later papers use varying recipes. Whether AdamW (vs Adam) and cosine decay are the official R50 configuration is unresolved; I specified AdamW + cosine and require the baseline to run both schedules.
3. **PA+DADA (AAAI 2024)**: seed count and uncertainty unreported, per the prompt; I additionally could not confirm whether its PA base is R50-at-512 under the same sampler I require. My B0 reproduction forecast assumes it is.
4. Two possible novelty collisions I could not resolve to primary sources, named in §3.
5. I have **not** run any of these experiments. Every number in §5 is a frozen prior forecast, not a measurement, and I did not verify the reference values against the papers' own tables beyond the searches recorded here.

---

**Honest summary of what I am claiming.** The mechanism is: *the deployed metric is fixed and isotropic, the optimal metric is not, and no post-hoc fix reaches unseen identities — so make the encoder's conditional geometry isotropic during training.* It has a proof under a weak prior (Jensen at fixed trace), a minimax proof under no prior, an exactly-orthogonal gradient to the entire intra-class-compactness family, `O(d)`-invariance that immunizes it against the rotation cheats that afflict basis-dependent decorrelation losses, a derived rather than tuned admissible `λ` range, and essentially zero cost. Its two real vulnerabilities are the sign conflict with NIR (CVPR 2022) and the possibility that novel-class signal lives in the low-variance directions of `Σ_W` — both of which I have made cheap, decisive, and pre-registered to test rather than argued around.

**Sources:**
- [Non-isotropy Regularization for Proxy-based Deep Metric Learning (Roth, Vinyals, Akata, CVPR 2022)](https://openaccess.thecvf.com/content/CVPR2022/papers/Roth_Non-Isotropy_Regularization_for_Proxy-Based_Deep_Metric_Learning_CVPR_2022_paper.pdf)
- [Potential Field Based Deep Metric Learning (PFML, CVPR 2025)](https://openaccess.thecvf.com/content/CVPR2025/html/Bhatnagar_Potential_Field_Based_Deep_Metric_Learning_CVPR_2025_paper.html)
- [Proxy Anchor Loss for Deep Metric Learning (Kim et al., CVPR 2020)](https://openaccess.thecvf.com/content_CVPR_2020/papers/Kim_Proxy_Anchor_Loss_for_Deep_Metric_Learning_CVPR_2020_paper.pdf) · [official repo](https://github.com/sung-yeon-kim/Proxy-Anchor-CVPR2020)
- [Towards Improved Proxy-based DML via Data-Augmented Domain Adaptation (DADA, AAAI 2024)](https://arxiv.org/html/2401.00617v1)
- [Revisiting Training Strategies and Generalization Performance in DML (Roth et al., ICML 2020)](http://proceedings.mlr.press/v119/roth20a/roth20a.pdf)
- [Within-Class Covariance Normalization for SVM-based Speaker Recognition (Hatch et al., Interspeech 2006)](https://www.isca-archive.org/interspeech_2006/hatch06_interspeech.pdf)
- [Neural representational geometry underlies few-shot concept learning (Sorscher, Ganguli, Sompolinsky, PNAS 2022)](https://www.pnas.org/doi/abs/10.1073/pnas.2200800119)
- [SIGReg / LeJEPA](https://github.com/rbalestr-lab/lejepa) · [Weak-SIGReg (OpenReview)](https://openreview.net/forum?id=hDy4bmURLS)
- [Eigenfeature Regularization and Extraction in Face Recognition (Jiang et al., TPAMI 2008)](https://www3.ntu.edu.sg/home/EXDJiang/JiangX.D.-PAMI-08P.pdf)
- [Intra-class Correlation Regularizer for speech embeddings](https://arxiv.org/pdf/2310.17049) · [Deep Metric Learning with Density Adaptivity](https://arxiv.org/pdf/1909.03909) · [Joint Representation Diversification (ICML 2020)](http://proceedings.mlr.press/v119/chu20a/chu20a.pdf) (last two: flagged as unresolved possible collisions)
