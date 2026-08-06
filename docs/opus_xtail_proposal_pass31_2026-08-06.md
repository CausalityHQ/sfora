I did the novelty search first — the results changed one design decision (the nearest prior art is AnchorFace/UniTSFace, which solve the same depth problem by *enlarging the sample*, not by extrapolating). Here is the frozen proposal.

---

# XTail — Extrapolated-Tail Metric Learning

**Lane: A** (ResNet-50, 512-D normalized global descriptor, 224 px, single-view cosine, 200 epochs, batch 150). All forecasts and comparisons below are Lane A only.

---

## 0. One-sentence statement

Every deep-metric loss controls the negative-similarity distribution at the depth its batch can observe (~1/150); retrieval R@1 is decided at the depth of the deployment gallery (up to 1/60,502). XTail fits a Generalized Pareto tail to each anchor's in-batch negative exceedances, differentiates through the closed-form probability-weighted-moment estimator, and uses the **extrapolated 1/N_train quantile** as that anchor's margin target — so the loss penalizes the tail *shape* (index ξ), not only the tail *location*, at ~1.00× cost and with a bit-identical deployment model.

---

## 1. Executable mathematics

### 1.1 Learned objects (complete list)

| Object | Shape | Init | LR |
|---|---|---|---|
| Backbone θ (ResNet-50) | — | ImageNet-1K | η |
| Embedding `E`, `b` | 512×2048, 512 | Kaiming / 0 | η |
| Proxies `W = [w_1…w_C]` | 512×C | N(0, 1) rows | 100η |

`ξ̄` (below) is a non-learned EMA **buffer**. XTail introduces **zero new parameters**.

### 1.2 Forward / deployment

`h = GAP(ResNet50(x)) ∈ ℝ²⁰⁴⁸`, `z = Eh + b`, `f = z/‖z‖₂ ∈ S⁵¹¹`.
**Test:** resize shorter side 256 → center crop 224 → one forward → L2-normalize → cosine NN. One view, one descriptor, no TTA, no reranking, no gallery statistics.

### 1.3 Base loss — Proxy-Anchor, reproduced exactly (Kim, Kim, Cho, Kwak, CVPR 2020, Eq. 4)

$$\mathcal L_{PA}(X)=\frac{1}{|P^+|}\sum_{p\in P^+}\log\!\Big(1+\!\!\sum_{x\in X_p^+}\!\!e^{-\alpha(s(x,p)-\delta_{PA})}\Big)+\frac{1}{|P|}\sum_{p\in P}\log\!\Big(1+\!\!\sum_{x\in X_p^-}\!\!e^{\alpha(s(x,p)+\delta_{PA})}\Big)$$

`P` = all proxies; `P⁺` = proxies with ≥1 positive in batch; `X_p⁺` = batch embeddings of class p; `X_p⁻ = X∖X_p⁺`; `s` = cosine. **α = 32, δ_PA = 10⁻¹** (paper: "for all experiments"). Proxies normal-initialized; proxy LR ×100; AdamW; η = 10⁻⁴ (CUB/Cars), 6×10⁻⁴ (SOP/In-Shop); batch 150; random crop 224 + horizontal flip; center crop at test.

**Unresolved source ambiguities in this baseline** (§6 revisits): the CVPR paper does not state weight decay, BN freezing, LR schedule, warm-up, or a P×M sampler (it says "input batches are randomly sampled"); its 512-D main table is Inception-BN (CUB 68.4, Cars 86.1, SOP 79.1, In-Shop 91.5), not ResNet-50; and the Lane-A budget is 200 epochs, not the paper's 40/60. **I therefore inherit no published Proxy-Anchor number.** Arm A0 is my own matched reproduction.

### 1.4 The XTail term

Batch `B` = P classes × M images (P=30, M=5, |B|=150). Compute Gram `G = FFᵀ`. Per anchor `i`: `N_i = {j : y_j ≠ y_i}` (n_i ≈ 145), `P_i = {j≠i : y_j = y_i}`.

**(a) Threshold.** `u_i` = the `⌈(1−ζ)n_i⌉`-th ascending order statistic of `{s_ij}_{j∈N_i}`, **ζ = 0.25**. Exceedances `E_i = {j : s_ij > u_i}`, `k_i = |E_i| ≈ 36`.

**(b) Probability-weighted moments** (closed form, differentiable in the sorted exceedances `z_{i,(1)} ≤ … ≤ z_{i,(k_i)}`, `z_{ij} = s_ij − u_i`):

$$a_i=\frac1{k_i}\sum_{r} z_{i,(r)},\qquad b_i=\frac1{k_i}\sum_{r=1}^{k_i}\frac{k_i-r}{k_i-1}\,z_{i,(r)}$$

**(c) Shape (Hosking–Wallis GPD PWM,** convention `1−F(z) = (1+ξz/σ)^{−1/ξ}`**):**

$$\hat\xi^{\,\text{raw}}_i \;=\; 2-\frac{a_i}{a_i-2b_i+\epsilon},\qquad \epsilon=10^{-6}$$

**(d) Pooled shape + empirical-Bayes shrinkage.** Standardize `ẑ_ij = z_ij/a_i`, pool across all anchors (≈5,400 exceedances/batch), run the same PWM → `ξ̂_pool`. EMA buffer `ξ̄ ← 0.9·ξ̄ + 0.1·ξ̂_pool` (gradient flows only through the current 0.1 term). Then

$$\hat\xi_i=(1-\lambda_i)\,\bar\xi+\lambda_i\,\hat\xi^{\,\text{raw}}_i,\qquad \lambda_i=\frac{k_i}{k_i+\tau}\ \ (\tau=40),\quad \lambda_i:=0 \text{ if } k_i<k_{\min}=12$$

`ξ̃_i = clip(ξ̂_i, −0.5, +0.15)`, zero gradient outside the clip.

**(e) Extrapolation coefficient.** With `R = ζ·n_target` and using `σ_i = (1−ξ)·𝔼[Z] = (1−ξ̃_i)a_i`:

$$A_i \;=\; A(R,\tilde\xi_i)\;=\;\frac{(1-\tilde\xi_i)\big(R^{\tilde\xi_i}-1\big)}{\tilde\xi_i},\qquad A\big|_{\xi\to0}=\ln R$$

(Taylor branch for `|ξ|<10⁻³`: `A ≈ ln R + ξ(½ln²R − ln R)`.)
**`n_target = N_train`** — training-set size only, never a test quantity: CUB 5,864 · Cars 8,054 · In-Shop 25,882 · SOP 59,551. Hence `R` ∈ {1466, 2014, 6471, 14888} and `A_i ∈ [2.9, 18.3]`.

**(f) Deep-quantile target, positive, loss.**

$$q_i=\min\big(1,\;u_i+A_i a_i\big),\qquad s_i^+=\tfrac1\gamma\log\Big(\tfrac1{|P_i|}\sum_{j\in P_i}e^{\gamma s_{ij}}\Big),\ \gamma=8$$
$$m_i=q_i+\delta_X-s_i^+,\qquad \boxed{\ \mathcal L_{\text{XTail}}=\frac{1}{|B|}\sum_i \frac{1}{\beta}\,\mathrm{softplus}(\beta m_i)\ }$$

`δ_X = 0.10`, `β = 16`. **Total:** `L = L_PA + λ(t)·L_XTail`, `λ(t) = λ_max·min(1, t/T_w)`, `λ_max = 1.0`, `T_w = 10` epochs.

### 1.5 The mechanism in one line

Substituting `a_i = mean_{E_i}(s) − u_i`:

$$m_i \;=\; \underbrace{A_i\!\cdot\!\operatorname*{mean}_{j\in E_i} s_{ij}\;-\;(A_i-1)\,u_i}_{q_i}\;+\;\delta_X-s_i^+$$

Because `A_i ≥ 2.9` over the clipped range, a **uniform upward shift of the whole tail costs weight 1**, while the **spread of the tail costs weight A_i**. Standard losses (LSE/soft-max weighted) cost only location. That difference is the method.

### 1.6 Gradient paths (explicit)

With `g_i = σ(βm_i) ∈ (0,1)`, `c = λ/|B|`, sphere Jacobian `J_i = (I − f_i f_iᵀ)/‖z_i‖`:

- **positives:** `∂L/∂s_ij = −c·g_i·softmax_γ(s_ij)`, `j ∈ P_i`
- **exceedance mean:** `∂L/∂s_ij ⊃ c·g_i·A_i/k_i`, `j ∈ E_i`
- **threshold:** `∂L/∂u_i = c·g_i·(1−A_i) < 0`, routed to the single order-statistic element (subgradient)
- **tail-shape path (the novel one):** `∂L/∂s_ij ⊃ c·g_i·a_i·ρ·(∂A/∂ξ)·(∂ξ̂_i/∂s_ij)`, with

$$\frac{\partial A}{\partial \xi}=-g(\xi)+(1-\xi)g'(\xi),\quad g=\tfrac{R^\xi-1}{\xi},\quad g'=\tfrac{\xi R^\xi\ln R-(R^\xi-1)}{\xi^2}$$

`ρ = 1` (ablated at 0). At `ξ=0, R=14888`: `∂A/∂ξ ≈ 37` — large, hence the clip and shrinkage. Finally `∂s_ij/∂f_i = f_j`, and `∂f_i/∂z_i = J_i`, into `E` and θ.

### 1.7 Recipe (all arms identical)

ResNet-50 ImageNet-1K, BN **frozen**, 512-D, AdamW **wd = 10⁻⁴** on backbone+embedding, **wd = 0** on proxies, η as §1.3, 5-epoch linear warm-up then cosine decay, 200 epochs, batch 150 (30×5), grad-clip 1.0, RandomResizedCrop(224, scale 0.16–1) + hflip, AMP off for the PWM block (fp32). Every hyperparameter selected on a **class-disjoint split of the training classes** (last 20% of train class IDs held out as pseudo-unseen). Test set touched once per arm.

---

## 2. Causal zero-shot error mode + degeneracy attack

### 2.1 The error mode

R@1 fails for query `i` iff `max_{j∈G, y_j≠y_i} s_ij > max_{j∈G, y_j=y_i} s_ij`. The left term is an order statistic of anchor `i`'s negative-similarity law at depth `1/|G|`. Training observes depth `1/n_i ≈ 1/145`. Extrapolation factors under standard splits: **CUB 41× · Cars 56× · In-Shop 87× · SOP 417×**.

**(i) Depth bias, quantified.** GPD with `u=0.30, a=0.05, ξ=−0.2` (σ = 0.06):
`q(145) = 0.30 + (0.06/−0.2)(37.25^{−0.2}−1) = 0.4545`; `q(60,502) = 0.30 + (0.06/−0.2)(15126^{−0.2}−1) = 0.5562`.
Training equalizes positives against **0.455**; deployment demands **0.556**. A systematic **0.10-cosine under-margin** — comparable to the entire margin `δ_PA = 0.1`. Equivalently `A` moves 3.09 → 5.12.

**(ii) Shape blindness — the causal part.** Two anchors with *identical hardest in-batch negative* but different `ξ` have very different deployment quantiles (`A(−0.4)=3.6` vs `A(+0.1)=13.5` at SOP's R). LSE/max-weighted losses rank difficulty by tail **location**, so they assign both anchors near-identical, near-zero gradient. The heavy-`ξ` anchor is precisely the one that fails at gallery depth. Under the protocol's constraints this is uncorrectable at test time (no transduction, no reranking, no gallery fitting) and invisible in training accuracy.

**Load-bearing assumption:** `ξ` is location-, scale-, and rotation-free — a property of how the *embedding geometry* packs a generic class against generic others, not of any class identity — so it transfers from seen to unseen classes. This is the assumption most likely to be wrong; F5 falsifies it.

### 2.2 Proof-level attack on the cheapest degeneracies

**D1 — Collapse `f_i ≡ f`.** All `s = 1 ⟹ u_i = 1, a_i = 0, q_i = 1, s_i⁺ = 1, m_i = δ_X > 0`, so `L_XTail = softplus(βδ_X)/β > 0` and `∂L/∂s_i⁺ = −σ(βδ_X) ≠ 0`. Collapse is **not a critical point** and is strictly dominated by any configuration with `s_i⁺ − q_i > δ_X`. (`L_PA` independently forbids it.)

**D2 — Rotation / norm shortcuts.** Every term is a function of cosines of L2-normalized vectors ⟹ `L_XTail` is exactly O(512)-invariant and invariant to positive rescaling of `z`. There is no gauge or norm to exploit.

**D3 — Gaming the threshold.** `m_i` has coefficient `−(A_i−1) < 0` on `u_i`, so "raise `u_i`" looks exploitable. It is not: `u_i` is an order statistic of the same similarities and `mean_{E_i}(s) ≥ u_i` by construction, hence
$$m_i \;\ge\; u_i+\delta_X-s_i^+ \quad\text{pointwise, with equality iff the tail is flat.}$$
So **XTail provably upper-bounds a hard 75th-percentile quantile-margin loss**; minimizing it necessarily minimizes the 75th-percentile negative similarity. No minimizer exists below that floor.

**D4 — Gaming `a_i → 0`.** `a_i = 0` means the top quartile of negatives is exactly flat — no straggler negatives. The loss then reduces to the quantile-margin loss, which still drives `u_i` down and `s_i⁺` up. **The estimator's degenerate point coincides with the geometric optimum**, not with a shortcut.

**D5 — Gaming the PWM ratio.** `a_i − 2b_i → 0` blows up `ξ̂ᵣₐw`. Guarded three ways (ε-ridge; hard clip with zero gradient; shrinkage `λ_i`, disabled below `k_min`). Crucially the *sign* is protective: inflating `ξ̂` **increases** `A_i` and therefore **increases** the loss. There is no direction in which corrupting the estimator lowers the objective.

**D6 — One-good-positive chaining.** Pure max-positive admits a chain solution (each image needs only one same-class neighbour) which is R@1-legal on seen classes but does not transfer. Mitigated by `γ=8` (soft-max retains gradient to all positives; ablated at `γ∈{0,8,∞}`) and by `L_PA`'s proxy-pull, which enforces per-image compactness to a class center that a chain violates.

**D7 — Batch-composition gaming.** The negative pool spans only `P−1 = 29` classes. Countered because `R` is fixed by `n_target ≫ n_i`, so the objective is dominated by tail *shape*, which is class-generic; F5/F7 test the residual.

---

## 3. Adversarial novelty search (primary sources)

**Inside DML**

1. **Proxy-Anchor** (CVPR 2020) — LSE-weighted margins against proxies. *Distinction:* XTail's target is a **fitted deep quantile of the empirical negative law**, not a fixed margin against observed proxies.
2. **Multi-Similarity** (CVPR 2019) / Circle / Ranked-List Loss — reweight pairs by relative similarity. *Distinction:* they reweight what is **inside** the batch; XTail sets a target **beyond the batch's support**.
3. **Sampling Matters / distance-weighted sampling** (ICCV 2017) — analytic uniform-on-sphere model of the **bulk** pairwise-distance density, used to **sample** negatives. *Distinction:* XTail fits the **empirical upper tail** (POT/GPD), assumes no isotropy, and produces a **margin target**, not a sampling distribution.
4. **Recall@k Surrogate** (CVPR 2022) / **Smooth-AP** (ECCV 2020) — smooth the rank indicator, then push batch size very large (plus similarity mixup) to approach deployment depth. *Distinction:* XTail closes the same depth gap by **extrapolating a fitted tail** at |B| = 150 and ~1.0× cost, instead of enlarging the empirical sample.
5. **Cross-Batch Memory** (CVPR 2020) — stale-embedding queue enlarges the negative pool. *Distinction:* same "more samples" answer as (4); XTail needs no queue and composes with one (arm A8).
6. **PFML / SoftTriple / HIER** — richer *class* models (potential fields, multi-proxy, hierarchy). *Distinction:* those change **what a class is**; XTail changes **at what depth the negative law is controlled** — orthogonal, tested as A3.
7. **Range Loss and long-tail DML** — "tail" = class-frequency imbalance. *Distinction:* XTail's tail is the upper tail of the **similarity** distribution; unrelated to class frequency.

**Outside DML**

8. **AnchorFace (AAAI 2022), OneFace (ECCV 2022), UniTSFace (NeurIPS 2023)** — *the nearest work overall.* They optimize TAR@FAR by computing a FAR-anchored threshold from the batch **plus a MoCo-style online-updating feature set**, then softening the indicator; AnchorFace explicitly notes that insufficient negatives make threshold estimation non-robust. *Distinction:* they reach the operating depth **empirically, by enlarging the negative sample**; XTail reaches it by **fitting a GPD shape and extrapolating**, which (a) needs no memory bank, (b) yields a **per-anchor** target driven by that anchor's own tail index rather than one global threshold, and (c) targets retrieval gallery depth rather than a verification operating point.
9. **OpenMax (CVPR 2016), Extreme Value Machine (TPAMI 2018), GPD/GEV open-set classifiers (arXiv:1808.09902)** — EVT on distance/activation tails. *Distinction:* all are **post-hoc test-time** calibration/rejection over a frozen embedding; XTail **differentiates through** the EVT fit and uses it to shape the embedding at train time.
10. **ViEVL — Vocabulary-informed Extreme Value Learning (arXiv:1705.09887)** — EVT constraints in visual-semantic embedding. *Distinction:* requires vocabulary/semantic side information (**forbidden by this protocol**) and Weibull-fits distances to class centers for a max-margin constraint, rather than extrapolating a deployment-depth quantile of the image–image negative law.
11. **Extreme Quantile Regression Networks (EQRN, arXiv:2208.07590)** — POT + NN for extrapolating conditional extreme quantiles. *Distinction:* XTail runs the same machinery in reverse — the extrapolated quantile is not the prediction, it is the **adversary to be pushed down**, with gradient flowing into the representation that generates the sample.
12. **α-ReQ / RankMe** — spectral diagnostics of representation quality. *Distinction:* measurement-only, and about the covariance spectrum, not the similarity tail.

I found no primary source that (i) fits a GPD to in-batch negative-similarity exceedances, (ii) differentiates through the PWM estimator, and (iii) uses the extrapolated `1/N_train` quantile as a per-anchor margin target. That conjunction is the claimed novel object.

**Highest-risk novelty claim:** the AnchorFace PDF was not text-extractable in this session (binary stream), so distinction (8) rests on the paper's own method summary and on OneFace/UniTSFace descriptions. If AnchorFace does fit a parametric tail, (a) and (b) survive but (c) weakens materially. Flagged for the adjudicator.

---

## 4. Matched-compute controls

All arms share backbone, init, augmentation, optimizer, schedule, epochs, batch, and the class-disjoint validation split. 5 seeds each.

| Arm | What it is | What it kills if it matches A1 |
|---|---|---|
| **A0** | Proxy-Anchor, matched Lane-A reproduction | — (only legitimate Δ baseline) |
| **A1** | A0 + XTail | the method |
| **A2** | A0 with K proxies/class (15 CUB/Cars, 2 SOP/In-Shop, max aggregation) | "just use a richer class model"; PFML surrogate |
| **A3** | A2 + XTail | orthogonality; frontier attempt |
| **A4** | A1 with `A_i` → tuned **constant** A ∈ {2,4,6,8,10,14} | **decisive:** EVT fit is dead weight, method = tuned margin |
| **A5** | A1 with pooled `ξ̄` only (no per-anchor shrinkage) | per-anchor adaptivity is unnecessary |
| **A6** | A1 with `ρ = 0` (stop-grad through `ξ̃`) | "actively flatten the tail" is unsupported |
| **A7** | A1 with `n_target = n_i` (**no extrapolation**, same code path) | **decisive:** gain is quantile-margin, not extrapolation |
| **A8** | A0 + XBM queue K=4096, matched wall clock | "enlarge the sample" beats "extrapolate" |
| **A9** | A1 fitting the tail on anchor→proxy similarities | wrong negative population chosen |

**A4 and A7 must both fail to match for the mechanism claim to stand.** A7 is the cleanest single control in the design: identical estimator, identical code, only `R` changes.

---

## 5. Frozen forecasts, frontier arithmetic, falsification

**Lane A, R@1, mean ± SD over 5 seeds.** Primary datasets: **SOP and In-Shop** (deepest galleries). A0/A2 are my reproductions; no published number is inherited.

| Dataset | A0 (PA repro) | **A1 (+XTail)** | Δ | A2 (multi-proxy) | **A3 (A2+XTail)** | Lane-A reference | Verdict |
|---|---|---|---|---|---|---|---|
| CUB | 0.693 ± 0.005 | 0.702 ± 0.006 | **+0.009** | 0.719 ± 0.005 | 0.727 ± 0.006 | PFML 0.734 ± 0.003 | **miss −0.007** |
| Cars | 0.884 ± 0.005 | 0.895 ± 0.005 | **+0.011** | 0.916 ± 0.004 | 0.926 ± 0.005 | PFML 0.927 ± 0.003 | tie −0.001 |
| In-Shop | 0.913 ± 0.003 | 0.929 ± 0.003 | **+0.016** | 0.918 ± 0.003 | 0.934 ± 0.003 | PA+DADA 0.930 (SD unreported) | **cross +0.004**, unadjudicable |
| SOP | 0.801 ± 0.003 | 0.821 ± 0.004 | **+0.020** | 0.811 ± 0.003 | 0.830 ± 0.004 | PFML 0.829 ± 0.003 | **cross +0.001** = tie |

**Frontier arithmetic, explicit.**
- **SOP:** target 0.829. A0 = 0.801 ⟹ required lift +0.028. Multi-proxy supplies +0.010 (A2 = 0.811); XTail supplies +0.020 standalone, of which I assume ~+0.019 survives on top of A2 (orthogonal mechanisms, mild saturation) ⟹ A3 = 0.830. Crossing margin **+0.001 against a reference with SD 0.003** ⟹ **not a decisive crossing**; I put ~55% on A3 ≥ PFML.
- **In-Shop:** target 0.930. 0.913 + 0.016 + 0.005 = 0.934, margin +0.004. The weakness here is on the *reference* side: PA+DADA's seed count and uncertainty are unreported, so a +0.004 gap is **not adjudicable**.
- **CUB:** target 0.734, forecast 0.727 — **I predict a miss and do not claim it.** CUB's extrapolation factor is only 41× and its bottleneck is fine-grained feature quality, not tail depth.
- **Cars:** target 0.927, forecast 0.926 — tie.

**Honest summary of what I would defend:** the *matched* Δ and its *ordering*, not a frontier sweep. XTail is forecast to be a large, near-free, base-orthogonal gain that reaches the Lane-A frontier on SOP/Cars, crosses it unadjudicably on In-Shop, and falls short on CUB.

**Preregistered falsifiers (fixed before any run).**

- **F1 (scale).** Δ(A1−A0) on SOP < +0.008 at 5 seeds (Welch p > 0.05) ⟹ mechanism refuted at claimed magnitude.
- **F2 (ordering — the primary mechanism test).** The depth story *requires* Δ monotone in extrapolation factor: **Δ(SOP) > Δ(In-Shop) > Δ(CUB)** and **Δ(SOP) ≥ 1.5·Δ(CUB)**. (Arithmetic supporting the threshold: at ξ=−0.2 the extrapolated-vs-batch coefficient ratio is A(R_SOP)/A(R_batch) = 5.12/3.09 = 1.66 versus CUB's 4.61/3.09 = 1.49; the residual comes from R@1's `1−F(s⁺)ⁿ` sensitivity at n = 60k. Cars is unconstrained — too little headroom.) If gains are flat, or CUB ≥ SOP, **retract the mechanism claim even if the numbers improve.**
- **F3 (necessity of the fit).** |A4 − A1| ≤ 0.003 on all four ⟹ EVT machinery is dead weight; demote to "a tuned quantile-margin loss".
- **F4 (necessity of extrapolation).** |A7 − A1| ≤ 0.003 ⟹ retract.
- **F5 (ξ transfer).** At convergence measure `ξ̂` on seen-class and on held-out-class negatives. |ξ̂_seen − ξ̂_unseen| > 0.15 ⟹ the transfer assumption fails.
- **F6 (depth-scaling, no test data).** On the class-disjoint *training* split, evaluate R@1 at sub-gallery sizes {1k, 2k, 4k, 8k, full}. If the A1−A0 advantage is not monotone increasing in sub-gallery size, the mechanism is refuted at diagnostic level.
- **F7 (no free lunch).** If A1 gains R@1 but loses > 0.01 mAP@R, it is exploiting the R@1-specific max-positive structure (D6); report and re-tune γ.

---

## 6. Cost, benchmark and contamination risk

**Training cost.** One extra Gram (150²×512 ≈ 11.5 MFLOP) + 150 row sorts, against ResNet-50 fwd+bwd on 150×224² ≈ 1.85 TFLOP — under 10⁻⁵ of compute. Forecast wall clock **≤ 1.02×**, memory +150² fp32 = 90 KB. Compare: DADA ≈1.06× epoch time / 1.01× memory; AdvRF adds a ResNet-34/U-Net reconstruction system plus distillation; VAPNet adds attribute machinery. **Deployment cost: bit-identical to baseline** — zero extra parameters, zero extra ops, same 512-D descriptor.

**Risks, plainly.**
- **Estimator variance is the binding constraint.** 36 exceedances/anchor at |B|=150. Shrinkage + EMA mitigate; if variance dominates, the fallback is A8's queue, which reintroduces XBM machinery and weakens the "no extra machinery" claim.
- **ξ-path sensitivity.** `∂A/∂ξ ≈ 37` at ξ=0, R=14888. The clip bounds it; if training destabilizes, `ρ<1` is the remedy and A6 becomes primary — still novel, but demoted from "flatten the tail" to "read an adaptive margin off the fitted tail".
- **Hyperparameter inflation.** XTail adds 10 (ζ, τ, k_min, two clips, δ_X, β, γ, λ_max, T_w, ρ). All are selected on the class-disjoint *training* split. This is the single most common way a result of this shape gets faked in DML, and it is why I do not tune on test.
- **Contamination.** ImageNet-1K contains bird and car classes semantically overlapping CUB/Cars — this inflates all Lane-A absolutes equally but caps interpretability of CUB/Cars deltas. SOP and In-Shop contain near-duplicate images that inflate R@1 for every method.
- **Cross-paper risk.** PFML's proxy counts, seed protocol, epoch budget and weight decay were not verifiable from primary sources here; secondary sources place PFML ("Potential Field based Deep Metric Learning") at **CVPR 2024**, the brief at CVPR 2025 — unresolved. PA+DADA's uncertainty is unreported. Cross-paper comparisons in §5 are labelled as such; only A1−A0 and A3−A2 are internally valid.
- **Scale is operational, not cosmetic.** AdamW's decoupled decay means the `L_PA : L_XTail` ratio is not a free normalization. I fix wd = 10⁻⁴ (backbone/embedding), 0 (proxies) across every arm and report `λ_max` as a real hyperparameter, not an absorbed constant.

---

**Sources:**
[Proxy Anchor Loss (CVPR 2020)](https://ar5iv.labs.arxiv.org/html/2003.13911) · [AnchorFace (AAAI 2022)](https://ojs.aaai.org/index.php/AAAI/article/view/20063) · [UniTSFace (NeurIPS 2023)](https://arxiv.org/pdf/2311.02523) · [OneFace (ECCV 2022)](https://link.springer.com/chapter/10.1007/978-3-031-19775-8_32) · [Sampling Matters in Deep Embedding Learning (ICCV 2017)](https://arxiv.org/abs/1706.07567) · [Recall@k Surrogate Loss (CVPR 2022)](https://arxiv.org/abs/2108.11179) · [Multi-Similarity Loss (CVPR 2019)](https://arxiv.org/pdf/1904.06627) · [The Extreme Value Machine](https://arxiv.org/pdf/1506.06112) · [EVT for Open Set Classification — GPD and GEV](https://arxiv.org/pdf/1808.09902) · [Vocabulary-informed Extreme Value Learning](https://arxiv.org/pdf/1705.09887) · [Extreme Quantile Regression Networks](https://arxiv.org/html/2208.07590v3) · [PFML quick review](https://liner.com/review/potential-field-based-deep-metric-learning) · [Unbiased Evaluation of DML](https://arxiv.org/pdf/1911.12528)
