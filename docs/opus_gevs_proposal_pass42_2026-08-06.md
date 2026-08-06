I did not inspect the filesystem, git state, or any prior conversation. Grounding is web/primary-literature only. Returning **one** method.

---

# GEVS — Gallery-Scale Extreme-Value Supervision

**Lane: A** (ResNet-50, 512-D normalized descriptor, ~224 px, single-view cosine retrieval, 200 epochs). All forecasts, baselines and references below are Lane A only.

**One-line mechanism.** Every DML loss aggregates negatives over a mini-batch of *M* negatives; deployment takes a maximum over a gallery of *N* ≈ 10²–10³·*M* negatives drawn from **classes the loss never pushed away**. GEVS makes the training loss target the *extrapolated* gallery-scale negative extreme by fitting a peaks-over-threshold generalized-Pareto tail to the in-batch negative similarities, correcting for same-class clustering with the extremal index, and extrapolating to a **training-set-derived** exposure *N*. The standard log-sum-exp loss is recovered exactly as the (ξ=0, θ=1, N=M) special case.

---

## 1. Executable mathematics

### 1.1 Model and deployment

Backbone `f_θ`: ResNet-50 (torchvision `IMAGENET1K_V1`), stage-4 GAP → **g** ∈ ℝ²⁰⁴⁸. Head `W` ∈ ℝ⁵¹²ˣ²⁰⁴⁸ (+bias), then

$$z = \frac{Wg+b}{\lVert Wg+b\rVert_2}\in S^{511}.$$

Train aug: `RandomResizedCrop(224, scale=(0.16,1.0))` + `RandomHorizontalFlip`. Test: `Resize(256)` → `CenterCrop(224)`. Deployment: one model, one view, 512-D, cosine NN. **Zero deployment overhead vs the carrier.**

### 1.2 Carrier loss (reproduced exactly, then held fixed across all arms)

Proxy-Anchor (Kim et al., CVPR 2020), one proxy per class, α = 32, δ = 0.1:

$$\mathcal L_{\text{PA}}=\frac{1}{|P^+|}\sum_{p\in P^+}\log\Big(1+\!\!\sum_{x\in X_p^+}\!\!e^{-\alpha(s(x,p)-\delta)}\Big)+\frac{1}{|P|}\sum_{p\in P}\log\Big(1+\!\!\sum_{x\in X_p^-}\!\!e^{\alpha(s(x,p)+\delta)}\Big)$$

Primary-source recipe (official repo): AdamW, lr 1e-4, proxy lr ×100, weight decay 1e-4, backbone frozen for 5 warm-up epochs, batch 120 (CUB/Cars). Published R50/512-D: **CUB 69.9, Cars 87.7**; SOP/In-Shop are published for Inception-BN only (79.2 / 91.9) — see §7 ambiguities.

### 1.3 The GEVS term

Batch = *P* classes × *K* images, *B* = *PK*. For anchor *i*: positives `P_i` (K−1), negatives `N_i` (M = B−K), similarities `s_ij = z_i·z_j`.

**Positive side (R@1 needs the *best* positive):**
$$s_i^+=\tfrac1\gamma\log\!\!\sum_{p\in P_i}\!e^{\gamma s_{ip}},\qquad \gamma=32.$$

**Step 1 — threshold.** Sort negatives descending `s_(1) ≥ … ≥ s_(M)`. Set `k = ⌈ρM⌉`, ρ = 0.2, threshold `u_i = stopgrad(s_(k+1))`, exceedance rate `ζ = k/M`. Excesses `y_j = s_(j) − u_i`, j ≤ k.

**Step 2 — shape (detached).** Probability-weighted moments (Hosking–Wallis). Ascending excesses `y_[1] ≤ … ≤ y_[k]`:

$$a_0=\tfrac1k\textstyle\sum_j y_{[j]},\qquad a_1=\tfrac1k\textstyle\sum_j y_{[j]}\tfrac{k-j}{k-1},\qquad \hat\xi_i=2-\frac{a_0}{a_0-2a_1}.$$

(Derivation check: for GPD, `a_r = σ/[(r+1)(r+1−ξ)]`, so `a_0/(a_0−2a_1) = 2−ξ` exactly.)

Shrink and project onto the admissible set. Cosine similarity is bounded by 1 ⇒ the tail is **Weibull-domain, ξ < 0**, with finite right endpoint:

$$\tilde\xi_i=\text{clamp}\big((1-\kappa)\hat\xi_i+\kappa\bar\xi,\;-2,\;-0.02\big),\quad \kappa=0.5,$$

`ξ̄` = EMA (momentum 0.99) of `ξ̂` across anchors/steps, detached.

**Step 3 — extremal index (dependence, detached).** Same-class gallery images are a *cluster*: the max over N dependent scores behaves like the max over θN independent ones. Runs estimator on the exceedance set:

$$\hat c_i=\frac{k}{\#\{\text{distinct classes among the top-}k\}},\qquad \hat c_i^{\text{test}}=\min\!\big(\hat c_i\cdot \tfrac{\bar n}{K},\;\bar n\big),\qquad \hat\theta_i=1/\hat c_i^{\text{test}},$$

where `n̄` = mean images-per-class of the **training** split (CUB 58.6, Cars 82.2, SOP 5.26, In-Shop 6.48).

**Step 4 — exposure.** `N = |D_train|` (5 864 / 8 054 / 59 551 / 25 882). **The test gallery size is never used** (§6).

**Step 5 — extrapolated gallery-scale negative extreme.** With scale `σ_i = (1−ξ̃_i)·ȳ_i` (method-of-moments given ξ) and `ȳ_i = (1/k)Σ_{j≤k} y_j`:

$$\boxed{\;\hat s_i(N)=u_i+c_i\Big(\tfrac1k\textstyle\sum_{j\le k}s_{(j)}-u_i\Big),\qquad c_i=\text{clamp}\Big(\tfrac{1-\tilde\xi_i}{-\tilde\xi_i}\big(1-(N\hat\theta_i\zeta)^{\tilde\xi_i}\big),\,0,\,8\Big)\;}$$

with the endpoint guard `σ_i ← min(σ_i, (−ξ̃_i)(1−u_i))`, which enforces `ŝ_i(N) ≤ 1` for every *N*.

**Step 6 — loss.**
$$\mathcal L^{\text{GEVS}}_i=\tfrac1\eta\log\!\big(1+e^{\eta(\hat s_i(N)+m-s_i^+)}\big),\quad \eta=16,\; m=0.1;\qquad \mathcal L=\mathcal L_{\text{PA}}+\lambda\cdot\tfrac1B\textstyle\sum_i \mathcal L^{\text{GEVS}}_i .$$

**Gradient path.** `u_i`, `ξ̃_i`, `θ̂_i`, `ζ`, `N` all detached. The only differentiable objects are the top-*k* negative similarities and the positive LSE:

$$\frac{\partial \hat s_i}{\partial s_{(j)}}=\frac{c_i}{k}>0\;(j\le k),\qquad \frac{\partial \mathcal L_i}{\partial s^+_i}<0 .$$

So GEVS is, in gradient terms, a **top-*k* negative penalty whose per-anchor strength `c_i` is an analytic function of the tail shape, the cluster size, and the batch-to-gallery exposure ratio.** No new parameters, no memory.

### 1.4 Exact reduction of the baseline

The standard soft-max negative aggregation `(1/α)log Σ_{n=1}^M e^{α s_n}` equals, under POT, the (1−1/M)-quantile `u + σ log(Mζ)` with **σ = 1/α**, i.e. the ξ→0 (Gumbel/exponential-tail) limit at θ=1, N=M. GEVS therefore relaxes exactly three implicit assumptions of every LSE-based DML loss — *unbounded tails*, *independent negatives*, *N = M* — and the three ablations C3/C4/C6 below map one-to-one onto them.

### 1.5 Schedules and full hyperparameters

| | CUB | Cars | SOP | In-Shop |
|---|---|---|---|---|
| P × K (batch) | 30×4 | 30×4 | 45×4 | 45×4 |
| k = ⌈0.2M⌉ | 24 | 24 | 36 | 36 |
| N | 5 864 | 8 054 | 59 551 | 25 882 |
| n̄ | 58.6 | 82.2 | 5.26 | 6.48 |

Fixed everywhere: ρ=0.2, m=0.1, η=16, γ=32, κ=0.5, ξ̄ momentum 0.99, ξ∈[−2,−0.02], c∈[0,8]. λ ramps linearly 0→λ_max over epochs 5→25, λ_max = 1.0. AdamW lr 1e-4 (proxies ×100), wd 1e-4, 5 warm-up epochs (backbone frozen), **cosine LR decay to 0 over 200 epochs** — a declared deviation from PA's step schedule, applied identically to every arm including the λ=0 baseline. 5 seeds, mean ± std.

**Scale is operational, not cosmetic.** `z` is L2-normalized, so the loss is scale-invariant in `W` and AdamW's *decoupled* weight decay alone sets the equilibrium ‖W‖ and hence the effective angular learning rate. Adam's second-moment normalization means changing λ does **not** simply rescale the GEVS influence — it reweights the PA/GEVS gradient mixture *and* shifts the weight-decay equilibrium. λ must therefore be tuned **jointly with weight decay**, and the λ=0 baseline must be re-tuned over the same wd grid. I pre-register the 2-D grid λ ∈ {0.25, 0.5, 1, 2} × wd ∈ {1e-4, 4e-4} for both the method and the baseline.

---

## 2. Causal zero-shot error mode + degeneracy attack

### 2.1 The error mode: *truncated-tail overconfidence*

Two compounding biases, both specific to zero-shot retrieval:

1. **Exposure bias.** Training sees `M ≈ 120–176` negatives from `P−1 ≈ 29–44` classes; deployment takes a max over 5.9 k–60.5 k images from 97–11 315 classes. Log-ratios of *negative-class* counts: CUB 0.81, Cars 0.79, In-Shop 4.50, SOP 5.55 nats.
2. **Familiarity bias.** Every negative the loss ever sees belongs to a class it is simultaneously repelling, so the seen-class negative-similarity distribution is *truncated from above* over 200 epochs. Test negatives are unseen classes that were never repelled; their upper tail is not truncated.

Both move the operative statistic up. The trained margin `E[s⁺ − max_{n∈batch} s⁻]` is calibrated to `q_{1−1/M}(F_seen)`; deployment faces `q_{1−1/N}(F_unseen)`. The gap is not a constant — it grows with the exposure ratio and is therefore **largest exactly on the largest benchmarks**.

Independent corroboration from primary sources: the Multi-Similarity paper's batch ablation reports SOP R@1 **71.40 at batch 20 → 78.35 at batch 1000**, while **CUB degrades** with larger batches; XBM lifts contrastive on SOP by ~13 points purely by materializing more negatives; Recall@k Surrogate needs ~4000-sample batches. Every one of these is a brute-force payment for the exposure gap. GEVS pays it analytically.

### 2.2 Proof-level attack on the cheapest degeneracies

**D1 — N-driven margin runaway.** *Blocked analytically.* Because `s ∈ [−1,1]`, the tail is Weibull-domain (ξ<0) with endpoint `x_F = u + σ/(−ξ)`, and the projection `σ ≤ (−ξ)(1−u)` forces `x_F ≤ 1`. Hence `ŝ_i(N) ≤ 1` for all N, and

$$\frac{\partial \hat s_i}{\partial \log N}=\sigma_i (N\theta\zeta)^{\tilde\xi_i}\xrightarrow[N\to\infty]{}0 .$$

The objective is *asymptotically insensitive* to N. Contrast the Gumbel/LSE surrogate: `∂ŝ/∂log N = σ = 1/α`, constant — **unbounded in N**. This is the precise sense in which the naive "just inflate the margin by (1/α)log(N/M)" correction is wrong, and it is directly testable (C6).

**D2 — estimator gaming.** The network could try to lower `ŝ` by manipulating the *statistic* instead of the geometry. Blocked by construction: `ξ̃, θ̂, ζ, N` are stop-grad, so the only differentiable path is `∂ŝ/∂s_(j) = c_i/k > 0`. The loss is **monotone non-increasing in every top-k negative similarity**: it can only be reduced by actually lowering real negative similarities or raising the positive.

**D3 — perverse gradient on the threshold (a real bug, found and fixed).** Writing `ŝ = (1−c)u + (c/k)Σ_{j≤k}s_(j)`: whenever `c > 1` the coefficient on `u` is **negative**, so a differentiable threshold would reward *increasing* the (k+1)-th negative similarity. Typical values are `c ≈ 1.5–3` (e.g. ξ=−0.5, Nθζ=3.8e3 ⇒ c=2.95), so this is not hypothetical. Detaching `u` removes it and leaves all coefficients ≥ 0. Clamping `c ≤ 1` would also fix it but would gut the extrapolation; detaching is the correct fix and is also standard POT practice (u is a pre-chosen level, not a fitted parameter).

**D4 — total collapse.** At `z_i ≡ z`: all `s = 1`, `ȳ = 0`, `ŝ = 1 = s⁺`, `L = softplus(m)/η > 0` — not a minimum. First order, `∂L/∂z_i = w⁻Σ_{j∈top-k} z_j − w⁺Σ_{p∈P_i} z_p` with `w⁻, w⁺ > 0`; the configuration is a strict saddle, since separating any class decreases the negative term at first order while the positive term responds only at second order. (First-order argument, not a full Hessian proof — stated as such.)

**D5 — degenerate PWM.** `a_0 − 2a_1 → 0` (all excesses equal) sends `ξ̂ → 2`. Handled by the clamp to [−2,−0.02] plus EMA shrinkage, and it cannot be exploited because ξ̂ is detached (D2).

---

## 3. Adversarial novelty search — nearest works and mechanism distinctions

**Inside contrastive/metric learning**

1. **WEINCE, "When Softmax Fails at the Top: Extreme-Value Corrections for InfoNCE" (arXiv 2606.00262, 2026)** — the nearest work; it also fits Weibull/POT tails to bounded cosine scores inside a contrastive loss, but it reweights **in-batch** logits with stop-grad statistics, has **no exposure parameter N**, no extremal-index/cluster correction, and is evaluated on SSL pretraining (CIFAR/STL/ImageNet-32, SimCSE); GEVS's entire content is the batch→gallery extrapolation WEINCE does not perform.
2. **XBM (CVPR 2020)** — enlarges the negative pool by *materializing* stale past embeddings under a slow-drift assumption; GEVS materializes nothing, has no memory, no staleness, and an explicit exposure knob.
3. **Recall@k Surrogate (CVPR 2022)** — attacks the same gap by *actually* training at batch ≈4000 via a two-pass memory trick; GEVS obtains the large-N statistic at batch 180 by extrapolation.
4. **Multi-Similarity (CVPR 2019) / Ranked List Loss / Smooth-AP / FastAP** — all reweight or rank *within* the batch; every one inherits the batch's negative cardinality, which is exactly the quantity GEVS corrects.
5. **PFML (CVPR 2025)** — changes the *interaction potential* between embeddings and proxies (ψ ∝ ±r^−α, δ-capped, M=15 proxies on CUB/Cars, M=2 on SOP); GEVS leaves the interaction form untouched and changes the *sample size at which the interaction is evaluated* — the two compose.
6. **PA+DADA (AAAI 2024)** — closes a sample↔proxy domain gap by data-augmented domain adaptation; orthogonal to exposure and independent of gallery cardinality.
7. **Threshold-Consistent Margin loss (arXiv 2307.04047)** — equalizes operating thresholds *across classes/domains*; GEVS equalizes nothing across classes, it rescales the negative extreme to a cardinality.
8. **Sampled softmax / logQ correction (two-tower retrieval; ACM RecSys 2025 "Correcting the LogQ Correction")** — corrects in-batch *sampling-frequency* bias with a per-item additive logit shift; GEVS corrects the *extreme-order-statistic scale*, which for bounded scores is a saturating power law in N, not an additive log-frequency term.

**Outside DML**

9. **EVT for recognition scores — Scheirer et al. (ECCV 2010 robust fusion; EVM; OpenMax), Furon & Jégou (EVT for image search), "Surprise: Result List Truncation via EVT" (SIGIR 2020)** — all fit EVT tails at **test** time to calibrate, threshold or truncate a *frozen* model's scores; GEVS puts the fit inside the training loss so gradients reshape the embedding.
10. **Hydrology/insurance return-level design (Hosking & Wallis PWM 1987; Coles 2001) and the extremal index for clustered extremes (Leadbetter; Ferro & Segers 2003)** — GEVS imports the *design-value* calculation (extrapolate an observed sample's tail to a far larger exposure) and the *extremal index* (dependence within clusters) into a differentiable loss, where "exposure" is the retrieval gallery and "clusters" are same-class gallery images. To my knowledge no DML method makes the deployment gallery cardinality an explicit term in its training loss.

---

## 4. Decisive matched-compute controls

All arms share seeds, sampler, P, K, B, epochs, augmentation, LR/wd grid. Method cost ≈ +0.4 % step time.

| ID | Arm | What it kills if it matches GEVS |
|---|---|---|
| **C1** | `c_i ≡ c` constant, grid-searched over {0.5,1,2,3,4} | the *per-anchor adaptivity* — GEVS reduces to tuned top-k mining |
| **C2** | `c_i ≡ 0` (pure top-(k+1) hinge), margin *m* re-tuned | the *extrapolation* — it's just hard-negative mining + a margin |
| **C3** | `θ̂ ≡ 1` (independence assumed) | the *cluster/extremal-index* correction |
| **C4** | Gumbel limit `c = log(Nθζ)`, unbounded in N | the *bounded-support (Weibull)* premise (WEINCE's premise, tested here) |
| **C5** | XBM memory (1 024 / 8 192) and larger batch (B=360, gradient-accumulated) at matched wall-clock/memory | the *no-memory* selling point (demotes rather than refutes) |
| **C6** | N swept over {0.25, 1, 4}×\|D_train\| | the *EVT parametrization* — GEVS must be **flat** (saturation, D1); a margin schedule cannot be |
| **C7** | Gradients allowed through ξ̂ | tests the D2 hardening (predicted to *degrade*) |
| **C8** | GEVS on top of a PFML carrier reproduction | composability; separates carrier from mechanism |

C6 is the single most decisive test: no constant-margin or LSE-temperature reparametrization can be simultaneously non-trivial and flat in N over a 16× sweep.

---

## 5. Frozen forecasts, falsification thresholds, frontier arithmetic

**Lane A. R@1, 5 seeds, mean ± 1σ. Frozen 2026-08-06.**

Matched baselines I will run (λ=0, identical everything). CUB/Cars baselines are anchored to the official PA repo; **SOP/In-Shop R50/512-D PA numbers are not published, so those baselines are my forecasts, not literature values.**

| | PA baseline (mine) | **GEVS forecast** | Δ | References (Lane A) |
|---|---|---|---|---|
| **SOP** | 80.2 ± 0.2 | **82.0 ± 0.3** | **+1.8** | PFML 82.9±0.2; PA+DADA 81.0 |
| **In-Shop** | 91.3 ± 0.2 | **92.6 ± 0.3** | **+1.3** | PA+DADA 93.0 (seeds/σ unreported) |
| **Cars196** | 87.7 ± 0.4 | 88.1 ± 0.4 | +0.4 | PFML 92.7±0.3; PA+DADA 92.1 |
| **CUB** | 69.9 ± 0.4 | 70.0 ± 0.5 | +0.1 (≈ null) | PFML 73.4±0.3; PA+DADA 72.9 |

**The near-null CUB/Cars forecast is a prediction, not a hedge.** The mechanism scales with the exposure ratio (0.8 nats on CUB/Cars vs 4.5–5.6 on In-Shop/SOP), and the MS-loss batch ablation independently shows CUB *degrading* with more negatives. A method that gained uniformly across all four datasets would falsify the stated mechanism.

**Frontier arithmetic.**
- **SOP:** 82.0 vs PA+DADA 81.0 → **crosses by +1.0**. vs PFML 82.9±0.2 → **−0.9, does not cross.** With a PFML carrier (C8), forecast = my PFML reproduction P̂ + 1.4; if P̂ = 82.9 this gives 84.3. I do **not** inherit 82.9: PFML's batch size, weight decay and LR schedule are undisclosed (§7), so the claim is Δ over *my* reproduction, and if P̂ < 82.9 the frontier claim fails with it. (AdvRF's 84.2 is Lane B, 2048-D — noted for orientation only, not claimed.)
- **In-Shop:** 92.6 vs PA+DADA 93.0 → **−0.4, does not cross**; and since DADA's seed count and uncertainty are unreported, that comparison is formally unresolvable. With a DADA-strength carrier, 93.0 + ~1.0 = 94.0 — conditional on a DADA reproduction I have not verified.
- **CUB / Cars:** explicitly forecast **not** to cross PFML, by ~3.4 and ~4.6 points.

**Falsification thresholds (pre-registered).**
- **F1** Δ_SOP < **+0.8** (5-seed mean) ⇒ mechanism refuted.
- **F2** Δ_CUB ≥ Δ_SOP ⇒ the exposure-scaling story is refuted regardless of absolute gains.
- **F3** C1 (best constant *c*) within **0.3** of GEVS on SOP ⇒ per-anchor adaptivity refuted; method demoted to "tuned top-k mining with an N-derived constant".
- **F4** C6 sweep varies by > **0.5** R@1 across N ∈ {0.25,1,4}×\|D_train\| ⇒ EVT parametrization refuted (saturation prediction fails).
- **F5** C2 within **0.3** of GEVS on SOP ⇒ the whole extrapolation is inert.
- **F6** C5 (XBM/large batch at matched cost) ≥ GEVS on SOP ⇒ demoted, not refuted: materializing beats extrapolating.

Confidence, stated plainly: I hold **F2 and F4 with high confidence** (they follow from the mathematics plus the published batch ablations). I hold the **absolute +1.8 on SOP with low-to-moderate confidence** — the historical range for exposure-driven mechanisms on SOP is roughly +1 to +7, but those were measured from weak operating points, and the saturation the method itself predicts (D1) argues the remaining headroom at a strong 80.2 baseline is small. **F3 is the threat I rate most likely to fire.**

---

## 6. Cost, benchmark and contamination risk

**Training cost.** One B×B similarity matrix (180² × 512 ≈ 17 MFLOP vs ≈ 2.2 TFLOP for the ResNet-50 fwd+bwd on 180 images — under 0.001 %), one row-wise sort (O(B log B)), O(Bk) reductions. Measured overhead expectation **< 0.5 % epoch time, +130 KB activation, 0 parameters, 0 memory bank.** Strictly cheaper than DADA (1.06× time / 1.01× memory), and far cheaper than AdvRF (extra ResNet-34 + U-Net + distillation) or VAPNet (attribute machinery). **Deployment cost is exactly the carrier's.**

**Contamination.** *N* is the **training** split size and `n̄` the **training** images-per-class; no test cardinality, no test image, no gallery statistic enters training. The method is deliberately insensitive to *N* (D1), so the choice cannot smuggle in test information even implicitly — and C6 verifies that insensitivity empirically.

**Benchmark risk.** These four benchmarks have no conventional validation split, so the field routinely tunes on test (Musgrave et al., "A Metric Learning Reality Check"). GEVS adds hyperparameters (ρ, λ, m, η, κ, clamps), which makes this risk materially worse if ignored. Mitigation, pre-registered: **carve a class-disjoint validation split from the training classes (last 20 % of training identities), select every hyperparameter there, freeze, then retrain on all training classes.** The λ=0 baseline gets the identical treatment and the identical (λ, wd) grid.

**Residual risks I do not claim to have resolved.**
- ImageNet-1K pretraining is semantically entangled with Cars/CUB. Pre-existing in the lane, shared with every reference, not introduced here.
- *k* ≈ 24 exceedances is a thin sample for a 2-parameter GPD; EMA shrinkage helps but the per-anchor ξ̂ is noisy. This is exactly the weakness F3 would expose.
- The extremal-index transfer from batch (K=4) to gallery (n̄) is a modelling approximation, not a theorem; C3 bounds its contribution.
- Fitting the tail on *seen*-class negatives corrects the exposure bias but **not** the familiarity bias of §2.1. A probe-class variant (a rotating 25 % of classes masked from the discriminative term each step, used only for the tail fit) is the obvious extension; I deliberately left it out of the method to keep one mechanism, and pre-register it as an exploratory arm only.

---

## 7. Unresolved source ambiguities

- **PFML (CVPR 2025 / arXiv 2405.18560):** discloses M=15 (CUB, Cars) / M=2 (SOP), 200 epochs, Adam lr 5e-4 with proxy lr ×100, 224 center crops from 256, δ ∈ [0.1,0.3], α ∈ [0,6], R@1 73.4±0.3 / 92.7±0.3 / 82.9±0.2 over 5 runs. **Batch size, classes-per-batch, weight decay, LR schedule, and per-dataset final (δ, α) are not disclosed** in what I could retrieve. No In-Shop row was retrievable. A matched reproduction is therefore not possible from the source alone.
- **PA+DADA In-Shop 0.930:** seed count and uncertainty unreported (as the task statement notes); I could not retrieve the AAAI PDF's exact recipe.
- **Proxy-Anchor:** the official repo publishes R50/512-D for CUB (69.9) and Cars (87.7) only; SOP (79.2) and In-Shop (91.9) are Inception-BN. My SOP/In-Shop baselines are forecasts, flagged as such.
- **Recall@k Surrogate:** I could not retrieve its R50/512-D table (CVPR/arXiv PDFs blocked or unparseable). I therefore use **no** RS@k number as a reference — only its qualitative design (batch ≈ 4000).
- **MS-loss batch ablation (SOP 71.40 @ B=20 → 78.35 @ B=1000; CUB degrading):** taken from a search summary of the MS paper, **not** verified against the PDF. It is load-bearing for §2.1 and §5 and must be verified before any write-up cites it.
- **WEINCE (arXiv 2606.00262):** read via HTML render; I did not verify its Eq. 3.1 against a PDF.

---

**Sources:** [PFML (CVPR 2025)](https://openaccess.thecvf.com/content/CVPR2025/html/Bhatnagar_Potential_Field_Based_Deep_Metric_Learning_CVPR_2025_paper.html) · [PFML arXiv](https://arxiv.org/abs/2405.18560) · [WEINCE](https://arxiv.org/html/2606.00262) · [XBM](https://arxiv.org/abs/1912.06798) · [Recall@k Surrogate](https://arxiv.org/abs/2108.11179) · [Proxy-Anchor](https://openaccess.thecvf.com/content_CVPR_2020/papers/Kim_Proxy_Anchor_Loss_for_Deep_Metric_Learning_CVPR_2020_paper.pdf) · [Proxy-Anchor repo](https://github.com/tjddus9597/Proxy-Anchor-CVPR2020/blob/master/README.md) · [PA+DADA (AAAI 2024)](https://ojs.aaai.org/index.php/AAAI/article/view/29400) · [Multi-Similarity](https://arxiv.org/pdf/1904.06627) · [Threshold-Consistent Margin](https://arxiv.org/html/2307.04047v2) · [LogQ correction](https://arxiv.org/abs/2507.09331) · [EVT recognition score normalization](https://www.wjscheirer.com/papers/wjs_eccv2010_fusion.pdf) · [Surprise: EVT list truncation](https://arxiv.org/pdf/2010.09797) · [Metric Learning Reality Check](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123700681.pdf)
