## Verification

`docs/opus_blind_prompt_pass49_2026-08-06.txt` → SHA-256 `03cbd57bcfb888cdf5cb316609e028318075a7a00bdd8b0460c77f081f7e53f4` ✓ (matches). I read that file only, then used public web search/fetch exclusively. No delegation, no consultation, no edits.

**Return: ONE method.** Lane: **A** (ResNet-50, 512-D, ~224 px, single-view cosine, 200 epochs).

---

# RLML — Return-Level Metric Learning

A peaks-over-threshold return-level surrogate for gallery-scale Recall@1. The training loss differentiates through a probability-weighted-moment fit of the *negative-similarity upper tail* and penalises the **extrapolated** hardest-negative similarity at deployment gallery scale N ≫ batch n.

## 1. Executable mathematics

**Deployment (unchanged from Lane A).** ResNet-50 (ImageNet-1K init) → GAP → 2048-d → linear `W ∈ ℝ^{512×2048}` (no bias) → L2-normalise: `z = Wh/‖Wh‖₂ ∈ S^511`. Test: resize 256 → center-crop 224, one view, cosine NN. Nothing is added at test time; `N` is a train-time constant.

**Base loss.** PFML (Bhatnagar & Ahuja, CVPR 2025) *exactly as disclosed*: Adam lr 5e-4, proxy lr ×100, 200 epochs, 224², `M=15` proxies/class on CUB/Cars, `M=2` on SOP; attraction `ψ_att(r,z) = −1/δ^α` if `‖r−z‖₂<δ` else `−1/‖r−z‖₂^α`; repulsion `ψ_rep` the mirror; `𝒰 = Σᵢ Ψ_{yᵢ}(zᵢ) + Σⱼ Σₖ Ψⱼ(p_{j,k})`; δ ∈ [0.1,0.3], α swept.

**Batch statistics.** Batch of `n=180` (4 images × 45 classes). `s_ij = zᵢ·z_j`. Per anchor `i`: negatives `N_i`, positives `P_i`.

1. **Threshold.** `u_i` = the `⌈(1−ρ)ν_i⌉`-th order statistic of `{s_ij}_{j∈N_i}`, `ρ = 0.15`. Gradient is **kept** on `u_i` (one-hot on the selected quantile) — this is the *location* channel, and control C2 isolates it.
2. **Exceedances.** `e_ij = s_ij − u_i > 0`, pooled across all anchors → sorted `e_(1) ≤ … ≤ e_(K)`, `K ≈ ρν n ≈ 4700`. If `K < 64`, fall back to `M̂_i = u_i`.
3. **PWM fit** (Hosking–Wallis 1987), with plotting position `c_r = (K−r)/(K−1)`:

   `a₀ = (1/K) Σ_r e_(r)` , `a₁ = (1/K) Σ_r c_r e_(r)`

   **Endpoint-stable closed form** (this is the whole estimator — ~10 lines of code):

   `ξ̂ = (a₀ − 4a₁)/(a₀ − 2a₁)` , `σ̂/ξ̂ = 2a₀a₁/(a₀ − 4a₁)` , `σ̂ = 2a₀a₁/(a₀ − 2a₁)`

   Sanity: exponential tail `(a₀=s, a₁=s/4)` → `ξ̂=0, σ̂=s` ✓. Uniform-on-[0,σ] → `ξ̂=−1, σ̂=σ` ✓.
4. **Return level.** `m = θ ρ N`, `N := |D_train|` (train-split cardinality — a *train-only* quantity), `θ = 1/n̄_c` the extremal-index discount for clustered extremes (`n̄_c` = mean images/class in the train split).

   `R = (σ̂/ξ̂)(m^{ξ̂} − 1)` for `|ξ̂| > 10⁻³`; else `R = σ̂ log m ·(1 + ξ̂ log m/2 + ξ̂² log²m/6)`.

   **Cosine-cap projection** (physical: similarity ≤ 1): `M̂_i = u_i + (1−u_i)·tanh(R/(1−u_i))`.
5. **Stabilisation.** Clamp `ξ̂ ∈ [−4, 0.3]` (straight-through); EMA `ξ̄,σ̄` with momentum 0.9 used as *value*, current batch as *gradient* (`ξ̃ = ξ̄ + ξ̂ − sg[ξ̂]`).
6. **Positive side.** R@1 is a max over positives: `s_i⁺ = (1/β) log[(1/π_i) Σ_{p} e^{βs_ip}]`, β = 16 (log-mean-exp, lower-biased smooth max, π_i-free).
7. **Loss.** `L_tail = (1/n) Σ_i (1/γ)·softplus(γ·clamp(M̂_i + Δ − s_i⁺, ≤1))`, γ = 8, Δ = 0.05.
8. **Total.** `L = L_PFML + λ(t)·L_tail`; λ ramps 0→λ_max over epochs 20→40 (before that the tail is uninformative and the POT threshold meaningless).

**λ must be gradient-norm matched, not a fixed scalar.** PFML's `1/r^α` potentials (α up to 6) have magnitudes not commensurable with a softplus hinge. At end of warmup, measure `ḡ_base = ‖∂L_PFML/∂Z‖_F`, `ḡ_tail = ‖∂L_tail/∂Z‖_F` over 50 steps; set `λ_max = κ·ḡ_base/ḡ_tail`, κ ∈ {0.1, 0.25, 0.5}. Related: weight decay on `W` does not change `z` (scale cancels under L2-norm) but *does* change the effective angular learning rate — the absolute margin Δ is therefore operational, not cosmetic, and the κ-sweep must be re-run if weight decay changes.

**Gradient paths.**
- `∂L/∂s_ip = −(1/n)·logistic(γ·arg)·w_ip`, `w_ip = softmax_β(s_ip)` — hardest positive pulled hardest.
- `∂L/∂u_i = +(1/n)·logistic(γ·arg)` onto the ρ-quantile negative — the location channel.
- `∂L/∂e_(r)`: since `a₀, a₁` are linear in `e_(r)` with weights `1/K` and `c_r/K`, the total return-level gradient is **affine in `c_r`**: `dR/de_(r) = A + B·c_r`. Worked at ξ≈0: `A = log m(log m − 1)/K`, `B = log m(8 − 4log m)/K`, sign flip at `c* = (log m − 1)/(4 log m − 8)`. For SOP (`log m ≈ 7.4`): `c* ≈ 0.29`, i.e. **exceedances below the ~71st percentile of the tail are pulled *closer*, above are pushed away.** No hardness-monotone pair loss can produce a negative-repulsion gradient of either sign; this is the mechanism fingerprint and ablation C8/F5 tests it directly.

## 2. Causal zero-shot error mode + degeneracy attack

**Gallery-scale extremal mismatch under identity shift.** R@1 error `= P(max_{g∈G⁻} s(q,g) > max_{p∈G⁺} s(q,p))`. Batch losses supervise the `(1−1/n)` quantile of the negative-similarity law; the test decision sits at `(1−1/N)`. Under POT the unsupervised gap is

`Q(1−1/N) − Q(1−1/n) = (σ/ξ)[(ρN)^ξ − (ρn)^ξ] → σ·log(N/n)` at ξ→0.

For SOP, `N/n ≈ 336`; with σ ≈ 0.03 that is ≈ 0.17 in cosine — **the same order as the entire margin budget (δ = 0.1)**, and it is a quantity no batch-max loss ever observes. Zero-shot amplifies it twice: the gallery grows, and the identities generating the extremes were never trained against.

**Degeneracies, attacked:**

- **P1 (collapse strictly dominated).** If all `z_i = z`: `s_ij=1`, `u_i=1`, `K=0` → fallback `M̂_i = u_i = 1`, `s⁺=1`, `L_tail = softplus(γΔ)/γ`. Compare `z_i = c_{y_i}` with distinct centroids, `max_{c≠c'}⟨c,c'⟩ = τ < 1−Δ`: `M̂_i = τ`, `s⁺ = 1`, `L_tail = softplus(γ(τ+Δ−1))/γ < softplus(0)/γ`. Numerically 0.089 vs 0.0015 — **60× penalty**. ∎
- **P2 (tail flattening does not blow up, and does not pay).** As exceedances degenerate to a point mass, `a₁ → a₀/2`, so `ξ̂ → −∞` and `σ̂ → ∞` *individually*, but `R = (σ̂/ξ̂)(m^{ξ̂}−1) → 2a₀(a₀/2)/(−a₀)·(−1) = a₀` — the estimator converges to the correct degenerate endpoint. The endpoint-stable parameterisation makes this numerically exact; there is no exploitable singularity. ∎
- **Global contraction shortcut.** Uniformly shrinking all similarities scales `R` and `(s⁺−u_i)` together, so it reduces the loss magnitude — but Δ is an *absolute* floor (`L → softplus(γΔ)/γ > 0`), and PFML's `1/r^α` repulsion diverges under contraction. Control C4 detects any residual "you just added margin" effect.
- **Threshold gaming.** Lowering `u_i` by pushing most negatives away while keeping a few very close raises the exceedances, hence `σ̂` and `R`. Penalised.
- **Shape estimator variance.** Per-anchor `k_i ≈ 26` is too few (`sd(ξ̂) ≈ 0.25`); pooling gives `K ≈ 4700` (`sd ≈ 0.015`). Pooling buys precision at the cost of assumption A1 below.

## 3. Adversarial novelty search (primary sources) — nearest works and distinctions

| Work | One-sentence mechanism distinction |
|---|---|
| **WEINCE**, Erol et al., arXiv:2606.00262, 29 May 2026 — *nearest work overall* | WEINCE **stop-grads** its tail statistics (`"we treat (λᵢ, β̂ᵢ) as stop_grad statistics"`) and blends *within-batch* logits; RLML backpropagates *through* the PWM fit and its entire purpose is a return-level extrapolation to `N ≫ n`, which WEINCE never computes — and it is self-supervised InfoNCE on CIFAR/STL/ImageNet-32, never DML benchmarks. |
| **OpenMax** (Bendale & Boult), **EVM**, **MetaMax**, GPD-FAR extrapolation in biometrics | These fit EVT tails *post hoc* to a **frozen** network's scores to calibrate rejection or extrapolate FAR; RLML differentiates the tail fit to *change the representation*. |
| **EQRN**, Pasche & Engelke, AoAS 2024 | EQRN puts a **GPD head** on the network and trains by GPD deviance; RLML has no head and no likelihood — ξ, σ are emergent statistics of the embedding geometry and the loss is a return-level hinge against the positive similarity. |
| **XBM**, Wang et al., CVPR 2020 | XBM *observes* more negatives at O(Md) memory and only for **seen** classes; RLML *models* the tail at O(1) and extrapolates to a gallery of unseen identities no memory can hold. |
| **Recall@k Surrogate** (Patel et al., CVPR 2022), **Smooth-AP** | These *pay* for gallery scale with very large batches and memory tricks; RLML *predicts* the gallery-scale order statistic from a small batch, and its gradient contains a `∂R/∂ξ` term no rank surrogate has. |
| **ROADMAP**, Ramzi et al., NeurIPS 2021 | ROADMAP names the batch-vs-training-set decomposability gap and closes it with an **absolute-threshold calibration** (a location fix on *seen* data); RLML's fix is a tail-**shape** extrapolation to a larger, disjoint-identity gallery. |
| **CVaR / DRO / hardness-weighted sampling** | CVaR is a linear functional of order statistics with **non-negative** weights and no extrapolation; RLML's weights are affine in `c_r` and **change sign** at `c*`. |
| **Multi-Similarity, Circle, Ranked List Loss, hard mining** | All assign non-negative repulsion to every negative; RLML *attracts* sub-`c*` exceedances. |
| **Roth et al. ICML 2020 (ρ-spectral), S2SD, NIR** | These regularise the global feature spectrum/isotropy; RLML constrains only the negative-similarity upper tail, with an explicit deployment scale `N`. |
| **Proxy Synthesis, Embedding Expansion, HDML** | They fabricate *samples*; RLML fabricates no data — it fabricates the *order statistic* those samples would have produced. |

## 4. Decisive matched-compute controls

All at identical epochs/batch/backbone, 5 seeds.

- **C1** `M̂ ← max_{j∈N_i} s_ij` (observed batch max) — separates extrapolation from hard-negative hinge.
- **C2** `M̂ ← u_i` (pure quantile hinge) and `M̂ ← u_i + mean(e)` (CVaR_ρ) — separates tail *shape* from tail *location*.
- **C3** `ξ̂ ≡ 0`, `σ̂` detached → `M̂ = u_i + σ̄ log m`. **Sharpest control**: isolates the learned shape parameter.
- **C4** `M̂ ← u_i + c`, `c` tuned to match C3's mean offset — rules out "extra margin."
- **C5** `N`-sweep ∈ {n, 10n, |D_train|, 100|D_train|} — mechanism predicts an interior optimum near `|D_train|`.
- **C6** PFML + XBM with memory `M = |D_train|` — the "observe instead of extrapolate" upper bound. If XBM-full ≥ RLML at equal epochs, extrapolation adds nothing.
- **C7** EMA off; per-anchor (unpooled) fit — tests A1.
- **C8** Clamp `∂L/∂e_(r) ≤ 0` (kill the pull-closer half). **Self-falsification**: if gains survive, RLML is just fancy hard-negative weighting and the novelty claim collapses.

## 5. Frozen forecasts, Lane A

`m = θρN`: **CUB 15** (log m 2.7), **Cars 15** (2.7), **SOP 1698** (7.4), **In-Shop 598** (6.4). The mechanism therefore *predicts near-zero gain on CUB/Cars* — with `m ≈ 15 < k_i ≈ 26`, the extrapolation does not even leave the batch. This is a feature of the design, not a hedge: CUB/Cars are **null-prediction controls**.

**Baselines must be reproduced in-house** (PFML's batch size is undisclosed; see §7). Do **not** inherit 0.734/0.927/0.829.

| Dataset | Reference (Lane A) | Predicted Δ (80% interval) | Predicted RLML |
|---|---|---|---|
| **SOP** | PFML 0.829 ± 0.002 (5 seeds) | **+1.1 pp** [+0.4, +1.9] | **0.840** [0.833, 0.848] |
| **In-Shop** | PA+DADA 0.930 (seeds unreported) | **+1.3 pp** [+0.5, +2.2] | **0.934** [0.926, 0.943] |
| Cars196 | PFML 0.927 ± 0.003 | +0.4 pp [−0.3, +1.1] | 0.931 |
| CUB | PFML 0.734 ± 0.003 | +0.3 pp [−0.4, +1.0] | 0.737 — **does not cross** |

**Frontier arithmetic (SOP).** PFML SEM = 0.002/√5 = 0.0009. Assuming a comparable RLML seed std, difference SE = √(0.0009²+0.0009²) = 0.0013; a 2-SE crossing needs Δ ≥ **+0.25 pp**. Central forecast +1.1 pp = 8.5 SE; low-interval +0.4 pp = 3.1 SE. **SOP is a well-powered frontier test either way.** In-Shop crossing 0.930 is ≈ a coin flip and, because the reference reports no seeds, a crossing there is weakly supported *on the reference side* — say so rather than claim it.

**Pre-registered falsification thresholds.**
- **F1 (primary, ordering).** Gain must be monotone in `log m`: SOP ≳ In-Shop > Cars ≈ CUB. **If ΔCUB exceeds ΔSOP by > 0.3 pp, the mechanism is falsified even if mean gain is positive.** (Hard-negative-mining improvements historically go the *other* way — largest on fine-grained CUB/Cars — so this is a genuinely discriminating prediction.)
- **F2.** `N = n` must give ≤ +0.1 pp; `N = 100|D_train|` must degrade ≥ 0.2 pp vs `N = |D_train|`. A flat `N`-curve falsifies.
- **F3.** C3 (ξ frozen at 0) must recover ≤ 50% of the gain. ≥ 80% ⇒ report as a log-N margin schedule, not an EVT method.
- **F4 (cheapest, run first).** On a *vanilla* PFML checkpoint, fit ξ̂ by the same POT procedure on train-split and test-split negatives. `|ξ̂_train − ξ̂_test| > 0.15` ⇒ the tail does not transfer to unseen identities and the extrapolation is mis-calibrated. **≈ 1 GPU-hour, no training — this can kill the method before any run.**
- **F5.** C8 must lose ≥ 50% of the gain.
- **F6.** Any dataset regressing > 0.5 pp at matched compute ⇒ reject.

## 6. Cost, and risks

**Train cost.** One extra `n×n` cosine block: 2·180²·512 = 33 MFLOP/step against ResNet-50 fwd+bwd ≈ 2.2 TFLOP/step → **1.5 × 10⁻⁵ relative FLOPs**. One sort of ~4700 scalars. Memory 180² floats ≈ 130 KB. Realistic wall-clock **1.00–1.02×**, launch-latency-dominated. Compare PA+DADA 1.06× epoch / 1.01× memory; AdvRF adds a whole ResNet-34/U-Net reconstruction system plus distillation; VAPNet adds attribute machinery. **Deployment cost: exactly zero delta** — same ResNet-50, same 512-D, same cosine NN.

**Risks.**
- **A1 (pooled homogeneity).** Shape/scale pooled across anchors and classes. Diagnostic: per-anchor ξ̂ IQR at epochs 50/100/200; IQR > 0.5 ⇒ A1 shaky. C7 tests it.
- **A2 (train→test tail transfer).** The load-bearing zero-shot assumption; F4 tests it directly and cheaply.
- **A3 (dependence).** Return levels assume exchangeable draws; gallery negatives cluster by identity. Handled by θ. Sensitivity is bounded and sub-linear: only `m^ξ` enters, so a 30× misspecification of `m` moves R by ≈ 18% of the endpoint gap at ξ = −0.5.
- **Soft-leakage optics.** `N = |D_train|` happens to land within ~2% of the test gallery size on all four benchmarks. This is a consequence of the standard 50/50 class split, not test knowledge — but a reviewer can reasonably press on it. F2's `N`-sweep is the mitigation: the method must be insensitive within an order of magnitude.
- **Knob inflation.** 7 new hyperparameters. Mitigation: **freeze ρ=0.15, β=16, γ=8, EMA=0.9, θ=1/n̄_c across all four datasets**; tune only λ (via gradient-norm matching, κ ∈ {0.1,0.25,0.5}) and Δ ∈ {0.02,0.05}, on CUB only, and transfer. Report the frozen set.
- **Contamination.** ImageNet-1K pretraining overlaps CUB/Cars semantics — matched with every cited reference, not differential. No test data, no external data, no text/VLM, no reranking, no transduction.

## 7. Unresolved source ambiguities

1. **PFML batch size, per-class sampler, LR schedule/decay, weight decay, warmup, and augmentation are not disclosed** in what I could read; `"δ between [0.1,0.3], α between {0,6}"` is itself ambiguous (set vs. interval). Consequence: RLML depends on batch composition, so PFML **must be reproduced in-house** across batch ∈ {90,150,180}, and the reproduction reported alongside the published 0.734/0.927/0.829 before any Δ is claimed.
2. **Whether PFML's proxies are L2-normalised is unstated** — this determines whether its potentials act on the sphere and therefore how the tail term composes with it.
3. **PFML does not evaluate In-Shop.** The In-Shop Lane A reference (PA+DADA 0.930) reports no seeds/uncertainty; an In-Shop baseline must be established in-house.
4. **Proxy Anchor's ResNet-50/512 row is unconfirmed.** The project page I could read shows CUB 72.2 / Cars 88.1 / SOP 79.2 / In-Shop 88.3, but I could not verify these are the ResNet-50 512-D rows (the PDF fetch returned unreadable binary). I therefore rely on **no** specific PA number.
5. Standard split cardinalities used for `N` (CUB 5,864 / Cars 8,054 / SOP 59,551 / In-Shop 25,882) are from recall, not verified this session; the implementation should read them from the split files.

**Honest summary of where this is weakest:** WEINCE (May 2026) already brings EVT to contrastive logit geometry, so the *domain* is no longer virgin — the distinction rests entirely on (a) backpropagating through the tail fit rather than stop-grad, and (b) the return-level extrapolation to `N ≫ n`. If C8 or C3 shows those two are not load-bearing, this reduces to a log-N margin schedule and should be reported as such.

---

Sources: [PFML (CVPR 2025)](https://arxiv.org/html/2405.18560v2) · [PFML CVF page](https://openaccess.thecvf.com/content/CVPR2025/html/Bhatnagar_Potential_Field_Based_Deep_Metric_Learning_CVPR_2025_paper.html) · [WEINCE (arXiv:2606.00262)](https://arxiv.org/abs/2606.00262) · [EQRN, Pasche & Engelke](https://arxiv.org/html/2208.07590v3) · [XBM (CVPR 2020)](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_Cross-Batch_Memory_for_Embedding_Learning_CVPR_2020_paper.pdf) · [Recall@k Surrogate (CVPR 2022)](https://arxiv.org/abs/2108.11179) · [ROADMAP (NeurIPS 2021)](https://arxiv.org/abs/2110.01445) · [MetaMax / OpenMax lineage](https://arxiv.org/html/2211.10872v2) · [Hosking–Wallis PWM for GPD](https://www.tandfonline.com/doi/abs/10.1080/00401706.1987.10488243) · [Proxy Anchor (CVPR 2020)](https://cvlab.postech.ac.kr/research/ProxyAnchor/) · [PA+DADA (AAAI 2024)](https://arxiv.org/pdf/2401.00617) · [AdvRF (ICCV 2025)](https://arxiv.org/abs/2507.21742) · [Roth et al., ICML 2020](http://proceedings.mlr.press/v119/roth20a/roth20a.pdf)
