# Opus blind proposal, Pass 26

Consultation ID: `c49f49e6e8a94628`  
Provider/model: Claude Opus  
Caller: `sfora/emafactorial`  
Status: completed, exit 0  
Prompt: `docs/opus_blind_prompt_pass26_2026-08-06.txt`  
Prompt SHA-256: `3e4040980b388bd9d899a1c7f574ccb721eef99cb59525778d2b0f05ae60d04d`  
Native result truncated: false

I ran the novelty search cold. One near-miss surfaced (WEINCE, arXiv 2606.00262) that I had to read before I could commit; it constrains but does not occupy the mechanism. Here is my proposal.

---

# PORT — Peaks-Over-threshold Retrieval Training

**One sentence.** Existing DML losses apply their hinge to an *aggregate of the negatives the batch happens to contain*; PORT applies the same hinge to a **differentiable extreme-value estimate of the hardest confuser in an identity population the batch never contains**, obtained by fitting a Generalized Pareto tail to the batch's own declustered negative similarities and extrapolating past the observed maximum.

**Lane: A** (ResNet-50, 512-D, ~224 px, single-view cosine, 200 epochs). All forecasts and comparisons below are Lane A only.

---

## 1. Executable mathematics

### 1.0 Architecture, data path, recipe

- Backbone `φ`: ResNet-50, ImageNet-1K supervised init. GAP → `h ∈ R^2048`.
- Embedding: `z = W h`, `W ∈ R^{512×2048}`, no bias. `f = z/‖z‖₂ ∈ S^511`.
- Train aug (ordinary, no colour jitter): `RandomResizedCrop(224, scale=(0.16,1), ratio=(3/4,4/3))` + hflip.
- Test: resize 256 → centre crop 224, single view, `f`, cosine NN. **Deployment is byte-identical to the baseline.**
- Optimizer AdamW; lr 1.2e-5 backbone, 1.2e-4 on `W`; cosine decay to 0, 5-epoch linear warmup; 200 epochs. BN **unfrozen** (stated because frozen-BN moves CUB by several points and must be matched).
- Weight decay 5e-4 on backbone conv/linear, **0 on BN params and 0 on `W`** — see §3(D5); this is not a cosmetic choice.
- Batch `B = 256`: `P=64` classes × `K=4` for CUB/Cars; `P=128` × `K=2` for SOP/In-Shop.
- **The EVT block runs in fp32** even under bf16 autocast (§6).

Base loss is Multi-Similarity (Wang et al., CVPR 2019), reduced exactly as published:

```
L_MS(i) = (1/α)·log[1 + Σ_{k∈P_i} e^{−α(S_ik − λ)}]  +  (1/β)·log[1 + Σ_{k∈N_i} e^{ β(S_ik − λ)}]
```
`α=2, β=50`. **Source ambiguity, stated:** the MS paper specifies `λ=1` with mining margin `ε=0.1`; the widely-used reimplementation uses `λ=0.5`. I use `λ=0.5` and flag that `λ` is the hinge offset on the same scale as PORT's increment, so it is operational, not cosmetic. I do **not** inherit MS's published numbers (different backbone/batch); C0 below is my own matched reproduction.

PORT **replaces the negative term only**.

### 1.1 Declustering (POT prerequisite)

`S_ab = ⟨f(x_a), f(x_b)⟩`. For anchor `a`, for each negative class `j` in the batch:

```
v_j(a) = (1/β_d)·log Σ_{b: y_b=j} exp(β_d · S_ab),      β_d = 24
```

POT for *dependent* sequences requires declustering; instances of one identity are exactly a dependence cluster. Class soft-maxima give `M = P−1` approximately independent block maxima. Overestimation bias is `≤ log(K)/β_d` (0.058 at K=4, 0.029 at K=2) — a constant absorbed into the threshold.

### 1.2 Threshold and excesses

Sort `v_(1) ≥ … ≥ v_(M)`. `k = ⌊κM⌋`, `κ = 0.25` (k=15 CUB/Cars, k=31 SOP/In-Shop).
`u_a = v_(k+1)` (kept differentiable). Excesses `e_i = v_(i) − u_a > 0`, `i=1..k`.

### 1.3 Closed-form differentiable GPD fit (probability-weighted moments)

With excesses re-ordered ascending `e_{1:k} ≤ … ≤ e_{k:k}` (Hosking–Wallis 1987):

```
a₀ = (1/k) Σ_j e_{j:k}
a₁ = (1/k) Σ_j [(k−j)/(k−1)] · e_{j:k}
D  = a₀ − 2a₁
ξ̂  = 2 − a₀/D
σ̂  = 2·a₀·a₁ / D
```

Verified on the exponential case (`ξ=0`, scale `σ`): `a₀=σ`, `a₁=σ/4`, `D=σ/2` ⇒ `ξ̂=0`, `σ̂=σ`. ✓

**Why PWM, not MLE.** GPD MLE has no closed form, needs inner iterations, and its regularity conditions *fail* for `ξ < −0.5` — precisely our regime (cosine similarity is bounded above by 1, so the true tail is Weibull-domain, `ξ<0`). PWM is a smooth rational function of the order statistics, so exact reverse-mode gradients exist with no implicit differentiation.

Guards. For a true GPD, `D/a₀ = 1/(2−ξ) ∈ [0.34, 0.67]` on `ξ∈[−0.9,0.3]`, so the degeneracy `D→0` is far from the operating point and the guard rarely fires. Blend to the exponential branch (`ξ̂=0, σ̂=a₀`) with weight `sigmoid((D/a₀ − 0.05)/0.02)`. Soft-clip `ξ̃ = ξ_lo + sp_τ(ξ̂−ξ_lo) − sp_τ(ξ̂−ξ_hi)`, `[ξ_lo,ξ_hi]=[−0.95, 0.30]`, `τ=0.05`. The upper clamp also enforces PWM's `ξ<0.5` validity condition; `ξ̂>0` here is a small-sample artifact, so clamping is principled rather than a patch.

### 1.4 Extrapolation to an unobserved identity population

Exceedance rate at `u_a` is `p̂ = k/M = κ`. Define the **log identity-density gain** `L = log(κN)`. From `P(X > u+y) = p̂(1+ξy/σ)^{−1/ξ} = 1/N`:

```
q̂_a(N) = u_a + σ̂ · expm1(ξ̃·L)/ξ̃          (Taylor branch σ̂L(1+ξ̃L/2) for |ξ̃L|<1e-3)
```

For `ξ̃<0` and large `L` this saturates at the estimated **right endpoint** `u_a + σ̂/|ξ̃|` — the worst confuser the population could ever produce.

Schedule, and the tension I am resolving explicitly: theory wants `N = ` deployment identity count; POT statistics are only trustworthy ~10–20× beyond the sample. So

```
L(t) = L₀ + (L_max − L₀)·½(1 − cos(π·min(t/T_ramp,1))),   L₀ = log k,  T_ramp = 0.3T
L_max = min( log(κ·C_train),  log k + Λ ),                Λ = 3.0
```

At `t=0`, `L = log k` ⇒ `q̂ ≈` the observed max ⇒ PORT ≡ hard mining, so early-training tail noise cannot hurt. The cap `Λ` yields, **with no per-dataset tuning**:

| | C_train | k | L_max | extrapolation past observed max |
|---|---|---|---|---|
| CUB | 100 | 15 | 3.22 (population-capped) | **1.7×** |
| Cars | 98 | 15 | 3.20 (population-capped) | **1.6×** |
| SOP | 11318 | 31 | 6.43 (Λ-capped) | **20×** |
| In-Shop | 3997 | 31 | 6.43 (Λ-capped) | **20×** |

This asymmetry is not a convenience — it is the method's **pre-registered falsifiable signature** (§5, F2).

### 1.5 Anti-gaming floor and the loss

```
ŝ⁻_a = max( q̂_a , v_(1) )
L⁻_a = (1/β)·log(1 + exp(β(ŝ⁻_a − λ)))
L⁺_a = (1/α)·log(1 + Σ_{b∈P_a} exp(−α(S_ab − λ)))
L    = (1/B) Σ_a (L⁺_a + L⁻_a)
```

New hyperparameters, total: `κ, Λ, β_d`, plus the `ξ` clamp. Everything else is inherited from MS.

### 1.6 Gradient path, in closed form

`∂L⁻/∂ŝ⁻ = σ(β(ŝ⁻−λ))`. On the `q̂` branch, with `E(ξ) := expm1(ξL)/ξ`:

```
∂q̂/∂u = 1 ;   ∂q̂/∂σ̂ = E(ξ̃) ;   ∂q̂/∂ξ̃ = σ̂·E′(ξ̃),  E′(ξ) = [ξL e^{ξL} − e^{ξL} + 1]/ξ²
∂σ̂/∂a₀ = −4a₁²/D² ;  ∂σ̂/∂a₁ = +2a₀²/D²
∂ξ̂/∂a₀ = +2a₁/D²  ;  ∂ξ̂/∂a₁ = −2a₀/D²
∂a₀/∂e_{j:k} = 1/k ;  ∂a₁/∂e_{j:k} = (k−j)/(k(k−1))
∂v_j/∂S_ab = softmax_{β_d} weight  → W → φ
```

`E′(ξ) > 0` everywhere (let `g(ξ)=ξLe^{ξL}−e^{ξL}+1`; `g(0)=0`, `g′(ξ)=ξL²e^{ξL}` has the sign of `ξ`, so `g>0` for `ξ≠0`), i.e. **lowering the fitted tail shape directly lowers the loss** — `ξ̃` is a genuine optimization target, not a statistic.

**Fingerprint of the mechanism.** Composing the chain at the extreme rank `j=k` (`w_k=0`) gives `σ̂`-channel `−4a₁²E/(kD²) < 0` and `ξ̃`-channel `+2σ̂E′a₁/(kD²) > 0`; at `j=1` the two signs *reverse*. So PORT's per-rank gradient weights are **sign-mixed and determined by the fitted tail, not by rank**. Every occupied weighting is single-signed and rank-determined: exponential (MS/InfoNCE), uniform-over-top-k (CVaR), one-hot (hard mining), inverse-power (PFML). This is directly measurable and is the cleanest empirical handle on whether the claimed mechanism is the one operating. It is also a risk — see D6.

---

## 2. The causal zero-shot error mode

**Population-extreme under-estimation ("small-world margin").**

Deployment R@1 is decided by a maximum over `N_test` *unseen* identities. Every batch loss is a functional of at most `M = P−1` seen identities: it controls the `(1−1/M)` quantile, while deployment needs `(1−1/N)`. Under the POT model the gap is

```
Δ(M→N) = (σ/ξ)·(κM)^ξ·[(N/M)^ξ − 1]   →   σ·log(N/M)  as ξ→0
```

Three consequences, and they are the whole argument:

1. **It does not vanish.** It decays only as `M^ξ` with `|ξ| ≲ 0.5`. At `ξ=−0.3`, halving the gap requires `10×` the batch. Batch scaling is a losing race, not a fix.
2. **It is quantitatively first-order.** SOP: `M=127`, `N_test=11,316` ⇒ `log(N/M)=4.49`. Measured `σ̂` for cosine similarities on a trained R50/512 is `O(0.015–0.03)` ⇒ gap **0.07–0.13 in cosine units**, comparable to the entire MS hinge band. This is not a second-order correction.
3. **It is anchor-dependent.** `σ, ξ` vary per query, so training systematically under-protects exactly the anchors sitting in heavy-tailed, crowded neighbourhoods — which is where zero-shot errors concentrate.

**Why a tail fitted on seen identities should transfer to unseen ones.** `(u_a, σ̂_a, ξ̂_a)` are not properties of the training label set; they describe how fast the mass of *everything that is not this identity* accumulates as similarity rises toward `f(x_a)` — a local property of the map `f` on the sphere. Unseen identities are embedded by the same `f` into the same sphere. **This is the load-bearing assumption and it is the method's main scientific risk.** It is cheaply testable and I would run that test first (§7).

---

## 3. Proof-level attacks on the cheapest degeneracies

**Lemma 1 (domination — the anti-gaming guarantee).** `ŝ⁻_a = max(q̂_a, v_(1)) ≥ v_(1)`, and `x ↦ (1/β)log(1+e^{β(x−λ)})` is strictly increasing. Hence `L_PORT ≥ L_hard` pointwise in `θ`, so `L_PORT(θ) ≤ ℓ ⟹ L_hard(θ) ≤ ℓ`. PORT's sublevel sets are contained in hard-mining's. ∎

*Consequence:* the EVT machinery can only ever **add** pressure. There exists no parameter setting that lowers PORT by making the estimator lie. This kills the single cheapest shortcut available to any loss built on a learned statistic.

**D1 — total collapse.** `f ≡ const` ⇒ all `S=1`, all `v_j=1`, `u=1`, `e≡0` ⇒ `a₀=a₁=0` ⇒ guard fires ⇒ `ξ̃=0, σ̂=0` ⇒ `q̂=u=1` ⇒ `ŝ⁻=1`, the **maximum** of `L⁻` on the reachable set. `∂L⁻/∂ŝ⁻ = σ(β(1−λ)) ≈ 1 ≠ 0`, so any separating perturbation strictly decreases `L⁻` at first order. Collapse is a strict maximum, not a stationary point. ∎

**D2 — estimator gaming.** By Lemma 1 the increment `q̂ − v_(1)` can only be driven to zero by making the fitted right endpoint coincide with the observed maximum, i.e. by making the negative-similarity law *terminate* at the observed max. That is not a cheat — it is exactly the target geometry ("no worse confuser exists in the population"). The degenerate solution and the desired solution coincide. ∎

**D3 — subset gaming.** Could the network satisfy the terminating-tail property only on the class subsets it happens to see? Classes are resampled i.i.d. each step, and under the POT domain-of-attraction condition `û + σ̂/|ξ̂|` is consistent for the population endpoint as `k→∞`; a persistent gap therefore contributes a nonzero-mean gradient every step and is not a stationary point of the expected loss. **This is an argument under an assumption, not a theorem** — POT consistency is asserted for a similarity law that training is itself changing. Stated as a limitation, not closed.

**D4 — threshold gaming.** Pushing `v_(k+1)` down without touching `v_(1)` inflates the excesses, inflates `σ̂`, and *raises* `q̂`. The estimator penalizes exactly this cliff-shaped profile; and `q̂ ≥ v_(1)` regardless.

**D5 — the scale question, answered rather than waved past.** `f` is L2-normalized, so `‖z‖` is a gauge freedom — but the normalization is **not** harmless. (i) Weight decay on `W` shrinks all singular values equally while the loss resists only along used directions, so `wd(W)>0` is an active pressure toward low-rank `W`; at `C_train=100` with a 512-D output this is a real rank-collapse channel. I set `wd(W)=0` and require the identical setting in every control. (ii) `σ̂` is in **raw cosine units** and `β=50` is fixed, so a more concentrated embedding yields a smaller `σ̂` and a smaller increment: PORT's effective margin is *coupled to* the representation's concentration. That coupling is the mechanism (it is what makes the margin adaptive), but it means `σ̂` must never be rescaled or standardized, and `β_d`, `κ`, `λ` all shift the operating point. I claim none of these normalizations are inert.

**D6 — the sign-mixed gradient is a genuine risk.** Some ranks receive an *attractive* contribution, which can pull mid-tail negatives together. Partial counterweights: Lemma-1 floor, the `+1` repulsive `u`-channel, and `L⁺`. Isolated by control C7, which is designed to *lose* if the mechanism is real.

---

## 4. Adversarial novelty search — nearest works and the one-sentence distinction

Searched inside DML and outside it (open-set recognition, EVT/statistics, contrastive SSL, DRO/tail-risk, hubness).

| Nearest work | Distinction |
|---|---|
| **WEINCE**, "When Softmax Fails at the Top: Extreme-Value Corrections for InfoNCE" (arXiv 2606.00262, 2026) — **closest** | Uses EVT to reshape the *link function* over the observed batch negatives with the tail statistics held explicitly `stop_grad` and with no extrapolation beyond the batch, on SSL benchmarks (CIFAR/STL/ImageNet-32/SimCSE); PORT differentiates *through* the fit and applies the hinge to an out-of-sample quantile at an identity population the batch never contains. |
| **XBM** (CVPR'20) | Enlarges the *observed* negative set with stale memory; PORT observes nothing extra and infers the unobserved extreme from the batch's own tail shape. |
| **Recall@k Surrogate** (CVPR'22) | Closes the batch→database gap by making the batch large enough to approximate the database; PORT closes the same gap by extrapolation instead of observation. |
| **Large-batch gradient bias** (NeurIPS'22) | Corrects a *mean* functional (softmax denominator) toward infinite negatives, a bias that is `O(1/M)` and vanishes; PORT corrects an *extreme* functional whose gap is `Θ(σ log(N/M))` and does not. |
| **OpenMax / EVM / MetaMax** | Fit Weibulls to non-match scores *post hoc* on a frozen model to make an open-set decision; PORT makes the fitted tail an optimization target during training and never touches test scores. |
| **Histogram Loss** (NIPS'16) | Models the empirical pos/neg similarity distributions and minimizes an AUC-type average with no support beyond the observed range; PORT targets a single quantile *outside* that range. |
| **PFML** (CVPR'25) | Minimizes a superposed inverse-power potential energy — a *sum* over observed embeddings and proxies; PORT minimizes an inferred *maximum* over an unobserved identity population. |
| **PA+DADA** (AAAI'24) | Closes the sample–proxy *domain* gap by feature-mixup alignment; PORT closes the batch–population *extremal* gap and uses no proxies. |
| **Robinson et al.** (ICLR'21), **MoCHi** (NeurIPS'20) | Reweight or synthesize within/near the observed negative set; PORT changes the argument of the hinge, not the sampling distribution. |
| **CVaR / DRO losses** | Average the worst observed `τ`-fraction, a level bounded by `1/M`; PORT evaluates level `1/N ≪ 1/M`, which no within-sample statistic can reach. |
| **MagFace / AdaFace / CurricularFace** | Set the margin from feature norm or a difficulty heuristic; PORT's margin *is* the estimated gap between the sample max and a fitted population quantile, and has no margin hyperparameter of its own. |
| **Hubness correction** (Dinu'15; ridge-regression ZSL) | Global/test-time rescaling of the NN rule; PORT is train-time only and never touches the gallery. |
| **AdvRF** (ICCV'25, Lane B) | Training-only ResNet-34/U-Net reconstruction plus distillation; PORT adds no network, no teacher, <1% compute. |

**Residual novelty risk, stated plainly:** WEINCE is a 2026 preprint that appeared at/after my knowledge cutoff and I read only its HTML. Its two disclaimers — `stop_grad` tail statistics, no extrapolation — are exactly the two things PORT does. If the final version adds either, PORT's novelty narrows to "out-of-sample identity-population extrapolation for zero-shot DML," which is thinner. Control C5 is deliberately placed on that boundary.

---

## 5. Matched-compute controls, forecasts, falsification

### Controls (same recipe, seeds, augmentation, `wd(W)=0`, matched wall-clock)

- **C0** plain MS (no declustering, standard MS mining) — the only legitimate reference for Δ.
- **C1** MS + declustering, `L ≡ L₀` (no extrapolation). Separates declustering from the mechanism.
- **C2 — decisive.** Replace `Δ_a = q̂_a − v_(1)` by the dataset-mean scalar `Δ̄`. If C2 ≈ PORT, the mechanism is a margin.
- **C3** shuffled increment: permute `{Δ_a}` across anchors. Preserves the marginal, destroys anchor coupling.
- **C4** CVaR: `q̂ ←` mean of top-k. Tail averaging without extrapolation.
- **C5 — the WEINCE boundary.** Detach `σ̂, ξ̃` (keep `u` differentiable).
- **C6** force `ξ̃≡0` ⇒ `q̂ = u + σ̂L`. Isolates shape from scale.
- **C7** sign-clipped: zero every attractive component of `∂L⁻/∂S_ab`.
- **C8** MS+XBM sized to `e^{L_max}/κ` effective negatives, matched wall-clock.
- **C9** C0 at 2× and 4× batch, matched wall-clock (fewer steps).
- **C10** placebo: `(σ̂,ξ̃)` resampled from their empirical marginal, anchor-independent.

### Frozen forecasts — Lane A, R50 / 512-D / 224 px / 200 ep / 5 seeds

| | C0 (my MS base) | C1 | **PORT** | Δ vs C0 | Lane-A reference |
|---|---|---|---|---|---|
| CUB | 0.691 ± 0.006 | 0.694 | **0.704 ± 0.006** | +0.013 | PFML 0.734 ± 0.003 |
| Cars196 | 0.884 ± 0.006 | 0.887 | **0.898 ± 0.005** | +0.014 | PFML 0.927 ± 0.003 |
| SOP | 0.799 ± 0.003 | 0.803 | **0.821 ± 0.003** | +0.022 | PFML 0.829 ± 0.002 |
| In-Shop | 0.905 ± 0.004 | 0.908 | **0.925 ± 0.004** | +0.020 | PA+DADA 0.930 (seeds unreported) |

### Frontier-crossing arithmetic — stated without varnish

**PORT on an MS base does not cross the Lane-A frontier on any dataset.** Deficits: CUB −0.030, Cars −0.029, SOP −0.008, In-Shop −0.005.

A second, **conditional and unclaimed** forecast: PORT is base-agnostic, and the natural composition replaces PFML's *repulsive superposition* with the extrapolated extreme while keeping its attractive term. PFML's `α`-power repulsion already spreads pressure over far negatives, so the marginal Δ should shrink. Conditional on a faithful PFML reproduction landing within ±0.004 of published:

- CUB `0.739 ± 0.006` — crossing margin +0.005, **inside the combined ±0.007 band; not claimable.**
- Cars `0.932 ± 0.005` — +0.005, **not claimable.**
- SOP `0.841 ± 0.003` — +0.012, `≈3.3σ`, **the only claimable crossing.**
- In-Shop — **PFML does not evaluate In-Shop**, so no inheritable base exists; PORT on a PA base forecasts `0.927 ± 0.004`, below 0.930.

I do not claim the conditional row. Per the brief's own rule I cannot inherit a frontier without a matched reproduction, and **PFML's weight decay, batch size, backbone-vs-embedding LR split, LR schedule, BN handling, augmentation details, and proxy initialization are not disclosed** in the sources I could read (I could read the arXiv HTML; the CVF PDF returned 403). Disclosed and reproduced exactly: potential `ψ_att = −1/max(r,δ)^α`, `ψ_rep = +1/max(r,δ)^α` capped at `1/δ^α`, total energy `U = Σ_i Ψ_{y_i}(z_i) + Σ_{j,k} Ψ_j(p_{j,k})`, `α ∈ {0..6}` and `δ ∈ [0.1,0.3]` cross-validated, Adam 5e-4 network / 5e-2 proxies, 200 epochs, R50 512-D, 224 train / 256→224 centre-crop eval, M=15 proxies/class on CUB & Cars and M=2 on SOP. **Only the C0-vs-PORT contrast is a real claim.**

### Pre-registered falsification thresholds

- **F1** `Δ(SOP) < +0.010` over C0 at 5 seeds ⇒ method fails.
- **F2 — the signature.** Require `Δ(SOP)` and `Δ(In-Shop)` each `≥ Δ(CUB)+0.004` and `≥ Δ(Cars)+0.004`. This ordering falls out of `L_max` being population-capped on CUB/Cars and Λ-capped on SOP/In-Shop, with no per-dataset tuning. More than one inversion ⇒ the gain is not extrapolation.
- **F3** C2 within 0.004 of PORT on SOP ⇒ the mechanism is a margin; method dead.
- **F4** C5 within 0.004 on SOP ⇒ the differentiable-fit claim dies and WEINCE occupies the ground.
- **F5** C6 within 0.004 on SOP ⇒ shape estimation inert; degenerates to "scale-adaptive margin," weaker and more occupied.
- **F6** mean `ξ̂` pinned at the upper clamp on >20% of anchors ⇒ Weibull-domain assumption violated, extrapolation unjustified.
- **F7** wall-clock overhead >3% or peak memory >2% over C0 ⇒ the cost claim fails.
- Prediction on C8/C9: PORT ≈ XBM on SOP at ~1/10 the memory and with no staleness knob, and PORT+XBM > either. If C8 strictly beats PORT by >0.005 at matched wall-clock *and* PORT adds nothing on top, the extrapolation claim is substantially weakened.

---

## 6. Cost, and benchmark / contamination risk

**Training cost.** The `B×B` similarity matrix already exists in MS. PORT adds one soft class-max reduction (`B×B → B×P`), one top-(k+1) partial sort per anchor, two linear forms, ~10 scalar ops: `≈3·B·P ≈ 0.05 MFLOP` against `≈3.1 TFLOP` for the R50 forward+backward at B=256 — about `10⁻⁸` of compute. Expected measured overhead **<1% wall-clock, +32 KB memory**, dominated by kernel launch. Compare DADA at ~1.06× epoch time / 1.01× memory, and AdvRF's extra ResNet-34 + U-Net.

**Deployment cost: zero.** One ResNet-50, one 224 view, one 512-D descriptor, cosine NN — identical to C0.

**Risks.**
- Compliance: no test data, no generated data, no text/VLM, no extra annotation, no transduction, no reranking, no gallery fitting. Clean.
- **Hyperparameter contamination is the live risk.** `κ` and `Λ` must be selected on a held-out *training-identity* split (CUB: fit on classes 1–80, select on 81–100, refit on 1–100). I would report both held-out-selected and test-selected values so the gap is visible rather than hidden.
- CUB/Cars test sets are small (5,924 / 8,131 images); seed σ ≈ 0.5–0.9 pt. Any Δ < 0.010 on CUB/Cars is uninterpretable at 5 seeds — which is precisely why my CUB/Cars forecasts are near the interpretability floor and I do not lean on them.
- ImageNet-1K pretraining is semantically close to CUB/Cars. Lane-wide confound, not method-specific.
- BN handling is undisclosed by PFML and moves CUB by several points; cross-paper comparison is not apples-to-apples.
- **Numerical hazard worth naming:** `D = a₀ − 2a₁` is a cancellation. In bf16 this silently corrupts `ξ̂`. The EVT block must be fp32 or the results are wrong without erroring.

---

## 7. Where I think this is weakest

1. **The transfer assumption in §2 is asserted, not proved.** It is cheaply testable and I would run it as a kill-test *before* the 200-epoch sweep: fit `ξ̂, σ̂` on train identities and on test identities of an already-trained model; the premise requires them to be close. If they are not, PORT's extrapolation is fitted to the wrong law and nothing downstream matters.
2. **Late-training saturation.** As seen-identity similarities saturate, `σ̂ → 0` and PORT degenerates to C0. The `L`-ramp mitigates but does not eliminate this. Failure signature: Δ collapsing over the last 50 epochs.
3. **`k=15` on CUB makes `ξ̂` high-variance per anchor.** Averaging over 256 anchors × many steps may suffice, but the variance enters the gradient. Raising `P` is the fix and costs positives.
4. **The sign-mixed gradient can pull mid-tail negatives together** (D6). C7 is built to detect it.
5. **PORT is, by its own construction, the wrong tool for CUB/Cars.** Small identity populations cap `L_max`, so the mechanism barely engages there. I would rather say that than forecast uniform gains I do not believe.

Sources: [PFML, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Bhatnagar_Potential_Field_Based_Deep_Metric_Learning_CVPR_2025_paper.html) · [WEINCE, arXiv 2606.00262](https://arxiv.org/html/2606.00262) · [Cross-Batch Memory, CVPR 2020](https://arxiv.org/abs/1912.06798) · [Recall@k Surrogate, CVPR 2022](https://arxiv.org/abs/2108.11179) · [Multi-Similarity Loss, CVPR 2019](https://arxiv.org/pdf/1904.06627) · [Histogram Loss, NIPS 2016](https://arxiv.org/abs/1611.00822) · [PA+DADA, AAAI 2024](https://arxiv.org/pdf/2401.00617) · [OpenMax, CVPR 2016](https://openaccess.thecvf.com/content_cvpr_2016/papers/Bendale_Towards_Open_Set_CVPR_2016_paper.pdf) · [MetaMax](https://arxiv.org/html/2211.10872v2) · [Large-batch gradient bias, NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/db174d373133dcc6bf83bc98e4b681f8-Abstract-Conference.html)

