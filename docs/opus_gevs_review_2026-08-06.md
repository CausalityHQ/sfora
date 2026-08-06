## VERDICT: **DEAD** — earliest failed gate: **Gate 1 (executable mathematics / mechanism)**

**Decisive mechanism-level reason.** The exposure term is algebraically inert at the operating point that carries the entire claim. Because `θ̂_i = K/(ĉ_i·n̄)`, the product `N·θ̂_i·ζ` reduces **exactly** to `K·ζ·C_train/ĉ_i` — the dataset size *N* cancels against `n̄`, leaving the training **class count**. Since ξ<0 forces saturation, GEVS's target sits within `c_∞·ȳ·(Nθζ)^ξ ≈ 0.002` cosine units of its own N→∞ endpoint limit on SOP, while the PWM shape estimator at k=36 injects ≈0.044 cosine units of noise into the same target. The advertised signal is **~20× below the noise of the estimator that computes it**, and that ratio is independent of `ȳ`. Gate 3 fails independently (§10 below).

---

## 1. Estimator reproduction — mostly correct, one stated limit wrong

Re-derived from scratch with `ā_r = E[Y·F̄(Y)^r] = σ/[(r+1)(r+1−ξ)]`:

- **PWM shape — correct.** `a_0/(a_0−2a_1) = [σ/(1−ξ)]/[σ/((1−ξ)(2−ξ))] = 2−ξ`, so `ξ̂ = 2 − a_0/(a_0−2a_1)`. Matches Hosking–Wallis (1987) under `k = −ξ`. The ascending/survival-weight convention `a_1 = (1/k)Σ y_[j](k−j)/(k−1)` is the correct plotting-position estimator: weight 1 at `y_[1]` (F̄≈1), weight 0 at `y_[k]` (F̄≈0). ✓
- **MoM scale — correct.** `E[Y]=σ/(1−ξ) ⇒ σ=(1−ξ)ȳ`. ✓
- **Coefficient c — correct.** Return level `u + (σ/ξ)[(Nθζ)^ξ − 1]` with `σ=(1−ξ)ȳ` gives exactly `c = (1−ξ̃)/(−ξ̃)·(1−(Nθζ)^ξ̃)`. ✓
- **ξ→0 limit — correct.** `c → log(Nθζ)`, consistent with C4. ✓
- **D3 numerical example — correct.** ξ=−0.5, Nθζ=3.8e3 → 3×(1−0.01622) = **2.951**. ✓
- **D1 derivative — correct for the written program.** `∂ŝ/∂log N = ȳ(1−ξ)(Nθζ)^ξ = σ(Nθζ)^ξ`. ✓

**Wrong: D5's limit.** By Chebyshev's sum inequality (`y_[j]` ascending, weights `1−2(k−j)/(k−1)` ascending, summing to 0), `a_0−2a_1 ≥ 0` **always**, so `ξ̂ ≤ 2` always and 2 is the *supremum*, not the degenerate value. All-excesses-equal gives `a_0−2a_1 → 0⁺` hence **ξ̂ → −∞**, not +2. The direction is inverted: the protective clamp is the −2 bound, not the −0.02 bound. Minor in effect, but it shows the limit was asserted, not computed.

## 2. Endpoint guard — **dead code; the D1 guarantee is false**

The boxed `c_i` is a function of `ξ̃_i, N, θ̂_i, ζ` only. `ŝ_i = u_i + c_i·ȳ_i`. **`σ_i` appears in neither.** It was pre-substituted into `c_i`'s `(1−ξ̃_i)` factor; clipping the substituted-out intermediate afterwards propagates nowhere. Per instruction I do not reconnect it.

Consequence: **`ŝ_i ≤ 1` does not follow.** Concrete failure — SOP, `u=0.35`, `ȳ=0.09`, `ξ̂` above −0.02 (which I estimate happens for ~10–25% of anchors per step given SE(ξ̂)≈0.5 and κ=0.5 shrinkage), so `ξ̃` clamps to −0.02: `c = clamp(51·(1−5030^−0.02), 0, 8) = clamp(7.99,0,8) = 8`, and `ŝ = 0.35 + 8(0.09) = **1.07**`. Since `s⁺ ≤ 1`, the hinge argument at the *global optimum* `s⁺=1` is `+0.17`, giving `L = 0.174` with sigmoid weight 0.938 — **94% of full gradient strength that never vanishes**, pushing every top-k negative down with force `0.938·c/k = 0.21` in perpetuity. That is precisely the D1 runaway the guard was written to block.

Had the guard been wired (`σ ≤ (−ξ)(1−u)`), `ŝ ≤ u+(1−u)(1−(Nθζ)^ξ) ≤ 1` would follow. But `c ≤ 8` and a separate σ-clip are two clamps on **the same quantity in two parameterizations**; clamping both independently and asserting a bound is incoherent regardless.

Second dead branch: `ĉ_i = k/#distinct ≤ k/⌈k/K⌉ ≤ K = 4` **always**, so `ĉ·n̄/K ≤ n̄` and the `min(·, n̄)` cap **never binds**.

## 3. LSE reduction — **not exact; the "N=M" corollary does not follow**

`L_α(s) = (1/α)log Σe^{αs_n}` is a deterministic function of a finite vector, bounded by `max_n s_n ≤ L_α ≤ max_n s_n + log M/α`. The POT object is `u + σ̂ log(Mζ)` with `σ̂` **estimated** from data.

Counterexample: M=100, α=32, ρ=0.2, all `s_n = 0`. LSE = log(100)/32 = **0.1439**. POT: u=0, all excesses 0, ŝ = **0**. Gap = log M/α, nonzero for every M>1.

Even under an exact exponential tail the two differ: `E[max] ≈ u + σlog(Mζ) + γσ` (γ≈0.577), so LSE tracks the *expected max* while the return level is the 1−1/M *quantile*. And `Mζ = k = ρM`, so the arguments differ by `log ρ = −1.61` nats. The identification `σ = 1/α` swaps a fixed hyperparameter for an estimated statistic; they coincide only when `ȳ = 1/α` by accident.

Therefore **"every LSE loss assumes N=M" is false.** A fixed-temperature LSE makes no commitment about N; it is a smooth-max whose relation to a gallery quantile is simply unspecified. The "three implicit assumptions" framing and the one-to-one C3/C4/C6 mapping rest on a false identity. Compounding this: the carrier's negative term aggregates *images per proxy*, not the `M = B−K` of the GEVS term — the reduction is not even connected to the loss it is bolted onto.

## 4. Extremal index — form defensible, transfer not

`k/#distinct` is a label-defined **mean cluster size**, and `θ ≈ 1/E[cluster size]` is standard. But it is *not* a runs or inter-exceedance estimator (Smith & Weissman 1994; Ferro & Segers 2003): those require an ordered sequence, a declustering parameter, and inferred clusters. Here there is no ordering and clusters come from exogenous labels. Naming aside, the transfer fails on four counts:

- **`ĉ·n̄/K` assumes cluster size is linear in class size.** At a high threshold it is not; the number of a class's images above a *fixed* level depends on within-class concentration relative to that level, and the batch's 80th-percentile threshold is nowhere near the 1−1/N gallery level.
- **`n̄` is a mean over a heavily skewed distribution** (SOP mode ≈2, mean 5.26). The max is produced by *one* confusing class whose size is a tail draw, not the mean.
- **In-Shop breaks the model.** Retrieval is query(14,218)→gallery(12,612 images / 3,985 classes ⇒ **3.16 img/class**), so both `N=25,882` and `n̄=6.48` are wrong by ≈2×. They cancel in `N·θ̂` (which is why the error is invisible), but the structural mismatch is real.
- **The sampler manufactures the statistic.** P×K forces exactly K=4 images/class, capping in-batch cluster size at 4 and making the cluster-size distribution a sampler artifact with no deployment analogue.

**The identity nobody states:** `N·θ̂_i·ζ = K·ζ·C_train/ĉ_i = 0.8·C_train/ĉ_i`. So CUB ≈ 44, Cars ≈ 44, SOP ≈ 5,030, In-Shop ≈ 1,776. The headline "training-set-derived exposure N" is, mechanically, the **training class count**, and per-anchor adaptivity is a coarsely quantized step function over `ĉ_i = 36/d`, `d ∈ {9..36}`, compressed further by `L^ξ`. This is never disclosed, and it means C6 (sweeping N with n̄ fixed) is not the test the proposal thinks it is.

## 5. Tail identifiability — the mechanism is below its own noise floor

Delta-method at ξ=0 (σ=1): `∂ξ̂/∂a_0 = 2`, `∂ξ̂/∂a_1 = −8`, `Var(a_0)=1/n`, `Var(a_1)≈0.0116/n`, `Cov≈0` ⇒ `Var(ξ̂) ≈ 4.74/n`. Effective n is **distinct clusters**, not exceedances: k/ĉ ≈ 20 (SOP), ≈13 (CUB). So **SE(ξ̂) ≈ 0.49 / 0.60**; after κ=0.5 shrinkage, SD(ξ̃) ≈ 0.24 / 0.30.

At ξ=−0.5, L=5030: `∂c/∂ξ ≈ 3.67` ⇒ **SD(c) ≈ 0.88** on a mean of 2.96 — 30% noise in the loss weight, every anchor, every step.

Now the signal. The entire exposure content of `c` is its deviation from the endpoint limit: `c_∞·L^ξ = 3 × 0.0141 = 0.042`. Ratio **SD(c)/(c_∞L^ξ) = 20.8**, and — critically — **this ratio is free of `ȳ`**, so it holds at any training stage.

Over the pre-registered 16× C6 sweep: `Δc = c_∞[(0.25L)^ξ − (4L)^ξ] = 3(0.0282−0.0071) = 0.063` — *smaller than one SD of the shape noise.* **F4 is guaranteed to pass whether or not the mechanism is real**, so C6 has zero discriminating power. And since a constant `c` is trivially flat in N and non-trivial, C1 and "flat under C6" are simultaneously satisfiable — the claim that "no constant-margin reparametrization can be simultaneously non-trivial and flat in N" is false, and the "single most decisive test" is vacuous against its own most likely null.

**The forecast contradicts the mechanism.** `c_∞L^ξ` = 0.452 (CUB) vs 0.049 (SOP): the exposure correction does **9× more work on CUB than on SOP** — the two datasets forecast at ≈null — and is essentially inert on the dataset carrying +1.8 and F1.

Additionally: ρ=0.2 is the upper **bulk**, not a tail. No mean-residual-life or threshold-stability diagnostic (Coles 2001 §4.3.1) is offered, ρ is fixed everywhere, and **there is no ρ control in C1–C8** — despite ρ jointly setting k, ζ, and u, i.e. every component of c. The parent is a nonstationary mixture (hard same-superclass negatives, easy negatives, augmentation structure) shifting over 200 epochs; the GPD limit is a fixed-parent statement.

## 6. Degeneracy and gradient claims

- **D4 is void.** The gradient is computed w.r.t. `z` as if unconstrained. With `∂z/∂v = (I−zz^T)/‖v‖`, at `z_i ≡ z` every term is parallel to z, so the projected gradient is **exactly zero** — collapse is a genuine critical point, not a non-minimum. "L > 0 ⇒ not a minimum" is a non-sequitur about criticality. The stated hedge understates the gap: the first-order argument itself does not survive the sphere Jacobian.
- **NaN, unhandled.** At/near collapse all excesses → 0, `a_0−2a_1 → 0` with `a_0 → 0`: **0/0 → NaN**, which `torch.clamp` propagates, which enters the EMA `ξ̄` — **poisoning `c_i` for every anchor and every subsequent step**. No ε, no NaN guard, no dtype policy. Under AMP this is likely rather than exotic: fp16 near 1.0 has spacing ≈4.9e−4, so cosines in [0.9,1.0] quantize to a coarse grid, ties become pervasive, and `a_0−2a_1` hits exact zero. No tie-breaking rule is given for `s_(k)=s_(k+1)` either.
- **D2 is a one-step argument only.** Stop-grad blocks a *differentiable* exploit; it says nothing about the 200-epoch trajectory. `ŝ = u + c·ȳ` rewards **small excess spread**: an embedding that compresses the top-k into a tight shell at `u` reduces the loss without lowering `u` — and near-equidistant negatives is exactly the geometry that destroys R@1 separation. Untouched by the stop-grad defense.
- **No ranking pressure inside top-k.** All `j ≤ k` receive identical weight `c/k`. The supervision object is `mean(top-k)`, not an extreme.
- **The reduction the proposal states itself.** With u, ξ, θ, ζ, N detached, `L_i = softplus_η(c_i·mean_k(s⁻) + [(1−c_i)u_i + m] − s⁺_i)` — an adaptively weighted top-k-mean-minus-best-positive hinge. Every EVT object enters through two detached scalars. §1.3's own sentence ("in gradient terms, a top-k negative penalty whose per-anchor strength c_i is an analytic function of…") is correct and is the concession.

## 7. Prior art — verified against primary sources; **this is the proposal's strongest section**

**WEINCE is real and correctly characterized.** [arXiv:2606.00262](https://arxiv.org/abs/2606.00262) (Erol, Evren, Ozel, Morgan, Ryu, Zheng; ICML 2026). Fetching the [HTML](https://arxiv.org/html/2606.00262) confirms: logits `ℓ_ij = (1−λ_i)s_ij/τ + λ_i·(−β̂_i log(1−s_ij))`; endpoint **fixed at x_F=1** (not estimated); Weibull **line fit** on the K_tail smallest shortfalls with AIC model selection, explicitly *not* POT/GPD during training; `(λ_i, β̂_i)` stop-grad; **no N, no M, no extremal index**; benchmarks CIFAR-10/100, STL-10, ImageNet-32, Tiny-ImageNet + SimCSE/STS under frozen-feature linear/kNN eval. Every distinction the proposal draws holds. Credit — it found the nearest work and described it accurately.

The distinctions from XBM, Recall@k Surrogate, MS/RLL/Smooth-AP/FastAP, PFML, DADA, TCM, logQ, and the OpenMax/EVM/Furon–Jégou/SIGIR-2020 test-time-EVT line are also correctly drawn *at the level of machinery*. **But the correct level is the supervision object**, and there the answer is §6's reduction: an adaptive top-k mean penalty. Judged on supervision object and optimization action rather than the wrapper, the novelty is the formula for one detached scalar weight — which §5 shows is dominated by its own estimation noise.

One deflation worth recording: `c/c_∞ = 1−L^ξ` = **0.984 on SOP**, so GEVS's target is 98.4% of the way from `u` to the fitted Weibull endpoint. At its operating point GEVS *is* an endpoint-shortfall correction; the extrapolation supplies the remaining 1.6%.

## 8. Causal error mode

- **CUB/Cars log-ratios are wrong.** `log(99/44)=0.811` and `log(97/44)=0.790` reproduce the stated 0.81/0.79 — but that uses **P−1=44**, the SOP/In-Shop batch, while §1.5's table specifies **30×4** for CUB/Cars. Correct values: `log(99/29)=1.23`, `log(97/29)=1.21` — 52% larger. SOP 5.55 and In-Shop 4.50 check out. The ordering survives; the load-bearing contrast was overstated by understating the small side.
- **MS batch ablation — not verified, and I could not verify it either.** [ar5iv render of 1904.06627](https://ar5iv.labs.arxiv.org/html/1904.06627) reports no batch-size ablation table (plausibly the numbers live in a figure and are unextractable); the CVF PDF returns 403 and the arXiv PDF failed to parse. So the **"CUB degrades with larger batches"** half — the sole independent corroboration for the near-null CUB forecast and the stated basis for holding F2 "with high confidence" — remains unconfirmed. The proposal flags this itself; holding a falsifier at high confidence on an unverified search summary is a breach of its own §7 standard.
- **Familiarity bias is a feedback loop, not just a residual.** The tail is fit on seen-class negatives whose upper tail the loss is actively truncating. As training converges, `ȳ` shrinks and `ξ̂` grows more negative, driving `c → 1` — **the GEVS term self-extinguishes precisely in the late-training regime that determines R@1.** Unaddressed.
- **Asymmetric extrapolation.** The negative extreme is extrapolated to gallery scale; the positive is not. At deployment CUB queries face ~58 positives vs 3 in batch — a *larger* correction than the negative-side one on CUB. Consistent with the null CUB forecast, but by an unnamed mechanism.
- **No link is established from a train-identity tail to unseen-identity R@1.** This is the central assumption and it is asserted, not argued.
- **A flat 16× sweep is not corroboration.** It is predicted equally by GEVS and by constant-c, and §5 shows it is arithmetically guaranteed. It cannot support a mechanism whose load-bearing input is N.

## 9. Controls and falsifiers

- **No non-adaptive control exists.** With `c` constant, `ŝ = (1−c)u_i + c·mean_k` — still per-anchor through `u_i`. C2 (`c=0`) gives `ŝ = u_i` — also per-anchor. So F3/F5 cannot separate "EVT adaptivity" from "any per-anchor threshold statistic." **Missing:** matched adaptive-margin and rank-affine top-k controls.
- **Search spaces are grossly unequal.** §1.5 pre-registers λ∈{0.25,0.5,1,2}×wd∈{1e-4,4e-4} "for both the method and the baseline" — but the baseline *is* λ=0, so its real grid is 2 configs against GEVS's 8. C1 gets 40. **The headline Δ_SOP is measured against a baseline with ¼ the selection capacity**, and F1's +0.8 threshold is unprotected against that asymmetry.
- **Validation split cannot bear the load.** "Last 20% of training identities" = **20 classes** on CUB/Cars. Selecting ρ, λ, m, η, κ and two clamps on 20-class R@1 does not mitigate the Musgrave et al. critique. Worse, carving 20% of identities changes `C_train` 100→80, which changes `N·θ·ζ = 0.8·C_train/ĉ` — **the method's own key quantity is not held fixed between selection and final retrain.**
- **Sampler feasibility unaddressed.** 45×4 requires 45 classes with ≥4 images. SOP (mean 5.26, mode ≈2) and In-Shop have many 2–3-image classes. Dropping them biases both the tail fit and `n̄`; sampling with replacement creates exact duplicates → ties → the PWM degeneracy of §6. Neither is specified.
- **EMA state is unsynchronized and arm-dependent.** `ξ̄` is one global scalar; DDP synchronization is unstated, and C3/C4 do not exercise it identically.
- **No power analysis, no multiplicity control.** With 5 seeds and σ≈0.2–0.3, SE of a mean difference is ≈0.19, so the 0.3-point equivalence margins in F3/F5 sit ~1.6 SE out and will resolve near-randomly. Six falsifiers at an implicit 5% level give a family-wise false-refutation rate near 26%.

## 10. Protocol and standing-objective arithmetic — **Gate 3 fails independently**

- **Corrected paired In-Shop screening: absent.** No paired control is proposed. The In-Shop entry is a R50/512-D forecast against a self-generated baseline, not a same-seed pairing with the corrected BN-Inception control (raw 0.9163 / final 0.9137). Different backbone; cannot be screened without a new run. The requirement is never acknowledged.
- **Raw vs independently selected/final: not decomposed.** §6 states the select→freeze→retrain protocol; §5 reports one number per cell.
- **Second dataset: effectively unavailable.** The mechanism forecasts null on CUB and Cars, so confirmation rests on SOP and In-Shop — and In-Shop is the dataset with the 2× N/n̄ and query/gallery-structure errors. SOP alone carries the claim.
- **No frontier is crossed on the frozen standalone forecasts.** Against the audited matched Lane-A references: SOP 82.0 vs 0.829 → **−0.9**; CUB 70.0 vs 0.734 → **−3.4**; Cars 88.1 vs 0.927 → **−4.6**. In-Shop 92.6 vs VAPNet 0.939 → −1.3, vs CRT 0.9448 → −1.9, vs PA+DADA 0.930 → −0.4. **Zero crossings on all four datasets.** The one claimed crossing (SOP over PA+DADA 81.0) is against a non-matched reference.
- **The only frontier path is uncreditable.** 84.3 requires the C8/PFML composition, which §7 concedes cannot be matched-reproduced from the sources (batch size, classes/batch, wd, LR schedule, per-dataset (δ,α) all undisclosed). The additive constant +1.4 appears nowhere else (standalone is +1.8 SOP / +1.3 In-Shop) and is unexplained. Additivity is also not orthogonal: PFML changes the interaction potential, hence the similarity distribution `ξ̂` is fit to. Per instruction, no credit.
- **Cost arithmetic ≈ correct.** 180²×512 = 17 MMAC vs ~2.2 TFLOP fwd+bwd ⇒ **0.0015%**, not "<0.001%" — immaterial. +130 KB for the B×B fp32 matrix ✓ (PA computes B×|P|, so this is genuinely new). "Measured overhead expectation <0.5%" is unmeasured; per-anchor sort + distinct-class count are latency, not FLOP, costs.

---

## Correct subcomponents, preserved separately from the verdict

Recorded so they survive the rejection and can be reused:

1. **PWM shape estimator with the ascending/survival-weight convention** — verified correct against Hosking & Wallis (1987).
2. **MoM scale `σ=(1−ξ)ȳ`** — correct.
3. **`c = (1−ξ)/(−ξ)·(1−(Nθζ)^ξ)`** as return-level-over-mean-excess, and its ξ→0 limit `log(Nθζ)` — both correct.
4. **The D3 finding** — that a differentiable threshold gives coefficient `(1−c) < 0` on u whenever c>1, rewarding *increases* in `s_(k+1)`, and that detaching u is the right fix (standard POT practice) — **correct, non-obvious, and worth keeping**. The numerical example c=2.951 checks out exactly.
5. **`∂ŝ/∂s_(j) = c/k` and the self-description as a top-k penalty with analytic strength** — correct.
6. **`∂ŝ/∂log N = σ(Nθζ)^ξ`** — correct for the written program.
7. **Training-split arithmetic** (5864/8054/59551/25882; n̄ = 58.6/82.2/5.26/6.48) and the SOP/In-Shop log-ratios (5.55/4.50) — correct.
8. **The WEINCE novelty distinction** — verified accurate against the primary source on every stated axis.
9. **§7's ambiguity disclosures** and the explicit refusal to inherit PFML 82.9 — correct scientific practice, and the reason several defects above were findable at all.
10. **The identity `N·θ̂·ζ = K·ζ·C_train/ĉ`** — my derivation, not the proposal's, but a correct and reusable fact about any extremal-index-corrected exposure term under a P×K sampler.

## Unresolved uncertainty

- **SE(ξ̂) ≈ 0.49/0.60** is my delta-method calculation at ξ=0 propagated to the clustered effective sample size; Hosking & Wallis's Monte Carlo is the authority and I did not retrieve their variance table. A factor-of-2 error here would still leave the signal ~10× below noise, so the conclusion is robust, but the constant is mine.
- **The MS-loss batch ablation is unverified in both directions.** One automated read found no such table; the numbers may be in a figure. I neither confirm nor refute 71.40→78.35, and specifically not "CUB degrades."
- **In-Shop query/gallery counts** (14,218 / 12,612 / 3,985 classes) are from memory, not re-fetched. If wrong, the §4 In-Shop point weakens; the `N·θ̂ ∝ C_train` identity does not depend on them.
- **`ȳ ≈ 0.05` and `u ≈ 0.35`** are my assumptions for the §2 counterexample. The §5 signal-to-noise ratio is `ȳ`-free and does not rest on them; the ŝ>1 counterexample does.
- I did not verify PA's repo numbers (69.9 / 87.7) or PA+DADA's recipe against primaries.

**Repair note.** Several defects have obvious fixes (wire the guard, drop the LSE-exactness claim, add a ρ control, add a non-adaptive control, equalize grids). Per the framing, a substantive repair is a **new proposal** and must not upgrade this frozen object. The signal-to-noise result in §5 is not a repair target — it is a property of fitting a 2-parameter tail to ~13–20 effective clusters and reading a 1.6% saturation residual off it.

**Sources:** [WEINCE abs](https://arxiv.org/abs/2606.00262) · [WEINCE HTML](https://arxiv.org/html/2606.00262) · [Multi-Similarity arXiv](https://arxiv.org/pdf/1904.06627) · [MS ar5iv render](https://ar5iv.labs.arxiv.org/html/1904.06627) · [Recall@k Surrogate](https://openaccess.thecvf.com/content/CVPR2022/papers/Patel_Recallk_Surrogate_Loss_With_Large_Batches_and_Similarity_Mixup_CVPR_2022_paper.pdf)
