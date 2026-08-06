+I have what I need. Here is the review.

---

# Independent Adversarial Review — XTail (frozen, pass 31)

**Verdict: DEAD.** Earliest failure: **§1.4(c)–(e)**, the estimator and extrapolator themselves, at ζ = 0.25 with k ≈ 36. Everything downstream (gradients, controls, forecasts) inherits it. All numbers below are my own fresh computation (`numpy`, no repo files read); scripts are at `/tmp/xtail_audit{,2,3}.py`.

---

## 1. The POT/GPD mathematics — correct, and I could not break it

I re-derived the Hosking–Wallis PWM estimator from scratch under the proposal's survival convention `1−F(z) = (1+ξz/σ)^(−1/ξ)`. With `a_s = E[Z(1−F(Z))^s] = σ/((s+1)(s+1−ξ))`:

`a₀/(a₀−2a₁) = 2−ξ` exactly ⟹ **`ξ̂ = 2 − a/(a−2b)` is right**, and the order-statistic weights `(k−r)/(k−1)` on ascending `z_(r)` are the correct unbiased PWM weights for `a₁`. Monte-Carlo at k = 2×10⁵ recovers ξ to ±0.005 for ξ ∈ {−0.4, −0.2, 0, 0.1}.

The scale plug-in is better than the proposal claims: `(1−ξ̂_raw)·a ≡ 2ab/(a−2b)`, i.e. it is *identically* the exact H–W PWM scale estimator, not an approximation. `A(R,ξ) = (1−ξ)(R^ξ−1)/ξ` is the correct POT quantile coefficient with `R = ζ·n_target`; `A|_{ξ→0} = ln R`; the Taylor branch `ln R + ξ(½ln²R − ln R)` has relative error 5×10⁻⁶ at |ξ| = 10⁻³ and 1×10⁻³ at |ξ| = 10⁻², so the 10⁻³ switch point is well chosen. `R ∈ {1466, 2014, 6471, 14888}` and `A ∈ [2.92, 18.28]` are exact. `∂A/∂ξ|₀ = ½ln²R − ln R = 36.55` at R = 14888 — their "≈37" is right.

Every dataset count is correct (N_train 5864/8054/25882/59551; galleries 5924/8131/12612/60502) and the extrapolation factors **41 / 56 / 87 / 417 = |G|/145 are exactly right**.

Two slips, neither material: §2.1(i) uses `R = 37.25` where n_i = 145 gives 36.25 (changes q by 0.0008); PFML's SOP SD is ±0.002, not ±0.003 ([arXiv:2405.18560](https://arxiv.org/html/2405.18560v2)).

**One real mathematical defect.** Cosine similarity is bounded by 1, so the data lie in the Weibull domain: ξ < 0 *and* the right endpoint `u + σ/(−ξ) ≤ 1` must hold. For u = 0.30, a = 0.05 that requires **ξ ≤ −0.077**. The clip `[−0.5, +0.15]` admits an entire region — including all of ξ > 0 — where the fitted model has an infinite or > 1 right endpoint. The endpoint constraint is never imposed, and ξ > 0 is not a variance nuisance; it is outside the admissible parameter space for the data-generating process.

---

## 2. Can 36 exceedances identify a tail 41×–417× deeper? No — and the failure is systematic, not just noisy

**Population level (infinite data, zero sampling noise).** I computed the exact `(1−1/N_train)` quantile for 16 plausible negative-similarity laws and asked what the ζ = 0.25 POT extrapolator returns:

| law | u₇₅ | a | ξ_fit | A_fit | **A*** (needed) | error in q |
|---|---|---|---|---|---|---|
| block mix (class-sd .10, within .05) | 0.275 | 0.067 | −0.256 | 4.49 | **5.74** | −0.084 |
| Beta(5,5)→[−1,1] | 0.216 | 0.172 | −0.386 | 3.50 | **4.04** | −0.093 |
| isotropic cosine d=512 | 0.030 | 0.026 | −0.260 | 4.45 | **5.66** | −0.032 |
| Normal (CLT limit) | 0.674 | 0.597 | −0.258 | 4.47 | **5.84** | −0.819 |
| **90% N(.15,.08) + 10% N(.45,.10) — a hard-negative cluster** | 0.227 | 0.115 | **+0.081** | **13.38** | **5.09** | **+0.953** |

ζ = 0.25 is a bulk quantile, not a tail threshold; the GPD approximation simply does not hold there. The extrapolator closes only **52–62%** of the true batch→gallery depth gap for every realistic law. That residual (≈0.08 cosine on the block model) is the same order as the entire 0.10 "depth bias" §2.1(i) exists to fix.

The last row is the important one. On the *only* law that represents what §2.1(ii) is about — an anchor with a genuinely hard negative cluster — the low-threshold fit reads the mixture step as a heavy tail, returns ξ̂ = +0.081 and A = 13.4 against a true requirement of A* = 5.1, and demands q = 1.77 → saturated at 1. **The estimator inverts the one case the method exists for.** A P×M batch guarantees this structure: I measured only **~12.8 distinct classes** among the 36 exceedances (5th–95th pct: 10–16), so the effective sample size is ~13, not 36, and ξ̂'s SD inflates 1.67×.

**Finite sample.** SD(ξ̂_raw) at k = 36 is 0.20–0.28 depending on law; 16–31% of anchors fall below the −0.5 clip. In simulations where *all anchors share one true ξ*, so 100% of the observed spread is estimation noise:

| law | median ξ̂ | A_SOP | A_batch | **whole depth signal** | **sd(A_i) from noise** |
|---|---|---|---|---|---|
| block cosine mix | −0.302 | 4.078 | 2.854 | **1.224** | **1.056 (86% of signal)** |
| isotropic cosine d=512 | −0.250 | 4.550 | 2.963 | 1.587 | 1.063 (67%) |

Estimator noise alone is 67–86% of the entire batch→SOP extrapolation effect, and injects **±0.069 cosine** of noise into the per-anchor target — 69% of δ_X = 0.10. Saturation of `min(1,q)` is *not* a practical problem for realistic cosine scales (0.15%), and the softplus operates near its elbow (g ≈ 0.31–0.50), so those two concerns are unfounded — I checked and they are fine.

---

## 3. R@1 causality — the depth lever is logarithmic, and the controls cannot isolate it

The failure criterion in §2.1 is correct as far as it goes, and `n_target = N_train` is a *legal* proxy: it is a train-only quantity, fixed a priori, and it matches the actual candidate-negative count to within 0.5% on CUB, 0.0% on Cars, 1.6% on SOP. In-Shop is off by **2.05×** (25,882 vs a 12,612-image gallery) — but because A is logarithmic in R this costs only 0.10 in A units (2.4%) at the realistic ξ ≈ −0.26.

That harmlessness *is* the problem. **The entire extrapolation content is a factor 1.19–2.68 on one coefficient**:

| ξ | A(R=14888) | A(R=36.25) | ratio |
|---|---|---|---|
| −0.35 | 3.724 | 2.759 | 1.35 |
| −0.25 | 4.547 | 2.962 | 1.54 |
| 0.00 | 9.608 | 3.590 | 2.68 |

So **A7 (`n_target = n_i`) is not "no extrapolation" — it is A1 with the mean-excess term rescaled by ~1.43× at the realized ξ**, entirely confoundable with `λ_max`, which is frozen at 1.0 across arms. And **F2 has n = 4 with perfectly collinear covariates** — extrapolation factor is rank-identical to N_train, class count, dataset size and images/class — so a monotone Δ cannot identify depth as the cause.

Separately, §2.1 controls only the max-negative side. R@1 error is `P(max_neg > max_pos)`, and `s⁺` with γ = 8 over |P_i| ≈ 4 is ≈ mean + 4·Var — effectively the *mean* positive, not the max. That is conservative for R@1 (good), but it means the loss is not aligned with the stated criterion; whether a train-identity mixture tail predicts unseen-identity retrieval is untested by any arm, and F5 will pass vacuously (see §7).

---

## 4. The degeneracy proofs: D1 and D5 are wrong; D3's conclusion does not follow; D4 is refuted by descent

**D1 is false.** On the sphere, `∇_{z_i}L = J_i Σ_j (∂L/∂s_ij) f_j` and at full collapse every `f_j = f`, so the sum is a multiple of `f`, which `J_i = (I − f fᵀ)/‖z‖` annihilates exactly. I verified numerically: at the collapsed configuration `|∂L/∂z|_max = 4.7×10⁻⁵` against a loss value of 284 — machine zero. **Collapse is a critical point.** The proposal writes `J_i` in §1.6 and then omits it in D1; a non-zero loss *value* is not a non-zero *gradient*. The same argument voids "L_PA independently forbids it." Also, at exact collapse the strict `s_ij > u_i` rule gives **k_i = 0**, so `a_i = 0/0` and `b_i` divides by `k−1 = −1`. Unhandled NaN.

**The displayed gradients in §1.6 are the ρ = 0 gradients, not the ρ = 1 ones.** Finite-differencing the full pipeline (shrinkage + clip included) at a realistic operating point (u = 0.201, a = 0.059, ξ̃ = −0.212, A = 4.97):

| direction | measured | §1.6 claims |
|---|---|---|
| uniform shift of all 145 | **+1.0000** | +1 ✓ |
| uniform shift of the 36 exceedances | **−7.220** | +A ≈ +4.97 ✗ |
| `∂q/∂u` (threshold element) | **+8.221** | `1−A` = −3.97, "< 0" ✗ |
| smallest exceedance z₍₁₎ | −0.917 | — |
| largest exceedance z₍₃₆₎ | +0.515 | — |

**23 of 36 exceedance ranks receive an *attractive* gradient**, and the per-rank profile sums to −7.22. Both signs match the proposal exactly when I set ρ = 0 (I get +4.968 and −3.968). The cause is structural: `ξ̂ = 2 − a/D` with D the Gini dispersion, so raising the mean excess at fixed dispersion makes the tail look *lighter*, lowers A, and lowers q. `∂L/∂ξ̂ = c·g·a·(∂A/∂ξ) > 0` **unconditionally** — descent always pushes the estimated shape down, with no counterbalancing term.

**D5 is therefore false as stated.** "There is no direction in which corrupting the estimator lowers the objective" — corrupting ξ̂ *downward* lowers A, lowers q, lowers the loss, and the loss supplies that direction itself. The ε-ridge and the clip are protective; the sign argument is not.

**D3's inequality is true, its conclusion is not.** `q ≥ mean_E(s) ≥ u` holds (equality iff a = 0), so `L_XTail ≥ L_quantile-margin` — a valid bound. But descending on an upper bound does not descend on the bounded quantity. Direct descent on the 145 cosines under the specified loss (400 steps, ρ = 1):

| | start | end | Δ |
|---|---|---|---|
| max negative cosine | 0.4457 | 0.3523 | **−0.093** ✓ |
| **u (75th pct of negatives)** | 0.2691 | 0.3050 | **+0.036** ✗ |
| a (mean excess) | 0.0563 | 0.0082 | −0.048 |
| ξ_raw | −0.104 | **−0.537** | → lower clip |
| A | 5.03 | 3.36 | → toward min |

**D4 is refuted.** `a → 0` is reached by *raising the 75th percentile*, not by lowering the tail — the loss believes it improved the deep quantile by 0.219 while the only verifiable quantity, the in-batch max, improved by 0.093 and the 75th percentile got worse. And ρ = 0 (arm A6) produces nearly the same geometry (max −0.059, u +0.030), so the "novel" shape path buys little. Note where ξ_raw ends: **at the lower clip, where the shape path has zero gradient and XTail is literally a constant-A loss.**

---

## 5. Distinguishability from a tuned quantile/CVaR margin — it isn't, and I can show it without a GPU

Algebraically, `q_i = A·mean_E(s) − (A−1)·u = A·CVaR₀.₇₅ − (A−1)·VaR₀.₇₅`. With A constant this is exactly a linear combination in the VaR/CVaR family — the [average top-k loss / CVaR](https://papers.nips.cc/paper/6653-learning-with-average-top-k-loss) family. And `a − 2b = ½·(Gini mean difference)`, so `ξ̂ = 2 − 2·mean/GMD`: the whole "EVT" content is a smooth function of exactly three linear order-statistic functionals (u, a, GMD).

The decisive test. Across the 16 laws above, the constant A* that *exactly* hits the true deep quantile has **mean 5.39, sd 0.73, range [3.22, 5.88]** — comfortably inside A4's grid {2,4,6,8,10,14}. The fitted A has mean 4.85 but **sd 2.24, range [2.98, 13.38]**, and undershoots A* in **15 of 16** laws.

With genuinely heterogeneous anchors (three block laws with different true deep quantiles), 30,000 anchors, k = 36:

| predictor | RMSE | corr with truth |
|---|---|---|
| XTail (fitted ξ) | 0.1575 | +0.790 |
| **tuned constant A = 5.68** | **0.0918** | **+0.882** |
| u alone | — | +0.747 |
| a alone | — | +0.844 |
| **ξ̂ alone** | — | **+0.009** |

**Incremental R² of ξ̂ over a linear fit in (u, a): 0.0087.** (u, a) explain 80.6%. Even in the model-correct best case (parent *is* GPD, wide true-ξ spread U[−0.5, 0.1]) XTail's RMSE beats the tuned constant by only 12% and reliability of ξ̂ is 0.371; at realistic narrow spread the constant **wins** (RMSE 0.052 vs 0.071) and reliability drops to **0.063**.

A4 and A7 are necessary. They are not sufficient — neither isolates the added positive-pull channel (no arm sets `q_i` to a constant), and A7 is confounded with a 1.43× loss rescaling. But sufficiency is moot: **A4 is predicted to match A1**, which fires F3 by the proposal's own preregistration.

---

## 6. Prior art — the flagged risk resolves *for* the proposal; a different one resolves against it

**AnchorFace, resolved.** I extracted the [AAAI PDF](https://cdn.aaai.org/ojs/20063/20063-13-24076-1-2-20220628.pdf) that was unreadable in the proposal's session. It does **not** fit a parametric tail. It maintains "an online-updating set S ∈ R^{N×K×d}" inspired by MoCo, notes that "if the numbers of positive pairs and negative pairs are insufficient, the threshold estimation is not robust," and obtains "the corresponding Anchor Threshold t_A under the Anchor FAR" as an empirical quantile — "a scalar value," global, not per-anchor. **Distinctions (a), (b) and (c) all survive intact.** The proposal's self-flagged highest-risk claim is vindicated.

**LDReg (ICLR 2024) is not.** [arXiv:2401.10474](https://arxiv.org/pdf/2401.10474) uses the Method-of-Moments LID estimator from [Amsaleg et al., *Extreme-value-theoretic estimation of local intrinsic dimensionality*](https://link.springer.com/article/10.1007/s10618-018-0578-6): `LID* = −µ_k/(µ_k − w_k)`, with µ_k the mean of the k nearest-neighbour distances and w_k the k-th. Its pseudocode is structurally XTail's block:

```
r = torch.cdist(data, reference, p=2)   # in-batch pairwise matrix
a, idx = torch.sort(r, dim=1)           # per-row sort
m = torch.mean(a[:, 1:k], dim=1)        # mean of k order statistics
lids = m / (a[:, k] - m)                # EVT tail index as a mean/dispersion ratio
total_loss = loss + beta * lid_reg      # differentiated into the representation
```

k = 64/128, in-batch, chosen "for incorporation within a gradient descent framework." So §3's framing — that items 9–11 are all "post-hoc test-time calibration over a frozen embedding" while XTail uniquely "differentiates through the EVT fit and uses it to shape the embedding at train time" — **is false**. Per-sample EVT tail-index estimation from in-batch order statistics, differentiated through to shape a learned embedding, is published.

Also uncited and relevant: [Furon & Jégou, *Using extreme value theory for image detection*](https://inria.hal.science/hal-00789804/preview/RR-8244.pdf) (INRIA RR-8244) and [*Relevance prediction in similarity-search systems using EVT*](https://www.sciencedirect.com/science/article/abs/pii/S1047320319300720) fit EVT to the upper tail of non-matching similarity scores to extrapolate deep retrieval probabilities — the identical statistical object, post-hoc. I could not fetch the RR-8244 body (host blocked), so I state this from the abstract-level description with corresponding uncertainty.

Residual novelty after this: extrapolating the fit to a *deep quantile used as a margin target*. That is real, and narrower than §3 claims.

---

## 7. Recipe and controls

- **Sampler infeasible on the primary dataset.** SOP has 59,551 train images over 11,318 classes (mean 5.26, min 2). M = 5 forces duplicates for a large share of classes — standard `MPerClassSampler` behaviour is "if a class has less than m samples, then there will be duplicates." Duplicates give `s⁺ = 1.0` exactly, hence `m_i = q + 0.1 − 1 ≤ 0`, hence `g_i ≈ 0.002`: **XTail switches itself off for exactly those anchors, on SOP and In-Shop.** Not addressed anywhere.
- **λ_max = 1.0 does not transfer.** L_PA's negative term is averaged over |P| = C proxies — 100 on CUB, 11,318 on SOP — so the L_PA : L_XTail gradient ratio differs by ~2 orders of magnitude across datasets. Worse, holding out the last 20% of train class IDs changes C (100→80, 11,318→9,054), so λ_max tuned on the pseudo-unseen split is calibrated against a different L_PA normalization than the arm it is used in. §6 correctly says scale is operational; the recipe does not honour that.
- **EMA autograd is under-specified with a silent-failure mode.** "Gradient flows only through the current 0.1 term" requires `ξ̄_new = 0.9·ξ̄.detach() + 0.1·ξ̂_pool`, used live and stored detached. A naive in-place buffer either raises *backward through the graph a second time* or silently detaches — in which case the (1−λ_i) ≈ 0.53 share of the shape path is dead and A5/A6 become partially degenerate without anyone noticing.
- **Pooled standardization is biased.** Dividing each anchor's exceedances by its own `a_i` forces the pooled mean to exactly 1 and shrinks between-block dispersion, biasing ξ̂_pool low. Modest, but it is not the estimator the shrinkage formula assumes.
- **"Empirical-Bayes shrinkage" is a misnomer.** `λ_i = k/(k+τ)` uses no variance-component estimate. Measured reliability at k = 36 is 0.06–0.37; λ = 0.474 **under-shrinks by roughly 2–8×**.
- **Ties/k-degeneracy.** With 120 tied values + 25 spread, k_i = 24, not 36; k_min = 12 only disables shrinkage while a_i, b_i still feed q_i, and b_i divides by k−1 (undefined at k = 1).
- **AMP.** "AMP off for the PWM block" must include the Gram itself; casting only the PWM ops leaves fp16 cosines (ulp ≈ 5×10⁻⁴ at 0.5) feeding a cancellation-heavy `a−2b`. Small vs the 0.28 sampling SD, but it must be stated as `F.float()`.
- **F5 will pass vacuously.** ξ̂ is dominated by the ζ = 0.25 threshold choice and shrinkage toward a shared ξ̄, not by class identity; |ξ̂_seen − ξ̂_unseen| > 0.15 will essentially never trigger, so F5 cannot falsify the transfer assumption it names.
- **F3/F4 are unmeasurable.** With 5 seeds and sd ≈ 0.0035, the 95% CI half-width on an arm difference is **±0.0043** and the minimum detectable difference is **0.0062**. Both falsifiers use a 0.003 equivalence bound — inside the noise. ~21 seeds/arm would be needed. F1 (+0.008 floor) is adequately powered at n ≈ 3.
- **Clean:** no test-conditioned leak (n_target is train-only and fixed a priori); F6's sub-gallery sweep on a class-disjoint train split is leak-free and is the single best diagnostic in the document; test touched once per arm; A0 as an own-reproduction baseline.

---

## 8. Cost and forecast arithmetic

The cost model is **correct and defensible**. Gram = 150²×512 = 11.5 MMAC vs ResNet-50 fwd+bwd on 150×224² ≈ 1.85 TFLOP → 6.2×10⁻⁶. The Gram is genuinely extra (Proxy-Anchor is proxy-based and needs no sample–sample Gram) and is counted as such. Retained autograd state is ~500 KB, not 90 KB — the sort indices (150×145 int64 = 174 KB), the sorted values, and the pooled block are omitted — but the absolute number is irrelevant. No new parameters ⟹ no optimizer-state growth. ≤1.02× wall clock is plausible if vectorized. Deployment is genuinely bit-identical.

Forecast arithmetic is internally consistent (I re-derived every row). The problems are the premises. It requires multi-proxy alone to deliver **+0.032 on Cars** and **+0.026 on CUB**, and then XTail to be **89–100% additive** on top. And the headline Δ of **+0.020 R@1 on SOP** from a near-free auxiliary term sits at the very top of the historical range — while the proposal itself cites [Musgrave et al.](https://arxiv.org/pdf/1911.12528), whose finding is that matched-condition DML deltas collapse to ~1 point or less.

On the standing objective: **no.** By the proposal's own table, CUB is a declared miss (−0.007), Cars a tie (−0.001), SOP a +0.001 crossing against PFML 0.829 ± 0.002 over 5 runs (I confirmed the reference: combined SE ≈ 0.002, z ≈ 0.5), and In-Shop +0.004 against an uncertainty-free number. A +0.001 tie and an unadjudicable +0.004 cannot satisfy a frontier objective, and the document says so.

---

## 9. Verdict

**DEAD.**

**Earliest failure:** §1.4(c)–(e) — the shape estimator and its extrapolator at ζ = 0.25, k ≈ 36. Not the gradients, not the controls, not the forecast.

**Single most decisive mechanism-level reason:** at the specified threshold and sample size, the per-anchor shape estimate carries **no information about the quantity it is supposed to control** — corr(ξ̂, true deep quantile) = **+0.009**, incremental R² over (u, a) = **0.0087** — while a single tuned constant A ∈ [3.2, 5.9] predicts that quantile *better* than the fit on every plausible similarity law (RMSE 0.092 vs 0.158). `A(ξ̂_i)` is therefore a noisy, biased reparameterization of a constant, and XTail is arm A4 plus noise. Its own preregistered falsifier F3 is predicted to fire — and cannot be measured at 5 seeds. The second, independent kill is that the mechanism runs backwards: `∂L/∂ξ̂ > 0` unconditionally, so descent drives ξ̂ to the lower clip (where XTail *becomes* constant-A) along a path that **raises the 75th percentile of negative similarity by +0.036**, with 23 of 36 exceedances receiving attractive gradients that §1.6 does not display.

**Not BLOCKED.** No no-GPU evidence resolves this, because the evidence is already no-GPU: the estimator's population-level bias and its ~1% incremental information content are properties of ζ = 0.25 and k = 36, not of any network.

**Valid subcomponents to preserve, separately from the verdict:**

1. The Hosking–Wallis PWM derivation, the order-statistic weights, `ξ̂ = 2 − a/(a−2b)`, and the identity `(1−ξ̂)a ≡ 2ab/(a−2b)` — all exactly correct and independently verified.
2. `A(R,ξ)`, its ξ→0 limit, the Taylor branch, and the 10⁻³ switch point — correct and well-calibrated.
3. All dataset counts and the 41/56/87/417 extrapolation factors.
4. The cost model and the bit-identical-deployment argument.
5. D2 (exact O(512) and scale invariance).
6. **F6** — sub-gallery-size R@1 sweep on a class-disjoint train split — a genuinely leak-free depth diagnostic worth keeping independently of XTail.
7. The AnchorFace distinction, which I confirmed from the primary PDF.
8. The methodological discipline: own-reproduction baseline, class-disjoint tuning, one test touch per arm, and forecasts that predict their own misses.

If any variant is revisited, the two things that would have to change first are the threshold (ζ = 0.25 is a bulk quantile, and a proper POT threshold leaves k ≈ 3–8 per anchor, which is worse) and the supervision object (a tuned VaR/CVaR combination already dominates the fit at the population level, and is what A4 tests).

**Sources:** [Proxy-Anchor (CVPR 2020)](https://ar5iv.labs.arxiv.org/html/2003.13911) · [AnchorFace (AAAI 2022) PDF](https://cdn.aaai.org/ojs/20063/20063-13-24076-1-2-20220628.pdf) · [LDReg (ICLR 2024)](https://arxiv.org/pdf/2401.10474) · [Amsaleg et al., EVT estimation of LID](https://link.springer.com/article/10.1007/s10618-018-0578-6) · [Furon & Jégou, EVT for image detection (RR-8244)](https://inria.hal.science/hal-00789804/preview/RR-8244.pdf) · [Relevance prediction in similarity search using EVT](https://www.sciencedirect.com/science/article/abs/pii/S1047320319300720) · [Learning with Average Top-k Loss (NeurIPS 2017)](https://papers.nips.cc/paper/6653-learning-with-average-top-k-loss) · [PFML](https://arxiv.org/html/2405.18560v2) · [SOP dataset stats](https://www.tensorflow.org/datasets/catalog/stanford_online_products) · [MPerClassSampler](https://kevinmusgrave.github.io/pytorch-metric-learning/samplers/) · [Unbiased Evaluation of DML](https://arxiv.org/pdf/1911.12528) · Hosking & Wallis, *Technometrics* 29(3):339–349, 1987 — [JSTOR](https://www.jstor.org/stable/1269343) (cited as the standard reference; I verified the estimator by re-derivation and Monte Carlo rather than by fetching the paper).

