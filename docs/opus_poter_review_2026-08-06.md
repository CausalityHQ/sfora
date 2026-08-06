## Verdict: **DEAD**

**Earliest failed section:** §1.4(c) — *"Scale is per-query with empirical-Bayes shrinkage, **gradient kept**"*. The failure surfaces as three false claims in §1.5 and D2, but it originates in that one design decision.

**Single most decisive mechanism-level reason:** For the Hosking–Wallis PWM scale estimator the proposal uses, the *largest* exceedance carries PWM weight exactly **zero** in â₁. Therefore

$$\frac{\partial \hat\sigma}{\partial m_{(1)}} \;=\; \frac{1}{k}\frac{\partial\hat\sigma}{\partial \hat a_0} \;=\; -\,\frac{4\hat a_1^{\,2}}{k\,(\hat a_0-2\hat a_1)^2}\;<\;0 \quad\text{identically}$$

Combined with ∂ℓ/∂σ̃ > 0 — which the proposal itself asserts and which holds whenever any quadrature point has s⁺ − Ê_r − u > 0 — this gives **dℓ/dm_(1) < 0 in every configuration where POTER has a live gradient at all**. Gradient descent therefore *raises* the query's similarity to its single hardest negative **class**: the exact score that decides R@1. POTER's flagship "dimension-expansion pressure" (D2) is, at the top of the tail, a hardest-negative-class **attraction** term.

Verified FD-vs-closed-form to 1.1e-7 relative over 400 random configurations, k ∈ {4,5,8,16,64}, all shapes; strictly negative in 400/400:

```
d sigma_hat / d m_(5)  = +1.5945  -> loss wants this score DOWN
d sigma_hat / d m_(4)  = +1.1288  -> DOWN
d sigma_hat / d m_(3)  = +0.6632  -> DOWN
d sigma_hat / d m_(2)  = +0.1975  -> DOWN
d sigma_hat / d m_(1)  = -0.2681  -> UP        <-- the top-1 competitor
```

It is not a spread compressor. It is a spread *inverter*: it pushes the maximum up and everything beneath it down. §1.5's second bullet ("compresses the spread of the top-k class scores") is false at the only coordinate that matters.

---

## 1. PWM/GPD estimator and Λ

**Correct.** a₀ = σ/(1−ξ), a₁ = σ/(2(2−ξ)) ⇒ ξ̂ = 2 − a₀/(a₀−2a₁), σ̂ = 2a₀a₁/(a₀−2a₁) = a₀(1−ξ̂). Recovered exactly for ξ ∈ {0, ±0.3, ±0.5, 0.2}. The paper's own spot-checks (ξ=0 ⇒ â₁=σ/4; ξ=0.5 ⇒ â₁=σ/3) are right. Sample weights (k−j)/(k−1) match [Hosking & Wallis (1987)](https://www.jstor.org/stable/1269343). Finite-sample bias at k=5 is severe but unclaimed (mean ξ̂ = −0.235 when ξ=0; mean σ̂ = 1.24σ) — CUB/Cars run at k=5.

**Support handling.** ξ<0: base clamps to 0, 0^{−1/ξ} = 0 — correct value, but an *exactly* dead gradient branch. ξ>0 with y < −σ̃/ξ: base clamps to 0 and **0^{−1/ξ} = +inf**; `min(1,inf)` returns the constant branch, but autograd multiplies 0 by pow's inf local derivative → NaN backward. §1.4 specifies no numerically safe ξ→0 branch, and under the specified AMP fp16 this term's gradient is O(1/σ̃²) ≈ 10⁶ near σ_min = 1e-3 while L_PA's is O(1) — one GradScaler scalar cannot serve both.

**Gradient signs through the live estimator — the audit the brief asked for.** §1.5 omits that u enters σ̂ as well as the threshold. The exact identity is ∂σ̂/∂u = −(â₀+2â₁)/(â₀−2â₁) = −(3−2ξ) ∈ [−4, −2.6] over the clipped shape range. Total derivative:

$$\frac{d\ell}{du} \;=\; \frac{\Lambda}{(1+\Lambda)\tilde\sigma}\Big[\,1 - c_\gamma(3-2\xi)\,\bar r\,\Big],\qquad c_\gamma = \gamma + \gamma/B$$

Scale-free: the sign flips once r̄ ≳ 0.5–0.76, and r̄ ≈ ln(A/Λ) with A = θC_gal ζ_u ≈ k. **The published operating points force the flip**: Λ* = −ln R@1 ∈ [0.073, 0.309] against A ∈ {5.05, 64.0} ⇒ r̄ ∈ [2.8, 6.5]. No embedding-geometry assumption is needed.

Batch-mean finite differences (B=120, λ/B applied), against PA's *measured* negative-term gradient on the same coordinate:

| | k | A | dℓ/du | PA on u (measured) | ratio | dℓ/dm_(1) |
|---|---|---|---|---|---|---|
| CUB | 5 | 5.05 | **−0.102** | +0.0078 | 13× | **−0.0277** |
| Cars | 5 | 5.05 | **−0.056** | ~0.008 | ~7× | **−0.0145** |
| In-Shop | 64 | 64.0 | **−0.125** | ≤0.0080 (bound) | ≥16× | **−0.00099** |
| SOP | 64 | 64.0 | **−0.207** | +0.000218 | **951×** | **−0.00189** |

Fine scan over the whole trajectory, 0 ≤ s⁺ ≤ 0.70: **max positive dℓ/dm_(1) over every live grid point on all four datasets = 0.0000.** dℓ/du < 0 on 96/114 (CUB), 86/108 (Cars), 141/158 (In-Shop), 138/152 (SOP) live points; the positive-sign region is the saturated `min[1,·]` plateau where the gradient is ~0 anyway. On SOP the net gradient on the rank-65 negative proxy is *positive* by ~800×: this is a sign inversion of the total training signal, not an offset.

## 2. e_ic nonnegativity, the pooled law, and Q_{K−1}

**e_ic ≥ 0 is false.** Exact 2-D counterexample with the *most favourable* proxy (normalized class mean): z_a,z_b at ±20°, query at the mean ⇒ e = −0.0603. In d=512 with the proxy at the exact class centre, **991/4000 (25%) of e_ic are negative**. PA proxies are free parameters, so this is the best case, not the typical one.

**K→n̄ block-max transform: algebraically correct.** H_n̄ = H_K^{n̄/K} ⇒ Ê_r = Q_K(p^{K/n̄}), verified against direct n̄-block-max sampling to ≤0.08 sd-units. Δ⁺ = Q_{K−1}(0.5^{(K−1)/n̄}) − Q_{K−1}(0.5) has the right sign and direction.

**Q_{K−1} is not defined by the algorithm.** §1.4(d) constructs only Q_K. Q_{K−1}(p) = Q_K(p^{K/(K−1)}) is derivable but unstated. Worse: Q_K is the law of **negative-class** excess seen from an outside query; Δ⁺ needs the law of **same-class** similarity seen from inside. "(same law, used once)" is a category error, not a shorthand — the two are different distributions with different max-growth, and this one is not flagged anywhere.

**Pooling bias.** (E_mix[H^K])^{n̄/K} ≠ E_mix[H^n̄] — measured error up to 1.3 sd-units at r=32 for n̄=58.64. Conservative by Jensen, but a bias in the object the whole claim rests on.

## 3. The hierarchy's probability model — **second decisive failure**

The independence assumption m_c ⊥ E_c does not merely fail empirically; it fails by **spherical geometry**, and it is decidable by simulation, not GPU:

| variant | corr(m,E) | E[E] pooled | E[E \| top 5%] | shift / σ̃ |
|---|---|---|---|---|
| proxy = class centre | **−0.719** | +0.0739 | −0.0616 | 6.8× |
| proxy = L2-normalised class mean | **−0.643** | +0.0727 | +0.0022 | 3.5× |
| proxy jittered 0.15 off centre | **−0.752** | +0.0715 | −0.0834 | 7.7× |
| proxy jittered 0.40 off centre | **−0.776** | +0.0730 | −0.0966 | 8.5× |
| heterogeneous class spread | **−0.701** | +0.0697 | −0.0623 | 6.6× |
| queries drawn near the classes | **−0.727** | +0.0164 | −0.1245 | 7.0× |

The mechanism is unavoidable: if z_i is close to μ_c (high m), every z_j = normalize(μ_c + noise) moves *away* from z_i, so E < 0 exactly on the classes that dominate Λ. POTER estimates E on all 39 in-batch classes but applies it to the top-k **selected** classes. Since Λ ∝ e^{E/σ̃}, a shift of 3.5–8.5 σ̃ misestimates Λ by **30× to 5000×** — the same size as, or larger than, the 8×–1543× quantile gap that §2.1 says is the entire reason the method exists. **The proposal's own F6 (|corr| > 0.2 ⇒ misspecified) fires at 3–4× threshold before any GPU run.** C10 also measures only the unconditional correlation, not the selection-conditioned shift, which is the operationally relevant quantity.

Other model defects: heterogeneous class sizes (using n̄ rather than E over n_c inflates E[max] by 0.28 sd-units on SOP-like, 0.24 on In-Shop-like; CUB is fine at 0.006); θ=1 with correlated/duplicate product images is conservative as claimed. The R=64 quadrature is a legitimate midpoint-quantile rule for E_E[·], but on CUB/Cars it reads only the top 22%/16% of the pooled e-distribution, so Ê is carried by a handful of extreme order statistics.

**Deployment constants (primary-source check).** CUB/Cars/SOP train-vs-test-gallery n̄ agree to 1–2%. **In-Shop does not.** The query/gallery protocol splits the test images: gallery = 12,612 images over 3,985 classes ⇒ **n̄ = 3.165**, against POTER's 6.475 — a **2.05×** overstatement of the block size, on the one dataset where a frontier crossing is claimed. ([DeepFashion In-Shop](https://liuziwei7.github.io/projects/DeepFashion.html))

## 4. Shortcut attacks

- **Gradients outside span(P): NO for the σ̃ path.** σ̂_i is a function of {⟨z_i,p_c⟩} only ⇒ ∇_{z_i}σ̂_i = Σ_c (∂σ̂/∂m_ic)p_c ∈ span(P) exactly. Verified numerically: orthogonal component 2.1e-6 relative. **D2's central claim — "POTER supplies the missing gradient there" — is false for the mechanism it names.** POTER does reach outside span(P), but only via s⁺ and e_ic, which are not the σ̃/D2 mechanism and are not what §3 item 10 differentiates from ρ-spectrum/Anti-Collapse.
- **Exact collapse is not handled.** â₀ = â₀−2â₁ = 0 ⇒ ξ̂ = 0/0 → NaN (clamp(NaN) = NaN, gradients NaN); §2.2 D1's "σ̂→σ_min" is wrong, it is 0/0. Near collapse, σ̃ is pinned at σ_min (clip kills the σ̃ path) *and* min[1,·] is active (kills s⁺ and u) ⇒ **POTER's gradient is exactly zero at POTER's maximum value log(1+k)**. It is a zero-gradient plateau for the POTER term; only L_PA escapes. Confirmed in the descent experiment: from a PA-warm-started point with Λ = 4.66, 600 steps of PA+POTER moved nothing.
- **Gaming σ̃: yes, and it is the dominant path.** The σ̃ term's gradient is ~2.8–6.5× the s⁺ term's at any live operating point, and its cheapest routes are raising u (∂σ̂/∂u ≈ −3) and raising m_(1) (∂σ̂/∂m_(1) < 0) — both anti-retrieval. D2's claimed lower bound σ̃ ≳ c√(log(C/k)/d_eff) also has the wrong C-dependence: for Gaussian-tailed scores E[m_(1)−m_(k+1)] ≈ s[√(2 log C) − √(2 log(C/k))], which *decreases* in C.
- **ζ_u: not gameable** (D6 correct). **ξ̃/θ: not gameable** (D4 correct). **s⁺ soft-max weakens collapse pressure** (D5's premise correct).
- **min/clamp:** kills gradient on the hardest queries (the saturated branch) and NaNs for ξ>0 deep in the negative-y branch.

## 5. Novelty

Moot given the mechanism failure, but for the record: with σ̃ detached — the only version whose gradients point the right way — §1.4(f) collapses (its own ξ→0 form) to

`ℓ = log(1 + const · e^{−(s⁺ − u)/σ̃_detached})`

a temperature-scaled margin between the soft-max positive and the (k+1)-th hardest **proxy**, plus an additive constant. That is precisely control **C2** (global temperature) composed with **C4** (tuned constant offset) — the two the proposal itself names as "the dangerous one," whose success is pre-registered falsification **F4**. So the object is a more elaborate estimator wrapped around a proxy-space top-k margin, not a new supervision object. I found no train-time EVT/GPD loss for DML in the search; the distinction from [Meta-Recognition](https://www.wjscheirer.com/projects/meta-recognition/)/[OpenMax](https://arxiv.org/pdf/1511.06233)/EVM (test-time) and from [XBM](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_Cross-Batch_Memory_for_Embedding_Learning_CVPR_2020_paper.pdf) (observation vs extrapolation) is fairly drawn. I could not verify EVEREST (arXiv 2601.19022).

## 6. Recipe, feasibility, cost

- **B=120 is correct** for ResNet-50/512-D in the [official PA repo](https://github.com/tjddus9597/Proxy-Anchor-CVPR2020) (180 is the Inception-BN setting). §1.2's recipe fidelity holds.
- **K=3 is infeasible on SOP and In-Shop.** SOP products have 2–12 images ([Song et al., CVPR 2016](https://arxiv.org/abs/1511.06452)); roughly 40% of classes have fewer than 3. Sampling with replacement makes e_ic a max over a multiset with duplicates, so H_K ≠ H_1^K and the K→n̄ transform is applied to a law it does not describe. §1.3 asserts "K ≥ 3 is required" and never addresses that two of four datasets cannot supply it.
- **Test-set contact.** §1.6 promises "a single retrain … and one test evaluation," but C1–C10 + C6's batch ladder + C7's 3-point gallery ladder + the λ sweep + the θ ablation is ~29 arms × 4 datasets × 5 seeds ≈ **550–600 200-epoch runs, order 4,000 A100-hours**. No total cost is stated anywhere in §6.
- Power is adequate for the +1.6 headline (paired, n=5) but not for the ±0.2–0.3 "parity" claims.
- Cost claims (≤1.03× step, ≤1.01× memory) check out arithmetically.

## 7. The forecast, taken literally

Section 5's arithmetic is internally consistent — I reproduced every nat, every κ·nats, every Λ*, every q*, and the 0.58–0.62 additivity ratios. Taken literally:

- Standalone PA+POTER: **0 of 4 crossings**. It is −1.8 / −2.8 / −0.3 / +0.0 against the stated references.
- The single crossing (In-Shop, +1.1) requires (i) DADA, a method POTER does not own; (ii) an explicitly assumed 60% additivity; (iii) comparison against a point estimate with unreported uncertainty; and (iv) a deployment constant (n̄ = 6.475) that is **2.05× the true In-Shop gallery block size**, which is the very quantity the crossing's mechanism depends on.
- On SOP — where the mechanism predicts its *largest* effect — the forecast is **−0.6**.

So even if every forecast landed exactly, the frozen method delivers one assumption-contingent, composed +1.1 at ~2.2σ of its own seed spread against an uncertainty-free reference, and nothing standalone. That does not satisfy a frontier-crossing objective.

---

## Preserved as genuinely correct

These are independent of the verdict and worth keeping:

1. **The PWM/GPD estimator itself** — exact, differentiable, correctly transcribed from Hosking–Wallis, with correct closed-form verification points.
2. **ζ_u = k/(C−1) as an order-statistic threshold** — exact, constant, ungameable. D6 is sound.
3. **The K→n̄ block-max quantile transform** Q_K(p^{K/n̄}) — algebraically correct and empirically validated; likewise the Δ⁺ direction.
4. **The R=64 midpoint-quantile quadrature** as an estimator of E_E[·] — legitimate.
5. **The estimand framing in §2.1** — that R@1 is a two-level extreme and that Λ* = −ln R@1, q* = Λ*/(C_gal−1) gives an 8×–1543× class-level resolution gap, is correct arithmetic and a real observation.
6. **∂L_PA/∂z ∈ span(P)** and the rank-≤C limitation — true, and a real pathology.
7. **D4, D5, D6 blocks** and the θ=1 conservatism — correctly reasoned.
8. **§1.2 recipe fidelity and the cost analysis** — verified.

## Stated uncertainty

- I confirmed PFML exists at the claimed venue ([CVPR 2025, pp. 25549–25559](https://openaccess.thecvf.com/content/CVPR2025/html/Bhatnagar_Potential_Field_Based_Deep_Metric_Learning_CVPR_2025_paper.html), [arXiv 2405.18560](https://arxiv.org/abs/2405.18560)) but did **not** independently verify its .734/.927/.829 R@1 values. §7.4 already flags this.
- I did not verify DADA's PA rows or EVEREST (arXiv 2601.19022).
- ∂ℓ/∂σ̃ > 0 is the proposal's own assertion; I verified it holds whenever any quadrature point has y_r > 0, which is every live regime.
- The synthetic m-distribution (Gaussian, σ=0.045) supplies *magnitudes* only. The decisive results — the two PWM sensitivity identities and the scale-free flip condition r̄ ≳ 0.66 forced by the published R@1 values — do not depend on it.

Scratch code is under `/tmp/poter_audit/` (`audit.py` … `audit6.py`); no repository file was read beyond the frozen proposal, and none was modified.

**Sources:** [Proxy Anchor Loss, CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/papers/Kim_Proxy_Anchor_Loss_for_Deep_Metric_Learning_CVPR_2020_paper.pdf) · [official repo](https://github.com/tjddus9597/Proxy-Anchor-CVPR2020) · [Hosking & Wallis, PWM for the GPD, Technometrics 1987](https://www.jstor.org/stable/1269343) · [Stanford Online Products, CVPR 2016](https://arxiv.org/abs/1511.06452) · [DeepFashion In-Shop](https://liuziwei7.github.io/projects/DeepFashion.html) · [PFML, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Bhatnagar_Potential_Field_Based_Deep_Metric_Learning_CVPR_2025_paper.html) · [Cross-Batch Memory, CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_Cross-Batch_Memory_for_Embedding_Learning_CVPR_2020_paper.pdf) · [OpenMax](https://arxiv.org/pdf/1511.06233) · [Meta-Recognition](https://www.wjscheirer.com/projects/meta-recognition/)
