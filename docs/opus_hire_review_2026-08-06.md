## VERDICT: **DEAD**

**Earliest failed section:** §1.4, estimator **(T1)** — the cross-class U-statistic.

**Most decisive mechanism-level reason:** `tr(C_c C_c′)` is a measure of *residual-subspace overlap between two different identities*, not of isotropy. `L_shape = log(dU + 1e-8)` is monotone increasing in `U` with **no term anywhere in `L` pushing `U` up**, so its actual minimiser is *mutually orthogonal, rank-(K−1) per-class residual subspaces* — i.e. maximal per-identity anisotropy **and** maximal heteroscedasticity of shape, the two configurations §2.1 identifies as what destroys retrieval. The advertised target (`L_shape = 0`) is an interior point the loss descends *away from*. Settled by CPU arithmetic and a 4,000-step CPU descent; no GPU run is needed or admissible to repair it.

---

## 1. What U actually estimates

Two independent defects, both verified numerically (`/tmp/adv/check1_ustat.py`).

**(a) The identification `E[δδᵀ] = Σ_c/tr Σ_c` is false.** δ is a *direction*; direction-normalisation compresses anisotropy nonlinearly. For `2Σ = diag(9,1,0.5,0.5)`, 4M samples:

| | e₁ | e₂ | e₃ | e₄ |
|---|---|---|---|---|
| `diag E[δδᵀ]` (true) | 0.6205 | 0.1720 | 0.1037 | 0.1038 |
| `diag Σ/tr Σ` (claimed) | 0.8182 | 0.0909 | 0.0455 | 0.0455 |

Max discrepancy **0.198**. Consequently `r_eff := 1/‖C̄‖²_F` is the participation ratio of the *directional* second moment, not "the effective rank of the conditional residual covariance" (§1.4). It systematically **overstates** isotropy — which matters because C7/F4 are defined on it.

**(b) Isotropy of the average does not imply per-identity isotropy, and the estimator is blind to the difference.** At P=30, K=5, d=512:

| configuration | d·U | `L_shape` |
|---|---|---|
| per-class **isotropic full-rank** (advertised target) | 0.9736 | −0.027 |
| per-class **rank-4, random subspaces** (κ = ∞) | 0.9725 | −0.028 |
| per-class **rank-4, mutually orthogonal** | **0.0210** | **−3.862** |

The maximally anisotropic configuration is *indistinguishable* from the target (row 2 vs row 1: Δ = 0.001), and the orthogonalised version is **strictly preferred by 3.84 nats**.

**(c) Why the unbiasedness proof fails.** For a fixed finite class set,

```
J = (1/C(C−1)) Σ_{c≠c′} tr(C_c C_c′) = C/(C−1)·[ ‖C̄‖²_F − (1/C)·mean_c ‖C_c‖²_F ]
```

`E[U] = ‖C̄‖²_F` requires `E[C_c C_c′] = E[C_c]E[C_c′]`, i.e. class shapes drawn *independently*. A trained encoder makes them a deterministic, jointly optimised family — that factorisation is precisely what SGD destroys. Minimising `J` explicitly **rewards increasing `mean_c ‖C_c‖²_F`**, i.e. per-class anisotropy. Feasibility of the exact-zero solution: CUB `C(K−1) = 100×4 = 400 ≤ 512`; Cars `98×4 = 392 ≤ 512`. Both frontier datasets admit `U ≡ 0` exactly.

**(d) The ten overlapping pairs.** Expectation is unaffected (cross-class products are independent given θ), so the cross-class construction does what it claims *for the mean*. It does not remove the variance inflation from the ten index-sharing pairs, and — decisively — the per-batch `Ĉ_c` from 5 points has rank ≤ 4 out of 512 regardless.

**(e) Dynamical confirmation** (`check5_descent.py`, exact pseudocode, embeddings optimised directly, Adam 3e-3, d=64, P=K as specified, ±`L_PA`):

```
step      U*d   L_shape   r_eff(Σ_c)/64   κ(Σ_c|supp)   tr(C_c C_c′)   [target 1/d=0.0156]
   0   1.0043   +0.0043       3.77           1.86          0.01583
4000   0.0000  -11.5427       3.46           3.14          0.00000     (identical with L_PA)
```

Starting from a configuration HIRE's own certificate scores as isotropic (d·U = 1.004), descent drives `L_shape` to −11.5 while `r_eff(Σ_c)` *falls* and cross-class shape overlap collapses to 0. `L_homo → 0.0003`, `L_scale → 0.03`: no other term objects. Adding `L_PA` changes nothing except that κ rises faster (1.86 → 4.45). The gradient direction `−∇U` is architecture-independent; a ResNet-50 need not reach exact orthogonality, but there is no opposing force.

## 2. The derivative

`sg` is on the norm, so the live Jacobian is `∂δ/∂d = I/s`, **verified exactly** by `torch.autograd.functional.jacobian`. The proposal's stated `(I − δδᵀ)/ρ` is the derivative of *unstopped* normalisation. The claim "shape and scale gradients are orthogonal by construction" is therefore false:

- radial component of `∂⟨δ_a,δ_b⟩²/∂d_a` = **+0.025418**, matching the analytic `2⟨δ_a,δ_b⟩²/ρ_a` = +0.025418 exactly;
- without `sg`, the same component is **−6.9e−18** (machine zero), i.e. the quoted formula's property holds only for the formula, not the code.

Descent on `L_shape` therefore **shrinks ‖d_a‖** — the shape term is a collapse force, amplified by `∂L/∂U = 512` at the target.

**Clamped regime.** `ρ_min² = R*/4` exactly. Below it: `‖δ‖ = ρ/ρ_min < 1`, so **tr C ≠ 1**, `U` is not floored at `1/d`, and a pair's contribution decays as **ρ⁴** — a second unbounded reward channel. `P(ρ² < R*/4)` is 0.0000 at rank 512 but **0.0902 at rank 4**: the clamp is inert at the advertised optimum and fires on ~9% of pairs in the regime the loss actually produces.

**Is `L_shape` minimised at the claimed target?** No. Target = 0; attainable floor = `log(1e-8)` = **−18.42**.

## 3. Collapse and numerics (pseudocode as written)

- **ρ = 0** (exact duplicates): torch 2.13 special-cases the norm subgradient at 0 to **0**, so `rho**2` yields a *silently wrong zero* gradient rather than a NaN. Not a crash — worse, an undetected one.
- **R_c = 0** (collapsed class): `L_homo = nan`, `L_scale = inf`, gradients NaN. D1/D2's "repelled with divergent force" is numerically **run-destroying**, not repelling.
- **U → 0**: `∂L/∂U` runs 5.12e2 → 1.00e4 → 1.00e6 → **5.12e10**. The steepest gradients occur exactly along the degenerate descent path.
- **fp32**: `1.0 + 1e-8 == 1.0` (eps = 1.19e−7). The guard is invisible near the target and engages only once `d·U < ~1e−7`, i.e. only in the runaway regime.
- `torch.log(R_c).var()` uses correction=1 (÷29); immaterial. Gram FLOPs are `2M²d = 9.2e7`, not 4.6e7 — but the ratio to `1.9e12`–`3.7e12` for the backbone is 2.5e−5 either way, so §1.6's overhead claim stands.

## 4. Kantorovich and the discriminability ratio

The bound itself is **correct** and the `≈0.62` arithmetic reproduces (`1/√(λ̄·(1/λ)‾)` = 0.6178; MC over random w = 0.6371 ± 0.0764). Two claims around it do not survive:

- **"equality iff κ=1" is false.** With λ_k = 1/k, κ=100, floor 0.1980: efficiency is **exactly 1.0000** for `w = e₁`, `e₅₀`, *and* `e₁₀₀`, and equals the floor 0.1980 only at `w = (e₁+e₁₀₀)/√2`. Anisotropy is *perfectly harmless* for eigenvector-aligned between-class directions.
- **"governed entirely by the condition number" is false.** κ bounds only the worst direction. Efficiency = 0.895 / 0.825 / 0.759 for w confined to the top-5 / top-10 / top-20 eigendirections — the plausible DML case, where class means separate along the high-variance identity subspace. Over 2e5 random w, `P(eff < 0.30) = 0`.

## 5. Sphere feasibility

A common **full-rank isotropic** Σ_c with a non-degenerate class mean is **geometrically impossible** on S^511. With `z = αv + w`, `α = √(1−‖w‖²)`, the mean direction is second-order rigid: `Var(α) ≈ τ⁴/2d` against the required `τ²/d`. Monte-Carlo, d=512:

| τ² | λ_rad | λ_tan | required τ²/d | κ floor | analytic 2/τ² |
|---|---|---|---|---|---|
| 0.10 | 1.078e−5 | 1.952e−4 | 1.953e−4 | **18** | 20 |
| 0.06 | 3.719e−6 | 1.173e−4 | 1.172e−4 | **32** | 33 |

So §5's mechanism forecast "**κ ≤ 8**" is unreachable under HIRE's own τ². And the objective cannot see it: that one direction contributes `‖Ĉ_c − I/d‖²_F = 3.4e−6`, leaving `r_eff = 511.1 / 512` while κ = 20. **The optimised functional (r_eff, Frobenius) does not control the functional in the causal argument (κ).** At K=5 the losses can enforce only (i) equality of `R_c` across the P classes in a batch and (ii) a floor on `R_c`; `L_homo`'s null floor from pure sampling noise is `2/((K−1)r)` — verified 0.0010 at r=512 and 0.0130 at r=40 (predicted 0.0125). Since §5 forecasts the PA base at r_eff = 20–60, `L_homo` at λ_ho = 1.0 is dominated by *estimator noise*, and its minimiser at fixed geometry is "make R_c deterministic," which a content-independent within-class configuration satisfies exactly — a D3 channel the C1 control does not cover.

## 6. Literature

- **NIR resolved from primary source.** Roth, Vinyals & Akata (CVPR 2022) use **Normalizing Flows over sample-to-proxy residual densities**, not second moments. So HIRE's shape term is *not* a sign-flip of NIR, and A4 does not fire literally. But NIR's premise — "proxy-based methods can induce **locally isotropic** sample distributions, leading to crucial semantic context being missed" — is the **inverse diagnosis** of §2.1 for the same base loss and the same benchmarks. §3's reconciliation ("second-moment isotropy is max-entropy and therefore *increases* the residual informativeness NIR wants") does not hold: maximum entropy at fixed second moment is exactly the absence of class-specific residual structure, i.e. HIRE's own D3. NIR's published gains are prima-facie evidence against §2.1's error mode.
- **Missed prior art on the core hypothesis.** §3 sources the homogeneity hypothesis only to Box's M and Flury's CPC. Two closer hits: Hamsici & Martinez, **"Spherical-Homoscedastic Distributions"** (JMLR 8, 2007) — exactly the question of when spherical and Gaussian classification agree on the hypersphere, on point for §2.1/§5; and Zhu et al., **"Spherical Feature Transform for Deep Metric Learning"** (arXiv 2008.01469), whose abstract reads *"It relaxes the assumption of identical covariance between classes to an assumption of similar covariances of different classes on a hypersphere."* SFT's *action* (feature transfer) is distinguishable from HIRE's (a differentiable constraint), so this is an incomplete novelty framing, not an independent kill.

## 7. Recipe, sampler, frontier

- **SOP/In-Shop are not executable at P=30, K=5.** SOP train: 11,318 classes / 59,551 images, per-class count min **2**, max 12, mean 5.3, sd 3.0. Every class with <5 images must be sampled with replacement. A 2-image class sampled 5× yields **4 of its 10 pairs at ρ = 0 exactly**: they contribute 0 to U (the shape loss *rewards* duplication), deflate R_c by ~16% (spuriously firing `L_scale`), and inject a `Var_c(log R_c)` signal that is pure per-class image count. In-Shop (3,997 classes / 25,882 images) is the same defect class. The SOP row of §5 is unsupported for reasons independent of the T1 failure.
- **A2 partially resolves against the proposal.** The official README's ResNet-50 CUB command is `--batch-size 120 --warm 5 --lr-decay-step 5`, against §1.2's batch 150, warm-up 1, decay every 10. Optimizer, weight decay, proxy-lr multiplier, epochs, resolution, augmentation, α, δ and pooling are indeed undocumented, as A2 states.
- **Frontier arithmetic reproduces exactly** — CUB 5.37σ / 3.74σ, Cars 2.68σ / 1.87σ, SOP 2.91σ / 2.33σ. But the comparator is the *published* PFML number, and every crossing row requires the PFML base: PA+HIRE at CUB 0.731 is **−0.003 below** PFML 0.734. With A1 conceding the PFML recipe was not retrieved, F6 is gated on an artefact that does not exist. **A PFML+HIRE forecast without an implemented matched PFML base cannot support the standing objective** — even setting aside §1.4, this alone would be BLOCKED rather than LIVE.
- **Cost.** The per-step claim (~1.00×) is sound. The pre-registered *programme* is never costed: main table 190 + tuning 206 + C1–C6/C9 524 + C8 286 ≈ **1,206 GPU-h** (~6 days on 8 GPUs, at 0.35 s/step).
- **The proposal's own gates would not catch the failure.** F2/F3 compare against noise and marginal-isotropy controls, which the degenerate solution does not resemble. F4 is defined on `κ` and `r_eff` — but `r_eff` is computed from the same broken statistic (it reads ≈1/d on the degenerate configuration, per §1(b) above), and per-class `κ` on CUB test identities (~30 images in 512-D) is rank-deficient and unmeasurable. F4 is not executable as specified.

## Preserved as correct

1. **§1.3's decomposition is exact.** `‖Σ_c − (τ²/d)I‖²_F = s_c²‖Ĉ_c − I/d‖²_F + (1/d)(s_c − τ²)²` verifies algebraically, as does the level/spread split.
2. **`E[d_a d_aᵀ] = 2Σ_c`** and **`E[R_c] = 2s_c`** are correct; the mean-free pairwise handle genuinely avoids a class-mean estimate.
3. **The Kantorovich inequality and the 0.62 average-case number** are correctly stated and reproduce.
4. **Orthogonal invariance (§1.6)** is real and is a genuine distinction from VICReg/Barlow-Twins-style coordinate-wise penalties.
5. **The cost overhead claim** (2.5e−5 FLOPs, ~1 MB) is arithmetically sound.
6. **The in-house PA baseline forecasts** (CUB 0.705 / Cars 0.885 / SOP 0.808) are well-calibrated against published PA R50-512 (0.697 / 0.877 / 0.805) at a longer schedule.
7. **D5 is a correct observation** — but it understates: the loss does not merely fail to certify per-identity shape, it actively rewards its worst case. Per the adjudication rules, the caveat does not earn credit because it invalidates the advertised mechanism.

## Uncertainty

- The SOP per-class distribution (min 2, max 12, mean 5.3, sd 3.0) is from secondary reports of the Song et al. CVPR 2016 split; I could not enumerate the raw counts here. The min = 2 fact, which is what makes K=5 infeasible, is corroborated by two independent sources.
- NIR and SFT were resolved from abstracts and search-surfaced verbatim quotes; both primary PDFs returned 403/binary through the available fetcher. The NIR claims I rely on ("locally isotropic sample distributions", "enforce unique translatability", normalizing-flow density on residuals) are verbatim from the CVF/arXiv abstract. I did **not** verify NIR's loss equations directly, so I cannot exclude a second-moment term inside it.
- PFML's recipe and whether its ±0.003 is sd or SEM remain unresolved, exactly as A1 states.
- `check5_descent.py` optimises embeddings directly. A ResNet-50 under image constraints may not reach exact orthogonality; what the experiment establishes is the **sign and absence of an opposing force**, which is architecture-independent, plus that CUB/Cars have the dimensional capacity (400/392 ≤ 512) for the exact solution.

Reproduction scripts: `/tmp/adv/{check1_ustat,check2_grad,check3,check4_arith,check5_descent}.py` (numpy 2.5.1, scipy 1.18.0, torch 2.13.0+cpu). No repository file was modified.

**Sources:** [NIR, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Roth_Non-Isotropy_Regularization_for_Proxy-Based_Deep_Metric_Learning_CVPR_2022_paper.html) · [NIR arXiv](https://arxiv.org/abs/2203.08547) · [Spherical-Homoscedastic Distributions, JMLR 2007](https://www.jmlr.org/papers/v8/hamsici07a.html) · [Spherical Feature Transform for DML](https://arxiv.org/abs/2008.01469) · [Proxy-Anchor official code](https://github.com/tjddus9597/Proxy-Anchor-CVPR2020) · [SOP dataset stats](https://www.tensorflow.org/datasets/catalog/stanford_online_products)
