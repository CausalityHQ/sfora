I have reproduced the mathematics independently (pure-Python Monte Carlo, exact 3-scalar reduction of the 512-dim dither; script at `/tmp/darc_check/sim2.py`, outside the repo). No repository files were edited and no git history, prior review, ledger, or conversation was consulted.

---

# ADJUDICATION — DARC (Discriminant-Axis Residual Capacity), Pass 36

# VERDICT: **DEAD**

**Earliest failed gate:** §1.5 (*The DARC term* — the executable estimator), with its root cause displayed one section later at §2.2 D1.

**One decisive mechanism-level reason:** The renormalization after dither divides by `‖z+σε‖ ≈ √(1+σ²d)`, not `√(1+σ²)`. At d=512, σ=0.10 that is 2.474, not 1.005. Because the proposal calibrated its absolute constants against the latter, the differentiable clamp `softplus(τ̂/s)·s` with s=10⁻³ injects a **constant `s·ln2 = 6.93×10⁻⁴` into the capacity numerator that is 0.37×–1.13× the entire noise denominator** across the pre-registered σ grid. On every axis carrying no reliable within-class signal — which at σ=0.10 is every axis holding less than ~3.1× the isotropic share of a *full* unit variance budget — `L_DARC` reports 0.16–0.38 nats of capacity from literally nothing and emits a ν̂-gradient 0.72×–2.25× the size of its τ̂-gradient. The executable term is therefore not a capacity allocation over most of its operating range; it is a **cross-view augmentation-invariance penalty restricted to 16 detached directions** — VICReg's invariance term, class-conditioned and rotated. §4 contains no arm isolating that objective, and the one control that would expose it (C3) carries no pre-registered falsification threshold.

---

## Gate 1 — Formulas, gradients, and the dimension-free floor

**Reproduced correct.** `∂C/∂τ = 1/(2(ν+τ))` and `∂C/∂ν = −τ/(2ν(ν+τ))` are exact for `C=½log(1+τ/ν)`. `exp(C_k)=√(1+τ/ν)` is the correct Gaussian-channel level count. ε=10⁻⁶ is harmless (≤0.16% of ν̂ at every σ).

**The dimension-free bound is false.** With `‖ε‖²≈d`, `E[ν̂_k] = σ²/(1+σ²d)·[(1−c²) + c²S²/(1+S)²]`, `S=σ²d`, `c=u_kᵀz`. Monte Carlo (N=3×10⁵/cell) against the claimed `σ²/(1+σ²)`:

| σ | σ√d | S=σ²d | signal power frac | **claimed floor** | MC, c=0 | MC, c=1 | worst-case overstatement |
|---|---|---|---|---|---|---|---|
| 0.03 | 0.679 | 0.461 | 0.685 | 8.99e-04 | 6.14e-04 | **1.28e-04** | 14.7× (analytic) |
| 0.05 | 1.131 | 1.280 | 0.439 | 2.49e-03 | — | — | 7.2× |
| **0.10** | **2.263** | **5.120** | **0.163** | **9.90e-03** | **1.64e-03** | **1.26e-03** | **8.7×** |
| 0.20 | 4.525 | 20.48 | 0.047 | 3.85e-02 | 1.86e-03 | 1.74e-03 | 22.7× |

The claim is violated at **every σ and every alignment tested**, by 1.5–22×. At the default σ=0.10 the vector fed to the estimator is 84% dither; at σ=0.20 it is 95% dither.

**The floor is also encoder-controllable, contradicting "no parameter setting reduces this floor."** Renormalization suppresses the noise component parallel to the direction the descriptor already occupies, so driving `|u_kᵀz_i|→1` cuts the measured floor: at σ=0.03, MC 6.14e-04 → 1.28e-04 (**4.8×**, ≈0.78 nats free); at σ=0.10, 1.64e-03 → 1.26e-03 (23%). This is bounded by Bessel (`Σ_k c_ik² ≤ 1`) so it cannot be harvested on all 16 axes, but the gain is convex in c², so the optimum is to concentrate on the highest-w_k axis — and doing so requires **increasing between-class mass along the top certified axis, which Proxy-Anchor already rewards and which raises λ₁ hence w₁**. D1 and D4 jointly claim to block this; they do not. It is a positive-feedback channel carrying zero within-class information.

**What survives.** The `1/(1+σ²d)` attenuation cancels between numerator and denominator, so `τ̂/ν̂ → τ_raw/σ²` *is* genuinely dimension-free (MC-confirmed: at σ=0.10, τ_raw=0.05 → τ̂_meas 8.20e-03 vs predicted 8.17e-03). The *idea* that σ supplies an absolute unit is sound; the *stated bound* and every absolute constant tuned against it are not.

## Gate 2 — Unbiasedness under two views, one source image, and finite-batch centering

**ν̂ is unbiased — correctly, and non-obviously so.** Centering by the dithered class mean cancels exactly in the view difference (`a_{ik1}−a_{ik2}=ξ_{ik1}−ξ_{ik2}`). MC: 1.5989e-03 vs ν=1.6e-03. **Credit this.**

**τ̂ is not unbiased. The boxed claim at line 68 is false.** Centering leaves

`E[τ̂_k] = (1−1/K)·τ_k − ν_k/(2K)`

MC confirms exactly: (τ=1, ν=1, K=6) → 0.7521 vs prediction 0.750, claim 1.0. (τ=0, ν=1.6e-3, K=6) → **−1.31e-04** vs prediction −1.333e-04, claim 0. On SOP (K=3) the multiplicative loss is 1/3 and the additive term doubles.

**The softplus creates a material positive floor.** `softplus(0)·s = 6.93e-04`:

| σ | ν actual | [τ̂]₊ at τ̂=0 | ratio to ν | spurious C_k | \|∂L/∂ν̂\| | \|∂L/∂τ̂\| | ν/τ pressure |
|---|---|---|---|---|---|---|---|
| 0.03 | 6.16e-04 | 6.93e-04 | **1.13** | **0.376 nats** | 430 | 191 | **2.25** |
| 0.05 | 1.10e-03 | 6.93e-04 | 0.63 | 0.245 | 177 | 140 | 1.26 |
| 0.10 | 1.63e-03 | 6.93e-04 | 0.42 | 0.177 | 91 | 107 | 0.85 |
| 0.20 | 1.86e-03 | 6.93e-04 | 0.37 | 0.158 | 73 | 98 | 0.74 |

Real signal only beats the clamp scale when `τ_raw > s(1+S)` = 1.5e-03 (σ=.03) to 2.1e-02 (σ=.20) — i.e. 0.75× to 11× the isotropic per-axis share of a *full* unit budget; against a realistic within-class budget W≈0.2–0.4 that is ~2× to ~28×. **Note the perverse ordering: the artifact is largest at the smallest σ, and DARC's own logic (finer ruler ⇒ more measured capacity) drives the σ sweep toward small σ.** The method will select into its own artifact.

## Gate 3 — Cheapest optimization shortcuts

- **D2/D3 hold.** Augmentation-parameter leakage is view-inconsistent by construction and lands in ν̂ where it is penalized. The sign inversion against AugSelf/E-SSL and against the EAAI-2024 augmentation-intensity line is genuine. **Credit.**
- **Frame steering (D4): correctly labelled partial, but weaker than presented.** ρ=0.99 → ~100-step time constant against ~6,500 total steps (5,864/180×200). The lag is 1.5% of training; over training the frame is encoder-determined.
- **Alignment shortcut: unblocked and unmentioned** (Gate 1 above).
- **D6 brake 2 (power budget) fails quantitatively.** Bits available on 16 axes under water-filling (`Σ_k ½log₂(1+τ_raw/σ²)`, the (1+S) factors cancel) versus the 12.52 bits needed to index CUB's 5,864 training images:

| W on 16 axes | σ=0.03 | σ=0.05 | σ=0.10 | σ=0.20 |
|---|---|---|---|---|
| 0.10 | **23.9 b** | **14.5 b** | 5.6 b | 1.7 b |
| 0.20 | **31.2 b** | **20.7 b** | 9.4 b | 3.1 b |
| 0.40 | **38.8 b** | **27.7 b** | **14.5 b** | 5.6 b |

Over most of the pre-registered grid the sphere budget affords 1.2–3.1× the bits required. "Hash bits are paid for out of the same sphere that class separation needs" is true and insufficient. Brake 1 (saturation) caps τ but not below the requirement; brake 3 is conceded as non-proof. **D6 is unblocked.** And it is not even the leading risk: view-consistent *background* scores τ̂ with no memorization at all, and the proposal lists that risk without blocking it.

- **C5 cannot detect D6.** With one image per identity label there is no held-out split; setting `W = Zᵀ` gives `w_iᵀz_i = 1 > z_jᵀz_i` for any distinct unit descriptors, so **train top-1 is ~100% for any injective embedding, baseline included**. The probe measures injectivity, not memorization. F3 is therefore inert.

## Gate 4 — The reverse-water-filling claim

The algebra `τ*_k = [βw_k/(2g_k) − ν_k]₊` follows from the stated stationarity. The *identification* does not.

1. **It is not reverse water-filling.** Reverse water-filling holds distortion at a level **common to all channels**; here the level `βw_k/(2g_k)` is per-axis. β is not the water level. It is an ordinary equal-marginal-return KKT condition.
2. **The sphere never enters.** The proposal calls `‖z‖=1` the power budget that "makes the allocation a genuine trade," yet no Lagrange multiplier appears; the entire trade in the displayed equation comes from `g_k`. With the constraint, stationarity reads `g_k + λ = βw_k/(2(ν_k+τ_k))`, and λ is unestimated.
3. **ν_k's own stationarity is omitted** although the same loss drives it (Gate 2 table: at σ=0.03 the ν-pressure is 2.25× the τ-pressure). τ and ν are not independent coordinates — τ̂ is *defined* as `(1/n)Σā² − ν̂/2`.
4. `g_k`, `w_k`, `λ_k`, `u_k`, and the class means all move; "axis k" is not a persistent coordinate.

It proves neither a Gaussian rate–distortion allocation, nor an interior optimum, nor uniqueness. Uniqueness in τ_k holds only for *fixed* g_k, and D6 brake 1 leans on exactly that fixed-g_k reading.

## Gate 5 — Causal direction

The top eigenvectors of S_B are the directions in which **seen** labels separate. DARC raises within-class variance on precisely those axes, i.e. it lowers the Fisher ratio where it is highest. It is anti-LDA on the labels' own best directions.

The objective's only selectors are *view-reliable* and *on a top-S_B axis*. Neither distinguishes (a) variation that would be between-class under a finer label set (useful for unseen identities) from (b) variation that is within-class at any granularity — sex, individual, pose, illumination, background. On CUB, (b) dominates within-species variation, and background–species correlation is strong enough that top-S_B axes are partly context axes. Nothing in the mechanism selects continuous semantic attributes. D7 is presented as one adverse axis; the argument above makes it the generic case.

**The C7 mediator does not identify causality.** Leg (i) — raise train-class `Σw_kC_k` — *is the training objective*; it is a manipulation check that must pass, not a mediator. Leg (ii) — reduce test-class level-snapping — is mechanically satisfied by any intervention that increases within-class dispersion on those axes, including plain noise injection, weaker weight decay, or label smoothing. Both legs can pass while the claimed causal path is absent. F2 is therefore not a mechanism test.

## Gate 6 — Prior art and the exact new supervision object

The proposal's own §3 is well-constructed and honest about the motivation being occupied. Two additions from primary sources:

- **DVML (Lin et al., ECCV 2018)** states DARC's problem statement verbatim: *"Most existing methods usually enforce the model to be indiscriminating to intra-class variance, which makes the model over-fitting to the training set… and leads to low generalization power on unseen classes."* Absent from §3.
- **Ranked List Loss (Wang et al., CVPR 2019)** explicitly *"learn[s] a hypersphere for each class… rather than shrinking each class to a single point,"* to *"preserve intraclass data distribution."* Absent from §3.
- **Sharing Matters (Milbich et al., TPAMI 2020)** targets characteristics shared across training classes precisely because they recur in novel categories. Absent from §3.

**Exact new supervision object or action: none.** Decomposed, the executable term is: VICReg's invariance term (ν̂↓) + VICReg's variance floor made class-conditional and rotated (τ̂↑), combined as a log-SNR that is Cronbach's ICC / Shannon's Gaussian channel, measured on LDA between-class scatter eigenvectors, with an exogenous additive-noise ruler of the kind used to give rate an absolute scale in Ballé et al. (ICLR 2017) and VIB, and a sign that is exactly negated CEB (Fischer 2020, which minimizes I(X;Z|Y)). No new label, annotation, measurement, or invariance is introduced. Per the review constraint, a novel assembly of occupied terms is not credited as a new supervision object. The proposal concedes this ("my novelty has to live entirely in the mechanism") — and Gates 1–2 show the mechanism is not what is written down.

## Gate 7 — Controls and protocol

- **Compute matching holds under the strict reading** (180 forward passes/step, equal steps/epoch, 1.01× epoch time) — but **unique-image exposure is halved**: 200 DARC epochs = 100 dataset passes. This is never stated. It cuts against DARC, so it is not a validity threat to a positive result, but it means **C1 is the only admissible comparator**.
- **F1 is defined against the wrong arm.** F1 measures Δ over "my own baseline" (30×6, 180 unique). DARC's arm is 15×6×2, which also changes `|P⁺|` in Proxy-Anchor's positive term from 30 to 15 — a material change to the base loss's normalization. C1 isolates this, but **no pre-registered rule links C1 to F1**, so F1 can pass on a batch-construction artifact that §4 explicitly names as disqualifying.
- **F1's conjunction is too weak**: falsification requires failing on CUB *and* Cars. +0.9/+0.0 survives.
- **C1, C3, C4, C8 carry no thresholds.** C3 is called "the control that proves the dither is the mechanism" yet has no consequence attached. Its stated prediction is also wrong: with σ=0, cross-view correlation is `τ/(τ+ν)`, whose maximizer is any augmentation-invariant non-collapsed encoder — a real invariance penalty, not "a near no-op."
- **C6 omits the leading occupied alternative**: a plain cross-view consistency penalty (minimize Σw_kν̂_k), which is what DARC reduces to under Gate 2. F5 therefore cannot fire on the most likely explanation.
- **F4 is non-discriminating**: essentially every regularizer weight yields a non-monotone R@1(β) with an interior optimum.
- **F5 is underpowered**: a 0.3-pt band against a 5-seed paired SE of ≈0.22 pt is ~1.4 SE.
- **Good practice, credited**: hyperparameters tuned on a class-disjoint holdout carved from *training* identities (80/20 on CUB) with retraining on all 100 at frozen values; "beat the best of C6, not the default"; frame fitted on train images only and discarded before evaluation.
- **Protocol order is inadmissible.** The governing protocol requires a corrected In-Shop-first screen. §6 declines it — for a stated and defensible reason (PA+DADA's 0.930 has no reported seed count) — and pre-registers five-seed CUB/Cars decisions instead. A reason to prefer a different order is not authority to substitute one. The correct response is a corrected In-Shop forecast with the uncertainty stated, or an explicit protocol amendment, not silent reordering.

## Gate 8 — Forecasts, frontier, and expected value

**The arithmetic reproduces exactly.** Standalone: CUB −2.6 pt (4.46σ), Cars −3.3 pt (5.7σ), SOP −2.3 pt (4.3σ); also below every DADA matched-cost row (−2.1/−2.7/−0.4). Composed at 0.55×: 0.7439 (+0.99 pt, 2.33σ), 0.9342 (+0.71 pt, 1.69σ), 0.8328 (+0.39 pt, 1.36σ) — matching the stated +1.0/2.4σ, +0.7/1.7σ, +0.4/1.4σ. Cost estimates check: 512²×4 B = 1.05 MB; 512³ = 1.34e8 flops/20 steps ≈ 3e-6 of a ResNet-50 step (the stated "<0.001%" is conservative and true).

**Against an objective to outperform the matched frontier, the proposal forecasts failure.** Standalone: 0 of 3 crossings. Every crossing runs through a PFML composition whose recipe the proposal states it does not have, at an additive fraction (0.5–0.7×) supported by no evidence. Its own P(all three cross) = 0.08.

**Worse, the additivity assumption points the wrong way.** DARC's error mode is defined by one proxy per class producing a C-level quantizer (§2.1). PFML uses **15 proxies/class on CUB/Cars** (§1.1). A 15C-level codebook has already mitigated ~15× of the coarseness DARC exists to remove. The mechanism-consistent shrink factor on that base is plausibly ≈0, not 0.55. The composition row is not merely unsupported; it is anti-supported by the proposal's own mechanism.

**Is the programme justified before the base is reproduced?** No. A 60-point σ×β×K sweep, eight controls, and five-seed runs on three datasets spend the budget on a path whose terminal standalone node is known in advance to sit 2.3–3.3 pts below the frontier, and whose only crossing path is gated on reproducing a recipe the proposal states is unavailable to it.

---

## Preserved subcomponents (correct, independent of the verdict)

1. **ν̂ is exactly unbiased under within-class centering by the dithered sample class mean** — MC-confirmed. The view-difference construction annihilates the centering error. This is a genuinely good estimator design and is reusable.
2. **The SNR ratio is dimension-free even though the stated bound is not**: `τ̂/ν̂ → τ_raw/σ²`, MC-confirmed. The concept of an exogenous dither as an absolute ruler for a within-class reliability objective survives the failure of §2.2 D1's arithmetic.
3. **Both displayed derivatives are exact**, and `exp(C_k)=√(1+τ/ν)` is the correct Gaussian-channel level count.
4. **D2/D3 are sound**: augmentation-driven variance provably lands in ν̂ and is penalized. The sign inversion against AugSelf/E-SSL and against the augmentation-intensity intra-variance line is real, not cosmetic.
5. **The frontier arithmetic and cost model are correct** and the standalone non-crossing is reported plainly rather than dressed up.
6. **Holdout discipline is correct**: class-disjoint holdout carved from training identities, retrain at frozen values, frame discarded before evaluation.
7. **C6's standard** ("beat the best, not the default") and **F5** are the strongest parts of the protocol.
8. **The Cox–Reid rejection is right for the stated reason**: as separation grows, `j_pp→0` and the `−½logdet` adjustment ceases to oppose the likelihood.

## Uncertainty

- The alignment shortcut (Gate 1) is verified in the exact dither geometry but not in a trained network; how much `|u_kᵀz_i|` the encoder can actually buy against Proxy-Anchor's other pressures is untested here.
- The hash-capacity table (Gate 3) depends on W, the within-class variance resident on the 16 axes, which the proposal never quantifies; I bracketed W∈[0.1,1.0] and the conclusion holds across σ≤0.05 for all of it and across all σ≤0.10 for W≥0.4.
- Gate 5 is a mechanism argument, not a measurement. It is refutable — but not by C5 or C7 as specified.
- The DVML / RLL / Sharing-Matters omissions are incompleteness in §3, not contradictions of the proposal's own framing, which already concedes the motivation is occupied.

**None of these uncertainties reaches the verdict.** §1.5's estimator and clamp are wrong at the proposal's own defaults by margins established with exact asymptotics and confirmed by Monte Carlo, and the section that would have caught it, §2.2 D1, contains the dimension error that produced it.
