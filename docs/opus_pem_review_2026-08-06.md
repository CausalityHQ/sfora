## VERDICT: **DEAD**

**Earliest failed gate:** §1.2 (Numerics) + §1.5 — the executable specification does not deliver a gradient to `a`/κ, the sole declared scalar learned object. **Earliest failed *mechanism* gate: §2.2 bullet 3 / §2.3 D3.**

**Decisive mechanism-level reason:** For unit vectors, `R_A = |A|·√(1 − tr Σ_A)` exactly. So `E(π) = Σ_A φ(|A|√(1−tr Σ_A))` depends on the embedding **only through within-cluster covariance traces and subset means** — it is blind to covariance *shape*. Its global minimiser is `tr Σ_A → 0`: per-class collapse to a point with separated means. That is exactly the instance-memorising map D3 claims cannot suffice, and it passes every family with the **largest** available margin (exact, κ=96, K=8, means 90° apart: Δ_merge = **−124.3 nats**, Δ_split = **−127.7 nats**, moves strongly negative ⇒ `L_PEM ≈ 0`). D3 is false, and with it the §2.2 "split ⇒ isotropy ⇒ cosine is LR-optimal" chain: isotropy is attained only vacuously, by deleting all within-class variation including the pose/viewpoint variation §2.1 names as the target.

---

### 1. Mathematics — verified exactly (pure-Python `log ₀F₁` in log space, d=512, ν=255)

**Correct.** `p(Z_A) = C_d(κ)^{|A|}C_d(0)/C_d(κR_A)`; `φ(r) = log[Γ(ν+1)(2/κr)^ν I_ν(κr)] = log ₀F₁(;d/2;κ²r²/4)`; `φ(0)=0`; `φ′(r)=κ I_{d/2}(κr)/I_{d/2−1}(κr)` (I re-derived via `I_ν′/I_ν = I_{ν+1}/I_ν + ν/x`); convexity ⇒ superadditivity; `φ ≈ κ²r²/(2d)` small-r. **Cluster-count constant:** `log C_d(0) ≈ 867.9` nats/cluster is *not* partition-independent, but it is correctly absorbed by defining φ with `φ(0)=0` — this is handled right, and the merge/split deltas carry it correctly.

**The one worked number in §2.2 is wrong.** Claimed: at κ=96, R=6.8, separation requirement satisfied for `cos(θ/2) ≲ 0.83`, θ ≳ 68°. Exact: **θ\* = 74.61°**, and at θ=68° the exact `Δ_merge = +34.57` nats — the merge hypothesis *beats* the truth by e³⁴·⁶ ≈ 10¹⁵ in the loss. The error is in the unsafe direction. Cause: the naive `I_ν(x) ≈ e^x/√(2πx)` needs `x ≫ ν² = 65,025`, but the operating point is `x = 653`. Exact `φ(6.8) = 285.15`; the §1.2 "large argument" expansion gives 334.5 — **49 nats off**. Every §2.2/D4 argument built on that expansion is unsupported (D4's *conclusion* survives on monotonicity alone; its derivation does not).

### 2. Executability

- **No gradient reaches κ.** §1.2 specifies only `∂φ/∂z_i`, forward via a spline table rebuilt "whenever κ changes by >1 %". The table is piecewise-constant in κ, so autodiff through it yields **exactly zero**; `∂φ/∂κ = r·A_d(κr)` is never given. §1.1 (LR 1e-2) and §1.5 ("then learned freely") are non-executable as written. Repairable.
- **Table domain too small.** Grid is `r ∈ [0,K]`. Merge needs `‖S_a+S_b‖` up to **2K=16**; move needs `‖S_b+z_i‖` up to **K+1=9**. `φ(8)=366.1` vs `φ(16)=977.8` — spline extrapolation here is not a rounding issue. Repairable.

### 3. κ is not identified — it pins at the clamp, reintroducing a hand-tuned margin

Exact `dΔ_merge/dκ` at R=6.8: **−1.96** (κ=96, θ=90°), **−3.08** (κ=256, θ=90°), **−1.43** (κ=256, θ=68°). Move deltas are cluster-count-neutral so the `−ν log` terms cancel and `dΔ_move/dκ ≈ ‖S_a−z_i‖+‖S_b+z_i‖−R_a−R_b < 0` for any coherent cluster; split deltas fall too whenever excess `R_1+R_2−R_a < ν/κ`. **All three families push κ monotonically up.** Once repaired, κ runs to the ceiling and stops. Consequence: θ\* = **86.1° / 74.6° / 59.2°** at κ = 32 / 96 / 256. The "absolute, dimension-calibrated threshold" is set by the **hand-chosen clamp endpoint 256** — precisely the hand-tuned margin the one-line mechanism claims to have eliminated. F7 (`κR̄/ν < 0.8`) is then vacuous: κR̄/ν = 6.8 by construction.

**Stiffness / loss dynamics.** Exact `dΔ_merge/dθ` = −5.2 to −6.1 nats/deg (κ=96), −16 to −19 (κ=256). With ~8,464 hypotheses the LSE is a max (e⁶ per degree), so **L_PEM ≈ softplus(max_π Δ_π)** — hardest-hypothesis mining, not a partition posterior. And Δ = −89 at θ=90°, κ=96 ⇒ `softplus(−89) ≈ 2×10⁻³⁹`: **float32 underflow, gradient identically zero**, including to κ. The transition from full gradient to none spans ≈3°. This is a hard on/off objective, not the "weak gradients / rich-get-richer" risk named in §6. The `log|𝒩| ≈ 9`-nat offset from partition count is real but second-order against these scales.

### 4. C1 — "the single most important experiment" — is not a valid control

`φ̃(r) = κr` is **linear ⇒ additive**, so by the triangle inequality `Δ_merge = κ(R_ab−R_a−R_b) ≤ 0` **identically** (the whole merge family is vacuously satisfied, including at total collapse) and `Δ_split = κ(R_1+R_2−R_a) ≥ 0` **identically** (vacuously violated unless within-class scatter is exactly zero). C1 also silently drops the per-cluster `log C_d(0)`. So C1 deletes the superadditivity that D1/D2/D3 credit for everything and inverts the sign structure of two of three families. F3 cannot bear weight in either direction. Worse: **the Occam factor and the convexity are the same object** — there is no way to remove `−ν log(κr)` while keeping convexity — so the proposal's decomposition (Occam = mechanism, convexity = anti-collapse) is not a real decomposition. C2 is also unmatched: κ=8 sits outside PEM's own clamp and changes the loss scale by ~(96/8)² ≈ 144×, conflating regime with effective LR.

### 5. Sampler / protocol

In-Shop train ≈ 3,997 classes / 25,882 images ≈ **6.5 img/class**; K=8 without replacement excludes a large fraction of identities (modal item has 3–7 views) — unquantified. SOP ≈ 11,318 / 59,551 ≈ **5.3 img/class, min 2**; K=4 excludes every 2- and 3-image class — unquantified. C6's K∈{2,4,8} sweep therefore changes the *identity pool* between arms, confounding the regime control with a data-quantity change. In-Shop is forecast but demoted to "non-target"; there is no In-Shop-first screen.

### 6. Causal provenance and forecasts

§2.1 offers **no measurement** that free per-class proxies absorb calibration/anisotropy — and its own frontier contradicts it: the diagnosis predicts more free per-class capacity hurts zero-shot, yet PFML (**15 proxies/class** on CUB/Cars) is the strongest cited baseline, and SoftTriple (multi-centre) likewise beats single-proxy PA. §5's forecasts follow from no measured premise: no pilot, no scaling relation, B0/B1 unreproduced. I confirmed the arithmetic (SEMs 0.001342/0.002236; pooled 0.002608, **t=2.684**; Cars pooled 0.002236, **t=2.236**; Reading 2 pooled 0.003742, **t=1.871**) — it is correct, and it says the **sole** crossing is CUB, only under Reading 1, only at α=0.05. Minor slips, both conservative: the α=0.05 requirement is Δ≈0.0090 (not 0.0105) and ~9 seeds (not 12). **A programme is not warranted when the only payoff is a coin-flip on the meaning of "±0.003" in a paper the author has not reproduced.**

### 7. Novelty

§3 self-declares the same *slot* as Song et al. 2017. The evidence functional is the standard Bayesian-vMF marginal likelihood (Banerjee et al. 2005; Gopal & Yang 2014). Integrating out the class latent and training the network on the resulting marginal LR is end-to-end PLDA training in speaker verification (e.g. Rohdin et al., ICASSP 2018) in Gaussian form. §3 omits three works the brief names: **Deep Spectral Clustering Learning** (Law, Urtasun & Zemel, ICML 2017), **large-margin metric learning for partitioning** (Lajugie, Arlot & Bach, ICML 2014), and **DeepDPM** (Ronen et al., CVPR 2022). §3's row conflates "Non-isotropy Regularization" (CVPR 2022, arXiv 2203.08549) with the cited arXiv 2207.03784 ("A Non-isotropic Probabilistic Take…", ECCV 2022). **New supervision object: none.** A known evidence functional composed with a known structured-partition supervision is a new scoring functional, not new supervision.

---

### Preserved as correct (independent of the verdict)

The §1.2 derivation in full (marginal likelihood, φ's closed form, the `₀F₁`/Bessel identity, the analytic `φ′ = κA_d(κr)`, the backward `κA_d(κR_A)·S_A/R_A`, `φ(0)=0`, small-r expansion, convexity ⇒ superadditivity); the correct handling of the cluster-count constant; **D1** — collapse is a near-maximiser (exact: Δ_merge = **+245.7** nats/pair at κ=96, so L ≈ 252 nats, far above the loose 6.2-nat bound quoted); **D2** and **D5**; §1.6's arithmetic (R̄=6.8, κ≥37.5, κR̄/ν=2.56; SOP 336/255=1.32); all parameter/FLOP counts (5.8 M SOP proxies, 1.05 M head, 0.26/4.2/1.05 MFLOP, ≪0.1 % of a step); the test-time reduction to plain cosine NN; §5's t-arithmetic; and the document's own disclosures (C9 warm-up contamination, straight-through split gradient, "compare only against my own B1", the open-novelty admission).

**Uncertainty I hold:** PFML's authorship and its 0.734/0.927 values are unverified by me; In-Shop/SOP per-class histograms are cited from published split statistics, not recounted here; the two unread arXiv entries in §3 remain unresolved. None of these changes the verdict.
