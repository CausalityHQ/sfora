## Verdict: **DEAD**

**Earliest failed gate: Gate 1 (provenance).**

**Decisive mechanism-level reason.** CoDiF's every flatness term is a functional of the *shape* of the pair-distance CDF between self-anchored endpoints, and is therefore exactly blind to the quantity it was invented to forbid. Because the kernel uses a relative bandwidth, `σ((θ_m−d)/(βθ_m))` is invariant under `d→λd, θ→λθ`; the intra grid runs from `ε` to a quantile of the same-class distances, so both endpoints rescale with the within-class scale. Hence **`L_flat^intra` is *identically* unchanged under `a→λa`** — the only same-class term supplies zero opposition to the scale gap at any λ. The pooled term is worse than blind: shrinking `a` shrinks `ε`, widening `log ρ = log(θ_max/ε)/8`, and since `|Δ²g| ≤ 4|log η| ≈ 44` is bounded, `L_flat = mean(Δ²g)²/(log ρ)⁴ → 0`. The frozen loss is **monotonically reduced by widening the scale gap it claims to penalize**, and its anti-collapse force vanishes precisely as collapse is approached.

---

### 1. Gate-1 provenance — FAILS

§2.1 asserts four things the admissible premise says no corrected checkpoint measurement establishes: the a/b gap, its growth over training, curvature↔official-query-error association, and unseen-identity distinctions living at the training within-class median scale. Literature does not supply this: neural collapse (Papyan–Han–Donoho, PNAS 2020) is a terminal-phase statement about *training* classes under cross-entropy, not a causal account of official-query errors in this pipeline. **Candidate 225's locked ratios (0.9312, 0.9287, 0.9345 — all below 1.15 and below one) are the nearest corrected measurement and point against the premise**: unseen-identity variation is *not* preferentially carried in the training within-class subspace. F3 is not provenance — it is a *different training run* (50 classes, half the data), on CUB rather than the corrected In-Shop screen, and the proposal itself designates it a falsifier for D6. A pre-registered future falsifier cannot retroactively found the premise it is conditioned on. Corrected ARCG's 0.3631–0.3640 same-class retention is likewise not a curvature/error association.

### 2. Grid boundaries and definedness — FAILS

- **`ε=0` and exact view/class collapse:** `ε=0, θ_max=0 ⇒ ρ=(0/0)^{1/8}=NaN ⇒ θ_m=NaN ⇒ L_flat, L_dim, L_flat^intra = NaN`, poisoning MPA's gradient too. **No first-order escape at exact collapse — the run dies.**
- **`θ_max=ε`:** `ρ=1, log ρ=0` ⇒ `0/0` in both `L_flat` and `D̄`. Undefined, unguarded.
- **`θ_max<ε`:** `log ρ<0`, grid descends into the sub-floor region, all `C_m≈0`, all `g_m=log η`, `Δ²g=0` ⇒ **`L_flat=0` exactly**. A view-inconsistent encoder zeroes the core term silently.
- **`θ_0=ε`:** `C_0` counts *different-image* pairs while `ε` is the median *same-image* distance, so `C_0≈0`, `g_0` pinned at `log η` with `σ'≈2.6×10⁻¹²`. The grid anchor is a gradient-dead coordinate; the "extra decades" `L_ε` buys are all saturated.
- **`L_ε` reduces the anchor:** `stop_grad` blocks only within-step credit assignment. `L_ε=(1/B)Σεᵢ²` explicitly minimizes the median, so `ε` drifts down across steps and deflates `L_flat` by `(log ρ)⁻⁴` with no geometric change. D3 does not close this.
- **Magnitude:** at `ε=0.5, θ_max≈1.42`, `(log ρ)²=0.017`, `Δ²g≈5` ⇒ `L_flat≈3×10⁴`, `λ_f L_flat≈1.5×10⁴` against `L_MPA≈O(1–20)`. D5's "blocked by `L_MPA`" fails by 3–4 orders of magnitude at the frozen λ.

*Correct:* the pseudocount `η=1/(2B(B−1))` does prevent `log 0`; the relative bandwidth `βθ_m` is the right choice for a scale-covariant estimator and the justification in (d) is sound; `w≈0.028` is arithmetically right (900/32220).

### 3. "Zero curvature iff power law" — FAILS both directions

⟹ is false: nine collinear log-log samples constrain `C` only at nine points, not on the interval. ⟸ is also false *as frozen*, twice: `g=log(C+η)≠log C`, so an exact power law is curved wherever `C~η` — which is the bottom of the grid by construction; and the smoothing is exactly scale-covariant only for a power law on all of `(0,∞)`, so truncation (no mass below the floor, `d≤2` on `S^511`, `C(θ_max)≡0.6`) biases both endpoints. The memorylessness clause has the direction wrong: `C` is the CDF, so `C∝θ^D` is memorylessness of `−log d`, not `log d`. Hill and Levina–Bickel are *local/tail* estimators; equating them with the global pooled Grassberger–Procaccia integral requires local homogeneity — exactly what D7 concedes is absent. Unstated finite-batch condition: `D ≤ log N/log R`, i.e. `≤4.06` at `N=32220, R=13`.

### 4. Zero set semantics — FAILS

The pooled term is 97.2% different-class pairs. Thirty class means give 435 distances — ample to lay the pooled log-log curve straight across nine bins **while every class is internally collapsed**. `L_flat` therefore forbids NC2 (equiangularity), not NC1 (the claimed failure). The §2.2 two-step staircase argument holds only for the ETF idealization, which the proxy loss does not require. `L_flat^intra` does not close this: as shown above it is *exactly* scale-invariant, so any self-similar within-class shape satisfies it at radius→0. **No frozen term converts smooth distance scale into unseen-identity supervision.** Private class radii, stable background/pose codes, and mixtures whose pooled CDF is a power law all sit in the zero set.

### 5. Second-moment blindness lemma — FAILS

The ETF algebra is right (`(1/C)Σp_cp_cᵀ=(1/(C−1))Π`). Every consequence drawn from it is wrong.
- **Effective rank:** value `C−1=29` is correct; "at its optimum" is not. The batch has n=180 samples, so the attainable maximum is 180. "Maximum *for C classes*" restricts the comparison class to collapse itself — circular. 180 spread points with a decaying spectrum give exp-entropy > 29.
- **Covariance decorrelation:** `Π`'s off-diagonals in the standard basis are generically nonzero under arbitrary orientation. VICReg's covariance term is basis-dependent and is *not* exactly satisfied.
- **Variance floor:** hinge on per-coordinate sd; trace 1 over 512 coords gives sd≈0.044 ≪ γ=1. Massively violated, and its gradient favors more spread.
- **Coding rate:** flatly false. `log det(I+αΣ)` is strictly increasing on the PSD cone, so adding within-class variance strictly *increases* total coding rate; it is not maximal at collapse.
- **"Training-class collapse ⇒ zero unseen-class resolution":** non-sequitur. Zero variance on ~6000 training images does not make φ constant on unseen identities — and Candidate 225's <1 ratios measure against it.

§3 states the distinctness claim "rests entirely on the Lemma." The Lemma is false, so the distinctness claim has no support.

### 6. `L_dim` — structurally infeasible

`C_M` is pinned at ≈0.60 *by construction* (`θ_max` is the 0.60 quantile), so `g_M≈log 0.6` always and the only live lever is `g_0`. Then `D̄ ≤ (log 0.6 − log η)/(8 log ρ) = 10.56/log(θ_max/ε)`. At `θ_max/ε=13`, `D̄ ≤ 4.12`, against `D*=D_0≈20–40`. Reaching `D̄≥20` requires `θ_max/ε ≤ 1.70` — a grid spanning less than a factor of two, which voids the multiscale premise. **`L_dim` and `L_flat`'s stated design goal are mutually infeasible.** So `L_dim` is a permanently saturated ≈256-magnitude tilt, not a floor; D2's fractal-thread defense is vacuous. Worse, its one live gradient pushes `C_0→0` — pushing pairs *out* of the low-scale band, i.e. **widening the gap**. Separately, `D̄` is a two-point chord slope over the *upper* 60% (where `C→1` forces flattening), not `lim_{θ→0} d log C/d log θ`; and `D_0` on unnormalized non-negative 2048-D GAP features has no defined sampler, `ε`, or grid (frozen features have no two-view structure), so `D*=κD_0` compares incommensurable estimators. A scalar dimension is invariant under any shape-preserving diffeomorphism and cannot identify useful directions.

### 7. MPA carrier — invalid, not matched

`s_c = (1/γ)log Σ_k exp(γ⟨u,p̂_k⟩)` without `−log K/γ` gives `s_c ∈ [−1+log K/γ, 1+log K/γ]`; duplicate proxies add a free offset `log K/γ` = **0.271 (K=15)** vs **0.069 (K=2)**. Proxy Anchor's terms are absolute-margin, not shift-invariant, so with α=32 the positive term is multiplied by `e^{−8.67}≈1.7×10⁻⁴` and the negative by `e^{+8.67}≈5.8×10³` — a 3.4×10⁷ pull/push swing. Effective positive margin becomes `δ−log K/γ = −0.171`: **positives count as satisfied at negative cosine with their own class**, while negatives must reach cos < −0.371, which is impossible for C≥100 unit vectors (mean pairwise cosine ≥ −1/(C−1)). At K=15 the frozen carrier is essentially pure repulsion. SoftTriple-style multi-proxy uses a normalized convex combination precisely to avoid this. The K=15/K=2 split therefore does *not* hold the proxy axis fixed — it introduces an uncontrolled per-dataset margin shift, confounding every cross-dataset comparison. A C1 delta measured on this carrier cannot be composed with PFML, whose loss the proposal concedes it never retrieved.

### 8. Prior art — supervision object and action are occupied

- **[LDReg (ICLR 2024)](https://arxiv.org/pdf/2401.10474)** makes local intrinsic dimensionality of the *distance distribution* a differentiable training regularizer against dimensional collapse, via Fisher–Rao on local distance distributions with LID/Hill estimation. This directly refutes the proposal's stated distinctive claim that all such estimators have been used only as "measurements or stopping criteria." `L_dim` is a coarse chord-slope version of LDReg's object and action. LDReg appears nowhere in §3.
- **[MDR (AAAI 2021)](https://arxiv.org/abs/2102.04223)**, Kim & Park, regularizes the pairwise-distance distribution into *multiple levels* under mean/std normalization, on exactly CUB/Cars/SOP/In-Shop. Same supervision object (pooled pair-distance distribution), same action (auxiliary regularizer alongside a metric loss), same benchmark family. Absent from §3.
- **NIR (CVPR 2022)** and MCR² (NeurIPS 2020) are likewise unaddressed.

Per the instruction to judge by object/action rather than stencil, the nine-point curvature form is a parameterization choice inside an occupied neighbourhood.

*Correct:* the ρ-spectral-decay recollection (Roth et al., ICML 2020: tuple-sampling intervention) is accurate.

### 9. D1–D7 and C0–C8 — FAIL

- **View-shared nuisance is unpenalized.** Both views come from one image, so background, pose, illumination and photometric attributes stable under RandomResizedCrop+flip give `ε_i≈0` while generating a rich, smooth, scale-free spread across *different-image* pairs. This is the cheapest route to flatness, it is invisible to `L_ε`, and it is exactly the junk that destroys unseen-identity retrieval. D1's cross-view argument covers only view-*varying* nuisance.
- **D6's proxy cost does not block junk.** `αδω/2 = 1.6ω` ≈ 0.48 nats at ω=0.3, against `λ_f L_flat ≈ 10⁴`. Off by four orders of magnitude; the junk subspace is bought at negligible price.
- **The intra term does not close heterogeneity** — it is scale-invariant (above), so per-class radii may differ by any factor provided each pooled shape is self-similar.
- **C8 is not a placebo.** Permuting grid indices does not preserve magnitude: the sorted order approximately minimizes `Σ(Δ²g)²`, so the permuted objective is systematically larger. And because `C_m` is necessarily monotone in `m`, permuted-order smoothness is *unattainable*; the placebo is a permanently violated, larger-gradient term whose only reachable minimizer equalizes all `g_m` (mass driven entirely outside the grid). F5's ≥50% threshold is not diagnostic.
- **C3 is post-outcome model selection.** Its variance floor is tuned to match *CoDiF's measured final a/b* — a downstream outcome of the treatment arm. F4 then evaluates the treatment against a control configured from the treatment's result.

### 10. Cost, forecast, protocol — FAILS

The `B×B` pair-distance matrix is **new** cost, not "reused": proxy losses compute `B×C` sample–proxy similarities and never form it. Two-view exposure is honestly disclosed at ~1.9–2.0×, with the BN caveat — but the arm count is uncosted: C0–C8 with C6a/b/c ≈ 11 arms × 5 seeds × 4 datasets ≈ 220 runs, plus the six-point λ_f sweep, F3, and two `D*` variants — **≈260 ResNet-50 200-epoch runs, most at 2×**, against a §6 that reports only "<0.5% overhead."

On the frontier: by the proposal's own arithmetic every standalone forecast is below reference — CUB −0.007, Cars −0.014, SOP −0.012, In-Shop −0.006. The single crossing (SOP vs DADA, +0.007) is not against the frontier. The CoDiF⊕PFML composition is explicitly disclaimed by the author and is independently inadmissible under §7. The required corrected paired In-Shop screen is not run first, is not reported raw vs. independently-selected/final, has no OOS confirmation, no second-dataset replication, and no falsifier attached (F1 covers CUB/Cars only) — and its forecast Δ vs C1 is +0.8 against a stated ±0.4 seed SD. **The candidate cannot meet the standing objective even if every forecast came true.**

---

### Preserved correct subcomponents

The ETF second-moment algebra; `w≈0.028`; the pseudocount's `log 0` protection; the scale-covariance argument for relative bandwidth `βθ_m` (§1.3d), which is genuinely the right estimator choice; the ρ-spectral-decay characterization; the honest disclosure of PFML non-retrieval, the GAP-vs-GAP+GMP deviation, the ImageNet bird/car contamination floor as a shared rather than differential problem, the 2× training cost, and the explicit non-assertion of the composed numbers.

### Uncertainty

Numeric illustrations use `ε≈0.5, θ_max≈1.42` as a plausible mid-training operating point; the qualitative conclusions (the exact invariance of `L_flat^intra`, the `(log ρ)⁻⁴` deflation, the `D̄ ≤ 10.56/log(θ_max/ε)` ceiling, the `log K/γ` margin shift, the NaN at collapse) are algebraic and do not depend on those values. I did not retrieve PFML's or IDRR's primary text this session; the IDRR characterization is taken from the proposal's own description, which is sufficient because the Lemma fails on the coding-rate and variance-floor branches independently. Item 1's provenance ruling and item 4's zero-set ruling are each independently dispositive.

Sources: [LDReg (ICLR 2024)](https://arxiv.org/pdf/2401.10474), [LDReg OpenReview](https://openreview.net/forum?id=oZyAqjAjJW), [Multi-level Distance Regularization (AAAI 2021)](https://arxiv.org/abs/2102.04223), [MDR proceedings](https://ojs.aaai.org/index.php/AAAI/article/view/16277)

Any clamp on `ε`, per-class radius constraint, corrected `−log K/γ` carrier, or PFML composition would be a new proposal, not a repair of this one.
