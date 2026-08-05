# Pass 22 independent review: CINA

Date: 2026-08-05 UTC  
Frozen proposal: `docs/fable_cina_proposal_pass22_2026-08-05.md`  
Frozen review prompt: `docs/fable_cina_review_prompt_2026-08-05.txt`  
Review prompt SHA-256: `cb55dd10cfa26240c807d218e712ef720721ad9a7cd98b4df997a60a462d2e73`  
Native durable consultation: `202f2f67b8b44602`  
Caller attribution: `sfora/emafactorial`  
Provider path: Fable, then configured same-job Claude Opus fallback  
Durable status: failed (`exit 1`); result untruncated and cleanly ended.

The following is the exact durable `consult_result`, without repair or adjudication.

---
I'm the cold reviewer for this frozen proposal. I will not touch the local repository; my review uses only the frozen text plus web/primary-literature checks. Plan: first verify the factual anchors (PFML, ProxyAnchor recipe constants, PA+DADA), then run the prior-art sweep the proposal itself requests, then deliver the gate-by-gate verdict. Loading the web tools now.
# Verdict: **DEAD**

---

## The single strongest decisive reason

**The debias null is provably, structurally noisier than the statistic it debiases — for every `(C, m)` — so `L_CINA = max(0, D(A,B) − sg[D_null])` is systematically over‑subtracted, and on SOP/In‑Shop it is a no‑op while the proposal simultaneously forecasts a positive SOP Δ.**

The two branches' degrees of freedom are fixed by the spec:

| | statistic dof | null dof | ratio |
|---|---|---|---|
| general | `n = (C/2)(m−1)` | `n_null = C(⌊m/2⌋−1)` | `2(⌊m/2⌋−1)/(m−1)` |
| CUB/Cars (`C=20, m=9`) | **80** | **60** | 0.75 |
| SOP/In‑Shop (`C=36, m=5`) | **72** | **36** | 0.50 |

Since `2(⌊m/2⌋−1) ≤ m−2 < m−1`, we have `n_null < n` **strictly, for all m ≥ 2**. There is no reading of step 8 that repairs it: restricting the null to `C/2` identities gives 30 dof on CUB (worse); using all `m` samples violates disjointness. The null is *definitionally un‑matchable* to the statistic under this batch shape.

This falsifies Prop. 2's load‑bearing parenthetical — "*estimated under the same rank/sample budget*" — which is simply false as written. `D` under H₀ is a projected matrix‑variate Beta/Wachter statistic whose log‑eigenvalue dispersion grows monotonically in `γ = k/dof`. The cross branch runs at `γ = 24/80 = 0.30` (CUB) and `24/72 = 0.33` (SOP); the null runs at `γ = 0.40` and `0.67`. So `E[D_null] > E[D | H₀]` always, and the hinge does not open until the true signal exceeds the *gap between two mismatched noise floors*, not the noise floor.

Scale that against the signal the proposal itself derives: `E[D] = (2/h)·V`. On SOP, `h = 18` ⟹ the signal is attenuated to `V/9`, while the null‑vs‑statistic noise gap at `γ: 0.33 → 0.67` is order ~1 in log‑eigenvalue variance units. Activation would require `V ≳ 9`, i.e. per‑identity log‑shape dispersion with std ≳ 3 (≈25× eigenvalue ratios between identities). **`L_CINA ≡ 0` on SOP/In‑Shop with near‑certainty.** On CUB/Cars the gap is smaller (`γ: 0.30 → 0.40`) and the hinge plausibly does open — but at a zero‑point that is an uncontrolled function of batch shape rather than of the quantity being penalized.

This is decisive on its own: the frozen spec is inert on two of the four legal benchmarks *and* forecasts +0.004 on one of them (§5 Tier 1). That is an internal contradiction between the mechanism and the forecast, not a tuning issue. Any fix is a new proposal.

---

## 1. Executability and differentiability

**Correct as claimed:** residuals and the `−1/m` mean term; `n = (C/2)(m−1)` as the Bessel‑corrected pooled estimator; `Σ̂_A ⪯ Ŝ` ⟹ `μ_i ∈ [0,1]`; `eig(Ŝ^{−1/2}Σ̂_BŜ^{−1/2}) = 1−μ_i` on the same basis; `ν_i = log(μ_i/(1−μ_i))` are the log generalized eigenvalues of `(Σ̂_A, Σ̂_B)`; `D` is symmetric under `A↔B` (`ν → −ν`); `D_max = (log 999)² = 47.7` ✓; `Cm = 180` for both configurations ✓; `λ(t)` is executable; deployment is byte‑identical to base ✓.

**Defects:**

- **§6 cost accounting omits the only expensive operation in §1.** Step 4 requires the top‑`k` eigenvectors of `S ∈ R^{512×512}` **every step** (and a second one for the null branch, if `U` is recomputed). §6 counts "two 24×24 eigensolves" and nothing else. A 512×512 symmetric eigensolve is latency‑bound on GPU (`~1–5 ms`, poorly overlapped), not FLOP‑bound, so the "<0.01% FLOP / +0.5–1.0% wall‑clock" claim is not derived from the specified algorithm. "*measured target*" is also not a measurement.
- **The eigengap jitter is misspecified.** "`10⁻⁶·tr(Ŝ)/k`" as a diagonal ridge shifts every eigenvalue equally and **leaves all eigengaps unchanged** — it cannot fix eigengap‑driven gradient blow‑up. Underdefined whether it is a ridge on the matrix or a floor on backward‑pass denominators. Ironically the remedy targets a problem that largely isn't there (`D` is a *symmetric* function of the eigenvalues, so `∂D/∂M = V diag(∂f/∂λ) Vᵀ` is smooth under degeneracy) and does not address the one that is (the Daleckii–Krein derivative of `Ŝ^{−1/2}` needs `λ_min(Ŝ) > 0`).
- **The `[η, 1−η]` clamp zeroes the gradient exactly on the worst offenders.** Clamped `μ_i` contribute a constant to `ν̄` and no gradient. The directions with the largest shape disagreement are precisely those excluded from the learning signal.
- **Numerical behavior under near‑collapse is unhandled.** As `Σ → 0`, `tr(Ŝ) → 0` and the jitter scales with it; there is no absolute floor, so the eigensolve degrades to noise in fp32 with no guard.
- **Underdefined:** whether `U` is recomputed from `S_null` or shared with the cross branch (step 8 says "repeat 3–7", which includes step 4). This is not cosmetic — it changes whether the two branches are measured in the same window at all.

## 2. `D = 0`, affine invariance, and null comparability

- **`D = 0 ⟺ Σ̂_A ∝ Σ̂_B`: correct** on the projected matrices (all `ν_i` equal ⟺ `Σ̂_B^{−1/2}Σ̂_AΣ̂_B^{−1/2} = ρI`). But the object constrained is the **pooled** covariance of two identity *sets*, not per‑identity covariances. `D(A,B) = 0` places **zero constraint on any individual `Σ_c`** — identity‑specific shapes need only average out. The step‑7 "⟺" is true and the inference drawn from it in Prop. 2 is not.
- **Signal attenuation is admitted and then ignored.** Prop. 2 states `E[D] = (2/h)V`: the estimator discards ~80–90% of the target signal by pooling, and the residual is then compared against a *larger*, mismatched noise floor. With a hinge on top, the steps that produce gradient are selected for **large noise realizations**, so the gradient is preferentially fitting which images landed in `A` vs `B`.
- **Affine invariance does not survive the stopped top‑k projection.** `U` = top‑`k` eigenvectors of `S` is **not** equivariant under `z → Az`; the measured subspace itself moves. Exactly preserved: `O(d) × R⁺` (verified: under `z → cQz`, `U → QU`, `Σ̂_A → c²Σ̂_A`, `Ŝ → c²Ŝ`, `μ_i` unchanged). The proposal's "*only approximately*" understates it: at `k=24` of `d=512`, the broken part of `GL(512)` is essentially all of it (the full symmetric part, `512·513/2 − 1` dimensions).
- **The null is non‑comparable on three axes, not one:** dof (80 vs 60; 72 vs 36), class composition (disjoint identity sets vs the *identical* identity set in both halves, so the class‑scale mixture cancels in the null and does not in the statistic — the two have different *effective* dof under scale heterogeneity), and subspace selection (underdefined). `D − sg[D_null]` is therefore neither unbiased for `V` nor a comparable statistic. Its bias is dataset‑ and training‑state‑dependent with an uncontrolled sign.
- Credit where due: selecting `U` from the *sum* `S` is a better choice than selecting on `Σ_A` alone (cf. the Wishart–Beta independence of `S` and `S^{−1/2}Σ_AS^{−1/2}`). But that classical result covers the unprojected `k×k` Beta matrix under a *common* `Σ`, not the top‑`k`‑projected version under heterogeneous class scales, and the proposal offers no argument for the extension.

## 3. Degeneracies

- **D2's cost floor is invalid in the regime it must cover.** "`∂L_base/∂s ≥ α/4`" holds only when the ProxyAnchor softmax weight is `≥ 1/2`. The attractive term is `log(1 + Σ e^{−α(s−δ)})`; as training converges its gradient decays **exponentially**. The bound is asserted "in the active regime" and then used as a universal floor — the degeneracy would be exploited precisely where the floor has vanished.
- **The dominant version of D2 is not the one attacked.** The proposal prices *increasing* total scatter `Δs > 0`. The actual cheap move is **reallocation at constant `s`**: shift within‑identity variance out of identity‑specific directions and into a common high‑rank nuisance subspace. `Δs = 0`, so the pre‑registered cost bound prices it at zero, and it reduces `D` directly.
- **D4's defense covers rank‑1 only.** "Top‑`k` of `S` anti‑selects" applies to a rank‑1 shared nuisance. A **rank‑≥24 common nuisance subspace** is exactly what the penalty rewards, is fully compatible with the top‑`k` construction, and would *raise* — not lower — the F5 diagnostic. **F5 points the wrong way for the live attack.**
- **D3 (head absorption) is not blocked; the argument is category‑wrong.** Prop. 1b reasons about `A ∈ GL(d)` acting on `z ∈ R^d`. The actual head is `W : R^{2048} → R^{512}`, a **rank‑reducing** map. Its principal degree of freedom is *which* 512 directions of `u` to keep, and selecting a subspace on which within‑identity shapes agree is head absorption that no `GL(d)` invariance argument touches. Worse, the penalty is only ever evaluated on **24 of 512** directions, chosen by the network's own `S`; identity‑specific covariance structure at ranks 25–512 is never measured. `sg[U]` makes this worse, not better: it removes the moving‑window gradient term, so the loss never sees the cost of shuffling its own measurement window. The Corollary's "*it must be produced by the backbone*" is therefore **false**.
- **Singular covariance:** `rank(Σ_A) ≤ 80 ≪ 512`. The projection saves executability, but `n/k = 3.3` (CUB) and the SOP null's `n/k = 1.5` are deep in the Marchenko–Pastur regime; at `γ = 0.67` pure noise spreads sample eigenvalues over roughly `[0.03, 3.3]`.
- **Too‑few‑samples / repeated images — verified fatal on two datasets.** SOP training set: min 2, **mean 5.3, std 3.0**, max 12 images per class. In‑Shop training set: **min 1**, mean 6.5, std 6.4, max 162. So `m = 5` is **unsatisfiable for a large fraction of SOP classes and for In‑Shop singleton classes**. The sampler must either repeat images — making `r_i` linearly dependent, `Σ_c` rank‑deficient, and the residuals from a duplicate‑contaminated mean non‑independent, which breaks *both* branches — or filter to classes with `≥ 5` images, which is an unstated recipe change and a biased class subset. In‑Shop classes with 1 image have `r_i ≡ 0` identically. The null (`⌊m/2⌋ = 2` per half) is unsatisfiable for any class with fewer than 4 distinct images.
- **D5 (split gaming)** was never a plausible attack: the split is drawn after the forward pass, so the network cannot condition on it. The defense is fine but defends nothing.
- **Radial/norm coding:** CINA‑Z measures covariance in the space where `‖z‖` is deployment‑invisible. The proposal flags this and offers CINA‑T, which is the right instinct, but the selection between them lands on a val split that cannot resolve it (see §7 below).

## 4. Is the target the right target?

**No — the Corollary is false, and it is explicitly "the whole design rationale."**

Prop. 1 itself is correct but trivial (`AΣ_cAᵀ = σ_c²I ∀c ⟺ Σ_c = σ_c²Σ`). The Corollary — *cosine can be made Bayes‑optimal for every identity by some linear head reparameterization **iff** the identity covariances share a common shape* — does not follow:

1. **Proportionality is necessary‑ish but not sufficient.** The Bayes test for same/different identity with *unknown* identities is the PLDA likelihood ratio (Ioffe 2006; Prince & Elder 2007), a bilinear‑plus‑quadratic form `z₁ᵀΛz₂ + z₁ᵀΓz₁ + z₂ᵀΓz₂ + const`. Whitening the within‑covariance to isotropy is WCCN, not Bayes‑optimality: the LR reduces to cosine only when `Λ ∝ I` *after* within‑whitening, which additionally requires the **between**‑identity covariance to be isotropic in that frame (plus norm control). CINA constrains nothing about between‑identity geometry. It targets a necessary condition and calls it "the exact obstruction."
2. **L2 normalization is not addressed by the linear‑algebra framing.** An invertible `A` is not an isometry of `S^511`; `normalize(Az)` is a projective, angle‑distorting map. The proposition lives in `z`‑space; deployment lives on the sphere.
3. **The Gaussian premise contradicts the intended base.** `Σ_c` is a sufficient statistic only if within‑identity distributions are unimodal Gaussian. PFML/SoftTriple‑style multi‑proxy models (the proposal cites "15 proxies/class") assert the opposite by construction. The §3 claim that CINA is "orthogonal to how the mean structure is parameterized" is wrong in sign: a 15‑proxy class model *is* a claim of identity‑specific multimodal within‑class geometry, whose second moment is identity‑specific by design. CINA and its own base are **antagonistic**, not orthogonal.

On identification of *useful* nuisance: within‑identity covariance measured on training images under ordinary augmentation is substantially augmentation‑induced (identity‑independent in input space by construction). The proposal never separates "identity‑specific nuisance that fails to transfer" from "arbitrary training‑image covariance," and has no measurement of `D` at initialization against which F3's "≥30% drop" could be evaluated.

## 5. Prior art

The statistic is **classical, and older than the proposal credits**: the proportional‑covariance model `Σ_i = ρ_i Ω` and its likelihood‑ratio test are Flury (1986) and, independently, Eriksen (1987), sitting inside Flury's (1984) Common Principal Components hierarchy; the LRT has no closed form and is solved by the FG algorithm. Box's M (1949) is the equality (not proportionality) test. The affine‑invariant/log‑Euclidean SPD geometry that makes `Σ(ν_i − ν̄)²` a quotient distance is Pennec et al. (2006) / Arsigny et al. (2006). The proposal concedes all of this and claims novelty *in the use*, which is the honest framing.

The residual novelty question is how far "in the use" is from: Deep CORAL (Sun & Saenko 2016); **Fishr** (Rame et al., ICML 2022), which matches second‑order gradient statistics across environments with an SPD‑matrix distance and is closer in spirit than anything in the §3 table; the IRM/REx family; and covariance/second‑order alignment in domain adaptation and re‑id. My targeted searches did not surface an exact "cross‑class covariance‑shape homogeneity regularizer for DML" precedent — but **I ran only a handful of queries under a hard budget cap, so I record novelty as `not established either way` and explicitly do not rest the verdict on it.** Novelty is not the decisive issue here; the mechanism and the estimator are.

## 6. Controls — several stated nulls are wrong

- **C3's stated null is a logical error.** `‖Σ_A/tr Σ_A − Σ_B/tr Σ_B‖²_F = 0 ⟺ Σ_A ∝ Σ_B` — **the identical zero set to `D`**. C3 and CINA optimize the same target under different geometries. So "if C3 matches, Prop. 1's mechanism claim is false" is backwards: C3 matching would *corroborate* the shared‑shape mechanism and falsify only the necessity of the affine‑invariant metric. The proposal's most emphasized decisive control cannot do what it is claimed to do.
- **C7 (label shuffle) is not magnitude‑matched.** With shuffled labels, residuals are taken about wrong means, `Σ` becomes ≈ the total covariance, both halves converge to the same thing, and `D` collapses. C7 therefore trains at a much weaker effective penalty. C7 failing to match is consistent with "generic regularization at a different strength," so its stated null ("kills generic regularization noise") is not established.
- **C4 is not statistic‑matched** for the same dof reason as the null (it *is* the null branch).
- **Missing control — the most likely confound.** Nothing controls for plain **within‑identity compactness / intra‑class variance regularization**, which is what CINA's gradient does through the "shrink the disagreeing directions" channel. C2 (whitening), C5 (ρ‑spectrum), C9 (epochs) do not cover it. Also missing: a **random `k`‑dim subspace** control, which is the only way to isolate the top‑24 measurement window from the mechanism.
- **F2 is under‑powered by the proposal's own numbers.** §6 states CUB/Cars seed variance ≈ 0.003–0.005; a paired 5‑seed sem is then ≈ 0.002, so "within 0.002" is at the measurement floor.
- **F6 is confounded by construction.** Correlating `D`‑reduction with R@1 gain "across ≥12 runs spanning seeds **and λ**" builds in a λ‑driven correlation that is not mechanism evidence. And `ρ = 0.5` at `n = 12` has a 95% CI of roughly `[−0.1, 0.83]` — the test cannot discriminate.
- **F3 is ill‑defined.** `D` is measured in each model's *own* top‑`k` subspace with its own noise floor; "held‑out‑identity `D` falls ≥30% vs base" compares two quantities measured in two different, model‑chosen windows.
- **F5's "effective rank" is undefined** (participation ratio? stable rank? entropy?), and as noted points the wrong way for the live degeneracy.
- C1, C8 (mandatory, correctly identified), C5, C9 are well‑posed. C6 is well‑posed and, as the proposal concedes, load‑bearing rather than confirmatory.

## 7. Forecasts, premises, frontier arithmetic

- **The crossing thresholds are arithmetically right** (`0.003/√5 = 0.00134`; `2√2·0.00134 = 0.0038` ⟹ CUB ≥ 0.738, Cars ≥ 0.931; `0.002/√5 = 0.00089` ⟹ SOP ≥ 0.832). Flagging the std/sem ambiguity rather than assuming the favorable reading is correct practice. `2·sem` is still not a valid `t`‑test at `n=5`, and it accounts for **no** selection over the 4 λ × 2 variants = 8 configurations.
- **No forecast derives from a measured premise.** `s*` is unmeasured yet the λ ceiling is a function of it; `D` at init is unmeasured yet F3 is a percentage drop from it; wall‑clock is a "target"; and **the base is unread**. PFML exists and is real (Bhatnagar & Ahuja, CVPR 2025, pp. 25549–25559; arXiv:2405.18560; the full PDF is on CVF and the arXiv abstract page resolves), so the retrieval failure was a session artifact, not an absent source — but the consequence stands: no reproduced matched base, so no frontier is inherited.
- **The fallback base cannot reach the frontier.** ProxyAnchor at ResNet‑50/512‑D is ~4–6 points below the stated CUB/Cars references. Tier 2 — the only tier that touches the frontier — is conditional on "the base reproducing PFML within 0.004," i.e. on a branch the proposer could not verify. Tier 1 (`+0.008 / +0.006 / +0.004`) is the number actually stood behind, and it is at or below the crossing thresholds.
- **Probability aggregation is internally inconsistent.** With `(0.42, 0.40, 0.35)` independent, `P(≥2 of 3) = 0.337`; the stated **0.28 is `P(exactly 2)`**, and positive correlation (a shared mechanism) pushes `P(≥2)` up, not down. Separately, the Tier‑1 80% CIs (`[+0.002,+0.015]`, `[+0.001,+0.012]`, `[0.000,+0.008]`) imply each marginal `P(Δ>0) ≥ 0.90`, so `P(Δ>0 on all three) ≥ 0.73` under independence — the stated **0.60 is below the product of the proposal's own marginals**. `P(cross | CUB) = 0.42` is also not reconstructible from the stated Tier‑2 mean/CI (which gives ≈0.68 at the 0.738 threshold, ≈0.44 at 0.742) without an unstated `P(base reproduces)` factor.
- The proposal states plainly that "the modal outcome is a real but sub‑frontier gain." Taken at its own word, this does not clear a frontier‑crossing bar even before the defects above.

## 8. Protocol, data, tuning, capacity

- **Legal:** deployment (one ResNet‑50, one view, 512‑D, cosine NN) is untouched; no test images, gallery statistics, transduction, reranking, generated data, or extra encoders; λ selected on train identities. No contamination found.
- **λ ceiling contradicts the frozen grid.** The stated ceiling `λ_max ≤ α·s*/(8·D_max) = 32s*/381.6 = 0.084·s*` also **disagrees with the proposal's own worked example** by 2× (its cost floor `(α/4)Δs` implies `λ_max ≤ 8s*/D_max = 0.168·s*`). With `s* = E‖r‖²` for unit‑norm `f`, realistically `s* ≈ 0.3–0.8`, so the admissible ceiling is ≈`0.025–0.067` (stated formula) or ≈`0.05–0.13` (its own arithmetic). The frozen grid `{0.05, 0.1, 0.2, 0.4}` therefore contains **at most one admissible value and possibly none** — and the D2 defense is void for the rest of the grid.
- **The selection split cannot resolve the effect being selected for.** 20 held‑out CUB identities ≈ 600 images ⟹ `sem(R@1) ≈ √(0.25/600) ≈ 0.018`, against a target effect of `+0.008`. Choosing among 8 configurations at that resolution is close to random, and it is also a regime shift (80 train identities at selection, 100 at report) — precisely the transfer failure ooDML documents.
- **Capacity/compute:** genuinely light; 0 extra parameters, views, or inference cost — the one part of the proposal that is unambiguously as advertised, modulo the omitted `d×d` eigensolve.
- `m = 9` on CUB/Cars is correctly identified as a real recipe delta with C8 mandatory. Good.

---

## Summary of false, inconsistent, or underdefined operations

**False:** Prop. 1 Corollary (proportionality is not sufficient for cosine Bayes‑optimality; the LR is a PLDA form requiring isotropic between‑identity structure, and `A` is not a spherical isometry); Prop. 1b as applied (the head is rank‑reducing 2048→512, so subspace selection is an unaddressed absorption channel — "must be produced by the backbone" is false); Prop. 2's "same rank/sample budget" (60 vs 80; 36 vs 72); D3's "blocked"; C3's stated null (identical zero set); the D2 cost floor's universality; §3's "orthogonal to how the mean structure is parameterized."

**Inconsistent:** λ ceiling formula vs its own worked example (2×), and vs the frozen grid; §6 cost accounting vs §1 step 4 (omits the 512×512 eigensolve); `P(≥2 of 3) = 0.28` vs marginals; `P(Δ>0 on all three) = 0.60` vs the stated CIs; a positive SOP forecast vs a specification that is inert on SOP.

**Underdefined:** whether `U` is recomputed for the null branch; the meaning and placement of the eigengap jitter; "effective rank" in F5; the measurement space (`z` or `f`) for F3/F4; the SOP/In‑Shop sampler's handling of classes with `<5` (and `<4`, and `=1`) images; whether λ‑selection runs retrain on 100 classes after selecting on 80; `k = 24` has no derivation and is not in the search.

**Honest limits of this review:** I did not run any of it, and my prior‑art sweep was 4 queries under a hard budget cap — enough to confirm the classical provenance (Flury/Eriksen proportionality LRT) and the SOP/In‑Shop class‑size distributions that make the sampler infeasible, but not enough to settle novelty against 2024–2026 primary sources. The verdict does not depend on the novelty question.

---

**Sources:** [PFML — CVF Open Access](https://openaccess.thecvf.com/content/CVPR2025/papers/Bhatnagar_Potential_Field_Based_Deep_Metric_Learning_CVPR_2025_paper.pdf) · [PFML — arXiv:2405.18560](https://arxiv.org/abs/2405.18560) · [PFML — IEEE Xplore](https://ieeexplore.ieee.org/document/11093291/) · [Flury's tests for common covariance structures (CPC package)](https://rdrr.io/github/tpepler/cpc/man/flury.test.html) · [A test for proportional covariance matrices — CSDA](https://www.sciencedirect.com/science/article/abs/pii/S0167947399000328) · [High-dimensional proportionality test of two covariance matrices](https://www.tandfonline.com/doi/full/10.1080/24754269.2021.1984373) · [SOP/In-Shop per-class statistics — Adaptive Cross Batch Normalization (arXiv:2303.17127)](https://arxiv.org/pdf/2303.17127) · [Non-isotropy Regularization (NIR), CVPR 2022](https://arxiv.org/pdf/2203.08547) · [ooDML generalization study](https://arxiv.org/pdf/2107.09562)

*(Unrelated: the claude.ai Gmail/Calendar/Drive/IBKR connectors need authorization via your claude.ai connector settings before they can be used — not needed for this task.)*

