# Pass 79 GCNP independent review (verbatim)

 I have what I need. I did not edit the proposal and ran no GPU work.
 
 ---
 
 # Verdict: **DEAD**
 
 Dead independently at **Gate 1** (premise measured in-repo and contradicted, at exactly the proposal's own parameter), at **Gate 2** (the novelty claim collapses into its own control arm by Jensen, and the operator is occupied), and on **frontier arithmetic**. Any one is dispositive under `docs/search_protocol.md`.
 
 ## Single strongest reason
 
 **GCNP's F1 has already been run in this repository, in a stronger form, at k = 32, and it failed.**
 
 `docs/candidate_225_nuisance_transfer_preregistration_2026-08-02.md` (quoted in `docs/pass77_nam_review_2026-08-07.md:22`) estimated the pooled within-class covariance on one In-Shop identity fold and evaluated its leading-`k` subspace on a **disjoint** identity fold — three digest-bound seeds, both fold directions, Haar-random and label-permuted nulls, thresholds locked before the numbers were read. At `k=32`: the source-fold within-class subspace captured **35–37% of target within-class variance and 38–40% of target between-class variance**.
 
 GCNP's F1 metric is *removed-within-class-energy ÷ removed-between-class-energy*, abandon if **< 1.3**. The measured value is **≈ 0.36/0.39 ≈ 0.92** — not merely below 1.3, but **below 1.0**: on identities the estimator never saw, the consensus within-class subspace carries *more* identity signal than nuisance. This inverts the causal mechanism in §"Causal mechanism" ("for unseen classes those directions inject raw noise"). It is the same premise that killed NAM one pass ago and CINA (22), EFML (24), FRAME (27), HIRE (32), CNW (40) before it.
 
 *Applied symmetrically, per `RSPG_SPECDEFECT.md`:* candidate 225 is In-Shop, epoch 10, arithmetic-pooled. Two of those three gaps close below; the third (dataset) does not, and GCNP names Cars196 primary — but Cars196 **has no corrected-recipe baseline in this repository at all** (`HANDOFF_BRIEF.md`, "Cars196, which has never been run under corrected recipes"; `docs/pass77_nam_review_2026-08-07.md:56`, "no repo base under corrected recipes"). So the premise is contradicted on the dataset the protocol makes decisive and unmeasured on the one the proposal elects primary.
 
 ## Gate 2 — the novelty claim is self-annihilating (this is the load-bearing math)
 
 The proposal isolates its entire contribution in one sentence: log-det consensus, *not* the arithmetic pooled covariance, and F3 concedes null-novelty if arm-4 matches arm-3.
 
 `log det` is concave on the PSD cone and `A ↦ UᵀAU` is linear, so by Jensen, for **every** `U ∈ St(512,r)`:
 
 $$J(U)=\tfrac1C\textstyle\sum_c \log\det(U^\top\tilde\Sigma_c U)\ \le\ \log\det\big(U^\top\bar\Sigma U\big),\quad \bar\Sigma=\tfrac1C\textstyle\sum_c\tilde\Sigma_c$$
 
 with equality **iff all `UᵀΣ̃_cU` are equal**. The right-hand side is maximized (Poincaré separation) by the top-`r` eigenvectors of `Σ̄` — which is *exactly* arm 4, i.e. NAP / Radenović-style supervised whitening's suppressed subspace. The pointwise gap is precisely the cross-class heterogeneity of the covariances restricted to `U`.
 
 So the proposal is caught in a scissors:
 
 - If a subspace exists "along which **every** training class spreads" (the stated premise, §"One-line"), the `UᵀΣ̃_cU` are near-identical there, the Jensen gap `η ≈ 0`, and the two maximizers coincide to `O(η)` → **arm 3 ≡ arm 4, F3 fires, null novelty.**
 - If the gap is large, the covariances are heterogeneous, there is no consensus subspace, and the causal story ("noisy for *every* class") is false.
 
 The "log diverges to −∞" protection — the sole stated novelty mechanism — is then **deleted by the proposal's own regularizers**. `Σ̃_c=(S_c+εI)/tr(S_c+εI)` floors every eigenvalue at `ε/tr`, so `log det ≥ r·log(ε/tr)`: bounded, with protection strength set entirely by an unspecified `ε`. On In-Shop it is worse: 25,882 train images / 3,997 classes ≈ **6.5 images/class**, so `rank(S_c) ≤ 5` in `d=512`, and Ledoit–Wolf's optimal intensity at `n/d ≈ 0.013` shrinks toward the pooled covariance with `ρ ≈ 0.99`. `Σ̃_c → Σ̄` ⟹ `J(U) → log det(UᵀΣ̄U)` ⟹ **arm 3 is arm 4 to ~1% on the only dataset with a verified four-seed reference.** The proposal itself mandates that shrinkage (failure mode 3). Its "stress test" is where its novelty provably vanishes.
 
 **Occupied operator.** The verdict ledger already records this family: entry **23** ("Interventional differencing nuisance residualization: occupied by NAP", `docs/method_search_verdict.md:945`) — Solomonoff, Campbell & Quillen, Nuisance Attribute Projection (2004–2007): *estimate within-identity nuisance directions by an eigenproblem, orthogonally remove them, compare identity in the retained space*, with "weighted and nonlinear NAP variants cover[ing] covariance weighting and learned feature spaces." Entry **372** (`:4789`) killed an EMA/shrinkage/backprop-through-covariance whitening proposal with the ruling that "shrinkage, EMA estimation, and backpropagation through the covariance alter the estimator and optimization, **not the method mechanism**" — which is verbatim the four differences GCNP claims. Add WCCN (Hatch et al., Interspeech 2006), RCA, entries 176 and 312, and CNW (pass at `:6514`). Radenović TPAMI 2018 is correctly identified as highest risk but under-weighted: with `γ=1`, hard rank-`r` removal is the `α→∞` limit of Lw along its lowest-eigenvalue directions, and the repo's own post-hoc whitening sweep on CUB (`docs/opus_pass60_none_2026-08-06.md`, `:7834`) measured `α = .0/.1/.25/.5/.75/1.0 → R@1 .6818/.6826/.6870/.6913/.6828/.6676` — a peak at partial suppression and **−1.42 pt at full suppression**. GCNP's declared expected-best config (`γ=1.0`) sits past the measured turning point.
 
 **Roth ICML 2020 is adverse, not adjacent.** ρ-spectral evidence is that *higher* embedding spectral entropy predicts better transfer. Zeroing 32 of 512 directions lowers it. The proposal reads Roth as a differentiation risk; it is closer to a contrary measurement in the same benchmark family. Deep LDA and INLP are fairly characterized.
 
 ## Unaddressed dynamical defect: the estimator is self-referential
 
 At `γ=1` the loss depends on `z` only through `(I−UUᵀ)z`, so `∂L/∂z` has **exactly zero component in `span(U)`**. Yet `U` is refreshed from EMA statistics of the **unprojected** `ẑ`. The criterion is therefore evaluated on a component the objective has been made blind to, and the fixed point is decided by weight decay, not by nuisance:
 
 - with decay on the head, `W`'s `U`-rows decay → within-class variance along `U` collapses → `log det → −∞` → `U` must rotate each refresh (failure mode 2 is *predicted*, not merely risked, at the declared best config);
 - without decay, the `U`-component is an unconstrained passive readout whose variance is maximal *because* it is unconstrained → `U` is self-confirming and detects nothing.
 
 Neither branch is "the head reallocates capacity into the complement." The 20°-principal-angle diagnostic detects the first branch and is blind to the second. (Argument, not measurement — flagged as such.)
 
 The gradient algebra `∇_U J = (2/C)Σ_c Σ̃_c U(UᵀΣ̃_cU)^{-1}` is correct, as is the `min_g cos_g ≤ mean_g cos_g` reasoning that discarded the polyspherical axis.
 
 ## Gate 1 — provenance of every quoted number
 
 **Zero repository measurements are cited.** Every entry in the forecast table is a forecast, and the lane is wrong: the proposal specifies **ResNet-50, 512-d, 224²**, while the audited controlled lane is **BN-Inception/512-D on the corrected 256-pixel corpus** (`docs/inshop_corrected_reference_seed3_result_2026-08-06.md`, which explicitly warns that the legacy `proxy-anchor-resnet50-512` protocol-family string is *misleading*). The proposal appears to have inherited exactly the string the repo flagged. F1's "100 training classes" is CUB's count; Cars196 has 98 — a second provenance tell.
 
 **In-Shop "~91.9 ± 0.2" is not a repo number.** The corrected four-seed reference is raw best-over-training **0.9181495 ± 0.0012560** and frozen-final **0.9153889 ± 0.0013196**; the 0.276-pt raw-to-final gap is checkpoint-selection optimism. `91.9` grants a phantom **+0.36 pt** head start over the frozen-final base and silently adopts the raw convention the protocol requires to be reported alongside, not instead of, final.
 
 ## F1–F4 validity
 
 | | Verdict |
 |---|---|
 | **F1** | **Invalid and already failed.** It splits *training* classes 75/25 on a model trained on all of them, so the 25 are seen identities where proxies *have* compressed — it cannot test a claim about unseen classes. Candidate 225 ran the protocol-clean version (disjoint identity folds, both directions, nulls, locked thresholds) and returned ≈0.92 against GCNP's 1.3 threshold. |
 | **F2** | **Unrunnable as costed.** Cars196 has no corrected-recipe baseline; establishing one precedes the falsifier and is not in the 58-run budget. |
 | **F3** | **Underpowered, and fires by construction.** A 0.2-pt equivalence margin "on all datasets" is inside noise on CUB (6-seed sd 0.367 ⟹ SE of a 5-seed difference ≈ 0.23), and `HANDOFF_BRIEF.md` discipline #7/#3 records that the sd estimate itself is unstable (n=3 → 0.153, n=6 → 0.367). Independently, the Jensen and Ledoit–Wolf arguments predict arm 3 ≡ arm 4 whenever the premise holds. |
 | **F4** | **The only sound falsifier**, and the one most likely to fire: the proposal's own export identity `W' = (I−γUUᵀ)W` concedes the deployed object *is* a post-hoc linear projection; the CUB α-sweep shows the post-hoc operator already delivers +0.95 pt at α=0.5. |
 
 Model selection via leave-class-out folds inside the training split is **clean** and materially better than NAM's — no test-split leakage found. Compute matching, the search-budget-matching demand, MAP@R reporting, and data legality all check out.
 
 ## Frontier arithmetic
 
 Against the corrected four-seed In-Shop reference (mean **0.9153889**, SD **0.0013196**, four-seed power floor **0.9185**, i.e. +0.31 pt):
 
 - Nominal +0.7 → 0.9224 — clears.
 - **The proposal's own selection correction (−0.3 to −0.5) → +0.2 to +0.4 → 0.9174–0.9194**, straddling 0.9185.
 - Corrected 90% interval [−0.3, +0.9] → **includes zero**. The In-Shop arm cannot resolve its own forecast.
 
 None of the three arms crosses an audited Lane-A horizon even at the uncorrected optimum: CUB 0.697 vs 0.734; Cars 0.887 vs 0.927; In-Shop 0.922 vs 0.930.
 
 ## Exact kill gates (what a revival would have to clear, in order)
 
 1. **Gate 1, ≤1 CPU-hour.** Re-run candidate 225's disjoint-identity-fold protocol on the corrected In-Shop packs with the **log-det consensus** subspace substituted for the pooled one, thresholds locked in advance, both fold directions, ≥3 digest-bound seeds, Haar-random and label-permuted nulls. Required: removed-within ÷ removed-between **> 1.3** at r = 32. Measured today at 0.92 for the arithmetic estimator. **This gate must be passed before any GPU.**
 2. **Gate 2, no GPU.** Report `η = max_U [log det(UᵀΣ̄U) − J(U)]` on those same packs, plus the Ledoit–Wolf intensity actually selected. If `η` is small, or `ρ > 0.5`, arm 3 is arm 4 and the pass is a re-derivation of NAP/WCCN/Lw — record it as such.
 3. **Gate 3.** State the deployment lane as BN-Inception/512-D on the corrected 256-px corpus, and re-quote all three baselines as repo measurements with recipe digests, raw **and** frozen-final. Cars196 cannot be primary until it has a corrected-recipe reference.
 4. **Gate 4.** F4 before F2: post-hoc projection on the existing frozen seed-0–3 In-Shop checkpoints costs no training and settles the train-time claim. The CUB α-sweep predicts it recovers most of the gain and that `γ=1` is past the optimum.
 5. **Frontier.** Any In-Shop forecast must be **selection-corrected before** comparison and must exceed **0.9185** at four seeds on that corrected number.
 
 Nothing here is repairable by tuning: gates 1 and 2 are a measurement that already exists and a Jensen inequality.
 
 I have not written this to `docs/` or updated `method_search_verdict.md`. Say the word and I'll record it as `docs/pass79_gcnp_review_2026-08-07.md` plus a ledger entry, per the protocol's "when a candidate dies" rule.
