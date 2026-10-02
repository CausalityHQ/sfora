# Completed research review through declared fallback

Consultatione824c230047e4784/group6272a38d847743e9; Opus5.5 OAuth session expired, no review; same leg automatically continued GPT-6.1 Sol/XHigh, actual exit0/720s. Static review only.

**Independent research / first-principles critic (Codex): GO-to-implement.** The fixed trial is scientifically defensible. Native execution still requires the frozen closure and reconciled reviews.

I reviewed `02e077ea` and the subsequent gate additions at `e47f1f8c`; the implementation files under review are unchanged. The additions address the previously unspecified stationarity, timing, and source-improvement claims.

1. **The objective tests supervised contraction, with no guarantee of better retrieval.** Raw prototype regression penalizes within-class variation, including variation in output norm. It does not directly optimize angular separation, negative rankings, or packed retrieval. Relevant prior art combines compactness with discrimination: [center loss](https://ydwen.github.io/papers/WenECCV16.pdf) accompanies softmax, while [supervised retrieval whitening](https://cmp.felk.cvut.cz/~chum/papers/Radenovic-ECCV16.pdf) uses matching and nonmatching covariance. These comparisons support the hypothesis but supply no evidence that this particular residual will improve quality. Keep the unchanged source and matched linear arm; no additional research grid is necessary.

2. **Member-inclusive prototypes are legitimate TRAIN supervision.** I verified the pinned partition hash and zero row/identity overlap among TRAIN, selection, and validation. The [existing partition predicate](/home/rb/worktrees/sfora-positive-causality/scripts/export_siglip2_genuine_views.py:160) additionally binds identity assignments to the manifest. Including a member attenuates its leave-one-out residual by `(n_c−1)/n_c`; singleton residuals are zero. That is a weighting choice, not held-out leakage. Preserve complete warm-source authentication and canonical label/order checks.

3. **Omitting the target intercept is justified.** With image-weighted, member-inclusive prototypes, each class satisfies
   \[
   \sum_{i\in c}(P_c-H_i)=0,
   \]
   so `mean(E)=0` in exact arithmetic. “No fitted intercept” still permits the coefficient-dependent offset introduced by centering the basis. Pass **uncentered V/E** to [fit_ridge_stitch](/home/rb/worktrees/sfora-positive-causality/src/sfora/foundation_adapter.py:505), retain `weight.T`, squeeze `source_mean` to `[32]`, and omit `target_mean` at inference. A small synthetic check with nonzero target mean should catch accidental use of `transform()`.

4. **FP32 is reasonable for this regularized system.** For nonzero positive-semidefinite `G`, the prescribed regularization gives the exact-arithmetic bound
   \[
   \kappa_2(G+\lambda I)\le1+32/0.1=321.
   \]
   This supports attempting FP32 without a precision rescue. The [updated gate](/home/rb/worktrees/sfora-positive-causality/docs/inshop_prototype_residual_gate_2026-10-02.md:13) supplies the missing stationarity and positive-lambda checks. Make its norms explicitly Frobenius norms and verify the actual finite system and independent coefficient agreement. Using `solve` directly is appropriate. [PyTorch documentation](https://docs.pytorch.org/docs/2.14/generated/torch.linalg.solve.html)

5. **The interpretation must remain narrower than production improvement.** Quadratic can beat a harmful linear correction by `0.002` while merely matching the original source. The updated requirement to report both source deltas and treat selection GO as permission for validation resolves that claim problem. Likewise, equal coefficient counts establish matched parameter budgets; diagonal squares in one frozen latent basis do not establish a general advantage for quadratic representations.

   The [pinned bootstrap](/home/rb/worktrees/sfora-positive-causality/scripts/score_inshop_crop_view_pair.py:78) resamples query-product contributions while retaining the gallery. Label its intervals as conditional on **the frozen source and fixed gallery**. They do not include source-training, gallery-sampling, or prior selection uncertainty. Preserve the shared 5,000 draws/179019 and sealed validation; repeated deterministic refits add reproducibility evidence only.

The smallest decisive path is the declared sequence: CPU120 qualification, independent linear300 and quadratic300 fits, evaluatorCPU120, then selectionscore300. Before selection cache access, require authenticated solver/source closure, corrupted-label/mean/source rejection, fresh reconstruction without reused sufficient statistics, strict reload parity, and both cost ratios ≤1.50. Compare actual fit-terminal service costs; sum both independently measured fit-core passes per arm and report shared qualification/scoring costs separately.

Replay every archived source query exactly at **96.30911188004614% R1 / 80.57229533585297% mAP@R**, then apply the frozen early KILL and final gain/interval gates. Only GO admits sealed validation. Public speed, full-TRAIN fitting, and official SOP/InShop confirmation remain unmeasured; the old cost KILLs remain final.

No files were edited, and no native work, quality calculations, children, consultations, or operator contact were initiated.
