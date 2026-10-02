# Completed engineering review

Consultation36304e444d4c48df, group6272a38d847743e9; GPT-6 Astra/XHigh; exit0;335s; static only.

**Independent engineering/release critic: GO-to-implement, with bounded corrections before execution. No release GO.**

The proposed trial is scientifically defensible. I found no reason to introduce an optimizer, additional seeds, or a research grid. Four issues need explicit closure:

1. **P1 — Selection GO can occur without improving source quality.**  
   The gate allows quadratic to equal the source while beating a degraded linear residual. For example, quadratic could reproduce every source prediction while linear loses five R1 hits across different products and sufficient mAP; the candidate–linear thresholds and confidence bounds could pass. The [specified source floor]( /home/rb/worktrees/sfora-positive-causality/docs/inshop_prototype_residual_gate_2026-10-02.md:11) permits this.

   Preserve the frozen thresholds, but label GO as permission to continue validation. Report candidate–source deltas alongside candidate–linear deltas. Equality with source must not be presented as actual quality improvement; the production objective remains unmet.

2. **P1 — Ridge-helper reuse needs an explicit authenticated loading contract.**  
   The plan names `fit_ridge_stitch`, but the existing [origin audit](/home/rb/worktrees/sfora-positive-causality/scripts/train_siglip2_quadratic_readout.py:352) rejects loaded origins outside the accepted source/warm union. An ordinary import from the current checkout does not establish admissibility.

   Pin the helper’s exact bytes in the new closure and authenticate its loading path after source admission. Preserve original source predicates; do not broaden their historical authority or rebind helpers. Likewise, reuse applicable predicates individually: the existing [endpoint validator](/home/rb/worktrees/sfora-positive-causality/scripts/evaluate_siglip2_quadratic_readout.py:456) requires TRAIN1000, CUDA identities and mechanics replay, so it cannot validate a ridge endpoint.

3. **P1 — Numerical acceptance and fit-core timing are not yet reproducible specifications.**  
   The [plan’s `1e-5` stationarity threshold and `1.50` cost ratios](/home/rb/worktrees/sfora-positive-causality/docs/evidence/compact_metric/sop-siglip2-substrate-v1/prototype-residual-ridge-v1/completed-plan.md:63) lack exact normalization and timing boundaries.

   Freeze these before native work. One suitable stationarity definition is
   \[
   \frac{\|MW-B\|_F}{\|M\|_F\|W\|_F+\|B\|_F}\le10^{-5},
   \quad M=G+\lambda I,\quad B=\Phi^\top Y,\quad W=A^\top,
   \]
   with finite, positive denominator and independently reconstructed inputs. Separately reject nonfinite/zero energy, nonpositive/nonfinite λ, and nonfinite/zero coefficients. The [existing solver](/home/rb/worktrees/sfora-positive-causality/src/sfora/foundation_adapter.py:542) does not supply those procedure-specific checks.

   Define fit-core as a fixed contiguous interval covering normalization, head/basis evaluation, prototypes, both means, Gram/RHS construction and solve. Record authentication, reload and serialization separately; include everything in whole-service cost. Use identical boundaries and prescribed fit counts for both arms.

4. **P2 — The helper interface has concrete shape and centering traps.**  
   `fit_ridge_stitch` already centers both inputs and returns `source_mean` shaped `[1,32]`; the [readout requires `[32]` means](/home/rb/worktrees/sfora-positive-causality/scripts/quadratic_readout.py:74). Pass **uncentered `V` and `E`**, retain `weight.T`, explicitly remove the singleton mean dimension, and compare against both independently fitted TRAIN means. Passing the recipe’s already centered `Phi/Y` would center again and change FP32 results.

   Keep `target_mean` out of inference. The [existing ridge test](/home/rb/worktrees/sfora-positive-causality/tests/test_foundation_adapter.py:229) tests recovery **with** an intercept; it does not verify this procedure.

The mathematical choices themselves are reasonable:

- Member-inclusive class prototypes are legitimate supervised TRAIN targets. The [existing partition predicate](/home/rb/worktrees/sfora-positive-causality/scripts/export_siglip2_genuine_views.py:172) checks both row and identity disjointness; preserve it and the authenticated dense-label ordering.
- With image weighting, each class’s residuals sum to zero in exact arithmetic. Omitting the target-mean intercept is therefore coherent; FP32 centering remains part of the fixed recipe.
- For nonzero positive-semidefinite `G`, the prescribed regularization bounds the exact-arithmetic condition number of `G+λI` by **321**. FP32 is a reasonable starting point; no precision search is justified.
- The unchanged bootstrap resamples product groups of fixed per-query deltas. Its intervals are conditional on this source, gallery and scoring protocol. Exposed selection remains exploratory; neither repeated refits nor query intervals establish training-seed or new-gallery population uncertainty.

The smallest meaningful executable check should cover unequal class sizes, discarded nonzero target mean, rank-deficient and zero-energy bases, and rejection of altered labels, means and source bytes—including unchanged-version mutation. Independent refits must rebuild statistics from authenticated raw inputs. Updated checkpoint reload must verify raw/unit/int8/inverse-norm bits. Existing [AST-only checks](/home/rb/worktrees/sfora-positive-causality/scripts/test_quadratic_readout.py:119) cannot establish those numerical properties.

**Fast falsifier:** freeze the corrected contract, then run CPU120 → linear fit300 → quadratic fit300 → evaluator CPU120 → selection score300 sequentially. Close engineering and cost admission before selection access. Replay the archived source wires and every query exactly, then apply the single frozen paired gate. Any failure closes this trial without cap extensions, retuning or partial reuse. Only accepted selection GO admits sealed validation; public paired p99, full TRAIN and official quality remain separate gates.

Static review only: no files edited, tests or native jobs run, or original remote artifacts authenticated. The earlier TRAIN1000, residency and serializer cost KILL decisions remain intact.
