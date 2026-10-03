# Nearest-ranking research_critic review

Group f64090982b394ae0; completed exit 0; actual codex/gpt-6.1-sol . Pinned Opus attempt failed OAuth authentication; the recorded runner fallback completed under Sol. This is not an Opus review.

**GO with explicit corrections for one bounded encoder trial.** The recipe is a plausible way to produce a useful trained model. It is not ready for native launch until its new state and numerical contracts are implemented and qualified. Quality, convergence within 32 updates, and whole-unit runtime remain unknown; the committed evidence does not establish their failure.

I reviewed commit `61d17d4e`, including the full proposal, selection decision, receipt structure and relevant inventories, both requested scripts, and authenticated encoder/export/TRAIN sources. Static hashes matched the receipt’s fitter, readout, genuine-view trainer/exporter, source qualifier, and partition pins. No files were edited or native experiments run.

1. **Keep the causal claim narrow.**

   The [decision receipt](/home/rb/worktrees/sfora-positive-causality/docs/evidence/compact_metric/sop-siglip2-substrate-v1/prototype-residual-ridge-v1/signed-concat-selection-score-v1/decision.json) supports preserving concat: its mAP@R gain is +0.217187 percentage points, while its +0.057670-point R1 gain fails the frozen gate.

   That result does not identify the encoder as the bottleneck. Absence of cached-separability evidence establishes neither inseparability nor an encoder cause. This trial tests whether **adding nearest-impostor supervision to this particular four-tensor adaptation improves retrieval**, conditional on the accepted concat initialization and fixed readout.

   Repository prior art already supplies classification and valid-anchor ranking supervision. The proposed distinction is the new miner, frozen teacher gallery, and final-MLP adaptation. Treat it as a bounded objective comparison, without requiring another cached readout experiment or architecture search.

2. **Separate original encoder provenance from updated model ownership.**

   This is the principal concrete implementation correction. The accepted concat payload’s `encoder` member describes an authenticated external immutable encoder. It does not contain trained vision tensors. The [original ownership checks](/home/rb/worktrees/sfora-positive-causality/docs/evidence/compact_metric/sop-siglip2-substrate-v1/quadratic-readout-v1/train-source-v7/train_siglip2_quadratic_readout.py:519) explicitly require all 448 inventory entries to remain frozen.

   The new payload must therefore own the complete updated vision state, retaining original provenance separately. Verify that exactly the four named MLP tensors may change and the other 444 parameters, nonpersistent position buffer, configuration, pooling/post-LN, warm head, classifier, concat coefficients, and fitted means remain unchanged.

   The [source factory](/home/rb/worktrees/sfora-positive-causality/scripts/qualify_siglip2_substrate_cpu.py:357) initially enables 205 parameters. Reusing that factory requires freezing everything and enabling only the four authorized tensors **before creating the optimizer**. Otherwise the intervention differs from the proposal.

   Independently construct and strictly reload the updated endpoint. Verify its complete state and raw/unit/packed outputs. Export an inference artifact that loads the updated encoder and fixed concat readout without needing training caches or teacher targets.

3. **Qualify the actual objective and gradient path.**

   The existing [readout helper](/home/rb/worktrees/sfora-positive-causality/scripts/prototype_residual_readout.py:57) blocks encoder gradients through `no_grad` and detachments. The proposal correctly requires a differentiable counterpart. Preserve operation order, normalization, `.float()`, and FP32 readout arithmetic; an algebraically equivalent folded expression is insufficient for exact-forward claims.

   There is also a validation-contract mismatch: the authenticated helper requires `A` to be a trainable leaf, while this trial freezes it. Keep that helper unchanged as the forward oracle, using an isolated coefficient copy satisfying its contract. The actual training coefficient must remain frozen and outside AdamW.

   Reconstruct teacher `T` from the accepted **concat output**, then compute the new class means `P`. Do not reuse the checkpoint’s stored `prototypes`: the [fitter constructs those from the pre-correction warm-head output](/home/rb/worktrees/sfora-positive-causality/scripts/fit_siglip2_prototype_residual.py:618).

   Require finite positive `e0`, but do not interpret it as proof that the control has useful gradients. On actual scheduled native anchors, qualify finite gradients and real updates in all four tensors. Separately demonstrate that an active ranking term contributes a nonzero encoder-gradient difference; a nonzero total candidate gradient could come entirely from MSE.

   Preserve self exclusion, all-same-class negative exclusion, original-row tie breaking, and singleton handling. Define ranking normalization across the entire B64 update: summing valid-anchor losses and dividing by the total valid count avoids unintentionally weighting microbatches differently when their valid counts differ.

   Use fresh, empty four-tensor AdamW state in both arms. Do not inherit the warm head optimizer’s moments or counters. Resume qualification must cover parameter-to-moment ownership, scaler, update counters, schedules, CPU/CUDA RNG, teacher targets, and numerical flags—not only final weights.

4. **Keep teacher arithmetic and live-gallery arithmetic distinct.**

   The [canonical export](/home/rb/worktrees/sfora-positive-causality/scripts/export_siglip2_genuine_views.py:499) used native FP16 autocast, GPU normalization, B32 plus tail19, and persisted normalized FP32 features. The accepted teacher then uses CPU arithmetic and the original TRAIN normalization path. Training uses live micro16 images.

   These are different numerical contexts. Pin their normalization sequence and native CUDA flags explicitly. Require exact wrapper/oracle parity on identical inputs within each role; do not demand or claim cross-context bit equality without evidence. Use existing bounded TRAIN witnesses to record initialization drift from cached teacher descriptors.

   Best-other-positive mining is defensible for R1: at least one genuine gallery image must outrank the impostor. It is less directly aligned with mAP@R, and the best TRAIN positive may be absent from the production gallery subset. Hardest-positive or live-gallery training would change the objective and add cost; neither is necessary for this trial.

   The MSE control is a valid shared contraction objective, not an untreated baseline. Preserve the concat floor and disclose that +0.20 points over this control does not necessarily mean +0.20 points over preserved concat.

5. **Make the TRAIN triplet screen diagnostic rather than a model-utility veto.**

   Keep its frozen panel, original ordering, and position before selection. Re-encoding all three roles through each updated model is useful evidence about transfer from frozen teacher mining to a moving gallery.

   Its proposed stopping conditions are not decisive utility falsifiers. Margins can improve without crossing 0.05; the initially selected impostor can cease to be nearest; and the first 128 ordered anchors need not represent selection errors. Consequently, unchanged violation counts or a nonpositive median on these fixed triples can coexist with improved retrieval elsewhere. Conversely, passing this screen does not exclude new live-gallery impostors.

   **Prospectively remove its automatic pre-selection KILL.** Report both measures unchanged and label failure as failure to support the proposed mechanism. Do not remine the panel or tune afterward. A valid, resource-qualified endpoint should receive its single frozen selection comparison regardless of this diagnostic’s sign.

   Retain hard stops for integrity failure, nonfinite computation, broken gradient ownership, and resource violations. A teacher-only finding of zero active anchors should not alone establish zero native training pressure: numerical drift and MSE updates can change activation. Resolve objective activity during the already-required mechanics run.

The exact ordered path should be:

1. **Freeze the implementation and contracts:** accepted concat/source closure, TRAIN rows and targets, teacher construction, differentiable forward, four-tensor inventory, optimizer/RNG state, diagnostic interpretation, unit boundaries, timers, and unchanged quality thresholds.
2. **Run one CPU qualification unit, ≤500 seconds, CUDA hidden, without quality access.** Establish source/layout ownership, forward parity, miner edge cases, frozen-state mutation rejection, and serialization contracts.
3. **Run discarded DGX Spark mechanics sequentially for both arms:** uninterrupted17 versus independently restored8+9, ≤300 seconds per arm. Verify full updated-state replay, gradient ownership, objective activity, image/pixel identity, and raw/unit/packed reload parity. Record construction, target building, updates, serialization, reload, and exit-audit times.
4. **Only after those passes, run two fresh 32-update endpoints**, ≤300 seconds each. Do not reuse mechanics state. Measure candidate/control whole-service and training-core ratios against these fresh matched runs; both must be ≤1.50. Include target construction, mining, decoding, forward/backward, and optimizer work in the declared core. Report preparation separately.
5. **Independently reload both endpoints; run the frozen TRAIN diagnostic, then one native selection export and score.** Updated descriptors must come from updated image encoders, never the original feature cache. Freeze export/scoring unit boundaries before execution. Replay archived source wires and metrics first.
6. **Apply the unchanged selection gate:** both gains ≥0.002 fraction, both paired-product 95% lower bounds >0, 5,000 shared draws/seed179019, existing immediate failures/source floors, and preserved concat R1/mAP@R floor. Only selection GO admits sealed validation. Official SOP/InShop quality and matched end-to-end serving speed remain subsequent requirements.

Maintain both lifetime locks, 8GiB host limit, zero swap/disallowed events, whole-unit CUDA allocated below 10GB, normal terminal exit, and uncached exit authentication throughout.

**The smallest decisive native readiness falsifier is the required mechanics run.** It can reject this implementation for ineffective gradients, incorrect reload, or resource failure without consuming selection exposure. The [earlier 300-second timeout](/home/rb/worktrees/sfora-positive-causality/docs/evidence/compact_metric/sop-siglip2-substrate-v1/late-dense-v1/native256-train-large-179032-timeout-v6.json) completed all 100 updates and timed out during exit authentication. That proves update timing alone is insufficient; it does not prove this smaller So400 trial infeasible.

Preserve concat and every valid updated endpoint even if this procedure fails. These corrections provide a direct route to trained models without adding grids or further qualification campaigns.

Reviewer provenance: this is the current Codex-session critique. The exact pinned reviewer model and completed Opus5.5/Astra pair are not exposed to me; I cannot certify that pair. No consultation or authentication fallback was invoked.

