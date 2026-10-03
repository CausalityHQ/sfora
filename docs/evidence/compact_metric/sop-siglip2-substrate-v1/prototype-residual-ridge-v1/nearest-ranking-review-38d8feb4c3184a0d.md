# Nearest-ranking engineering_critic review

Group f64090982b394ae0; completed exit 0; actual codex/gpt-6-astra . Astra completed independently.

**GO with explicit corrections for one bounded 32-update trial. Production release remains NO-GO.** The proposal offers a plausible route to a trained model, but its current TRAIN veto can reject a useful endpoint, and the existing checkpoint/readout machinery cannot be reused unchanged.

I inspected committed evidence at `61d17d4e`, including the complete proposal, selection decision/receipt, authenticated source implementations, runtime receipts, and TRAIN partition. The relevant current scripts’ SHA256 values match the selection receipt’s source pins. No files were edited or native execution, GPU work, tests, children, or consultations launched.

1. **High — Make the fixed-triplet screen diagnostic, not a model-usefulness veto.**

   The [proposal’s stop rule](/home/rb/worktrees/sfora-positive-causality/docs/evidence/compact_metric/sop-siglip2-substrate-v1/prototype-residual-ridge-v1/nearest-ranking-plan-6808efe4c21a4f52.md:118) requires fewer violations **and** a positive median margin difference on up to 128 initially selected triples.

   Neither condition is necessary for retrieval improvement. A model can improve margins without crossing 0.05, improve a minority of important queries while leaving the median unchanged, or replace the original positive with another legitimate nearest positive. Conversely, improving the frozen negative’s margin does not exclude a different image becoming the nearest impostor. Ascending original-row selection also provides no representative sampling guarantee.

   Re-encoding all three roles through each updated encoder is the correct diagnostic for gallery movement. Preserve that check, its frozen panel, and its position before selection. Record failure as **“mechanism not demonstrated on this panel”**; do not use it alone to deny the one frozen selection comparison.

   Similarly, zero violations among initial **teacher** anchors does not establish zero pressure for **native** anchors or after MSE updates. Stop for disconnected/nonfinite training or an invalid objective, not merely an empty initial diagnostic panel. Freeze this correction before execution; do not change thresholds after observing results.

2. **High — The differentiable wrapper must change validation contracts as well as remove detachments.**

   [The existing helper](/home/rb/worktrees/sfora-positive-causality/scripts/prototype_residual_readout.py:23) requires `A` to be a trainable leaf. Its forward then runs the source path under `no_grad`, detaches features and `h0`, and detaches `phi` before the correction. Freezing `A` makes the existing validator reject it; retaining its trainable status violates the proposed four-tensor inventory.

   The new procedure needs a frozen-`A` contract and a connected path through **both** `h0` and the concat correction. Frozen pooling, post-LN, and head parameters must still propagate gradients to the final MLP; “frozen” cannot mean wrapping those downstream operations in `no_grad`.

   Keep exact helper-versus-wrapper forward parity on identical inputs within each arithmetic role. Also verify the complete native forward used for training against the corresponding inference path.

   **Smallest decisive native falsifier:** within the first planned mechanics unit, use one authenticated scheduled B64 update, micro16. Demonstrate finite gradients and actual changes confined to the four named tensors, and separately demonstrate a nonzero ranking-gradient contribution when a valid hinge is active. Total-loss gradients alone could come entirely from MSE. Detach mining indices and teacher vectors, then compute the selected similarities from the connected anchor.

3. **High — Updated encoder ownership requires a new endpoint schema and loader.**

   The current [fitter reconstruction](/home/rb/worktrees/sfora-positive-causality/scripts/fit_siglip2_prototype_residual.py:735) owns an immutable encoder descriptor; [the underlying contract](/home/rb/worktrees/sfora-positive-causality/scripts/train_siglip2_quadratic_readout.py:520) explicitly requires the original checkpoint and all-frozen exported inventory. Existing reload reconstructs that source and restores fitted readout members. That is appropriate for concat ridge, but cannot certify an encoder update.

   Preserve original provenance separately from the complete updated vision state. The new checkpoint must own the four updated tensors, the byte-identical remaining 444 vision tensors, configuration, nonpersistent position buffer, fixed head/readout, and processor contract. Independent reload must load the updated vision strictly.

   Add one decisive mutation test: substituting original vision for updated vision must fail endpoint authentication. Verify reloaded native raw/unit/packed outputs against pre-save outputs. The [current evaluator consumes cached panel features](/home/rb/worktrees/sfora-positive-causality/scripts/evaluate_siglip2_prototype_residual.py:995); reusing that path would evaluate the old encoder.

   Create a **fresh four-member optimizer**. Do not restore the warm head’s five-member optimizer as active training state. Bind parameter names/order to moments and counters; compare the complete `17` versus `8+9` continuation, including scaler, RNG, schedule position, teacher tensors, and frozen bytes. Restore saved teachers rather than reconstructing them from updated vision.

4. **Medium — Resolve a real loss-reduction ambiguity before implementation.**

   The authenticated [TRAIN export receipt](/home/rb/worktrees/sfora-positive-causality/docs/evidence/compact_metric/sop-siglip2-substrate-v1/genuine-view-v1/export-source-v2/export-receipt.json) contains **12 singleton identities** among 1,008 classes.

   “Average over valid anchors” and “four equally weighted microbatches” can produce different ranking gradients. Implement the intended B64 valid-anchor average as each microbatch’s hinge **sum divided by the full batch’s valid-anchor count**. Regression remains averaged over all 64 anchors. Handle an entirely invalid batch without evaluating invalid positive indices or producing NaNs.

   Also pin the squared-error convention: `mean_i ||r_i−P_yi||²` sums across 128 coordinates before averaging images. Mixing that definition of `e0` with an elementwise MSE mean changes the ranking-to-regression weight by 128.

   A small fixture with unequal valid counts across microbatches should establish the accumulated gradient’s agreement with the B64 objective. Include self-exclusion, all same-class negative exclusions, original-row ties, singleton handling, and frozen-coefficient parity. Existing [prototype tests](/home/rb/worktrees/sfora-positive-causality/scripts/test_siglip2_prototype_residual.py:837) exercise algebra with stand-ins; they do not establish the new native gradient path.

5. **Medium — The causal question is legitimate but narrower than “encoder limitation.”**

   The comparison tests the incremental effect of nearest-wrong-class pressure during a fixed, short encoder adaptation. It cannot establish that cached features were inseparable or that the encoder caused the remaining errors.

   Best **other-image** positive is coherent with top-1 retrieval, and singleton regression-only handling is sensible. Frozen teacher mining is a bounded surrogate; its usefulness when the whole gallery moves remains unknown.

   Construct `P` from the accepted concat descriptors as proposed. Do **not** reuse the checkpoint’s old `prototypes`: [the fitter constructs those from the warm head’s `H`](/home/rb/worktrees/sfora-positive-causality/scripts/fit_siglip2_prototype_residual.py:638), before the concat correction.

   MSE control is not automatically degenerate: it targets distinct fixed class means. However, it can degrade retrieval. Therefore the preserved concat floor is essential, and both valid trained endpoints should be retained with their exact status. A failed candidate does not erase a potentially useful control.

6. **Medium — Source arithmetic and whole-unit feasibility remain unqualified.**

   The teacher cache was produced with native FP16 vision, FP32 pooled normalization, and B32+tail19; the fitter then performs CPU normalization and head arithmetic. The [export source](/home/rb/worktrees/sfora-positive-causality/scripts/export_siglip2_genuine_views.py:499) makes that sequence explicit. Native micro16 anchors are a separate numerical role.

   Freeze each role’s normalization sequence and flags. Require exact wrapper parity within roles; report zero-update teacher/native descriptor and margin differences on already-required TRAIN witnesses. Do not demand unjustified cross-role bit identity or silently redefine teachers to obtain it.

   There is no measured proof that the proposed units fit 300 seconds. The [concat fit receipt](/home/rb/worktrees/sfora-positive-causality/docs/evidence/compact_metric/sop-siglip2-substrate-v1/prototype-residual-ridge-v1/signed-concat-fit-concat-v1/receipt.json) records approximately **119.31 seconds of authority work** and **29 seconds of exit authentication**, despite only **0.322 seconds of fitting core**. The earlier [TRAIN100 timeout](/home/rb/worktrees/sfora-positive-causality/docs/evidence/compact_metric/sop-siglip2-substrate-v1/late-dense-v1/native256-train-large-179032-timeout-v6.json) completed all updates and final reload, then timed out during exit rehash. It is a different model/training path, not a throughput estimate for this trial.

   Mechanics costs **34 updates per arm**, plus reconstruction and reload. Measure those complete units before fresh training. The previous export’s host peak was about **7.72 GiB**; new ownership must avoid simultaneous original/updated model copies and uncontrolled checkpoint-page retention. This is a feasibility risk, not proof of OOM or timeout.

The exact dependency order should be:

1. **Freeze one corrected recipe and source closure.** Resolve the TRAIN diagnostic policy, reductions, arithmetic roles, updated-state schema, exact four optimizer members, and cost boundaries. Preserve the signed-concat procedure’s existing KILL decision.
2. **Run the single CPU engineering qualification, ≤500 seconds, CUDA hidden.** Authenticate complete source/runtime/data ownership; check teacher construction, miner fixtures, forward parity, malformed-state rejection, and checkpoint contracts. No quality evaluation.
3. **Run sequential discarded mechanics units, ≤300 seconds per arm.** Embed the one-update falsifier above, then complete `17` versus independent `8+9`, strict updated reload, and output parity. Use measured phase costs to establish whether fresh32 fits with save, reload, receipt, and exit authentication included.
4. **Run one fresh control32 and one fresh candidate32.** Same initialization, teacher bytes, first32 authenticated schedule, and numerical contract. Enforce DGX Spark only, both lifetime locks, 8 GiB host, zero swap/disallowed events, CUDA allocated below 10 GB, and original normal terminal exit. No precision, step-count, or hyperparameter rescue.
5. **Admit complete endpoints and matched fresh costs.** Require both whole-service and total-training-core ratios ≤1.50. Include target construction, decoding, mining, forward/backward, and optimizer work; report preparation separately. Historical ridge timings provide no denominator. Since control also mines for diagnostics, describe the ratio as instrumentation-matched training cost.
6. **Run the frozen TRAIN diagnostic, then one selection comparison for technically valid, cost-qualified endpoints.** Replay archived source metrics/wires; freshly encode candidate and control query/gallery images and independently verify packed scoring. Preserve both gains ≥0.002 fraction, both product-bootstrap 95% lower bounds >0, shared 5,000 draws/seed179019, query intervals, original failures/floors, candidate R1 above **96.4821222607%**, and mAP at least **81.7775403554%**.
7. **Only selection GO admits sealed validation.** Production acceptance still requires the unchanged SOP/InShop quality and matched end-to-end speed protocol, including B1/B32 serving qualification.

The [accepted decision](/home/rb/worktrees/sfora-positive-causality/docs/evidence/compact_metric/sop-siglip2-substrate-v1/prototype-residual-ridge-v1/signed-concat-selection-score-v1/decision.json) supports preserving concat: mAP improved, while the R1 gain and product lower bound failed the frozen gate. It supplies no encoder-training feasibility or public-speed evidence.

No authentication or model fallback was invoked in this review; this response does not certify completion of the separate pinned Opus/Astra pair.

