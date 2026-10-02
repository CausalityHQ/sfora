# engineering_critic

Group f418ef478c664414; consultation 7f20bc0a26574805; completed model gpt-6-astra; exit 0.

**GO for bounded implementation. NO-GO for native execution until the new qualification passes.** I found no design blocker requiring another method or architecture.

**Compositional CPU qualification is valid here.** The [accepted source proof](/home/rb/worktrees/sfora-positive-causality/docs/evidence/compact_metric/sop-siglip2-substrate-v1/late-dense-v1/native256-source-cpu-so400-v4.json) covers all 448 vision tensors, config, processor, roles and the nonpersistent position buffer. The [accepted cache export](/home/rb/worktrees/sfora-positive-causality/scripts/export_siglip2_genuine_views.py:559) independently reconstructs the encoder and replays outputs. The existing [genuine CPU qualification](/home/rb/worktrees/sfora-positive-causality/scripts/train_siglip2_genuine_views.py:857) already composes inherited encoder evidence with fresh cached-head qualification.

Therefore, four new vision factories are unnecessary for this **cached-readout certificate**. Reuse requires an explicit, complete proof-to-source-to-cache mapping and current byte authentication; it cannot confer a new public-encoder certificate.

Before freezing implementation, resolve these concrete hazards:

1. **P1 — Existing entry points silently change the experiment.** [`head_from('candidate')`](/home/rb/worktrees/sfora-positive-causality/scripts/train_siglip2_cached_readout.py:378) selects GELU; [`view_inputs(..., 'candidate')`](/home/rb/worktrees/sfora-positive-causality/scripts/train_siglip2_genuine_views.py:512) selects augmented rows. Both existing optimizers train five tensors. Both new arms must retain the original **control** factory and canonical inputs, with exactly one FP32 optimizer member, `A[128,32]`. Freeze primary/down/up, classifier and all buffers explicitly. The original source constructor’s 205 trainable-role flags are provenance; the accepted exporter’s **all-448-frozen** mapping is the relevant encoder allocation.

2. **P1 — Freeze machine arithmetic for the means, not just the formula.** [`training_features`](/home/rb/worktrees/sfora-positive-causality/scripts/train_siglip2_genuine_views.py:555) normalizes cached rows on CPU; the head normalizes again internally. [`cache_rows`](/home/rb/worktrees/sfora-positive-causality/scripts/evaluate_siglip2_genuine_views.py:732) passes original FIT panel rows directly to that head. Preserve each path independently. Specify mean-fit device, FP32 operations, row order, batch layout and reduction order before execution. Fit each canonical TRAIN row once; use `mean(z.square())`, not `mean(z).square()` or centered-square features. Do not divide by `preactivation_std` again or feed already internally normalized `u` back through `head.forward`. Both export receipts record FP32 vision/F16 autocast, but that does **not** establish cache-byte equivalence.

3. **P1 — The fresh CPU proof must exercise an updated state.** Authenticate the entire warm payload before extracting members, following [complete warm admission](/home/rb/worktrees/sfora-positive-causality/scripts/train_siglip2_postln_adaptation.py:415). Preserve its stale bank initially; refresh it only through the frozen pre-update/last-duplicate rule. For each arm, require:
   - Initial learned-source/control/candidate raw, unit, int8 and FP16 inverse-bit equality.
   - Actual finite, nonzero **data** gradient for A on the fixed first B64/micro16 objective; successful A movement; no frozen-member gradients or byte changes.
   - Independent fresh-head reconstruction after a real update, restoring A, both means, complete base/head buffers, classifier, bank, typed optimizer moments, scaler, schedules/masks, counters, RNG and numerical flags.
   - A complete logical-state identity binding the unchanged encoder’s authenticated inventory/checkpoint to the new dynamic payload. JSON presentation must not replace typed-state hashing.

   Preserve current-use hashes and fresh uncached exit checks. Version counters alone cannot detect `.data` mutation. Historical source proofs remain complete dependencies, not permission to omit vision/config/buffer predicates.

4. **P1 — Reusing the old quality gate misses the source floor.** [`first_gate` and `quality_gate`](/home/rb/worktrees/sfora-positive-causality/scripts/evaluate_siglip2_genuine_views.py:338) compare candidate against control only. Add the frozen source061 nonregression predicate explicitly, with exact original baseline replay. On sealed validation, compare against source061 on **that validation panel**, after selection authorization—not against selection’s numeric baseline. Reuse scoring mathematics without the existing scorer’s helper-global rebinding.

The smallest useful falsifier is the new CPU unit: authenticate dependencies, construct both fresh heads, fit/reconstruct means, execute the first real objective/update, then compare uninterrupted step2 with independently reloaded step1→2, including complete state and raw/unit/packed bytes. Pair this with narrow rejection checks for wrong factory, augmented inputs, extra optimizer membership, mutated means/frozen bytes and wrong typed state. **Only a normal terminal including full exit authentication within CPU120 qualifies it.** Resource fit remains unknown; subtracting factory timings from the failed norm run proves nothing.

Then require discarded17 versus8+9 for both arms and fresh TRAIN first17 replay, under the unchanged resource limits and locks. The first061 matched pair remains the quality falsifier; failure closes this recipe before069/validation.

A survivor still needs the actual authenticated So4001152→128 factored public path: the [current loader](/home/rb/worktrees/sfora-positive-causality/src/sfora/siglip2_compact_serving.py:110) rejects it. No unqualified folding, public-speed inference or official-quality claim follows from cached wires. The GELU negative supplies no positive evidence for quadratic readout.

Static review only; no files changed or native tests executed.

