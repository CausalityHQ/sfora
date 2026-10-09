**Verdict: CONDITIONAL for one cache-based asymmetric method falsifier. REJECT the proposed CPU-only execution and direct VAL continuation.** This can test whether archived candidate queries work better against a specifically defined cached gallery. It cannot yet isolate encoder freshness as the cause.

The symmetric MLP and pooling-probe decisions remain **KILL**.

1. **The control-equivalence blocker is real.** V3 completed zero counterfactual cells. Its control061 comparison found 1,902 raw-byte differences and ten wire-byte differences, all within query indices 1728–1733. That localizes the discrepancy; it does not prove batching caused it. The second control was never reached. The existing [control predicate](/home/rb/worktrees/sfora-positive-causality/scripts/diagnose_connected_gallery_freshness.py:260) correctly rejects these differences.

   Preserve CUDA FP32 readout and normalization, original flags, role order, B32 batches and tails. CPU FP32, tolerance checks, extra normalization, or copying expected output bytes would invalidate the comparison.

2. **Reconstructing the original FIT batches would breach the validation boundary.** I independently checked the partition: FIT groups `13216..13247` and `13248..13279` contain **14 sealed validation rows**. Their images cannot be opened to repair this TRAIN-only diagnostic. The [role audit](/home/rb/worktrees/sfora-positive-causality/docs/evidence/compact_metric/sop-siglip2-substrate-v1/connected-next-mechanism-research-v1/tail-original-group-role-audit.json) agrees.

   The admissible finite oracle is the **original accepted selection-query B6**, FIT rows `[13239,13240,13241,13242,13254,13257]`, under its original live execution context. This is a new matched-context oracle; it does not retroactively make the failed all-cache comparison exact.

3. **Input coverage is established; current availability is not.** Local metadata verifies 13,283 FIT rows, 3,449 selection rows, 1,734 queries, 1,715 gallery rows and 498 products. Selection rows and products are disjoint from the 6,355-row control training scope. Every selected row originally came from a complete encoder B32 cache batch.

   However, the [input ledger](/home/rb/worktrees/sfora-positive-causality/docs/evidence/compact_metric/sop-siglip2-substrate-v1/connected-next-mechanism-research-v1/freshness-parent-input-ledger.json) explicitly says remote bytes have not been freshly checked. Historical V3 execution supports availability then; launch still requires fresh hashes, shapes, typed endpoint checks and original terminal admission.

4. **Existing replay is not actual native-search parity.** [`native_wire_quality`](/home/rb/worktrees/sfora-positive-causality/scripts/diagnose_connected_gallery_freshness.py:566) uses CPU Torch integer multiplication and sorting. It is useful independent wire replay, but it never calls Rust/CUDA search. Likewise, equal control descriptors cannot prove equality of all underlying 1,152-dimensional features under a different candidate readout.

   Require independent same-input, same-batch readout output parity for candidate A/C, plus actual native IDs, score bits and ordinal ties on the new mixed wire. Reuse the existing [`CutilePackedInt8Gallery.search_packed`](/home/rb/worktrees/sfora-positive-causality/src/sfora/cutile_int8.py:188) interface and native tie witness. Historical parity on a different gallery is insufficient.

5. **The claimed deployment and causal conclusions exceed the evidence.** Candidate A/C applied to frozen features generally creates a new gallery artifact and index. Training used cached canonical features while serving uses different numerical contexts; cache extraction and accepted live export also differ in batching and cuDNN TF32 flags. A positive result would support this cache-specific method, not prove a pure freshness effect.

   Optimizer causation, exclusion of coverage, core-44 gate impossibility, unchanged cost, and family-wide closure after a negative result remain unsupported. Per-request byte checks must remain intact. The [current inference path](/home/rb/worktrees/sfora-positive-causality/scripts/train_siglip2_connected_mlp.py:1258) explicitly detects current `.data` substitutions.

**The smallest admissible implementation proposal** is a separate `scripts/diagnose_connected_asymmetric_cache.py` and `scripts/test_connected_asymmetric_cache.py`, with a fresh freeze directory containing `execution.json`, `authority.json`, `source-ledger.json` and the launch command. Preserve historical diagnostic files and receipts. Proposed CLI:

```text
<PINNED_PYTHON> -B <FROZEN_DIR>/diagnose_connected_asymmetric_cache.py \
  --execution-sha256 <SHA> \
  --authority <AUTHORITY_JSON> \
  --authority-sha256 <SHA> \
  --output <EXCLUSIVE_NEW_DIRECTORY>
```

Freeze the stage and seed in authority, not optional runtime tuning flags.

The source ledger must bind these exact roles:

| Role | Required binding |
|---|---|
| Cache | Original `fit.npy`, SHA `c418df0354408c8b33dca07f21e7b3cbf9d5089f7f278b93003a614f21085716`; normalized little-endian FP32, `13283×1152` |
| Rows | Original FIT manifest, partition and scope; selection ordinal → original FIT index → image/product → query/gallery role |
| Query F | Archived candidate query raw/unit/wire rows, endpoint and export receipt, separately for each seed |
| Gallery S | Original cache gallery rows through that **same candidate’s** head/A/C/means/μ |
| Controls | Each seed’s own control endpoint and archived descriptors; do not confuse candidate `SS` with the trained control |
| Sources/runtime | Original v3 source-role map, interpreter, package/native origins, flags and terminal receipts |
| Optional tail oracle | Exact six selection images, accepted RGB/pixel/output hashes, unchanged control bundle and original B6 context |

I verified the historical identity source SHA `840c5d8277a89ccdac02c9e231cbe6eddf386e2b23915ecd1bbec1136c51dee8` and its readout AST equality with the connected helper. **Use the historical packing pin:** current repository `joint_relational_compaction.py` does not match that archived source hash.

Execution should follow this fixed order:

1. **Admit and replay before new quality.** Reuse `prepare`, `panel_mapping`, `check_export_mapping`, `Sources`, `origin_audit`, `terminal_admission`, `check_endpoint_payload` and `fresh_values`. Authenticate archived descriptors and replay their per-query R@1/AP.

2. **Resolve controls before candidate scoring.** Preserve complete raw/unit/code/inverse/wire equality. If needed, independently reconstruct only the predeclared accepted query B6 using `load_inference` and the AST-checked capture seam in [the existing batch observer](/home/rb/worktrees/sfora-positive-causality/scripts/observe_connected_control_batch_execution.py:186). Require original RGB/pixel/output hashes and repeatability. Keep that live-tail provenance separate from cache provenance. Any other mismatch stops the run; no expanding image subsets or padding search.

3. **Construct only FS.** Keep candidate query rows byte-identical to archive. Compute gallery rows using the pinned identity `fullfeature_raw_features` and original `packed_outputs`, CUDA FP32, gallery B32/tail19 grouping, without another outer feature normalization. Use independent reloads and the accepted explicit concat-plus-centered-residual oracle; retain omitted-C and wrong-μ negatives.

4. **Persist, read back and verify actual search.** Bind the mixed artifact’s independent query/gallery roles and gallery hash. Require exact output and native parity before releasing its quality result. Cache-to-live encoder equivalence remains unproved unless separately witnessed; do not label this a pure causal freshness result.

5. **Stop on engineering or resource failure.** Retain the 700-second whole-job cap, 8 GiB host cap, zero swap/events, CUDA allocation below 10 GB, both locks, uncached exit authentication and genuine cleanup. Include oracle construction, hashing, native search and teardown. V3’s 19.643 seconds is time to failure, not an estimate of successful cost. Emit no qualifying result after a failed gate.

6. **Apply staged scientific gates unchanged.** Start with seed179061. Kill this method if FS does not improve R@1 over archived candidate FF, or fails the existing first-stage control/source/concat/AP/floor/cost rules. Only then admit seed179069 and full paired evaluation. Require improvement over FF in each seed, existing per-seed conditions, both mean gains ≥0.2 percentage points, and both product lower bounds strictly positive using the original 5,000 draws/179019; retain query intervals and complete cost accounting. The [existing gate implementation](/home/rb/worktrees/sfora-positive-causality/scripts/evaluate_siglip2_identity_diversity.py:917) is the authority.

Passing permits the next frozen evaluation stage. It does not permit a direct VAL jump, revive symmetric KILL, establish either official benchmark win, or establish matched image-to-top-k speed. No displacement/cosine side study, LR change, loss change or quality sweep is needed.

Not checked: current remote bytes, runtime tests, numerical/native parity, quality or latency. No edits, native/Torch imports, SSH, images, jobs, children or operator contact; SSH35033 was untouched.  
Risk: the finite oracle may still fail; that must terminate this proposal without weakening equivalence or resource gates.
