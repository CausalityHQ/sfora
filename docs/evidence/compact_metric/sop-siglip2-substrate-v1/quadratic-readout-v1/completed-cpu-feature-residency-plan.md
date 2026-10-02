Recommend **keeping the authoritative canonical feature matrix on CPU and transferring only each selected microbatch to CUDA**. This changes storage placement while retaining the frozen learning recipe and every fresh integrity predicate.

The production callers support this intervention:

- [`fresh`](/home/rb/worktrees/sfora-positive-causality/scripts/train_siglip2_quadratic_readout.py:778) currently moves all FP32 `[6355,1152]` features onto CUDA.
- [`update`](/home/rb/worktrees/sfora-positive-causality/scripts/train_siglip2_quadratic_readout.py:1050) calls `integrity` before and after each update. Each call hashes the entire feature matrix through `check_complement → frozen_tree → fingerprint`.
- Training consumes only four 16-row microbatches per update. The two complement checks therefore copy **58,567,680 feature bytes back to CPU per update**, while the required training input is only **294,912 bytes**. These are source-derived byte counts, not measured savings.

The [profile ledger](/home/rb/worktrees/sfora-positive-causality/docs/evidence/compact_metric/sop-siglip2-substrate-v1/quadratic-readout-v1/cost-attribution-v1/profile-result-v1/repeated-integrity-callers.json) measured 3.678 seconds in the adapted serializer’s `Tensor.cpu` calls; that window overlaps other work and does not isolate removable transfer time. The failed cat comparison measured only 43.637 ms for the complete three-fingerprint workload. **Neither measurement establishes enough savings to meet 300 seconds.**

1. **Implement one storage change from the qualified source-v5 closure.**

   Use the four-file source at `cffdb936`, retaining frame-helper SHA `1928b288…b2e8db`. The rejected cat helper currently in HEAD must not enter the new closure.

   Modify [`train_siglip2_quadratic_readout.py`](/home/rb/worktrees/sfora-positive-causality/scripts/train_siglip2_quadratic_readout.py):

   - In `fresh`, retain the already normalized CPU `features.detach()`.
   - Add one shared `feature_rows(state, rows)` helper returning `state['features'][rows].to(device=state['A'].device)`.
   - Route `update`, `calibration`, and the initial source comparison in `cpu_witnesses` through it. In `update`, use the Python batch slice for CPU feature indexing; retain the CUDA index tensor for targets and positives.
   - Require CPU residency explicitly in `integrity`. Keep the complete matrix in `frozen_tensors`, both fresh complement hashes, pointer/version checks and `features_sha256`. Keep diagnostic row hashing unchanged.

   The removable duplication is **full-matrix device copying**, not byte authentication: both boundaries still hash every current feature byte. Full typed fingerprints remain unchanged for identical values because the original serializer excludes tensor device from identity. Preserve normalization, FP32 arithmetic, optimizer, bank refresh, RNG, checkpoint reconstruction and uncached source/exit admission.

2. **One fast source falsifier.**

   Add `ContractTests.test_cpu_feature_residency_preserves_fresh_bytes_and_rows` to the existing trainer test file; bound it to **5 seconds, stdlib/fake tensors only**.

   Assert ordered duplicate rows survive all four microbatches; uploads use A’s exact device and preserve dtype/bytes; CUDA loss indices remain separate. Compare CPU/GPU representations with the untouched typed serializer. Mutate a feature through unchanged-version `.data` after a successful boundary: the next boundary must reread and reject it. Assert that integrity performs no full-matrix CUDA transfer and retains both complete hashes.

3. **One decisive native gate, then the existing training path.**

   After freezing the new exact closure and passing fresh **CPU120**, root runs **one fresh control061 mechanics300**, including full17 versus independent8+9, strict reload, complete fresh exit and original terminal/resource admission. Verify uploaded microbatch bytes against ordinary CUDA-resident gathers for the first17 schedule.

   Before proceeding, require:

   - First17 mean **and** median update time ≤ **0.180 seconds**.
   - Whole-service non-update remainder ≤ **110 seconds**.
   - Every parity, integrity and resource check passes.

   This is a prospective engineering screen: `1000×0.180 + 110 = 290 seconds` is an **estimate**, not acceptance evidence. Any failure closes this intervention before candidate mechanics or TRAIN.

   A pass permits candidate mechanics, then **one fresh control061 TRAIN1000/300** with first17 replay and full terminal acceptance. Only accepted control completion permits candidate061 and the unchanged selection, seed069 and sealed-validation gates. Survivors still need public endpoint parity/speed, full-TRAIN fitting, and official SOP/InShop quality.

The principal risk is insufficient savings: CPU hashing remains, and four uploads are added. Incorrect indexing or an unintended arithmetic/device change is the correctness risk. Preserve the 531/666/705 timeouts and cat KILL; none becomes reusable state or a quality verdict.

No files were edited, tests run, native packages imported, or jobs launched.
