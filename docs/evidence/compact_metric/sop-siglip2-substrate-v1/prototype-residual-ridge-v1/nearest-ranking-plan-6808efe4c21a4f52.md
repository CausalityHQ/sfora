# Prospective nearest-ranking plan

Status: read-only proposal; native training, mechanics, quality and cost unrun. SAME production goal remains unmet. Original consultation 6808efe4c21a4f52 completed exit 0 in 752 seconds; actual gpt-6.1-sol / xhigh. Prompt SHA256 46bc2bd7726587c5cbebe3ac7f672848b51a57b4683c9deb0bf607f7c72b6756. Retain signed-concat accepted endpoints and procedure KILL; no duplicate consultation.

Root source inspection independently confirmed the current readout helper detaches input/teacher paths. A differentiable wrapper and exact inference parity are mandatory. Lack of source evidence of cached separability does not prove cached inseparability or establish the encoder as the cause. The proposed four-tensor trial tests that hypothesis. Recipe awaits independent consequential critique and actual source/resource qualification.

**Recommend one final-MLP encoder ranking trial, with the accepted concat readout fixed.** The available evidence does not justify claiming that the remaining TRAIN nearest collisions are separable in the cached 1,152-dimensional vectors. Under your fallback rule, I would advance encoder training rather than another cached readout experiment.

The [original fitter3](/home/rb/worktrees/sfora-positive-causality/docs/evidence/compact_metric/sop-siglip2-substrate-v1/prototype-residual-ridge-v1/train-source-v3/fit_siglip2_prototype_residual.py:374) regresses member-inclusive class-mean residuals. The current concat fitter expands that correction to `[Z32,H0raw128]`, but still fits class means. The warm control already received classification and valid-anchor ranking supervision; the missing intervention is direct nearest-impostor supervision during this next adaptation.

**Measured:** concat improved selection mAP by +0.217187pp over linear, with positive product-bootstrap lower bound, but its +0.057670pp R1 gain failed the frozen gate. The signed receipt’s TRAIN64 entries are parity witnesses, not collision or separability measurements. **Inferred:** retaining the improved readout while adjusting the encoder offers a plausible way to change missing distinctions. Its quality gain and runtime are unmeasured.

1. **Freeze the initialization and trainable inventory.**

   Initialize both arms from the actual concat endpoint in the [accepted selection receipt](/home/rb/worktrees/sfora-positive-causality/docs/evidence/compact_metric/sop-siglip2-substrate-v1/prototype-residual-ridge-v1/signed-concat-selection-score-v1/receipt.json):

   - Checkpoint: `/home/riomus/runs/sfora-so400-signed-concat-fit-concat-v1/resume.pt`
   - File SHA256: `b702e03847be88540ee9711b420279e3bc131fafc475f361eca3d57cf3b8bbcf`
   - Complete typed-state SHA256: `a118fd98cce0b8fafa51c897be70b2b6e2ec93ebb226b2382dd264721776b644`

   Authenticate its complete original vision dependency, warm061 payload, head, concat coefficients, means, buffers, configuration and source closure.

   Train **only** these four existing So400 tensors:

   ```text
   encoder.layers.26.mlp.fc1.weight  [4304,1152]
   encoder.layers.26.mlp.fc1.bias    [4304]
   encoder.layers.26.mlp.fc2.weight  [1152,4304]
   encoder.layers.26.mlp.fc2.bias    [1152]
   ```

   Their inventory is present in the receipt: **9,921,872 trainable scalars**, derived from those shapes. Freeze the other vision tensors, pooling/post-LN, complete warm head, classifier, concat `A`, and both fitted means. No new architecture, normalization statistics or ridge solve.

2. **Use exactly TRAIN6355/1008 to construct supervision.**

   Bind the original partition SHA `702f763e…676c`, canonical cache SHA `55d37d06…6bdb8`, ordered original-row mapping and dense TRAIN targets. The canonical cache is already normalized FP32 pooled output; “raw canonical” in the fitter does **not** mean unnormalized pooled features.

   Independently reconstruct, in each arm, the accepted concat TRAIN raw descriptors \(T_i\) using its qualified CPU arithmetic. Define:

   ```text
   V_i = normalize(T_i)
   P_c = mean(T_i for TRAIN members of class c), including each member
   e0  = mean_i ||T_i - P_yi||²
   ```

   Freeze `T`, `V`, `P`, class counts and finite-positive `e0`. They are teacher targets, not trainable banks. Selection and validation rows, labels and metric results cannot construct targets, miners, schedules or hyperparameters.

   Online anchors are the corresponding **canonical native256 TRAIN images**, using the pinned processor and no augmentation. Preserve exact image/row hashes. Native vision uses FP16 autocast; pooled values, readout and losses use FP32. CPU cached-teacher and native-image arithmetic remain separately identified roles.

3. **Freeze one objective comparison and one miner.**

   For current raw descriptor \(r_i\), let \(u_i=\operatorname{normalize}(r_i)\). Use the shared regression term:

   \[
   L_{\mathrm{mean}}=\operatorname{mean}_i\|r_i-P_{y_i}\|^2/e_0.
   \]

   Mine against **all 6,355 frozen teacher vectors**, separately for every current anchor:

   - Positive: maximum cosine among **other original images of the same TRAIN identity**.
   - Negative: maximum cosine among **all different-identity TRAIN images**.
   - Exclude the anchor itself; exclude all same-identity rows from negatives.
   - Break exact ties by ascending original-row ID.
   - Singleton identities contribute regression only. Average ranking over valid anchors; an entirely invalid batch has zero ranking term.

   With cosine margin **0.05**:

   \[
   L_{\mathrm{rank}}=
   \operatorname{mean}_{\mathrm{valid}\ i}
   \max(0,\ 0.05+u_i^\top V_{n_i}-u_i^\top V_{p_i})/0.05.
   \]

   **Control:** `Lmean`.  
   **Candidate:** `Lmean + Lrank`.

   Both arms perform the same mining and diagnostics; only the candidate ranking term participates in backward. Miner indices and teacher descriptors are detached. The current anchor remains connected to the encoder.

   This positive role targets top-1 retrieval’s requirement that at least one genuine gallery image outrank the nearest impostor. It does not force the farthest positive to become nearest.

4. **Fix the training budget and numerical contract prospectively.**

   Use **32 optimizer updates**, B64/micro16: **2,048 image presentations per arm**. Reuse the first32 batches of the authenticated genuine TRAIN schedule179061, regenerate them independently, and require exact agreement. The target bank uses all6355 rows/1008 classes; this is not a full pass over every TRAIN image.

   Use AdamW on exactly the four MLP tensors:

   ```text
   lr=1e-5; betas=(0.9,0.999); eps=1e-8; weight_decay=0.05
   amsgrad=False; foreach=False; fused=False
   global gradient clipping norm=1
   FP32 parameters, gradients and Adam moments
   FP16 vision autocast; FP32 readout/loss; initial scaler=128
   ```

   Accumulate four equally weighted microbatches per update. Retain source numerical flags and CUDA workspace contract; put vision in evaluation mode while allowing gradients. Reject nonfinite values or skipped scaler updates. No LR, margin, boundary, step-count or precision rescue.

   Thirty-two updates are a deliberately bounded delivery, **not** a claim of sufficient convergence. Earlier native TRAIN100 whole-unit timeouts make a larger initial commitment unjustified under300 seconds.

5. **Implement the useful model path and qualify it before training.**

   Proposed files:

   - `scripts/train_siglip2_nearest_ranking.py`: admission, differentiable concat forward, miner, losses, CPU/mechanics/TRAIN phases and complete resume state.
   - `scripts/evaluate_siglip2_nearest_ranking.py`: independent model loading, TRAIN falsifier, native export, wire verification and frozen scoring.
   - `scripts/test_siglip2_nearest_ranking.py`: role exclusions, ties/singletons, objective differences, gradient reachability, mutation rejection and resume parity.
   - `docs/inshop_nearest_ranking_gate_2026-10-03.md`: recipe, source closure, cost definitions and stop rules.

   A critical implementation issue is that [the existing readout helper](/home/rb/worktrees/sfora-positive-causality/scripts/prototype_residual_readout.py:57) detaches features, `h0` and `phi`. Calling it during training would block encoder gradients. Add a procedure-owned differentiable forward with the **same operation order and frozen coefficients**, without those detachments. Require exact forward agreement with the authenticated helper on identical inputs within each qualified device/precision role.

   Qualify finite, nonzero gradients on all four MLP tensors and no gradients or byte changes in the frozen complement. CPU qualification is one prospectively frozen engineering unit **≤500s**, CUDA hidden, with no quality read. Discarded GPU mechanics qualify each arm with uninterrupted17 versus independent8+9 replay, strict updated reload and final raw/unit/packed parity, **≤300s each**.

   Resume state must contain complete updated vision/config/buffers, unchanged head/classifier/concat coefficients/means, teacher targets, TRAIN mapping, schedule, optimizer moments, scaler, counters, RNG and numerical flags. Restoring it must not regenerate targets from an updated encoder.

   Export a complete inference model with the updated vision and fixed concat readout, preserving packed128 output. This produces a usable model rather than a cache-only correction.

6. **Use one bounded causal falsifier, then the original quality gate.**

   During required TRAIN-target construction, freeze at most128 valid anchors whose initial teacher margin is below0.05, ordered by original-row ID, together with their initial positive and negative rows. If no TRAIN anchor activates the ranking objective, stop before training: this recipe lacks the proposed training pressure.

   After both complete32-update endpoints qualify, independently reload them and re-encode the fixed triples’ actual images—at most384 distinct rows per model. Evaluate **all three roles through the same updated encoder**, so this checks whether fixed-teacher training transfers when gallery descriptors also move.

   Close the intervention before selection if:

   - Candidate has no better fixed-triplet margin-violation count than control; or
   - Median candidate-minus-control positive–negative cosine margin is ≤0.

   This is a TRAIN mechanism falsifier, not evidence of unseen-identity quality. Do not select another panel or tune after failure.

   A survivor receives exactly one selection1734q/1715g/498 comparison. Replay archived source metrics and wires first. Retain the original immediate failures, source floors, **both candidate–control gains ≥0.002 fraction**, and both paired-product95% lower bounds>0, using shared5000 draws/seed179019; report query intervals. Also require candidate R1 above the preserved concat result and mAP at least its **81.7775403554%**, preventing a degraded control from manufacturing a “win.”

   Only GO advances to unchanged sealed validation. Prior selection exposure remains disclosed.

Every native unit runs on **DGX Spark only**, sequentially, with both lifetime locks, **8GiB host, zero swap/disallowed events, CUDA allocated<10GB**, complete current-byte authentication, strict reload, original normal terminal exit and uncached exit checks. Fresh TRAIN units are **300s each**. Define candidate/control whole-service and total-training-core ratios **≤1.50**; core includes target construction, mining, decoding, forward/backward and optimizer work. Report preparation separately. Historical ridge timings are not the denominator.

The principal risks are weak learning within32 updates, fixed-teacher/current-gallery mismatch, mAP loss from emphasizing nearest negatives, and source/reload overhead exhausting300s. None is resolved by the current receipts. Failures close this procedure only; preserve concat and the complete updated endpoints where valid. Official SOP/InShop quality and matched public speed remain subsequent requirements, including B1/B32 serving qualification and the existing paired-latency protocol.

This is the concrete recipe for the root’s next fixed Opus/Astra critique. No files were edited or native work, metrics, jobs, children or consultations launched.

