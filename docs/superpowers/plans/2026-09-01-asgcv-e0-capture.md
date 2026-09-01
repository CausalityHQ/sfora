# ASG-CV E0 Capture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a resumable, authenticated Cars-training-only pipeline that captures exact Qwen semantic patch gradients, fits the frozen ASG-CV predictor, and emits the claim-ineligible E0 falsifier without opening retrieval quality.

**Architecture:** Keep scientific authority in small `src/sfora` modules and GPU/model orchestration in one offline script. Reuse the already-tested Qwen replay semantics, but add a real-image pair boundary and phase receipts so completion generation, gradient capture, predictor fitting, and E0 evaluation cannot silently share mutable state. Every dense row is fp32 on disk and exact fp32-to-fp64 widening in E0.

**Tech Stack:** Python 3.12, PyTorch, NumPy, Transformers/Qwen3-VL, canonical JSON, SHA-256, pytest, Ruff, mypy.

**Spec:** `docs/pass212_amortized_semantic_gradient_control_variate_2026-08-31.md`

## Global Constraints

- Cars official test examples and labels remain inaccessible through E0.
- Predictor-train, E0-validation, and E1-optimization classes and image IDs are disjoint and bound by `AsgcvPartitionAuthority`.
- Candidate schedules, rollout seeds, completion groups, and eligible schedules are sealed before exact gradient replay.
- Every accepted exact field has shape `2 x 49 x D`, dtype fp32, finite values, and one canonical gradient-sample receipt.
- Predictor fitting and E0 validation each contain exactly 64 strata of eight pairs (512 pair rows), from disjoint class bands; predictor rank is exactly 16.
- The one-of-eight selected index is derived after predictor output is sealed.
- No E0 artifact is claim eligible and no retrieval metric is read or emitted.
- The running three-seed control is never modified, restarted, or overlapped by this pipeline.

---

### Task 1: Phase and artifact authority

**Files:**
- Create: `src/sfora/asgcv_e0_capture.py`
- Create: `tests/test_asgcv_e0_capture.py`

**Interfaces:**
- Produces: `AsgcvE0CaptureManifest`, `AsgcvE0PhaseReceipt`, `canonical_capture_manifest_bytes`, `validate_capture_manifest_bytes`, `canonical_phase_receipt_bytes`, and `validate_phase_receipt_bytes`.
- Consumes: `AsgcvPartitionAuthority`, candidate/eligible schedule digests, rollout authority, model revision, fixture digest, pooler digest, and predictor digest.

- [ ] **Step 1: Write the failing manifest test**

```python
def test_capture_manifest_binds_disjoint_phase_inputs_and_forbids_test_access(tmp_path):
    manifest = canonical_capture_manifest_bytes(
        source_commit="1" * 40,
        dataset_manifest_sha256="2" * 64,
        partition_authority=_partition(),
        rollout_authority=_rollout(),
        candidate_schedule_sha256="3" * 64,
        eligible_schedule_sha256="4" * 64,
        model_revision="5" * 40,
        fixture_sha256="6" * 64,
        official_test_access=False,
    )
    assert validate_capture_manifest_bytes(manifest)["official_test_access"] is False
```

- [ ] **Step 2: Run the RED**

Run: `.venv/bin/pytest -q tests/test_asgcv_e0_capture.py::test_capture_manifest_binds_disjoint_phase_inputs_and_forbids_test_access`

Expected: import failure for `sfora.asgcv_e0_capture`.

- [ ] **Step 3: Implement strict manifest and phase receipts**

Use exact concrete types, sorted canonical JSON plus one LF, exact key sets, SHA-256 self-digests, phase names `eligibility`, `capture`, `fit`, and `evaluate`, monotone input/output digest lists, `claim_eligible=false`, and `official_test_access=false`. Reject a phase receipt unless all inputs equal the preceding sealed outputs.

- [ ] **Step 4: Mutation-lock every field and phase transition**

Test missing/extra keys, bool-as-int, source/model/partition/schedule drift, reordered output digests, duplicate output digests, phase skips, test access, and self-digest drift.

- [ ] **Step 5: Run and commit**

```bash
.venv/bin/pytest -q tests/test_asgcv_e0_capture.py
.venv/bin/ruff check src/sfora/asgcv_e0_capture.py tests/test_asgcv_e0_capture.py
.venv/bin/mypy src/sfora/asgcv_e0_capture.py
git diff --check
git add src/sfora/asgcv_e0_capture.py tests/test_asgcv_e0_capture.py
git commit -m "feat: add ASG-CV E0 capture authority"
```

### Task 2: Real-image Qwen pair preparation

**Files:**
- Modify: `scripts/diagnose_saga_gb10_feasibility.py`
- Create: `tests/test_asgcv_qwen_pair.py`

**Interfaces:**
- Produces: `QwenSagaAdapter.prepare_image_pair(images, prompt_utf8, attribute_token_span, patch_tokens_per_image) -> PreparedPair`.
- Consumes: exactly two authenticated RGB uint8 arrays and the existing processor/model adapter.

- [ ] **Step 1: Write the failing pair-preparation test**

Use a fake processor returning the exact five-tensor schema. Require two image token ranges, 49 patches per image, no label argument, and identical output to `prepare_pair` for the existing generated fixture images.

- [ ] **Step 2: Run the RED**

Run: `.venv/bin/pytest -q tests/test_asgcv_qwen_pair.py`

Expected: missing `prepare_image_pair`.

- [ ] **Step 3: Extract the common processor boundary**

`prepare_pair` resolves fixture images and delegates to `prepare_image_pair`. The new method validates two RGB uint8 HWC arrays, copies them before processor access, rejects test-split metadata by construction (there is no label/split argument), and preserves the existing exact processor schema and patch-span checks.

- [ ] **Step 4: Run existing and new adapter tests**

```bash
.venv/bin/pytest -q tests/test_asgcv_qwen_pair.py tests/test_diagnose_saga_gb10_feasibility.py
.venv/bin/ruff check scripts/diagnose_saga_gb10_feasibility.py tests/test_asgcv_qwen_pair.py
git diff --check
```

- [ ] **Step 5: Commit**

```bash
git add scripts/diagnose_saga_gb10_feasibility.py tests/test_asgcv_qwen_pair.py
git commit -m "feat: prepare authenticated ASG-CV image pairs"
```

### Task 3: Forward eligibility and resumable exact-gradient capture

**Files:**
- Create: `scripts/run_asgcv_e0.py`
- Create: `tests/test_run_asgcv_e0.py`
- Modify: `src/sfora/asgcv_e0_capture.py`

**Interfaces:**
- Produces: an `eligibility` phase with sealed completion groups/schedules and a `capture` phase containing separate 512-row predictor-training and 512-row E0-validation `sample-<ordinal>.json`, `patch-<ordinal>.npy`, and `gradient-<ordinal>.npy` triples plus sealed phase receipts.
- Consumes: capture manifest, authenticated train-only image object table, pair schedules, rollout authority, completion protocol, and local Qwen snapshot.

- [ ] **Step 1: Write a fake-adapter RED**

The fake generates eight token-ID completions and returns fp32 `2 x 49 x D` tensors. Assert candidate schedule order, source-derived rollout seeds, exact token-level classification, four same/four different refill per stratum, separate training/validation bands, one write per eligible ordinal, atomic temporary-to-final rename, refusal to overwrite mismatched content, and resume only after revalidating every existing triple with `validate_gradient_sample_bundle`.

- [ ] **Step 2: Run the RED**

Run: `.venv/bin/pytest -q tests/test_run_asgcv_e0.py -k capture`

Expected: missing `eligibility` and `capture` phases.

- [ ] **Step 3: Implement bounded capture**

Eligibility loads one pair at a time, runs the exact eight source-derived completions without autograd, classifies them with `classify_asgcv_completion_group`, and seals the final refill schedule before capture begins. Capture then calls the existing `capture_asgcv_patch_gradient`, synchronously copies only the two fp32 arrays to CPU, emits the canonical receipt, clears model graphs, records exact elapsed ns and CUDA peaks, and advances the durable ordinal. Neither phase can access predictor state or the selection seed.

- [ ] **Step 4: Add interruption and corruption tests**

Kill after ordinals 0, 1, and 511; mutate each array/receipt identity; inject nonfinite values; reuse an image from another partition; and verify resume either continues from the first absent ordinal or fails without rewriting evidence.

- [ ] **Step 5: Run and commit**

```bash
.venv/bin/pytest -q tests/test_run_asgcv_e0.py -k capture
.venv/bin/ruff check scripts/run_asgcv_e0.py tests/test_run_asgcv_e0.py src/sfora/asgcv_e0_capture.py
.venv/bin/mypy src/sfora/asgcv_e0_capture.py
git diff --check
git add scripts/run_asgcv_e0.py tests/test_run_asgcv_e0.py src/sfora/asgcv_e0_capture.py
git commit -m "feat: capture resumable ASG-CV gradients"
```

### Task 4: Predictor fitting without selection leakage

**Files:**
- Modify: `scripts/run_asgcv_e0.py`
- Modify: `src/sfora/asgcv_e0_capture.py`
- Modify: `tests/test_run_asgcv_e0.py`

**Interfaces:**
- Produces: a canonical predictor state plus optimizer/update receipt.
- Consumes: only predictor-training class-band samples and the sealed SRHT authority; it cannot read E0 validation receipts or selection seeds.

- [ ] **Step 1: Write the fit-phase RED**

Require source-bound initialization, fixed rank 16, fixed dense-plus-SRHT loss, fixed optimizer/update count from the manifest, deterministic sample order, uniform receipt identities, and byte-identical state digests across repeated CPU controls.

The registered full-shape authority is 512 rows, batch size 4, 32 epochs and
4,096 updates of single-tensor fp32 AdamW with exact rational parameters
`lr=3/10,000`, `weight_decay=1/10,000`, `betas=(9/10,999/1,000)`, and
`eps=1/100,000,000`. The manifest binds the SRHT authority and distinct
initialization/sample-order seeds before capture; tests may use a separately
marked reduced shape but may not relax concrete-type, divisibility, or derived
update-count checks.

- [ ] **Step 2: Implement fit phase**

Load one bounded minibatch of fp32 patch/gradient pairs at a time. Detach teachers/tokens, update only predictor parameters, reject zero/nonfinite teacher energies, and seal `predictor_state_sha256` before E0 inputs or selection authority become available.

- [ ] **Step 3: Mutation-lock leakage and determinism**

Prove the phase rejects an E0 sample path, a selection seed, shuffled input order, changed optimizer hyperparameter, changed update count, changed SRHT rows, and nondeterministic output state.

- [ ] **Step 4: Run and commit**

```bash
.venv/bin/pytest -q tests/test_run_asgcv_e0.py -k fit
git diff --check
git add scripts/run_asgcv_e0.py src/sfora/asgcv_e0_capture.py tests/test_run_asgcv_e0.py
git commit -m "feat: fit sealed ASG-CV predictor"
```

### Task 5: E0 evaluation and custody

**Files:**
- Modify: `scripts/run_asgcv_e0.py`
- Modify: `src/sfora/asgcv_bias.py`
- Modify: `src/sfora/asgcv_custody.py`
- Modify: `tests/test_run_asgcv_e0.py`
- Modify: `tests/test_asgcv_bias.py`
- Modify: `tests/test_asgcv_custody.py`

**Interfaces:**
- Produces: canonical E0 result, custody receipt, relation-control evidence, projected randomization p-value, and diagonal-free projected U z-score.
- Consumes: exactly 512 E0 validation receipts, their fp32 arrays, sealed predictor state, selection seed, SRHT authority, and runtime measurements.

- [ ] **Step 1: Write the evaluation RED**

Require exact `[64,8,2,49,D]` schedule ordering, predictor output before selection reveal, source-derived selected indices, exact fp32-to-fp64 widening, all existing E0 gates, custody of all 512 sample digests, pair-exchange evidence, relation liveness, 10,000-draw mean evidence, and diagonal-free U evidence.

- [ ] **Step 2: Implement one-pass evaluation**

Predict all eight rows per stratum under `torch.no_grad`, seal the prediction digest, derive the selected index, compute exact metrics from widened arrays, then write the three canonical artifacts. Randomization and U statistics are evidence fields, not adaptive tuning gates.

- [ ] **Step 3: Mutation-lock scientific relations**

Change one row, receipt, selected index, predictor tensor, class band, relation sign, SRHT row, semantic timing, or custody digest. Each mutation must fail before a result is accepted. Add a zero relation-conditioning predictor that fails the liveness control.

- [ ] **Step 4: Run and commit**

```bash
.venv/bin/pytest -q tests/test_run_asgcv_e0.py -k evaluate tests/test_asgcv_bias.py tests/test_asgcv_custody.py tests/test_asgcv_predictor.py
git diff --check
git add scripts/run_asgcv_e0.py src/sfora/asgcv_bias.py src/sfora/asgcv_custody.py tests/test_run_asgcv_e0.py tests/test_asgcv_bias.py tests/test_asgcv_custody.py
git commit -m "feat: evaluate authenticated ASG-CV E0"
```

### Task 6: Offline launcher and resource monitor

**Files:**
- Create: `scripts/run_asgcv_e0.sh`
- Create: `tests/test_run_asgcv_e0_shell.py`

**Interfaces:**
- Produces: phase-specific one-shot invocations with pressure, progress, PID, scratch, and terminal receipts.
- Consumes: local content-addressed inputs only; acquisition is a separately authorized outer step.

- [ ] **Step 1: Write shell refusal tests**

Assert exact source/tree, offline model variables, no official-test path, no network-capable flag, one process group, bounded scratch, named cleanup, RSS/PSI/swap/progress stops, and no automatic retry after a terminal.

- [ ] **Step 2: Implement the launcher**

Expose one explicit phase per invocation. Authenticate every local input before launch, set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`, monitor the original process group, preserve stdout/stderr/status, and delete only registered scratch files after PID clearance.

- [ ] **Step 3: Run and commit**

```bash
.venv/bin/pytest -q tests/test_run_asgcv_e0_shell.py
.venv/bin/ruff check tests/test_run_asgcv_e0_shell.py
bash -n scripts/run_asgcv_e0.sh
git diff --check
git add scripts/run_asgcv_e0.sh tests/test_run_asgcv_e0_shell.py
git commit -m "feat: add bounded ASG-CV E0 launcher"
```

### Task 7: Synthetic end-to-end and repository assurance

**Files:**
- Modify: `tests/test_run_asgcv_e0.py`
- Modify: `docs/pass212_amortized_semantic_gradient_control_variate_2026-08-31.md`

**Interfaces:**
- Produces: one complete fake-adapter four-phase run and verified documentation matching implemented authority.
- Consumes: all prior task interfaces.

- [ ] **Step 1: Add a reduced-shape end-to-end test**

Run eligibility, capture, fit, and evaluate in separate invocations over a test-only reduced shape. Prove phase replay is byte-identical, capability leakage fails, interrupted capture cannot be misclassified complete, and the final result is claim ineligible.

- [ ] **Step 2: Update the method document**

Record the custody artifact, exact mean/U evidence semantics, relation controls, phase separation, resumability, and that none are retrieval-quality evidence.

- [ ] **Step 3: Run final assurance once**

```bash
.venv/bin/pytest -q tests/test_asgcv.py tests/test_asgcv_bias.py tests/test_asgcv_custody.py tests/test_asgcv_predictor.py tests/test_asgcv_protocol.py tests/test_asgcv_e0_capture.py tests/test_run_asgcv_e0.py tests/test_run_asgcv_e0_shell.py tests/test_diagnose_saga_gb10_feasibility.py
.venv/bin/ruff format --check src/sfora/asgcv*.py scripts/run_asgcv_e0.py tests/test_asgcv*.py tests/test_run_asgcv_e0*.py
.venv/bin/ruff check src/sfora/asgcv*.py scripts/run_asgcv_e0.py tests/test_asgcv*.py tests/test_run_asgcv_e0*.py
.venv/bin/mypy src/sfora/asgcv.py src/sfora/asgcv_bias.py src/sfora/asgcv_custody.py src/sfora/asgcv_e0_capture.py src/sfora/asgcv_predictor.py src/sfora/asgcv_protocol.py
python3 -m py_compile scripts/run_asgcv_e0.py
git diff --check
```

- [ ] **Step 4: Independent review and verified repair**

Request read-only cross-provider review of the exact diff. Reproduce every accepted blocker with the narrowest RED, repair it, rerun the affected test, then rerun the final assurance once.

- [ ] **Step 5: Commit and push**

```bash
git add docs/pass212_amortized_semantic_gradient_control_variate_2026-08-31.md src/sfora/asgcv*.py scripts/run_asgcv_e0.py scripts/run_asgcv_e0.sh tests/test_asgcv*.py tests/test_run_asgcv_e0*.py
git commit -m "feat: complete ASG-CV E0 falsifier"
git push origin HEAD:devbox/emafactorial
test "$(git rev-parse HEAD)" = "$(git ls-remote origin refs/heads/devbox/emafactorial | awk '{print $1}')"
```

### Task 8: Separate DGX scientific execution

**Files:**
- No repository edits during the run.

**Interfaces:**
- Produces: authenticated phase receipts, exact sample corpus, E0 result, custody receipt, bias evidence, resource terminal, and cleanup evidence.
- Consumes: one clean committed source revision and frozen local inputs.

- [ ] **Step 1: Wait for the original three-seed control terminal**

Preserve all three seed receipts and the original aggregate. Do not stop or overlap it.

- [ ] **Step 2: Run one-pair SAGA feasibility first**

Use the existing bounded diagnostic to measure actual GB10 replay memory/time. If it fails authority, attention, memory, determinism, or time, stop ASG-CV before the 512-pair capture.

- [ ] **Step 3: Run a bounded 64-pair capacity pilot**

Only a passing capacity-floor receipt permits predictor fitting and E0 capture.

- [ ] **Step 4: Execute the four E0 phases once**

No adaptive rank, sample count, threshold, or architecture change is allowed after exact gradients open. Preserve the original terminal and never auto-restart.

- [ ] **Step 5: Classify and report**

Report actual wall time, peak resources, each gate, custody status, bias evidence, and whether E0 passes. A pass authorizes E1 design only; it is not a retrieval or SOTA claim.
