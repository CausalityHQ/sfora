# Qwen Geometry Control Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a paired, vision-only Qwen Proxy Anchor experiment that isolates learned attention pooling from mean pooling while measuring both retrieval quality and cost.

**Architecture:** A pure authority module owns the frozen protocol, two pooling implementations, paired result validation, and decisions. A local-only runner adapts authenticated Qwen patch tokens and Cars manifests to smoke/train/aggregate phases; the controller publishes only complete checkpoint-bound receipts.

**Tech Stack:** Python 3.12, PyTorch, Transformers offline loading, pytest, Ruff, mypy.

**Spec:** `docs/superpowers/specs/2026-09-05-qwen-geometry-control-design.md`

## Global Constraints

- Sfora files only; do not modify Borsuk.
- No official Cars test access, language forward, generation, tokenizer, Hub, HTTP, or network API in the scientific child.
- Pooling is the sole arm difference; all six attempts execute exactly 183 successful updates.
- Every implementation change follows an observed RED then narrow GREEN.
- Preserve the four pre-existing untracked handoff entries.

---

### Task 1: Frozen Protocol and Pooling Core

**Files:**
- Create: `src/sfora/qwen_geometry_control.py`
- Create: `tests/test_qwen_geometry_control.py`

**Interfaces:**
- Produces: `QwenGeometryProtocol`, `MeanProjectionPooler`, `SingleQueryAttentionPooler`, `build_geometry_pooler`, and `pool_patch_tokens`.

- [ ] **Step 1: Write failing protocol and pooling tests**

  Cover exact constants, concrete-type rejection, finite rank-three patch inputs, normalized 4096-D output, deterministic attention weights, and a behavior test proving the two arms share input/output contracts but compute different pooling rules.

- [ ] **Step 2: Run the Task-1 RED**

  Run: `PYTHONDONTWRITEBYTECODE=1 uv run --offline --locked python -m pytest -q -p no:cacheprovider tests/test_qwen_geometry_control.py`

  Expected: collection fails only because `sfora.qwen_geometry_control` is absent.

- [ ] **Step 3: Implement the minimal core**

  Define frozen dataclasses and modules. Mean pooling averages patches before a bias-free output projection. Attention pooling applies a learned query and bias-free key projection, divides logits by the square root of token width, softmaxes over patches, and applies a bias-free output projection. Both validate finite nonzero output and return fp32 unit vectors.

- [ ] **Step 4: Run Task-1 GREEN and static checks**

  Run the Task-1 selector, then `uv run --offline --locked ruff check src/sfora/qwen_geometry_control.py tests/test_qwen_geometry_control.py`, `uv run --offline --locked mypy src/sfora/qwen_geometry_control.py`, and `git diff --check`.

- [ ] **Step 5: Commit the independently green core**

  Commit only the spec, plan, core, and core tests with the configured operator identity and no attribution trailers.

### Task 2: Paired Sampler, Schedule, and Optimizer Roles

**Files:**
- Modify: `src/sfora/qwen_geometry_control.py`
- Modify: `tests/test_qwen_geometry_control.py`

**Interfaces:**
- Produces: `GeometryBatchPlan`, `derive_epoch_batches`, `learning_rate_multiplier`, `parameter_role_manifest`, and `optimizer_groups`.

- [ ] **Step 1: Write failing behavioral tests**

  Hand-derive one small class-balanced example schedule. Assert 16 classes by four distinct images in production shape, arm-invariant batch digests, exactly 183 update indices, warm-up values at updates 0 and 9, cosine endpoints, zero proxy decay, and complete/disjoint parameter roles.

- [ ] **Step 2: Run the narrow RED**

  Run only the new schedule/role nodes and require missing-symbol failures.

- [ ] **Step 3: Implement deterministic schedules and roles**

  Use domain-separated SHA-256 counter streams for class/image/augmentation order. Return optimizer groups with explicit role names and reject missing, duplicated, frozen, or unexpected parameters.

- [ ] **Step 4: Run narrow and full Task-1/2 GREEN**

  Run `tests/test_qwen_geometry_control.py`, Ruff, mypy, and diff-check.

- [ ] **Step 5: Commit the schedule/role slice**

### Task 3: Logical-Batch Replay and Update Equivalence

**Files:**
- Create: `scripts/run_qwen_geometry_control.py`
- Create: `tests/test_run_qwen_geometry_control.py`

**Interfaces:**
- Produces: `replayed_proxy_anchor_step`, `GeometryStepEvidence`, and `state_sha256`.

- [ ] **Step 1: Write the failing real-module replay test**

  Use a small deterministic token tower and literal labels. Compare full-batch versus two-microbatch loss, score cotangent, each gradient tensor, clipped gradient norm, AdamW moments, and post-update parameter bytes. Include a mutation that advances the scheduler after a rejected update.

- [ ] **Step 2: Run the exact RED**

  Expected: import fails at the missing step API, not fixture construction.

- [ ] **Step 3: Implement one successful update boundary**

  Materialize logical scores without gradients, differentiate Proxy Anchor once with respect to scores, replay each microbatch against its exact score cotangent, verify every intended gradient, clip once, step AdamW once, and advance the schedule once. Reject nonfinite or skipped steps.

- [ ] **Step 4: Run replay GREEN plus core regressions**

  Run both new test files and `tests/test_siglip_proxy_control.py`, followed by Ruff, mypy, py_compile, and diff-check.

- [ ] **Step 5: Commit the replay slice**

### Task 4: Checkpoint, Resume, and Three-Update Smoke

**Files:**
- Modify: `src/sfora/qwen_geometry_control.py`
- Modify: `scripts/run_qwen_geometry_control.py`
- Modify: `tests/test_qwen_geometry_control.py`
- Modify: `tests/test_run_qwen_geometry_control.py`

**Interfaces:**
- Produces: `GeometryCheckpointAuthority`, `write_geometry_checkpoint`, `restore_geometry_checkpoint`, `run_smoke`, and `SmokeReceipt`.

- [ ] **Step 1: Write failing checkpoint/resume/smoke tests**

  Prove path+size+SHA binding, reject every identity/state mutation, compare uninterrupted and resumed states, and require two restored three-update executions to match input and state digests.

- [ ] **Step 2: Run the narrow RED**

- [ ] **Step 3: Implement atomic checkpoint publication and smoke receipts**

  Store model, proxy, optimizer, schedule, RNG, sampler, protocol, arm, seed, update, and source identities. Write a temporary file, fsync, rename, hash, then publish its canonical receipt.

- [ ] **Step 4: Run checkpoint/smoke GREEN and static checks**

- [ ] **Step 5: Commit the checkpoint/smoke slice**

### Task 5: Local Qwen/Cars Adapter and Closed CLI

**Files:**
- Modify: `scripts/run_qwen_geometry_control.py`
- Modify: `tests/test_run_qwen_geometry_control.py`

**Interfaces:**
- Produces: `load_vision_only_qwen`, `load_geometry_bands`, `smoke`, `train`, `aggregate`, and `main`.

- [ ] **Step 1: Write failing adapter and CLI tests**

  Use a complete fake Qwen output containing patch tokens. Assert no language module, tokenizer, generation, official-test, Hub, HTTP, arbitrary dataset, or unknown flag can be reached. Test exact class bands, RGB/224 input, sealed manifest/digest, and explicit execution flags.

- [ ] **Step 2: Run adapter/CLI RED**

- [ ] **Step 3: Implement local-only adapter and commands**

  Reuse the authenticated Qwen snapshot checks from `scripts/diagnose_saga_gb10_feasibility.py`, but expose only the vision tower. Implement command-specific parsers so `smoke`, `train`, and `aggregate` receive only their required capabilities.

- [ ] **Step 4: Run focused GREEN and dependency-complete local verification**

  Run both geometry test files plus SigLIP control tests, Ruff, mypy, py_compile, and diff-check.

- [ ] **Step 5: Commit the adapter/CLI slice**

### Task 6: Aggregate Evaluation, Cost Profile, and Decision

**Files:**
- Modify: `src/sfora/qwen_geometry_control.py`
- Modify: `scripts/run_qwen_geometry_control.py`
- Modify: `tests/test_qwen_geometry_control.py`
- Modify: `tests/test_run_qwen_geometry_control.py`

**Interfaces:**
- Produces: `SeedPairEvidence`, `GeometryCampaignResult`, `decide_geometry_campaign`, and `canonical_geometry_campaign_bytes`.

- [ ] **Step 1: Write failing decision and mutation tests**

  Hand-derive three paired seed rows. Mutation-lock per-query outcomes, integer hits, R@1, MAP@R, update counts, wall ratios, raw latency samples, p95, checkpoint identities, and result SHA. Cover each quality/cost failure independently and exhaustive precedence.

- [ ] **Step 2: Run decision RED**

- [ ] **Step 3: Implement one-time clean aggregation**

  Authenticate all six terminal checkpoints before loading classes `49..81`. Evaluate identical images/ties for both arms, retain raw per-query correctness and latency samples, recompute metrics and gates, and emit canonical newline JSON with `claim_eligible=false`.

- [ ] **Step 4: Run complete focused verification**

  Run the Astra-prescribed focused command, Ruff, mypy, py_compile, and diff-check.

- [ ] **Step 5: Commit the evaluation slice**

### Task 7: Repository Assurance, Deploy, and Smoke

**Files:**
- Create: `scripts/deploy_qwen_geometry_control_v1.sh`
- Create: `tests/test_deploy_qwen_geometry_control.py`

**Interfaces:**
- Produces: a commit-addressed, SHA-manifested DGX deployment and exactly one monitored smoke process.

- [ ] **Step 1: Write and run a failing deployment lifecycle test**

  Exercise a temporary fake remote root. Prove new-directory deployment, manifest verification before execution, GPU-idle check, one process group, no restart, pressure/timeout stop, terminal preservation, and named cleanup.

- [ ] **Step 2: Implement the deployer and make the lifecycle test GREEN**

- [ ] **Step 3: Run repository assurance once**

  Run full pytest, changed-file Ruff, mypy, py_compile, diff-check, and the repository's configured assurance command. Commit any verified repair, push the configured branch, and verify local/remote SHA equality.

- [ ] **Step 4: Deploy the exact commit and run one smoke**

  Check DGX process/GPU/PSI state, deploy to a new commit-addressed directory, verify every file digest, then start one smoke. Poll the original process at intervals no longer than 55 seconds and preserve its terminal receipt.

- [ ] **Step 5: Apply the frozen stop/go rule**

  A passing smoke authorizes six serial attempts. A scientific or resource failure is sent to Astra with the exact receipt before any design change. An authority/implementation failure is repaired test-first without changing scientific constants.
