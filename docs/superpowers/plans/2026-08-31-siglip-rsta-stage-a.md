# SigLIP RSTA Stage-A Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an authenticated, optimization-only SigLIP RSTA causal falsifier that can run after the three sealed control checkpoints and decide `PASS_ONWARD`, `FAIL`, `UNRESOLVED`, or `INVALID` without loading clean, burned, or test classes.

**Architecture:** A focused authority/scoring module owns deterministic roles, exact statistics, gates, and canonical receipts. A thin diagnostic script authenticates sealed control artifacts, reuses the existing logical-batch replay, and computes receiver fields with matrix-free VJP/JVP actions. A separate launcher owns local-to-DGX acquisition, pressure monitoring, and cleanup; it cannot expose forbidden evidence roles to the scientific child.

**Tech Stack:** Python 3.12, PyTorch 2.12, torchvision, transformers, NumPy PCG64, pytest, existing SFORA canonical JSON and SigLIP control code.

**Spec:** `docs/superpowers/specs/2026-08-31-siglip-rsta-stage-a-design.md`

## Global Constraints

- Work only in `/home/rb/worktrees/sfora-emafactorial`; never edit or coordinate Borsuk code.
- Do not read partial control metrics; scientific execution requires three sealed epoch-60 checkpoints and the aggregate receipt.
- Scientific inputs contain Cars classes `0..48` only; classes `49..195` are forbidden.
- The primary field uses the complete trainable SigLIP vision tower plus bias-free projection; projection-only is forbidden.
- Use matrix-free eager `torch.func` actions; no dense Jacobian, `torch.compile`, or custom attention kernel.
- Every scientific result is canonical, newline-terminated, and `claim_eligible=false`.
- Every test run uses `uv run --offline --locked pytest -q -p no:cacheprovider <paths>`; "focused GREEN" means that command scoped to the task's test module.
- Do not launch Stage A, Stage B, method training, clean evaluation, or test evaluation from this implementation plan.

---

### Task 1: Freeze authority, roles, gates, and canonical receipt

**Files:**
- Create: `src/sfora/siglip_rsta_stage_a.py`
- Create: `tests/test_siglip_rsta_stage_a.py`

**Interfaces:**
- Produces: `RstaControlBinding`, `RstaStageAConfig`, `RstaRolePanel`, `RstaReceiverRow`, `RstaSeedEvidence`, `RstaAggregate`, `select_rsta_roles`, `summarize_rsta_stage_a`, and `rsta_stage_a_result_bytes`.
- Consumes: only an outcome-blind `rsta-control-binding-v1` projection and
  `sfora.pass209_m4.canonical_json_bytes`. It must not import script-private
  `ControlRunAuthority`.

- [ ] **Step 1: Write failing role-selection tests**

  Cover exact 49-class ordering, ranks 0--14, two 120-row primary batches,
  alternate nonoverlap, including exclusion of ranks 0--5 and all primary refills,
  147 receivers, duplicate IDs, insufficient class size, forbidden labels, and
  deterministic replay. Mutation-lock every hash domain.

- [ ] **Step 2: Run the focused RED**

  Run: `uv run --offline --locked pytest -q -p no:cacheprovider tests/test_siglip_rsta_stage_a.py::TestRstaRoles`

  Expected: import failure for `sfora.siglip_rsta_stage_a` only.

- [ ] **Step 3: Implement immutable config and role selection**

  Use concrete-type validation. Return frozen tuples; do not return mutable source
  dictionaries. Require labels exactly `0..48` and at least 15 unique examples per
  label.

- [ ] **Step 4: Write failing gate/receipt mutation tests**

  Cover all pass, fail-precedence, unresolved, invalid, seed-unanimity, bootstrap,
  concrete-type, nonfinite, field-order, digest, forbidden-role, backend, resource,
  autocast/checkpointing policy, all-three-seed failure, and `claim_eligible`
  mutations.

- [ ] **Step 5: Implement aggregation and canonical serialization**

  Recompute every derived field from receiver rows before serialization. Never trust
  caller-provided pooled metrics or verdicts.

- [ ] **Step 6: Run Task 1 GREEN and commit**

  Run the two focused test classes, Ruff on both files, `python3 -m py_compile` on
  both files, and `git diff --check`.

  Commit: `feat: add SigLIP RSTA Stage-A authority`

### Task 2: Implement exact contextual cotangent and parameter-direction extraction

**Files:**
- Modify: `src/sfora/siglip_rsta_stage_a.py`
- Modify: `tests/test_siglip_rsta_stage_a.py`
- Read/reuse: `src/sfora/siglip_proxy_control.py`

**Interfaces:**
- Produces: `contextual_rsta_direction(model, inputs, labels, replay_config) -> ContextualDirectionEvidence`.
- Consumes: `recomputed_proxy_anchor_backward` and the registered normalized proxy table.

- [ ] **Step 1: Write a failing dense linear fixture**

  Prove `dbar == -(dL/dscores) @ normalized_proxies`, the unclipped replay gradient
  equals `-g`, proxies are absent from `g`, and clip/weight-decay/Adam state cannot
  affect the result.

- [ ] **Step 2: Run the exact RED**

  Expected: missing `contextual_rsta_direction`.

- [ ] **Step 3: Implement the minimal replay adapter**

  Snapshot and clear parameter gradients, call the existing replay once, copy the
  lexicographically ordered tower/projection gradient tuple before clipping, and
  fail closed if any selected gradient is missing or nonfinite.

- [ ] **Step 4: Mutation-lock operator boundaries**

  Reject proxy inclusion, parameter-order drift, active dropout, BatchNorm,
  wrong score shape, altered Proxy Anchor alpha/delta, and bf16/fp32 role confusion.

- [ ] **Step 5: Run focused GREEN and commit**

  Commit: `feat: extract exact SigLIP contextual direction`

### Task 3: Implement matrix-free receiver fields and backend preflight

**Files:**
- Modify: `src/sfora/siglip_rsta_stage_a.py`
- Modify: `tests/test_siglip_rsta_stage_a.py`

**Interfaces:**
- Produces: `receiver_rsta_fields`, `preflight_rsta_jvp_backend`, `RstaJvpBackendEvidence`.
- Consumes: the ordered parameter direction from Task 2 and one receiver tensor/cotangent.

- [ ] **Step 1: Write dense-Jacobian, adjoint, and degeneracy RED tests**

  Use a normalized affine-plus-nonlinear tower where dense `J` is tractable. Compare
  `b=Jg` and `s=JJ^T dbar` to explicit float64 matrices. Add the bias-free linear
  head proof that `s` is parallel to tangential `dbar`.

- [ ] **Step 2: Write backend mutation tests**

  Cover forward-mode success, forced coverage failure, registered double-backward
  fallback, fallback disagreement above `1e-5`, wrong selected backend, nonfinite
  tangents, parameter-shape/order drift, train/eval drift, active checkpointing,
  checkpointing-equivalence failure, and the no-backend `INVALID` terminal.

- [ ] **Step 3: Implement eager functional receiver actions**

  Use `torch.func.functional_call`, `vjp`, and `jvp` with receiver batch size one.
  Keep constants outside the selected parameter tree. Project output fields into the
  descriptor tangent exactly once.

- [ ] **Step 4: Implement preregistered double-backward fallback**

  Activate only from the preflight decision; never retry or change backend after a
  scientific row begins.

- [ ] **Step 5: Run focused GREEN and commit**

  Commit: `feat: add matrix-free RSTA receiver fields`

### Task 4: Implement outcome directions, receiver scoring, and controls

**Files:**
- Modify: `src/sfora/siglip_rsta_stage_a.py`
- Modify: `tests/test_siglip_rsta_stage_a.py`

**Interfaces:**
- Produces: `proxy_free_margin_direction` and `score_rsta_receiver`.
- Consumes: receiver/support/foreign descriptors and Task 3 field evidence.

- [ ] **Step 1: Write hand-derived scoring tests**

  Cover top-32 foreign selection/ties, logmeanexp, tangent projection, `Delta`,
  `A_desc`, `rho=sqrt(1-cos^2)`, log ratio, cross contribution, random target, and
  cyclic derangement.

- [ ] **Step 2: Run RED, implement fp32 scoring, and run GREEN**

  Use stable logsumexp and concrete finite checks. Do not introduce a second recall
  or nearest-neighbor authority.

- [ ] **Step 3: Add rotation and radial mutation tests**

  Reject radial fraction above `1e-3`, zero projected controls, rotation residuals,
  and receiver/support role leakage.

- [ ] **Step 4: Commit**

  Commit: `feat: score SigLIP RSTA causal fields`

### Task 5: Build the authenticated local scientific CLI

**Files:**
- Create: `scripts/diagnose_siglip_rsta_stage_a.py`
- Create: `tests/test_diagnose_siglip_rsta_stage_a.py`
- Modify: `src/sfora/siglip_rsta_stage_a.py`

**Interfaces:**
- Produces: strict local-file CLI and one canonical result on stdout.
- Consumes: one outcome-blind control binding, three checkpoint files, pinned
  optimization dataset manifest/image root, Task 1 roles, and Tasks 2--4 scoring.

- [ ] **Step 1: Write parser/refusal RED tests**

  Require all local paths and exact registered identities. Reject network, S3,
  clean/burned/test, alternate checkpoint, threshold, seed, and backend override
  flags. Require `--execute-stage-a`.

- [ ] **Step 2: Implement authority-only loading**

  Authenticate the outcome-blind binding and checkpoint bytes and cross-bind the
  three seeds before constructing any model. Prove control receipts and their
  quality metrics, plus clean/burned/test bands, are absent from the scientific
  child capability manifest.

  Use a model-state-only checkpoint reader: validate schema, finality, seed, config,
  run-authority digest, and `claim_eligible is false`, then strictly load
  `model_state` without constructing an optimizer. Snapshot and restore all Python,
  NumPy, torch CPU, and torch CUDA RNG states around each load.

- [ ] **Step 3: Write reduced-shape end-to-end tests**

  Run three tiny sealed checkpoints through both panels, backend preflight,
  aggregation, and canonical output. Include authority, forbidden-class, interrupted
  row, duplicate, nondeterminism, and no-partial-result mutations.

- [ ] **Step 4: Implement the scientific loop**

  Run seed/batch/receiver serially, delete graphs and clear gradients between
  receiver actions, repeat the first receiver bitwise, and publish no result until
  all rows and controls validate.

- [ ] **Step 5: Run focused GREEN and commit**

  Commit: `feat: add authenticated SigLIP RSTA diagnostic`

### Task 6: Build the DGX launcher and lifecycle receipts

**Files:**
- Create: `scripts/run_siglip_rsta_stage_a.py`
- Create: `tests/test_run_siglip_rsta_stage_a.py`

**Interfaces:**
- Produces: acquisition/preflight/execution/cleanup controller; no scientific calculations.
- Consumes: the Task 5 local CLI and sealed control artifact locations.

- [ ] **Step 1: Write capability and lifecycle RED tests**

  Cover exact source/host/path allowlists, checkpoint finality, absent partial result,
  process-group ownership, 96 GiB RSS+CUDA-reserved stop, PSI/swap stops, timeout,
  progress, terminal preservation, named cleanup, and no restart.

- [ ] **Step 2: Implement an explicit phase controller**

  Separate authority projection, backend preflight, and scientific execution. The
  authority phase alone reads complete final seed/aggregate receipts through the
  existing authenticated control loader and emits `rsta-control-binding-v1` with no
  metric or verdict fields. It authenticates the full image-free control manifest,
  derives a canonical optimization-only manifest/digest, and includes only that
  digest in the binding. The scientific child receives only that binding, the
  authenticated checkpoints, the optimization-only ID/label manifest, and
  optimization images; receipt and clean/burned/test pixel paths do not exist in
  its namespace or argument vector, and test IDs/labels are absent entirely.

- [ ] **Step 3: Run no-science integration tests**

  Use fake checkpoints and a fake scientific child. Prove failures remove scratch,
  preserve the original terminal, and never publish partial evidence.

- [ ] **Step 4: Run focused GREEN and commit**

  Commit: `feat: add RSTA Stage-A execution controller`

### Task 7: Repository assurance and adversarial review

**Files:**
- Modify only files required by verified failures in Tasks 1--6.

**Interfaces:**
- Produces: verified source commit eligible for a separately authorized DGX preflight.

- [ ] **Step 1: Run grouped Python tests**

  First run:

  ```bash
  uv run --offline --locked pytest -q -p no:cacheprovider tests/test_siglip_rsta_stage_a.py tests/test_diagnose_siglip_rsta_stage_a.py tests/test_run_siglip_rsta_stage_a.py tests/test_siglip_proxy_control.py tests/test_run_siglip_proxy_control.py
  ```

  Then run one dependency-complete
  `uv run --offline --locked pytest -q -p no:cacheprovider`.

- [ ] **Step 2: Run static gates**

  Run `uv run --offline --locked ruff check` on all new/modified Python files,
  `python3 -m py_compile` on those files, and `git diff --check`. This repository
  has no research-doc validator; do not borrow one from another worktree.

- [ ] **Step 3: Obtain read-only cross-provider review**

  Ask Claude to inspect the exact spec, plan, diff, algebra, capability boundary,
  role leakage, gate adaptation, and tests. Repair only independently reproduced
  Critical/Important findings with focused RED/GREEN cycles.

- [ ] **Step 4: Run one final full assurance gate**

  Do not overlap it with the active DGX control. Record pressure and original exit.

- [ ] **Step 5: Commit and push the verified SFORA slice**

  Preserve configured operator identity and add no AI attribution. Verify local and
  remote commit equality and a clean tracked worktree. Leave the four protected
  pre-existing untracked paths untouched.

### Task 8: Separately authorized post-control execution

**Files:**
- No source edits.

**Interfaces:**
- Consumes: Task 7 commit and three sealed epoch-60 control checkpoints.
- Produces: one claim-ineligible Stage-A result or one terminal failure receipt.

- [ ] **Step 1: Verify control completion without reading partial metrics**

  Require all three final seed receipts, final checkpoints, and aggregate receipt.

- [ ] **Step 2: Run exact pinned-module JVP preflight only**

  Seal the backend choice before scientific input roles are opened.

- [ ] **Step 3: Request and execute one bounded Stage-A attempt**

  Budget less than one DGX hour. No restart or adaptive repair after any scientific
  row begins.

- [ ] **Step 4: Validate and record the canonical outcome**

  `PASS_ONWARD` permits only a new Stage-B preregistration. Every other outcome stops
  the RSTA line. No benchmark or test-band run follows automatically.
