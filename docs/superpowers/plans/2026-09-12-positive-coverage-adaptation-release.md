# Positive-Coverage Adaptation Release Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:test-driven-development and execute this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Release deterministic, dataset-agnostic class-exposure scheduling and hard-negative mining around Sfora's positive-coverage objective.

**Architecture:** Extend the existing teacher-anchored distillation module with small immutable scheduling and mining primitives. Dataset-specific launchers consume those primitives but do not enter the public package. Scientific receipts and official-test policy stay in tracked experiment code.

**Tech Stack:** Python 3.12+, NumPy 2, PyTorch 2.4+, pytest, Ruff, mypy.

**Spec:** `docs/superpowers/specs/2026-09-12-positive-coverage-adaptation-release-design.md`

## Global Constraints

- No dataset names, paths, split rules, or metric thresholds in `src/sfora`.
- Preserve exact lower-row tie authority without a full score sort or device transfer.
- Preserve ineligible rows in the negative bank while excluding them from anchor sampling.
- Add behavior through focused RED/GREEN cycles and run full repository assurance once.

---

### Task 1: Dataset-invariant class-exposure scheduling

**Files:**
- Modify: `src/sfora/teacher_anchored_distillation.py`
- Modify: `src/sfora/__init__.py`
- Modify: `tests/test_teacher_anchored_distillation.py`

**Interfaces:**
- Produces: `ClassBalancedAnchorSchedule`, `exposure_normalized_update_count`, `class_balanced_anchor_schedule`

- [ ] **Step 1: Write failing update-count and schedule tests**

  Assert the reference case is 2,000 updates, 100 eligible classes produce 23,
  3,985 produce 881, singleton rows are not anchors, every update has distinct
  classes and rows, identical seeds reproduce the matrix/digest, and invalid
  concrete types/shapes/inventories fail closed.

- [ ] **Step 2: Preserve focused RED**

  Run: `.venv/bin/pytest -q tests/test_teacher_anchored_distillation.py -k 'exposure_normalized or class_balanced_anchor'`

  Expected: import failure for the three absent symbols.

- [ ] **Step 3: Implement immutable schedule authority**

  Use integer numerator/denominator arithmetic for the ceiling. Build eligible
  class row inventories once, use one caller-seeded PCG64 generator, return a
  defensive read-only copy, and hash little-endian shape/data plus configuration.

- [ ] **Step 4: Run focused GREEN**

  Run the Task 1 selector and require every new node to pass.

### Task 2: Exact device-resident hard-negative membership

**Files:**
- Modify: `src/sfora/teacher_anchored_distillation.py`
- Modify: `src/sfora/__init__.py`
- Modify: `tests/test_teacher_anchored_distillation.py`

**Interfaces:**
- Consumes: float32 unit embedding tensors and int64 label tensors on one device
- Produces: `stable_different_class_topk(anchor_codes, bank_codes, anchor_labels, bank_labels, *, k)`

- [ ] **Step 1: Write scalar differential and mutation tests**

  Compare random inputs and exact cutoff ties against a Python scalar oracle;
  require lower row ordinals at the boundary, different-label membership, exact
  width, device residence, and rejection of nonfinite/non-unit/shape/type drift.

- [ ] **Step 2: Preserve focused RED**

  Run: `.venv/bin/pytest -q tests/test_teacher_anchored_distillation.py -k stable_different_class_topk`

- [ ] **Step 3: Implement bounded top-k cutoff selection**

  Mask same-label scores, obtain only the kth cutoff with `torch.topk`, combine
  all strictly-above rows with the lowest equal-score ordinals using cumulative
  equality counts, and return contiguous `[anchors, k]` int64 membership. Explicitly
  disable ambient autocast around the float32 authority calculation.

- [ ] **Step 4: Run focused GREEN and a real-width timing check**

  Require the focused tests and one 128-by-47,704 CUDA call to complete without
  CPU transfer or a full sort. Record timing diagnostically, not as a public SLA.

### Task 3: Tracked experiment and provenance boundary

**Files:**
- Create: `scripts/positive_coverage_artifacts.py`
- Create: `scripts/replay_sop_retrieval_local_rank.py`
- Create: `scripts/run_sop_positive_coverage_metric.py`
- Create: `scripts/run_cub_positive_coverage_replication.py`
- Create: `scripts/run_inshop_positive_coverage_replication.py`
- Create: `tests/test_positive_coverage_artifacts.py`

**Interfaces:**
- Consumes: authenticated local embedding archives, explicit split adapter, fixed recipe, seed
- Produces: checkpoints followed by canonical complete receipt containing every input, schedule, state, runtime, and metric digest

- [ ] **Step 1: Write strict CLI/provenance REDs**

  Reject network URIs, absent/duplicate flags, output overwrite, unregistered
  dataset adapters, and missing hashes. Require both control and treatment
  checkpoints to exist and authenticate before the complete receipt is written.

- [ ] **Step 2: Implement adapters outside the library**

  Keep SOP, CUB, and In-Shop archive parsing in explicit script-local adapters.
  Call only the Task 1/2 public primitives for schedules and mining. Include source
  revision, driver SHA, archive SHA, ordered split SHA, schedule SHA, parameter SHA,
  environment versions, seed, and metrics in sorted compact newline JSON. Publish
  checkpoints by exclusive temporary creation and no-replace linking before the
  receipt, preserve competing writers' partials/finals, and make replication
  checkpoints self-contained by storing both the base affine head and adaptation.

- [ ] **Step 3: Replay one retained seed per dataset**

  Require exact schedule and parameter hashes and numerically identical scientific
  fields after removing elapsed-time fields.

### Task 4: Assurance and frozen evaluation

**Files:**
- Modify: `docs/research/` evidence ledger selected by repository convention

**Interfaces:**
- Consumes: terminal multi-seed receipts and checkpoints
- Produces: verified release commit and frozen official-test protocol

- [ ] **Step 1: Run focused and full assurance**

  Run the affected pytest files, Ruff, mypy, `python -m compileall`, dependency-complete
  Python discovery, and the repository's full assurance command. Stop at the first
  failure, repair narrowly, then repeat the final full gate once.

- [ ] **Step 2: Record all seeds and negative replications**

  Report SOP, In-Shop, and CUB without suppressing failures. Separate packed and
  float metrics, treatment versus pooled control, and class/identity-clustered bounds.

- [ ] **Step 3: Freeze official-test confirmation**

  After recipe and seed panel are immutable, evaluate SOP official TEST once. A
  second backbone/dataset is required before claiming architecture-independent SOTA.
