# Representation-Ceiling Diagnosis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and execute a train-only diagnosis that separates source-information, width, head-training, and quantization limits before spending further DGX time.

**Architecture:** A dataset-independent library module implements deterministic class splits and closed-form transforms. A strict local-only research script authenticates paired archives, scores fixed arms, computes clustered uncertainty, and publishes one canonical claim-ineligible receipt. A teacher-neighborhood sampler is a conditional follow-up, never part of the first run.

**Tech Stack:** Python 3.12, PyTorch, NumPy, hashlib, pytest, Ruff, mypy.

**Spec:** `docs/superpowers/specs/2026-09-09-representation-ceiling-diagnosis-design.md`

## Global Constraints

- Keep all dataset-specific loading, metric, and receipt code outside `src/sfora`.
- Use only training rows; the first receipt must not accept test tensors.
- All class splits are deterministic, class-disjoint, and mutation-locked.
- Every output is authenticated, canonical, no-clobber, and claim-ineligible.
- Run only one monitored DGX science process and retain negative evidence.

---

### Task 1: Deterministic class partition and normalized affine primitives

**Files:**
- Create: `src/sfora/representation_ceiling.py`
- Modify: `src/sfora/__init__.py`
- Create: `tests/test_representation_ceiling.py`

**Interfaces:**
- Produces: `ClassDisjointPartition`, `deterministic_class_partition`,
  `CenteredPcaTransform`, `fit_centered_pca`, `AffineMap`,
  `fit_ridge_affine`, and `apply_normalized_affine`.
- Consumes: CPU contiguous float32 tensors and concrete integer class labels.

- [ ] **Step 1: Write failing partition tests**

  Cover determinism, exact class disjointness, ordered indices, split-seed
  sensitivity, invalid fractions/types, singleton validation classes, and no
  dependency on research scripts.

- [ ] **Step 2: Run the partition tests and preserve RED**

  Run: `pytest -q tests/test_representation_ceiling.py -k partition`
  Expected: import failure for the absent module or symbols.

- [ ] **Step 3: Implement the minimal partition API**

  Hash `seed || concrete class id` with SHA-256, order classes by digest and
  class ID, choose the nearest valid prefix to the requested fit fraction, and
  return strictly increasing row indexes plus ordered class IDs. Reject bool as
  int and every invalid or degenerate authority.

- [ ] **Step 4: Write and preserve affine/PCA/ridge REDs**

  Mutation-lock centered PCA fit-only behavior, deterministic sign orientation,
  float64 ridge equations, unpenalized intercept, exact ridge penalty,
  normalized application, rank/singularity/nonfinite/zero-norm failures, and
  input immutability.

- [ ] **Step 5: Implement closed-form transforms and run focused GREEN**

  Run: `pytest -q tests/test_representation_ceiling.py`
  Expected: all tests pass with no warning.

- [ ] **Step 6: Verify and commit the generic primitive slice**

  Run Ruff, strict mypy on the new module, pycompile, and `git diff --check`.
  Commit only the module, public exports, tests, spec, and plan.

### Task 2: Strict train-only ceiling evaluator

**Files:**
- Create: `scripts/probe_representation_ceiling.py`
- Create: `tests/test_probe_representation_ceiling.py`

**Interfaces:**
- Consumes: two authenticated local paired-embedding archives and the Task 1
  primitives.
- Produces: one canonical `sfora-representation-ceiling-v1` JSON receipt.

- [ ] **Step 1: Write failing authority and arm tests**

  Use synthetic paired archives to require the six frozen arms, outer seeds
  `(17,1729,65537)`, inner ridge penalties `(1e-6,1e-4,1e-2)`, fit-only
  transforms, self exclusion, ordinal ties, and no test input access.

- [ ] **Step 2: Preserve the evaluator RED**

  Run: `pytest -q tests/test_probe_representation_ceiling.py`
  Expected: missing evaluator/CLI symbols only.

- [ ] **Step 3: Implement scoring, selection, uncertainty, and decisions**

  Reuse authenticated SOP archive loading but not SOP-specific logic in the
  library. Emit all per-query AP/R@1 outcomes, 10,000 paired class resamples,
  exact selected penalties, environment, source hashes, numeric gates, and an
  exhaustive decision: `neighborhood-sampling-warranted`,
  `wider-code-warranted`, or `backbone-quality-work-warranted` with simultaneous
  flags retained when multiple gates fire.

- [ ] **Step 4: Implement strict CLI and no-clobber publication**

  Require absolute source/teacher/output paths, lowercase SHA-256 values, an
  exact 40-character source commit, and `--execute-representation-ceiling`.
  Reserve the output partial before archive loading; reject any final/partial
  collision; clean only the process-owned partial.

- [ ] **Step 5: Run focused and static GREEN gates**

  Run both new test files, Ruff, strict mypy on production sources, pycompile,
  and `git diff --check`.

- [ ] **Step 6: Obtain independent contextless Astra and Fable review**

  Give each reviewer only the concrete evaluator diff and the authenticated
  prior measurements. Repair Critical/Important correctness or scientific
  blockers with focused RED/GREEN cycles; do not add adaptive arms.

- [ ] **Step 7: Commit and verify the frozen evaluator**

  Push `HEAD:master` and require local HEAD, `origin/master`, and remote
  `refs/heads/master` equality before scientific execution.

### Task 3: One monitored DGX ceiling run

**Files:**
- Add: `docs/evidence/representation_ceiling/sop-representation-ceiling-v1.json`
- Modify: this plan checkbox state only after evidence authenticates.

**Interfaces:**
- Consumes: exact paired SOP train archives already present on DGX.
- Produces: one immutable canonical result; no model is deployed from this run.

- [ ] **Step 1: Preflight exact source, archive, environment, and output authority**

  Verify clean committed checkout, both registered archive digests, output
  absence, CUDA health, memory pressure, and the external environment. Start no
  science if any preflight differs.

- [ ] **Step 2: Run one original monitored process**

  Enforce a two-hour wall cap, RSS and PSI stop rules, forward-progress checks,
  explicit scratch cleanup, and no restart after any terminal. Preserve stdout,
  stderr, exit status, and pressure evidence.

- [ ] **Step 3: Authenticate and interpret the receipt**

  Recompute canonical bytes, every input/source hash, partitions, ridge
  selections, aggregate metrics, bootstrap bounds, and decisions. Record
  failure as evidence; do not inspect SOP test rows.

- [ ] **Step 4: Commit evidence and choose only the authorized branch**

  If the ridge gain gate passes, proceed to Task 4. If only the width gate
  fires, freeze a 192/256D bytes-quality study. Otherwise move primary quality
  work to generic backbone distillation. Push and verify remote equality.

### Task 4: Conditional teacher-neighborhood sampler

**Files:**
- Modify: `src/sfora/joint_relational_compaction.py`
- Modify: `tests/test_joint_relational_compaction.py`
- Create only after Task 3 authorization: focused sampler script/tests/evidence.

**Interfaces:**
- Consumes: fit-only source/teacher rows and a fixed sampler configuration.
- Produces: deterministic row-index batches; the trainer objective and codec
  remain unchanged.

- [ ] **Step 1: Stop unless Task 3 ridge full-width gate passed**

- [ ] **Step 2: TDD a generic deterministic teacher-neighborhood sampler**

  Require 64 uniform anchors, 15 self-excluding teacher neighbors each,
  deduplication, uniform fill to 1,024, paired seeds, finite normalized teacher
  rows, exact ties, and bounded memory.

- [ ] **Step 3: Run the frozen paired train-only comparison**

  Use exactly 1,000 updates and advance only for mean int4 gain at least 0.003,
  positive gain in all three seeds, float improvement, and positive paired
  class-bootstrap lower bound. Retain negative evidence.
