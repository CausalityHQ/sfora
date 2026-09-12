# Single-Stage Differential PQ Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add and run a generic joint projection/PQ screen that targets neighborhood-differential quantization error while retaining exact 24-byte asymmetric scoring.

**Architecture:** Extend the existing tested PQ loss with one explicit differential term, then add a separate authenticated SOP driver that owns the four-arm experiment and publication. Keep scientific primitives in `src/sfora`; keep SOP paths, hashes, controls, and gates in `scripts`.

**Tech Stack:** Python 3.12/3.13, PyTorch, NumPy, pytest, Ruff, mypy, CUDA.

**Spec:** `docs/superpowers/specs/2026-09-13-single-stage-differential-pq-design.md`

## Global Constraints

- Work only in the Sfora repository.
- Never touch the SOP official test during fitting, selection, or this development screen.
- Preserve exactly 24 database bytes/vector, zero per-vector sidecar, and the existing 24-lookup asymmetric PQ scorer.
- Use one original DGX process and preserve its terminal result; do not overlap or silently restart scientific runs.
- Keep results `claim_eligible=false`.

---

### Task 1: Differential quantization loss

**Files:**
- Modify: `src/sfora/product_quantization.py`
- Modify: `tests/test_product_quantization.py`

**Interfaces:**
- Extend `NeighborhoodAdcDistillationLoss` with `differential: torch.Tensor`.
- Extend `neighborhood_adc_distillation_loss(..., differential_weight: float = 0.0, neighbor_pairs: torch.Tensor | None = None)`.
- Produce `neighborhood_differential_error(values, hard_reconstructions, neighbor_pairs)` as a strict generic primitive.

- [x] **Step 1: Write REDs** for the independent differential formula, gradients into rows and selected codewords, invalid pairs/nonfinite inputs, and exact zero-weight compatibility.
- [x] **Step 2: Run** `pytest -q tests/test_product_quantization.py -k 'differential or neighborhood_adc'` and require the missing interface/formula RED.
- [x] **Step 3: Implement minimally.** Compute residuals from hard decoded codewords, gather registered pair endpoints, average squared residual differences, and add `differential_weight * differential` to the existing total.
- [x] **Step 4: Rerun the focused file** and require GREEN.

### Task 2: Four-arm authenticated driver

**Files:**
- Create: `src/sfora/joint_pq_projection.py`
- Create: `tests/test_joint_pq_projection.py`
- Create: `scripts/run_sop_single_stage_differential_pq.py`
- Create: `tests/test_run_sop_single_stage_differential_pq.py`

**Interfaces:**
- Produce generic `JointPqTrainingSpec`, `JointPqProjection`, and `fit_joint_pq_projection` in the library.
- Produce driver-owned `initialize_joint_pq_arms`, `joint_pq_decision`, `canonical_joint_pq_receipt`, and explicit `main`.
- Consume the existing paired snapshots, direct affine checkpoint, PQ24 parent checkpoint/receipt, and frozen failure receipt.

- [x] **Step 1: Write library REDs** for exact hard-forward training, step-zero equality, input-specific gradients, fixed update order, finite traces, and immutable returned checkpoint tensors.
- [x] **Step 2: Run** `pytest -q tests/test_joint_pq_projection.py` and preserve the missing-module RED, then implement the minimal generic trainer and require GREEN.
- [x] **Step 3: Write driver REDs** for exact four-arm names, frozen fitting-only candidate/pair identities, deployment byte/lookup accounting, decision boundaries, canonical no-clobber publication, source authority, and CLI refusal.
- [x] **Step 4: Implement the driver** by reusing `load_paired_train_archives`, `deterministic_class_partition`, the existing label-free candidate builder, parent PQ loader, exact PQ scorer, and the generic trainer. Do not duplicate metric or codec implementations.
- [ ] **Step 5: Add a reduced end-to-end test** that performs real optimization and validates the reloaded canonical receipt/model artifact.
- [x] **Step 6: Run the complete focused tests** and require GREEN.

### Task 3: Assurance and independent review

**Files:**
- Modify only the four Task-1/2 code files and these two docs.

**Interfaces:**
- Produce one clean committed source revision suitable for DGX execution.

- [x] **Step 1: Run targeted Ruff, strict mypy on production files, `py_compile`, and `git diff --check`.**
- [x] **Step 2: Run the complete Python test suite once** and require zero failures.
- [x] **Step 3: Obtain cold read-only Astra and Fable/Opus reviews** of the exact diff and reconcile only concrete correctness/scientific blockers.
- [ ] **Step 4: Commit and push to canonical `master`; verify HEAD, tracking ref, remote ref, and clean status.**

### Task 4: DGX screen and evidence

**Files:**
- Modify after the terminal: `docs/positive_coverage_adaptation_result_2026-09-12.md`

**Interfaces:**
- Consume the Task-3 commit and existing authenticated SOP artifacts.
- Produce one canonical seed-0 result and model, plus conditional seed-1/2 results only after a preregistered go.

- [ ] **Step 1: Deploy a clean detached checkout and run the focused DGX tests.** Confirm the official test is absent from the invocation.
- [ ] **Step 2: Run one monitored seed-0 process.** Record GPU/RSS/progress, preserve the original terminal, and do not duplicate or restart after a scientific terminal.
- [ ] **Step 3: Authenticate receipt/model bytes and classify every absolute/mechanism gate.** Stop on kill or ambiguous; only a go authorizes unchanged seeds 1 and 2.
- [ ] **Step 4: Record positive and negative evidence, verify docs, commit/push, and notify the operator.**
