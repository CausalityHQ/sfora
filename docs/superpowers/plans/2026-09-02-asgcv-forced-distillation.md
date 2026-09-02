# ASG-CV Forced-Gradient Distillation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Capture deterministic relation-correct Qwen gradients and test whether the existing rank-16 student predicts them on disjoint train-only Cars classes.

**Architecture:** A pure authority module defines schedules, receipts, and metrics. A local-only runner resumes Qwen captures, fits the existing predictor with a frozen recipe, and emits one canonical validation result.

**Tech Stack:** Python 3.13, NumPy, PyTorch, existing Sfora Qwen adapter and ASG-CV predictor.

**Spec:** `docs/superpowers/specs/2026-09-02-asgcv-forced-distillation-design.md`

## Global Constraints

- Work only in the Sfora repository.
- Never read Cars196 official-test artifacts.
- Use exact local model/input/source authorities and deterministic schedules.
- Preserve relation-correct targets as float32 `[2,196,2048]` arrays.
- Use TDD and run the narrowest test after every change.

---

### Task 1: Authority, schedule, and capture receipt

**Files:**
- Create: `src/sfora/asgcv_forced_distill.py`
- Create: `tests/test_asgcv_forced_distill.py`

**Interfaces:**
- Produces: `ForcedDistillAuthority`, `ForcedDistillCapture`, `build_forced_distill_schedule`, `canonical_forced_distill_capture_bytes`.

- [ ] Write failing tests for exact schema/types, balanced deterministic train and validation schedules, role-separated seeds, array shape/digest binding, relation-sign orientation, and mutation rejection.
- [ ] Run `uv run pytest -q tests/test_asgcv_forced_distill.py` and require failure at missing symbols.
- [ ] Implement only the authority, schedule, capture, and canonical serialization needed by those tests.
- [ ] Rerun the focused test and require green.
- [ ] Commit the independently reviewable authority slice.

### Task 2: Reopenable local capture runner

**Files:**
- Create: `scripts/run_asgcv_forced_distill.py`
- Create: `tests/test_run_asgcv_forced_distill.py`

**Interfaces:**
- Consumes: Task 1 authority and existing `QwenSagaAdapter.collapsed_verdict_patch_gradient`.
- Produces: `capture_forced_distill_pair`, `run_capture_phase`, strict local CLI.

- [ ] Write failing tests with a fake adapter for fixed SAME/DIFFERENT prefix order, exact relation-sign target orientation, atomic three-file writes, digest validation, ordinal-gap rejection, and idempotent resume.
- [ ] Run the focused runner test and require the missing-interface RED.
- [ ] Implement capture by persisting target `patch_tokens` and `relation_sign * predicted_gradient`; never persist a generated completion.
- [ ] Add strict CLI tests that reject network, official-test, duplicate, and unknown flags.
- [ ] Rerun both focused files and require green.
- [ ] Commit the capture slice.

### Task 3: Frozen student fit and validation metrics

**Files:**
- Modify: `src/sfora/asgcv_forced_distill.py`
- Modify: `scripts/run_asgcv_forced_distill.py`
- Modify: `tests/test_asgcv_forced_distill.py`
- Modify: `tests/test_run_asgcv_forced_distill.py`

**Interfaces:**
- Produces: `ForcedDistillResult`, `fit_forced_distill_predictor`, `evaluate_forced_distill`, `canonical_forced_distill_result_bytes`.

- [ ] Write failing tests for fixed initialization/order/optimizer recipe, unchanged exact targets, canonical predictor-state digest, per-pair cosine recomputation, median/positive-rate gates, nonfinite rejection, and result mutation rejection.
- [ ] Run the exact new nodes and preserve the RED.
- [ ] Implement 20 ordinal-order epochs of streaming batch-one AdamW (`lr=1e-3`, `weight_decay=1e-4`) with the existing `predictor_training_loss` and a source-bound 256-dimensional SRHT; evaluate all 32 validation pairs once without adapting on them.
- [ ] Rerun the exact nodes, then both focused files, and require green.
- [ ] Commit the fit/evaluation slice.

### Task 4: Repository verification and scientific execution

**Files:**
- Modify: `docs/asgcv_forced_p32_result_2026-09-02.md` only after a terminal result.

**Interfaces:**
- Consumes: committed runner and authenticated DGX inputs.
- Produces: canonical claim-ineligible distillation result and evidence update.

- [ ] Run focused tests, Ruff, py_compile, `git diff --check`, then the repository's dependency-complete Python test command once.
- [ ] Commit and push the verified Sfora slice with configured operator identity and no attribution trailers.
- [ ] Export and authenticate the exact commit on DGX, run one original capture/fit process, and monitor memory PSI/GPU/RSS without duplicates.
- [ ] Validate canonical bytes, all capture digests, predictor-state digest, metrics, process clearance, and no official-test access.
- [ ] Record the terminal evidence, rerun the research-doc validator and diff check, then commit/push the one-doc update.
