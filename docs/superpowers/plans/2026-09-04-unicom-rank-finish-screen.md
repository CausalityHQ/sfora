# UniCOM Rank-Finish Screen Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and execute one claim-ineligible DGX screen of a deterministic Smooth-AP rank-finishing phase resumed from the preserved imprinted epoch-4 checkpoint.

**Architecture:** A focused library module owns the deterministic identity-balanced schedule and Smooth-AP loss. A thin script authenticates the preserved FEPF evidence, restores the model/optimizer/scheduler, runs epochs 5--8, evaluates only at epochs 6 and 8, and emits one canonical result outside the FEPF artifact tree.

**Tech Stack:** Python 3.12+, PyTorch, NumPy, pytest, existing Sfora UniCOM loaders and retrieval evaluator.

**Spec:** `docs/superpowers/specs/2026-09-04-unicom-rank-finish-screen-design.md`

## Global Constraints

- Work only in the Sfora repository and preserve the untracked `img` path.
- Training must never consume holdout or standard-test records.
- The result is always `claim_eligible=false`; no publication claim follows from one seed.
- Run one DGX process with the stops and gates fixed in the spec.

---

### Task 1: Deterministic balanced schedule

**Files:**
- Create: `src/sfora/unicom_rank_finish.py`
- Create: `tests/test_unicom_rank_finish.py`

**Interfaces:**
- Produces: `identity_balanced_batches(labels, *, batch_size, images_per_identity, seed, epoch, steps) -> tuple[tuple[int, ...], ...]`.

- [ ] Write tests for exact replay, epoch separation, 32 distinct identities per batch, four examples per identity, no duplicate index before per-identity cycling, invalid labels, and impossible shapes.
- [ ] Run `.venv/bin/pytest -q tests/test_unicom_rank_finish.py` and preserve the missing-interface RED.
- [ ] Implement the smallest deterministic schedule using NumPy PCG64 streams derived from `(seed, epoch)`.
- [ ] Run the focused tests to GREEN.
- [ ] Commit the schedule slice.

### Task 2: Smooth AP loss in deployment geometry

**Files:**
- Modify: `src/sfora/unicom_rank_finish.py`
- Modify: `tests/test_unicom_rank_finish.py`

**Interfaces:**
- Produces: `smooth_ap_finish_loss(embeddings, labels, *, dimensions=512, temperature=0.01) -> torch.Tensor`.

- [ ] Write scalar-oracle tests for normalized 768-D input, first-512-coordinate squared distances, exact positive/self masks, perfect and reversed rankings, finite gradients, ties, and rejected nonfinite/shape/identity inputs.
- [ ] Run the exact loss tests and preserve RED.
- [ ] Implement the fixed-order anchor/positive formulation from the spec without a memory bank or holdout input.
- [ ] Run focused tests to GREEN, then run the whole new test file.
- [ ] Commit the loss slice.

### Task 3: Authenticated resume screen

**Files:**
- Create: `scripts/screen_unicom_rank_finish.py`
- Create: `tests/test_screen_unicom_rank_finish.py`

**Interfaces:**
- Consumes: the Task-1 schedule and Task-2 loss plus existing UniCOM model, checkpoint, partition, and retrieval helpers.
- Produces: canonical `unicom-rank-finish-screen-v1` JSON.

- [ ] Write CLI/authority tests covering exact source/checkpoint/run-receipt/partition digests, output no-replace behavior, optimization-only training records, epoch-6 abort, epoch-8 reject/promote gates, canonical bytes, and `claim_eligible=false`.
- [ ] Run `.venv/bin/pytest -q tests/test_screen_unicom_rank_finish.py` and preserve RED.
- [ ] Implement strict argument parsing, evidence authentication, state restoration, the four-epoch loop, epoch-6/8 evaluation, resource accounting, and canonical result publication.
- [ ] Run the script tests and the library tests to GREEN.
- [ ] Run Ruff, py_compile, and `git diff --check`; commit and push the verified slice to `master` without attribution trailers.

### Task 4: DGX falsifier

**Files:**
- No repository changes.

**Interfaces:**
- Consumes: a clean detached checkout at the exact implementation commit and the preserved epoch-4 control authorities.
- Produces: one canonical claim-ineligible screen result.

- [ ] Validate the clean checkout and all input digests without loading CUDA state.
- [ ] Run the focused DGX CUDA canary for balanced batches, loss gradients, and one optimizer step.
- [ ] Launch exactly one process with the two-hour, PSI, OOM, and progress stops from the spec.
- [ ] Poll the same process at intervals no longer than 55 seconds and report user-visible liveness approximately every five minutes.
- [ ] Authenticate the terminal JSON, classify it using the frozen epoch-6/8 gates, and delete only registered scratch after PID clearance.

### Task 5: Decision

**Files:**
- Modify only the appropriate research ledger after a terminal result exists.

- [ ] If rejected, record the exact failure and close this Smooth-AP finish configuration.
- [ ] If exploratory-only, record it but do not spend on confirmation.
- [ ] If promoted, write a new multi-seed confirmation spec with non-inferiority recall gates before any further GPU run.
- [ ] Run the research-doc validator and `git diff --check`, then commit/push the evidence-only update.

