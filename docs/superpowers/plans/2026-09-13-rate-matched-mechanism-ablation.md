# Rate-Matched PQ Mechanism Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and execute a strict mechanism evaluator that separates representation geometry,
quantization efficiency, optimizer convergence, and fixed query overhead.

**Architecture:** A focused local-only experiment script reuses the authenticated transfer archive
and retrieval primitives, adds bounded float retrieval, fixed representation construction, staged
timing, strict receipt validation, and atomic publication. It does not alter the public codec API.

**Tech Stack:** Python 3.12, PyTorch, NumPy, scikit-learn, pytest, Ruff, mypy.

**Spec:** `docs/superpowers/specs/2026-09-13-rate-matched-mechanism-ablation-design.md`

## Global Constraints

- Work only in the Sfora repository and push the configured operator identity to `master`.
- Fit transforms/codebooks on official training rows only; evaluation labels never affect fitting.
- Keep all distance computations query-bounded and all outputs `claim_eligible=false`.
- Do not claim ANN serving p99 from exhaustive-scan component timing.
- Preserve exact archive/source authority and atomic mode-0600 publication.

---

### Task 1: Bounded float retrieval and neighborhood overlap

**Files:**
- Modify: `scripts/evaluate_rate_matched_pq_transfer.py`
- Modify: `tests/test_evaluate_rate_matched_pq_transfer.py`

**Interfaces:**
- Produces: `score_float_retrieval(queries, gallery, labels, *, batch_size, neighbor_width)`.
- Produces: per-query AP@R, Recall@1, stable top-100 ordinals, and aggregate metrics.

- [x] Write tests for leave-self-out behavior, stable ties, R greater than one, bounded distance
  calls, nonfinite rejection, and exact neighborhood overlap.
- [x] Run the focused tests and preserve the missing-interface RED.
- [x] Implement the bounded squared-L2 scorer by reusing `_lowest_distance_candidates`.
- [x] Run the focused tests and static checks to GREEN.
- [x] Commit the focused reusable scoring slice.

### Task 2: Frozen representation construction

**Files:**
- Create: `scripts/evaluate_rate_matched_pq_mechanism.py`
- Create: `tests/test_evaluate_rate_matched_pq_mechanism.py`

**Interfaces:**
- Produces: `MechanismConfig`, `Representation`, `build_representations`, and strict CLI parsing.
- Consumes: authenticated `TransferArchive` and train-only rows.

- [ ] Write tests fixing all six representation names/order, train-only mean/PCA fitting,
  deterministic random orthonormal projection, normalization semantics, and centered-L2 ranking
  invariance.
- [ ] Run the focused tests and preserve the missing-interface RED.
- [ ] Implement the minimal constructors with float64 fitting arithmetic and float32 contiguous
  outputs.
- [ ] Run focused tests, Ruff, mypy, py_compile, and `git diff --check`.
- [ ] Commit the representation slice.

### Task 3: Quantized mechanism arms and convergence control

**Files:**
- Modify: `scripts/evaluate_rate_matched_pq_mechanism.py`
- Modify: `tests/test_evaluate_rate_matched_pq_mechanism.py`

**Interfaces:**
- Produces: fixed OPQ24/PQ24/OPQ32 arm fitting and ADC scoring for seeds 50 through 54.
- Produces: reconstruction error trajectories and matched-wall-time/convergence classifications.

- [ ] Write tests for exact arm matrix, code widths, block widths, fitter inputs, seed propagation,
  fit schedules, ADC-vs-decoded-distance agreement, and train/evaluation separation.
- [ ] Run the focused RED.
- [ ] Implement the arm loop using existing generic quantizers without changing their public API.
- [ ] Run focused GREEN and static checks.
- [ ] Commit the quantized-arm slice.

### Task 4: Stage timing, memory, and strict receipt

**Files:**
- Modify: `scripts/evaluate_rate_matched_pq_mechanism.py`
- Modify: `tests/test_evaluate_rate_matched_pq_mechanism.py`

**Interfaces:**
- Produces: raw transform/ADC/top-k nanoseconds, recomputed percentiles, peak RSS, contrasts,
  classifications, `validate_mechanism_result`, and `canonical_mechanism_result_bytes`.

- [ ] Write mutation tests for every raw timing, percentile, metric, overlap, fitted byte count,
  source/input identity, classification, seed, and `claim_eligible` field.
- [ ] Run the authority RED.
- [ ] Implement synchronized warm-up plus measurement, resource sampling, strict recomputation,
  exclusive partial reservation, and canonical publication.
- [ ] Run complete focused GREEN and static checks.
- [ ] Commit and push the evaluator, then verify HEAD/origin/remote equality and a clean tree.

### Task 5: Frozen DGX mechanism screen

**Files:**
- Modify: `docs/positive_coverage_adaptation_result_2026-09-12.md`

**Interfaces:**
- Consumes: committed evaluator and authenticated CUB/SOP UniCOM-B16 archives.
- Produces: one canonical CUB receipt followed, only if structurally valid, by one canonical SOP
  receipt for the unchanged matrix.

- [ ] Run the exact CUB cell once under RSS/PSI/swap/wall monitoring and preserve its receipt.
- [ ] Classify the mechanism without changing arms, seeds, metrics, or thresholds.
- [ ] Run the exact SOP cell once; do not rerun or tune based on CUB/SOP results.
- [ ] Record only verified findings, limitations, receipt hashes, and resource evidence.
- [ ] Verify docs, commit, push to `master`, and confirm remote equality and a clean tree.
