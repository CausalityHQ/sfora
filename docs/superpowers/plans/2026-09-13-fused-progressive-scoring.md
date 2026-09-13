# Fused Progressive Candidate Scoring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a calibrated fixed-shape compiled scorer for progressive residual candidates, validate it on CUDA, then use it to test a fused 3-to-5-plane cascade before optimizing candidate ADC/top-k.

**Architecture:** A new scoring module wraps a tensor-only compiler-friendly graph while retaining the existing eager codec as reference and fallback. Fixed geometry and padded tail batches prevent timed recompilation; calibration and boundary repair prevent silent numerical quality drift. Candidate generation remains a separately measured kernel.

**Tech Stack:** Python 3.12, PyTorch 2.x, TorchInductor/Triton on CUDA, pytest, NumPy, Ruff, mypy.

**Spec:** `docs/superpowers/specs/2026-09-13-fused-progressive-scoring-design.md`

## Global Constraints

- Work only in `/home/rb/worktrees/sfora-emafactorial-release`; never modify Borsuk.
- Do not change the progressive residual wire format or existing eager score semantics.
- No original gallery vectors, labels, truth IDs, or evaluation-query statistics enter fitting or compilation.
- Compile and steady-state timing are separate; every timed batch has one fixed physical shape.
- Every result remains `claim_eligible=false` until untouched multi-seed, multi-dataset validation passes.

---

### Task 0: Tail-safe compiled base-PQ ADC/top-k

**Files:**
- Create: `src/sfora/pq_candidate_scoring.py`
- Create: `tests/test_pq_candidate_scoring.py`
- Modify: `src/sfora/__init__.py`

**Interfaces:**
- Produces: `PqCandidateScoringSpec(metric, candidate_width, compiled_batch_rows, row_tile=64)`.
- Produces: `CompiledPqCandidateScorer` and `compile_pq_candidate_scorer`.
- Returns owned candidate ordinals and backend/padding evidence without exposing full score matrices.

- [ ] **Step 1: Write RED geometry, masking, and ownership tests**

Use hand-derived PQ codebooks and gallery sizes both divisible and non-divisible by 64. Require sentinel
scores to be masked, no returned ordinal to address padding, fixed query-tail padding, and cloned output
ownership across repeated compiler replays. Cover bool-as-int, width/range/device/dtype/contiguity drift,
duplicate candidates, compile failure, and nonfinite scores.

- [ ] **Step 2: Write RED eager differential tests**

For plain PQ and OPQ, compare every candidate-set ID with `asymmetric_squared_distances` plus eager top-k.
Keep query normalization/rotation outside the compiled graph and include it in timing.

- [ ] **Step 3: Run the focused RED**

Run `pytest -q tests/test_pq_candidate_scoring.py`; require missing interfaces only.

- [ ] **Step 4: Implement the minimum tail-safe scorer**

Pad gallery codes to `row_tile`, mask sentinels to the metric-worst score, compile fixed-shaped ADC/top-k,
and clone each CUDA-graph output immediately. Retain a portable eager fallback and exact physical padding
accounting.

- [ ] **Step 5: Run focused/regression GREEN and static checks**

Run the new file, existing PQ/progressive/evaluator tests, Ruff, strict mypy, py_compile, and diff-check.

- [ ] **Step 6: Run one DGX CUDA differential**

Require all 5,000 candidate IDs on GloVe and at least 4,999.999 mean overlap on SIFT, at least 2x speed,
no sentinel IDs, and no hidden fallback. Record compilation separately.

- [ ] **Step 7: Commit and push Task 0**

Commit only the module/export/tests after verification and confirm canonical `master` equality.

### Task 1: Validated scoring geometry and pure eager graph

**Files:**
- Create: `src/sfora/progressive_residual_scoring.py`
- Create: `tests/test_progressive_residual_scoring.py`
- Modify: `src/sfora/__init__.py`

**Interfaces:**
- Produces: `ProgressiveScoringSpec`.
- Produces: an internal tensor-only score graph for angular and squared L2.
- Produces: `CompiledProgressiveCandidateScorer` and `compile_progressive_candidate_scorer`.

- [ ] **Step 1: Write RED validation tests**

Cover bool-as-int, invalid widths, repair width smaller than return width, candidate/spec geometry drift,
device drift, duplicate/out-of-range candidates, partial final batches, and nonfinite inputs.

- [ ] **Step 2: Run the focused RED**

Run `pytest -q tests/test_progressive_residual_scoring.py`; require missing interfaces rather than a
fixture error.

- [ ] **Step 3: Implement the spec and tensor-only eager graph**

Reuse codec codebooks and exact prefix-center arithmetic. Keep validation outside the graph. Pad a final
short query batch by repeating its last valid row and slice outputs back to the logical row count.

- [ ] **Step 4: Run focused GREEN and static checks**

Run the scoring and quantization tests, Ruff, strict mypy, py_compile, and `git diff --check`.

- [ ] **Step 5: Commit and push Task 1**

Commit only the focused module, export, and tests; push `HEAD:master` and verify exact remote equality.

### Task 2: Compile, calibration, fallback, and boundary repair

**Files:**
- Modify: `src/sfora/progressive_residual_scoring.py`
- Modify: `tests/test_progressive_residual_scoring.py`

**Interfaces:**
- Produces: fixed-shape `torch.compile(fullgraph=True, dynamic=False)` activation.
- Produces: truth-free calibration evidence and explicit eager fallback reason.
- Produces: boundary-prefix eager repair and reread byte accounting.

- [ ] **Step 1: Write RED compiler-boundary tests**

Inject a compile callable to test one compile, cache reuse, padded-tail reuse, compile failure, nonfinite
scores, membership mismatch, exact ordinal ties, and deterministic fallback without requiring CUDA.

- [ ] **Step 2: Write RED boundary-repair tests**

Construct approximate score inversions inside and outside the repair prefix. Require exact eager order
inside the prefix, explicit fallback when the runtime boundary is unsafe, and correct reread bytes.

- [ ] **Step 3: Run RED and implement the minimum behavior**

Compile score generation only. Keep validation and final stable selection outside the compiled graph.
Never time first compilation as query latency.

- [ ] **Step 4: Run focused/regression GREEN and static checks**

Run scoring, progressive codec, PQ reranking, evaluator tests, Ruff, strict mypy, py_compile, and diff-check.

- [ ] **Step 5: Commit and push Task 2**

Verify clean `HEAD == origin/master == ls-remote refs/heads/master`.

### Task 3: CUDA differential and performance gate

**Files:**
- Modify: `tests/test_progressive_residual_scoring.py`
- Modify: `scripts/evaluate_progressive_residual_ann.py`
- Modify: `tests/test_evaluate_progressive_residual_ann.py`

**Interfaces:**
- Produces: canonical fused-score screen with compile time, latency distribution, throughput, memory,
  numerical deltas, membership equality, recall, and fallback counts.

- [ ] **Step 1: Add skipped-without-CUDA differential tests**

Cover both metrics, random values, exact ties, subnormals, large finite values, query counts both divisible
and non-divisible by compiled batch rows, and return widths 10/100.

- [ ] **Step 2: Add strict receipt validation tests**

Recompute percentiles and quality; reject missing raw samples, metric/shape/backend drift, noncanonical
JSON, score nonfiniteness, membership mismatch, hidden fallback, and `claim_eligible=true`.

- [ ] **Step 3: Run local GREEN and synchronize one exact commit to DGX**

Use a content-addressed bundle and verify the detached DGX checkout before execution.

- [ ] **Step 4: Run one GloVe and one SIFT burned-query screen**

Require identical top-100 membership and recall, at least 2x speedup, and zero fallback for 1,000 queries.
Preserve original sessions and canonical results; do not retry scientific failures.

- [ ] **Step 5: Deliver the verified kernel slice**

Run full local assurance once, commit, push, and record only verified evidence.

### Task 4: Fused 3-to-5-plane cascade

**Files:**
- Modify: `src/sfora/progressive_residual_scoring.py`
- Modify: `tests/test_progressive_residual_scoring.py`
- Modify: `scripts/evaluate_progressive_residual_ann.py`
- Modify: `tests/test_evaluate_progressive_residual_ann.py`

**Interfaces:**
- Produces: one fused first-pass score at three planes and five-plane refinement for fixed shortlist widths
  128, 192, and 256.

- [ ] **Step 1: Write hand-derived incremental-score RED tests**

Require the five-plane reconstruction to equal full decode. For L2 verify the exact incremental squared
distance identity; for angular verify numerator and norm updates before final normalization.

- [ ] **Step 2: Implement one-read fused cascade and exact accounting**

Do not decode a complete three-plane tensor and then call the five-plane scorer. Gather shared base codes
and scales once and read only the selected fine planes for the shortlist.

- [ ] **Step 3: Run fixed-width development falsifiers**

Stop if every width either loses more than 0.002 absolute recall on either dataset or fails to improve
measured latency/bytes by 20%. Keep width 192 only if it wins under the implemented kernel.

- [ ] **Step 4: Verify, commit, and push**

Run the focused and repository gates before delivery.

### Task 6: Untouched and cross-dataset release gate

**Files:**
- Modify: `docs/positive_coverage_adaptation_result_2026-09-12.md`

- [ ] **Step 1: Freeze profiles before reading untouched results**

Register queries 1,000..9,999, seeds 50..52, batch sizes 1/8/32/128, a third high-dimensional dataset,
and one 10M-or-larger workload.

- [ ] **Step 2: Run optimized progressive and equally optimized controls**

Include scalar, residual-PQ, Faiss PQ/FastScan, and feasible official LVQ/RaBitQ controls. Report p50/p95/p99,
throughput, bytes, build time, peak memory, and confidence intervals.

- [ ] **Step 3: Apply release/publication gates**

Require at least 20% speed or storage improvement at matched recall on three materially different datasets.
If the result misses, report the exact frontier rather than describing it as SOTA.

- [ ] **Step 4: Run repository assurance and deliver**

Run dependency-complete pytest, Ruff, strict mypy, py_compile, docs validation, and diff-check; commit and
push only verified evidence.
