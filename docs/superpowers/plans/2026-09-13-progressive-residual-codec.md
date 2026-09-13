# Progressive Residual Codec Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a generic angular/L2 product-code residual codec whose single maximum-rate bitstream supports physically contiguous lower-rate prefixes, plus a strict evaluator for frozen ANN-Benchmarks studies.

**Architecture:** A new focused library module owns metric preparation, base PQ composition, MSB-first residual bitplanes, prefix decoding, candidate reranking, artifact validation, and exact byte accounting. A separate script owns HDF5 acquisition authority, truth evaluation, timing, and canonical claim-ineligible receipts. ANN routing and adaptive precision policy are deliberately deferred until the codec passes untouched-query and matched-storage gates.

**Tech Stack:** Python 3.12, PyTorch, NumPy, scikit-learn, h5py, pytest, Ruff, mypy.

**Spec:** `docs/superpowers/specs/2026-09-13-progressive-residual-ann-design.md`

## Global Constraints

- Work only in `/home/rb/worktrees/sfora-emafactorial-release`; never modify Borsuk.
- Fit transforms and codebooks from caller-supplied training rows only; no truth IDs or labels enter the library API.
- Support only explicit `angular` and `squared_l2` metrics; preserve each dataset's native metric.
- One maximum-rate residual encoding must decode every lower-rate prefix; separately encoded rates are controls, not progressive evidence.
- Count physical row-major bitplane bytes, the float16 scale, base codes, shared parameters, IDs, and later routing metadata separately.
- Every experiment result remains `claim_eligible=false`; exhaustive batch averages are not ANN p99.
- Use focused RED/GREEN tests, frequent commits, and push verified commits to canonical `master` with the configured operator identity and no AI attribution.

---

### Task 1: Progressive wire geometry and exact bitplanes

**Files:**
- Create: `src/sfora/progressive_residual_quantization.py`
- Create: `tests/test_progressive_residual_quantization.py`

**Interfaces:**
- Produces: `ProgressiveResidualSpec(metric, base_spec, maximum_residual_bits=8)`.
- Produces: `pack_residual_bitplanes(indexes, *, dimensions)` and `unpack_residual_prefix(planes, *, dimensions, residual_bits)`.
- Produces: `ProgressiveResidualCodes(base_codes, scales, residual_planes)` as an immutable tensor owner.

- [ ] **Step 1: Write validation and byte-arithmetic tests**

```python
def test_progressive_spec_counts_physical_prefix_bytes() -> None:
    spec = ProgressiveResidualSpec(
        metric="angular",
        base_spec=balanced_product_quantization_spec(dimensions=100, bytes_per_vector=24),
    )
    assert spec.residual_plane_bytes == 13
    assert spec.bytes_per_vector(residual_bits=3) == 65
    assert spec.bytes_per_vector(residual_bits=5) == 91
    assert spec.bytes_per_vector(residual_bits=8) == 130
```

Also reject bool-as-int, unknown metrics, residual bit counts outside 1..8, inconsistent dimensions,
non-`uint8` indexes/planes, nonzero padding bits, wrong shapes, and non-`float16` scales.

- [ ] **Step 2: Write exact packing and prefix tests**

Use rows containing `0x00`, `0x01`, `0x7f`, `0x80`, `0xfe`, and `0xff`; assert exact MSB-first bytes for dimensions 8, 9, and 100. For each `b in 1..8`, require unpacked prefixes to equal `indexes >> (8-b)`. Flip every padding bit and require rejection.

- [ ] **Step 3: Run the focused RED**

Run: `pytest -q tests/test_progressive_residual_quantization.py`

Expected: collection failure for the missing module or missing interfaces.

- [ ] **Step 4: Implement only wire types and packing**

Use tensor shifts and fixed byte weights; never serialize through Python integer lists. Store residual planes as contiguous `uint8[N, 8, ceil(D/8)]`, with planes ordered from bit 7 to bit 0. Validate concrete types and zero padding in one private helper reused by packing, unpacking, and `ProgressiveResidualCodes`.

- [ ] **Step 5: Run focused GREEN and static checks**

Run:

```bash
pytest -q tests/test_progressive_residual_quantization.py
ruff check src/sfora/progressive_residual_quantization.py tests/test_progressive_residual_quantization.py
mypy --strict src/sfora/progressive_residual_quantization.py
python -m py_compile src/sfora/progressive_residual_quantization.py tests/test_progressive_residual_quantization.py
git diff --check
```

- [ ] **Step 6: Commit and push the wire slice**

```bash
git add src/sfora/progressive_residual_quantization.py tests/test_progressive_residual_quantization.py
git commit -m "Add progressive residual wire format"
git push origin HEAD:master
```

Verify local HEAD, `origin/master`, and `git ls-remote origin refs/heads/master` are identical and the worktree is clean.

### Task 2: Deterministic fit, encode, and prefix decode

**Files:**
- Modify: `src/sfora/progressive_residual_quantization.py`
- Modify: `tests/test_progressive_residual_quantization.py`

**Interfaces:**
- Consumes: Task 1 wire types and existing `ProductQuantizer`/`fit_product_quantizer`.
- Produces: `ProgressiveResidualQuantizer` and `fit_progressive_residual_quantizer(values, spec, *, seed, maximum_iterations)`.
- Produces: `prepare_values`, `encode`, `decode_prefix`, `export_artifact`, and `from_artifact`.

- [ ] **Step 1: Add metric preparation tests**

For angular rows, assert one float64-norm normalization roundtrip and reject zero/nonfinite rows. For squared L2, assert bit-exact float32 identity. Reject input dimension, dtype, device, and contiguity drift.

- [ ] **Step 2: Add hand-derived encode/decode tests**

Construct a two-dimensional base quantizer with fixed codebooks. Assert that encode computes the residual from the decoded base code, rounds the scale through float16 before assignment, emits one eight-plane code, and that every prefix uses:

```python
step = 1 << (8 - residual_bits)
center = prefix.float() * step + (step - 1) / 2.0 - 127.5
expected = base_decode + center * scales.float()[:, None]
```

Require angular output normalization and no L2 normalization. Cover zero residuals, clipping, float16 scale underflow/overflow, exact half ties, and CPU/CUDA equality when CUDA is available.

- [ ] **Step 3: Add train-only fitting and artifact mutation tests**

Monkeypatch `fit_product_quantizer` to record its input and prove angular normalization versus L2 identity. Mutation-lock schema, metric, block geometry, codebook size, maximum bits, every codebook tensor, and unexpected/missing keys. Roundtrip a real small fitted codec and require byte-identical outputs.

- [ ] **Step 4: Run the Task 2 RED**

Run the exact new test nodes and preserve failures at missing codec/fitter methods.

- [ ] **Step 5: Implement the minimal codec**

Compose one existing `ProductQuantizer`; do not duplicate k-means. Quantize from the float16-rounded scale, pack once at eight bits, and make `decode_prefix` consume only the requested prefix planes. Keep artifact tensors CPU-portable and register codebooks through the existing quantizer implementation.

- [ ] **Step 6: Run focused GREEN, all module tests, and static checks**

Run the Task 2 nodes, the complete progressive test file, existing product/rate-matched PQ tests, Ruff, strict mypy, py_compile, and `git diff --check`.

- [ ] **Step 7: Commit and push the codec slice**

Commit only the module and its test, push to `master`, and verify exact remote equality and a clean tree.

### Task 3: Bounded candidate reranking and causal accounting

**Files:**
- Modify: `src/sfora/progressive_residual_quantization.py`
- Modify: `tests/test_progressive_residual_quantization.py`

**Interfaces:**
- Consumes: `ProgressiveResidualQuantizer`, `ProgressiveResidualCodes`, queries, and `int64[Q,L]` candidate ordinals.
- Produces: `ProgressiveCandidateResult(ordinals, scores, base_bytes_read, residual_bytes_read)`.
- Produces: `score_candidates(queries, codes, candidate_ordinals, *, residual_bits, return_width)`.

- [ ] **Step 1: Write exact angular/L2 ranking tests**

Use hand-derived candidates and require angular ordering by `(-similarity, ordinal)` and L2 ordering by `(distance, ordinal)`. Cover exact ties, repeated/out-of-range candidates, `return_width > L`, nonfinite queries, device mismatch, and candidate matrices large enough to prove scoring remains candidate-bounded.

- [ ] **Step 2: Write stage-byte accounting tests**

For `Q` queries and `L` candidates require:

```python
base_bytes_read == Q * L * spec.base_spec.bytes_per_vector
residual_bytes_read == Q * L * (2 + residual_bits * spec.residual_plane_bytes)
```

Keep these counters separate so a future fused index may prove base codes were already resident or cached rather than silently reporting zero.

- [ ] **Step 3: Run the Task 3 RED**

Run only the new candidate tests; expected failure is the missing result/scorer boundary.

- [ ] **Step 4: Implement gather, prefix decode, score, and stable selection**

Gather candidate rows without materializing `Q*N`. Sort ordinals first, then stable-sort metric scores so equal scores choose the lowest row ordinal. Return contiguous tensors and exact Python integers for byte counts.

- [ ] **Step 5: Run focused and regression GREEN**

Run the complete progressive test file, `tests/test_pq_candidate_reranking.py`, product/rate-matched PQ tests, Ruff, strict mypy, py_compile, and diff-check.

- [ ] **Step 6: Commit and push the candidate slice**

Commit only the focused files and verify canonical remote equality and a clean tree.

### Task 4: Strict ANN-Benchmarks evaluator

**Files:**
- Create: `scripts/evaluate_progressive_residual_ann.py`
- Create: `tests/test_evaluate_progressive_residual_ann.py`

**Interfaces:**
- Consumes: authenticated HDF5 path, SHA-256, dataset metric, fixed fit/query ranges, seeds, candidate widths, and prefix widths.
- Produces: per-query containment/recall/stage timing records and canonical `sfora-progressive-residual-ann-result-v1` JSON.
- Produces: `validate_progressive_ann_result` and `canonical_progressive_ann_result_bytes`.

- [ ] **Step 1: Write HDF5 authority tests**

Create tiny angular and L2 files with `train`, `test`, `neighbors`, and `distances`. Require exact SHA before parsing; exact rank/dtype/schema; finite rows; metric attribute; in-range neighbor IDs; and top-k set correctness under equal-distance ties. Mutation-lock missing/extra datasets, dtype/shape, nonfinite values, duplicate truth IDs, and digest drift.

- [ ] **Step 2: Write candidate-containment tests that prevent the development bug**

Use a candidate matrix wider than truth and separately pass `candidate_width` and `truth_width`. Assert that top-10 truth containment inside top-1,000 candidates examines all 1,000 candidates, not only the first ten. Mutation-lock candidate width, truth width, and duplicate IDs.

- [ ] **Step 3: Write receipt recomputation tests**

Require raw per-query hits and nanoseconds. Recompute aggregate Recall@10/100, containment, p50/p95/p99, physical bytes, shared bytes, seed summaries, and bootstrap confidence intervals. Reject bool-as-int, reordered/missing queries, percentile drift, metric/digest drift, partial seeds, `claim_eligible=true`, and noncanonical output.

- [ ] **Step 4: Run the evaluator RED**

Run `pytest -q tests/test_evaluate_progressive_residual_ann.py`; expected missing-script interfaces only.

- [ ] **Step 5: Implement bounded loading and evaluation**

Stream or memory-map gallery inputs where possible, fit only the registered sample, encode bounded batches, score query blocks, and synchronize CUDA around every measured stage. Use the library scorer for final ranking; never duplicate its quantization math in the evaluator.

- [ ] **Step 6: Implement strict CLI and atomic publication**

Require explicit metric, source URI/SHA, seed set, fit rows, development/evaluation query ranges, candidate widths, prefixes, device, batch size, and output path. Refuse overwrite, reserve a mode-0600 partial, validate the complete result, fsync, and rename atomically.

- [ ] **Step 7: Run focused GREEN and repository static checks**

Run both focused test files, Ruff, strict mypy, py_compile, and diff-check. Run dependency-complete `pytest -q` only after focused tests are stable.

- [ ] **Step 8: Commit and push the evaluator slice**

Commit the script/test only after the full local gate passes. Verify HEAD/origin/remote equality and a clean tree.

### Task 5: Frozen untouched-query codec screen

**Files:**
- Modify: `docs/positive_coverage_adaptation_result_2026-09-12.md`

**Interfaces:**
- Consumes: official GloVe and SIFT HDF5 files, remaining query ordinals 1,000..9,999, seeds 50..52, prefixes 3/5/8, and candidate widths 1,000/5,000.
- Produces: two canonical claim-ineligible receipts and one evidence-ledger update.

- [ ] **Step 1: Pre-register immutable commands and gates**

Record exact input SHA/size, code commit, fit/query ordinals, seeds, metric, prefixes, candidate widths, RSS/PSI/swap/wall stops, and the 0.95 Recall@10/100 gate before execution.

- [ ] **Step 2: Run GloVe once on the DGX**

Preserve the original session, monitor PID/CPU/RSS/PSI, and do not restart after a scientific terminal. Validate the receipt before interpreting quality.

- [ ] **Step 3: Run SIFT once on the DGX**

Use the identical code and corresponding native L2 metric. Preserve and validate the original terminal and receipt.

- [ ] **Step 4: Classify representation and prefix penalty**

Require both datasets to pass the 0.95 high-quality gate at one common residual bit count. Compare nested prefixes with separately encoded controls and reject the progression claim if recall loss exceeds 0.005 without a measured systems benefit.

- [ ] **Step 5: Record and deliver evidence**

Update only verified numbers, confidence intervals, limitations, receipt hashes, and classifications. Verify docs, commit, push, and confirm a clean canonical remote.

### Task 6: Matched-storage falsifiers before ANN routing

**Files:**
- Modify: `scripts/evaluate_progressive_residual_ann.py`
- Modify: `tests/test_evaluate_progressive_residual_ann.py`
- Modify: `docs/positive_coverage_adaptation_result_2026-09-12.md`

**Interfaces:**
- Consumes: frozen candidate matrices from the codec screen.
- Produces: direct SQ4/6/8, full-dimensional OPQ24, and fixed-rate residual controls with honest bytes and identical ranking metrics.

- [ ] **Step 1: Add test-first baseline interfaces and byte accounting**

Mutation-lock per-vector scale bytes, scalar packed payloads, OPQ rotation/codebooks, shared parameters, and candidate-set equality across methods.

- [ ] **Step 2: Implement only the registered cheap falsifiers**

Reuse existing OPQ. Implement scalar controls in the evaluator, not the public library. Do not add LVQ/RaBitQ/QINCo2 until an official dependency or faithful implementation plan is separately reviewed.

- [ ] **Step 3: Run the frozen controls once**

Compare matched total storage and matched candidate populations. Continue a new-codec claim only if progressive residuals improve quality materially or later demonstrate a systems advantage.

- [ ] **Step 4: Complete local assurance and delivery**

Run focused tests, full pytest, Ruff, strict mypy, py_compile, diff-check, then commit/push verified evidence.

## Deferred follow-on

Only after Tasks 1--6 identify a surviving codec should a new design/plan add IVF routing,
stage-wise survival accounting, boundary-aware escalation, fused kernels, and official system
baselines. Its first gate is at least 5x fewer scanned base codes with no more than 0.002 absolute
final-recall loss; its publication gate is at least a 20% matched-frontier improvement on three
materially different datasets.
