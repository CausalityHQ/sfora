# Joint Product Quantization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:test-driven-development for
> every implementation task and superpowers:verification-before-completion before delivery.

**Goal:** Recover most of the observed 128-dimensional float retrieval quality in an exact
24-byte database code while retaining additive asymmetric scoring and a generic, label-free
training interface.

**Architecture:** Add a reusable variable-block product quantizer to Sfora. Hard database codes
use one byte per block; float queries are scored by the exact sum of per-block squared-distance
tables. A straight-through hard reconstruction exposes the identical hard forward path during
training. Before that expensive intervention, compare it to a full-dimensional 24-stage
residual/additive quantizer, which changes the codec family while retaining 24 additive lookup
operations per candidate. Only then does a separate authenticated SOP development driver adapt
the selected codec or representation from fitting rows using frozen-teacher neighborhood
distillation. Dataset labels are used only for the established evaluation metric, never by codec
optimization or candidate construction.

**Tech Stack:** Python 3.12, PyTorch, NumPy, scikit-learn, pytest, Ruff, mypy.

## Global constraints

- Work only in the Sfora repository.
- Keep the frozen 768-dimensional teacher input, existing 128-dimensional float-head
  initialization, class-disjoint fit/validation partition, and evaluator unchanged.
- Encode every database vector into exactly 24 bytes: 24 independent 256-entry codebooks whose
  block dimensions sum to 128. The initial registered layout is sixteen 5-dimensional blocks
  followed by eight 6-dimensional blocks.
- Float-query/database-code scoring is exactly negative one-half of the sum of squared block
  distances. Never renormalize a decoded database vector or substitute symmetric scoring.
- Candidate construction and optimization are label-free. Validation labels are evaluation-only.
- SOP validation is burned development evidence. Results remain `claim_eligible=false`; no result
  alone authorizes a generic or production claim.
- Freeze one recipe before the seed-0 screen. Do not branch into a codec sweep on the same split.
- The observed 24-byte PQ used all 128 dimensions as sixteen 5D and eight 6D blocks; no dimensions
  were truncated or padded. Preserve this fact in the baseline receipt.

---

### Task 1: Generic hard and trainable product quantizer

**Files:**
- Create: `src/sfora/product_quantization.py`
- Create: `tests/test_product_quantization.py`

**Interfaces:**
- Produces: `ProductQuantizationSpec` with exact block/code-width authority.
- Produces: `ProductQuantizer` with hard encode/decode, exact asymmetric distances, canonical
  codebook state, and hard-forward straight-through reconstruction.
- Produces: a neighborhood-distillation loss whose hard ADC, float-student, and reconstruction
  terms are separately observable.

- [ ] Write failing tests for input authority, uneven 128/24 partitioning, exact `uint8` code
  shape, nearest-codeword ties, decode, table-based ADC equality to explicit hard decode,
  serialization/state roundtrip, and rejection of nonfinite inputs.
- [ ] Write failing autograd tests proving the forward value is the hard codebook reconstruction
  while gradients reach both representation rows and selected codewords.
- [ ] Write failing loss tests that independently recompute teacher KL, hard-ADC KL, float KL,
  and reconstruction terms and reject invalid temperatures/shapes.
- [ ] Preserve the missing-module RED, implement the minimum module, and rerun focused tests.

### Task 2: Deterministic initializers and additive-codec preflight

**Files:**
- Modify: `src/sfora/product_quantization.py`
- Modify: `tests/test_product_quantization.py`
- Create: `src/sfora/residual_quantization.py`
- Create: `tests/test_residual_quantization.py`

- [x] Add tests for fitting-only deterministic codebook initialization, empty-cluster repair,
  seed reproducibility, and one-byte-per-block state authority.
- [x] Add tests for a 24-stage full-dimensional residual quantizer: exact greedy residual updates,
  decoded-vector sums, one-byte-per-stage codes, float-query additive-dot lookup equality, stable
  ties, and deterministic fitting. Do not claim cosine or squared-L2 equivalence.
- [x] On seed 0, compare the fixed 24-stage residual quantizer to post-hoc PQ24 and PQ32 before
  representation adaptation. The exact greedy arm failed at `0.526219` raw-dot mAP@R and
  `0.561729` under decoded-cosine diagnosis versus matched PQ24 `0.570721` and PQ32 `0.578923`.
  Retire this greedy construction without claiming that all additive quantizers fail.
- [ ] Add a blockwise, label-free teacher candidate builder. For each registered query row it
  must exclude self and return unique near, middle-rank, and uniform-tail identities with stable
  distance/row tie-breaking.
- [ ] Implement and verify without importing experiment-specific SOP code into the library.

### Task 2A: Deterministic OPQ control and local-ordering ceiling

**Files:**
- Modify: `src/sfora/product_quantization.py`
- Modify: `tests/test_product_quantization.py`
- Modify: `scripts/run_sop_additive_codec_preflight.py`
- Modify: `tests/test_run_sop_additive_codec_preflight.py`

- [x] Prove that PQ24 candidate containment is not the binding failure: exact float reranking of
  its top 32 reproduces the `0.591265 / 0.832700` float result, while a fitting-only linear
  conditional decoder reaches only `0.573793 / 0.823500`.
- [x] Add a generic fixed-rotation product quantizer with exact hard encode/decode and asymmetric
  distance equivalence, strict whole-matrix orthogonality and device authority, and canonical
  component extraction.
- [x] Add a deterministic OPQ alternation with float64 Procrustes updates, fitting-data-only model
  selection, and ordinary-PQ non-regression.
- [ ] Run the authenticated seed-0 OPQ24 control. Require it to match PQ32 before making it the
  initialization for hard-score joint training; record distortion and the exact rotation/codebook
  checkpoint even on failure.

### Task 3: Authenticated joint-codec development driver

**Files:**
- Create: `scripts/run_sop_joint_product_quantization.py`
- Create: `tests/test_run_sop_joint_product_quantization.py`
- Modify: `scripts/positive_coverage_artifacts.py`
- Modify: `tests/test_positive_coverage_artifacts.py`

- [ ] Freeze the independently reviewed recipe in constants and mutation-lock them: 128D head,
  exact 24-byte codec selected by Task 2, temperature, loss coefficients, candidate strata,
  optimizer schedule, refresh cadence, and seed-0 kill gate.
- [ ] Authenticate the same source/teacher snapshots and parent direct-head checkpoint used by the
  projection-capacity experiment. Initialize the student head identically to that checkpoint.
- [ ] Fit codebooks only on fitting representations. Train the head and codebooks against the
  actual hard asymmetric score; report a frozen-head/codebook-only causal control and the joint
  arm, plus joint float quality, distortion, utilization, assignment churn, and ranking flips.
- [ ] Publish no-clobber checkpoints and one canonical complete receipt; refuse implicit execution.

### Task 4: Seed-0 kill screen and evidence

- [ ] Run focused tests, Ruff, pycompile, mypy, and `git diff --check`; then one full repository
  test suite.
- [ ] Commit/push the exact Sfora slice to canonical `master`, deploy that exact commit to the DGX,
  and rerun focused tests against checkout `src`.
- [ ] Run exactly one seed-0 SOP development screen. Kill the recipe unless hard-ADC mAP@R reaches
  at least `0.5809` (recovering at least 0.010 of the 0.0207 post-hoc loss), with no more than
  0.002 float-student regression. The desired research threshold is `0.5859`.
- [ ] On failure, preserve diagnostics and seek fresh independent failure analysis before changing
  the recipe. On success, run the frozen remaining seeds and require paired class-cluster evidence.

### Task 5: Generality and serving qualification

- [ ] Freeze a fresh dataset and protocol before observing results. Require noninferiority to the
  stronger 24-byte post-hoc control and report all seeds; SOP-only evidence cannot pass this task.
- [ ] Separate representation, codec, and ANN losses using float exhaustive, coded exhaustive, and
  actual indexed retrieval.
- [ ] Measure the complete 100M serving representation: 2.4 GB code plane, ID strategy, routing,
  models, allocator/runtime, and buffers. Require total resident memory <=3 GiB and measured p99
  <=15 ms on declared hardware and concurrency.
- [ ] Only after fresh-domain quality and 100M systems gates pass, update release claims and docs.
