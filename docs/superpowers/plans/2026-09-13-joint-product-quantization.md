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
- Encode every database vector into exactly 24 bytes: one `uint8` assignment for each of 24
  256-entry codebooks. The PQ controls use sixteen 5-dimensional blocks followed by eight
  6-dimensional blocks; the Task 3 candidates use 24 full-dimensional additive codebooks.
- PQ-control scoring is negative one-half of summed squared block distances. Task 3 additive
  scoring is exactly `sum_m q dot A_m[c_m]`, with no database sidecar or decoded-vector
  renormalization. Never substitute symmetric scoring for either registered rule.
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
- [x] Run the authenticated seed-0 OPQ24 control. It improved matched PQ24 from `0.570721` to
  `0.572754` mAP@R but did not match PQ32 `0.578923`; retain it as an initialization/control, not
  as a sufficient solution. The canonical receipt SHA-256 is `b0922d2c...f9ba10b`.

### Task 2B: Candidate-set ordering repair before joint codec training

**Files:**
- Create: `src/sfora/pq_candidate_reranking.py`
- Create: `tests/test_pq_candidate_reranking.py`
- Create: `scripts/run_sop_pq_candidate_reranking.py`
- Create: `tests/test_run_sop_pq_candidate_reranking.py`

- [x] Add query-independent per-codeword residual second moments and exact query/code features:
  all 24 partial dot products, 24 directional variances, 24 residual energies, decoded norm,
  and the bounded candidate-set code similarity matrix. These are shared metadata only; database
  rows remain exactly 24 bytes.
- [x] Add a permutation-equivariant residual set scorer that begins at the exact hard-ADC
  baseline, listwise float-teacher distillation, deterministic fitting, and an exact top-32
  candidate constructor with leave-self-out and row-index tie authority.
- [x] Before training, measure PQ24 seed variance and preserve the existing PQ24/PQ32/OPQ24/float
  controls. A gain smaller than the observed seed variation is not evidence.
- [x] Train one frozen label-free recipe on fitting rows only: 32 ADC candidates, a three-layer
  128-wide/four-head set scorer, temperature `0.05`, KL plus `0.1` row-centered (shift-invariant)
  score MSE, AdamW, and a fixed schedule. Validation labels remain evaluation-only. Report the
  exact baseline and reranked candidate lists, per-query AP, paired deltas, objective components,
  residual gate, target entropy/effective support, codebook seed, model bytes, and top-32 ceiling.
- [x] Kill the ordering-repair path below PQ32 `0.578923`; call it promising only at or above the
  preregistered `0.58563` target, above measured PQ seed variation, and without more than `0.002`
  R@1 loss from its own PQ24 baseline. A pass remains SOP development evidence and must transfer
  to a fresh domain before any generic claim. A non-improving fit objective is an optimization
  failure, not a scientific rejection of candidate-set reranking.
- [x] If this bounded reranker fails, proceed to Task 3's joint hard-code ranking objective. Do not
  spend another experiment on reconstruction-only codebook or decoder improvements.

The initial substrate remains PQ24 so the experiment isolates ordering repair against the exact
matched hard-PQ control. OPQ24 remains a separately reported control and a possible initializer for
Task 3; mixing it into this screen would change both the representation and the ordering mechanism.

The completed seed-0 screen reached `0.578615516 / 0.822655525`, gaining `0.007894843` mAP@R
over PQ24, or 8.99 measured PQ-codebook seed standard deviations. It nevertheless missed PQ32 by
`0.000307669` and the target by `0.007014484`, so the registered path is closed. The exact-float
top-32 ceiling remains `0.591225085`, localizing the remaining problem to information and ordering
inside the fixed 192-bit representation rather than candidate containment.

### Task 2C: Fixed-code lookup-table distillation diagnostic

**Files:**
- Create: `src/sfora/pq_lookup_distillation.py`
- Create: `tests/test_pq_lookup_distillation.py`
- Create: `scripts/run_sop_pq_lookup_distillation.py`
- Create: `tests/test_run_sop_pq_lookup_distillation.py`

- [x] Add a generic query-dependent additive scorer
  `s(q,c)=s_PQ(q,c)+sum_m h(q)^T E_m[c_m]`, with a `128 -> 128 -> 32` GELU query network and
  24 zero-initialized, mean-centered `256 x 32` correction tables. The database remains exactly
  24 bytes. This diagnostic computes the existing 24-lookup PQ baseline and the 24 correction
  lookups separately; a fused 24-lookup production table is possible but is not implemented or
  claimed by this experiment.
- [x] Mutation-lock permutation independence, exact zero-correction equality to PQ24, table
  centering, direct-score/lookup equality, deterministic state, and rejection of nonfinite or
  malformed codes and queries.
- [x] Train two frozen label-free controls on fitting classes only, using teacher-top128, the
  first 128 additional compressed candidates outside that teacher set (mined from compressed
  top256), and a 128-row uniform tail: teacher-score MSE and listwise
  teacher KL plus 256 soft pairwise comparisons at temperature `0.03`. The uniform tail uses
  independent per-query SplitMix64 rejection sampling. Train for four epochs and refresh the
  compressed candidates at the start of epochs two through four; validation labels remain
  evaluation-only. Authenticate and bind the exact prior reranker receipt and compare the final
  fitted loss against the unchanged scorer on both the initial fixed pool and the same final
  refreshed pool. Pairwise strata are
  64 teacher-top32 pairs, 64 compressed-exclusive-top32 pairs, and 128 near-versus-tail pairs;
  listwise KL relates all candidate groups. Record an objective failure per arm rather than
  aborting the other arm or suppressing the result receipt.
- [x] Report both fixed-candidate and own exhaustive rankings, teacher pairwise agreement,
  mAP@R, Recall@1, model/table bytes, PQ seed variation, and exact input/output authorities.
  Kill this frozen fixed-code recipe unless its fixed-PQ24-top32 ranking beats the matched prior
  reranker `0.578615516` by more than the measured PQ-codebook seed-variation heuristic and its
  exhaustive ranking crosses PQ32. This is not learned-scorer run variation and cannot reject the
  whole fixed-code model family. Only reaching
  `0.58563` without more than `0.001` Recall@1 loss can defer Task 3. This seed-0 screen always
  remains `passed=false`; a target-screen result requires a separately frozen multi-seed run.
- [ ] Add a fitting-shard-only contested-margin spectrum diagnostic before any anisotropic codec:
  compare global PCA energy, PQ residual error, and centered float top-32 pairwise decision energy.
  Kill contested-set anisotropy when the registered alignment ratio is at least `0.6`; treat a
  value below `0.4` as evidence to retain it as a later additive-quantizer ablation.

The completed fixed-code screen regressed despite healthy matched-pool optimization. The ranking
arm reached only `0.566232738 / 0.819279142` exhaustive mAP@R / Recall@1, while the score-MSE arm
reached `0.557812453 / 0.809403224`; plain PQ24 remained `0.570720672 / 0.822233477`. Its
canonical receipt SHA-256 is `e3555a41...9af44ca0`. This closes the frozen lookup-correction
recipe and makes Task 3's learned hard-code assignments the next representation boundary.

### Task 3: Authenticated joint-codec development driver

**Files:**
- Create: `scripts/run_sop_joint_product_quantization.py`
- Create: `tests/test_run_sop_joint_product_quantization.py`
- Modify: `scripts/positive_coverage_artifacts.py`
- Modify: `tests/test_positive_coverage_artifacts.py`

- [ ] Add `src/sfora/additive_quantization.py` and focused tests for a generic full-dimensional
  additive codec. Its database representation is exactly 24 `uint8` assignments; its shared
  codebooks have shape `24 x 256 x 128`; query/database scoring is exactly
  `sum_m q dot A_m[c_m]`, requiring 24 table lookups and no database norm, scale, residual, or
  sidecar bytes. Mutation-lock hard decode, lookup/direct-dot equality, finite/device/type
  authority, incumbent-preserving coordinate-descent ties, and exact padded-PQ initialization.
- [ ] Freeze a three-arm exact-192-bit comparison: matched PQ24 control; re-encoded additive
  quantization with ordinary reconstruction; and re-encoded additive quantization with
  anisotropic reconstruction
  `||x-x_hat||^2 + 3 * (x dot (x-x_hat))^2`. Initialize the 24 full-dimensional codebooks by
  embedding each existing PQ24 codeword in its original block and initialize every encode from
  the original PQ24 bytes. Use eight outer rounds, two hard coordinate-descent assignment sweeps
  per round, two complete fixed-assignment Adam codebook passes per round, batch size 1,024,
  learning rate `1e-3` for rounds 1--4 and `3e-4` for rounds 5--8, no weight decay, and one retry
  at one-quarter learning rate when the complete fitting objective increases. Held-out vectors
  are freshly encoded from their own PQ24 initialization with eight coordinate sweeps.
- [ ] Do not use labels, candidate sampling, hard-negative refresh, KL, or another learned query
  scorer in this screen. The prior fixed-code losses optimized their sampled objectives while
  worsening exhaustive retrieval, so changing both encoding and mining would make the next
  result uninterpretable. Validation labels remain evaluation-only.
- [ ] Authenticate the same source/teacher snapshots and parent direct-head checkpoint used by the
  projection-capacity experiment. Initialize the student head identically to that checkpoint.
- [ ] Fit codebooks only on fitting representations and keep the 128D head frozen for this first
  representation test. Report initial-padded/fixed-code, re-encoded isotropic, and re-encoded
  anisotropic controls plus a fixed-codebook anisotropic re-encoding control; exhaustive
  mAP@R/R1; reconstruction and parallel/tangential error;
  float-head top-1/8/32/128 overlap; float-head-top32 pairwise inversions; coded top32 intruders
  from outside float-head top128; float reranking within each arm's top32; utilization; assignment churn;
  and the full-gallery score-error upper tail. This separates containment, local ordering,
  representation change, and objective effects.
- [ ] Publish no-clobber checkpoints and one canonical complete receipt; refuse implicit execution.

### Task 4: Seed-0 kill screen and evidence

- [ ] Run focused tests, Ruff, pycompile, mypy, and `git diff --check`; then one full repository
  test suite.
- [ ] Commit/push the exact Sfora slice to canonical `master`, deploy that exact commit to the DGX,
  and rerun focused tests against checkout `src`.
- [ ] Run exactly one seed-0 SOP development screen. Kill the recipe unless additive-dot mAP@R
  reaches at least `0.5809`, with no more than `0.002` R1 regression from the matched padded-PQ24
  additive-dot control. The desired research threshold is `0.58563`; PQ32 hard-ADC remains a named
  cross-functional quality bar rather than the matched scoring-rule control. This is a
  preregistered best-of-two claim-ineligible screen, not a single-arm publication claim.
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
