# Rate-Matched Transfer Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the fixed PCA80+OPQ24 transfer result into reproducible paired evidence on standardized embedding archives without coupling the Sfora library to one dataset.

**Architecture:** Add one experiment-only evaluator that authenticates a generic train/test embedding archive, fits fixed-rate OPQ controls and the public rate-matched codec using training rows only, and emits canonical paired retrieval evidence. Keep fitting/scoring primitives in the library and dataset/provenance policy in the script. Run the frozen configuration on SOP first and one independently named second-domain archive without selecting parameters from either result.

**Tech Stack:** Python 3.12, PyTorch/CUDA, NumPy NPZ, Sfora product quantization APIs, pytest, Ruff, mypy.

**Spec:** `docs/positive_coverage_adaptation_result_2026-09-12.md` (Reduced-dimension fixed-rate diagnostic and Deployable library composition sections)

## Global Constraints

- Fit centering, projection, OPQ rotation, and codebooks from training rows only.
- Freeze `dimensions=80`, `bytes_per_vector=24`, `codebook_size=256`, `seed=50`, `maximum_iterations=20`, and `rotation_iterations=4` before evaluation.
- Compare against ordinary OPQ24 and OPQ32 fit from the identical normalized training matrix.
- Retain per-query AP@R and Recall@1 for paired uncertainty; aggregate-only output is insufficient.
- Use bounded evaluation encode and score batches; the current public OPQ fitting path still performs full-population assignment and therefore requires an external RSS cap. Full-corpus evaluation time is diagnostic throughput, not a serving-latency claim.
- Emit `claim_eligible=false` until the protocol and input identities are prospectively frozen and independently reviewed.
- Never import SOP, CUB, In-Shop, class names, or dataset paths into `src/sfora`.
- Treat only the four named embedding/label arrays as semantic inputs; authenticate the complete NPZ byte stream and record every contained key so provenance arrays remain bound without coupling the evaluator to a dataset-specific key set.
- Freeze paired bootstrap seed 50 and 10,000 complete-class resamples.
- Freeze the formal inputs before either canonical run: SOP UniCOM-B16 SHA-256 `6bc0d8383251685eaccd472eeda357861caffb3bfb4129f18e0124c3ddc72818`, then CUB-200-2011 UniCOM-B16 SHA-256 `c5edb5b3e1caee1d4e09ff0ccbb005ae4efd7e0840b2ae91ffd317719a699e68`.
- Define replication before observing the canonical cells: on both inputs the lower endpoint of the paired 95% mAP@R interval versus OPQ24 must be greater than 0, and the lower endpoint versus OPQ32 must be at least -0.005. Recall@1 is secondary corroboration, not a hidden selection gate.

---

### Task 1: Generic paired ADC evaluator

**Files:**
- Create: `scripts/evaluate_rate_matched_pq_transfer.py`
- Create: `tests/test_evaluate_rate_matched_pq_transfer.py`

**Interfaces:**
- Consumes: normalized or normalizable CPU float32 train/test matrices, concrete test labels, and `OptimizedProductQuantizer` or `RateMatchedProductQuantizer`.
- Produces: `normalize_embedding_rows`, `encode_in_batches`, `score_adc_retrieval`, and `paired_class_bootstrap_interval`.

- [x] **Step 1: Write failing numerical tests**

Add literal fixtures proving that `score_adc_retrieval` excludes the same row, uses stable lowest-ordinal ties, computes AP@R over exactly `class_count - 1` positions, and returns literal per-query AP/R1 values. Add a codec fake whose maximum observed encode and score batch sizes prove bounded execution without asserting mock call existence.

- [x] **Step 2: Run the focused RED**

Run: `uv run --locked pytest -q tests/test_evaluate_rate_matched_pq_transfer.py -k 'normalize or encode or score or bootstrap'`

Expected: collection fails only because the new evaluator module or named interfaces do not exist.

- [x] **Step 3: Implement the bounded evaluator primitives**

Normalize rows in float32 after finite/nonzero validation. Encode consecutive slices no larger than the explicit batch size. Score consecutive query slices against the resident code matrix, set the matching query/gallery diagonal to infinity, rank by `(distance, gallery_ordinal)`, and calculate AP@R/R1 from literal labels. Return aggregates plus immutable per-query tuples. Bootstrap complete classes with a fixed seed and percentile interpolation already used by Sfora's paired evaluators.

- [x] **Step 4: Run focused GREEN**

Run the Step-2 command and require every selected test to pass with no unexpected warning.

### Task 2: Strict archive, experiment, and receipt boundary

**Files:**
- Modify: `scripts/evaluate_rate_matched_pq_transfer.py`
- Modify: `tests/test_evaluate_rate_matched_pq_transfer.py`

**Interfaces:**
- Consumes: `TransferEvaluationConfig`, one local NPZ path, its SHA-256, one absent output path, and an explicit execution flag.
- Produces: `load_transfer_archive`, `evaluate_transfer`, `validate_transfer_result`, `canonical_transfer_result_bytes`, and `main`.

- [x] **Step 1: Write failing authority tests**

Create a real temporary NPZ containing the four required semantic arrays plus one authenticated provenance array. Mutation-lock required-key presence, dtypes, rank, shared width, finite/nonzero rows, label concreteness/cardinality, file digest, fixed configuration, output nonexistence, duplicate/unknown CLI flags, and refusal without `--execute-transfer-evaluation`. Validate a hand-authored result whose paired rows recompute all aggregates and contrasts; mutate each aggregate, storage width, timing classification, and `claim_eligible` field.

- [x] **Step 2: Run authority RED**

Run: `uv run --locked pytest -q tests/test_evaluate_rate_matched_pq_transfer.py -k 'archive or arguments or result or canonical'`

Expected: failures occur only at missing strict loader/parser/result interfaces.

- [x] **Step 3: Implement the minimal strict experiment**

Fit arms in the immutable order `pca80-opq24`, `opq24`, `opq32`; record fit seconds and full-evaluation seconds separately; retain each arm's per-query arrays; derive paired per-query and class-cluster intervals; record exact code bytes per vector, shared parameter bytes, source commit, script SHA-256, input SHA-256, CUDA/runtime identity, and the explicit warning that evaluation duration is not serving p99. Write sorted compact JSON plus one newline through an exclusive `.partial` reservation and hard-link publication.

- [x] **Step 4: Run complete evaluator GREEN and static checks**

Run serially:

```bash
uv run --locked pytest -q tests/test_evaluate_rate_matched_pq_transfer.py
uv run --locked ruff format --check scripts/evaluate_rate_matched_pq_transfer.py tests/test_evaluate_rate_matched_pq_transfer.py
uv run --locked ruff check scripts/evaluate_rate_matched_pq_transfer.py tests/test_evaluate_rate_matched_pq_transfer.py
uv run --locked mypy scripts/evaluate_rate_matched_pq_transfer.py tests/test_evaluate_rate_matched_pq_transfer.py
python3 -m py_compile scripts/evaluate_rate_matched_pq_transfer.py tests/test_evaluate_rate_matched_pq_transfer.py
git diff --check
```

### Task 3: Frozen DGX replication and interpretation

**Files:**
- Modify: `docs/positive_coverage_adaptation_result_2026-09-12.md`
- Produce on DGX: one canonical SOP receipt and one canonical second-domain receipt outside the repository.

**Interfaces:**
- Consumes: committed evaluator/source hashes and authenticated embedding archives.
- Produces: paired quality contrasts, storage comparison, diagnostic throughput, and an explicit replication classification.

- [ ] **Step 1: Commit and push the evaluator before opening evaluation rows**

Run the complete focused/static gate, stage only the evaluator/test/plan, commit with configured operator identity and no attribution trailers, push `HEAD:master`, and verify local HEAD, `origin/master`, and `git ls-remote origin refs/heads/master` are identical.

- [ ] **Step 2: Run the fixed SOP cell once**

Use the authenticated UniCOM-B16 SOP archive, exact fixed configuration above, CUDA, bounded batches, and an absent output. Preserve the original terminal and canonical result SHA-256. Do not rerun based on outcome.

- [ ] **Step 3: Run one fixed second-domain cell once**

Use the same configuration and scoring semantics on the frozen CUB-200-2011 UniCOM-B16 archive above. Do not change rank, bytes, seeds, iteration counts, or gates after the SOP result.

- [ ] **Step 4: Classify without overclaim**

Call the rate-matching mechanism replicated only if both frozen cells satisfy the exact paired interval rules above at 25% lower code storage than OPQ32. Otherwise report which representation/domain or comparison fails. Regardless of quality, keep the result diagnostic until candidate-index p99 and total 100M-row memory are measured.

- [ ] **Step 5: Verify and commit the evidence note**

Run the documentation validator, `git diff --check`, assert only the intended evidence document changed, commit, push to `master`, and verify remote equality and a clean worktree.
