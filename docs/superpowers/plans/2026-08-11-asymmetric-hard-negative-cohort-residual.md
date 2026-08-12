# AHNCR Held-Out Falsifier Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans and superpowers:test-driven-development task by task.

**Goal:** Implement and run one deterministic CPU-only, train-archive-only falsifier for Asymmetric Hard-Negative Cohort Residual similarity.

**Architecture:** A small library module owns exact label splitting, cohort centroids, retrieval comparisons, controls, and the frozen decision. A separate CLI strictly loads only the registered training archive, constructs a class-disjoint held-out retrieval task, validates a canonical report, and publishes it without replacement. The official In-Shop query/gallery archives and the prior ALSP result are not CLI inputs and must never be read.

**Tech stack:** Python 3.12, NumPy 2.5, standard-library JSON/hashlib/pathlib, pytest, Ruff.

## Frozen constraints

- Input is exactly `inshop_corrected_pa_seed0_train_final.npz`, SHA-256 `67aa387c9815fd300e7db0da9f1a781e4b95191bc9db715e7d06850c9a7e6fea`.
- CPU only with `CUDA_VISIBLE_DEVICES=''` and one OMP/MKL/OpenBLAS thread.
- Fixed `k=50`, block size `256`, float32 products, float64 means/statistics, stable first-index ties.
- Labels are indivisible. The 80/20 cohort/evaluation split, one-query-per-label choice, and four reporting shards follow design commit `67814a5` exactly.
- Controls are raw, global mean, positive expansion, unary cohort density, symmetric adaptive normalization, and 20 centroid-assignment permutations from PCG64 seed `20260814`.
- No coefficient, normalization, `k`, seed, threshold, or split may change after the result is observed.
- One scientific execution only. A failed predicate is a scientific KILL, not permission to tune.

---

### Task 1: Exact held-out construction and cohort residuals

**Files:**
- Create: `src/sfora/asymmetric_cohort_residual.py`
- Create: `tests/test_asymmetric_cohort_residual.py`

**Interfaces:**
- `split_heldout_labels(labels) -> HeldoutSplit`
- `select_query_and_gallery(labels, example_ids, evaluation_labels) -> RetrievalSplit`
- `reporting_shards(evaluation_labels) -> np.ndarray`
- `hard_negative_centroids(queries, cohort, *, k=50, block_size=256) -> np.ndarray`
- `unary_cohort_density(gallery, cohort, *, k=50, block_size=256) -> np.ndarray`

- [ ] Write RED tests with independent literal SHA-256 ordering oracles. Cover singleton cohort inclusion, indivisible labels, exact 80/20 floor, Unicode IDs, stable query selection, and all four nonempty domain-separated shards.
- [ ] Run `pytest -q tests/test_asymmetric_cohort_residual.py -k 'split or query or shard'`; require missing-module RED.
- [ ] Implement exact concrete dtypes and validation. Reject bool-as-int, duplicate/empty IDs, non-int64 labels, overlap, and empty partitions.
- [ ] Write RED tests that independently enumerate top-k centroids/densities, including stable tie behavior, block invariance, `k` bounds, nonfinite arrays, and non-unit rows.
- [ ] Implement blockwise float32 products and stable top-k by `(-similarity, cohort_index)`, reducing selected rows in float64 then returning concrete float32 centroids and float64 densities.
- [ ] Run the complete new test file and Ruff; commit `add AHNCR heldout construction`.

---

### Task 2: Shared-product retrieval controls and decision

**Files:**
- Modify: `src/sfora/asymmetric_cohort_residual.py`
- Modify: `tests/test_asymmetric_cohort_residual.py`

**Interfaces:**
- `evaluate_ahncr(queries, query_labels, gallery, gallery_labels, cohort, shard_ids, *, k=50, block_size=256, permutation_seed=20260814) -> Evaluation`
- `exact_mcnemar(wrong_to_right, right_to_wrong) -> float`
- `decide_ahncr(evaluation) -> tuple[bool, dict[str, bool]]`

- [ ] Write RED fixtures whose exact top-1 outcomes are independently enumerable for every named arm. Assert AHNCR is exactly `(2q-m(q)) @ g.T`, the gallery is unchanged, and the residual is not normalized.
- [ ] Require every arm to reuse the same raw `q @ g.T` product. Independently verify stable first-index argmax, pooled/shard recall, transitions, and exact binomial McNemar p-values.
- [ ] Add 20 fixed centroid-row permutations. Assert the same observed centroid matrix is reused, each permutation is a bijection, assignments are deterministic, and the linear empirical 95th percentile matches an independent oracle.
- [ ] Parameterize every decision boundary: gain `>= .003`, raw paired `p < .01`, at least 3/4 positive shards, four control gaps `>= .001`, every direct AHNCR-versus-control comparison favors AHNCR with Bonferroni `p < .0125`, shuffled `p95` strictly lower, and W→R > R→W. Mutating any predicate must flip PASS to KILL.
- [ ] Implement the minimum code to make tests green, preserving exact predicate order.
- [ ] Run pytest/Ruff/py_compile; commit `add AHNCR retrieval decision`.

---

### Task 3: Strict train-only evaluator and atomic report

**Files:**
- Create: `scripts/evaluate_inshop_ahncr.py`
- Create: `tests/test_evaluate_inshop_ahncr.py`

**Interfaces:**
- `build_ahncr_report(train_path: Path, *, block_size: int = 256) -> dict[str, Any]`
- `validate_ahncr_report(value: object) -> dict[str, Any]`
- CLI: `--train`, `--output`, `--block-size`; no query/gallery/result arguments.

- [ ] Write archive RED tests for exact SHA, split name `train`, float32/unit/finite embeddings, int64 labels, unique nonempty Unicode IDs, and row alignment. Prove the parser rejects every extra CLI argument, especially `--query` and `--gallery`.
- [ ] Freeze report order: `schema_version`, `input`, `environment`, `configuration`, `split`, `arms`, `null`, `decision`. Record input SHA, NumPy/thread runtime, every split/count/hash, all arm pooled/shard statistics, 20 null gains, percentile, predicates, and `passes_falsifier`.
- [ ] Add exhaustive recursive mutations: missing/extra/reordered keys; wrong concrete types; NaN/infinity; inconsistent counts/hashes/recall/gains/transitions/p-values/shards/direct-control comparisons/null percentile/predicates/final decision.
- [ ] Add a deterministic synthetic end-to-end archive in which the candidate succeeds and each causal control has a known weaker outcome. Independently recompute all top-1 vectors outside production code.
- [ ] Add candidate-isolation tests: install read/open sentinels for the official query/gallery filenames and ALSP result, then execute `build_ahncr_report`; no forbidden path may be reached.
- [ ] Add atomic-publication tests for success, existing destination/symlink, existing temp, link race, serialization/write/fsync/reload/validation failures, unchanged sentinel bytes, and owned-temp-only cleanup.
- [ ] Implement canonical UTF-8 JSON plus LF, same-directory exclusive temp, file fsync, hard-link no-replace, directory fsync, strict reload/revalidation. Valid PASS and KILL both exit 0; structural failures exit 2.
- [ ] Run both test files, Ruff, py_compile, and diff-check; commit `add frozen AHNCR evaluator`.

---

### Task 4: Independent review and single frozen execution

**Files:**
- Create after execution: `reports/generated/inshop_ahncr_train_holdout.json`
- Create after execution: `docs/inshop_ahncr_result_2026-08-11.md`

- [ ] Start one read-only review consultation with explicit `models=["opus", "gpt-5.6-sol"]`. Ask it to audit leakage, hash splits, top-k/ties, shared products, controls, exact statistics, strict schema, atomic publication, and novelty scope. Reproduce findings before changing code; do not duplicate the review.
- [ ] Run the affected tests, Ruff, py_compile, and `git diff --check`. Run the repository suite once only if it does not repeat the known memory/cgroup-thrashing path.
- [ ] Confirm output/temp absence and the frozen input SHA. Execute exactly once:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src \
/home/rb/worktrees/sfora-emafactorial/.venv/bin/python -B scripts/evaluate_inshop_ahncr.py \
  --train /home/rb/reranking-inputs-2026-08-11/inshop_corrected_pa_seed0_train_final.npz \
  --output reports/generated/inshop_ahncr_train_holdout.json \
  --block-size 256
```

- [ ] Independently strict-load and recompute the split hashes, cohort centroids, every arm’s top-1 vector, recalls/transitions/p-values, four shard gains, 20 permutations/percentile, all predicates, and final decision. Verify regular nonsymlink mode-0600 output and no temp.
- [ ] Record PASS or KILL first in the result note. A KILL closes AHNCR without tuning. A PASS authorizes only a separately preregistered frozen-embedding replication on an independent dataset; GPU training remains unauthorized.
- [ ] Commit the exact artifact and note as `record frozen AHNCR falsifier`.
