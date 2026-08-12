# Lorentz L0/L1 Falsifiers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CPU-only, no-training evaluator that can close the Lorentz compression lane or nominate a narrowly described norm-weighted rescoring function without making a premature hyperbolic claim.

**Architecture:** Add one pure NumPy arithmetic module, one strict frozen-bundle/report layer, and one no-clobber CLI. L0 measures train-only four-point structure against paired nulls. L1 fits PCA on train only, evaluates fixed radial scales and matched functional controls, and prices scale selection inside a paired query-identity bootstrap.

**Tech Stack:** Python 3.12, NumPy 2.5, pytest, Ruff, the existing `EmbeddingBundle` loader and stable retrieval reducer.

## Global Constraints

- Source authority is `docs/superpowers/specs/2026-08-12-lorentz-compression-rider-design.md` at commit `7c15452`.
- Dimensions are exactly `(8, 16, 32, 64)`; curvature is `-1`; maximum radius is `2.5`.
- Target median train radii are exactly `(0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)`.
- L0 uses ten train-only subsamples of exactly 2,000 rows, seeds `7000 + 100 * dataset_index + replicate`, and `delta_rel = 2 * delta / diameter`.
- L0 null seeds are `7500 + 100 * dataset_index + replicate`.
- L1 uncertainty is a paired query-identity bootstrap with 10,000 replicates and PCG64 seed `205`.
- PCA mean, covariance, and components use train rows only. Canonicalize every component sign by making its largest-absolute loading positive, breaking ties by lowest coordinate index.
- Retrieval uses stable row-index tie breaking. Identical query/gallery archives exclude only the same row index, never every zero-distance duplicate.
- The correct MIPS query transform is `(-q0, qs)`; `(q0, -qs)` is a mandatory failing mutant.
- No learned head, GPU, Riemannian optimizer, backbone training, ANN index, custom kernel, or result claim belongs in this plan.

---

### Task 1: Exact L0 Arithmetic and Nulls

**Files:**
- Create: `src/sfora/lorentz_rider.py`
- Create: `tests/test_lorentz_rider.py`

**Interfaces:**
- Produces: `l0_subsample_indices(train_count: int, dataset_index: int, replicate: int) -> np.ndarray`.
- Produces: `pairwise_chord_distances(values: np.ndarray) -> np.ndarray`.
- Produces: `medoid_index(distances: np.ndarray) -> int`.
- Produces: `gromov_delta_rel(distances: np.ndarray, base: int) -> DeltaEstimate`.
- Produces: `delta_bruteforce(distances: np.ndarray) -> float` for test arrays with at most 32 rows.
- Produces: `column_permutation_null(values: np.ndarray, seed: int) -> np.ndarray` and `spectrum_gaussian_null(values: np.ndarray, seed: int) -> np.ndarray`.

- [ ] **Step 1: Write the L0 formula and sampling RED tests**

```python
def test_four_point_oracles_and_scale_invariance():
    line = np.abs(np.arange(6)[:, None] - np.arange(6)[None, :]).astype(np.float64)
    cycle4 = np.asarray([[0,1,2,1],[1,0,1,2],[2,1,0,1],[1,2,1,0]], dtype=np.float64)
    square = pairwise_chord_distances(np.asarray([[0,0],[1,0],[1,1],[0,1]], np.float32))
    assert delta_bruteforce(line) == 0.0
    assert 2.0 * delta_bruteforce(cycle4) / cycle4.max() == 1.0
    assert 2.0 * delta_bruteforce(square) / square.max() == pytest.approx(2.0 - np.sqrt(2.0))
    base = medoid_index(square)
    assert gromov_delta_rel(17.0 * square, base).relative == pytest.approx(
        gromov_delta_rel(square, base).relative
    )

def test_l0_subsamples_are_exact_sorted_and_reproducible():
    first = l0_subsample_indices(25_882, 0, 0)
    assert first.shape == (2_000,)
    assert np.array_equal(first, np.sort(first))
    assert np.array_equal(first, l0_subsample_indices(25_882, 0, 0))
    assert not np.array_equal(first, l0_subsample_indices(25_882, 0, 1))
```

- [ ] **Step 2: Run the RED selector**

Run: `.venv/bin/pytest -q tests/test_lorentz_rider.py -k 'four_point or l0_subsample'`

Expected: collection fails because `sfora.lorentz_rider` does not exist.

- [ ] **Step 3: Implement exact small-array oracle and blocked base estimator**

Implement the three sorted sums formula in `delta_bruteforce`. In `gromov_delta_rel`, construct `A[i,j] = (D[i,b] + D[j,b] - D[i,j]) / 2` in float64, update a float64 `maximum` matrix for each `k` with `maximum(maximum, minimum(A[:,k,None], A[k,None,:]))`, and return `delta=max(maximum-A)`, `diameter=max(D)`, `relative=2*delta/diameter`. Reject non-square, nonfinite, asymmetric, negative, nonzero-diagonal, or zero-diameter matrices.

- [ ] **Step 4: Add null-preservation RED tests and implement nulls**

```python
def test_nulls_preserve_registered_structure():
    values = np.arange(60, dtype=np.float32).reshape(12, 5)
    permuted = column_permutation_null(values, 7500)
    for column in range(values.shape[1]):
        assert np.array_equal(np.sort(permuted[:, column]), np.sort(values[:, column]))
    gaussian = spectrum_gaussian_null(values, 7500)
    assert gaussian.shape == values.shape
    assert gaussian.dtype == np.float32
    assert np.isfinite(gaussian).all()
```

Fit the Gaussian null to the observed mean and covariance eigenvalues with PCG64 standard-normal scores and the same canonical eigenvector signs. Do not use query/gallery rows.

- [ ] **Step 5: Run Task 1 GREEN and commit**

Run: `.venv/bin/pytest -q tests/test_lorentz_rider.py -k 'four_point or l0_subsample or null' && .venv/bin/ruff check src/sfora/lorentz_rider.py tests/test_lorentz_rider.py`

```bash
git add src/sfora/lorentz_rider.py tests/test_lorentz_rider.py
git commit -m "add Lorentz L0 geometry falsifier"
```

---

### Task 2: Frozen PCA, Lorentz Lift, and Correct MIPS

**Files:**
- Modify: `src/sfora/lorentz_rider.py`
- Modify: `tests/test_lorentz_rider.py`

**Interfaces:**
- Produces: `FrozenPCA(mean: np.ndarray, components: np.ndarray)`.
- Produces: `fit_frozen_pca(train: np.ndarray, dimension: int) -> FrozenPCA` and `apply_frozen_pca(values: np.ndarray, fit: FrozenPCA) -> np.ndarray`.
- Produces: `scale_for_target_median(train_projected: np.ndarray, target: float) -> float`.
- Produces: `lorentz_lift(values: np.ndarray, scale: float, clip: float = 2.5) -> np.ndarray`.
- Produces: `lorentz_mips_scores(query: np.ndarray, gallery: np.ndarray) -> np.ndarray` and `lorentz_distance_block(query: np.ndarray, gallery: np.ndarray) -> np.ndarray`.

- [ ] **Step 1: Write PCA leakage/sign and lift RED tests**

```python
def test_pca_uses_train_only_and_has_canonical_signs():
    train = np.asarray([[2,0],[1,0],[-1,0],[-2,0]], np.float32)
    fit = fit_frozen_pca(train, 1)
    assert fit.mean.tolist() == [0.0, 0.0]
    assert fit.components.shape == (1, 2)
    pivot = int(np.argmax(np.abs(fit.components[0])))
    assert fit.components[0, pivot] > 0.0
    assert apply_frozen_pca(np.asarray([[10,7]], np.float32), fit).shape == (1, 1)

def test_lift_satisfies_hyperboloid_and_mips_selects_nearest():
    values = np.asarray([[1,0],[-1,0]], np.float32)
    lifted = lorentz_lift(values, 1.0)
    constraint = -(lifted[:,0] ** 2) + np.sum(lifted[:,1:] ** 2, axis=1)
    assert np.allclose(constraint, -1.0, atol=1e-6)
    scores = lorentz_mips_scores(lifted[:1], lifted)
    assert int(np.argmax(scores[0])) == 0
```

- [ ] **Step 2: Verify RED, then implement PCA and lift**

Run: `.venv/bin/pytest -q tests/test_lorentz_rider.py -k 'pca or lift'`

Use float64 covariance/eigh, descending eigenvalues, canonical signs, float32 outputs. Lift with `radius=min(scale*norm, clip)`, `x0=cosh(radius)`, and `xs=sinh(radius)*direction`, with zero vectors mapped to `(1,0,...,0)`.

- [ ] **Step 3: Write the sign, ambient-dot, and stable-distance mutant tests**

```python
def test_wrong_sign_transform_selects_farthest():
    lifted = lorentz_lift(np.asarray([[1,0],[-1,0]], np.float32), 1.0)
    wrong_query = lifted[:1].copy()
    wrong_query[:,1:] *= -1
    assert int(np.argmax(wrong_query @ lifted.T)) == 1
    assert int(np.argmax(lorentz_mips_scores(lifted[:1], lifted)[0])) == 0

def test_fp32_distance_matches_nonnegative_fp64_oracle():
    values = np.asarray([[1,0],[np.cos(1e-4),np.sin(1e-4)],[-1,0]], np.float32)
    lifted = lorentz_lift(values, 2.5)
    actual = lorentz_distance_block(lifted, lifted)
    assert np.isfinite(actual).all()
    assert np.all(actual >= 0.0)
    assert np.allclose(np.diag(actual), 0.0, atol=2e-4)
```

Implement `u=(||xs-ys||^2-(x0-y0)^2)/2`, clip only negative roundoff to zero, and `log1p(u+sqrt(u*(u+2)))`. Implement MIPS by multiplying queries by `(-1,+1,...,+1)` and using one ordered FP32 matrix product.

- [ ] **Step 4: Run Task 2 GREEN and commit**

Run: `.venv/bin/pytest -q tests/test_lorentz_rider.py && .venv/bin/ruff check src/sfora/lorentz_rider.py tests/test_lorentz_rider.py`

```bash
git add src/sfora/lorentz_rider.py tests/test_lorentz_rider.py
git commit -m "add Lorentz lift and MIPS controls"
```

---

### Task 3: L1 Functional Controls and Selection-Priced Bootstrap

**Files:**
- Modify: `src/sfora/lorentz_rider.py`
- Modify: `tests/test_lorentz_rider.py`

**Interfaces:**
- Produces: `score_control_blocks(query_projected, gallery_projected, scale, clip) -> dict[str, Iterable[np.ndarray]]` with exact order `lorentz`, `pca_euclidean`, `pca_cosine`, `spatial_only`, `power_1`, `power_3`.
- Produces: `paired_identity_max_interval(baseline_correct: np.ndarray, interior_correct: np.ndarray, labels: np.ndarray, *, samples: int = 10_000, seed: int = 205) -> MaxInterval` where `interior_correct` has shape `(scale_count, query_count)`.
- Produces: `l1_decision(*, endpoint_gain: float, endpoint_lower: float, standard_errors: float, spatial_gain: float, power_gain: float) -> L1Decision` for one dataset. Cross-dataset status remains prohibited until separately persisted In-Shop, Cars196, and SOP reports exist.

- [ ] **Step 1: Write the endpoint-identity and max-selection RED tests**

```python
def test_l1_cannot_pass_from_endpoint_difference_alone():
    labels = np.asarray(["a","a","b","b"])
    euclidean = np.asarray([1,1,0,0], bool)
    cosine = np.asarray([0,0,1,1], bool)
    interiors = np.stack([euclidean, cosine])
    result = paired_identity_max_interval(
        np.maximum(euclidean, cosine), interiors, labels, samples=200, seed=205
    )
    assert result.point <= 0.0
    assert result.lower <= 0.0

def test_function_family_tie_closes_geometry_claim():
    decision = l1_decision(
        endpoint_gain=0.02, endpoint_lower=0.01, standard_errors=4.0,
        spatial_gain=0.00, power_gain=0.00,
    )
    assert decision.status == "CLOSE_GEOMETRY"
```

- [ ] **Step 2: Verify RED and implement one-resample max statistic**

For each replicate, sample identity indices once, compute every interior-minus-best-endpoint R@1 difference on that same resample, then store the maximum. The reported point uses the same maximum over full arrays; lower/upper are 2.5/97.5 percentiles and `standard_errors = point / std(replicates, ddof=1)`.

- [ ] **Step 3: Add exact score-family tests and implement controls**

Assert the small-radius Lorentz ranking equals PCA Euclidean, the fully clipped equal-radius ranking equals PCA cosine, and the spatial-only control differs from Lorentz only by the time-coordinate term. Assert a power-family tie prevents `GEOMETRY_SURVIVES` even when Lorentz beats both endpoints.

- [ ] **Step 4: Run Task 3 GREEN and commit**

Run: `.venv/bin/pytest -q tests/test_lorentz_rider.py && .venv/bin/ruff check src/sfora/lorentz_rider.py tests/test_lorentz_rider.py`

```bash
git add src/sfora/lorentz_rider.py tests/test_lorentz_rider.py
git commit -m "add Lorentz L1 matched controls"
```

---

### Task 4: Strict Bundle Evaluation, Report, and CLI

**Files:**
- Create: `src/sfora/lorentz_rider_io.py`
- Create: `scripts/evaluate_lorentz_rider.py`
- Create: `tests/test_lorentz_rider_io.py`
- Create: `tests/test_evaluate_lorentz_rider.py`

**Interfaces:**
- Consumes: authenticated `EmbeddingBundle` from `load_embedding_bundle`.
- Produces: `evaluate_l0_l1(bundle: EmbeddingBundle, *, dataset_index: int) -> dict[str, object]`.
- Produces: `validate_lorentz_report(value: object) -> None` and `publish_lorentz_report(path: Path, value: dict[str, object]) -> None`.
- Produces CLI: `python -I -B scripts/evaluate_lorentz_rider.py --bundle ABS --output ABS --dataset-index {0,1,2}`.

- [ ] **Step 1: Write strict report/roundtrip and mutation RED tests**

Require exact ordered top-level keys `schema_version,input,constants,l0,l1,runtime,decision,timing`. Recursively bind bundle SHA/array hashes, dimensions, PCA hashes, all seeds, per-subsample indices/hash/base/diameter/delta/relative/null values, every scale and control correctness hash, bootstrap replicate hash, and decision predicates. Generate mutations for missing/extra/reordered keys, bool-as-int, NaN/infinity, wrong convention, wrong MIPS sign label, changed scale, changed correctness hash, and a recomputed best-cell point that does not match persisted arrays.

- [ ] **Step 2: Implement evaluator and strict validator**

Run L0 before L1. L1 first runs the norm-informativeness probe at dimension 32; persist Spearman correlations with identity frequency and cosine nearest-neighbor margin. If both absolute correlations are below `0.05`, record `NORM_CHANNEL_UNINFORMATIVE` but still run the registered d=32 controls so the predeclared falsifier has direct evidence. Expand to all dimensions only when d=32 beats endpoints and functional controls.

- [ ] **Step 3: Write publication/CLI RED tests and implement atomic no-clobber output**

Use same-directory exclusive `xb` temporary creation, mode `0600`, flush/fsync, hard-link no-replace publication, directory fsync, strict reload/validation, inode-owned rollback, and foreign-temp preservation. CLI must reject relative paths, symlinks, existing output, and owned temporary residue before opening the bundle.

- [ ] **Step 4: Run Task 4 GREEN and commit**

Run: `.venv/bin/pytest -q tests/test_lorentz_rider.py tests/test_lorentz_rider_io.py tests/test_evaluate_lorentz_rider.py && .venv/bin/ruff check src/sfora/lorentz_rider.py src/sfora/lorentz_rider_io.py scripts/evaluate_lorentz_rider.py tests/test_lorentz_rider.py tests/test_lorentz_rider_io.py tests/test_evaluate_lorentz_rider.py && .venv/bin/python -m py_compile src/sfora/lorentz_rider.py src/sfora/lorentz_rider_io.py scripts/evaluate_lorentz_rider.py && git diff --check`

```bash
git add src/sfora/lorentz_rider.py src/sfora/lorentz_rider_io.py scripts/evaluate_lorentz_rider.py tests/test_lorentz_rider.py tests/test_lorentz_rider_io.py tests/test_evaluate_lorentz_rider.py
git commit -m "add Lorentz L0 L1 evaluator"
```

---

### Task 5: Review and Deferred Real Evaluation

**Files:**
- Modify: `docs/inshop_modern_baseline_reproducibility_audit_2026-08-12.md`

- [ ] **Step 1: Run focused and repository assurance once**

Run the Task 4 gate, then `.venv/bin/pytest -q`. If the known host cgroup reclaim pathology recurs in `tests/test_cli.py`, retain the original PID evidence, stop that one process, and run the complete suite excluding only that file; do not start overlapping suites.

- [ ] **Step 2: Request adversarial cross-provider review**

Use `provider="other"` with `models=["opus","gpt-5.6-sol"]`. Ask the reviewer to check the factor-two convention, medoid max-min product, train-only PCA, sign transform, endpoint equivalences, scale-selection bootstrap, power controls, self-match exclusion, report recomputation, and no-clobber mechanics. Reproduce every actionable finding before editing.

- [ ] **Step 3: Record readiness and commit**

Document exact commits, tests, review verdict, and the absent/present immutable bundle state. Commit with `git commit -m "record Lorentz falsifier readiness"`.

- [ ] **Step 4: Run one real evaluation only when its bundle exists**

Run In-Shop first after the active GPU queue is idle and the frozen UNICOM export exists. Do not interpret L0 as positive hierarchy evidence. Do not escalate beyond d=32 unless the endpoint and function-family gates pass. Cars196 and SOP require separately authenticated native-protocol adapters; no two-dataset or geometry claim is allowed before both exist.
