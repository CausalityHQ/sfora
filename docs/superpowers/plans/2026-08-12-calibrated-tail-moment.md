# Calibrated Tail Moment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and falsify the zero-training calibrated-tail-moment (CTM) descriptor as a 129-value, ordinary inner-product retrieval representation using authenticated frozen embedding exports.

**Architecture:** Keep the numerical method in a pure NumPy module, keep result-schema and publication code in a separate IO module, and expose one thin CLI over the existing strict `EmbeddingBundle` loader. Fit all bases and the scalar coefficient from train rows only; freeze them before query/gallery evaluation; compare CTM against matched row-width and equal-total-storage controls with stable rankings and query-identity bootstrap intervals.

**Tech Stack:** Python 3.12, NumPy 2.x, scikit-learn 1.5+, pytest, Ruff; no Torch, CUDA, Triton, FAISS, or remote GPU process is required for this plan.

## Global Constraints

- Implement only the CTM zero-training lane from `docs/superpowers/specs/2026-08-12-modern-pareto-program-design.md`; do not implement PARC, ORBIT, or a custom retrieval kernel.
- Treat every input embedding as FP32 and normalize it with ordered FP64 reductions before fitting or evaluation.
- Fit native/PCA bases and `lambda` on train rows only; do not use query/gallery embeddings, labels, R@1, or mAP@R to select width, basis, or coefficient.
- The fitted neighbor set is top-50 head-only inner product with self excluded and stable row-index tie breaking; the fitted response must not influence pair selection.
- Evaluate registered widths from `(64, 128, 256, 512)` only when strictly below the export width; the primary decision is native width 128, represented by exactly 129 FP32 values.
- Use query-identity cluster bootstrap with gallery fixed, 10,000 PCG64 resamples, seed 205; do not bootstrap gallery rows independently.
- Include tail-direction-permuted coefficient nulls with identical head-neighbor pairs and radii, PCG64 seeds 206 through 237, and exact one-sided p-value `(1 + count(null >= observed)) / 33 <= 0.05`.
- Report encoder cost separately from descriptor build, gallery storage, and search cost. CTM is not an encoder-speed claim.
- Preserve the active remote GPU queue; all implementation tests in this plan are CPU-only.
- Use RED-GREEN TDD, no output overwrite, strict JSON reload validation, and an ordinary Git commit after each independently reviewable task.

---

### Task 1: Pure CTM coefficient fit

**Files:**
- Create: `src/sfora/calibrated_tail_moment.py`
- Create: `tests/test_calibrated_tail_moment.py`

**Interfaces:**
- Consumes: FP32 C-contiguous embedding matrices and a builtin integer head width.
- Produces: `TailMomentFit`; `fit_tail_moment(unit: np.ndarray, *, width: int, basis_kind: str, neighbors: int = 50) -> TailMomentFit`; `head_neighbor_pairs(unit: np.ndarray, *, width: int, neighbors: int = 50, chunk_size: int = 256) -> np.ndarray`; and `encode_tail_moment(unit: np.ndarray, fit: TailMomentFit, *, basis_kind: str) -> np.ndarray`.

- [ ] **Step 1: Write failing validation and hand-computed fit tests**

```python
from __future__ import annotations

import numpy as np
import pytest

from sfora.calibrated_tail_moment import (
    encode_tail_moment,
    fit_tail_moment,
    head_neighbor_pairs,
)


def test_head_pairs_use_only_head_inner_product_and_stable_row_ties() -> None:
    unit = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    pairs = head_neighbor_pairs(unit, width=2, neighbors=1)
    assert pairs.tolist() == [[0, 1], [1, 0], [2, 0]]


def test_fit_matches_the_registered_ordered_fp64_formula() -> None:
    unit = np.asarray(
        [
            [0.8, 0.0, 0.6, 0.0],
            [0.8, 0.0, 0.0, 0.6],
            [0.0, 0.8, 0.6, 0.0],
        ],
        dtype=np.float32,
    )
    fit = fit_tail_moment(unit, width=2, basis_kind="native", neighbors=1)
    pairs = fit.pairs
    tail = unit[:, 2:].astype(np.float64)
    radius = np.linalg.norm(tail, axis=1)
    x = radius[pairs[:, 0]] * radius[pairs[:, 1]]
    y = np.sum(tail[pairs[:, 0]] * tail[pairs[:, 1]], axis=1, dtype=np.float64)
    expected = float(np.sum(x * y, dtype=np.float64) / np.sum(x * x, dtype=np.float64))
    assert fit.lambda_raw == expected
    assert fit.lambda_value == min(1.0, max(0.0, expected))
    encoded = encode_tail_moment(unit, fit, basis_kind="native")
    assert encoded.shape == (3, 3)
    assert encoded.dtype == np.float32


def _unit_fixture() -> np.ndarray:
    return np.asarray(
        [
            [0.8, 0.0, 0.6, 0.0],
            [0.8, 0.0, 0.0, 0.6],
            [0.0, 0.8, 0.6, 0.0],
            [0.0, 0.8, 0.0, 0.6],
        ],
        dtype=np.float32,
    )


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_fit_rejects_nonfinite_values(bad: float) -> None:
    values = np.eye(3, dtype=np.float32)
    values[0, 0] = bad
    with pytest.raises(ValueError, match="finite"):
        fit_tail_moment(values, width=1, basis_kind="native", neighbors=1)
```

- [ ] **Step 2: Run the focused tests and confirm the missing-module RED**

Run: `.venv/bin/pytest -q tests/test_calibrated_tail_moment.py`

Expected: collection fails with `ModuleNotFoundError: No module named 'sfora.calibrated_tail_moment'`.

- [ ] **Step 3: Implement exact data contracts, pair selection, fit, and encoding**

```python
@dataclass(frozen=True, eq=False)
class TailMomentFit:
    dimension: int
    basis_kind: str
    width: int
    neighbors: int
    pairs: np.ndarray
    lambda_raw: float
    lambda_value: float


def head_neighbor_pairs(
    unit: np.ndarray, *, width: int, neighbors: int = 50, chunk_size: int = 256
) -> np.ndarray:
    _require_unit_fp32(unit)
    if type(width) is not int or not 0 < width < unit.shape[1]:
        raise ValueError("width must be a builtin integer strictly below dimension")
    if type(neighbors) is not int or not 0 < neighbors < unit.shape[0]:
        raise ValueError("neighbors must be a positive builtin integer below row count")
    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive builtin integer")
    head = unit[:, :width].astype(np.float64)
    pairs = np.empty((head.shape[0], neighbors, 2), dtype=np.int64)
    indices = np.arange(head.shape[0], dtype=np.int64)
    for start in range(0, head.shape[0], chunk_size):
        scores = head[start : start + chunk_size] @ head.T
        for offset, row in enumerate(scores):
            query_index = start + offset
            row[query_index] = -np.inf
            partition = np.argpartition(row, row.size - neighbors)[-neighbors:]
            boundary = np.min(row[partition])
            above = np.flatnonzero(row > boundary)
            tied = np.flatnonzero(row == boundary)[: neighbors - above.size]
            candidates = np.concatenate((above, tied))
            order = candidates[np.lexsort((indices[candidates], -row[candidates]))]
            pairs[query_index, :, 0] = query_index
            pairs[query_index, :, 1] = order
    return np.ascontiguousarray(pairs.reshape(-1, 2))


def fit_tail_moment(
    unit: np.ndarray, *, width: int, basis_kind: str, neighbors: int = 50
) -> TailMomentFit:
    pairs = head_neighbor_pairs(unit, width=width, neighbors=neighbors)
    tail = unit[:, width:].astype(np.float64)
    radius = np.sqrt(np.sum(tail * tail, axis=1, dtype=np.float64))
    numerator_by_row, denominator_by_row = _row_sufficient_statistics(
        tail, radius, pairs, neighbors=neighbors, query_chunk_size=256
    )
    numerator = float(np.sum(numerator_by_row, dtype=np.float64))
    denominator = float(np.sum(denominator_by_row, dtype=np.float64))
    lambda_raw = 0.0 if denominator == 0.0 else numerator / denominator
    return TailMomentFit(
        unit.shape[1], basis_kind, width, neighbors, pairs, lambda_raw,
        float(np.clip(lambda_raw, 0.0, 1.0))
    )


def encode_tail_moment(
    unit: np.ndarray, fit: TailMomentFit, *, basis_kind: str
) -> np.ndarray:
    _require_unit_fp32(unit)
    if unit.shape[1] != fit.dimension or basis_kind != fit.basis_kind:
        raise ValueError("embedding basis/dimension differs from fit")
    tail = unit[:, fit.width :].astype(np.float64)
    radius = np.sqrt(np.sum(tail * tail, axis=1, dtype=np.float64))
    scalar = np.sqrt(fit.lambda_value) * radius
    return np.ascontiguousarray(
        np.column_stack((unit[:, : fit.width], scalar.astype(np.float32))), dtype=np.float32
    )
```

Implement `_require_unit_fp32` to require exact `np.ndarray`, FP32, two dimensions, C order, finite rows, nonzero norms, and FP64-computed norms within `2e-6` of one. `_row_sufficient_statistics` processes complete query rows in fixed 256-row chunks, sums each query's 50 pair contributions in pair order, writes two FP64 vectors in query-row order, and never allocates the full `(pairs, tail_dimension)` product. The final numerator and denominator are one ordered `np.sum` over those vectors, making the coefficient independent of chunk scheduling. Make `TailMomentFit.__eq__` compare pair bytes and all scalar fields exactly.

Add a test that monkeypatches `np.lexsort` and proves each call sorts at most `neighbors` candidates even when the train set has 64 rows, plus a boundary-tie oracle comparing the selected indices to a full stable sort. The full `N x N` score matrix and a full per-query sort are both forbidden.

- [ ] **Step 4: Add mutation tests for dtypes, order, dimensions, norms, widths, neighbor counts, and zero-tail behavior**

```python
def test_zero_tail_has_zero_coefficient_and_exact_zero_scalar() -> None:
    unit = np.eye(3, dtype=np.float32)
    fit = fit_tail_moment(unit, width=3 - 1, basis_kind="native", neighbors=1)
    encoded = encode_tail_moment(unit, fit, basis_kind="native")
    assert fit.lambda_value == 0.0
    assert encoded[:, -1].tolist() == [0.0, 0.0, 0.0]


def test_pair_selection_cannot_observe_tail(monkeypatch: pytest.MonkeyPatch) -> None:
    unit = _unit_fixture()
    changed = unit.copy()
    changed[:, 2:] = changed[[1, 0, 3, 2], 2:]
    assert np.array_equal(
        head_neighbor_pairs(unit, width=2, neighbors=1),
        head_neighbor_pairs(changed, width=2, neighbors=1),
    )


def test_negative_raw_fit_clips_to_zero_and_encoding_is_exact_lambda_zero() -> None:
    unit = np.asarray(
        [[0.8, 0.0, 0.6, 0.0], [0.8, 0.0, -0.6, 0.0]], dtype=np.float32
    )
    fit = fit_tail_moment(unit, width=2, basis_kind="native", neighbors=1)
    encoded = encode_tail_moment(unit, fit, basis_kind="native")
    assert fit.lambda_raw < 0.0
    assert fit.lambda_value == 0.0
    assert encoded[:, -1].tobytes() == np.zeros(2, dtype=np.float32).tobytes()


def test_encoding_rejects_wrong_dimension_and_basis() -> None:
    fit = fit_tail_moment(_unit_fixture(), width=2, basis_kind="native", neighbors=1)
    with pytest.raises(ValueError, match="basis/dimension"):
        encode_tail_moment(_unit_fixture(), fit, basis_kind="pca")
    with pytest.raises(ValueError, match="basis/dimension"):
        encode_tail_moment(np.eye(3, dtype=np.float32), fit, basis_kind="native")
```

- [ ] **Step 5: Run focused tests and static checks**

Run: `.venv/bin/pytest -q tests/test_calibrated_tail_moment.py`

Expected: all tests pass.

Run: `.venv/bin/ruff check src/sfora/calibrated_tail_moment.py tests/test_calibrated_tail_moment.py && .venv/bin/python -m py_compile src/sfora/calibrated_tail_moment.py tests/test_calibrated_tail_moment.py && git diff --check`

Expected: all commands exit zero.

- [ ] **Step 6: Commit Task 1**

```bash
git add src/sfora/calibrated_tail_moment.py tests/test_calibrated_tail_moment.py
git commit -m "add calibrated tail moment fit"
```

---

### Task 2: Train-only basis, coefficient uncertainty, and null falsifier

**Files:**
- Modify: `src/sfora/calibrated_tail_moment.py`
- Modify: `src/sfora/unicom_retrieval_audit.py`
- Modify: `tests/test_calibrated_tail_moment.py`
- Modify: `tests/test_unicom_retrieval_audit.py`

**Interfaces:**
- Consumes: `fit_tail_moment` from Task 1 and train identity strings used only to resample coefficient uncertainty.
- Produces: `ProjectionBasis`; `fit_projection_basis(train_unit: np.ndarray, *, kind: str) -> ProjectionBasis`; `project_unit(unit: np.ndarray, basis: ProjectionBasis) -> np.ndarray`; `cluster_lambda_interval(unit: np.ndarray, labels: np.ndarray, *, width: int, basis_kind: str, samples: int = 10_000, seed: int = 205) -> LambdaInterval`; and `permuted_tail_null(unit: np.ndarray, *, width: int, basis_kind: str, neighbors: int = 50, seeds: range = range(206, 238)) -> TailNull`.

- [ ] **Step 1: Write failing native/PCA no-leakage tests**

```python
def test_pca_basis_is_fit_from_train_only_and_is_sign_canonical() -> None:
    train = np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32)
    basis = fit_projection_basis(train, kind="pca")
    assert basis.kind == "pca"
    assert basis.matrix.dtype == np.float32
    assert basis.matrix.flags.c_contiguous
    for column in basis.matrix.T:
        pivot = int(np.argmax(np.abs(column)))
        assert column[pivot] >= 0.0


def test_project_unit_never_refits_on_evaluation_rows() -> None:
    train = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    basis = fit_projection_basis(train, kind="native")
    query = np.asarray([[0.6, 0.8]], dtype=np.float32)
    assert np.array_equal(project_unit(query, basis), query)


def test_pca_projection_subtracts_only_the_frozen_train_mean() -> None:
    train = np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32)
    query = np.asarray([[0.6, 0.8]], dtype=np.float32)
    basis = fit_projection_basis(train, kind="pca")
    expected = (query.astype(np.float64) - basis.mean.astype(np.float64)) @ basis.matrix
    expected /= np.linalg.norm(expected, axis=1, keepdims=True)
    assert np.allclose(project_unit(query, basis), expected.astype(np.float32), atol=2e-7)
```

- [ ] **Step 2: Run the basis selector and confirm missing-interface failures**

Run: `.venv/bin/pytest -q tests/test_calibrated_tail_moment.py -k 'basis or project_unit'`

Expected: failures identify the absent basis interfaces.

- [ ] **Step 3: Implement native and deterministic train-fit PCA bases**

Define:

```python
@dataclass(frozen=True, eq=False)
class ProjectionBasis:
    kind: str
    mean: np.ndarray
    matrix: np.ndarray


def fit_projection_basis(train_unit: np.ndarray, *, kind: str) -> ProjectionBasis:
    _require_unit_fp32(train_unit)
    if kind == "native":
        mean = np.zeros(train_unit.shape[1], dtype=np.float32)
        matrix = np.eye(train_unit.shape[1], dtype=np.float32)
    elif kind == "pca":
        mean64 = np.mean(train_unit.astype(np.float64), axis=0, dtype=np.float64)
        centered = train_unit.astype(np.float64) - mean64
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        matrix64 = vt.T
        for column in matrix64.T:
            pivot = int(np.argmax(np.abs(column)))
            if column[pivot] < 0.0:
                column *= -1.0
        mean = np.ascontiguousarray(mean64.astype(np.float32))
        matrix = np.ascontiguousarray(matrix64.astype(np.float32))
    else:
        raise ValueError("basis kind must be native or pca")
    return ProjectionBasis(kind=kind, mean=mean, matrix=matrix)
```

`project_unit` must return the already-normalized input object unchanged when `basis.kind == "native"`. For PCA it subtracts the frozen train mean, multiplies FP64 inputs by the frozen FP32 matrix, casts to FP32, then calls the existing ordered `l2_normalize`; it must never accept or derive a basis from evaluation rows. The bundle is normalized exactly once before either path, and native CTM plus native controls all consume those same bytes. The report accounts for both PCA mean and matrix storage.

- [ ] **Step 4: Write failing identity-cluster bootstrap and tail-null tests**

```python
def test_cluster_interval_resamples_train_identities_not_pair_rows() -> None:
    labels = np.asarray(["a", "a", "b", "b"])
    draws = cluster_lambda_interval(
        _unit_fixture(), labels, width=2, basis_kind="native", samples=32, seed=205
    )
    assert draws.samples == 32
    assert draws.seed == 205
    assert draws.lower <= draws.point <= draws.upper


def test_cluster_interval_reuses_the_observed_pairs_without_duplicate_row_refits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unit = _unit_fixture()
    labels = np.asarray(["a", "a", "b", "b"])
    calls = 0
    original = head_neighbor_pairs

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr("sfora.calibrated_tail_moment.head_neighbor_pairs", counted)
    cluster_lambda_interval(
        unit, labels, width=2, basis_kind="native", samples=32, seed=205
    )
    assert calls == 1


def test_tail_permutation_keeps_head_pairs_fixed() -> None:
    observed = fit_tail_moment(
        _unit_fixture(), width=2, basis_kind="native", neighbors=1
    )
    null = permuted_tail_null(
        _unit_fixture(), width=2, basis_kind="native", neighbors=1,
        seeds=range(206, 238)
    )
    assert np.array_equal(null.pairs, observed.pairs)
    assert len(null.lambda_raw_values) == 32
    assert null.seeds == tuple(range(206, 238))
    expected_p = (1 + sum(value >= observed.lambda_raw for value in null.lambda_raw_values)) / 33
    assert null.p_value == expected_p
```

- [ ] **Step 5: Implement the uncertainty and null records**

Use frozen dataclasses `LambdaInterval(point, lower, upper, samples, seed)` and `TailNull(seeds, pairs, lambda_raw_values, p_value)`. Select the head-only pairs exactly once. Compute fixed FP64 numerator and denominator contributions per pair with the same query-row grouping as Task 1. For every two-way cluster-bootstrap draw, sample the ordered unique train identities with replacement, derive integer multiplicities, weight each frozen pair contribution by `query_identity_count * gallery_identity_count`, and sum in pair order. This accounts for a row appearing on either side without duplicating embeddings or rerunning neighbor selection. Use percentiles `[2.5, 97.5]`.

For each null seed, hold every original radius fixed, define each nonzero tail direction as `t/r`, permute only those directions with `np.random.Generator(np.random.PCG64(seed)).permutation`, reconstruct `t_null = r_original * direction_permuted`, and recompute the raw coefficient with the observed pairs and denominator. Zero-radius rows retain the zero vector. Store the pair matrix once, store only 32 builtin-float raw coefficients, and compute the exact one-sided p-value `(1 + count(null >= observed)) / 33`; do not use an interpolated percentile as a decision threshold.

- [ ] **Step 6: Run Task 2 tests and commit**

Run: `.venv/bin/pytest -q tests/test_calibrated_tail_moment.py && .venv/bin/ruff check src/sfora/calibrated_tail_moment.py tests/test_calibrated_tail_moment.py && git diff --check`

Expected: all commands exit zero.

```bash
git add src/sfora/calibrated_tail_moment.py tests/test_calibrated_tail_moment.py
git commit -m "add CTM train-only falsifiers"
```

---

### Task 3: Stable retrieval views and matched-cost controls

**Files:**
- Modify: `src/sfora/calibrated_tail_moment.py`
- Modify: `tests/test_calibrated_tail_moment.py`

**Interfaces:**
- Consumes: frozen bases and fits from Tasks 1-2.
- Produces: shared `retrieval_metrics_from_scores(...)`; `ScoreView`; `evaluate_inner_product(query: np.ndarray, gallery: np.ndarray, query_labels: np.ndarray, gallery_labels: np.ndarray, *, values_per_row: int, fixed_bytes: int = 0, query_projection_seconds: float = 0.0, chunk_size: int = 256) -> ScoreView`; `evaluate_width(query_native: np.ndarray, gallery_native: np.ndarray, query_pca: np.ndarray, gallery_pca: np.ndarray, query_labels: np.ndarray, gallery_labels: np.ndarray, *, width: int, native_fit: TailMomentFit, pca_fit: TailMomentFit, pca_fixed_bytes: int) -> dict[str, ScoreView]`; and `query_identity_interval(baseline_correct: np.ndarray, candidate_correct: np.ndarray, query_labels: np.ndarray, *, samples: int = 10_000, seed: int = 205) -> PairedInterval`.

- [ ] **Step 1: Write failing stable-ranking and metric tests**

```python
def test_inner_product_breaks_score_ties_by_gallery_row() -> None:
    query = np.asarray([[1.0, 0.0]], dtype=np.float32)
    gallery = np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    view = evaluate_inner_product(query, gallery, np.asarray(["x"]), np.asarray(["y", "x"]))
    assert view.top1_indices.tolist() == [0]
    assert view.top1_correct.tolist() == [False]


def test_ctm_score_equals_registered_formula() -> None:
    unit = _unit_fixture()
    fit = fit_tail_moment(unit, width=2, basis_kind="native", neighbors=1)
    encoded = encode_tail_moment(unit, fit, basis_kind="native").astype(np.float64)
    direct = encoded @ encoded.T
    head = unit[:, :2].astype(np.float64)
    tail = unit[:, 2:].astype(np.float64)
    radius = np.linalg.norm(tail, axis=1)
    formula = head @ head.T + fit.lambda_value * np.outer(radius, radius)
    assert np.allclose(direct, formula, rtol=0.0, atol=2e-7)
```

- [ ] **Step 2: Run the retrieval selector and verify missing-interface REDs**

Run: `.venv/bin/pytest -q tests/test_calibrated_tail_moment.py -k 'inner_product or registered_formula or width_controls'`

Expected: failures identify absent retrieval interfaces.

- [ ] **Step 3: Implement stable inner-product evaluation and all controls**

First extract `retrieval_metrics_from_scores(scores, query_labels, gallery_labels) -> RetrievalView` in `unicom_retrieval_audit.py`. It must retain the current exact contract: require at least one relevant gallery row, select `min(max(30, relevant), gallery_count)` entries, break equal scores by ascending gallery row, compute AP over the first `relevant` ranks with denominator `relevant`, and return the existing `RetrievalView`. Refactor `retrieval_view` to call this shared core. Add a test that the refactored Euclidean path is byte-for-byte unchanged and a CTM test proving inner product on unit vectors matches `retrieval_view` recall, mAP@R, indices, and correctness.

Define `ScoreView(recall, map_at_r, top1_indices, top1_correct, values_per_row, fixed_bytes, total_bytes, query_projection_seconds)` with recalls `(1, 10, 20, 30)`. Compute scores in query chunks as ordered FP64 matrix products and call the shared metrics core. `total_bytes = gallery_count * values_per_row * 4 + fixed_bytes`; PCA's fixed bytes are its FP32 matrix plus mean, and its measured projection time is included in query latency. Project native and PCA train/query/gallery arrays once per bundle, not once per width. Implement `evaluate_width` without a train-array argument and return an exact ordered mapping with these keys:

```python
CONTROL_ORDER = (
    "native_ctm",
    "pca_ctm",
    "renormalized_prefix_plus_zero",
    "plain_prefix_plus_zero",
    "lambda_zero",
    "unicom_tail_energy",
    "pca_renormalized_prefix_plus_zero",
    "tail_sign_control",
    "tail_permuted_control",
    "official_512",
    "full_width_768",
)

MATCHED_ROW_WIDTH_CONTROLS = CONTROL_ORDER[2:9]
EQUAL_TOTAL_STORAGE_CONTROLS = (
    "renormalized_prefix_plus_zero",
    "plain_prefix_plus_zero",
    "lambda_zero",
    "unicom_tail_energy",
    "tail_sign_control",
    "tail_permuted_control",
)
```

Every matched row-width record contains exactly `width + 1` FP32 values. `native_ctm` is the primary candidate. `pca_ctm` applies the same fit after the frozen train-PCA transform and is diagnostic unless its matrix/mean overhead also yields a measured total-storage Pareto point; it cannot support the native 3.97x claim. `renormalized_prefix_plus_zero` is the first `width + 1` original coordinates renormalized to unit length. `plain_prefix_plus_zero` is the first `width + 1` original coordinates without renormalization. `lambda_zero` is `[h, 0]`, so it is not silently conflated with either truncation control. `unicom_tail_energy` ranks by `<h_q,h_g> + r_g**2/2` using a dedicated scorer because it is not a symmetric descriptor inner product. `tail_sign_control` negates the CTM scalar on alternating gallery rows fixed by row index. `tail_permuted_control` permutes gallery radii with PCG64 seed 238. `pca_renormalized_prefix_plus_zero` is the first `width + 1` train-PCA coordinates renormalized to unit length and is a quality control, but its matrix/mean overhead excludes it from equal-total-storage controls. `official_512` uses the published UNICOM normalize-then-first-512-unrenormalized geometry and is the quality-gap/storage anchor. `full_width_768` is a separately labelled diagnostic. For non-UNICOM future adapters, those named ceilings must be supplied by a dataset-specific contract rather than inferred.

- [ ] **Step 4: Write and implement query-identity clustered intervals**

```python
def test_query_bootstrap_keeps_gallery_fixed_and_clusters_duplicate_identity_rows() -> None:
    labels = np.asarray(["a", "a", "b", "c"])
    baseline = np.asarray([False, True, False, True], dtype=np.bool_)
    candidate = np.asarray([True, True, False, True], dtype=np.bool_)
    result = query_identity_interval(
        baseline, candidate, labels, samples=10_000, seed=205
    )
    assert result.samples == 10_000
    assert result.point == 0.25
    assert result.lower <= result.point <= result.upper
```

The implementation must sample the ordered unique query identities with replacement and include every query row for each sampled identity. It receives correctness vectors only, so gallery rows cannot be resampled accidentally. Its reported point is the original query-row mean; every bootstrap draw remains a pooled query-row mean after concatenating the sampled identity clusters.

- [ ] **Step 5: Add primary gate boundary tests**

Implement `ctm_decision(*, ctm_r1: float, ctm_map_at_r: float, renormalized_r1: float, renormalized_map_at_r: float, official_512_r1: float, paired_lower: float, control_r1: Mapping[str, float], ctm_total_bytes: int, control_total_bytes: Mapping[str, int], lambda_raw: float, lambda_lower: float, null_p_value: float, width_gains: tuple[tuple[int, float], ...], replication_status: str) -> CTMDecision`. `replication_status` is exactly `PENDING`, `PASSED`, or `FAILED`; no nullable boolean is serialized. `CTMDecision` has exact fields `status`, `reason`, `r1_gain_passed`, `r1_lower_passed`, `map_passed`, `gap_recovery_passed`, `quality_controls_passed`, `equal_storage_passed`, `coefficient_passed`, `some_width_signal_passed`, and `replication_passed`. Status is one of `CLOSE`, `USE_RENORMALIZED`, `REPLICATE`, or `GENERAL_CLAIM_READY`.

Short-circuit to `CLOSE` before query/gallery evaluation when the primary native-128 fit has `lambda_value == 0.0` or every encoded scalar would round to FP32 zero. Otherwise test equality boundaries: `+0.003` R@1 is inclusive, lower bound must be strictly positive, mAP@R loss `0.001` is inclusive, recovery of the positive `official_512 - renormalized_129` gap by `0.5` is inclusive, every key in `MATCHED_ROW_WIDTH_CONTROLS` must be strictly beaten in R@1, native CTM total bytes must equal the controls in `EQUAL_TOTAL_STORAGE_CONTROLS`, coefficient lower bound must be positive, null p-value must be at most `0.05`, and at least one registered width gain must be `>= 0.001`. If renormalized 129-D matches/exceeds `official_512`, select `USE_RENORMALIZED`; failed Cars replication selects `CLOSE`; passing In-Shop with pending replication selects `REPLICATE`.

- [ ] **Step 6: Run Task 3 tests and commit**

Run: `.venv/bin/pytest -q tests/test_calibrated_tail_moment.py tests/test_unicom_retrieval_audit.py && .venv/bin/ruff check src/sfora/calibrated_tail_moment.py src/sfora/unicom_retrieval_audit.py tests/test_calibrated_tail_moment.py tests/test_unicom_retrieval_audit.py && .venv/bin/python -m py_compile src/sfora/calibrated_tail_moment.py src/sfora/unicom_retrieval_audit.py tests/test_calibrated_tail_moment.py tests/test_unicom_retrieval_audit.py && git diff --check`

Expected: all commands exit zero.

```bash
git add src/sfora/calibrated_tail_moment.py src/sfora/unicom_retrieval_audit.py tests/test_calibrated_tail_moment.py tests/test_unicom_retrieval_audit.py
git commit -m "add CTM retrieval controls and gates"
```

---

### Task 4: Strict report schema and atomic CLI

**Files:**
- Create: `src/sfora/calibrated_tail_moment_io.py`
- Create: `scripts/evaluate_calibrated_tail_moment.py`
- Create: `tests/test_calibrated_tail_moment_io.py`
- Create: `tests/test_evaluate_calibrated_tail_moment.py`

**Interfaces:**
- Consumes: `EmbeddingBundle` from `sfora.unicom_audit_io` and CTM records from Tasks 1-3.
- Produces: `build_ctm_report(bundle: EmbeddingBundle, evaluation: Mapping[str, object]) -> dict[str, object]`; `validate_ctm_report(value: object, *, expected_widths: tuple[int, ...] = (64, 128, 256, 512), expected_neighbors: int = 50, expected_null_count: int = 32) -> None`; `publish_ctm_report(path: Path, payload: Mapping[str, object], *, expected_widths: tuple[int, ...] = (64, 128, 256, 512), expected_neighbors: int = 50, expected_null_count: int = 32) -> None`; `evaluate_bundle(bundle: EmbeddingBundle, *, widths: tuple[int, ...] = (64, 128, 256, 512), neighbors: int = 50, null_seeds: range = range(206, 238)) -> dict[str, object]`; and CLI `run(arguments: Sequence[str] | None = None) -> int`.

- [ ] **Step 1: Write the report-schema RED**

Create a small synthetic bundle and exact expected key order:

```python
REPORT_KEYS = (
    "schema_version",
    "phase",
    "input",
    "constants",
    "fits",
    "views",
    "primary",
    "timing",
    "storage",
    "runtime",
    "decision",
)


def _bundle(tmp_path: Path) -> EmbeddingBundle:
    train = np.tile(np.eye(4, dtype=np.float32), (13, 1))
    query = np.eye(4, dtype=np.float32)
    gallery = np.eye(4, dtype=np.float32)
    labels = np.asarray(["a", "b", "c", "d"])
    return EmbeddingBundle(
        path=(tmp_path / "synthetic.npz").resolve(),
        sha256="a" * 64,
        metadata={
            "schema_version": 1,
            "model_identifier": "synthetic-ctm",
            "model_revision": "b" * 40,
            "checkpoint_sha256": "c" * 64,
            "image_list_sha256": "d" * 64,
            "transform": "synthetic-unit",
            "embedding_dimension": 4,
            "split_counts": {"train": 52, "query": 4, "gallery": 4},
            "array_sha256": {
                "train_embeddings": "e" * 64,
                "train_labels": "f" * 64,
                "query_embeddings": "1" * 64,
                "query_labels": "2" * 64,
                "gallery_embeddings": "3" * 64,
                "gallery_labels": "4" * 64,
            },
        },
        train_embeddings=np.ascontiguousarray(train),
        train_labels=np.repeat(labels, 13),
        query_embeddings=query,
        query_labels=labels,
        gallery_embeddings=gallery,
        gallery_labels=labels,
    )


def _valid_report(tmp_path: Path) -> dict[str, object]:
    report = evaluate_bundle(_bundle(tmp_path), widths=(2,), neighbors=1, null_seeds=range(206, 210))
    validate_ctm_report(report, expected_widths=(2,), expected_neighbors=1, expected_null_count=4)
    return report


def test_report_roundtrips_with_exact_recursive_schema(tmp_path: Path) -> None:
    report = _valid_report(tmp_path)
    validate_ctm_report(report, expected_widths=(2,), expected_neighbors=1, expected_null_count=4)
    persisted = json.loads(json.dumps(report, allow_nan=False, separators=(",", ":")))
    validate_ctm_report(
        persisted, expected_widths=(2,), expected_neighbors=1, expected_null_count=4
    )
    assert tuple(persisted) == REPORT_KEYS


def test_validator_recomputes_primary_decision(tmp_path: Path) -> None:
    report = _valid_report(tmp_path)
    report["decision"]["status"] = "CONTINUE"
    with pytest.raises(ValueError, match="decision"):
        validate_ctm_report(
            report, expected_widths=(2,), expected_neighbors=1, expected_null_count=4
        )
```

- [ ] **Step 2: Run report tests and verify the missing-module RED**

Run: `.venv/bin/pytest -q tests/test_calibrated_tail_moment_io.py`

Expected: collection fails for absent `sfora.calibrated_tail_moment_io`.

- [ ] **Step 3: Implement exact report construction and validation**

Use builtin JSON-compatible types only. Bind the input bundle path/SHA/metadata and array SHA values. Record widths, bases, top-50 pair rule, bootstrap/null seeds and sample counts, every fitted coefficient/interval/null summary, all control metrics and correctness hashes, timing for basis fit/coefficient fit/descriptor build/search, gallery/query descriptor bytes, PCA matrix bytes, and runtime Python/NumPy versions. Recompute every decision boolean from persisted scalar fields in `validate_ctm_report`; reject unknown/missing/reordered keys, bool-as-int, nonfinite values, wrong list counts, malformed hashes, and inconsistent storage arithmetic.

The report is an exact `phase`-tagged union. `FIT_CLOSED` contains input/constants/fits/null/timing/runtime plus a `CLOSE` decision, and requires `views`, `primary`, and `storage` to be JSON null; it is emitted when native-128 clips or encodes to zero and proves query/gallery evaluation did not run. `EVALUATED` requires the complete view, primary, and storage objects. Production validation derives the applicable widths from `input.metadata.embedding_dimension`, so 512 is included only when strictly below the input width. The report stores all matched-control R@1 values, total bytes, per-width gains, replication enum, and enough fields to recompute every decision branch.

- [ ] **Step 4: Implement no-clobber publication and strict reload**

`publish_ctm_report` must call `validate_ctm_report`, serialize with `json.dumps(payload, allow_nan=False, indent=2) + "\n"`, create a same-directory mode-0600 temporary with exclusive create, fsync it, publish with hard-link no-replace, fsync the directory, delete only its owned temporary, strict-load the published bytes, and call `validate_ctm_report` again. Reuse no mutable implementation state between validation and reload.

- [ ] **Step 5: Write and implement the thin CLI**

```python
def test_cli_runs_registered_grid_once_and_no_clobbers(tmp_path, monkeypatch) -> None:
    module = _load_script()
    output = tmp_path / "ctm.json"
    calls: list[str] = []
    monkeypatch.setattr(module, "load_embedding_bundle", lambda path: _bundle(tmp_path))
    monkeypatch.setattr(module, "REGISTERED_WIDTHS", (2,))
    monkeypatch.setattr(module, "REGISTERED_NEIGHBORS", 1)
    monkeypatch.setattr(module, "REGISTERED_NULL_SEEDS", range(206, 210))
    monkeypatch.setattr(
        module,
        "evaluate_bundle",
        lambda bundle, **kwargs: calls.append("evaluate") or _valid_report(tmp_path),
    )
    assert module.run(["--bundle", "in.npz", "--output", str(output)]) == 0
    first = output.read_bytes()
    assert module.run(["--bundle", "in.npz", "--output", str(output)]) == 2
    assert calls == ["evaluate"]
    assert output.read_bytes() == first
```

CLI arguments are exactly `--bundle` and `--output`. Before loading the bundle or running an evaluation, `run` rejects an existing output path or any matching owned temporary; hard-link publication remains the race-safe final gate. `evaluate_bundle` runs native and train-only PCA bases for every registered width below the export dimension, freezes each fit before query/gallery encoding, and reports the primary native-128 decision. Production module constants are exactly `(64, 128, 256, 512)`, 50 neighbors, and `range(206, 238)`; `run` passes them to evaluation, validation, and publication. The keyword-only reduced constants shown in `_valid_report` exist only to make exhaustive CPU tests small and are recorded in the returned report; tests monkeypatch all three module constants together, while an unmodified production CLI cannot accept a reduced report. Evaluation may read query/gallery labels only inside retrieval scoring and query-cluster bootstrap, never in basis/coefficient fitting.

- [ ] **Step 6: Add failure-path and isolation tests**

Test a preexisting destination, preexisting owned-temp name, hard-link race, serialization failure, fsync failure, strict-reload mutation, malformed bundle, fit failure, and evaluation failure. In every case assert destination preservation and no owned temp. Import the CLI in a fresh `python -I` subprocess and assert `torch`, `faiss`, and training modules are absent from `sys.modules`.

- [ ] **Step 7: Run Task 4 tests and commit**

Run: `.venv/bin/pytest -q tests/test_calibrated_tail_moment.py tests/test_calibrated_tail_moment_io.py tests/test_evaluate_calibrated_tail_moment.py`

Expected: all tests pass.

Run: `.venv/bin/ruff check src/sfora/calibrated_tail_moment.py src/sfora/calibrated_tail_moment_io.py scripts/evaluate_calibrated_tail_moment.py tests/test_calibrated_tail_moment.py tests/test_calibrated_tail_moment_io.py tests/test_evaluate_calibrated_tail_moment.py && .venv/bin/python -m py_compile src/sfora/calibrated_tail_moment.py src/sfora/calibrated_tail_moment_io.py scripts/evaluate_calibrated_tail_moment.py && git diff --check`

Expected: all commands exit zero.

```bash
git add src/sfora/calibrated_tail_moment_io.py scripts/evaluate_calibrated_tail_moment.py tests/test_calibrated_tail_moment_io.py tests/test_evaluate_calibrated_tail_moment.py
git commit -m "add CTM frozen-bundle evaluator"
```

---

### Task 5: Run the synthetic assurance gate and prepare real exports

**Files:**
- Modify: `docs/inshop_modern_baseline_reproducibility_audit_2026-08-12.md`
- Test: all CTM files from Tasks 1-4 plus existing UNICOM IO/retrieval tests.

**Interfaces:**
- Consumes: committed CTM evaluator and the existing UNICOM/Cars/PA embedding bundle format.
- Produces: a verified CPU implementation ready to consume each immutable export without changing the active GPU queue.

- [ ] **Step 1: Run the complete focused assurance set**

Run: `.venv/bin/pytest -q tests/test_calibrated_tail_moment.py tests/test_calibrated_tail_moment_io.py tests/test_evaluate_calibrated_tail_moment.py tests/test_unicom_audit_io.py tests/test_unicom_retrieval_audit.py`

Expected: all tests pass.

- [ ] **Step 2: Run repository-wide assurance once**

Run: `.venv/bin/pytest -q`

Expected: all tests pass. If this gate fails, repair only the failing layer, rerun that layer, then run one final repository-wide gate; never launch overlapping suites.

- [ ] **Step 3: Record readiness and exact deferred run commands**

Append a CTM implementation section to the audit with commit IDs, test totals, and this command, but do not execute it until the UNICOM bundle exists and the active queue is idle:

```bash
.venv/bin/python -I -B scripts/evaluate_calibrated_tail_moment.py \
  --bundle reports/generated/unicom_inshop_embeddings.npz \
  --output reports/generated/unicom_inshop_ctm.json
```

State that UNICOM is the first decision. Explicitly record that the current Cars196 and PA exporters use incompatible split-specific archives, that this CLI must reject them, and that a separately reviewed adapter is required before Cars196 replication or a general claim. Do not report a quality/storage win until a strict persisted In-Shop result passes its gates; do not report a general claim until a later Cars196 adapter/result also passes.

- [ ] **Step 4: Review the implementation against the frozen design**

Request a read-only cross-provider critique with explicit fallback order `models=["opus", "gpt-5.6-sol"]`. Ask the reviewer to inspect the committed diff for train/test leakage, head-pair response leakage, two-way bootstrap clustering, ranking ties, matched-row and total-storage accounting, PCA storage/timing, and decision-boundary bugs. Independently reproduce every actionable finding before editing.

- [ ] **Step 5: Commit the readiness record**

```bash
git add docs/inshop_modern_baseline_reproducibility_audit_2026-08-12.md
git commit -m "record CTM evaluator readiness"
```

---

## Deferred follow-on plans

The following are separate projects and intentionally do not expand this plan:

1. **Modern anchor reproduction:** DADA six-epoch structural smoke, three-seed faithful reproduction, and VPTSP-G fallback after the current GPU queue exits.
2. **Cars/PA CTM adapters:** define a strict split-specific input contract; Cars196 uses test-as-query-and-gallery leave-one-out with self-example exclusion, while PA In-Shop needs its train/query/gallery split exports combined without coercing int labels into the UNICOM schema. Verify each adapter against its native evaluator before replication.
3. **Training throughput:** profile the reproduced winning anchor, test maintained AMP/fused-optimizer/channels-last/compile/input-pipeline changes individually, then gate their composition at at least 20% images/s plus end-to-end wall-time improvement and funded TOST equivalence.
4. **Custom kernel:** remain closed unless a profiler shows one unsupported operator consumes at least 10% of step time after maintained optimizations. A common optimized kernel is valuable only if it reduces measured training or inference latency without changing the quality comparison.
5. **Retrieval kernel:** remain closed until an actual served 1M+ workload with named consumer and recorded trace exists.
