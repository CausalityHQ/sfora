from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/evaluate_progressive_residual_ann.py"
SPEC = importlib.util.spec_from_file_location("evaluate_progressive_residual_ann", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
SUBJECT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SUBJECT
SPEC.loader.exec_module(SUBJECT)


@pytest.mark.parametrize(
    ("indexes", "bits", "wire"),
    [
        (np.array([[0, 1, 2, 3]], dtype=np.uint8), 2, [[0x1B]]),
        (np.array([[0, 1, 7]], dtype=np.uint8), 3, [[0x07, 0x80]]),
        (np.array([[0, 255]], dtype=np.uint8), 8, [[0x00, 0xFF]]),
    ],
)
def test_dense_scalar_codes_have_exact_packed_wire_bytes(
    indexes: np.ndarray,
    bits: int,
    wire: list[list[int]],
) -> None:
    packed = SUBJECT.pack_dense_unsigned_codes(indexes, bits=bits)

    assert packed.tolist() == wire
    assert packed.shape[1] == (indexes.shape[1] * bits + 7) // 8
    assert np.array_equal(
        SUBJECT.unpack_dense_unsigned_codes(
            packed,
            dimensions=indexes.shape[1],
            bits=bits,
        ),
        indexes,
    )


def test_dense_scalar_codes_reject_range_shape_and_padding_drift() -> None:
    indexes = np.array([[0, 1, 2]], dtype=np.uint8)
    packed = SUBJECT.pack_dense_unsigned_codes(indexes, bits=2)
    bad_padding = packed.copy()
    bad_padding[0, -1] |= 0x01

    for value, bits in (
        (indexes.astype(np.int64), 2),
        (np.array([[0, 1, 4]], dtype=np.uint8), 2),
        (indexes, True),
        (indexes, 0),
        (indexes, 9),
    ):
        with pytest.raises(ValueError, match="dense scalar code authority differs"):
            SUBJECT.pack_dense_unsigned_codes(value, bits=bits)
    for value, dimensions, bits in (
        (packed.astype(np.int64), 3, 2),
        (bad_padding, 3, 2),
        (packed, 5, 2),
        (packed, 3, True),
    ):
        with pytest.raises(ValueError, match="dense scalar code authority differs"):
            SUBJECT.unpack_dense_unsigned_codes(
                value,
                dimensions=dimensions,
                bits=bits,
            )


def test_dense_scalar_quantizer_fits_exact_endpoint_grid_and_storage() -> None:
    training = np.array([[0.0, 0.0], [3.0, 6.0]], dtype=np.float32)
    quantizer = SUBJECT.fit_dense_scalar_quantizer(
        training,
        bits=2,
        metric="squared_l2",
    )

    assert quantizer.bits == 2
    assert quantizer.metric == "squared_l2"
    assert quantizer.dimensions == 2
    assert quantizer.bytes_per_vector == 1
    assert quantizer.shared_bytes == 16
    assert quantizer.minimums.tolist() == [0.0, 0.0]
    assert quantizer.steps.tolist() == [1.0, 2.0]
    assert SUBJECT.encode_dense_scalar_quantizer(
        quantizer,
        np.array([[0.0, 2.0], [3.0, 4.0]], dtype=np.float32),
    ).tolist() == [[0x10], [0xE0]]


@pytest.mark.parametrize("metric", ["angular", "squared_l2"])
def test_dense_scalar_candidate_scoring_is_metric_exact_and_stable(metric: str) -> None:
    gallery = np.array(
        [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [0.0, 1.0]],
        dtype=np.float32,
    )
    if metric == "squared_l2":
        gallery = np.array(
            [[0.0, 0.0], [0.2, 0.0], [1.0, 0.0], [1.0, 0.0]],
            dtype=np.float32,
        )
    quantizer = SUBJECT.fit_dense_scalar_quantizer(gallery, bits=8, metric=metric)
    payload = SUBJECT.encode_dense_scalar_quantizer(quantizer, gallery)
    candidates = np.array([[3, 2, 1, 0]], dtype=np.int64)
    query = np.array([[1.0, 0.0]], dtype=np.float32)
    if metric == "squared_l2":
        query = np.array([[0.0, 0.0]], dtype=np.float32)

    ranked, scores = SUBJECT.score_dense_scalar_candidates(
        quantizer,
        payload,
        query,
        candidates,
        result_width=3,
    )

    assert ranked.tolist() == [[0, 1, 2]]
    assert scores.dtype == np.float64
    assert np.isfinite(scores).all()


@pytest.mark.parametrize(
    "mutation",
    ["fit-dtype", "fit-nonfinite", "metric", "encode-shape", "candidate-duplicate"],
)
def test_dense_scalar_quantizer_rejects_authority_drift(mutation: str) -> None:
    training = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    if mutation == "fit-dtype":
        training = training.astype(np.float64)
    if mutation == "fit-nonfinite":
        training[0, 0] = np.nan
    if mutation in {"fit-dtype", "fit-nonfinite", "metric"}:
        with pytest.raises(ValueError, match="dense scalar quantizer authority differs"):
            SUBJECT.fit_dense_scalar_quantizer(
                training,
                bits=4,
                metric="cosine" if mutation == "metric" else "squared_l2",
            )
        return
    quantizer = SUBJECT.fit_dense_scalar_quantizer(
        training,
        bits=4,
        metric="squared_l2",
    )
    payload = SUBJECT.encode_dense_scalar_quantizer(quantizer, training)
    if mutation == "encode-shape":
        with pytest.raises(ValueError, match="dense scalar quantizer authority differs"):
            SUBJECT.encode_dense_scalar_quantizer(quantizer, training[:, :1])
        return
    with pytest.raises(ValueError, match="dense scalar quantizer authority differs"):
        SUBJECT.score_dense_scalar_candidates(
            quantizer,
            payload,
            training[:1],
            np.array([[0, 0]], dtype=np.int64),
            result_width=1,
        )


def _write_ann_benchmark_fixture(
    path: Path,
    *,
    metric: str,
    mutation: str | None = None,
) -> str:
    train = np.array(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]],
        dtype=np.float32,
    )
    test = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    neighbors = np.array([[0, 3], [1, 0]], dtype=np.int32)
    distances = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32)
    if metric == "euclidean":
        train = np.array(
            [[0.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [2.0, 0.0]],
            dtype=np.float32,
        )
        test = np.array([[0.0, 0.0], [2.0, 0.0]], dtype=np.float32)
        neighbors = np.array([[0, 2], [3, 1]], dtype=np.int32)
        distances = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32)

    if mutation == "train-dtype":
        train = train.astype(np.float64)
    elif mutation == "test-nonfinite":
        test[0, 0] = np.nan
    elif mutation == "duplicate-neighbor":
        neighbors[0, 1] = neighbors[0, 0]
    elif mutation == "out-of-range-neighbor":
        neighbors[0, 1] = len(train)
    elif mutation == "wrong-truth-set":
        neighbors[0, 1] = 2 if metric == "angular" else 3
        distances[0, 1] = 2.0
    elif mutation == "shape":
        distances = distances[:, :1]

    with h5py.File(path, "w") as handle:
        handle.attrs["distance"] = metric
        handle.create_dataset("train", data=train)
        handle.create_dataset("test", data=test)
        handle.create_dataset("neighbors", data=neighbors)
        handle.create_dataset("distances", data=distances)
        if mutation == "extra-dataset":
            handle.create_dataset("labels", data=np.arange(len(train), dtype=np.int32))
        elif mutation == "missing-dataset":
            del handle["distances"]
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    ("stored_metric", "metric"),
    [("angular", "angular"), ("euclidean", "squared_l2")],
)
def test_load_ann_benchmark_authenticates_native_metric_and_tied_truth(
    tmp_path: Path,
    stored_metric: str,
    metric: str,
) -> None:
    path = tmp_path / "dataset.hdf5"
    digest = _write_ann_benchmark_fixture(path, metric=stored_metric)

    dataset = SUBJECT.load_ann_benchmark(
        path,
        expected_sha256=digest,
        expected_metric=metric,
    )

    assert dataset.source_sha256 == digest
    assert dataset.metric == metric
    assert dataset.train.shape == (4, 2)
    assert dataset.test.shape == (2, 2)
    assert dataset.truth_ordinals.tolist() == (
        [[0, 3], [1, 0]]
        if metric == "angular"
        else [[0, 2], [3, 1]]
    )
    assert dataset.truth_distances.tolist() == [[0.0, 1.0], [0.0, 1.0]]
    assert not dataset.train.flags.writeable
    assert not dataset.test.flags.writeable
    assert not dataset.truth_ordinals.flags.writeable
    assert not dataset.truth_distances.flags.writeable


@pytest.mark.parametrize(
    "mutation",
    [
        "extra-dataset",
        "missing-dataset",
        "train-dtype",
        "test-nonfinite",
        "duplicate-neighbor",
        "out-of-range-neighbor",
        "wrong-truth-set",
        "shape",
    ],
)
def test_load_ann_benchmark_rejects_schema_identity_and_truth_drift(
    tmp_path: Path,
    mutation: str,
) -> None:
    path = tmp_path / "dataset.hdf5"
    digest = _write_ann_benchmark_fixture(
        path,
        metric="angular",
        mutation=mutation,
    )

    with pytest.raises(ValueError, match="ANN-Benchmarks authority differs"):
        SUBJECT.load_ann_benchmark(
            path,
            expected_sha256=digest,
            expected_metric="angular",
        )


@pytest.mark.parametrize(
    ("digest", "metric"),
    [
        ("0" * 64, "angular"),
        (None, "angular"),
        ("placeholder", "angular"),
        ("observed", "cosine"),
    ],
)
def test_load_ann_benchmark_rejects_digest_and_metric_authority_drift(
    tmp_path: Path,
    digest: object,
    metric: object,
) -> None:
    path = tmp_path / "dataset.hdf5"
    observed = _write_ann_benchmark_fixture(path, metric="angular")
    expected_digest = observed if digest == "observed" else digest

    with pytest.raises(ValueError, match="ANN-Benchmarks authority differs"):
        SUBJECT.load_ann_benchmark(
            path,
            expected_sha256=expected_digest,
            expected_metric=metric,
        )


def test_ann_benchmark_recall_accepts_equal_distance_truth_ties(tmp_path: Path) -> None:
    path = tmp_path / "sift-like.hdf5"
    digest = _write_ann_benchmark_fixture(path, metric="euclidean")
    dataset = SUBJECT.load_ann_benchmark(
        path,
        expected_sha256=digest,
        expected_metric="squared_l2",
    )

    hits = SUBJECT.ann_benchmark_recall_hits(
        dataset,
        np.array([[0, 1], [3, 1]], dtype=np.int64),
        query_ordinals=np.array([0, 1], dtype=np.int64),
        count=2,
    )
    tied = SUBJECT.ann_benchmark_recall_hits(
        dataset,
        np.array([[0, 2], [3, 1]], dtype=np.int64),
        query_ordinals=np.array([0, 1], dtype=np.int64),
        count=2,
    )

    assert hits.tolist() == [2, 2]
    assert tied.tolist() == [2, 2]


def test_ann_benchmark_recall_scores_an_explicit_query_subset(tmp_path: Path) -> None:
    path = tmp_path / "sift-like.hdf5"
    digest = _write_ann_benchmark_fixture(path, metric="euclidean")
    dataset = SUBJECT.load_ann_benchmark(
        path,
        expected_sha256=digest,
        expected_metric="squared_l2",
    )

    hits = SUBJECT.ann_benchmark_recall_hits(
        dataset,
        np.array([[0, 2]], dtype=np.int64),
        query_ordinals=np.array([0], dtype=np.int64),
        count=2,
    )

    assert hits.tolist() == [2]


def test_ann_benchmark_containment_scores_all_candidates_with_truth_ties(
    tmp_path: Path,
) -> None:
    path = tmp_path / "sift-like.hdf5"
    digest = _write_ann_benchmark_fixture(path, metric="euclidean")
    dataset = SUBJECT.load_ann_benchmark(
        path,
        expected_sha256=digest,
        expected_metric="squared_l2",
    )

    hits = SUBJECT.ann_benchmark_containment_hits(
        dataset,
        np.array([[3, 0, 2], [0, 3, 1]], dtype=np.int64),
        query_ordinals=np.array([0, 1], dtype=np.int64),
        candidate_width=3,
        truth_width=2,
    )

    assert hits.tolist() == [2, 2]


@pytest.mark.parametrize(
    ("returned", "count"),
    [
        (np.array([[0, 0], [3, 1]], dtype=np.int64), 2),
        (np.array([[0, 4], [3, 1]], dtype=np.int64), 2),
        (np.array([[0, 1], [3, 1]], dtype=np.int32), 2),
        (np.array([[0, 1], [3, 1]], dtype=np.int64), True),
        (np.array([[0, 1], [3, 1]], dtype=np.int64), 3),
    ],
)
def test_ann_benchmark_recall_rejects_result_authority_drift(
    tmp_path: Path,
    returned: np.ndarray,
    count: object,
) -> None:
    path = tmp_path / "sift-like.hdf5"
    digest = _write_ann_benchmark_fixture(path, metric="euclidean")
    dataset = SUBJECT.load_ann_benchmark(
        path,
        expected_sha256=digest,
        expected_metric="squared_l2",
    )

    with pytest.raises(ValueError, match="ANN-Benchmarks result differs"):
        SUBJECT.ann_benchmark_recall_hits(
            dataset,
            returned,
            query_ordinals=np.arange(returned.shape[0], dtype=np.int64),
            count=count,
        )


@pytest.mark.parametrize(
    "query_ordinals",
    [
        np.array([0, 1], dtype=np.int32),
        np.array([0, 0], dtype=np.int64),
        np.array([-1, 1], dtype=np.int64),
        np.array([0, 2], dtype=np.int64),
        np.array([[0, 1]], dtype=np.int64),
    ],
)
def test_ann_benchmark_recall_rejects_query_identity_drift(
    tmp_path: Path,
    query_ordinals: np.ndarray,
) -> None:
    path = tmp_path / "sift-like.hdf5"
    digest = _write_ann_benchmark_fixture(path, metric="euclidean")
    dataset = SUBJECT.load_ann_benchmark(
        path,
        expected_sha256=digest,
        expected_metric="squared_l2",
    )

    with pytest.raises(ValueError, match="ANN-Benchmarks result differs"):
        SUBJECT.ann_benchmark_recall_hits(
            dataset,
            np.array([[0, 1], [3, 1]], dtype=np.int64),
            query_ordinals=query_ordinals,
            count=2,
        )


def test_candidate_containment_uses_full_candidate_width_not_truth_width() -> None:
    candidates = np.arange(1_000, dtype=np.int64).reshape(1, 1_000)
    truth = np.array([[0, 999]], dtype=np.int64)

    full = SUBJECT.candidate_containment_hits(
        candidates,
        truth,
        candidate_width=1_000,
        truth_width=2,
    )
    truncated = SUBJECT.candidate_containment_hits(
        candidates,
        truth,
        candidate_width=10,
        truth_width=2,
    )

    assert full.tolist() == [2]
    assert truncated.tolist() == [1]


@pytest.mark.parametrize(
    ("candidate_width", "truth_width", "mutation"),
    [
        (True, 2, None),
        (1_001, 2, None),
        (1_000, 0, None),
        (1_000, 3, None),
        (1_000, 2, "candidate-duplicate"),
        (1_000, 2, "truth-duplicate"),
        (1_000, 2, "negative"),
        (1_000, 2, "dtype"),
    ],
)
def test_candidate_containment_rejects_width_schema_and_identity_drift(
    candidate_width: object,
    truth_width: object,
    mutation: str | None,
) -> None:
    candidates = np.arange(1_000, dtype=np.int64).reshape(1, 1_000)
    truth = np.array([[0, 999]], dtype=np.int64)
    if mutation == "candidate-duplicate":
        candidates[0, -1] = 0
    elif mutation == "truth-duplicate":
        truth[0, -1] = 0
    elif mutation == "negative":
        candidates[0, -1] = -1
    elif mutation == "dtype":
        candidates = candidates.astype(np.int32)

    with pytest.raises(ValueError, match="candidate containment authority differs"):
        SUBJECT.candidate_containment_hits(
            candidates,
            truth,
            candidate_width=candidate_width,
            truth_width=truth_width,
        )
