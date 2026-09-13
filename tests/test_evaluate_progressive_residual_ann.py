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
