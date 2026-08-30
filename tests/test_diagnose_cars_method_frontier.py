from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "diagnose_cars_method_frontier.py"
_SPEC = importlib.util.spec_from_file_location("diagnose_cars_method_frontier", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_failure_signatures_distinguish_local_dispersion_from_centroid_overlap() -> None:
    local_embeddings = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.99, 0.10],
            [-1.0, 0.0],
        ],
        dtype=np.float64,
    )
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    local = _MODULE.classify_retrieval_failures(local_embeddings, labels, split="train")
    assert local[0] == "local_within_class_dispersion"

    overlap_embeddings = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.99, 0.10],
            [0.80, -0.20],
        ],
        dtype=np.float64,
    )
    overlap = _MODULE.classify_retrieval_failures(overlap_embeddings, labels, split="train")
    assert overlap[0] == "between_class_centroid_overlap"


def test_failure_decomposition_recomputes_counts_from_signatures() -> None:
    signatures = np.asarray(
        [
            "correct",
            "local_within_class_dispersion",
            "between_class_centroid_overlap",
            "correct",
            "between_class_centroid_overlap",
        ]
    )
    result = _MODULE.summarize_failure_signatures(signatures)
    assert result == {
        "queries": 5,
        "correct": 2,
        "failures": 3,
        "recall_at_1": 0.4,
        "local_within_class_dispersion": 1,
        "between_class_centroid_overlap": 2,
        "local_fraction_of_failures": 1.0 / 3.0,
        "between_class_fraction_of_failures": 2.0 / 3.0,
    }


def test_failure_signatures_reject_nonfinite_and_singleton_classes() -> None:
    with np.testing.assert_raises_regex(ValueError, "finite"):
        _MODULE.classify_retrieval_failures(
            np.asarray([[1.0, 0.0], [np.nan, 1.0], [0.0, 1.0]]),
            np.asarray([0, 0, 1]),
            split="train",
        )
    with np.testing.assert_raises_regex(ValueError, "at least two"):
        _MODULE.classify_retrieval_failures(
            np.eye(3, dtype=np.float64),
            np.asarray([0, 1, 1]),
            split="train",
        )


def test_failure_signatures_preserve_exported_coordinate_near_ties() -> None:
    embeddings = np.asarray(
        [
            [0.999999740485382, 2.939206875696321e-6, -7.36626053402835e-8, -3.794201970999852e-6],
            [
                0.9999998573013525,
                -3.118063378160621e-6,
                3.763879338067382e-6,
                -5.068315744825331e-6,
            ],
            [
                1.0000002267022614,
                6.328542689090506e-8,
                -1.1832455536608093e-6,
                -1.3681275262267475e-7,
            ],
            [0.9999998456849051, 4.067739915150416e-7, 1.4553019288128912e-6, 2.58763818282212e-7],
            [
                0.9999999537389478,
                -1.1386382926666143e-6,
                -1.0319240403881874e-7,
                1.830239947936335e-6,
            ],
            [
                1.0000000908603262,
                -2.3069892336120913e-6,
                -1.2268802944470123e-6,
                3.729797547454824e-7,
            ],
            [
                1.0000001256918891,
                -1.602603896057274e-6,
                -4.0959457772671215e-6,
                5.824869307477138e-7,
            ],
            [0.9999999321193099, 4.1391602154160095e-6, 2.380637514787195e-6, 8.930047013783728e-7],
        ],
        dtype=np.float64,
    )
    labels = np.repeat(np.arange(4), 2)
    result = _MODULE.summarize_failure_signatures(
        _MODULE.classify_retrieval_failures(embeddings, labels, split="train")
    )
    assert result["correct"] == 2
    assert result["recall_at_1"] == 0.25


def test_failure_signatures_use_leave_one_out_centroids_and_refuse_test_split() -> None:
    embeddings = np.asarray(
        [
            [-0.09511001269072203, 0.6078871624652467],
            [1.3566486363421149, -0.7482692071192554],
            [-1.663336778974566, 0.23217579699156704],
            [-1.3351987310415951, -0.5189801630433079],
            [-1.2472683817433068, 1.0616189018785118],
            [-0.6603853650793738, 0.8605300855958107],
        ],
        dtype=np.float64,
    )
    labels = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
    signatures = _MODULE.classify_retrieval_failures(embeddings, labels, split="train")
    assert signatures[0] == "between_class_centroid_overlap"
    with np.testing.assert_raises_regex(ValueError, "train split"):
        _MODULE.classify_retrieval_failures(embeddings, labels, split="test")


def _write_pack(
    path: Path,
    *,
    split: str,
    embeddings: np.ndarray,
    labels: np.ndarray,
    example_ids: np.ndarray,
    checkpoint_sha256: str = "a" * 64,
    report_sha256: str = "b" * 64,
) -> None:
    np.savez_compressed(
        path,
        embeddings=np.asarray(embeddings, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.int64),
        example_ids=np.asarray(example_ids),
        content_sha256=np.asarray([f"{index:064x}" for index in range(len(labels))]),
        artifact_selection=np.asarray("final_training_state"),
        split=np.asarray(split),
        checkpoint_sha256=np.asarray(checkpoint_sha256),
        report_sha256=np.asarray(report_sha256),
    )


def test_analyze_train_head_pack_applies_registered_two_to_one_condition() -> None:
    embeddings = np.asarray(
        [
            [-0.09511001269072203, 0.6078871624652467],
            [1.3566486363421149, -0.7482692071192554],
            [-1.663336778974566, 0.23217579699156704],
            [-1.3351987310415951, -0.5189801630433079],
            [-1.2472683817433068, 1.0616189018785118],
            [-0.6603853650793738, 0.8605300855958107],
        ],
        dtype=np.float32,
    )
    pack = {
        "embeddings": embeddings,
        "labels": np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64),
        "checkpoint_sha256": "a" * 64,
        "report_sha256": "b" * 64,
    }
    result = _MODULE.analyze_train_head_pack(pack)
    assert result["schema_version"] == "cars-method-frontier-train-v1"
    assert result["representation"] == "deployed_normalized_head"
    assert result["embedding_dimensions"] == 2
    assert result["failure_decomposition"]["correct"] == 2
    assert result["between_to_local_failure_ratio"] == 3.0
    assert result["registered_interference_condition_met"] is True
    assert result["authorizes_method"] is False


def test_analyze_train_head_pack_serializes_pure_between_class_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        _MODULE,
        "classify_retrieval_failures",
        lambda *_args, **_kwargs: np.asarray(["between_class_centroid_overlap"]),
    )
    pack = {
        "embeddings": np.asarray([[1.0, 0.0]], dtype=np.float32),
        "labels": np.asarray([0]),
        "checkpoint_sha256": "a" * 64,
        "report_sha256": "b" * 64,
    }

    result = _MODULE.analyze_train_head_pack(pack)

    assert result["between_to_local_failure_ratio"] is None
    assert result["registered_interference_condition_met"] is True
    assert _MODULE.canonical_json_bytes(result).endswith(b"\n")


def test_main_writes_canonical_train_only_result(tmp_path: Path) -> None:
    pack = tmp_path / "train.npz"
    output = tmp_path / "result.json"
    _write_pack(
        pack,
        split="train",
        embeddings=np.asarray(
            [
                [-0.09511001269072203, 0.6078871624652467],
                [1.3566486363421149, -0.7482692071192554],
                [-1.663336778974566, 0.23217579699156704],
                [-1.3351987310415951, -0.5189801630433079],
                [-1.2472683817433068, 1.0616189018785118],
                [-0.6603853650793738, 0.8605300855958107],
            ],
            dtype=np.float32,
        ),
        labels=np.asarray([0, 0, 0, 1, 1, 1]),
        example_ids=np.asarray(["a", "b", "c", "d", "e", "f"]),
    )
    assert _MODULE.main(["--train-head", str(pack), "--output", str(output)]) == 0
    raw = output.read_bytes()
    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")
    parsed = json.loads(raw)
    assert parsed["train_head_pack_sha256"] == _MODULE.sha256(pack)
    assert raw == _MODULE.canonical_json_bytes(parsed)
