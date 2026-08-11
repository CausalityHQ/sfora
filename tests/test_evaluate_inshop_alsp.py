from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).parents[1] / "scripts/evaluate_inshop_alsp.py"
SPEC = importlib.util.spec_from_file_location("evaluate_inshop_alsp", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _unit(rows: list[list[float]]) -> np.ndarray:
    values = np.asarray(rows, dtype=np.float32)
    return values / np.linalg.norm(values, axis=1, keepdims=True)


def _write_bundle(
    path: Path,
    embeddings: np.ndarray,
    labels: np.ndarray,
    *,
    prefix: str,
    split: str,
    checkpoint: str = "a" * 64,
) -> None:
    np.savez(
        path,
        embeddings=np.asarray(embeddings, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.int64),
        example_ids=np.asarray(
            [f"{prefix}-{index}" for index in range(len(labels))], dtype=np.str_
        ),
        split=np.asarray(split),
        checkpoint_sha256=np.asarray(checkpoint),
    )


def test_loader_requires_exact_arrays_split_and_checkpoint(tmp_path: Path) -> None:
    valid = tmp_path / "valid.npz"
    _write_bundle(
        valid,
        _unit([[1, 0], [0, 1]]),
        np.asarray([1, 2]),
        prefix="v",
        split="train",
    )
    bundle = MODULE.load_embedding_bundle(valid, expected_split="train")
    assert bundle.example_ids.tolist() == ["v-0", "v-1"]
    assert bundle.checkpoint_sha256 == "a" * 64

    with pytest.raises(ValueError, match="split"):
        MODULE.load_embedding_bundle(valid, expected_split="query")

    missing_checkpoint = tmp_path / "missing.npz"
    np.savez(
        missing_checkpoint,
        embeddings=_unit([[1, 0], [0, 1]]),
        labels=np.asarray([1, 2], dtype=np.int64),
        example_ids=np.asarray(["a", "b"]),
        split=np.asarray("train"),
    )
    with pytest.raises(ValueError, match="checkpoint_sha256"):
        MODULE.load_embedding_bundle(missing_checkpoint, expected_split="train")


def test_atomic_writer_roundtrips_validated_json_and_never_clobbers(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "result.json"
    payload = {"schema_version": "test", "value": 3}
    MODULE.write_json_atomic(destination, payload, validator=lambda value: value)
    assert json.loads(destination.read_text()) == payload
    before = destination.read_bytes()
    with pytest.raises(FileExistsError):
        MODULE.write_json_atomic(
            destination, {"schema_version": "other"}, validator=lambda value: value
        )
    assert destination.read_bytes() == before
    assert list(tmp_path.glob(".result.json.*.tmp")) == []


def test_atomic_writer_rolls_back_owned_publication_on_reload_failure(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "invalid.json"

    calls = 0

    def reject(_value: object) -> object:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("persisted payload rejected")
        return _value

    with pytest.raises(ValueError, match="persisted payload rejected"):
        MODULE.write_json_atomic(destination, {"value": 3}, validator=reject)
    assert not destination.exists()
    assert list(tmp_path.glob(".invalid.json.*.tmp")) == []


def test_build_report_runs_train_only_pipeline_and_validates_exact_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        monkeypatch.setenv(name, "1")
    train_path = tmp_path / "train.npz"
    query_path = tmp_path / "query.npz"
    gallery_path = tmp_path / "gallery.npz"

    train_labels = np.repeat(np.arange(20, dtype=np.int64), 3)
    train_angles = (
        np.repeat(np.linspace(0.0, 5.7, 20, dtype=np.float32), 3)
        + np.tile(np.linspace(-0.03, 0.03, 3, dtype=np.float32), 20)
    )
    train_embeddings = np.stack(
        (np.cos(train_angles), np.sin(train_angles)), axis=1
    ).astype(np.float32)
    _write_bundle(
        train_path,
        train_embeddings,
        train_labels,
        prefix="train",
        split="train",
    )

    gallery_angles = np.linspace(0.0, 5.9, 60, dtype=np.float32)
    gallery_embeddings = np.stack(
        (np.cos(gallery_angles), np.sin(gallery_angles)), axis=1
    ).astype(np.float32)
    gallery_labels = np.arange(60, dtype=np.int64)
    query_embeddings = _unit(
        [[float(np.cos(angle + 0.01)), float(np.sin(angle + 0.01))]
         for angle in gallery_angles]
    )
    _write_bundle(
        query_path,
        query_embeddings,
        gallery_labels,
        prefix="query",
        split="query",
    )
    _write_bundle(
        gallery_path,
        gallery_embeddings,
        gallery_labels,
        prefix="gallery",
        split="gallery",
    )

    report = MODULE.build_alsp_report(
        train_path, query_path, gallery_path, block_size=17
    )
    assert MODULE.validate_alsp_report(report) == report
    assert list(report) == [
        "schema_version",
        "inputs",
        "configuration",
        "split",
        "selection",
        "fit",
        "test",
        "decision",
    ]
    assert report["configuration"]["k"] == 50
    assert report["configuration"]["numpy_version"] == str(np.__version__)
    assert report["configuration"]["thread_environment"] == {
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    }
    assert report["fit"]["weights_sha256"] == MODULE.float64_sha256(
        np.asarray(report["fit"]["weights"], dtype=np.float64)
    )
    assert report["test"]["arms"]["constant"] == report["test"]["arms"]["raw"]
    assert len(report["test"]["assignment_null"]["gains"]) == 20
    assert list(report["test"]["scale_sensitivity"]["arms"]) == ["0.5", "1.0", "2.0"]
    assert list(report["test"]["scale_sensitivity"]["calibration"]) == [
        "slope",
        "intercept",
    ]

    mutation = copy.deepcopy(report)
    mutation["decision"]["passes_falsifier"] = not report["decision"][
        "passes_falsifier"
    ]
    with pytest.raises(ValueError, match="decision"):
        MODULE.validate_alsp_report(mutation)

    mutation = copy.deepcopy(report)
    mutation["test"]["arms"]["alsp"]["p_value"] = 0.987654321
    mutation["test"]["scale_sensitivity"]["arms"]["1.0"]["p_value"] = 0.987654321
    with pytest.raises(ValueError, match="p_value"):
        MODULE.validate_alsp_report(mutation)

    mutation = copy.deepcopy(report)
    mutation["test"]["arms"]["alsp"]["raw_recall"] -= 0.01
    mutation["test"]["arms"]["alsp"]["gain"] = (
        mutation["test"]["arms"]["alsp"]["corrected_recall"]
        - mutation["test"]["arms"]["alsp"]["raw_recall"]
    )
    mutation["test"]["scale_sensitivity"]["arms"]["1.0"] = copy.deepcopy(
        mutation["test"]["arms"]["alsp"]
    )
    with pytest.raises(ValueError, match="raw_recall"):
        MODULE.validate_alsp_report(mutation)

    mutation = copy.deepcopy(report)
    mutation["inputs"]["train"]["sha256"] = "z" * 64
    with pytest.raises(ValueError, match="sha256"):
        MODULE.validate_alsp_report(mutation)

    mutation = copy.deepcopy(report)
    mutation["configuration"]["block_size"] = False
    with pytest.raises(ValueError, match="configuration"):
        MODULE.validate_alsp_report(mutation)

    rotated = np.roll(gallery_embeddings, 7, axis=0)
    _write_bundle(
        gallery_path,
        rotated,
        gallery_labels,
        prefix="different-gallery",
        split="gallery",
    )
    _write_bundle(
        query_path,
        np.roll(query_embeddings, 11, axis=0),
        gallery_labels,
        prefix="different-query",
        split="query",
    )
    changed_test_report = MODULE.build_alsp_report(
        train_path, query_path, gallery_path, block_size=17
    )
    assert changed_test_report["split"] == report["split"]
    assert changed_test_report["selection"] == report["selection"]
    assert changed_test_report["fit"] == report["fit"]
