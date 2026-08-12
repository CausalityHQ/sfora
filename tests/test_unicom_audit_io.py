from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from sfora.unicom_audit_io import (
    build_audit_report,
    load_embedding_bundle,
    publish_json_no_clobber,
    validate_audit_report,
)
from sfora.unicom_retrieval_audit import audit_deployment_geometry
from sfora.unicom_shard_audit import audit_shard_sensitivity


def _sha(values: np.ndarray) -> str:
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def _write_bundle(path: Path) -> None:
    dimension = 8
    train_labels = np.repeat(np.asarray([f"c{i}" for i in range(8)]), 4)
    train = np.zeros((32, dimension), dtype=np.float32)
    for row, label in enumerate(train_labels):
        train[row, int(label[1:])] = 1.0
        train[row] += np.float32((row % 4 + 1) * 0.001)
    query_labels = np.asarray(["c0", "c1", "c2", "c3"])
    gallery_labels = np.asarray(["c0", "c1", "c2", "c3", "c0", "c1"])
    query = np.ascontiguousarray(train[[0, 4, 8, 12]])
    gallery = np.ascontiguousarray(train[[1, 5, 9, 13, 2, 6]])
    arrays = {
        "train_embeddings": train,
        "train_labels": train_labels,
        "query_embeddings": query,
        "query_labels": query_labels,
        "gallery_embeddings": gallery,
        "gallery_labels": gallery_labels,
    }
    metadata = {
        "schema_version": 1,
        "model_identifier": "UNICOM-ViT-B/16",
        "model_revision": "a" * 40,
        "checkpoint_sha256": "b" * 64,
        "image_list_sha256": "c" * 64,
        "transform": "official-unicom-eval-v1",
        "embedding_dimension": dimension,
        "split_counts": {"train": 32, "query": 4, "gallery": 6},
        "array_sha256": {name: _sha(values) for name, values in arrays.items()},
    }
    np.savez(
        path,
        metadata_json=np.asarray(json.dumps(metadata, separators=(",", ":"))),
        **arrays,
    )


def _valid_report(tmp_path: Path) -> dict[str, object]:
    bundle_path = tmp_path / "embeddings.npz"
    _write_bundle(bundle_path)
    bundle = load_embedding_bundle(
        bundle_path, expected_counts=(32, 4, 6), expected_dimension=8
    )
    geometry = audit_deployment_geometry(
        bundle.query_embeddings,
        bundle.gallery_embeddings,
        bundle.query_labels,
        bundle.gallery_labels,
        selected=4,
        random_count=2,
        bootstrap_samples=50,
        expected_official_r1=1.0,
    )
    sharding = audit_shard_sensitivity(
        bundle.train_embeddings,
        bundle.train_labels,
        class_count=8,
        examples_per_class=2,
        selected=4,
        trials=2,
        permutations=3,
    )
    return build_audit_report(
        bundle,
        geometry,
        sharding,
        selected_coordinates=4,
        geometry_bootstrap_samples=50,
        panel_classes=8,
        examples_per_class=2,
    )


def test_load_embedding_bundle_validates_exact_keys_hashes_and_shapes(tmp_path: Path) -> None:
    path = tmp_path / "embeddings.npz"
    _write_bundle(path)

    bundle = load_embedding_bundle(path, expected_counts=(32, 4, 6), expected_dimension=8)

    assert bundle.train_embeddings.shape == (32, 8)
    assert bundle.query_labels.tolist() == ["c0", "c1", "c2", "c3"]
    assert tuple(bundle.metadata) == (
        "schema_version",
        "model_identifier",
        "model_revision",
        "checkpoint_sha256",
        "image_list_sha256",
        "transform",
        "embedding_dimension",
        "split_counts",
        "array_sha256",
    )


@pytest.mark.parametrize("mutation", ["extra_key", "wrong_hash", "nonfinite", "wrong_count"])
def test_load_embedding_bundle_rejects_structural_mutations(
    tmp_path: Path, mutation: str
) -> None:
    original = tmp_path / "original.npz"
    _write_bundle(original)
    with np.load(original, allow_pickle=False) as archive:
        values = {name: archive[name] for name in archive.files}
    metadata = json.loads(str(values["metadata_json"]))
    if mutation == "extra_key":
        values["unexpected"] = np.asarray(1)
    elif mutation == "wrong_hash":
        metadata["array_sha256"]["train_embeddings"] = "d" * 64
    elif mutation == "nonfinite":
        values["query_embeddings"] = values["query_embeddings"].copy()
        values["query_embeddings"][0, 0] = np.nan
    else:
        metadata["split_counts"]["query"] = 5
    values["metadata_json"] = np.asarray(json.dumps(metadata, separators=(",", ":")))
    mutated = tmp_path / "mutated.npz"
    np.savez(mutated, **values)

    with pytest.raises((TypeError, ValueError), match="key|hash|finite|count"):
        load_embedding_bundle(mutated, expected_counts=(32, 4, 6), expected_dimension=8)


def test_report_roundtrips_through_strict_validator(tmp_path: Path) -> None:
    report = _valid_report(tmp_path)
    encoded = json.dumps(report, allow_nan=False, separators=(",", ":"))
    persisted = json.loads(encoded)

    validate_audit_report(persisted)
    assert persisted == report
    assert persisted["geometry"]["decision"]["primary"] in {
        "EVALUATOR_REPAIR",
        "COORDINATE_NONEXCHANGEABILITY",
        "GEOMETRY_NULL",
        "REPRODUCTION_FAILED",
    }


@pytest.mark.parametrize("mutation", ["extra", "nan", "bad_decision", "bad_count"])
def test_report_validator_rejects_mutations(tmp_path: Path, mutation: str) -> None:
    report = _valid_report(tmp_path)
    if mutation == "extra":
        report["unexpected"] = True
    elif mutation == "nan":
        report["sharding"]["independent_loss_range"] = float("nan")
    elif mutation == "bad_decision":
        report["sharding"]["decision"] = "PASS"
    else:
        report["constants"]["random_masks"] += 1

    with pytest.raises((TypeError, ValueError)):
        validate_audit_report(report)


def test_publish_json_is_atomic_strict_and_no_clobber(tmp_path: Path) -> None:
    report = _valid_report(tmp_path)
    output = tmp_path / "report.json"

    publish_json_no_clobber(output, report)

    original = output.read_bytes()
    assert original.endswith(b"\n")
    assert json.loads(original) == report
    assert not list(tmp_path.glob(f".{output.name}.*.tmp"))
    with pytest.raises(FileExistsError):
        publish_json_no_clobber(output, report)
    assert output.read_bytes() == original


def test_publication_preserves_foreign_race_winner(tmp_path: Path, monkeypatch) -> None:
    report = _valid_report(tmp_path)
    output = tmp_path / "report.json"
    real_link = os.link

    def race_link(source: Path, destination: Path) -> None:
        output.write_bytes(b"foreign\n")
        real_link(source, destination)

    monkeypatch.setattr(os, "link", race_link)

    with pytest.raises(FileExistsError):
        publish_json_no_clobber(output, report)
    assert output.read_bytes() == b"foreign\n"
    assert not list(tmp_path.glob(f".{output.name}.*.tmp"))
