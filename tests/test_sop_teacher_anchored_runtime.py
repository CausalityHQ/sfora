from __future__ import annotations

import hashlib
import importlib.util
import json
import struct
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from sfora.nested_rank_protocol import ordered_training_records_sha256


def _load_subject() -> ModuleType:
    path = Path(__file__).parents[1] / "scripts" / "sop_teacher_anchored_runtime.py"
    spec = importlib.util.spec_from_file_location("sop_teacher_anchored_runtime", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SUBJECT = _load_subject()


def test_registered_snapshot_metadata_matches_the_sealed_sop_authority() -> None:
    source = SUBJECT._REGISTERED_SNAPSHOT_METADATA["source"]
    teacher = SUBJECT._REGISTERED_SNAPSHOT_METADATA["teacher"]

    assert source["source_archive_sha256"] == (
        "6bc0d8383251685eaccd472eeda357861caffb3bfb4129f18e0124c3ddc72818"
    )
    assert teacher["source_archive_sha256"] == (
        "1ba27b2d6b9db39067aa6facd0ef8aafc303c4527f6feabed859b0512c7d921a"
    )
    assert source["train_array_sha256"]["train_embeddings"] == (
        "d88e9d35f8419c7a661bd1358c901ecb2c64d4111ecd6ec311229d0d7c76dd74"
    )
    assert teacher["train_array_sha256"]["train_embeddings"] == (
        "5a8629deee1adff92ac941a0f78f4fd55f1d4cb0db43b1459a0d6c92f84b46fc"
    )
    assert source["train_array_sha256"]["train_labels"] == (
        "d785d2eca417d91257195bf2c16d87b7805178984db82c0d0a0f5ca8546a216f"
    )
    assert (
        teacher["train_array_sha256"]["train_labels"]
        == source["train_array_sha256"]["train_labels"]
    )
    assert (
        source["model_revision"]
        == teacher["model_revision"]
        == ("d71992ed969e6c271436ac0a0ee1f3ca61474ac0")
    )


def _snapshot(
    path: Path,
    *,
    model: str,
    embeddings: np.ndarray,
    labels: np.ndarray,
    image_ids: np.ndarray,
    relative_paths: np.ndarray,
) -> str:
    arrays = {
        "train_embeddings": np.ascontiguousarray(embeddings, dtype=np.float32),
        "train_labels": np.ascontiguousarray(labels, dtype=np.int64),
        "train_image_ids": np.ascontiguousarray(image_ids, dtype=np.int64),
        "train_relative_paths": np.ascontiguousarray(relative_paths),
    }
    array_sha = {
        name: hashlib.sha256(value.tobytes(order="C")).hexdigest() for name, value in arrays.items()
    }
    metadata = {
        "schema": "sfora-nnrl-sop-train-snapshot-v1",
        "source_archive_sha256": "1" * 64 if model.endswith("B/16") else "2" * 64,
        "model_identifier": model,
        "model_revision": "3" * 40,
        "checkpoint_sha256": "4" * 64 if model.endswith("B/16") else "5" * 64,
        "embedding_dimension": embeddings.shape[1],
        "train_rows": len(labels),
        "train_classes": len(set(labels.tolist())),
        "train_array_sha256": array_sha,
        "excluded_test_array_sha256": {
            "test_embeddings": "6" * 64 if model.endswith("B/16") else "7" * 64,
            "test_labels": "8" * 64,
            "test_image_ids": "9" * 64,
            "test_relative_paths": "a" * 64,
        },
        "ordered_train_record_sha256": ordered_training_records_sha256(
            arrays["train_image_ids"],
            arrays["train_labels"],
            tuple(str(value) for value in arrays["train_relative_paths"]),
        ),
        "transform": "fixture-transform",
    }
    np.savez(
        path,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True, separators=(",", ":"))),
        **arrays,
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _paired_snapshots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch | None = None
) -> tuple[Path, str, Path, str]:
    labels = np.asarray([1, 1, 2, 2], dtype=np.int64)
    image_ids = np.asarray([11, 12, 21, 22], dtype=np.int64)
    paths = np.asarray(["train/a.jpg", "train/b.jpg", "train/c.jpg", "train/d.jpg"])
    source = tmp_path / "source.npz"
    teacher = tmp_path / "teacher.npz"
    source_sha = _snapshot(
        source,
        model="UNICOM-ViT-B/16",
        embeddings=np.arange(4 * 768, dtype=np.float32).reshape(4, 768) + 1.0,
        labels=labels,
        image_ids=image_ids,
        relative_paths=paths,
    )
    teacher_sha = _snapshot(
        teacher,
        model="UNICOM-ViT-L/14@336px",
        embeddings=np.arange(4 * 768, dtype=np.float32).reshape(4, 768) + 2.0,
        labels=labels,
        image_ids=image_ids,
        relative_paths=paths,
    )
    if monkeypatch is not None:
        registered = {}
        for role, path in (("source", source), ("teacher", teacher)):
            with np.load(path, allow_pickle=False) as archive:
                metadata = json.loads(str(archive["metadata_json"].item()))
            registered[role] = {
                key: metadata[key]
                for key in (
                    "checkpoint_sha256",
                    "excluded_test_array_sha256",
                    "model_identifier",
                    "model_revision",
                    "source_archive_sha256",
                    "train_array_sha256",
                )
            }
        monkeypatch.setattr(SUBJECT, "_REGISTERED_SNAPSHOT_METADATA", registered)
    return source, source_sha, teacher, teacher_sha


def test_load_pair_authenticates_train_only_roles_and_alignment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, source_sha, teacher, teacher_sha = _paired_snapshots(tmp_path, monkeypatch)

    pair = SUBJECT.load_authenticated_train_pair(
        source,
        source_sha,
        teacher,
        teacher_sha,
    )

    assert pair.labels.tolist() == [1, 1, 2, 2]
    assert pair.image_ids.tolist() == [11, 12, 21, 22]
    assert pair.relative_paths == (
        "train/a.jpg",
        "train/b.jpg",
        "train/c.jpg",
        "train/d.jpg",
    )
    assert pair.source_embeddings.shape == (4, 768)
    assert pair.teacher_embeddings.shape == (4, 768)


def test_load_pair_rejects_self_consistent_but_unregistered_arrays(tmp_path: Path) -> None:
    source, source_sha, teacher, teacher_sha = _paired_snapshots(tmp_path)

    with pytest.raises(ValueError, match="snapshot pair"):
        SUBJECT.load_authenticated_train_pair(source, source_sha, teacher, teacher_sha)


def test_load_pair_rejects_digest_role_and_row_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, source_sha, teacher, teacher_sha = _paired_snapshots(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="snapshot pair"):
        SUBJECT.load_authenticated_train_pair(source, "0" * 64, teacher, teacher_sha)

    labels = np.asarray([1, 1, 2, 2], dtype=np.int64)
    image_ids = np.asarray([12, 11, 21, 22], dtype=np.int64)
    paths = np.asarray(["train/a.jpg", "train/b.jpg", "train/c.jpg", "train/d.jpg"])
    teacher_sha = _snapshot(
        teacher,
        model="UNICOM-ViT-L/14@336px",
        embeddings=np.arange(4 * 768, dtype=np.float32).reshape(4, 768) + 2.0,
        labels=labels,
        image_ids=image_ids,
        relative_paths=paths,
    )
    with pytest.raises(ValueError, match="snapshot pair"):
        SUBJECT.load_authenticated_train_pair(source, source_sha, teacher, teacher_sha)


def _image_digest(root: Path, relative_paths: tuple[str, ...]) -> str:
    digest = hashlib.sha256(b"sfora-teacher-anchored-image-tree-v1\x00")
    for relative in relative_paths:
        payload = (root / relative).read_bytes()
        encoded = relative.encode("utf-8")
        digest.update(struct.pack("<Q", len(encoded)))
        digest.update(encoded)
        digest.update(struct.pack("<Q", len(payload)))
        digest.update(hashlib.sha256(payload).digest())
    return digest.hexdigest()


def test_image_binding_reads_only_ordered_train_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, source_sha, teacher, teacher_sha = _paired_snapshots(tmp_path, monkeypatch)
    pair = SUBJECT.load_authenticated_train_pair(source, source_sha, teacher, teacher_sha)
    root = tmp_path / "images"
    for index, relative in enumerate(pair.relative_paths):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"image-{index}".encode())
    forbidden = root / "test" / "never-open.jpg"
    forbidden.parent.mkdir(parents=True)
    forbidden.write_bytes(b"official-test")
    expected = _image_digest(root, pair.relative_paths)

    bound = SUBJECT.bind_authenticated_train_images(pair, root, expected)

    assert bound.relative_paths == pair.relative_paths
    assert bound.image_paths == tuple(root / value for value in pair.relative_paths)
    assert bound.sha256 == expected


def test_image_binding_rejects_path_escape_symlink_and_digest_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, source_sha, teacher, teacher_sha = _paired_snapshots(tmp_path, monkeypatch)
    pair = SUBJECT.load_authenticated_train_pair(source, source_sha, teacher, teacher_sha)
    root = tmp_path / "images"
    for relative in pair.relative_paths:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"image")
    expected = _image_digest(root, pair.relative_paths)
    (root / pair.relative_paths[0]).unlink()
    (root / pair.relative_paths[0]).symlink_to(root / pair.relative_paths[1])

    with pytest.raises(ValueError, match="image manifest"):
        SUBJECT.bind_authenticated_train_images(pair, root, expected)
    with pytest.raises(ValueError, match="image manifest"):
        SUBJECT.bind_authenticated_train_images(pair, root, "0" * 64)


def test_image_binding_rejects_symlinked_parent_before_reading_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, source_sha, teacher, teacher_sha = _paired_snapshots(tmp_path, monkeypatch)
    pair = SUBJECT.load_authenticated_train_pair(source, source_sha, teacher, teacher_sha)
    root = tmp_path / "images"
    target = root / "test"
    target.mkdir(parents=True)
    for relative in pair.relative_paths:
        (target / Path(relative).name).write_bytes(b"official-test")
    (root / "train").symlink_to(target, target_is_directory=True)
    expected = _image_digest(root, pair.relative_paths)

    with pytest.raises(ValueError, match="image manifest"):
        SUBJECT.bind_authenticated_train_images(pair, root, expected)


def test_image_binding_rejects_empty_relative_path_uniformly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, source_sha, teacher, teacher_sha = _paired_snapshots(tmp_path, monkeypatch)
    pair = SUBJECT.load_authenticated_train_pair(source, source_sha, teacher, teacher_sha)
    root = tmp_path / "images"
    root.mkdir()
    invalid = replace(pair, relative_paths=(".",))

    with pytest.raises(ValueError, match="image manifest"):
        SUBJECT.bind_authenticated_train_images(invalid, root, "0" * 64)
