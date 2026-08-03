from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "verify_sop_final_artifacts", Path("scripts/verify_sop_final_artifacts.py")
)
assert _SPEC is not None and _SPEC.loader is not None
verifier = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(verifier)

_NO_DUPLICATES = {
    "duplicate_groups": 0,
    "duplicate_rows": 0,
    "cross_label_groups": 0,
    "cross_label_rows": 0,
}


def _expect_small_clean_content(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        verifier,
        "EXPECTED_CONTENT_PROFILE",
        {"train": _NO_DUPLICATES, "test": _NO_DUPLICATES},
    )
    monkeypatch.setattr(
        verifier,
        "EXPECTED_CROSS_SPLIT_CONTENT",
        {"groups": 0, "rows": 0, "labels": 0},
    )


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_pack(
    path: Path,
    *,
    split: str,
    labels: list[int],
    ids: list[str],
    paths: list[str],
    checkpoint_hash: str,
    report_hash: str,
) -> None:
    embeddings = np.eye(len(labels), 3, dtype=np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    np.savez_compressed(
        path,
        embeddings=embeddings,
        labels=np.asarray(labels),
        example_ids=np.asarray(ids),
        source_paths=np.asarray(paths),
        artifact_selection=np.asarray("final_training_state"),
        split=np.asarray(split),
        checkpoint_sha256=np.asarray(checkpoint_hash),
        report_sha256=np.asarray(report_hash),
    )


def test_verify_binds_disjoint_packs_to_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(verifier, "EXPECTED", {"train": (2, 2), "test": (2, 2)})
    _expect_small_clean_content(monkeypatch)
    checkpoint = tmp_path / "model.pt"
    report = tmp_path / "report.json"
    checkpoint.write_bytes(b"weights")
    report.write_text("{}", encoding="utf-8")
    train = tmp_path / "train.npz"
    test = tmp_path / "test.npz"
    sources = [tmp_path / name for name in ("a.jpg", "b.jpg", "c.jpg", "d.jpg")]
    for index, source in enumerate(sources):
        source.write_bytes(f"image-{index}".encode())
    _write_pack(
        train,
        split="train",
        labels=[1, 2],
        ids=["a", "b"],
        paths=[str(sources[0]), str(sources[1])],
        checkpoint_hash=_hash(checkpoint),
        report_hash=_hash(report),
    )
    _write_pack(
        test,
        split="test",
        labels=[3, 4],
        ids=["c", "d"],
        paths=[str(sources[2]), str(sources[3])],
        checkpoint_hash=_hash(checkpoint),
        report_hash=_hash(report),
    )
    result = verifier.verify(train, test, checkpoint, report)
    assert result["status"] == "verified"
    assert result["train_test_label_overlap"] == 0


def test_verify_rejects_cross_split_overlap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(verifier, "EXPECTED", {"train": (2, 2), "test": (2, 2)})
    _expect_small_clean_content(monkeypatch)
    checkpoint = tmp_path / "model.pt"
    report = tmp_path / "report.json"
    checkpoint.write_bytes(b"weights")
    report.write_text("{}", encoding="utf-8")
    kwargs = {"checkpoint_hash": _hash(checkpoint), "report_hash": _hash(report)}
    train = tmp_path / "train.npz"
    test = tmp_path / "test.npz"
    paths = [tmp_path / name for name in ("a", "b", "c", "d")]
    for index, path in enumerate(paths):
        path.write_bytes(f"image-{index}".encode())
    _write_pack(
        train,
        split="train",
        labels=[1, 2],
        ids=["a", "b"],
        paths=[str(paths[0]), str(paths[1])],
        **kwargs,
    )
    _write_pack(
        test,
        split="test",
        labels=[2, 3],
        ids=["c", "d"],
        paths=[str(paths[2]), str(paths[3])],
        **kwargs,
    )
    with pytest.raises(ValueError, match="class labels overlap"):
        verifier.verify(train, test, checkpoint, report)


def test_verify_rejects_hash_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(verifier, "EXPECTED", {"train": (2, 2), "test": (2, 2)})
    _expect_small_clean_content(monkeypatch)
    checkpoint = tmp_path / "model.pt"
    report = tmp_path / "report.json"
    checkpoint.write_bytes(b"weights")
    report.write_text("{}", encoding="utf-8")
    train = tmp_path / "train.npz"
    test = tmp_path / "test.npz"
    kwargs = {"checkpoint_hash": "wrong", "report_hash": _hash(report)}
    paths = [tmp_path / name for name in ("a", "b", "c", "d")]
    for index, path in enumerate(paths):
        path.write_bytes(f"image-{index}".encode())
    _write_pack(
        train,
        split="train",
        labels=[1, 2],
        ids=["a", "b"],
        paths=[str(paths[0]), str(paths[1])],
        **kwargs,
    )
    _write_pack(
        test,
        split="test",
        labels=[3, 4],
        ids=["c", "d"],
        paths=[str(paths[2]), str(paths[3])],
        **kwargs,
    )
    with pytest.raises(ValueError, match="checkpoint hash"):
        verifier.verify(train, test, checkpoint, report)


def test_verify_rejects_unexpected_cross_split_content_overlap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(verifier, "EXPECTED", {"train": (2, 2), "test": (2, 2)})
    _expect_small_clean_content(monkeypatch)
    checkpoint = tmp_path / "model.pt"
    report = tmp_path / "report.json"
    checkpoint.write_bytes(b"weights")
    report.write_text("{}", encoding="utf-8")
    paths = [tmp_path / name for name in ("a", "b", "c", "d")]
    for path, payload in zip(paths, (b"shared", b"train", b"shared", b"test"), strict=True):
        path.write_bytes(payload)
    kwargs = {"checkpoint_hash": _hash(checkpoint), "report_hash": _hash(report)}
    train = tmp_path / "train.npz"
    test = tmp_path / "test.npz"
    _write_pack(
        train,
        split="train",
        labels=[1, 2],
        ids=["a", "b"],
        paths=[str(paths[0]), str(paths[1])],
        **kwargs,
    )
    _write_pack(
        test,
        split="test",
        labels=[3, 4],
        ids=["c", "d"],
        paths=[str(paths[2]), str(paths[3])],
        **kwargs,
    )
    with pytest.raises(ValueError, match="source-content overlap"):
        verifier.verify(train, test, checkpoint, report)
