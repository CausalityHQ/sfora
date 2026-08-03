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
    checkpoint = tmp_path / "model.pt"
    report = tmp_path / "report.json"
    checkpoint.write_bytes(b"weights")
    report.write_text("{}", encoding="utf-8")
    train = tmp_path / "train.npz"
    test = tmp_path / "test.npz"
    _write_pack(
        train,
        split="train",
        labels=[1, 2],
        ids=["a", "b"],
        paths=["/data/a.jpg", "/data/b.jpg"],
        checkpoint_hash=_hash(checkpoint),
        report_hash=_hash(report),
    )
    _write_pack(
        test,
        split="test",
        labels=[3, 4],
        ids=["c", "d"],
        paths=["/data/c.jpg", "/data/d.jpg"],
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
    checkpoint = tmp_path / "model.pt"
    report = tmp_path / "report.json"
    checkpoint.write_bytes(b"weights")
    report.write_text("{}", encoding="utf-8")
    kwargs = {"checkpoint_hash": _hash(checkpoint), "report_hash": _hash(report)}
    train = tmp_path / "train.npz"
    test = tmp_path / "test.npz"
    _write_pack(train, split="train", labels=[1, 2], ids=["a", "b"], paths=["a", "b"], **kwargs)
    _write_pack(test, split="test", labels=[2, 3], ids=["c", "d"], paths=["c", "d"], **kwargs)
    with pytest.raises(ValueError, match="class labels overlap"):
        verifier.verify(train, test, checkpoint, report)


def test_verify_rejects_hash_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(verifier, "EXPECTED", {"train": (2, 2), "test": (2, 2)})
    checkpoint = tmp_path / "model.pt"
    report = tmp_path / "report.json"
    checkpoint.write_bytes(b"weights")
    report.write_text("{}", encoding="utf-8")
    train = tmp_path / "train.npz"
    test = tmp_path / "test.npz"
    kwargs = {"checkpoint_hash": "wrong", "report_hash": _hash(report)}
    _write_pack(train, split="train", labels=[1, 2], ids=["a", "b"], paths=["a", "b"], **kwargs)
    _write_pack(test, split="test", labels=[3, 4], ids=["c", "d"], paths=["c", "d"], **kwargs)
    with pytest.raises(ValueError, match="checkpoint hash"):
        verifier.verify(train, test, checkpoint, report)
