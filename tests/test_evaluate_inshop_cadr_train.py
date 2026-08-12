from __future__ import annotations

import copy
import hashlib
import importlib.util
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).parents[1] / "scripts" / "evaluate_inshop_cadr_train.py"
SPEC = importlib.util.spec_from_file_location("evaluate_inshop_cadr_train", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
CLI = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CLI)


def test_build_train_report_uses_train_only_and_validates(tmp_path: Path, monkeypatch) -> None:
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        monkeypatch.setenv(name, "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    rows: list[list[float]] = []
    labels: list[int] = []
    for label in range(10):
        angle = label * 0.4
        for offset in (-0.03, 0.03):
            rows.append([np.cos(angle + offset), np.sin(angle + offset)])
            labels.append(label)
    path = tmp_path / "train.npz"
    np.savez(
        path,
        embeddings=np.asarray(rows, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.int64),
        example_ids=np.asarray([f"id-{i}" for i in range(20)]),
        source_paths=np.asarray([f"p-{i}" for i in range(20)]),
        artifact_selection=np.asarray("final_training_state"),
        split=np.asarray("train"),
        checkpoint_sha256=np.asarray("a" * 64),
        report_sha256=np.asarray("b" * 64),
    )
    monkeypatch.setattr(CLI, "EXPECTED_TRAIN_SHA256", hashlib.sha256(path.read_bytes()).hexdigest())
    report = CLI.build_train_report(path, k=1, block_size=4)
    validated = CLI.validate_train_report(report)
    assert validated["decision"]["status"] in {"PASS", "KILL"}
    assert validated["input"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    mutated = copy.deepcopy(report)
    mutated["cadr"]["validation_loss"] += 0.01
    with np.testing.assert_raises(ValueError):
        CLI.validate_train_report(mutated)


def test_atomic_writer_rolls_back_only_its_publication(tmp_path: Path, monkeypatch) -> None:
    destination = tmp_path / "result.json"
    monkeypatch.setattr(
        CLI, "validate_train_report", lambda _value: (_ for _ in ()).throw(ValueError("bad"))
    )
    with np.testing.assert_raises(ValueError):
        CLI._write(destination, {"x": 1})
    assert not destination.exists()
    assert not list(tmp_path.glob(".*.tmp.*"))
