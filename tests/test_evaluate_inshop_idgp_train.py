from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "evaluate_inshop_idgp_train.py"
SPEC = importlib.util.spec_from_file_location("evaluate_inshop_idgp_train", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
CLI = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CLI)


def _fold(label: int) -> int:
    encoded = np.asarray(label, dtype="<i8").tobytes()
    return hashlib.sha256(b"LE-IDGP-fold-v1:" + encoded).digest()[0] & 3


def _write_train_archive(path: Path, *, include_ineligible: bool = False) -> None:
    labels_for_fold: list[int] = []
    label = 0
    while len(labels_for_fold) < 90:
        if _fold(label) == 1:
            labels_for_fold.append(label)
        label += 1
    rng = np.random.default_rng(44)
    bases = rng.normal(size=(90, 64))
    bases /= np.linalg.norm(bases, axis=1, keepdims=True)
    rows: list[np.ndarray] = []
    labels: list[int] = []
    ids: list[str] = []
    for position, class_label in enumerate(labels_for_fold):
        for sample in range(4):
            row = bases[position] + 0.03 * rng.normal(size=64)
            row /= np.linalg.norm(row)
            rows.append(row.astype(np.float32))
            labels.append(class_label)
            ids.append(f"example-{class_label}-{sample}")
    if include_ineligible:
        singleton = rng.normal(size=64)
        singleton /= np.linalg.norm(singleton)
        rows.append(singleton.astype(np.float32))
        labels.append(1_000_000)
        ids.append("ineligible-singleton")
    np.savez(
        path,
        embeddings=np.asarray(rows, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.int64),
        example_ids=np.asarray(ids),
        source_paths=np.asarray([f"source-{value}" for value in ids]),
        artifact_selection=np.asarray("final_training_state"),
        split=np.asarray("train"),
        checkpoint_sha256=np.asarray("a" * 64),
        report_sha256=np.asarray("b" * 64),
    )


def _set_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("MKL_NUM_THREADS", "1")
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "1")


def test_build_report_is_train_only_and_relationally_validates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_environment(monkeypatch)
    path = tmp_path / "train.npz"
    _write_train_archive(path, include_ineligible=True)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(CLI, "EXPECTED_TRAIN_SHA256", digest)
    monkeypatch.setattr(CLI, "DENSITY_POOL_SIZE", 180)
    report = CLI.build_report(path)
    assert CLI.validate_report(report) is report
    assert report["input"]["sha256"] == digest
    assert report["decision"]["status"] in {"PASS", "KILL"}
    assert report["input"]["rows"] == 361
    assert report["configuration"]["cohort_rows"] == 180
    assert report["configuration"]["density_pool_size"] == 180
    assert report["density_pools"]["primary_rows"] == 180
    assert report["density_pools"]["alternate_rows"] == 180
    assert len(report["density_pools"]["primary_sha256"]) == 64
    assert len(report["density_pools"]["alternate_sha256"]) == 64
    assert report["cohorts"][0]["specificity_records"]

    mutated = copy.deepcopy(report)
    mutated["pooled"]["metrics"]["raw_advantage_mean"] += 0.01
    with pytest.raises(ValueError, match="pooled"):
        CLI.validate_report(mutated)
    mutated = copy.deepcopy(report)
    mutated["decision"]["status"] = "PASS" if report["decision"]["status"] == "KILL" else "KILL"
    with pytest.raises(ValueError, match="decision"):
        CLI.validate_report(mutated)

    mutated = copy.deepcopy(report)
    mutated["diagnostics"]["local_norm_mean"] += 0.1
    with pytest.raises(ValueError, match="diagnostics"):
        CLI.validate_report(mutated)
    mutated = copy.deepcopy(report)
    mutated["cohorts"][0]["unexpected"] = 1
    with pytest.raises(ValueError, match="cohort"):
        CLI.validate_report(mutated)
    mutated = copy.deepcopy(report)
    mutated["cohorts"][0]["diagnostics"]["reference_cosine_mean"] += 0.1
    with pytest.raises(ValueError, match="diagnostics"):
        CLI.validate_report(mutated)
    mutated = copy.deepcopy(report)
    mutated["density_pools"]["primary_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="density pools"):
        CLI.validate_report(mutated)
    mutated = copy.deepcopy(report)
    mutated["cohorts"][0]["specificity_records"][0]["two_sided"] += 0.1
    with pytest.raises(ValueError, match="pooled"):
        CLI.validate_report(mutated)


def test_loader_rejects_schema_symlink_and_nonfinite_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "train.npz"
    _write_train_archive(path)
    monkeypatch.setattr(CLI, "EXPECTED_TRAIN_SHA256", hashlib.sha256(path.read_bytes()).hexdigest())
    link = tmp_path / "link.npz"
    link.symlink_to(path)
    with pytest.raises(ValueError, match="regular"):
        CLI.load_train_archive(link)

    bad = tmp_path / "bad.npz"
    np.savez(bad, labels=np.asarray([0], dtype=np.int64), embeddings=np.asarray([[np.nan]]))
    monkeypatch.setattr(CLI, "EXPECTED_TRAIN_SHA256", hashlib.sha256(bad.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match="schema"):
        CLI.load_train_archive(bad)


def test_atomic_writer_strictly_reloads_and_rolls_back_owned_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "result.json"
    monkeypatch.setattr(
        CLI,
        "validate_report",
        lambda _value: (_ for _ in ()).throw(ValueError("invalid persisted report")),
    )
    with pytest.raises(ValueError, match="invalid persisted"):
        CLI.write_report(destination, {"x": 1})
    assert not destination.exists()
    assert not list(tmp_path.glob(".*.tmp.*"))

    destination.write_text("sentinel")
    with pytest.raises(FileExistsError):
        CLI.write_report(destination, {"x": 1})
    assert destination.read_text() == "sentinel"


def test_main_accepts_only_train_and_output_arguments() -> None:
    parser = CLI.build_parser()
    parsed = parser.parse_args(["--train", "train.npz", "--output", "result.json"])
    assert parsed.train == Path("train.npz")
    assert parsed.output == Path("result.json")
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--train",
                "train.npz",
                "--query",
                "query.npz",
                "--output",
                "result.json",
            ]
        )


def test_written_report_is_canonical_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    destination = tmp_path / "result.json"
    report = {"x": 1}
    monkeypatch.setattr(CLI, "validate_report", lambda value: value)
    CLI.write_report(destination, report)
    expected = json.dumps(report, ensure_ascii=False, separators=(",", ":")) + "\n"
    assert destination.read_text() == expected
    assert oct(os.stat(destination).st_mode & 0o777) == "0o600"
