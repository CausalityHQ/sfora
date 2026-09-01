"""Tests for the local-only SSOR diagnostic boundary."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import stat
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from sfora.siglip_ssor import (
    SSOR_BETA_GRID,
    SSORDiagnosticEvidence,
    SSORInnerFoldEvidence,
    SSOROuterFoldEvidence,
)

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "diagnose_siglip_ssor.py"
_SPEC = importlib.util.spec_from_file_location("diagnose_siglip_ssor", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def _valid_args() -> list[str]:
    return [
        "--feature-manifest",
        "/cache/ssor.json",
        "--feature-manifest-sha256",
        "1" * 64,
        "--source-commit",
        "2" * 40,
        "--checkpoint-sha256",
        "3" * 64,
        "--result",
        "/result/ssor.json",
        "--deployment-head",
        "/result/ssor-head.npy",
        "--execute-ssor",
    ]


def _write_npy(path: Path, value: np.ndarray) -> str:
    with path.open("xb") as stream:
        np.save(stream, value, allow_pickle=False)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_cache(tmp_path: Path) -> Path:
    generator = torch.Generator().manual_seed(912)
    pooler = torch.randn(48, 20, generator=generator, dtype=torch.float32)
    head = torch.randn(16, 20, generator=generator, dtype=torch.float32)
    descriptors = F.normalize(pooler @ head.T, dim=1)
    labels = torch.arange(12, dtype=torch.int64).repeat_interleave(4)
    root = tmp_path / "cache"
    root.mkdir()
    files = {
        "pooler": ("pooler.npy", pooler.numpy().astype("<f4", copy=False)),
        "descriptors": (
            "descriptors.npy",
            descriptors.numpy().astype("<f4", copy=False),
        ),
        "labels": ("labels.npy", labels.numpy().astype("<i8", copy=False)),
        "head": ("head.npy", head.numpy().astype("<f4", copy=False)),
    }
    rows: dict[str, object] = {}
    for role, (filename, value) in files.items():
        rows[role] = {
            "file": filename,
            "sha256": _write_npy(root / filename, value),
            "shape": list(value.shape),
        }
    payload = {
        "schema": "sfora-siglip-ssor-cache-v1",
        "claim_eligible": False,
        "official_test_access": False,
        "role": "optimization-train",
        "source_manifest_sha256": "4" * 64,
        "source_commit": "2" * 40,
        "checkpoint_sha256": "3" * 64,
        "example_ids": [f"row-{row:04d}" for row in range(48)],
        "files": rows,
    }
    manifest = root / "ssor.json"
    manifest.write_bytes(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    )
    return manifest


def _passing_evidence(cache: object) -> SSORDiagnosticEvidence:
    projector = _MODULE.seen_class_projector(
        cache.descriptors,
        cache.labels,
        fit_labels=tuple(range(12)),
    )
    beta_hits = (10, 10, 10, 10, 12, 11)
    all_beta_hits = (15, 16, 16, 16, 18, 17)
    folds: list[SSOROuterFoldEvidence] = []
    for ordinal in range(4):
        validation_labels = tuple(range(ordinal * 3, ordinal * 3 + 3))
        fit_labels = tuple(label for label in range(12) if label not in validation_labels)
        inner_folds: list[SSORInnerFoldEvidence] = []
        for inner_ordinal in range(3):
            inner_validation = fit_labels[inner_ordinal * 3 : inner_ordinal * 3 + 3]
            inner_fit = tuple(label for label in fit_labels if label not in inner_validation)
            inner_folds.append(
                SSORInnerFoldEvidence(
                    ordinal=inner_ordinal,
                    fit_labels=inner_fit,
                    validation_labels=inner_validation,
                    projector_rank=len(inner_fit),
                    mean_complement_energy=0.25,
                    query_count=20,
                    beta_hits=beta_hits,
                )
            )
        folds.append(
            SSOROuterFoldEvidence(
                ordinal=ordinal,
                fit_labels=fit_labels,
                validation_labels=validation_labels,
                projector_rank=len(fit_labels),
                mean_complement_energy=0.25,
                selected_beta=1.5,
                query_count=30,
                identity_hits=15,
                scalar_identity_hits=15,
                ssor_hits=18,
                scalar_ssor_hits=18,
                all_beta_hits=all_beta_hits,
                inner_fold_schedule_sha256=f"{ordinal + 1:x}" * 64,
                inner_folds=tuple(inner_folds),
            )
        )
    return SSORDiagnosticEvidence(
        beta_grid=SSOR_BETA_GRID,
        fold_schedule_sha256="a" * 64,
        folds=tuple(folds),
        selected_betas=(1.5, 1.5, 1.5, 1.5),
        deployment_beta=1.5,
        consensus_count=4,
        deployment_projector_rank=projector.rank,
        deployment_mean_complement_energy=projector.mean_complement_energy,
        query_count=120,
        identity_hits=60,
        identity_errors=60,
        materiality_eligible=True,
        ssor_hits=72,
        identity_recall_ppm=500_000,
        ssor_recall_ppm=600_000,
        delta_ppm=100_000,
        fold_wins=4,
        minimum_fold_delta_ppm=100_000,
        valid=True,
        passed=True,
    )


def _null_evidence(cache: object) -> SSORDiagnosticEvidence:
    evidence = _passing_evidence(cache)
    folds = tuple(
        replace(
            fold,
            identity_hits=20,
            scalar_identity_hits=20,
            ssor_hits=17,
            scalar_ssor_hits=17,
            all_beta_hits=(20, 19, 19, 19, 17, 18),
        )
        for fold in evidence.folds
    )
    return replace(
        evidence,
        folds=folds,
        identity_hits=80,
        identity_errors=40,
        ssor_hits=68,
        identity_recall_ppm=666_666,
        ssor_recall_ppm=566_666,
        delta_ppm=-100_000,
        fold_wins=0,
        minimum_fold_delta_ppm=-100_000,
        passed=False,
    )


def _main_args(manifest: Path, result: Path, deployment_head: Path) -> list[str]:
    return [
        "--feature-manifest",
        str(manifest),
        "--feature-manifest-sha256",
        hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "--source-commit",
        "2" * 40,
        "--checkpoint-sha256",
        "3" * 64,
        "--result",
        str(result),
        "--deployment-head",
        str(deployment_head),
        "--execute-ssor",
    ]


def test_ssor_cli_is_explicit_local_only_and_fail_closed() -> None:
    parsed = _MODULE.parse_args(_valid_args())
    assert parsed.execute_ssor is True
    assert parsed.feature_manifest == Path("/cache/ssor.json")

    for mutation in (
        [*_valid_args(), "--clean-validation", "/clean"],
        [*_valid_args(), "--aws-profile", "causality"],
        [*_valid_args(), "--feature-manifest", "/other.json"],
    ):
        with pytest.raises(SystemExit):
            _MODULE.parse_args(mutation)


def test_ssor_loader_authenticates_cache_reconstruction_and_optimization_role(
    tmp_path: Path,
) -> None:
    manifest = _write_cache(tmp_path)
    raw = manifest.read_bytes()

    cache = _MODULE.load_ssor_cache(
        manifest,
        expected_sha256=hashlib.sha256(raw).hexdigest(),
        expected_source_commit="2" * 40,
        expected_checkpoint_sha256="3" * 64,
        expected_pooler_dimensions=20,
        expected_descriptor_dimensions=16,
        expected_class_count=12,
    )

    assert cache.pooler.shape == (48, 20)
    assert cache.descriptors.shape == (48, 16)
    assert cache.labels.tolist() == sorted(cache.labels.tolist())
    assert cache.split_authority.role == "optimization-train"
    assert cache.maximum_reconstruction_cosine_deviation <= 1e-5

    descriptor_path = manifest.parent / "descriptors.npy"
    descriptor_path.write_bytes(descriptor_path.read_bytes() + b"x")
    with pytest.raises(ValueError, match="digest"):
        _MODULE.load_ssor_cache(
            manifest,
            expected_sha256=hashlib.sha256(raw).hexdigest(),
            expected_source_commit="2" * 40,
            expected_checkpoint_sha256="3" * 64,
            expected_pooler_dimensions=20,
            expected_descriptor_dimensions=16,
            expected_class_count=12,
        )


def test_ssor_loader_rejects_manifest_semantic_drift_after_rehash(tmp_path: Path) -> None:
    manifest = _write_cache(tmp_path)
    value = json.loads(manifest.read_bytes())
    value["official_test_access"] = True
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    manifest.write_bytes(raw)

    with pytest.raises(ValueError, match="manifest authority"):
        _MODULE.load_ssor_cache(
            manifest,
            expected_sha256=hashlib.sha256(raw).hexdigest(),
            expected_source_commit="2" * 40,
            expected_checkpoint_sha256="3" * 64,
            expected_pooler_dimensions=20,
            expected_descriptor_dimensions=16,
            expected_class_count=12,
        )


def test_ssor_paths_refuse_symlinks_overwrites_and_stale_partials(tmp_path: Path) -> None:
    manifest = _write_cache(tmp_path)
    manifest_link = tmp_path / "manifest-link.json"
    manifest_link.symlink_to(manifest)
    with pytest.raises(ValueError, match="manifest path"):
        _MODULE.load_ssor_cache(
            manifest_link,
            expected_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
            expected_source_commit="2" * 40,
            expected_checkpoint_sha256="3" * 64,
            expected_pooler_dimensions=20,
            expected_descriptor_dimensions=16,
            expected_class_count=12,
        )

    output = tmp_path / "output.json"
    _MODULE._write_exclusive(output, b"first\n")
    with pytest.raises(FileExistsError):
        _MODULE._write_exclusive(output, b"second\n")

    blocked = tmp_path / "blocked.json"
    blocked.with_name("blocked.json.partial").write_bytes(b"stale")
    with pytest.raises(FileExistsError):
        _MODULE._write_exclusive(blocked, b"value\n")

    symlink = tmp_path / "symlink.json"
    symlink.symlink_to(output)
    with pytest.raises(FileExistsError):
        _MODULE._write_exclusive(symlink, b"value\n")


def test_ssor_exclusive_write_fsyncs_file_and_parent_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    synced_kinds: list[str] = []
    real_fsync = _MODULE.os.fsync

    def record_fsync(file_descriptor: int) -> None:
        mode = _MODULE.os.fstat(file_descriptor).st_mode
        synced_kinds.append("directory" if stat.S_ISDIR(mode) else "file")
        real_fsync(file_descriptor)

    monkeypatch.setattr(_MODULE.os, "fsync", record_fsync)

    _MODULE._write_exclusive(tmp_path / "durable.json", b"sealed\n")

    assert synced_kinds == ["file", "directory"]


def test_ssor_exclusive_write_removes_output_when_directory_sync_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "not-durable.json"
    real_fsync = _MODULE.os.fsync

    def fail_directory_fsync(file_descriptor: int) -> None:
        if stat.S_ISDIR(_MODULE.os.fstat(file_descriptor).st_mode):
            raise OSError("injected directory sync failure")
        real_fsync(file_descriptor)

    monkeypatch.setattr(_MODULE.os, "fsync", fail_directory_fsync)

    with pytest.raises(OSError, match="directory sync failure"):
        _MODULE._write_exclusive(output, b"sealed\n")

    assert not output.exists()
    assert not output.with_name(output.name + ".partial").exists()


def test_ssor_direct_script_resolves_local_package_without_network_surface() -> None:
    completed = subprocess.run(
        [sys.executable, str(_SCRIPT), "--help"],
        cwd=_SCRIPT.parents[1],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "ModuleNotFoundError" not in completed.stderr
    assert "--execute-ssor" in completed.stdout
    for forbidden in ("aws", "network", "official-test", "clean-validation"):
        assert forbidden not in completed.stdout.lower()


def test_ssor_main_writes_and_reauthenticates_exact_passing_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest = _write_cache(tmp_path)
    cache = _MODULE.load_ssor_cache(
        manifest,
        expected_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        expected_source_commit="2" * 40,
        expected_checkpoint_sha256="3" * 64,
        expected_pooler_dimensions=20,
        expected_descriptor_dimensions=16,
        expected_class_count=12,
    )
    evidence = _passing_evidence(cache)
    monkeypatch.setattr(_MODULE, "load_ssor_cache", lambda *_args, **_kwargs: cache)
    monkeypatch.setattr(_MODULE, "_OPTIMIZATION_CLASS_COUNT", 12)
    monkeypatch.setattr(
        _MODULE,
        "run_ssor_nested_diagnostic",
        lambda *_args, **_kwargs: evidence,
    )
    result = tmp_path / "result.json"
    deployment_head = tmp_path / "head.npy"

    assert _MODULE.main(_main_args(manifest, result, deployment_head)) == 0

    payload = json.loads(result.read_bytes())
    stdout = json.loads(capsys.readouterr().out)
    head_raw = deployment_head.read_bytes()
    assert payload["passed"] is True
    assert payload["deployment_beta"] == 1.5
    assert payload["deployment_head_file_sha256"] == hashlib.sha256(head_raw).hexdigest()
    assert stdout["deployment_head_file_sha256"] == payload["deployment_head_file_sha256"]
    assert stdout["deployment_head_tensor_sha256"] == payload["deployment_head_sha256"]
    assert stdout["result_file_sha256"] == hashlib.sha256(result.read_bytes()).hexdigest()
    assert stdout["result_file_sha256"] != payload["result_sha256"]
    assert not tuple(tmp_path.glob("*.partial"))


def test_ssor_main_records_null_without_head_and_rolls_back_head_on_result_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest = _write_cache(tmp_path)
    cache = _MODULE.load_ssor_cache(
        manifest,
        expected_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        expected_source_commit="2" * 40,
        expected_checkpoint_sha256="3" * 64,
        expected_pooler_dimensions=20,
        expected_descriptor_dimensions=16,
        expected_class_count=12,
    )
    monkeypatch.setattr(_MODULE, "load_ssor_cache", lambda *_args, **_kwargs: cache)
    monkeypatch.setattr(_MODULE, "_OPTIMIZATION_CLASS_COUNT", 12)
    evidence = _null_evidence(cache)
    monkeypatch.setattr(
        _MODULE,
        "run_ssor_nested_diagnostic",
        lambda *_args, **_kwargs: evidence,
    )
    null_result = tmp_path / "null.json"
    null_head = tmp_path / "null-head.npy"

    assert _MODULE.main(_main_args(manifest, null_result, null_head)) == 0
    assert json.loads(null_result.read_bytes())["delta_ppm"] == -100_000
    assert not null_head.exists()
    assert json.loads(capsys.readouterr().out)["deployment_head"] is None

    passing = _passing_evidence(cache)
    monkeypatch.setattr(
        _MODULE,
        "run_ssor_nested_diagnostic",
        lambda *_args, **_kwargs: passing,
    )
    failed_result = tmp_path / "failed.json"
    failed_head = tmp_path / "failed-head.npy"
    original_write = _MODULE._write_exclusive

    def fail_result(path: Path, raw: bytes) -> None:
        if path == failed_result:
            raise OSError("injected result write failure")
        original_write(path, raw)

    monkeypatch.setattr(_MODULE, "_write_exclusive", fail_result)
    with pytest.raises(OSError, match="injected result"):
        _MODULE.main(_main_args(manifest, failed_result, failed_head))
    assert not failed_result.exists()
    assert not failed_head.exists()
    assert not tuple(tmp_path.glob("*.partial"))

    blocked_result = tmp_path / "blocked.json"
    blocked_head = tmp_path / "blocked-head.npy"
    blocked_result.write_bytes(b"sealed\n")
    with pytest.raises(FileExistsError, match="output already exists"):
        _MODULE.main(_main_args(manifest, blocked_result, blocked_head))
    assert not blocked_head.exists()


def test_ssor_main_reauthenticates_result_and_removes_corrupt_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _write_cache(tmp_path)
    cache = _MODULE.load_ssor_cache(
        manifest,
        expected_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        expected_source_commit="2" * 40,
        expected_checkpoint_sha256="3" * 64,
        expected_pooler_dimensions=20,
        expected_descriptor_dimensions=16,
        expected_class_count=12,
    )
    monkeypatch.setattr(_MODULE, "load_ssor_cache", lambda *_args, **_kwargs: cache)
    monkeypatch.setattr(_MODULE, "_OPTIMIZATION_CLASS_COUNT", 12)
    monkeypatch.setattr(
        _MODULE,
        "run_ssor_nested_diagnostic",
        lambda *_args, **_kwargs: _passing_evidence(cache),
    )
    result = tmp_path / "corrupt.json"
    deployment_head = tmp_path / "corrupt-head.npy"
    real_write = _MODULE._write_exclusive

    def corrupt_result(path: Path, raw: bytes) -> None:
        real_write(path, raw)
        if path == result:
            path.write_bytes(b"short\n")

    monkeypatch.setattr(_MODULE, "_write_exclusive", corrupt_result)

    with pytest.raises(ValueError, match="result artifact differs after write"):
        _MODULE.main(_main_args(manifest, result, deployment_head))

    assert not result.exists()
    assert not deployment_head.exists()
    assert not tuple(tmp_path.glob("*.partial"))
