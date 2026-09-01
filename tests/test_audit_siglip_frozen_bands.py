"""Tests for the local-only frozen SigLIP band-audit runner."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import stat
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from sfora.siglip_band_audit import (
    SiglipBandAuditAuthority,
    validate_siglip_band_audit_bytes,
)

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_siglip_frozen_bands.py"
_SPEC = importlib.util.spec_from_file_location("audit_siglip_frozen_bands", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def _args(result: Path) -> list[str]:
    return [
        "--result",
        str(result),
        "--source-commit",
        "1" * 40,
        "--source-tree-digest",
        "2" * 64,
        "--batch-size",
        "8",
        "--query-block",
        "32",
        "--execute-band-audit",
    ]


def _loaded() -> object:
    labels = torch.arange(98, dtype=torch.int64).repeat_interleave(2)
    descriptors = torch.zeros(196, 98, dtype=torch.float32)
    descriptors[torch.arange(196), labels] = 1.0
    class_names = tuple(f"class-{index:03d}" for index in range(196))
    authority = SiglipBandAuditAuthority(
        source_commit="1" * 40,
        source_tree_digest="2" * 64,
        dataset_revision="9abf6cf7d6dfa7b95152a0d6e791ea9435b47a40",
        dataset_examples_sha256="3" * 64,
        ordered_example_ids_sha256="4" * 64,
        descriptor_sha256="5" * 64,
        label_vector_sha256="6" * 64,
        class_names_sha256="9da9ec6333105a7a2f0d50d7a5a6afe18b1ec3ede7dd8f1df298e59eb859ce35",
        model_name="google/siglip-so400m-patch14-384",
        model_revision="9fdffc58afc957d1a03a25b10dba0329ab15c2a3",
        readout="vision_pooler_output",
        split="train",
        batch_size=8,
        query_block=32,
        cublas_workspace_config=":4096:8",
    )
    return _MODULE.LoadedBandAudit(descriptors, labels, class_names, authority)


def test_prepare_requires_cublas_workspace_before_cuda_or_dataset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.setattr(_MODULE.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        _MODULE,
        "load_image_retrieval_examples",
        lambda **_kwargs: pytest.fail("dataset loading reached before determinism authority"),
    )

    with pytest.raises(RuntimeError, match="CUBLAS_WORKSPACE_CONFIG"):
        _MODULE.prepare_band_audit(
            source_commit="1" * 40,
            source_tree_digest="2" * 64,
            batch_size=8,
            query_block=32,
        )


def test_band_audit_parser_is_explicit_and_refuses_capability_expansion(tmp_path: Path) -> None:
    parsed = _MODULE.parse_args(_args(tmp_path / "result.json"))
    assert parsed.batch_size == 8 and parsed.query_block == 32

    cases = (
        _args(tmp_path / "missing.json")[:-1],
        _args(tmp_path / "duplicate.json") + ["--query-block", "32"],
        _args(tmp_path / "unknown.json") + ["--checkpoint", "/tmp/model.pt"],
        _args(tmp_path / "upload.json") + ["--aws-profile", "default"],
        _args(tmp_path / "model.json") + ["--model-name", "other/model"],
        _args(tmp_path / "batch.json")[:-5]
        + ["--batch-size", "9", "--query-block", "32", "--execute-band-audit"],
    )
    for argv in cases:
        with pytest.raises(SystemExit):
            _MODULE.parse_args(argv)


def test_band_audit_exclusive_write_is_durable_and_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "sealed.json"
    kinds: list[str] = []
    real_fsync = _MODULE.os.fsync

    def record(file_descriptor: int) -> None:
        mode = _MODULE.os.fstat(file_descriptor).st_mode
        kinds.append("directory" if stat.S_ISDIR(mode) else "file")
        real_fsync(file_descriptor)

    monkeypatch.setattr(_MODULE.os, "fsync", record)
    _MODULE._write_exclusive(output, b"sealed\n")
    assert output.read_bytes() == b"sealed\n"
    assert kinds == ["file", "directory"]

    with pytest.raises(FileExistsError):
        _MODULE._write_exclusive(output, b"other\n")
    stale = tmp_path / "stale.json"
    stale.with_name(stale.name + ".partial").write_bytes(b"partial")
    with pytest.raises(FileExistsError):
        _MODULE._write_exclusive(stale, b"other\n")


def test_band_audit_main_publishes_reauthenticated_canonical_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    loaded = _loaded()
    monkeypatch.setattr(_MODULE, "prepare_band_audit", lambda **_kwargs: loaded)
    result = tmp_path / "result.json"

    assert _MODULE.main(_args(result)) == 0

    raw = result.read_bytes()
    value = validate_siglip_band_audit_bytes(raw, expected_authority=loaded.authority)
    stdout = json.loads(capsys.readouterr().out)
    assert value["strict_hits"] == 196
    assert stdout == {
        "result": str(result),
        "result_file_sha256": hashlib.sha256(raw).hexdigest(),
    }
    assert not tuple(tmp_path.glob("*.partial"))


def test_band_audit_main_removes_corrupt_postwrite_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = _loaded()
    monkeypatch.setattr(_MODULE, "prepare_band_audit", lambda **_kwargs: loaded)
    result = tmp_path / "corrupt.json"
    real_write = _MODULE._write_exclusive

    def corrupt(path: Path, raw: bytes) -> None:
        real_write(path, raw)
        path.write_bytes(b"short\n")

    monkeypatch.setattr(_MODULE, "_write_exclusive", corrupt)
    with pytest.raises(ValueError, match="differs after write"):
        _MODULE.main(_args(result))
    assert not result.exists()
    assert not tuple(tmp_path.glob("*.partial"))


def test_band_audit_direct_script_resolves_without_forbidden_surface() -> None:
    completed = subprocess.run(
        [sys.executable, str(_SCRIPT), "--help"],
        cwd=_SCRIPT.parents[1],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert "ModuleNotFoundError" not in completed.stderr
    assert "--execute-band-audit" in completed.stdout
    for forbidden in (
        "--checkpoint",
        "--head",
        "--official-test",
        "--aws",
        "--upload",
        "--model-name",
        "--model-revision",
    ):
        assert forbidden not in completed.stdout
