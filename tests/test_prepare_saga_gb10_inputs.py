from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest

from sfora.saga_feasibility import load_fixture_authority, load_snapshot_authority

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_saga_gb10_inputs.py"
_SPEC = importlib.util.spec_from_file_location("prepare_saga_gb10_inputs", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def test_snapshot_rows_streams_files_without_whole_file_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = (b"bounded-snapshot-chunk" * 131_073) + b"terminal"
    model_file = tmp_path / "model.safetensors"
    model_file.write_bytes(payload)

    def reject_read_bytes(_path: Path) -> bytes:
        raise AssertionError("snapshot hashing must not read a whole model file")

    monkeypatch.setattr(Path, "read_bytes", reject_read_bytes)
    monkeypatch.setattr(_MODULE, "_HASH_CHUNK_BYTES", 65_536)

    rows = _MODULE._snapshot_rows(tmp_path)

    assert len(rows) == 1
    assert rows[0].relative_path == "model.safetensors"
    assert rows[0].byte_length == len(payload)
    assert rows[0].sha256 == hashlib.sha256(payload).hexdigest()


def test_prepare_seals_immutable_snapshot_and_source_bound_fixture(
    tmp_path: Path,
) -> None:
    calls: list[dict[str, object]] = []

    def resolver(repository_id: str, **kwargs: object) -> str:
        calls.append({"repository_id": repository_id, **kwargs})
        local_dir = Path(kwargs["local_dir"])
        local_dir.mkdir(parents=True)
        (local_dir / "config.json").write_text('{"model_type":"qwen3_vl"}\n')
        (local_dir / "model.safetensors").write_bytes(b"weights")
        metadata = local_dir / ".cache" / "huggingface"
        metadata.mkdir(parents=True)
        (metadata / "download.json").write_text("{}\n")
        return str(local_dir)

    output = tmp_path / "sealed"
    result = _MODULE.prepare_inputs(
        output_root=output,
        repository_id="Qwen/Qwen3-VL-8B-Instruct",
        model_revision="1" * 40,
        source_commit="2" * 40,
        controller_commit="3" * 40,
        binary_sha256="4" * 64,
        environment_sha256="5" * 64,
        host="spark-fixture",
        snapshot_resolver=resolver,
    )

    assert calls == [
        {
            "repository_id": "Qwen/Qwen3-VL-8B-Instruct",
            "revision": "1" * 40,
            "local_dir": output.with_name("sealed.partial") / "model",
        }
    ]
    assert result == output
    assert not (output / "model" / ".cache").exists()
    assert not any(path.is_symlink() for path in output.rglob("*"))
    snapshot = load_snapshot_authority(
        root=output / "model", manifest_path=output / "snapshot.json"
    )
    fixture = load_fixture_authority(output / "fixture.json")
    assert snapshot.model_revision == "1" * 40
    assert fixture.source_commit == "2" * 40
    assert fixture.controller_commit == "3" * 40
    assert fixture.patch_tokens_per_image == 49
    assert all(path.stat().st_mode & 0o222 == 0 for path in output.rglob("*"))


def test_prepare_rejects_mutable_revision_and_existing_output(tmp_path: Path) -> None:
    def resolver(_repository_id: str, **_kwargs: object) -> str:
        raise AssertionError("resolver must not run")

    for revision in ("main", "1" * 39, "G" * 40):
        try:
            _MODULE.prepare_inputs(
                output_root=tmp_path / revision,
                repository_id="Qwen/Qwen3-VL-8B-Instruct",
                model_revision=revision,
                source_commit="2" * 40,
                controller_commit="3" * 40,
                binary_sha256="4" * 64,
                environment_sha256="5" * 64,
                host="spark-fixture",
                snapshot_resolver=resolver,
            )
        except ValueError:
            pass
        else:
            raise AssertionError("mutable revision accepted")

    existing = tmp_path / "existing"
    existing.mkdir()
    try:
        _MODULE.prepare_inputs(
            output_root=existing,
            repository_id="Qwen/Qwen3-VL-8B-Instruct",
            model_revision="1" * 40,
            source_commit="2" * 40,
            controller_commit="3" * 40,
            binary_sha256="4" * 64,
            environment_sha256="5" * 64,
            host="spark-fixture",
            snapshot_resolver=resolver,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("existing output accepted")
