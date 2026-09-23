"""The RC4 library replay must authenticate bytes before timing."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts/profile_packed_topk_rc4_library.py"
_SPEC = importlib.util.spec_from_file_location("profile_packed_topk_rc4_library", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def test_fixture_validation_rejects_changed_consumed_bytes(tmp_path: Path) -> None:
    codes = tmp_path / "gallery_1000000_codes.bin"
    codes.write_bytes(b"original")
    manifest = {
        "files": {
            codes.name: {
                "bytes": codes.stat().st_size,
                "sha256": hashlib.sha256(codes.read_bytes()).hexdigest(),
            }
        }
    }
    payload = json.dumps(manifest).encode()
    (tmp_path / "manifest.json").write_bytes(payload)
    manifest_hash = hashlib.sha256(payload).hexdigest()

    _MODULE.verify_fixture(tmp_path, manifest_hash)
    codes.write_bytes(b"tampered")

    with pytest.raises(ValueError, match="fixture"):
        _MODULE.verify_fixture(tmp_path, manifest_hash)


def test_fixture_validation_rejects_changed_manifest(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text('{"files": {}}')

    with pytest.raises(ValueError, match="fixture"):
        _MODULE.verify_fixture(tmp_path, "0" * 64)
