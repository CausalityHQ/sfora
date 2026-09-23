"""Ordered SOP image-byte manifest for the official evaluator."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

SCRIPT = Path(__file__).parents[1] / "scripts" / "build_sop_test_image_manifest.py"
SPEC = importlib.util.spec_from_file_location("build_sop_test_image_manifest", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
sys.path.insert(0, str(SCRIPT.parent))
SPEC.loader.exec_module(MODULE)
sys.path.pop(0)


def test_manifest_records_exact_image_bytes_in_official_order(tmp_path: Path):
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    first.write_bytes(b"first-image-bytes")
    second.write_bytes(b"second-image-bytes")
    records = (
        SimpleNamespace(image_path=first),
        SimpleNamespace(image_path=second),
    )
    manifest, total_bytes = MODULE.build_manifest_bytes(records)
    assert (
        manifest
        == hashlib.sha256(first.read_bytes()).digest()
        + hashlib.sha256(second.read_bytes()).digest()
    )
    assert total_bytes == len(first.read_bytes()) + len(second.read_bytes())
    second.write_bytes(b"changed")
    changed, _ = MODULE.build_manifest_bytes(records)
    assert changed != manifest
