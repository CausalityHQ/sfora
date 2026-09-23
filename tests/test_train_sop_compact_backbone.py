"""The training receipt must authenticate the modules the process executes."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "train_sop_compact_backbone.py"
SPEC = importlib.util.spec_from_file_location("train_sop_compact_backbone", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
sys.path.insert(0, str(SCRIPT.parent))
SPEC.loader.exec_module(MODULE)
sys.path.pop(0)


def test_source_binding_accepts_executed_modules_and_rejects_stale_package(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    MODULE.assert_source_imports()
    trained = sys.modules["sfora.sop_compact_training"]
    monkeypatch.setattr(trained, "__file__", str(tmp_path / "stale_package.py"))
    with pytest.raises(ValueError, match="executed SOP training source differs"):
        MODULE.assert_source_imports()
