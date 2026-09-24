"""The training receipt must authenticate the modules the process executes."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

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


def test_full_width_holdout_records_unicom_prefix_rule_separately() -> None:
    values = torch.zeros((4, 768), dtype=torch.float32)
    values[:, 0] = torch.tensor([1.0, 1.0, -1.0, -1.0])
    values[:, 512] = torch.tensor([10.0, -10.0, 10.0, -10.0])
    values = torch.nn.functional.normalize(values, dim=1)
    labels = (0, 0, 1, 1)

    result = MODULE.score_validation_features(values, labels)

    assert result["float"]["recall_at_1"] == 0.0
    assert result["upstream_prefix512_euclidean"]["recall_at_1"] == 1.0
    assert set(result) == {"float", "packed", "upstream_prefix512_euclidean"}


def test_compact_holdout_keeps_existing_metric_inventory() -> None:
    values = torch.tensor([[1.0, 0.0], [0.9, 0.1], [-1.0, 0.0], [-0.9, -0.1]])
    values = torch.nn.functional.normalize(torch.nn.functional.pad(values, (0, 126)), dim=1)

    result = MODULE.score_validation_features(values, (0, 0, 1, 1))

    assert set(result) == {"float", "packed"}
