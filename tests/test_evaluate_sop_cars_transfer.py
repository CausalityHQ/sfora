"""Cars transfer uses the standard class split across both source partitions."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "evaluate_sop_cars_transfer.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("evaluate_sop_cars_transfer", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_cars_evaluation_uses_last_98_classes_from_both_image_partitions() -> None:
    selected = MODULE.select_evaluation_rows((0, 97, 98, 195), (100, 96, 99), expected_count=4)
    assert selected == (("train", 2, 98), ("train", 3, 195), ("test", 0, 100), ("test", 2, 99))
    with pytest.raises(ValueError, match="Cars transfer split"):
        MODULE.select_evaluation_rows((98, 196), (99,), expected_count=3)
    with pytest.raises(ValueError, match="Cars transfer split"):
        MODULE.select_evaluation_rows((98,), (99,), expected_count=3)
