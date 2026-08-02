from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS))
_SPEC = importlib.util.spec_from_file_location(
    "measure_fragmentation_series_confounding",
    _SCRIPTS / "measure_fragmentation_series_confounding.py",
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_adjust_uses_only_cells_with_both_exposures() -> None:
    result = _MODULE._adjust(
        ["matched", "matched", "only-fragmented"],
        [False, True, True],
        [0.4, 0.7, 1.0],
    )
    assert result["matched_cells"] == 1
    assert result["effective_matched_weight"] == 1
    assert result["retained_classes"] == 2
    assert result["adjusted_fragmented_minus_connected_top1_points"] == pytest.approx(30.0)


def test_adjust_weights_by_smaller_arm_count() -> None:
    result = _MODULE._adjust(
        ["a", "a", "b", "b", "b"],
        [False, True, False, True, True],
        [0.0, 1.0, 0.0, 0.0, 0.0],
    )
    assert result["effective_matched_weight"] == 2
    assert result["adjusted_fragmented_minus_connected_top1_points"] == pytest.approx(50.0)
