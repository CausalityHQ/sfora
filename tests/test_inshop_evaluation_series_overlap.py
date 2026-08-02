from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS))
_SPEC = importlib.util.spec_from_file_location(
    "measure_inshop_evaluation_series_overlap",
    _SCRIPTS / "measure_inshop_evaluation_series_overlap.py",
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_measure_classifies_same_and_cross_series_availability() -> None:
    lines = [
        "6",
        "image_name item_id evaluation_status",
        "img/C/id_a/01_1_front.jpg id_a query",
        "img/C/id_a/01_2_side.jpg id_a gallery",
        "img/C/id_a/02_1_front.jpg id_a gallery",
        "img/C/id_b/03_1_front.jpg id_b query",
        "img/C/id_b/04_1_front.jpg id_b gallery",
        "img/C/id_c/05_1_front.jpg id_c train",
    ]
    result = _MODULE.measure(lines)
    assert result["query_category_counts"] == {
        "both": 1,
        "same_only": 0,
        "cross_only": 1,
        "none": 0,
    }
    assert result["queries_with_same_series_gallery_positive_fraction"] == 0.5
    assert result["same_series_query_positive_gallery_pair_fraction"] == pytest.approx(1 / 3)


def test_measure_refuses_declared_count_mismatch() -> None:
    with pytest.raises(ValueError, match="declared"):
        _MODULE.measure(["2", "header", "img/C/id_a/01_1_front.jpg id_a train"])
