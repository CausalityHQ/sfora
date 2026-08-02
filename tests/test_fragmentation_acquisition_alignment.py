from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS))
_SPEC = importlib.util.spec_from_file_location(
    "measure_fragmentation_acquisition_alignment",
    _SCRIPTS / "measure_fragmentation_acquisition_alignment.py",
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_parse_extracts_series_pose_and_view() -> None:
    value = "inshop-train-img/WOMEN/Dresses/id_00000002/02_1_front.jpg"
    assert _MODULE._parse(value) == ("02", "1", "front")


def test_parse_refuses_ambiguous_filename() -> None:
    with pytest.raises(ValueError, match="unparseable"):
        _MODULE._parse("id_1/front.jpg")


def test_summary_is_macro_and_size_weighted() -> None:
    result = _MODULE._summary([1.0, 0.0], [1, 3])
    assert result["macro_mean_ari"] == pytest.approx(0.5)
    assert result["class_size_weighted_mean_ari"] == pytest.approx(0.25)
