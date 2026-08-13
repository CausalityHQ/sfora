from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

_SCRIPT = Path(__file__).parents[1] / "scripts/evaluate_oml_interpolation.py"
sys.path.insert(0, str(_SCRIPT.parent))
_SPEC = importlib.util.spec_from_file_location("evaluate_oml_interpolation", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_interpolate_state_dict_blends_floats_and_preserves_exact_buffers() -> None:
    base = {"weight": torch.tensor([1.0, 3.0]), "count": torch.tensor(2)}
    trained = {"weight": torch.tensor([5.0, 7.0]), "count": torch.tensor(2)}

    result = _MODULE.interpolate_state_dict(base, trained, alpha=0.25)

    torch.testing.assert_close(result["weight"], torch.tensor([2.0, 4.0]))
    assert result["count"].item() == 2


def test_interpolate_state_dict_rejects_nonfloat_buffer_drift() -> None:
    try:
        _MODULE.interpolate_state_dict(
            {"count": torch.tensor(2)}, {"count": torch.tensor(3)}, alpha=0.5
        )
    except ValueError as error:
        assert "non-floating state differs" in str(error)
    else:
        raise AssertionError("non-floating drift was accepted")
