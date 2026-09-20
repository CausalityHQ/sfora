from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

_PATH = Path(__file__).parents[1] / "scripts" / "_scratch_compact_residual_head.py"
_SPEC = importlib.util.spec_from_file_location("scratch_compact_residual_head", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
ResidualCompactHead = _MODULE.ResidualCompactHead
geometry_anchor_loss = _MODULE.geometry_anchor_loss


def test_residual_head_starts_as_exact_affine_control() -> None:
    weight = torch.arange(24, dtype=torch.float32).reshape(4, 6) / 24.0
    bias = torch.linspace(-0.2, 0.2, 4)
    head = ResidualCompactHead(weight=weight, bias=bias, hidden_dimensions=3, seed=17)
    rows = torch.arange(30, dtype=torch.float32).reshape(5, 6) / 30.0

    expected = torch.nn.functional.normalize(torch.nn.functional.linear(rows, weight, bias), dim=1)
    actual = head(rows)

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    assert head.residual_parameter_count == (6 * 3) + (3 * 4)


def test_geometry_anchor_is_zero_only_for_the_frozen_initial_direction() -> None:
    initial = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    same = initial.clone()
    changed = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float32)

    assert geometry_anchor_loss(same, initial).item() == 0.0
    assert geometry_anchor_loss(changed, initial).item() == 1.0
