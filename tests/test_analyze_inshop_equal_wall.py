"""The equal-wall gate must reject a quality or cost failure."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from analyze_inshop_equal_wall import gate_pass  # noqa: E402


def test_gate_requires_all_quality_and_cost_conditions() -> None:
    assert gate_pass(0.001, -0.001, 1.01, 0.9)
    assert not gate_pass(0.0, -0.001, 1.01, 0.9)
    assert not gate_pass(0.001, -0.0021, 1.01, 0.9)
    assert not gate_pass(0.001, -0.001, 1.021, 0.9)
    assert not gate_pass(0.001, -0.001, 1.01, 1.001)
