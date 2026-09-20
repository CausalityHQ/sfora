from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch

SCRIPT = Path(__file__).parents[1] / "scripts" / "_scratch_unicom_preprojection_gate.py"


def _load_subject():
    spec = importlib.util.spec_from_file_location("scratch_unicom_preprojection_gate", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fixed_width_control_is_deterministic_and_uses_no_new_rows() -> None:
    subject = _load_subject()
    rows = torch.tensor([[3.0, 4.0], [4.0, 3.0]], dtype=torch.float32)

    expanded = subject.fixed_width_control(rows, output_dimensions=4, seed=17)
    repeated = subject.fixed_width_control(rows, output_dimensions=4, seed=17)

    assert expanded.shape == (2, 4)
    torch.testing.assert_close(expanded[:, :2], rows, rtol=0.0, atol=0.0)
    torch.testing.assert_close(expanded, repeated, rtol=0.0, atol=0.0)
    assert bool(torch.isfinite(expanded).all())


def test_paired_class_bootstrap_detects_uniform_candidate_gain() -> None:
    subject = _load_subject()
    labels = np.repeat(np.arange(8, dtype=np.int64), 4)
    control = np.linspace(0.1, 0.6, len(labels), dtype=np.float64)
    candidate = control + 0.01

    interval = subject.paired_class_bootstrap(candidate, control, labels)

    assert interval["lower"] > 0.0099
    assert interval["upper"] < 0.0101
