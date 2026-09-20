from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

SCRIPT = Path(__file__).parents[1] / "scripts" / "_scratch_unicom_preprojection_export.py"


def _load_subject():
    spec = importlib.util.spec_from_file_location("scratch_unicom_preprojection_export", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _ToyTeacher(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.feature = torch.nn.Sequential(
            torch.nn.Linear(4, 3, bias=False),
            torch.nn.BatchNorm1d(3),
            torch.nn.Linear(3, 2, bias=False),
            torch.nn.BatchNorm1d(2),
        )

    def forward(self, rows: torch.Tensor) -> torch.Tensor:
        return self.feature(rows)


def test_capture_preprojection_is_exact_input_to_final_projection() -> None:
    subject = _load_subject()
    model = _ToyTeacher().eval()
    rows = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]], dtype=torch.float32
    )

    preprojection, final = subject.capture_preprojection_batch(
        model, rows, preprojection_dimensions=3, final_dimensions=2
    )

    assert preprojection.shape == (2, 3)
    assert final.shape == (2, 2)
    expected = model.feature[3](model.feature[2](preprojection))
    torch.testing.assert_close(final, expected, rtol=0.0, atol=0.0)


def test_capture_preprojection_rejects_noncanonical_teacher_layout() -> None:
    subject = _load_subject()
    model = _ToyTeacher().eval()
    model.feature = torch.nn.Sequential(*list(model.feature.children())[:3])

    with pytest.raises(ValueError, match="teacher feature layout differs"):
        subject.capture_preprojection_batch(
            model,
            torch.ones(2, 4),
            preprojection_dimensions=3,
            final_dimensions=2,
        )


def test_reproduction_cosines_align_interleaved_rows_by_fit_mask() -> None:
    subject = _load_subject()
    final = np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]], dtype=np.float32
    )
    fit_mask = np.asarray([True, False, True, False])
    fit_reference = final[fit_mask].copy()
    evaluation_reference = final[~fit_mask].copy()

    cosine = subject.reproduction_cosines(
        final, fit_mask, fit_reference, evaluation_reference
    )

    np.testing.assert_allclose(cosine, np.ones(4), rtol=0.0, atol=0.0)
