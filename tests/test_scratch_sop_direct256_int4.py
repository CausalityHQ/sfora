from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_SCRIPT = Path(__file__).parents[1] / "scripts" / "_scratch_sop_direct256_int4.py"
_OFFICIAL_SCRIPT = (
    Path(__file__).parents[1] / "scripts" / "_scratch_evaluate_sop_direct256_int4_official.py"
)
sys.path.insert(0, str(_SCRIPT.parent))


def _subject():
    spec = importlib.util.spec_from_file_location("scratch_sop_direct256_int4", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_pca_affine_is_train_only_centered_and_has_requested_width() -> None:
    subject = _subject()
    rows = torch.tensor(
        [
            [3.0, 0.0, 0.0],
            [1.0, 2.0, 0.0],
            [-1.0, 0.0, 1.0],
            [-3.0, -2.0, -1.0],
        ],
        dtype=torch.float32,
    )

    weight, bias = subject.fit_pca_affine(rows, output_dimensions=2)
    projected = torch.nn.functional.linear(rows, weight, bias)

    assert weight.shape == (2, 3)
    assert bias.shape == (2,)
    torch.testing.assert_close(projected.mean(dim=0), torch.zeros(2), atol=1e-6, rtol=0.0)
    torch.testing.assert_close(weight @ weight.T, torch.eye(2), atol=1e-5, rtol=1e-5)


def test_train_fitted_int4_roundtrip_is_exactly_128_bytes_per_row() -> None:
    subject = _subject()
    fit = torch.linspace(-1.0, 1.0, 4 * 256).reshape(4, 256)
    values = torch.flip(fit, dims=(0,))

    decoded, evidence = subject.fit_pack_decode_int4(fit, values)

    assert decoded.shape == values.shape
    assert evidence["persistent_bytes_per_item"] == 128
    assert evidence["shared_scale_bytes"] == 1024
    assert np.isfinite(decoded.numpy()).all()
    with pytest.raises(ValueError, match="256 dimensions"):
        subject.fit_pack_decode_int4(fit[:, :-1], values[:, :-1])


def test_official_positioning_evaluator_requires_explicit_execution() -> None:
    result = subprocess.run(
        [sys.executable, str(_OFFICIAL_SCRIPT)], check=False, capture_output=True, text=True
    )
    assert result.returncode == 2
    assert "--execute-official-positioning" in result.stderr
