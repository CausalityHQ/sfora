import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

from probe_l14_head_lowrank import factorize_linear


def test_factorize_linear_exact_at_full_rank() -> None:
    source = nn.Linear(6, 4, bias=False)
    with torch.no_grad():
        source.weight.copy_(torch.arange(24, dtype=torch.float32).reshape(4, 6) / 7)
    compressed, captured = factorize_linear(source, 4)
    rows = torch.randn(8, 6)
    torch.testing.assert_close(compressed(rows), source(rows), atol=2e-5, rtol=2e-5)
    assert captured == pytest.approx(1.0)


def test_factorize_linear_rejects_invalid_geometry() -> None:
    with pytest.raises(ValueError, match="rank"):
        factorize_linear(nn.Linear(6, 4, bias=False), 5)
    with pytest.raises(ValueError, match="geometry"):
        factorize_linear(nn.Linear(6, 4, bias=True), 2)
