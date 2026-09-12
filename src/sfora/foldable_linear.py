"""Overparameterized linear maps with a single-affine deployment form."""

from __future__ import annotations

import math
from typing import cast

import torch
from torch import nn


class FoldableLinear(nn.Module):
    """Two linear factors that fold exactly to one affine deployment head."""

    def __init__(self, *, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        if (
            type(input_dim) is not int
            or type(hidden_dim) is not int
            or type(output_dim) is not int
            or min(input_dim, hidden_dim, output_dim) <= 0
        ):
            raise ValueError("foldable linear shape differs")
        self.input = nn.Linear(input_dim, hidden_dim, bias=False)
        self.output = nn.Linear(hidden_dim, output_dim, bias=True)

    def initialize_repeated_identity(self) -> None:
        """Initialize a square map as repeated balanced identity frames."""

        input_dim = self.input.in_features
        hidden_dim = self.input.out_features
        output_dim = self.output.out_features
        if input_dim != output_dim or hidden_dim % input_dim != 0:
            raise ValueError("repeated identity shape differs")
        repeats = hidden_dim // input_dim
        scale = 1.0 / math.sqrt(float(repeats))
        identity = torch.eye(
            input_dim, dtype=self.input.weight.dtype, device=self.input.weight.device
        )
        with torch.no_grad():
            self.input.weight.copy_(identity.repeat(repeats, 1) * scale)
            self.output.weight.copy_(identity.repeat(1, repeats) * scale)
            self.output.bias.zero_()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Apply the training-time factorization."""

        return cast(torch.Tensor, self.output(self.input(values)))

    def fold(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the equivalent single affine weight and bias."""

        return (
            torch.matmul(self.output.weight, self.input.weight).contiguous(),
            self.output.bias.contiguous(),
        )
