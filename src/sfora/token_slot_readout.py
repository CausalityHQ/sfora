"""Content-addressed token readouts for compact learned descriptors."""

from __future__ import annotations

import math
from typing import cast

import torch
from torch import nn
from torch.nn import functional as F


def _check_tokens(tokens: torch.Tensor, width: int) -> None:
    if (
        type(tokens) is not torch.Tensor
        or tokens.ndim != 3
        or tokens.shape[0] < 1
        or tokens.shape[1] < 1
        or tokens.shape[2] != width
        or not tokens.is_floating_point()
    ):
        raise ValueError("token readout geometry differs")


class MeanTokenReadout(nn.Module):
    """Project the permutation-invariant mean of a nonempty token grid."""

    def __init__(self, *, width: int, output_width: int) -> None:
        super().__init__()
        if width < 1 or output_width < 1:
            raise ValueError("token readout geometry differs")
        self.width = width
        self.output_width = output_width
        self.projection = nn.Linear(width, output_width)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        _check_tokens(tokens, self.width)
        return cast(torch.Tensor, self.projection(tokens.float().mean(dim=1)))


class SlotTokenReadout(nn.Module):
    """Pool tokens into learned content-addressed slots, then project."""

    def __init__(self, *, width: int, output_width: int, slots: int) -> None:
        super().__init__()
        if width < 1 or output_width < 1 or slots < 1:
            raise ValueError("token readout geometry differs")
        self.width = width
        self.output_width = output_width
        self.slots = slots
        self.queries = nn.Parameter(torch.zeros(slots, width))
        self.projection = nn.Linear(slots * width, output_width)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def initialize_from_mean(
        self, mean: MeanTokenReadout, *, seed: int, query_std: float = 0.01
    ) -> None:
        """Set projection to the mean map; small queries break slot symmetry."""

        if (
            type(mean) is not MeanTokenReadout
            or mean.width != self.width
            or mean.output_width != self.output_width
            or seed < 0
            or not math.isfinite(query_std)
            or query_std < 0
        ):
            raise ValueError("token readout geometry differs")
        generator = torch.Generator(device="cpu").manual_seed(seed)
        with torch.no_grad():
            self.queries.copy_(
                torch.randn((self.slots, self.width), generator=generator).to(self.queries.device)
                * query_std
            )
            tiled = mean.projection.weight.repeat(1, self.slots) / self.slots
            self.projection.weight.copy_(tiled.to(self.projection.weight.device))
            self.projection.bias.copy_(mean.projection.bias.to(self.projection.bias.device))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        _check_tokens(tokens, self.width)
        values = tokens.float()
        weights = F.softmax(
            (values @ self.queries.float().T).transpose(1, 2) / math.sqrt(self.width), dim=-1
        )
        pooled = weights @ values
        return cast(torch.Tensor, self.projection(pooled.flatten(start_dim=1)))
