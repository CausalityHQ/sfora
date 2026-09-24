"""Exact split of a flattened-token encoder at its final transformer block."""

from __future__ import annotations

import torch
from torch import nn

from sfora.unicom_tail_adapter import last_block_input, output_from_last_block_input


class TinyEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.patch_embed = nn.Conv2d(3, 4, kernel_size=2, stride=2)
        self.pos_embed = nn.Parameter(torch.randn(1, 4, 4))
        self.blocks = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])
        self.norm = nn.LayerNorm(4)
        self.feature = nn.Linear(16, 5)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_embed(images).flatten(2).transpose(1, 2) + self.pos_embed
        for block in self.blocks:
            tokens = block(tokens)
        return self.feature(self.norm(tokens).reshape(len(images), -1))


def test_split_matches_unsplit_encoder_and_preserves_final_block_gradient() -> None:
    torch.manual_seed(43)
    model = TinyEncoder().eval()
    images = torch.randn(3, 3, 4, 4)
    with torch.no_grad():
        expected = model(images)
        tokens = last_block_input(model, images)
    assert tokens.shape == (3, 4, 4)
    actual = output_from_last_block_input(model, tokens)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.blocks[-1].parameters():
        parameter.requires_grad_(True)
    output_from_last_block_input(model, tokens).sum().backward()
    assert all(parameter.grad is not None for parameter in model.blocks[-1].parameters())
    assert all(parameter.grad is None for parameter in model.blocks[0].parameters())


def test_split_rejects_wrong_token_grid() -> None:
    model = TinyEncoder().eval()
    bad = torch.randn(2, 3, 4)
    try:
        output_from_last_block_input(model, bad)
    except ValueError as error:
        assert "token" in str(error)
    else:
        raise AssertionError("wrong grid was accepted")
