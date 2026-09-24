"""Behavioral tests for the source-preserving UNICOM resolution adapter."""

import pytest
import torch
from torch import nn

from sfora.unicom_resolution_adapter import output_at_resolution


class _PatchEmbed(nn.Module):
    num_patches = 196

    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(3, 8, 16, 16)

    def forward(self, images):
        return self.proj(images).flatten(2).transpose(1, 2)


class _SourceShape(nn.Module):
    dim = 8

    def __init__(self):
        super().__init__()
        self.patch_embed = _PatchEmbed()
        self.pos_embed = nn.Parameter(torch.randn(1, 196, 8))
        self.blocks = nn.ModuleList([nn.Linear(8, 8)])
        self.norm = nn.LayerNorm(8)
        self.feature = nn.Sequential(nn.Linear(196 * 8, 8, bias=False), nn.BatchNorm1d(8))

    def forward(self, images):
        tokens = self.patch_embed(images) + self.pos_embed
        for block in self.blocks:
            tokens = block(tokens)
        return self.feature(self.norm(tokens.float()).reshape(len(images), -1))


def test_224_output_matches_source_exactly():
    torch.manual_seed(12)
    model = _SourceShape().eval()
    images = torch.randn(2, 3, 224, 224)
    assert torch.equal(output_at_resolution(model, images), model(images))


def test_336_output_has_original_feature_width_and_backward_path():
    torch.manual_seed(13)
    model = _SourceShape().eval()
    images = torch.randn(2, 3, 336, 336)
    output = output_at_resolution(model, images)
    assert output.shape == (2, 8)
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    assert model.pos_embed.grad is not None
    assert torch.isfinite(model.pos_embed.grad).all()


@pytest.mark.parametrize("size", [192, 240, 320])
def test_other_image_sizes_are_rejected(size):
    with pytest.raises(ValueError, match="224 or 336"):
        output_at_resolution(_SourceShape().eval(), torch.empty(2, 3, size, size))


def test_training_mode_is_rejected():
    with pytest.raises(ValueError, match="evaluation mode"):
        output_at_resolution(_SourceShape().train(), torch.empty(2, 3, 224, 224))


def test_wrong_pretrained_head_geometry_is_rejected():
    model = _SourceShape().eval()
    model.feature[0] = nn.Linear(441 * 8, 8, bias=False)
    with pytest.raises(ValueError, match="geometry"):
        output_at_resolution(model, torch.empty(2, 3, 336, 336))
