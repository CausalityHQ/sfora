"""Split a flattened-token UNICOM encoder at its final transformer block."""

from __future__ import annotations

import torch
from torch import nn


def _parts(model: nn.Module) -> tuple[nn.Module, torch.Tensor, nn.ModuleList, nn.Module, nn.Module]:
    patch_embed = getattr(model, "patch_embed", None)
    pos_embed = getattr(model, "pos_embed", None)
    blocks = getattr(model, "blocks", None)
    norm = getattr(model, "norm", None)
    feature = getattr(model, "feature", None)
    if (
        not isinstance(patch_embed, nn.Module)
        or not isinstance(pos_embed, torch.Tensor)
        or pos_embed.ndim != 3
        or pos_embed.shape[0] != 1
        or not isinstance(blocks, nn.ModuleList)
        or len(blocks) < 1
        or not isinstance(norm, nn.Module)
        or not isinstance(feature, nn.Module)
    ):
        raise ValueError("UNICOM token adapter model inventory differs")
    return patch_embed, pos_embed, blocks, norm, feature


def last_block_input(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Return detached prefix tokens for a fixed image view and encoder state."""

    patch_embed, pos_embed, blocks, _norm, _feature = _parts(model)
    if images.ndim != 4 or images.shape[0] < 1:
        raise ValueError("UNICOM token adapter image inventory differs")
    with torch.no_grad():
        tokens = patch_embed(images)
        if tokens.ndim == 4:
            tokens = tokens.flatten(2).transpose(1, 2)
        if tokens.ndim != 3 or tokens.shape[1:] != pos_embed.shape[1:]:
            raise ValueError("UNICOM token adapter token grid differs")
        tokens = tokens + pos_embed
        for block in blocks[:-1]:
            tokens = block(tokens)
    return tokens.detach().contiguous()


def output_from_last_block_input(model: nn.Module, tokens: torch.Tensor) -> torch.Tensor:
    """Run the final block, normalization and original flattened-token head."""

    _patch_embed, pos_embed, blocks, norm, feature = _parts(model)
    if (
        tokens.ndim != 3
        or tokens.shape[0] < 1
        or tokens.shape[1:] != pos_embed.shape[1:]
        or tokens.device != pos_embed.device
    ):
        raise ValueError("UNICOM token adapter token grid differs")
    final = blocks[-1](tokens)
    flattened = norm(final.float()).reshape(len(tokens), -1)
    return feature(flattened)


__all__ = ["last_block_input", "output_from_last_block_input"]
