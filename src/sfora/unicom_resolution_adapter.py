"""Source-preserving resolution adapter for the pretrained UNICOM B/16 graph."""

from __future__ import annotations

from typing import Any, cast

import torch
from torch import nn
from torch.nn import functional as F


def _check_source_geometry(model: nn.Module) -> int:
    """Reject graphs whose flattened pretrained feature head has another contract."""

    graph: Any = model
    try:
        patch = graph.patch_embed
        projection = patch.proj
        position = graph.pos_embed
        first_feature = graph.feature[0]
        width = int(graph.dim)
    except (AttributeError, IndexError, TypeError, ValueError) as error:
        raise ValueError("UNICOM B/16 source geometry differs") from error
    if (
        not isinstance(projection, nn.Conv2d)
        or projection.kernel_size != (16, 16)
        or projection.stride != (16, 16)
        or projection.in_channels != 3
        or projection.out_channels != width
        or patch.num_patches != 196
        or position.shape != (1, 196, width)
        or not isinstance(first_feature, nn.Linear)
        or first_feature.in_features != 196 * width
    ):
        raise ValueError("UNICOM B/16 source geometry differs")
    return width


def output_at_resolution(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Run source B/16 at 224 or 336 pixels through its unchanged feature head.

    The 224 path follows the pretrained model's operation order exactly. At
    336 pixels, bicubic position-grid interpolation precedes the original
    blocks; area resampling returns final tokens to the 14×14 head geometry.
    No parameters or source-model state are changed.
    """

    if model.training:
        raise ValueError("UNICOM resolution adapter requires evaluation mode")
    return _run_at_resolution(model, images)


def train_output_at_resolution(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Run the same source graph at 224 or 336 pixels with training enabled."""

    if not model.training:
        raise ValueError("UNICOM training adapter requires training mode")
    return _run_at_resolution(model, images)


def _run_at_resolution(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    if (
        images.ndim != 4
        or images.shape[1] != 3
        or images.shape[-2:]
        not in (
            (224, 224),
            (336, 336),
        )
    ):
        raise ValueError("UNICOM resolution adapter requires 224 or 336 pixel RGB images")
    width = _check_source_geometry(model)
    graph: Any = model
    batch = images.shape[0]
    tokens = graph.patch_embed(images)
    if images.shape[-1] == 224:
        position = graph.pos_embed
    else:
        position_grid = graph.pos_embed.reshape(1, 14, 14, width).permute(0, 3, 1, 2)
        position = (
            F.interpolate(position_grid, size=(21, 21), mode="bicubic", align_corners=False)
            .permute(0, 2, 3, 1)
            .reshape(1, 441, width)
        )
    tokens = tokens + position
    for block in graph.blocks:
        tokens = block(tokens)
    tokens = graph.norm(tokens.float())
    if images.shape[-1] == 336:
        grid = tokens.reshape(batch, 21, 21, width).permute(0, 3, 1, 2)
        tokens = F.interpolate(grid, size=(14, 14), mode="area").permute(0, 2, 3, 1)
    return cast(torch.Tensor, graph.feature(torch.reshape(tokens, (batch, 196 * width))))
