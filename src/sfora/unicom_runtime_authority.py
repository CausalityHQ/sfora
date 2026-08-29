"""Canonical CPU operation for device-independent UniCOM runtime authority."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import torch
from PIL import Image


def build_runtime_authority_descriptor(
    raw_model: torch.nn.Module,
    transform: Callable[[Image.Image], torch.Tensor],
    image_path: Path,
) -> torch.Tensor:
    """Build the registered 512-wide descriptor with pinned CPU execution."""

    if not isinstance(raw_model, torch.nn.Module):
        raise TypeError("runtime authority model differs")
    model_values = (*raw_model.parameters(), *raw_model.buffers())
    if any(value.device.type != "cpu" for value in model_values):
        raise ValueError("runtime authority model device differs")

    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        raw_model.eval()
        with Image.open(image_path) as image, torch.no_grad():
            transformed = transform(image.convert("RGB"))
            if (
                not isinstance(transformed, torch.Tensor)
                or transformed.device.type != "cpu"
            ):
                raise ValueError("runtime authority transform differs")
            full = raw_model(transformed.unsqueeze(0)).float()
            if full.shape != (1, 768) or not torch.isfinite(full).all():
                raise ValueError("runtime authority descriptor differs")
            full = torch.nn.functional.normalize(full, dim=1)
            descriptor = full[:, :512].detach().cpu().contiguous()
    finally:
        torch.set_num_threads(previous_threads)
    if descriptor.dtype != torch.float32 or not torch.isfinite(descriptor).all():
        raise ValueError("runtime authority descriptor differs")
    return descriptor
