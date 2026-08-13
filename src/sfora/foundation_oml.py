"""Minimal helpers for evaluating the released OML In-Shop ViT extractor."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

import torch


def oml_vit_state_dict(checkpoint: object) -> dict[str, torch.Tensor]:
    """Remove the exact Lightning/OML wrapper from a released extractor state."""

    if type(checkpoint) is not dict or type(checkpoint.get("state_dict")) is not dict:
        raise ValueError("OML checkpoint must contain a builtin state_dict")
    wrapped = checkpoint["state_dict"]
    assert isinstance(wrapped, Mapping)
    prefix = "model.model."
    if not wrapped or any(type(key) is not str or not key.startswith(prefix) for key in wrapped):
        raise ValueError("OML state keys differ from the released model.model wrapper")
    state: dict[str, torch.Tensor] = {}
    for key, value in wrapped.items():
        if not torch.is_tensor(value):
            raise ValueError("OML state values must be tensors")
        state[key.removeprefix(prefix)] = value
    return state


def load_oml_vit(checkpoint_path: str, *, device: Any) -> Any:
    """Build the released ViT-S/16 extractor without importing the OML package."""

    timm = importlib.import_module("timm")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = timm.create_model("vit_small_patch16_224", pretrained=False, num_classes=0)
    model.load_state_dict(oml_vit_state_dict(checkpoint), strict=True)
    return model.to(device).eval()
