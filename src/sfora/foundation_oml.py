"""Minimal helpers for evaluating the released OML In-Shop ViT extractor."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from sfora.data import ImageExample


def load_oml_inshop_examples(
    partition_path: Path, *, image_root: Path, split: str
) -> list[ImageExample]:
    """Load the released OML protocol, which maps ``img/`` into ``img_highres``."""

    if split not in {"train", "query", "gallery"}:
        raise ValueError("OML In-Shop split must be train, query, or gallery")
    lines = partition_path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 3 or lines[1].split() != [
        "image_name",
        "item_id",
        "evaluation_status",
    ]:
        raise ValueError("invalid In-Shop evaluation partition")
    try:
        declared_rows = int(lines[0])
    except ValueError as error:
        raise ValueError("invalid In-Shop evaluation partition count") from error
    parsed = [line.split() for line in lines[2:] if line.strip()]
    if len(parsed) != declared_rows or any(len(row) != 3 for row in parsed):
        raise ValueError("In-Shop evaluation partition row count differs")
    train_counts: dict[str, int] = {}
    for _, item_id, status in parsed:
        if status == "train":
            train_counts[item_id] = train_counts.get(item_id, 0) + 1
    label_statuses = {"train"} if split == "train" else {"query", "gallery"}
    item_ids = sorted(
        {
            row[1]
            for row in parsed
            if row[2] in label_statuses and (split != "train" or train_counts[row[1]] > 1)
        }
    )
    label_by_item = {item_id: label for label, item_id in enumerate(item_ids)}
    examples: list[ImageExample] = []
    for image_name, item_id, status in parsed:
        if status != split:
            continue
        if split == "train" and train_counts[item_id] <= 1:
            continue
        if not image_name.startswith("img/"):
            raise ValueError("OML In-Shop image path must start with img/")
        image_path = image_root / image_name.removeprefix("img/")
        if not image_path.is_file():
            raise ValueError(f"missing OML In-Shop image: {image_path}")
        examples.append(
            ImageExample(
                example_id=f"oml-inshop-{split}-{image_name}",
                image=image_path,
                label=label_by_item[item_id],
            )
        )
    return examples


def oml_vit_state_dict(checkpoint: object) -> dict[str, torch.Tensor]:
    """Remove the exact Lightning/OML wrapper from a released extractor state."""

    if type(checkpoint) is not dict or not isinstance(checkpoint.get("state_dict"), Mapping):
        raise ValueError("OML checkpoint must contain a state_dict mapping")
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


def configure_oml_input_size(model: Any, *, input_size: int) -> None:
    """Use timm's positional-embedding resize for a fixed square input."""

    if type(input_size) is not int or input_size < 224 or input_size % 16:
        raise ValueError("OML input size must be a multiple of the patch size")
    if input_size == 224:
        return
    setter = getattr(model, "set_input_size", None)
    if not callable(setter):
        raise ValueError("OML model cannot resize its positional embedding")
    setter((input_size, input_size))
