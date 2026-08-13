"""Trainable image-level foundation encoder primitives."""

from __future__ import annotations

from collections import defaultdict
from math import isfinite

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from sfora.data import ImageExample


def configure_vit_trainable_layers(model: nn.Module, *, trainable_blocks: int) -> tuple[str, ...]:
    """Freeze a ViT, then unfreeze its final blocks and output norm."""

    blocks = getattr(model, "blocks", None)
    norm = getattr(model, "norm", None)
    if (
        not isinstance(blocks, nn.ModuleList)
        or not isinstance(norm, nn.Module)
        or type(trainable_blocks) is not int
        or not 0 <= trainable_blocks <= len(blocks)
    ):
        raise ValueError("model or trainable block count differs from a ViT tail")
    model.requires_grad_(False)
    names: list[str] = []
    if trainable_blocks:
        start = len(blocks) - trainable_blocks
        for index in range(start, len(blocks)):
            blocks[index].requires_grad_(True)
            names.append(f"blocks.{index}")
        norm.requires_grad_(True)
        names.append("norm")
    return tuple(names)


def identity_disjoint_train_validation(
    examples: list[ImageExample], *, seed: int, validation_fraction: float
) -> tuple[list[ImageExample], list[ImageExample]]:
    """Split complete identities into optimization and validation roles."""

    if (
        type(seed) is not int
        or type(validation_fraction) is not float
        or not isfinite(validation_fraction)
        or not 0.0 < validation_fraction < 1.0
    ):
        raise ValueError("identity split configuration differs")
    counts: dict[int, int] = defaultdict(int)
    for row in examples:
        counts[row.label] += 1
    labels = np.asarray(
        sorted(label for label, count in counts.items() if count >= 2), dtype=np.int64
    )
    if labels.size < 2:
        raise ValueError("identity split requires at least two labels")
    rng = np.random.default_rng(seed)
    shuffled = labels.copy()
    rng.shuffle(shuffled)
    validation_count = max(1, min(labels.size - 1, round(labels.size * validation_fraction)))
    validation_labels = set(int(label) for label in shuffled[:validation_count])
    train = [row for row in examples if row.label not in validation_labels]
    validation = [row for row in examples if row.label in validation_labels]
    return train, validation


def query_gallery_from_identities(
    examples: list[ImageExample], *, seed: int
) -> tuple[list[ImageExample], list[ImageExample]]:
    """Select one deterministic query per identity and use the remainder as gallery."""

    grouped: dict[int, list[ImageExample]] = defaultdict(list)
    for row in examples:
        grouped[row.label].append(row)
    if not grouped or any(len(rows) < 2 for rows in grouped.values()):
        raise ValueError("validation identities require at least two images")
    rng = np.random.default_rng(seed)
    query: list[ImageExample] = []
    gallery: list[ImageExample] = []
    for label in sorted(grouped):
        rows = sorted(grouped[label], key=lambda row: row.example_id)
        query_index = int(rng.integers(0, len(rows)))
        query.append(rows[query_index])
        gallery.extend(row for index, row in enumerate(rows) if index != query_index)
    return query, gallery


def select_query_gallery_identity_subset(
    query: list[ImageExample],
    gallery: list[ImageExample],
    *,
    seed: int,
    fraction: float,
    complement: bool,
) -> tuple[list[ImageExample], list[ImageExample]]:
    """Select complete identities from an existing query/gallery protocol."""

    query_labels = {row.label for row in query}
    gallery_labels = {row.label for row in gallery}
    if (
        query_labels != gallery_labels
        or len(query_labels) < 2
        or type(seed) is not int
        or type(fraction) is not float
        or not isfinite(fraction)
        or not 0.0 < fraction < 1.0
        or type(complement) is not bool
    ):
        raise ValueError("query/gallery identity subset configuration differs")
    labels = np.asarray(sorted(query_labels), dtype=np.int64)
    rng = np.random.default_rng(seed)
    rng.shuffle(labels)
    count = max(1, min(labels.size - 1, round(labels.size * fraction)))
    selected = set(int(label) for label in labels[:count])
    if complement:
        selected = query_labels - selected
    return (
        [row for row in query if row.label in selected],
        [row for row in gallery if row.label in selected],
    )


class CosFaceHead(nn.Module):
    """Normalized classification head used only while fitting the encoder."""

    def __init__(
        self, *, embedding_dim: int, class_count: int, margin: float = 0.2, scale: float = 32.0
    ) -> None:
        super().__init__()
        if (
            type(embedding_dim) is not int
            or embedding_dim <= 0
            or type(class_count) is not int
            or class_count <= 1
            or type(margin) is not float
            or not 0.0 <= margin < 1.0
            or type(scale) is not float
            or not isfinite(scale)
            or scale <= 0.0
        ):
            raise ValueError("CosFace configuration differs")
        self.margin = margin
        self.scale = scale
        self.weight = nn.Parameter(torch.empty(class_count, embedding_dim))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if (
            embeddings.dtype != torch.float32
            or embeddings.ndim != 2
            or labels.dtype != torch.int64
            or labels.shape != (embeddings.shape[0],)
        ):
            raise ValueError("CosFace inputs differ")
        logits = F.normalize(embeddings, dim=1) @ F.normalize(self.weight, dim=1).T
        target = F.one_hot(labels, num_classes=self.weight.shape[0]).to(logits.dtype)
        return self.scale * (logits - self.margin * target)


class IdentityNeck(nn.Module):
    """A deployed linear neck initialized to preserve the released embedding exactly."""

    def __init__(self, dimension: int) -> None:
        super().__init__()
        if type(dimension) is not int or dimension <= 0:
            raise ValueError("neck dimension must be a positive builtin integer")
        self.projection = nn.Linear(dimension, dimension, bias=False)
        with torch.no_grad():
            self.projection.weight.copy_(torch.eye(dimension))

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.projection(embeddings)
