"""Trainable image-level foundation encoder primitives."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator, Sequence
from math import ceil, isfinite

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Sampler

from sfora.data import ImageExample


def configure_vit_trainable_layers(model: nn.Module, *, trainable_blocks: int) -> tuple[str, ...]:
    """Freeze a ViT, then unfreeze its final blocks and output norm."""

    blocks = getattr(model, "blocks", None)
    norm = getattr(model, "norm", None)
    if (
        not isinstance(blocks, (nn.ModuleList, nn.Sequential))
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


class TokenResidualGate(nn.Module):
    """Mix local patch evidence into a CLS descriptor with an identity start."""

    def __init__(self, dimension: int) -> None:
        super().__init__()
        if type(dimension) is not int or dimension <= 0:
            raise ValueError("token gate dimension must be a positive builtin integer")
        self.gate = nn.Parameter(torch.zeros(dimension, dtype=torch.float32))

    def forward(self, cls: torch.Tensor, patches: torch.Tensor) -> torch.Tensor:
        if (
            cls.dtype != torch.float32
            or patches.dtype != torch.float32
            or cls.ndim != 2
            or cls.shape != patches.shape
            or cls.shape[1] != self.gate.numel()
        ):
            raise ValueError("token gate inputs differ")
        base = F.normalize(cls, dim=1)
        local = F.normalize(patches, dim=1)
        return F.normalize(base + torch.tanh(self.gate) * local, dim=1)


def split_cls_patch_tokens(tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the CLS token and mean of only the local patch tokens."""

    if tokens.dtype != torch.float32 or tokens.ndim != 3 or tokens.shape[1] < 2:
        raise ValueError("ViT token tensor differs")
    return tokens[:, 0], tokens[:, 1:].mean(dim=1)


def batch_hard_soft_triplet(embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Return the softplus loss for each anchor's batch-hard positive and negative."""

    if (
        embeddings.dtype != torch.float32
        or embeddings.ndim != 2
        or labels.dtype != torch.int64
        or labels.shape != (embeddings.shape[0],)
    ):
        raise ValueError("batch-hard inputs differ")
    same = labels[:, None] == labels[None, :]
    same.fill_diagonal_(False)
    if not bool(same.any(dim=1).all()) or labels.unique().numel() < 2:
        raise ValueError("batch-hard labels require positives and negatives")
    distances = torch.cdist(embeddings, embeddings, p=2)
    hardest_positive = distances.masked_fill(~same, -torch.inf).amax(dim=1)
    hardest_negative = distances.masked_fill(
        same | torch.eye(labels.numel(), dtype=torch.bool, device=labels.device), torch.inf
    ).amin(dim=1)
    return F.softplus(hardest_positive - hardest_negative).mean()


class IdentityBalancedBatchSampler(Sampler[list[int]]):
    """Yield deterministic P-by-K identity batches, resampling examples when needed."""

    def __init__(
        self,
        labels: Sequence[int],
        *,
        labels_per_batch: int,
        instances_per_label: int,
        seed: int,
    ) -> None:
        if (
            not labels
            or type(labels_per_batch) is not int
            or type(instances_per_label) is not int
            or type(seed) is not int
            or labels_per_batch < 2
            or instances_per_label < 2
        ):
            raise ValueError("identity-balanced sampler configuration differs")
        by_label: dict[int, list[int]] = defaultdict(list)
        for index, label in enumerate(labels):
            if type(label) is not int:
                raise ValueError("identity-balanced labels must be builtin integers")
            by_label[label].append(index)
        if len(by_label) < labels_per_batch:
            raise ValueError("identity-balanced sampler has too few identities")
        self._by_label = dict(sorted(by_label.items()))
        self.labels_per_batch = labels_per_batch
        self.instances_per_label = instances_per_label
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return ceil(len(self._by_label) / self.labels_per_batch)

    def set_epoch(self, epoch: int) -> None:
        if type(epoch) is not int or epoch < 0:
            raise ValueError("sampler epoch must be a nonnegative builtin integer")
        self.epoch = epoch

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        available = np.asarray(list(self._by_label), dtype=np.int64)
        for _ in range(len(self)):
            selected = rng.choice(available, size=self.labels_per_batch, replace=False)
            batch: list[int] = []
            for raw_label in selected:
                candidates = self._by_label[int(raw_label)]
                chosen = rng.choice(
                    candidates,
                    size=self.instances_per_label,
                    replace=len(candidates) < self.instances_per_label,
                )
                batch.extend(int(index) for index in chosen)
            yield batch
