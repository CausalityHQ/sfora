"""Trainable image-level foundation encoder primitives."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator, Sequence
from math import ceil, comb, isfinite

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


def retrieval_query_values(
    query: torch.Tensor,
    query_labels: np.ndarray,
    gallery: torch.Tensor,
    gallery_labels: np.ndarray,
    *,
    chunk_size: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact per-query R@1 hits and AP@R for paired comparisons."""

    if (
        type(query_labels) is not np.ndarray
        or type(gallery_labels) is not np.ndarray
        or query_labels.dtype != np.dtype("int64")
        or gallery_labels.dtype != np.dtype("int64")
        or query_labels.ndim != 1
        or gallery_labels.ndim != 1
        or query.dtype != torch.float32
        or gallery.dtype != torch.float32
        or query.ndim != 2
        or gallery.ndim != 2
        or query.shape[1] != gallery.shape[1]
        or query.shape[0] != query_labels.shape[0]
        or gallery.shape[0] != gallery_labels.shape[0]
        or type(chunk_size) is not int
        or chunk_size <= 0
    ):
        raise ValueError("retrieval query inputs differ")
    query = F.normalize(query, dim=1)
    gallery = F.normalize(gallery, dim=1)
    relevant_counts = np.asarray(
        [np.count_nonzero(gallery_labels == label) for label in query_labels], dtype=np.int64
    )
    if bool((relevant_counts == 0).any()):
        raise ValueError("every query requires a gallery match")
    hits: list[np.ndarray] = []
    average_precision: list[np.ndarray] = []
    for start in range(0, query.shape[0], chunk_size):
        stop = min(start + chunk_size, query.shape[0])
        maximum_r = int(relevant_counts[start:stop].max())
        top = (
            torch.topk(
                query[start:stop] @ gallery.T,
                k=maximum_r,
                dim=1,
                largest=True,
                sorted=True,
            )
            .indices.detach()
            .cpu()
            .numpy()
        )
        matches = gallery_labels[top] == query_labels[start:stop, None]
        hits.append(matches[:, 0].copy())
        precision = np.cumsum(matches, axis=1) / np.arange(1, maximum_r + 1)[None, :]
        rows = [
            float((precision[row, :relevant] * matches[row, :relevant]).sum() / relevant)
            for row, relevant in enumerate(relevant_counts[start:stop])
        ]
        average_precision.append(np.asarray(rows, dtype=np.float64))
    return np.concatenate(hits), np.concatenate(average_precision)


def paired_retrieval_statistics(
    *,
    initial_hits: np.ndarray,
    final_hits: np.ndarray,
    initial_ap: np.ndarray,
    final_ap: np.ndarray,
    seed: int,
    bootstrap_replicates: int,
) -> dict[str, float | int]:
    """Summarize paired retrieval changes with exact McNemar and bootstrap uncertainty."""

    if (
        type(initial_hits) is not np.ndarray
        or type(final_hits) is not np.ndarray
        or type(initial_ap) is not np.ndarray
        or type(final_ap) is not np.ndarray
        or initial_hits.dtype != np.dtype("bool")
        or final_hits.dtype != np.dtype("bool")
        or initial_ap.dtype != np.dtype("float64")
        or final_ap.dtype != np.dtype("float64")
        or initial_hits.ndim != 1
        or not (initial_hits.shape == final_hits.shape == initial_ap.shape == final_ap.shape)
        or initial_hits.size == 0
        or type(seed) is not int
        or type(bootstrap_replicates) is not int
        or bootstrap_replicates < 100
    ):
        raise ValueError("paired retrieval inputs differ")
    lost = int(np.count_nonzero(initial_hits & ~final_hits))
    gained = int(np.count_nonzero(~initial_hits & final_hits))
    discordant = lost + gained
    if discordant:
        tail = sum(comb(discordant, k) for k in range(min(lost, gained) + 1)) / (2**discordant)
        mcnemar_p = min(1.0, 2.0 * tail)
    else:
        mcnemar_p = 1.0
    delta = final_ap - initial_ap
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(bootstrap_replicates, dtype=np.float64)
    for replicate in range(bootstrap_replicates):
        indexes = rng.integers(0, delta.size, size=delta.size)
        bootstrap[replicate] = delta[indexes].mean()
    lower, upper = np.quantile(bootstrap, [0.025, 0.975])
    return {
        "recall_lost": lost,
        "recall_gained": gained,
        "mcnemar_exact_p": float(mcnemar_p),
        "map_at_r_delta": float(delta.mean()),
        "map_at_r_delta_ci95_lower": float(lower),
        "map_at_r_delta_ci95_upper": float(upper),
        "bootstrap_replicates": bootstrap_replicates,
        "bootstrap_seed": seed,
    }


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
        rng = np.random.default_rng([self.seed, self.epoch])
        available = np.asarray(list(self._by_label), dtype=np.int64)
        ordered = rng.permutation(available)
        padding = len(self) * self.labels_per_batch - ordered.size
        if padding:
            ordered = np.concatenate([ordered, rng.choice(available, size=padding, replace=False)])
        for start in range(0, ordered.size, self.labels_per_batch):
            selected = ordered[start : start + self.labels_per_batch]
            batch: list[int] = []
            for raw_label in selected:
                candidates = self._by_label[int(raw_label)]
                chosen = np.resize(rng.permutation(candidates), self.instances_per_label)
                batch.extend(int(index) for index in chosen)
            yield batch
