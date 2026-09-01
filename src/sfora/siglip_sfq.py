"""Optimization-only Shrunk-Fisher-Quotient transfer diagnostics."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from sfora.siglip_head_screen import FeatureSplitAuthority


@dataclass(frozen=True, slots=True)
class SFQFold:
    """One class-disjoint fit/validation partition."""

    ordinal: int
    fit_labels: tuple[int, ...]
    validation_labels: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class SFQFoldSchedule:
    """Deterministic nearest-class-pair fold allocation."""

    folds: tuple[SFQFold, ...]
    sha256: str


def _schedule_sha256(
    *,
    folds: tuple[SFQFold, ...],
    split_authority: FeatureSplitAuthority,
) -> str:
    payload = bytearray(b"sfora-sfq-fold-schedule-v1\0")
    payload.extend(bytes.fromhex(split_authority.source_manifest_sha256))
    payload.extend(bytes.fromhex(split_authority.ordered_example_ids_sha256))
    payload.extend(bytes.fromhex(split_authority.feature_matrix_sha256))
    payload.extend(len(folds).to_bytes(8, "big"))
    for fold in folds:
        payload.extend(fold.ordinal.to_bytes(8, "big"))
        for values in (fold.fit_labels, fold.validation_labels):
            payload.extend(len(values).to_bytes(8, "big"))
            for value in values:
                payload.extend(value.to_bytes(8, "big", signed=True))
    return hashlib.sha256(payload).hexdigest()


def build_sfq_fold_schedule(
    features: torch.Tensor,
    labels: torch.Tensor,
    split_authority: FeatureSplitAuthority,
    *,
    fold_count: int = 4,
) -> SFQFoldSchedule:
    """Keep nearest class-mean pairs together in deterministic held-out folds."""

    if (
        type(features) is not torch.Tensor
        or features.ndim != 2
        or features.device.type != "cpu"
        or features.dtype != torch.float32
        or features.shape[0] < 2
        or features.shape[1] < 2
        or not bool(torch.isfinite(features).all())
        or type(labels) is not torch.Tensor
        or labels.shape != (features.shape[0],)
        or labels.device.type != "cpu"
        or labels.dtype != torch.int64
        or type(split_authority) is not FeatureSplitAuthority
        or type(fold_count) is not int
        or fold_count < 2
    ):
        raise ValueError("SFQ fold authority differs")
    split_authority.validated(features=features)
    unique_labels = tuple(sorted(int(value) for value in torch.unique(labels).tolist()))
    if (
        len(unique_labels) < 2 * fold_count
        or unique_labels != tuple(range(len(unique_labels)))
        or any(int((labels == label).sum()) < 2 for label in unique_labels)
    ):
        raise ValueError("SFQ class authority differs")

    normalized = F.normalize(features.double(), dim=1)
    means = []
    class_counts: dict[int, int] = {}
    for label in unique_labels:
        members = normalized[labels == label]
        mean = members.mean(dim=0)
        norm = torch.linalg.vector_norm(mean)
        if not bool(torch.isfinite(norm)) or float(norm) <= 0.0:
            raise ValueError("SFQ class mean authority differs")
        means.append(mean / norm)
        class_counts[label] = members.shape[0]
    mean_matrix = torch.stack(means)
    similarity = mean_matrix @ mean_matrix.T
    edges = sorted(
        (
            (-float(similarity[left, right]), unique_labels[left], unique_labels[right])
            for left in range(len(unique_labels))
            for right in range(left + 1, len(unique_labels))
        ),
        key=lambda edge: edge,
    )
    unused = set(unique_labels)
    groups: list[tuple[int, ...]] = []
    for _negative_similarity, left, right in edges:
        if left in unused and right in unused:
            groups.append((left, right))
            unused.remove(left)
            unused.remove(right)
    groups.extend((label,) for label in sorted(unused))
    if len(groups) < fold_count:
        raise ValueError("SFQ fold group authority differs")

    allocations: list[list[int]] = [[] for _ in range(fold_count)]
    example_counts = [0] * fold_count
    for group in groups:
        fold_ordinal = min(range(fold_count), key=lambda value: (example_counts[value], value))
        allocations[fold_ordinal].extend(group)
        example_counts[fold_ordinal] += sum(class_counts[label] for label in group)

    all_labels = set(unique_labels)
    folds = tuple(
        SFQFold(
            ordinal=ordinal,
            fit_labels=tuple(sorted(all_labels.difference(validation_labels))),
            validation_labels=tuple(sorted(validation_labels)),
        )
        for ordinal, validation_labels in enumerate(allocations)
    )
    if (
        any(not fold.fit_labels or not fold.validation_labels for fold in folds)
        or sorted(label for fold in folds for label in fold.validation_labels)
        != list(unique_labels)
    ):
        raise ValueError("SFQ fold partition differs")
    return SFQFoldSchedule(
        folds=folds,
        sha256=_schedule_sha256(folds=folds, split_authority=split_authority),
    )
