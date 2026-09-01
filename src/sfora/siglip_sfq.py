"""Optimization-only Shrunk-Fisher-Quotient transfer diagnostics."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import torch
from sklearn.covariance import LedoitWolf
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


@dataclass(frozen=True, slots=True)
class SFQProjectionEvidence:
    """One robust Fisher quotient and its whitening-only comparator."""

    weight: torch.Tensor
    whitening_weight: torch.Tensor
    ledoit_wolf_shrinkage: float
    minimum_within_eigenvalue: float
    maximum_within_eigenvalue: float
    bbp_threshold: float
    sample_spikes: tuple[float, ...]
    retained_spikes: tuple[float, ...]
    gains: tuple[float, ...]
    reliable_rank: int


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


def _canonicalize_rows(matrix: torch.Tensor) -> torch.Tensor:
    canonical = matrix.clone()
    for row in range(canonical.shape[0]):
        pivot = int(torch.argmax(torch.abs(canonical[row])))
        value = canonical[row, pivot]
        if not bool(torch.isfinite(value)) or float(torch.abs(value)) == 0.0:
            raise ValueError("SFQ projection direction differs")
        if float(value) < 0.0:
            canonical[row].neg_()
    return canonical


def _uncentered_reduction(
    normalized_features: torch.Tensor,
    factor: torch.Tensor,
    *,
    output_dimensions: int,
) -> torch.Tensor:
    transformed = normalized_features @ factor.T
    if int(torch.linalg.matrix_rank(transformed)) < output_dimensions:
        raise ValueError("SFQ projection rank differs")
    _left, _singular_values, right = torch.linalg.svd(transformed, full_matrices=False)
    projection = _canonicalize_rows(right[:output_dimensions])
    weight = projection @ factor
    if weight.shape != (output_dimensions, normalized_features.shape[1]) or not bool(
        torch.isfinite(weight).all()
    ):
        raise ValueError("SFQ projection factor differs")
    return weight


def fit_sfq_projection(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    output_dimensions: int,
) -> SFQProjectionEvidence:
    """Fit a parameter-free robust Fisher metric on one fit-only class set."""

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
        or type(output_dimensions) is not int
        or not 1 <= output_dimensions <= min(features.shape)
    ):
        raise ValueError("SFQ projection authority differs")
    unique_labels = tuple(sorted(int(value) for value in torch.unique(labels).tolist()))
    if len(unique_labels) < 4 or any(int((labels == label).sum()) < 2 for label in unique_labels):
        raise ValueError("SFQ projection class authority differs")

    normalized = F.normalize(features.double(), dim=1)
    if bool((torch.linalg.vector_norm(normalized, dim=1) <= 0).any()):
        raise ValueError("SFQ normalized feature authority differs")
    means = []
    residual_rows = []
    counts = []
    for label in unique_labels:
        members = normalized[labels == label]
        mean = members.mean(dim=0)
        means.append(mean)
        residual_rows.append(members - mean)
        counts.append(members.shape[0])
    residuals = torch.cat(residual_rows, dim=0).contiguous()
    estimator = LedoitWolf(assume_centered=True).fit(residuals.numpy())
    covariance = torch.from_numpy(estimator.covariance_).to(dtype=torch.float64)
    if covariance.shape != (features.shape[1], features.shape[1]) or not bool(
        torch.isfinite(covariance).all()
    ):
        raise ValueError("SFQ within covariance differs")
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    minimum_eigenvalue = float(eigenvalues[0])
    maximum_eigenvalue = float(eigenvalues[-1])
    if minimum_eigenvalue <= 0.0 or not math.isfinite(maximum_eigenvalue):
        raise ValueError("SFQ within covariance is not positive definite")
    whitening = (eigenvectors * torch.rsqrt(eigenvalues).unsqueeze(0)) @ eigenvectors.T

    global_mean = normalized.mean(dim=0)
    whitened_means = torch.stack(
        [
            math.sqrt(count) * ((mean - global_mean) @ whitening)
            for mean, count in zip(means, counts, strict=True)
        ]
    )
    _left, singular_values, right = torch.linalg.svd(whitened_means, full_matrices=False)
    class_count = len(unique_labels)
    dimension = features.shape[1]
    gamma = dimension / class_count
    threshold = (1.0 + math.sqrt(gamma)) ** 2
    sample_spikes = tuple(float(value) for value in (singular_values.square() / class_count))
    retained_indexes = tuple(
        index for index, value in enumerate(sample_spikes) if value > threshold
    )
    if not retained_indexes:
        raise ValueError("SFQ reliable rank is zero")
    retained_spikes = tuple(sample_spikes[index] for index in retained_indexes)
    gains = []
    for sample_spike in retained_spikes:
        radicand = (sample_spike - 1.0 - gamma) ** 2 - 4.0 * gamma
        if radicand < -1.0e-12:
            raise ValueError("SFQ population spike radicand differs")
        theta = (sample_spike - 1.0 - gamma + math.sqrt(max(0.0, radicand))) / 2.0
        if theta <= 0.0 or not math.isfinite(theta):
            raise ValueError("SFQ population spike differs")
        alignment = (1.0 - gamma / theta**2) / (1.0 + gamma / theta)
        gain = alignment * theta
        if alignment <= 0.0 or gain <= 0.0 or not math.isfinite(gain):
            raise ValueError("SFQ nonlinear gain differs")
        gains.append(gain)
    retained_directions = right[list(retained_indexes)]
    metric = torch.eye(dimension, dtype=torch.float64) + (
        retained_directions.T
        @ torch.diag(torch.tensor(gains, dtype=torch.float64))
        @ retained_directions
    )
    factor = metric @ whitening
    weight = _uncentered_reduction(
        normalized,
        factor,
        output_dimensions=output_dimensions,
    )
    whitening_weight = _uncentered_reduction(
        normalized,
        whitening,
        output_dimensions=output_dimensions,
    )
    return SFQProjectionEvidence(
        weight=weight.to(dtype=torch.float32).contiguous(),
        whitening_weight=whitening_weight.to(dtype=torch.float32).contiguous(),
        ledoit_wolf_shrinkage=float(estimator.shrinkage_),
        minimum_within_eigenvalue=minimum_eigenvalue,
        maximum_within_eigenvalue=maximum_eigenvalue,
        bbp_threshold=threshold,
        sample_spikes=sample_spikes,
        retained_spikes=retained_spikes,
        gains=tuple(gains),
        reliable_rank=len(retained_indexes),
    )
