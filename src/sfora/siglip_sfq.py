"""Optimization-only Shrunk-Fisher-Quotient transfer diagnostics."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import cast

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


@dataclass(frozen=True, slots=True)
class SFQFoldEvidence:
    """Primitive projection and integer retrieval evidence for one held-out fold."""

    ordinal: int
    fit_labels: tuple[int, ...]
    validation_labels: tuple[int, ...]
    fit_count: int
    query_count: int
    raw_hits: int
    spectral_hits: int
    whitening_hits: int
    sfq_hits: int
    ledoit_wolf_shrinkage: float
    minimum_within_eigenvalue: float
    maximum_within_eigenvalue: float
    bbp_threshold: float
    sample_spikes: tuple[float, ...]
    retained_spikes: tuple[float, ...]
    gains: tuple[float, ...]
    reliable_rank: int
    spectral_projection_sha256: str
    whitening_projection_sha256: str
    sfq_projection_sha256: str


@dataclass(frozen=True, slots=True)
class SFQResult:
    """Canonical aggregate whose gates are reconstructed from integer counts."""

    schema: str
    claim_eligible: bool
    official_test_access: bool
    source_manifest_sha256: str
    ordered_example_ids_sha256: str
    feature_matrix_sha256: str
    input_dimensions: int
    output_dimensions: int
    fold_count: int
    fold_schedule_sha256: str
    query_count: int
    raw_hits: int
    spectral_hits: int
    whitening_hits: int
    sfq_hits: int
    raw_recall_ppm: int
    spectral_recall_ppm: int
    whitening_recall_ppm: int
    sfq_recall_ppm: int
    sfq_minus_whitening_ppm: int
    valid: bool
    passed: bool
    folds: tuple[SFQFoldEvidence, ...]


def _schedule_sha256(
    *,
    folds: tuple[SFQFold, ...],
    split_authority: FeatureSplitAuthority,
) -> str:
    return _schedule_sha256_from_hashes(
        folds=folds,
        source_manifest_sha256=split_authority.source_manifest_sha256,
        ordered_example_ids_sha256=split_authority.ordered_example_ids_sha256,
        feature_matrix_sha256=split_authority.feature_matrix_sha256,
    )


def _schedule_sha256_from_hashes(
    *,
    folds: tuple[SFQFold, ...],
    source_manifest_sha256: str,
    ordered_example_ids_sha256: str,
    feature_matrix_sha256: str,
) -> str:
    payload = bytearray(b"sfora-sfq-fold-schedule-v1\0")
    payload.extend(bytes.fromhex(source_manifest_sha256))
    payload.extend(bytes.fromhex(ordered_example_ids_sha256))
    payload.extend(bytes.fromhex(feature_matrix_sha256))
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
    if any(not fold.fit_labels or not fold.validation_labels for fold in folds) or sorted(
        label for fold in folds for label in fold.validation_labels
    ) != list(unique_labels):
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


def _projection_sha256(role: str, weight: torch.Tensor) -> str:
    if (
        type(role) is not str
        or not role
        or type(weight) is not torch.Tensor
        or weight.ndim != 2
        or weight.device.type != "cpu"
        or weight.dtype != torch.float32
        or not bool(torch.isfinite(weight).all())
    ):
        raise ValueError("SFQ projection digest authority differs")
    payload = bytearray(b"sfora-sfq-projection-v1\0")
    encoded_role = role.encode("utf-8")
    payload.extend(len(encoded_role).to_bytes(8, "big"))
    payload.extend(encoded_role)
    payload.extend(weight.shape[0].to_bytes(8, "big"))
    payload.extend(weight.shape[1].to_bytes(8, "big"))
    payload.extend(weight.contiguous().numpy().astype("<f4", copy=False).tobytes(order="C"))
    return hashlib.sha256(payload).hexdigest()


def _spectral_projection(features: torch.Tensor, *, output_dimensions: int) -> torch.Tensor:
    if int(torch.linalg.matrix_rank(features.double())) < output_dimensions:
        raise ValueError("SFQ raw spectral rank differs")
    _left, _singular_values, right = torch.linalg.svd(features.double(), full_matrices=False)
    return _canonicalize_rows(right[:output_dimensions]).float().contiguous()


def _recall_at_one_hits(embeddings: torch.Tensor, labels: torch.Tensor) -> tuple[int, int]:
    if (
        type(embeddings) is not torch.Tensor
        or embeddings.ndim != 2
        or embeddings.shape[0] < 2
        or embeddings.device.type != "cpu"
        or not embeddings.is_floating_point()
        or not bool(torch.isfinite(embeddings).all())
        or type(labels) is not torch.Tensor
        or labels.shape != (embeddings.shape[0],)
        or labels.device.type != "cpu"
        or labels.dtype != torch.int64
    ):
        raise ValueError("SFQ retrieval authority differs")
    normalized = F.normalize(embeddings.double(), dim=1)
    if bool((torch.linalg.vector_norm(normalized, dim=1) <= 0).any()):
        raise ValueError("SFQ retrieval norm differs")
    similarity = normalized @ normalized.T
    similarity.fill_diagonal_(-torch.inf)
    nearest = torch.argmax(similarity, dim=1)
    return int((labels[nearest] == labels).sum()), labels.numel()


def _canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _fold_mapping(fold: SFQFoldEvidence) -> dict[str, object]:
    return {
        "ordinal": fold.ordinal,
        "fit_labels": list(fold.fit_labels),
        "validation_labels": list(fold.validation_labels),
        "fit_count": fold.fit_count,
        "query_count": fold.query_count,
        "raw_hits": fold.raw_hits,
        "spectral_hits": fold.spectral_hits,
        "whitening_hits": fold.whitening_hits,
        "sfq_hits": fold.sfq_hits,
        "ledoit_wolf_shrinkage": fold.ledoit_wolf_shrinkage,
        "minimum_within_eigenvalue": fold.minimum_within_eigenvalue,
        "maximum_within_eigenvalue": fold.maximum_within_eigenvalue,
        "bbp_threshold": fold.bbp_threshold,
        "sample_spikes": list(fold.sample_spikes),
        "retained_spikes": list(fold.retained_spikes),
        "gains": list(fold.gains),
        "reliable_rank": fold.reliable_rank,
        "spectral_projection_sha256": fold.spectral_projection_sha256,
        "whitening_projection_sha256": fold.whitening_projection_sha256,
        "sfq_projection_sha256": fold.sfq_projection_sha256,
    }


def _result_mapping(result: SFQResult) -> dict[str, object]:
    return {
        "schema": result.schema,
        "claim_eligible": result.claim_eligible,
        "official_test_access": result.official_test_access,
        "source_manifest_sha256": result.source_manifest_sha256,
        "ordered_example_ids_sha256": result.ordered_example_ids_sha256,
        "feature_matrix_sha256": result.feature_matrix_sha256,
        "input_dimensions": result.input_dimensions,
        "output_dimensions": result.output_dimensions,
        "fold_count": result.fold_count,
        "fold_schedule_sha256": result.fold_schedule_sha256,
        "query_count": result.query_count,
        "raw_hits": result.raw_hits,
        "spectral_hits": result.spectral_hits,
        "whitening_hits": result.whitening_hits,
        "sfq_hits": result.sfq_hits,
        "raw_recall_ppm": result.raw_recall_ppm,
        "spectral_recall_ppm": result.spectral_recall_ppm,
        "whitening_recall_ppm": result.whitening_recall_ppm,
        "sfq_recall_ppm": result.sfq_recall_ppm,
        "sfq_minus_whitening_ppm": result.sfq_minus_whitening_ppm,
        "valid": result.valid,
        "passed": result.passed,
        "folds": [_fold_mapping(fold) for fold in result.folds],
    }


def run_sfq_fold_diagnostic(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    split_authority: FeatureSplitAuthority,
    output_dimensions: int,
    fold_count: int = 4,
) -> bytes:
    """Fit on optimization fold complements and score only held-out optimization classes."""

    schedule = build_sfq_fold_schedule(
        features,
        labels,
        split_authority,
        fold_count=fold_count,
    )
    folds = []
    for fold in schedule.folds:
        fit_label_tensor = torch.tensor(fold.fit_labels, dtype=torch.int64)
        validation_label_tensor = torch.tensor(fold.validation_labels, dtype=torch.int64)
        fit_mask = torch.isin(labels, fit_label_tensor)
        validation_mask = torch.isin(labels, validation_label_tensor)
        fit_features = features[fit_mask].contiguous()
        fit_labels = labels[fit_mask].contiguous()
        validation_features = features[validation_mask].contiguous()
        validation_labels = labels[validation_mask].contiguous()
        projection = fit_sfq_projection(
            fit_features,
            fit_labels,
            output_dimensions=output_dimensions,
        )
        spectral_weight = _spectral_projection(
            fit_features,
            output_dimensions=output_dimensions,
        )
        raw_hits, query_count = _recall_at_one_hits(validation_features, validation_labels)
        spectral_hits, spectral_queries = _recall_at_one_hits(
            validation_features @ spectral_weight.T,
            validation_labels,
        )
        whitening_hits, whitening_queries = _recall_at_one_hits(
            validation_features @ projection.whitening_weight.T,
            validation_labels,
        )
        sfq_hits, sfq_queries = _recall_at_one_hits(
            validation_features @ projection.weight.T,
            validation_labels,
        )
        if (
            spectral_queries != query_count
            or whitening_queries != query_count
            or sfq_queries != query_count
        ):
            raise ValueError("SFQ fold query authority differs")
        folds.append(
            SFQFoldEvidence(
                ordinal=fold.ordinal,
                fit_labels=fold.fit_labels,
                validation_labels=fold.validation_labels,
                fit_count=fit_features.shape[0],
                query_count=query_count,
                raw_hits=raw_hits,
                spectral_hits=spectral_hits,
                whitening_hits=whitening_hits,
                sfq_hits=sfq_hits,
                ledoit_wolf_shrinkage=projection.ledoit_wolf_shrinkage,
                minimum_within_eigenvalue=projection.minimum_within_eigenvalue,
                maximum_within_eigenvalue=projection.maximum_within_eigenvalue,
                bbp_threshold=projection.bbp_threshold,
                sample_spikes=projection.sample_spikes,
                retained_spikes=projection.retained_spikes,
                gains=projection.gains,
                reliable_rank=projection.reliable_rank,
                spectral_projection_sha256=_projection_sha256("spectral", spectral_weight),
                whitening_projection_sha256=_projection_sha256(
                    "whitening", projection.whitening_weight
                ),
                sfq_projection_sha256=_projection_sha256("sfq", projection.weight),
            )
        )
    fold_evidence = tuple(folds)
    query_count = sum(fold.query_count for fold in fold_evidence)
    raw_hits = sum(fold.raw_hits for fold in fold_evidence)
    spectral_hits = sum(fold.spectral_hits for fold in fold_evidence)
    whitening_hits = sum(fold.whitening_hits for fold in fold_evidence)
    sfq_hits = sum(fold.sfq_hits for fold in fold_evidence)
    recalls = tuple(
        hits * 1_000_000 // query_count
        for hits in (raw_hits, spectral_hits, whitening_hits, sfq_hits)
    )
    result = SFQResult(
        schema="sfora-siglip-sfq-fold-diagnostic-v1",
        claim_eligible=False,
        official_test_access=False,
        source_manifest_sha256=split_authority.source_manifest_sha256,
        ordered_example_ids_sha256=split_authority.ordered_example_ids_sha256,
        feature_matrix_sha256=split_authority.feature_matrix_sha256,
        input_dimensions=features.shape[1],
        output_dimensions=output_dimensions,
        fold_count=fold_count,
        fold_schedule_sha256=schedule.sha256,
        query_count=query_count,
        raw_hits=raw_hits,
        spectral_hits=spectral_hits,
        whitening_hits=whitening_hits,
        sfq_hits=sfq_hits,
        raw_recall_ppm=recalls[0],
        spectral_recall_ppm=recalls[1],
        whitening_recall_ppm=recalls[2],
        sfq_recall_ppm=recalls[3],
        sfq_minus_whitening_ppm=recalls[3] - recalls[2],
        valid=True,
        passed=(
            recalls[3] - recalls[2] >= 2_000 and sfq_hits >= raw_hits and sfq_hits >= spectral_hits
        ),
        folds=fold_evidence,
    )
    raw = _canonical_bytes(_result_mapping(result))
    validate_sfq_result_bytes(raw)
    return raw


def _exact_keys(value: object, expected: set[str], *, error: str) -> dict[str, object]:
    if type(value) is not dict or set(value) != expected:
        raise ValueError(error)
    return cast(dict[str, object], value)


def _integer(value: object, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError("SFQ integer evidence differs")
    return value


def _finite_float(value: object) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError("SFQ float evidence differs")
    return value


def _hex_digest(value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("SFQ digest evidence differs")
    return value


def _integer_tuple(value: object) -> tuple[int, ...]:
    if type(value) is not list or any(type(item) is not int for item in value):
        raise ValueError("SFQ label evidence differs")
    return tuple(cast(list[int], value))


def _float_tuple(value: object) -> tuple[float, ...]:
    if type(value) is not list:
        raise ValueError("SFQ spectral evidence differs")
    return tuple(_finite_float(item) for item in value)


def _parse_fold(value: object, *, input_dimensions: int) -> SFQFoldEvidence:
    mapping = _exact_keys(
        value,
        {
            "ordinal",
            "fit_labels",
            "validation_labels",
            "fit_count",
            "query_count",
            "raw_hits",
            "spectral_hits",
            "whitening_hits",
            "sfq_hits",
            "ledoit_wolf_shrinkage",
            "minimum_within_eigenvalue",
            "maximum_within_eigenvalue",
            "bbp_threshold",
            "sample_spikes",
            "retained_spikes",
            "gains",
            "reliable_rank",
            "spectral_projection_sha256",
            "whitening_projection_sha256",
            "sfq_projection_sha256",
        },
        error="SFQ fold schema differs",
    )
    fit_labels = _integer_tuple(mapping["fit_labels"])
    validation_labels = _integer_tuple(mapping["validation_labels"])
    fit_count = _integer(mapping["fit_count"], minimum=1)
    query_count = _integer(mapping["query_count"], minimum=2)
    hits = tuple(
        _integer(mapping[name])
        for name in ("raw_hits", "spectral_hits", "whitening_hits", "sfq_hits")
    )
    shrinkage = _finite_float(mapping["ledoit_wolf_shrinkage"])
    minimum_eigenvalue = _finite_float(mapping["minimum_within_eigenvalue"])
    maximum_eigenvalue = _finite_float(mapping["maximum_within_eigenvalue"])
    threshold = _finite_float(mapping["bbp_threshold"])
    sample_spikes = _float_tuple(mapping["sample_spikes"])
    retained_spikes = _float_tuple(mapping["retained_spikes"])
    gains = _float_tuple(mapping["gains"])
    reliable_rank = _integer(mapping["reliable_rank"], minimum=1)
    gamma = input_dimensions / len(fit_labels) if fit_labels else math.nan
    expected_threshold = (1.0 + math.sqrt(gamma)) ** 2
    if (
        not fit_labels
        or not validation_labels
        or tuple(sorted(set(fit_labels))) != fit_labels
        or tuple(sorted(set(validation_labels))) != validation_labels
        or not set(fit_labels).isdisjoint(validation_labels)
        or fit_count < 2 * len(fit_labels)
        or query_count < 2 * len(validation_labels)
        or any(hit > query_count for hit in hits)
        or not 0.0 <= shrinkage <= 1.0
        or minimum_eigenvalue <= 0.0
        or maximum_eigenvalue < minimum_eigenvalue
        or not math.isclose(threshold, expected_threshold, rel_tol=0.0, abs_tol=1.0e-12)
        or tuple(sorted(sample_spikes, reverse=True)) != sample_spikes
        or reliable_rank != len(retained_spikes)
        or reliable_rank != len(gains)
        or retained_spikes != sample_spikes[:reliable_rank]
        or any(spike <= threshold for spike in retained_spikes)
        or any(spike > threshold for spike in sample_spikes[reliable_rank:])
    ):
        raise ValueError("SFQ fold relation differs")
    expected_gains = []
    for sample_spike in retained_spikes:
        radicand = (sample_spike - 1.0 - gamma) ** 2 - 4.0 * gamma
        theta = (sample_spike - 1.0 - gamma + math.sqrt(max(0.0, radicand))) / 2.0
        alignment = (1.0 - gamma / theta**2) / (1.0 + gamma / theta)
        expected_gains.append(alignment * theta)
    if any(
        not math.isclose(observed, expected, rel_tol=1.0e-12, abs_tol=1.0e-12)
        for observed, expected in zip(gains, expected_gains, strict=True)
    ):
        raise ValueError("SFQ gain evidence differs")
    return SFQFoldEvidence(
        ordinal=_integer(mapping["ordinal"]),
        fit_labels=fit_labels,
        validation_labels=validation_labels,
        fit_count=fit_count,
        query_count=query_count,
        raw_hits=hits[0],
        spectral_hits=hits[1],
        whitening_hits=hits[2],
        sfq_hits=hits[3],
        ledoit_wolf_shrinkage=shrinkage,
        minimum_within_eigenvalue=minimum_eigenvalue,
        maximum_within_eigenvalue=maximum_eigenvalue,
        bbp_threshold=threshold,
        sample_spikes=sample_spikes,
        retained_spikes=retained_spikes,
        gains=gains,
        reliable_rank=reliable_rank,
        spectral_projection_sha256=_hex_digest(mapping["spectral_projection_sha256"]),
        whitening_projection_sha256=_hex_digest(mapping["whitening_projection_sha256"]),
        sfq_projection_sha256=_hex_digest(mapping["sfq_projection_sha256"]),
    )


def validate_sfq_result_bytes(raw: bytes) -> SFQResult:
    """Parse canonical bytes and independently reconstruct every aggregate and gate."""

    if type(raw) is not bytes:
        raise ValueError("SFQ result bytes differ")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("SFQ result is not JSON") from error
    mapping = _exact_keys(
        value,
        {
            "schema",
            "claim_eligible",
            "official_test_access",
            "source_manifest_sha256",
            "ordered_example_ids_sha256",
            "feature_matrix_sha256",
            "input_dimensions",
            "output_dimensions",
            "fold_count",
            "fold_schedule_sha256",
            "query_count",
            "raw_hits",
            "spectral_hits",
            "whitening_hits",
            "sfq_hits",
            "raw_recall_ppm",
            "spectral_recall_ppm",
            "whitening_recall_ppm",
            "sfq_recall_ppm",
            "sfq_minus_whitening_ppm",
            "valid",
            "passed",
            "folds",
        },
        error="SFQ result schema differs",
    )
    input_dimensions = _integer(mapping["input_dimensions"], minimum=2)
    output_dimensions = _integer(mapping["output_dimensions"], minimum=1)
    fold_count = _integer(mapping["fold_count"], minimum=2)
    if type(mapping["folds"]) is not list:
        raise ValueError("SFQ fold bundle differs")
    folds = tuple(
        _parse_fold(fold, input_dimensions=input_dimensions)
        for fold in cast(list[object], mapping["folds"])
    )
    schedule_folds = tuple(
        SFQFold(
            ordinal=fold.ordinal,
            fit_labels=fold.fit_labels,
            validation_labels=fold.validation_labels,
        )
        for fold in folds
    )
    source_manifest_sha256 = _hex_digest(mapping["source_manifest_sha256"])
    ordered_example_ids_sha256 = _hex_digest(mapping["ordered_example_ids_sha256"])
    feature_matrix_sha256 = _hex_digest(mapping["feature_matrix_sha256"])
    expected_schedule_sha256 = _schedule_sha256_from_hashes(
        folds=schedule_folds,
        source_manifest_sha256=source_manifest_sha256,
        ordered_example_ids_sha256=ordered_example_ids_sha256,
        feature_matrix_sha256=feature_matrix_sha256,
    )
    query_count = sum(fold.query_count for fold in folds)
    totals = tuple(
        sum(getattr(fold, name) for fold in folds)
        for name in ("raw_hits", "spectral_hits", "whitening_hits", "sfq_hits")
    )
    recalls = tuple(total * 1_000_000 // query_count for total in totals)
    passed = recalls[3] - recalls[2] >= 2_000 and totals[3] >= totals[0] and totals[3] >= totals[1]
    all_validation_labels = sorted(label for fold in folds for label in fold.validation_labels)
    if (
        mapping["schema"] != "sfora-siglip-sfq-fold-diagnostic-v1"
        or type(mapping["claim_eligible"]) is not bool
        or mapping["claim_eligible"]
        or type(mapping["official_test_access"]) is not bool
        or mapping["official_test_access"]
        or output_dimensions > input_dimensions
        or fold_count != len(folds)
        or tuple(fold.ordinal for fold in folds) != tuple(range(fold_count))
        or all_validation_labels != list(range(len(all_validation_labels)))
        or any(
            set(fold.fit_labels) != set(all_validation_labels).difference(fold.validation_labels)
            for fold in folds
        )
        or _hex_digest(mapping["fold_schedule_sha256"]) != expected_schedule_sha256
        or _integer(mapping["query_count"], minimum=1) != query_count
        or tuple(
            _integer(mapping[name])
            for name in ("raw_hits", "spectral_hits", "whitening_hits", "sfq_hits")
        )
        != totals
        or tuple(
            _integer(mapping[name])
            for name in (
                "raw_recall_ppm",
                "spectral_recall_ppm",
                "whitening_recall_ppm",
                "sfq_recall_ppm",
            )
        )
        != recalls
        or type(mapping["sfq_minus_whitening_ppm"]) is not int
        or mapping["sfq_minus_whitening_ppm"] != recalls[3] - recalls[2]
        or type(mapping["valid"]) is not bool
        or not mapping["valid"]
        or type(mapping["passed"]) is not bool
        or mapping["passed"] is not passed
    ):
        raise ValueError("SFQ result relation differs")
    result = SFQResult(
        schema=cast(str, mapping["schema"]),
        claim_eligible=False,
        official_test_access=False,
        source_manifest_sha256=source_manifest_sha256,
        ordered_example_ids_sha256=ordered_example_ids_sha256,
        feature_matrix_sha256=feature_matrix_sha256,
        input_dimensions=input_dimensions,
        output_dimensions=output_dimensions,
        fold_count=fold_count,
        fold_schedule_sha256=expected_schedule_sha256,
        query_count=query_count,
        raw_hits=totals[0],
        spectral_hits=totals[1],
        whitening_hits=totals[2],
        sfq_hits=totals[3],
        raw_recall_ppm=recalls[0],
        spectral_recall_ppm=recalls[1],
        whitening_recall_ppm=recalls[2],
        sfq_recall_ppm=recalls[3],
        sfq_minus_whitening_ppm=recalls[3] - recalls[2],
        valid=True,
        passed=passed,
        folds=folds,
    )
    if _canonical_bytes(_result_mapping(result)) != raw:
        raise ValueError("SFQ result bytes are not canonical")
    return result
