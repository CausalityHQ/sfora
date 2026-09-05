"""Descriptor-only capacity diagnostics for SigLIP gallery compatibility."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

import torch
from torch import nn
from torch.nn import functional as F


def _validated_pair(
    student: torch.Tensor, teacher: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if (
        type(student) is not torch.Tensor
        or type(teacher) is not torch.Tensor
        or student.device.type != "cpu"
        or teacher.device.type != "cpu"
        or student.dtype != torch.float32
        or teacher.dtype != torch.float32
        or student.ndim != 2
        or teacher.ndim != 2
        or student.shape != teacher.shape
        or student.shape[0] < 2
        or student.shape[1] < 2
        or not bool(torch.isfinite(student).all())
        or not bool(torch.isfinite(teacher).all())
        or bool((torch.linalg.vector_norm(student, dim=1) <= 0).any())
        or bool((torch.linalg.vector_norm(teacher, dim=1) <= 0).any())
    ):
        raise ValueError("compatibility descriptor authority differs")
    return student, teacher


def compatibility_folds(labels: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    """Return the registered three-way class folds."""

    if (
        type(labels) is not tuple
        or len(labels) != 39
        or any(type(label) is not int for label in labels)
        or set(labels) != set(range(39))
    ):
        raise ValueError("compatibility class authority differs")
    ranked = sorted(
        (
            hashlib.sha256(
                b"sfora-compatibility-capacity-fold-v1\0" + str(label).encode("ascii")
            ).digest(),
            label,
        )
        for label in labels
    )
    return tuple(
        tuple(label for _, label in ranked[start : start + 13])
        for start in range(0, 39, 13)
    )


@dataclass(frozen=True, slots=True)
class CompatibilityRetrievalEvidence:
    """Exact compatibility retrieval and cross-space diagnostic evidence."""

    hits: tuple[bool, ...]
    average_precisions: tuple[float, ...]
    labels: tuple[int, ...]
    paired_cosines: tuple[float, ...]
    cross_score_mse: float
    top10_overlaps: tuple[float, ...]
    hub_counts: tuple[int, ...]

    @property
    def micro_r1(self) -> float:
        return sum(self.hits) / len(self.hits)

    @property
    def micro_map_at_r(self) -> float:
        return math.fsum(self.average_precisions) / len(self.average_precisions)

    @property
    def class_macro_r1(self) -> float:
        classes = sorted(set(self.labels))
        return math.fsum(
            sum(hit for hit, label in zip(self.hits, self.labels, strict=True) if label == cls)
            / self.labels.count(cls)
            for cls in classes
        ) / len(classes)

    @property
    def class_macro_map_at_r(self) -> float:
        classes = sorted(set(self.labels))
        return math.fsum(
            math.fsum(
                value
                for value, label in zip(
                    self.average_precisions, self.labels, strict=True
                )
                if label == cls
            )
            / self.labels.count(cls)
            for cls in classes
        ) / len(classes)


def _valid_descriptor_bank(value: object, rows: int, dimensions: int | None = None) -> bool:
    return (
        type(value) is torch.Tensor
        and value.device.type == "cpu"
        and value.dtype == torch.float32
        and value.ndim == 2
        and value.shape[0] == rows
        and value.shape[1] >= 1
        and (dimensions is None or value.shape[1] == dimensions)
        and bool(torch.isfinite(value).all())
        and not bool((torch.linalg.vector_norm(value, dim=1) <= 0).any())
    )


def compatibility_retrieval_evidence(
    query: torch.Tensor,
    gallery: torch.Tensor,
    *,
    query_ids: tuple[str, ...],
    gallery_ids: tuple[str, ...],
    query_labels: tuple[int, ...],
    gallery_labels: tuple[int, ...],
    reference_query: torch.Tensor,
    reference_gallery: torch.Tensor,
) -> CompatibilityRetrievalEvidence:
    """Recompute identity-bound cross-space retrieval and diagnostic vectors."""

    rows = len(query_ids)
    if (
        type(query_ids) is not tuple
        or type(gallery_ids) is not tuple
        or rows < 2
        or len(set(query_ids)) != rows
        or len(gallery_ids) != rows
        or len(set(gallery_ids)) != rows
        or set(query_ids) != set(gallery_ids)
        or any(type(value) is not str or not value for value in (*query_ids, *gallery_ids))
        or type(query_labels) is not tuple
        or type(gallery_labels) is not tuple
        or len(query_labels) != rows
        or len(gallery_labels) != rows
        or any(type(value) is not int or value < 0 for value in (*query_labels, *gallery_labels))
        or not _valid_descriptor_bank(query, rows)
        or not _valid_descriptor_bank(gallery, rows, query.shape[1])
        or not _valid_descriptor_bank(reference_query, rows, query.shape[1])
        or not _valid_descriptor_bank(reference_gallery, rows, query.shape[1])
    ):
        raise ValueError("compatibility retrieval authority differs")
    gallery_by_id = dict(zip(gallery_ids, range(rows), strict=True))
    order = torch.tensor([gallery_by_id[value] for value in query_ids], dtype=torch.int64)
    canonical_gallery_labels = tuple(gallery_labels[index] for index in order.tolist())
    if canonical_gallery_labels != query_labels:
        raise ValueError("compatibility retrieval authority differs")
    current_query = F.normalize(query, dim=1)
    current_gallery = F.normalize(gallery[order], dim=1)
    expected_query = F.normalize(reference_query, dim=1)
    expected_gallery = F.normalize(reference_gallery[order], dim=1)
    scores = current_query @ current_gallery.T
    expected_scores = expected_query @ expected_gallery.T
    paired_cosines = tuple(float(value) for value in scores.diag())
    mask = ~torch.eye(rows, dtype=torch.bool)
    cross_score_mse = float((scores[mask] - expected_scores[mask]).double().square().mean())
    scores = scores.clone()
    expected_scores = expected_scores.clone()
    scores.diagonal().fill_(-torch.inf)
    expected_scores.diagonal().fill_(-torch.inf)
    ranked = torch.argsort(scores, dim=1, descending=True, stable=True)
    expected_ranked = torch.argsort(expected_scores, dim=1, descending=True, stable=True)
    counts = {label: query_labels.count(label) for label in set(query_labels)}
    if not counts or min(counts.values()) < 2:
        raise ValueError("compatibility retrieval authority differs")
    label_tensor = torch.tensor(query_labels)
    hits: list[bool] = []
    average_precisions: list[float] = []
    neighbor_count = min(10, rows - 1)
    top10_overlaps: list[float] = []
    hubs = [0] * rows
    for row in range(rows):
        hits.append(query_labels[int(ranked[row, 0])] == query_labels[row])
        retained = ranked[row, : counts[query_labels[row]] - 1]
        relevant = label_tensor[retained] == query_labels[row]
        hit_count = 0
        terms: list[float] = []
        for rank, hit in enumerate(relevant.tolist(), 1):
            if hit:
                hit_count += 1
                terms.append(hit_count / rank)
        average_precisions.append(math.fsum(terms) / len(relevant))
        current_top = ranked[row, :neighbor_count].tolist()
        expected_top = set(expected_ranked[row, :neighbor_count].tolist())
        top10_overlaps.append(len(set(current_top) & expected_top) / neighbor_count)
        for index in current_top:
            hubs[index] += 1
    return CompatibilityRetrievalEvidence(
        hits=tuple(hits),
        average_precisions=tuple(average_precisions),
        labels=query_labels,
        paired_cosines=paired_cosines,
        cross_score_mse=cross_score_mse,
        top10_overlaps=tuple(top10_overlaps),
        hub_counts=tuple(hubs),
    )


def csls_scores(
    query: torch.Tensor, gallery: torch.Tensor, *, neighbors: int = 10
) -> torch.Tensor:
    """Return exact cross-domain similarity local scaling scores."""

    rows = query.shape[0] if type(query) is torch.Tensor and query.ndim == 2 else 0
    if (
        not _valid_descriptor_bank(query, rows)
        or not _valid_descriptor_bank(gallery, rows, query.shape[1])
        or type(neighbors) is not int
        or not 1 <= neighbors <= rows
    ):
        raise ValueError("compatibility CSLS authority differs")
    cosine = F.normalize(query, dim=1) @ F.normalize(gallery, dim=1).T
    query_density = torch.topk(cosine, neighbors, dim=1).values.mean(dim=1)
    gallery_density = torch.topk(cosine, neighbors, dim=0).values.mean(dim=0)
    return (2.0 * cosine - query_density[:, None] - gallery_density[None, :]).contiguous()


def select_compatibility_finalist(
    fold_results: Mapping[str, tuple[tuple[float, float], ...]],
) -> str:
    """Choose the fitting-only finalist from worst fold-direction macro mAP."""

    names = (
        "affine-0.0001",
        "affine-0.01",
        "affine-1",
        "teacher-anchored-residual",
    )
    if type(fold_results) is not dict or set(fold_results) != set(names):
        raise ValueError("compatibility finalist authority differs")
    scores: dict[str, float] = {}
    for name in names:
        folds = fold_results[name]
        if (
            type(folds) is not tuple
            or len(folds) != 3
            or any(
                type(pair) is not tuple
                or len(pair) != 2
                or any(
                    type(value) is not float
                    or not math.isfinite(value)
                    or not 0 <= value <= 1
                    for value in pair
                )
                for pair in folds
            )
        ):
            raise ValueError("compatibility finalist authority differs")
        scores[name] = min(value for pair in folds for value in pair)
    preference = {
        "teacher-anchored-residual": 0,
        "affine-0.0001": 1,
        "affine-0.01": 2,
        "affine-1": 3,
    }
    return max(names, key=lambda name: (scores[name], preference[name]))


@dataclass(frozen=True, slots=True)
class CompatibilityDecisionMetrics:
    """Plain-cosine compatibility metrics used by the terminal decision."""

    forward_r1: float
    forward_map_at_r: float
    reverse_r1: float
    reverse_map_at_r: float
    self_r1: float
    self_map_at_r: float

    def __post_init__(self) -> None:
        if any(
            type(value) is not float or not math.isfinite(value) or not 0.0 <= value <= 1.0
            for value in (
                self.forward_r1,
                self.forward_map_at_r,
                self.reverse_r1,
                self.reverse_map_at_r,
                self.self_r1,
                self.self_map_at_r,
            )
        ):
            raise ValueError("compatibility decision authority differs")


def _passes(metrics: CompatibilityDecisionMetrics, r1: float, map_at_r: float) -> bool:
    return (
        metrics.forward_r1 >= r1
        and metrics.forward_map_at_r >= map_at_r
        and metrics.reverse_r1 >= r1
        and metrics.reverse_map_at_r >= map_at_r
        and metrics.self_r1 >= 0.99
        and metrics.self_map_at_r >= 0.96
    )


def classify_compatibility_capacity(
    finalist: CompatibilityDecisionMetrics, oracle: CompatibilityDecisionMetrics
) -> str:
    """Classify post-hoc capacity without consulting CSLS evidence."""

    if (
        type(finalist) is not CompatibilityDecisionMetrics
        or type(oracle) is not CompatibilityDecisionMetrics
    ):
        raise ValueError("compatibility decision authority differs")
    if _passes(finalist, 0.97, 0.95):
        return "posthoc-passed"
    if _passes(oracle, 0.90, 0.90):
        return "coverage-failure"
    if (
        oracle.forward_r1 < 0.80
        or oracle.forward_map_at_r < 0.80
        or oracle.reverse_r1 < 0.80
        or oracle.reverse_map_at_r < 0.80
    ):
        return "information-failure"
    return "ambiguous-capacity"


def hubness_present(
    cosine_r1: tuple[float, float], csls_r1: tuple[float, float]
) -> bool:
    """Report whether CSLS provides a registered five-point directional lift."""

    if any(
        type(pair) is not tuple
        or len(pair) != 2
        or any(
            type(value) is not float
            or not math.isfinite(value)
            or not 0 <= value <= 1
            for value in pair
        )
        for pair in (cosine_r1, csls_r1)
    ):
        raise ValueError("compatibility hubness authority differs")
    return any(
        after - before >= 0.05
        for before, after in zip(cosine_r1, csls_r1, strict=True)
    )


def _hex_digest(value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("compatibility capacity result digest differs")
    return value


def _evidence_mapping(value: CompatibilityRetrievalEvidence) -> dict[str, object]:
    if type(value) is not CompatibilityRetrievalEvidence:
        raise ValueError("compatibility capacity result evidence differs")
    return {
        "hits": list(value.hits),
        "average_precisions": list(value.average_precisions),
        "labels": list(value.labels),
        "paired_cosines": list(value.paired_cosines),
        "cross_score_mse": value.cross_score_mse,
        "top10_overlaps": list(value.top10_overlaps),
        "hub_counts": list(value.hub_counts),
        "micro_r1": value.micro_r1,
        "micro_map_at_r": value.micro_map_at_r,
        "class_macro_r1": value.class_macro_r1,
        "class_macro_map_at_r": value.class_macro_map_at_r,
    }


def _parse_evidence(value: object) -> CompatibilityRetrievalEvidence:
    keys = {
        "hits",
        "average_precisions",
        "labels",
        "paired_cosines",
        "cross_score_mse",
        "top10_overlaps",
        "hub_counts",
        "micro_r1",
        "micro_map_at_r",
        "class_macro_r1",
        "class_macro_map_at_r",
    }
    if type(value) is not dict or set(value) != keys:
        raise ValueError("compatibility capacity result evidence differs")
    item = cast(dict[str, object], value)
    sequences = (
        item["hits"],
        item["average_precisions"],
        item["labels"],
        item["paired_cosines"],
        item["top10_overlaps"],
        item["hub_counts"],
    )
    if any(type(sequence) is not list for sequence in sequences):
        raise ValueError("compatibility capacity result evidence differs")
    rows = len(cast(list[object], item["hits"]))
    if rows < 2 or any(len(cast(list[object], sequence)) != rows for sequence in sequences):
        raise ValueError("compatibility capacity result evidence differs")
    hits = cast(list[object], item["hits"])
    aps = cast(list[object], item["average_precisions"])
    labels = cast(list[object], item["labels"])
    paired = cast(list[object], item["paired_cosines"])
    overlaps = cast(list[object], item["top10_overlaps"])
    hubs = cast(list[object], item["hub_counts"])
    summaries = (
        item["cross_score_mse"],
        item["micro_r1"],
        item["micro_map_at_r"],
        item["class_macro_r1"],
        item["class_macro_map_at_r"],
    )
    if (
        any(type(entry) is not bool for entry in hits)
        or any(
            type(entry) is not float
            or not math.isfinite(entry)
            or not 0 <= entry <= 1
            for entry in aps
        )
        or any(type(entry) is not int or entry < 0 for entry in labels)
        or any(
            type(entry) is not float
            or not math.isfinite(entry)
            or not -1 <= entry <= 1
            for entry in paired
        )
        or any(
            type(entry) is not float
            or not math.isfinite(entry)
            or not 0 <= entry <= 1
            for entry in overlaps
        )
        or any(type(entry) is not int or entry < 0 for entry in hubs)
        or any(type(entry) is not float or not math.isfinite(entry) for entry in summaries)
        or cast(float, item["cross_score_mse"]) < 0
    ):
        raise ValueError("compatibility capacity result evidence differs")
    evidence = CompatibilityRetrievalEvidence(
        hits=tuple(cast(list[bool], hits)),
        average_precisions=tuple(cast(list[float], aps)),
        labels=tuple(cast(list[int], labels)),
        paired_cosines=tuple(cast(list[float], paired)),
        cross_score_mse=cast(float, item["cross_score_mse"]),
        top10_overlaps=tuple(cast(list[float], overlaps)),
        hub_counts=tuple(cast(list[int], hubs)),
    )
    expected = {
        "micro_r1": evidence.micro_r1,
        "micro_map_at_r": evidence.micro_map_at_r,
        "class_macro_r1": evidence.class_macro_r1,
        "class_macro_map_at_r": evidence.class_macro_map_at_r,
    }
    if any(item[name] != result for name, result in expected.items()):
        raise ValueError("compatibility capacity result evidence differs")
    return evidence


def _decision_metrics(
    cells: Mapping[str, CompatibilityRetrievalEvidence], prefix: str
) -> CompatibilityDecisionMetrics:
    forward = cells[f"{prefix}-forward"]
    reverse = cells[f"{prefix}-reverse"]
    self_cell = cells[f"{prefix}-self"]
    return CompatibilityDecisionMetrics(
        forward_r1=forward.micro_r1,
        forward_map_at_r=forward.micro_map_at_r,
        reverse_r1=reverse.micro_r1,
        reverse_map_at_r=reverse.micro_map_at_r,
        self_r1=self_cell.micro_r1,
        self_map_at_r=self_cell.micro_map_at_r,
    )


def build_compatibility_capacity_result(
    *,
    checkpoint_sha256: str,
    descriptor_artifact_sha256: str,
    fold_results: Mapping[str, tuple[tuple[float, float], ...]],
    cells: Mapping[str, CompatibilityRetrievalEvidence],
    cosine_r1: tuple[float, float],
    csls_r1: tuple[float, float],
) -> bytes:
    """Build canonical, claim-ineligible compatibility-capacity evidence."""

    finalist = select_compatibility_finalist(fold_results)
    cell_names = {
        "finalist-forward",
        "finalist-reverse",
        "finalist-self",
        "oracle-forward",
        "oracle-reverse",
        "oracle-self",
    }
    if type(cells) is not dict or set(cells) != cell_names:
        raise ValueError("compatibility capacity result cell inventory differs")
    parsed_cells = {name: _evidence_mapping(cells[name]) for name in sorted(cell_names)}
    finalist_metrics = _decision_metrics(cells, "finalist")
    oracle_metrics = _decision_metrics(cells, "oracle")
    result = {
        "schema": "sfora-siglip-compatibility-capacity-v1",
        "claim_eligible": False,
        "external_evaluation_access": False,
        "checkpoint_sha256": _hex_digest(checkpoint_sha256),
        "descriptor_artifact_sha256": _hex_digest(descriptor_artifact_sha256),
        "fold_results": {
            name: [list(pair) for pair in fold_results[name]] for name in sorted(fold_results)
        },
        "finalist": finalist,
        "cells": parsed_cells,
        "cosine_r1": list(cosine_r1),
        "csls_r1": list(csls_r1),
        "classification": classify_compatibility_capacity(finalist_metrics, oracle_metrics),
        "hubness_present": hubness_present(cosine_r1, csls_r1),
    }
    raw = (
        json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode()
    validate_compatibility_capacity_result_bytes(raw)
    return raw


def validate_compatibility_capacity_result_bytes(raw: bytes) -> dict[str, object]:
    """Validate canonical bytes and independently reconstruct every decision."""

    try:
        value = json.loads(raw)
    except (TypeError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("compatibility capacity result is not JSON") from error
    keys = {
        "schema",
        "claim_eligible",
        "external_evaluation_access",
        "checkpoint_sha256",
        "descriptor_artifact_sha256",
        "fold_results",
        "finalist",
        "cells",
        "cosine_r1",
        "csls_r1",
        "classification",
        "hubness_present",
    }
    if type(raw) is not bytes or type(value) is not dict or set(value) != keys:
        raise ValueError("compatibility capacity result schema differs")
    result = cast(dict[str, object], value)
    canonical = (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode()
    if raw != canonical:
        raise ValueError("compatibility capacity result bytes differ")
    if (
        result["schema"] != "sfora-siglip-compatibility-capacity-v1"
        or type(result["claim_eligible"]) is not bool
        or result["claim_eligible"] is not False
        or type(result["external_evaluation_access"]) is not bool
        or result["external_evaluation_access"] is not False
    ):
        raise ValueError("compatibility capacity result authority differs")
    _hex_digest(result["checkpoint_sha256"])
    _hex_digest(result["descriptor_artifact_sha256"])
    fold_value = result["fold_results"]
    if type(fold_value) is not dict:
        raise ValueError("compatibility capacity result folds differ")
    folds: dict[str, tuple[tuple[float, float], ...]] = {}
    for name, entries in fold_value.items():
        if type(name) is not str or type(entries) is not list:
            raise ValueError("compatibility capacity result folds differ")
        converted: list[tuple[float, float]] = []
        for entry in entries:
            if type(entry) is not list or len(entry) != 2:
                raise ValueError("compatibility capacity result folds differ")
            converted.append(tuple(entry))  # type: ignore[arg-type]
        folds[name] = tuple(converted)
    expected_finalist = select_compatibility_finalist(folds)
    if type(result["finalist"]) is not str or result["finalist"] != expected_finalist:
        raise ValueError("compatibility capacity result finalist differs")
    cell_value = result["cells"]
    cell_names = {
        "finalist-forward",
        "finalist-reverse",
        "finalist-self",
        "oracle-forward",
        "oracle-reverse",
        "oracle-self",
    }
    if type(cell_value) is not dict or set(cell_value) != cell_names:
        raise ValueError("compatibility capacity result cells differ")
    cells = {name: _parse_evidence(cell_value[name]) for name in cell_names}
    if len({len(evidence.hits) for evidence in cells.values()}) != 1:
        raise ValueError("compatibility capacity result cells differ")
    cosine = result["cosine_r1"]
    csls = result["csls_r1"]
    if type(cosine) is not list or type(csls) is not list:
        raise ValueError("compatibility capacity result hubness differs")
    cosine_pair = tuple(cosine)
    csls_pair = tuple(csls)
    expected_hubness = hubness_present(cosine_pair, csls_pair)  # type: ignore[arg-type]
    expected_class = classify_compatibility_capacity(
        _decision_metrics(cells, "finalist"), _decision_metrics(cells, "oracle")
    )
    if (
        type(result["classification"]) is not str
        or result["classification"] != expected_class
        or type(result["hubness_present"]) is not bool
        or result["hubness_present"] is not expected_hubness
    ):
        raise ValueError("compatibility capacity result decision differs")
    return result


@dataclass(frozen=True, slots=True)
class AffineMap:
    """A validated FP64 affine descriptor map with normalized FP32 output."""

    weight: torch.Tensor
    bias: torch.Tensor

    def __post_init__(self) -> None:
        if (
            type(self.weight) is not torch.Tensor
            or type(self.bias) is not torch.Tensor
            or self.weight.device.type != "cpu"
            or self.bias.device.type != "cpu"
            or self.weight.dtype != torch.float64
            or self.bias.dtype != torch.float64
            or self.weight.ndim != 2
            or self.weight.shape[0] != self.weight.shape[1]
            or self.bias.shape != (self.weight.shape[0],)
            or not bool(torch.isfinite(self.weight).all())
            or not bool(torch.isfinite(self.bias).all())
        ):
            raise ValueError("compatibility affine authority differs")

    def apply(self, descriptors: torch.Tensor) -> torch.Tensor:
        """Apply this map and return normalized CPU FP32 descriptors."""

        if (
            type(descriptors) is not torch.Tensor
            or descriptors.device.type != "cpu"
            or descriptors.dtype != torch.float32
            or descriptors.ndim != 2
            or descriptors.shape[0] < 2
            or descriptors.shape[1] != self.weight.shape[0]
            or not bool(torch.isfinite(descriptors).all())
            or bool((torch.linalg.vector_norm(descriptors, dim=1) <= 0).any())
        ):
            raise ValueError("compatibility descriptor authority differs")
        mapped = F.normalize(descriptors.double() @ self.weight + self.bias, dim=1)
        if not bool(torch.isfinite(mapped).all()):
            raise ValueError("compatibility affine authority differs")
        return mapped.float().contiguous()


def fit_centered_similarity(student: torch.Tensor, teacher: torch.Tensor) -> AffineMap:
    """Fit the registered centered orthogonal descriptor map."""

    student, teacher = _validated_pair(student, teacher)
    source = student.double()
    target = teacher.double()
    source_mean = source.mean(dim=0)
    target_mean = target.mean(dim=0)
    left, _, right_t = torch.linalg.svd(
        (source - source_mean).T @ (target - target_mean), full_matrices=False
    )
    weight = (left @ right_t).contiguous()
    bias = (target_mean - source_mean @ weight).contiguous()
    return AffineMap(weight, bias)


def fit_regularized_affine(
    student: torch.Tensor, teacher: torch.Tensor, regularization: float
) -> AffineMap:
    """Fit the registered identity-regularized affine ridge map."""

    student, teacher = _validated_pair(student, teacher)
    if type(regularization) is not float or regularization not in {1e-4, 1e-2, 1.0}:
        raise ValueError("compatibility regularization differs")
    source = student.double()
    target = teacher.double()
    rows, dimensions = source.shape
    augmented = torch.cat((source, torch.ones(rows, 1, dtype=torch.float64)), dim=1)
    identity_target = torch.cat(
        (torch.eye(dimensions, dtype=torch.float64), torch.zeros(1, dimensions)), dim=0
    )
    system = augmented.T @ augmented / rows
    system += regularization * torch.eye(dimensions + 1, dtype=torch.float64)
    right = augmented.T @ target / rows + regularization * identity_target
    solution = torch.linalg.solve(system, right)
    return AffineMap(solution[:-1].contiguous(), solution[-1].contiguous())


class CompatibilityResidual(nn.Module):
    """The fixed low-rank residual descriptor adapter."""

    def __init__(
        self,
        dimensions: int,
        *,
        rank: int,
        seed: int,
        device: torch.device | None = None,
    ):
        super().__init__()
        if (
            type(dimensions) is not int
            or dimensions < 2
            or type(rank) is not int
            or rank != 32
            or type(seed) is not int
            or seed < 0
        ):
            raise ValueError("compatibility residual authority differs")
        target = torch.device("cpu") if device is None else device
        with torch.random.fork_rng(devices=[] if target.type == "cpu" else [target]):
            torch.manual_seed(seed)
            self.down = nn.Linear(dimensions, rank, bias=False, dtype=torch.float64, device=target)
            self.up = nn.Linear(rank, dimensions, bias=True, dtype=torch.float64, device=target)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, descriptors: torch.Tensor) -> torch.Tensor:
        """Apply the unnormalized residual so zero initialization is exact identity."""

        return descriptors + self.up(F.gelu(self.down(descriptors)))


@dataclass(frozen=True, slots=True)
class ResidualFit:
    """A frozen fitted residual and its complete optimization evidence."""

    state_dict: dict[str, torch.Tensor]
    losses: dict[str, tuple[float, ...]]
    anchor_ids: tuple[str, ...]
    relational: bool
    seed: int

    def apply(self, descriptors: torch.Tensor) -> torch.Tensor:
        """Apply the frozen residual to normalized CPU FP32 descriptors."""

        if type(descriptors) is not torch.Tensor or descriptors.ndim != 2:
            raise ValueError("compatibility descriptor authority differs")
        dimensions = self.state_dict["up.bias"].shape[0]
        model = CompatibilityResidual(dimensions, rank=32, seed=self.seed)
        model.load_state_dict(self.state_dict, strict=True)
        model.eval()
        with torch.inference_mode():
            return F.normalize(model(descriptors.double()), dim=1).float().contiguous()


def _masked_score_loss(
    actual: torch.Tensor,
    expected: torch.Tensor,
    query_ids: tuple[str, ...],
    anchor_ids: tuple[str, ...],
) -> torch.Tensor:
    mask = torch.tensor(
        [[query_id != anchor_id for anchor_id in anchor_ids] for query_id in query_ids],
        dtype=torch.bool,
        device=actual.device,
    )
    if not bool(mask.any()):
        raise ValueError("compatibility residual authority differs")
    return (actual[mask] - expected[mask]).square().mean()


def fit_teacher_anchored_residual(
    student: torch.Tensor,
    teacher: torch.Tensor,
    ids: tuple[str, ...],
    *,
    relational: bool,
    seed: int,
) -> ResidualFit:
    """Fit the registered paired or teacher-anchored residual adapter."""

    student, teacher = _validated_pair(student, teacher)
    if (
        type(ids) is not tuple
        or len(ids) != student.shape[0]
        or len(ids) < 256
        or len(set(ids)) != len(ids)
        or any(type(value) is not str or not value for value in ids)
        or type(relational) is not bool
        or type(seed) is not int
        or seed < 0
    ):
        raise ValueError("compatibility residual authority differs")
    anchor_indexes = sorted(
        range(len(ids)),
        key=lambda index: hashlib.sha256(
            b"sfora-compatibility-anchor-v1\0" + ids[index].encode("utf-8")
        ).digest(),
    )[:256]
    anchor_ids = tuple(ids[index] for index in anchor_indexes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source = student.double().to(device)
    target = teacher.double().to(device)
    anchor_index_tensor = torch.tensor(anchor_indexes, dtype=torch.int64, device=device)
    source_anchors = source[anchor_index_tensor]
    target_anchors = target[anchor_index_tensor]
    model = CompatibilityResidual(student.shape[1], rank=32, seed=seed, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    trajectories: dict[str, list[float]] = {
        "paired": [],
        "forward": [],
        "reverse": [],
        "self": [],
    }
    rows = student.shape[0]
    for update in range(2_000):
        query_indexes = (torch.arange(256, device=device) + update * 256) % rows
        query_ids = tuple(ids[int(index)] for index in query_indexes.cpu())
        source_query = source[query_indexes]
        target_query = target[query_indexes]
        mapped_query = F.normalize(model(source_query), dim=1)
        paired_loss = (1.0 - (mapped_query * target_query).sum(dim=1)).mean()
        if relational:
            mapped_anchors = F.normalize(model(source_anchors), dim=1)
            teacher_scores = target_query @ target_anchors.T
            forward_loss = _masked_score_loss(
                mapped_query @ target_anchors.T, teacher_scores, query_ids, anchor_ids
            )
            reverse_loss = _masked_score_loss(
                target_query @ mapped_anchors.T, teacher_scores, query_ids, anchor_ids
            )
            self_loss = _masked_score_loss(
                mapped_query @ mapped_anchors.T,
                source_query @ source_anchors.T,
                query_ids,
                anchor_ids,
            )
            loss = paired_loss + forward_loss + reverse_loss + self_loss
        else:
            forward_loss = reverse_loss = self_loss = paired_loss.new_zeros(())
            loss = paired_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        values = {
            "paired": float(paired_loss.detach().cpu()),
            "forward": float(forward_loss.detach().cpu()),
            "reverse": float(reverse_loss.detach().cpu()),
            "self": float(self_loss.detach().cpu()),
        }
        if any(not math.isfinite(value) for value in values.values()):
            raise ValueError("compatibility residual authority differs")
        for name, value in values.items():
            trajectories[name].append(value)
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    return ResidualFit(
        state_dict=state,
        losses={name: tuple(values) for name, values in trajectories.items()},
        anchor_ids=anchor_ids,
        relational=relational,
        seed=seed,
    )
