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

from sfora.siglip_spatial_tail_recovery import spatial_tail_class_split


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

    fitting, _development = spatial_tail_class_split(tuple(range(49)))
    if (
        type(labels) is not tuple
        or len(labels) != 39
        or any(type(label) is not int for label in labels)
        or labels != tuple(sorted(fitting))
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
        tuple(label for _, label in ranked[start : start + 13]) for start in range(0, 39, 13)
    )


def compatibility_oracle_halves(
    labels: tuple[int, ...],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return the two registered burned-development support/evaluation halves."""

    _fitting, development = spatial_tail_class_split(tuple(range(49)))
    if type(labels) is not tuple or labels != tuple(sorted(development)):
        raise ValueError("compatibility oracle class authority differs")
    ranked = sorted(
        (
            hashlib.sha256(
                b"sfora-compatibility-oracle-v1\0" + str(label).encode("ascii")
            ).digest(),
            label,
        )
        for label in labels
    )
    return tuple(label for _, label in ranked[:5]), tuple(label for _, label in ranked[5:])


@dataclass(frozen=True, slots=True)
class CompatibilityRetrievalEvidence:
    """Exact compatibility retrieval and cross-space diagnostic evidence."""

    ids: tuple[str, ...]
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
                for value, label in zip(self.average_precisions, self.labels, strict=True)
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
    scores = (current_query @ current_gallery.T).clamp(-1.0, 1.0)
    expected_scores = (expected_query @ expected_gallery.T).clamp(-1.0, 1.0)
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
        ids=query_ids,
        hits=tuple(hits),
        average_precisions=tuple(average_precisions),
        labels=query_labels,
        paired_cosines=paired_cosines,
        cross_score_mse=cross_score_mse,
        top10_overlaps=tuple(top10_overlaps),
        hub_counts=tuple(hubs),
    )


def combine_compatibility_retrieval_evidence(
    parts: tuple[CompatibilityRetrievalEvidence, ...],
    *,
    expected_ids: tuple[str, ...],
) -> CompatibilityRetrievalEvidence:
    """Combine disjoint held-out retrieval panels without cross-panel scoring."""

    if (
        type(parts) is not tuple
        or len(parts) < 2
        or any(type(part) is not CompatibilityRetrievalEvidence for part in parts)
        or type(expected_ids) is not tuple
        or any(type(value) is not str or not value for value in expected_ids)
        or len(set(expected_ids)) != len(expected_ids)
    ):
        raise ValueError("compatibility retrieval authority differs")
    locations: dict[str, tuple[CompatibilityRetrievalEvidence, int]] = {}
    weighted_mse = 0.0
    pair_count = 0
    for part in parts:
        if len(part.ids) < 2 or len(set(part.ids)) != len(part.ids):
            raise ValueError("compatibility retrieval authority differs")
        for index, identity in enumerate(part.ids):
            if identity in locations:
                raise ValueError("compatibility retrieval authority differs")
            locations[identity] = (part, index)
        pairs = len(part.ids) * (len(part.ids) - 1)
        weighted_mse += part.cross_score_mse * pairs
        pair_count += pairs
    if set(locations) != set(expected_ids) or pair_count == 0:
        raise ValueError("compatibility retrieval authority differs")

    def values(name: str) -> tuple[object, ...]:
        collected: list[object] = []
        for identity in expected_ids:
            part, index = locations[identity]
            collected.append(getattr(part, name)[index])
        return tuple(collected)

    return CompatibilityRetrievalEvidence(
        ids=expected_ids,
        hits=cast(tuple[bool, ...], values("hits")),
        average_precisions=cast(tuple[float, ...], values("average_precisions")),
        labels=cast(tuple[int, ...], values("labels")),
        paired_cosines=cast(tuple[float, ...], values("paired_cosines")),
        cross_score_mse=weighted_mse / pair_count,
        top10_overlaps=cast(tuple[float, ...], values("top10_overlaps")),
        hub_counts=cast(tuple[int, ...], values("hub_counts")),
    )


def csls_scores(query: torch.Tensor, gallery: torch.Tensor, *, neighbors: int = 10) -> torch.Tensor:
    """Return exact cross-domain similarity local scaling scores."""

    rows = query.shape[0] if type(query) is torch.Tensor and query.ndim == 2 else 0
    if (
        not _valid_descriptor_bank(query, rows)
        or not _valid_descriptor_bank(gallery, rows, query.shape[1])
        or type(neighbors) is not int
        or not 1 <= neighbors < rows
    ):
        raise ValueError("compatibility CSLS authority differs")
    cosine = F.normalize(query, dim=1) @ F.normalize(gallery, dim=1).T
    density_cosine = cosine.clone()
    density_cosine.diagonal().fill_(-torch.inf)
    query_density = torch.topk(density_cosine, neighbors, dim=1).values.mean(dim=1)
    gallery_density = torch.topk(density_cosine, neighbors, dim=0).values.mean(dim=0)
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
                    type(value) is not float or not math.isfinite(value) or not 0 <= value <= 1
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
    finalist: CompatibilityDecisionMetrics,
    oracle: CompatibilityDecisionMetrics,
    matched_finalist: CompatibilityDecisionMetrics,
) -> str:
    """Classify post-hoc capacity without consulting CSLS evidence."""

    if (
        type(finalist) is not CompatibilityDecisionMetrics
        or type(oracle) is not CompatibilityDecisionMetrics
        or type(matched_finalist) is not CompatibilityDecisionMetrics
    ):
        raise ValueError("compatibility decision authority differs")
    if _passes(finalist, 0.97, 0.95):
        return "posthoc-passed"
    if _passes(oracle, 0.90, 0.90) and not _passes(matched_finalist, 0.90, 0.90):
        return "coverage-failure"
    if (
        oracle.forward_r1 < 0.80
        or oracle.forward_map_at_r < 0.80
        or oracle.reverse_r1 < 0.80
        or oracle.reverse_map_at_r < 0.80
    ):
        return "registered-map-failure"
    return "ambiguous-capacity"


def hubness_present(cosine_r1: tuple[float, float], csls_r1: tuple[float, float]) -> bool:
    """Report whether CSLS provides a registered five-point directional lift."""

    if any(
        type(pair) is not tuple
        or len(pair) != 2
        or any(
            type(value) is not float or not math.isfinite(value) or not 0 <= value <= 1
            for value in pair
        )
        for pair in (cosine_r1, csls_r1)
    ):
        raise ValueError("compatibility hubness authority differs")
    return any(after - before >= 0.05 for before, after in zip(cosine_r1, csls_r1, strict=True))


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
        "ids": list(value.ids),
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
        "ids",
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
        item["ids"],
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
    ids = cast(list[object], item["ids"])
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
        any(type(entry) is not str or not entry for entry in ids)
        or len(set(cast(list[str], ids))) != rows
        or any(type(entry) is not bool for entry in hits)
        or any(
            type(entry) is not float or not math.isfinite(entry) or not 0 <= entry <= 1
            for entry in aps
        )
        or any(type(entry) is not int or entry < 0 for entry in labels)
        or any(
            type(entry) is not float or not math.isfinite(entry) or not -1 <= entry <= 1
            for entry in paired
        )
        or any(
            type(entry) is not float or not math.isfinite(entry) or not 0 <= entry <= 1
            for entry in overlaps
        )
        or any(type(entry) is not int or entry < 0 for entry in hubs)
        or any(type(entry) is not float or not math.isfinite(entry) for entry in summaries)
        or cast(float, item["cross_score_mse"]) < 0
    ):
        raise ValueError("compatibility capacity result evidence differs")
    evidence = CompatibilityRetrievalEvidence(
        ids=tuple(cast(list[str], ids)),
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


_SELECTABLE_FOLD_ARMS = {
    "affine-0.0001",
    "affine-0.01",
    "affine-1",
    "teacher-anchored-residual",
}
_CONTROL_FOLD_ARMS = {"centered-similarity", "paired-only-residual"}


def _fold_result_pairs(
    fold_evidence: Mapping[
        str,
        tuple[tuple[CompatibilityRetrievalEvidence, CompatibilityRetrievalEvidence], ...],
    ],
) -> dict[str, tuple[tuple[float, float], ...]]:
    return {
        name: tuple(
            (forward.class_macro_map_at_r, reverse.class_macro_map_at_r)
            for forward, reverse in folds
        )
        for name, folds in fold_evidence.items()
    }


def build_compatibility_capacity_result(
    *,
    checkpoint_sha256: str,
    descriptor_artifact_sha256: str,
    control_binding_sha256: str,
    optimization_manifest_sha256: str,
    spatial_artifact_sha256: str,
    image_manifest_sha256: str,
    preprocessing: str,
    fold_evidence: Mapping[
        str,
        tuple[tuple[CompatibilityRetrievalEvidence, CompatibilityRetrievalEvidence], ...],
    ],
    fitting_identity: CompatibilityRetrievalEvidence,
    residual_optimization: Mapping[str, tuple[Mapping[str, object], ...]],
    decision_optimization: Mapping[str, object],
    cells: Mapping[str, CompatibilityRetrievalEvidence],
    identity_cosine_r1: tuple[float, float],
    identity_csls_hits: tuple[tuple[bool, ...], tuple[bool, ...]],
    finalist_cosine_r1: tuple[float, float],
    finalist_csls_hits: tuple[tuple[bool, ...], tuple[bool, ...]],
) -> bytes:
    """Build canonical, claim-ineligible compatibility-capacity evidence."""

    all_fold_arms = _SELECTABLE_FOLD_ARMS | _CONTROL_FOLD_ARMS
    if type(fold_evidence) is not dict or set(fold_evidence) != all_fold_arms:
        raise ValueError("compatibility capacity result folds differ")
    for folds in fold_evidence.values():
        if (
            type(folds) is not tuple
            or len(folds) != 3
            or any(
                type(pair) is not tuple
                or len(pair) != 2
                or any(type(cell) is not CompatibilityRetrievalEvidence for cell in pair)
                for pair in folds
            )
        ):
            raise ValueError("compatibility capacity result folds differ")
    derived_fold_results = _fold_result_pairs(fold_evidence)
    fold_results = {name: derived_fold_results[name] for name in _SELECTABLE_FOLD_ARMS}
    control_fold_results = {name: derived_fold_results[name] for name in _CONTROL_FOLD_ARMS}
    finalist = select_compatibility_finalist(fold_results)
    cell_names = {
        "identity-forward",
        "identity-reverse",
        "identity-self",
        "identity-oracle-panel-forward",
        "identity-oracle-panel-reverse",
        "identity-oracle-panel-self",
        "finalist-forward",
        "finalist-reverse",
        "finalist-self",
        "finalist-oracle-panel-forward",
        "finalist-oracle-panel-reverse",
        "finalist-oracle-panel-self",
        "oracle-forward",
        "oracle-reverse",
        "oracle-self",
    }
    if type(cells) is not dict or set(cells) != cell_names:
        raise ValueError("compatibility capacity result cell inventory differs")
    parsed_cells = {name: _evidence_mapping(cells[name]) for name in sorted(cell_names)}
    if type(fitting_identity) is not CompatibilityRetrievalEvidence:
        raise ValueError("compatibility capacity result fitting evidence differs")
    fitting_mean = math.fsum(fitting_identity.paired_cosines) / len(fitting_identity.paired_cosines)
    development_mean = math.fsum(cells["identity-forward"].paired_cosines) / len(
        cells["identity-forward"].paired_cosines
    )
    finalist_metrics = _decision_metrics(cells, "finalist")
    matched_finalist_metrics = _decision_metrics(cells, "finalist-oracle-panel")
    oracle_metrics = _decision_metrics(cells, "oracle")
    csls_diagnostic: dict[str, object] = {}
    csls_r1: dict[str, tuple[float, float]] = {}
    for name, cosine_r1, direction_hits in (
        ("identity", identity_cosine_r1, identity_csls_hits),
        ("finalist", finalist_cosine_r1, finalist_csls_hits),
    ):
        directions: dict[str, object] = {}
        values: list[float] = []
        for direction, hits in zip(("forward", "reverse"), direction_hits, strict=True):
            evidence = cells[f"{name}-{direction}"]
            if (
                type(hits) is not tuple
                or len(hits) != len(evidence.ids)
                or any(type(hit) is not bool for hit in hits)
            ):
                raise ValueError("compatibility capacity result hubness differs")
            r1 = sum(hits) / len(hits)
            values.append(r1)
            directions[direction] = {
                "ids": list(evidence.ids),
                "labels": list(evidence.labels),
                "hits": list(hits),
                "r1": r1,
            }
        csls_r1[name] = (values[0], values[1])
        csls_diagnostic[name] = {
            "cosine_r1": list(cosine_r1),
            "directions": directions,
        }
    result = {
        "schema": "sfora-siglip-compatibility-capacity-v1",
        "claim_eligible": False,
        "external_evaluation_access": False,
        "checkpoint_sha256": _hex_digest(checkpoint_sha256),
        "descriptor_artifact_sha256": _hex_digest(descriptor_artifact_sha256),
        "control_binding_sha256": _hex_digest(control_binding_sha256),
        "optimization_manifest_sha256": _hex_digest(optimization_manifest_sha256),
        "spatial_artifact_sha256": _hex_digest(spatial_artifact_sha256),
        "image_manifest_sha256": _hex_digest(image_manifest_sha256),
        "preprocessing": preprocessing,
        "fold_results": {
            name: [list(pair) for pair in fold_results[name]] for name in sorted(fold_results)
        },
        "control_fold_results": {
            name: [list(pair) for pair in control_fold_results[name]]
            for name in sorted(control_fold_results)
        },
        "fold_evidence": {
            name: [
                {
                    "forward": _evidence_mapping(forward),
                    "reverse": _evidence_mapping(reverse),
                }
                for forward, reverse in fold_evidence[name]
            ]
            for name in sorted(fold_evidence)
        },
        "fitting_identity": _evidence_mapping(fitting_identity),
        "paired_cosine_gap": {
            "fitting_mean": fitting_mean,
            "development_mean": development_mean,
            "gap": fitting_mean - development_mean,
        },
        "residual_optimization": residual_optimization,
        "decision_optimization": decision_optimization,
        "finalist": finalist,
        "cells": parsed_cells,
        "csls_diagnostic": csls_diagnostic,
        "classification": classify_compatibility_capacity(
            finalist_metrics, oracle_metrics, matched_finalist_metrics
        ),
        "hubness_present": hubness_present(identity_cosine_r1, csls_r1["identity"])
        or hubness_present(finalist_cosine_r1, csls_r1["finalist"]),
    }
    raw = (
        json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()
    validate_compatibility_capacity_result_bytes(raw)
    return raw


def _validate_optimization_record(
    entry: object,
    *,
    relational: bool,
    seed: int,
    expected_anchors: list[str],
    expected_dimensions: int,
) -> None:
    if type(entry) is not dict or set(entry) != {
        "relational",
        "seed",
        "dimensions",
        "parameter_count",
        "anchor_ids",
        "losses",
        "device",
        "torch_version",
        "optimizer",
        "learning_rate",
        "weight_decay",
    }:
        raise ValueError("compatibility capacity result optimization differs")
    dimensions = entry["dimensions"]
    anchors = entry["anchor_ids"]
    losses = entry["losses"]
    if (
        entry["relational"] is not relational
        or type(entry["relational"]) is not bool
        or type(entry["seed"]) is not int
        or entry["seed"] != seed
        or type(dimensions) is not int
        or dimensions != expected_dimensions
        or type(entry["parameter_count"]) is not int
        or entry["parameter_count"] != 2 * dimensions * 32 + dimensions
        or type(anchors) is not list
        or len(expected_anchors) != 256
        or len(anchors) != 256
        or len(set(anchors)) != 256
        or anchors != expected_anchors
        or entry["device"] not in {"cpu", "cuda"}
        or type(entry["device"]) is not str
        or type(entry["torch_version"]) is not str
        or not entry["torch_version"]
        or entry["optimizer"] != "adamw"
        or type(entry["optimizer"]) is not str
        or type(entry["learning_rate"]) is not float
        or entry["learning_rate"] != 1e-3
        or type(entry["weight_decay"]) is not float
        or entry["weight_decay"] != 0.0
        or type(losses) is not dict
        or set(losses) != {"paired", "forward", "reverse", "self"}
    ):
        raise ValueError("compatibility capacity result optimization differs")
    for loss_name, trajectory in losses.items():
        if (
            type(trajectory) is not list
            or len(trajectory) != 2_000
            or any(
                type(loss) is not float or not math.isfinite(loss) or loss < 0
                for loss in trajectory
            )
            or (
                not relational and loss_name != "paired" and any(loss != 0.0 for loss in trajectory)
            )
        ):
            raise ValueError("compatibility capacity result optimization differs")


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
        "control_binding_sha256",
        "optimization_manifest_sha256",
        "spatial_artifact_sha256",
        "image_manifest_sha256",
        "preprocessing",
        "fold_results",
        "control_fold_results",
        "fold_evidence",
        "fitting_identity",
        "paired_cosine_gap",
        "residual_optimization",
        "decision_optimization",
        "finalist",
        "cells",
        "csls_diagnostic",
        "classification",
        "hubness_present",
    }
    if type(raw) is not bytes or type(value) is not dict or set(value) != keys:
        raise ValueError("compatibility capacity result schema differs")
    result = cast(dict[str, object], value)
    canonical = (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
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
    _hex_digest(result["control_binding_sha256"])
    _hex_digest(result["optimization_manifest_sha256"])
    _hex_digest(result["spatial_artifact_sha256"])
    _hex_digest(result["image_manifest_sha256"])
    if result["preprocessing"] != "siglip-evaluation-transform-v1":
        raise ValueError("compatibility capacity result preprocessing differs")
    fitting_identity = _parse_evidence(result["fitting_identity"])
    fitting_labels, _development_labels = spatial_tail_class_split(tuple(range(49)))
    if (
        set(fitting_identity.labels) != set(fitting_labels)
        or min(fitting_identity.labels.count(label) for label in fitting_labels) < 2
    ):
        raise ValueError("compatibility capacity result fitting evidence differs")
    fold_evidence_value = result["fold_evidence"]
    all_fold_arms = _SELECTABLE_FOLD_ARMS | _CONTROL_FOLD_ARMS
    if type(fold_evidence_value) is not dict or set(fold_evidence_value) != all_fold_arms:
        raise ValueError("compatibility capacity result folds differ")
    parsed_fold_evidence: dict[
        str,
        tuple[tuple[CompatibilityRetrievalEvidence, CompatibilityRetrievalEvidence], ...],
    ] = {}
    fitting_partition, _development_partition = spatial_tail_class_split(tuple(range(49)))
    registered_folds = compatibility_folds(tuple(sorted(fitting_partition)))
    identity_by_fold: list[tuple[tuple[str, ...], tuple[int, ...]]] = []
    for name, entries in fold_evidence_value.items():
        if type(name) is not str or type(entries) is not list or len(entries) != 3:
            raise ValueError("compatibility capacity result folds differ")
        converted: list[tuple[CompatibilityRetrievalEvidence, CompatibilityRetrievalEvidence]] = []
        for fold_index, entry in enumerate(entries):
            if type(entry) is not dict or set(entry) != {"forward", "reverse"}:
                raise ValueError("compatibility capacity result folds differ")
            forward = _parse_evidence(entry["forward"])
            reverse = _parse_evidence(entry["reverse"])
            fold_label_set = set(registered_folds[fold_index])
            expected_ids = tuple(
                identity
                for identity, label in zip(
                    fitting_identity.ids, fitting_identity.labels, strict=True
                )
                if label in fold_label_set
            )
            expected_labels = tuple(
                label for label in fitting_identity.labels if label in fold_label_set
            )
            if (
                forward.ids != reverse.ids
                or forward.labels != reverse.labels
                or forward.ids != expected_ids
                or forward.labels != expected_labels
                or set(forward.labels) != set(registered_folds[fold_index])
                or min(forward.labels.count(label) for label in registered_folds[fold_index]) < 2
            ):
                raise ValueError("compatibility capacity result folds differ")
            if name == sorted(all_fold_arms)[0]:
                identity_by_fold.append((forward.ids, forward.labels))
            elif (forward.ids, forward.labels) != identity_by_fold[fold_index]:
                raise ValueError("compatibility capacity result folds differ")
            converted.append((forward, reverse))
        parsed_fold_evidence[name] = tuple(converted)
    derived = _fold_result_pairs(parsed_fold_evidence)
    folds = {name: derived[name] for name in _SELECTABLE_FOLD_ARMS}
    control_folds = {name: derived[name] for name in _CONTROL_FOLD_ARMS}
    for key, expected in (
        ("fold_results", folds),
        ("control_fold_results", control_folds),
    ):
        encoded = result[key]
        expected_encoded = {
            name: [list(pair) for pair in values] for name, values in expected.items()
        }
        if encoded != expected_encoded:
            raise ValueError("compatibility capacity result folds differ")
    optimization = result["residual_optimization"]
    if type(optimization) is not dict or set(optimization) != {
        "teacher-anchored-residual",
        "paired-only-residual",
    }:
        raise ValueError("compatibility capacity result optimization differs")
    optimization_dimensions: set[int] = set()
    for name, relational in (
        ("teacher-anchored-residual", True),
        ("paired-only-residual", False),
    ):
        entries = optimization[name]
        if type(entries) is not list or len(entries) != 3:
            raise ValueError("compatibility capacity result optimization differs")
        for fold_index, entry in enumerate(entries):
            if type(entry) is not dict or set(entry) != {
                "relational",
                "seed",
                "dimensions",
                "parameter_count",
                "anchor_ids",
                "losses",
                "device",
                "torch_version",
                "optimizer",
                "learning_rate",
                "weight_decay",
            }:
                raise ValueError("compatibility capacity result optimization differs")
            dimensions = entry["dimensions"]
            anchors = entry["anchor_ids"]
            losses = entry["losses"]
            validation_labels = set(registered_folds[fold_index])
            training_ids = tuple(
                identity
                for identity, label in zip(
                    fitting_identity.ids, fitting_identity.labels, strict=True
                )
                if label not in validation_labels
            )
            expected_anchors = sorted(
                training_ids,
                key=lambda identity: hashlib.sha256(
                    b"sfora-compatibility-anchor-v1\0" + identity.encode("utf-8")
                ).digest(),
            )[:256]
            if (
                entry["relational"] is not relational
                or type(entry["relational"]) is not bool
                or type(entry["seed"]) is not int
                or entry["seed"] != 20260905 + fold_index
                or type(dimensions) is not int
                or dimensions < 2
                or type(entry["parameter_count"]) is not int
                or entry["parameter_count"] != 2 * dimensions * 32 + dimensions
                or type(anchors) is not list
                or len(anchors) != 256
                or len(set(anchors)) != 256
                or any(type(identity) is not str or not identity for identity in anchors)
                or anchors != expected_anchors
                or entry["device"] not in {"cpu", "cuda"}
                or type(entry["device"]) is not str
                or type(entry["torch_version"]) is not str
                or not entry["torch_version"]
                or entry["optimizer"] != "adamw"
                or type(entry["optimizer"]) is not str
                or type(entry["learning_rate"]) is not float
                or entry["learning_rate"] != 1e-3
                or type(entry["weight_decay"]) is not float
                or entry["weight_decay"] != 0.0
                or type(losses) is not dict
                or set(losses) != {"paired", "forward", "reverse", "self"}
            ):
                raise ValueError("compatibility capacity result optimization differs")
            optimization_dimensions.add(dimensions)
            for loss_name, trajectory in losses.items():
                if (
                    type(trajectory) is not list
                    or len(trajectory) != 2_000
                    or any(
                        type(loss) is not float or not math.isfinite(loss) or loss < 0
                        for loss in trajectory
                    )
                    or (
                        not relational
                        and loss_name != "paired"
                        and any(loss != 0.0 for loss in trajectory)
                    )
                ):
                    raise ValueError("compatibility capacity result optimization differs")
    if len(optimization_dimensions) != 1:
        raise ValueError("compatibility capacity result optimization differs")
    expected_finalist = select_compatibility_finalist(folds)
    if type(result["finalist"]) is not str or result["finalist"] != expected_finalist:
        raise ValueError("compatibility capacity result finalist differs")
    cell_value = result["cells"]
    cell_names = {
        "identity-forward",
        "identity-reverse",
        "identity-self",
        "identity-oracle-panel-forward",
        "identity-oracle-panel-reverse",
        "identity-oracle-panel-self",
        "finalist-forward",
        "finalist-reverse",
        "finalist-self",
        "finalist-oracle-panel-forward",
        "finalist-oracle-panel-reverse",
        "finalist-oracle-panel-self",
        "oracle-forward",
        "oracle-reverse",
        "oracle-self",
    }
    if type(cell_value) is not dict or set(cell_value) != cell_names:
        raise ValueError("compatibility capacity result cells differ")
    cells = {name: _parse_evidence(cell_value[name]) for name in cell_names}
    _fitting_labels, development_labels = spatial_tail_class_split(tuple(range(49)))
    if (
        len({len(evidence.hits) for evidence in cells.values()}) != 1
        or len({evidence.ids for evidence in cells.values()}) != 1
        or len({evidence.labels for evidence in cells.values()}) != 1
        or bool(set(fitting_identity.ids) & set(next(iter(cells.values())).ids))
        or set(next(iter(cells.values())).labels) != set(development_labels)
        or min(next(iter(cells.values())).labels.count(label) for label in development_labels) < 2
    ):
        raise ValueError("compatibility capacity result cells differ")
    decision_optimization = result["decision_optimization"]
    if type(decision_optimization) is not dict or set(decision_optimization) != {
        "oracle-halves",
        "finalist-refit",
    }:
        raise ValueError("compatibility capacity result optimization differs")
    oracle_entries = decision_optimization["oracle-halves"]
    if type(oracle_entries) is not list or len(oracle_entries) != 2:
        raise ValueError("compatibility capacity result optimization differs")
    development_evidence = cells["identity-forward"]
    oracle_halves = compatibility_oracle_halves(tuple(sorted(development_labels)))
    residual_dimensions = next(iter(optimization_dimensions))
    for fold_index, training_labels in enumerate(oracle_halves):
        training_label_set = set(training_labels)
        training_ids = tuple(
            identity
            for identity, label in zip(
                development_evidence.ids,
                development_evidence.labels,
                strict=True,
            )
            if label in training_label_set
        )
        expected_anchors = sorted(
            training_ids,
            key=lambda identity: hashlib.sha256(
                b"sfora-compatibility-anchor-v1\0" + identity.encode("utf-8")
            ).digest(),
        )[:256]
        _validate_optimization_record(
            oracle_entries[fold_index],
            relational=True,
            seed=20260909 + fold_index,
            expected_anchors=expected_anchors,
            expected_dimensions=residual_dimensions,
        )
    finalist_refit = decision_optimization["finalist-refit"]
    if expected_finalist == "teacher-anchored-residual":
        expected_anchors = sorted(
            fitting_identity.ids,
            key=lambda identity: hashlib.sha256(
                b"sfora-compatibility-anchor-v1\0" + identity.encode("utf-8")
            ).digest(),
        )[:256]
        _validate_optimization_record(
            finalist_refit,
            relational=True,
            seed=20260908,
            expected_anchors=expected_anchors,
            expected_dimensions=residual_dimensions,
        )
    elif finalist_refit is not None:
        raise ValueError("compatibility capacity result optimization differs")
    paired_gap = result["paired_cosine_gap"]
    fitting_mean = math.fsum(fitting_identity.paired_cosines) / len(fitting_identity.paired_cosines)
    development_mean = math.fsum(cells["identity-forward"].paired_cosines) / len(
        cells["identity-forward"].paired_cosines
    )
    expected_gap = {
        "fitting_mean": fitting_mean,
        "development_mean": development_mean,
        "gap": fitting_mean - development_mean,
    }
    if paired_gap != expected_gap:
        raise ValueError("compatibility capacity result paired cosine differs")
    diagnostic = result["csls_diagnostic"]
    if type(diagnostic) is not dict or set(diagnostic) != {"identity", "finalist"}:
        raise ValueError("compatibility capacity result hubness differs")
    expected_hubness = False
    for name in ("identity", "finalist"):
        cell = diagnostic[name]
        if type(cell) is not dict or set(cell) != {"cosine_r1", "directions"}:
            raise ValueError("compatibility capacity result hubness differs")
        cosine = cell["cosine_r1"]
        directions = cell["directions"]
        if (
            type(cosine) is not list
            or type(directions) is not dict
            or set(directions)
            != {
                "forward",
                "reverse",
            }
        ):
            raise ValueError("compatibility capacity result hubness differs")
        csls_values: list[float] = []
        for direction in ("forward", "reverse"):
            entry = directions[direction]
            evidence = cells[f"{name}-{direction}"]
            if type(entry) is not dict or set(entry) != {"ids", "labels", "hits", "r1"}:
                raise ValueError("compatibility capacity result hubness differs")
            hits = entry["hits"]
            if (
                entry["ids"] != list(evidence.ids)
                or entry["labels"] != list(evidence.labels)
                or type(hits) is not list
                or len(hits) != len(evidence.ids)
                or any(type(hit) is not bool for hit in hits)
                or type(entry["r1"]) is not float
                or entry["r1"] != sum(hits) / len(hits)
            ):
                raise ValueError("compatibility capacity result hubness differs")
            csls_values.append(entry["r1"])
        try:
            cell_hubness = hubness_present((cosine[0], cosine[1]), (csls_values[0], csls_values[1]))
        except ValueError as error:
            raise ValueError("compatibility capacity result hubness differs") from error
        expected_hubness = expected_hubness or cell_hubness
        prefix = name
        if cosine != [
            cells[f"{prefix}-forward"].micro_r1,
            cells[f"{prefix}-reverse"].micro_r1,
        ]:
            raise ValueError("compatibility capacity result hubness differs")
    expected_class = classify_compatibility_capacity(
        _decision_metrics(cells, "finalist"),
        _decision_metrics(cells, "oracle"),
        _decision_metrics(cells, "finalist-oracle-panel"),
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

        result: torch.Tensor = descriptors + self.up(F.gelu(self.down(descriptors)))
        return result


@dataclass(frozen=True, slots=True)
class ResidualFit:
    """A frozen fitted residual and its complete optimization evidence."""

    state_dict: dict[str, torch.Tensor]
    losses: dict[str, tuple[float, ...]]
    anchor_ids: tuple[str, ...]
    relational: bool
    seed: int
    device: str
    torch_version: str

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
    mask: torch.Tensor,
) -> torch.Tensor:
    if (
        type(mask) is not torch.Tensor
        or mask.dtype != torch.bool
        or mask.device != actual.device
        or mask.shape != actual.shape
        or expected.shape != actual.shape
        or not bool(mask.any())
    ):
        raise ValueError("compatibility residual authority differs")
    return (actual[mask] - expected[mask]).square().mean()


def _paired_cosine_loss(mapped: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if mapped.shape != target.shape or mapped.ndim != 2:
        raise ValueError("compatibility residual authority differs")
    cosine = (mapped * target).sum(dim=1).clamp(-1.0, 1.0)
    return (1.0 - cosine).mean()


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
    source = F.normalize(student.double(), dim=1).to(device)
    target = F.normalize(teacher.double(), dim=1).to(device)
    anchor_index_tensor = torch.tensor(anchor_indexes, dtype=torch.int64, device=device)
    source_anchors = source[anchor_index_tensor]
    target_anchors = target[anchor_index_tensor]
    identity_mask = torch.tensor(
        [[identity != anchor for anchor in anchor_ids] for identity in ids],
        dtype=torch.bool,
        device=device,
    )
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
        score_mask = identity_mask[query_indexes]
        source_query = source[query_indexes]
        target_query = target[query_indexes]
        mapped_query = F.normalize(model(source_query), dim=1)
        paired_loss = _paired_cosine_loss(mapped_query, target_query)
        if relational:
            mapped_anchors = F.normalize(model(source_anchors), dim=1)
            teacher_scores = target_query @ target_anchors.T
            forward_loss = _masked_score_loss(
                mapped_query @ target_anchors.T, teacher_scores, score_mask
            )
            reverse_loss = _masked_score_loss(
                target_query @ mapped_anchors.T, teacher_scores, score_mask
            )
            self_loss = _masked_score_loss(
                mapped_query @ mapped_anchors.T,
                source_query @ source_anchors.T,
                score_mask,
            )
            loss = paired_loss + forward_loss + reverse_loss + self_loss
        else:
            forward_loss = reverse_loss = self_loss = paired_loss.new_zeros(())
            loss = paired_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()  # type: ignore[no-untyped-call]
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
        device=device.type,
        torch_version=str(torch.__version__),
    )
