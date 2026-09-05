"""Cross-model retrieval evidence and frozen exploratory investment gates."""

from __future__ import annotations

import math
from collections.abc import Callable
from decimal import Decimal
from typing import Any

import torch

from sfora.qwen_geometry_control import (
    GeometryRetrievalEvidence,
    validate_geometry_retrieval_evidence,
)


def asymmetric_retrieval_evidence(
    query_descriptors: torch.Tensor,
    gallery_descriptors: torch.Tensor,
    *,
    query_ids: tuple[str, ...],
    gallery_ids: tuple[str, ...],
    query_labels: tuple[int, ...],
    gallery_labels: tuple[int, ...],
    check_time: Callable[[], None] | None = None,
) -> GeometryRetrievalEvidence:
    """Rank one model's queries against another model's matching-ID gallery."""
    n = len(query_ids)
    if (
        n < 2
        or len(set(query_ids)) != n
        or len(gallery_ids) != n
        or len(set(gallery_ids)) != n
        or set(query_ids) != set(gallery_ids)
        or len(query_labels) != n
        or len(gallery_labels) != n
        or any(type(value) is not str or not value for value in (*query_ids, *gallery_ids))
        or any(type(value) is not int or value < 0 for value in (*query_labels, *gallery_labels))
        or query_descriptors.device.type != "cpu"
        or gallery_descriptors.device.type != "cpu"
        or query_descriptors.dtype != torch.float32
        or gallery_descriptors.dtype != torch.float32
        or query_descriptors.ndim != 2
        or gallery_descriptors.ndim != 2
        or tuple(query_descriptors.shape) != tuple(gallery_descriptors.shape)
        or query_descriptors.shape[0] != n
        or query_descriptors.shape[1] < 1
        or not bool(torch.isfinite(query_descriptors).all())
        or not bool(torch.isfinite(gallery_descriptors).all())
    ):
        raise ValueError("asymmetric retrieval identity or descriptor authority differs")
    gallery_by_id = dict(zip(gallery_ids, range(n), strict=True))
    gallery_label_by_id = dict(zip(gallery_ids, gallery_labels, strict=True))
    if tuple(gallery_label_by_id[value] for value in query_ids) != query_labels:
        raise ValueError("asymmetric retrieval identity label binding differs")
    canonical_gallery = gallery_descriptors[
        torch.tensor([gallery_by_id[value] for value in query_ids], dtype=torch.int64)
    ]
    query_norms = torch.linalg.vector_norm(query_descriptors, dim=1)
    gallery_norms = torch.linalg.vector_norm(canonical_gallery, dim=1)
    if not bool((query_norms > 0).all()) or not bool((gallery_norms > 0).all()):
        raise ValueError("asymmetric retrieval descriptors must have finite nonzero norms")
    queries = torch.nn.functional.normalize(query_descriptors, dim=1)
    gallery = torch.nn.functional.normalize(canonical_gallery, dim=1)
    counts = {label: query_labels.count(label) for label in set(query_labels)}
    if min(counts.values()) < 2:
        raise ValueError("asymmetric retrieval classes require at least two images")
    label_tensor = torch.tensor(query_labels)
    nearest: list[int] = []
    correct: list[bool] = []
    aps: list[float] = []
    top_r: list[tuple[int, ...]] = []
    for start in range(0, n, 128):
        if check_time is not None:
            check_time()
        stop = min(start + 128, n)
        scores = queries[start:stop] @ gallery.T
        scores[torch.arange(stop - start), torch.arange(start, stop)] = -torch.inf
        ranked = torch.argsort(scores, dim=1, descending=True, stable=True)
        for local, row in enumerate(range(start, stop)):
            first = int(ranked[local, 0])
            nearest.append(first)
            correct.append(query_labels[first] == query_labels[row])
            r = counts[query_labels[row]] - 1
            retained = ranked[local, :r]
            top_r.append(tuple(int(value) for value in retained))
            relevant = (label_tensor[retained] == query_labels[row]).tolist()
            hits = 0
            terms = []
            for rank, hit in enumerate(relevant, 1):
                if hit:
                    hits += 1
                    terms.append(hits / rank)
            aps.append(math.fsum(terms) / r)
    evidence = GeometryRetrievalEvidence(
        tuple(range(n)),
        query_labels,
        tuple(nearest),
        tuple(top_r),
        tuple(correct),
        tuple(aps),
    )
    validate_geometry_retrieval_evidence(evidence)
    return evidence


def asymmetric_recovery_decision(cells: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Apply the frozen total investment gate to both student-to-teacher cells."""
    if set(cells) != {"pa", "relational"}:
        raise ValueError("asymmetric recovery arms differ")
    arms = {}
    for name, cell in cells.items():
        value = cell.get("map_at_r")
        if (
            type(cell.get("queries")) is not int
            or cell["queries"] != 2746
            or type(value) is not float
            or not math.isfinite(value)
            or not 0 <= value <= 1
        ):
            raise ValueError("asymmetric recovery quality authority differs")
        exact = Decimal(str(value))
        if exact >= Decimal("0.70"):
            classification = "alive"
        elif exact <= Decimal("0.462"):
            classification = "dead"
        else:
            classification = "inconclusive-not-alive"
        arms[name] = {"map_at_r": value, "classification": classification}
    selected = next(
        (name for name in ("pa", "relational") if arms[name]["classification"] == "alive"),
        None,
    )
    return {
        "claim_eligible": False,
        "surface": "exploratory-reuse-49..81",
        "alive_map_at_r_minimum": 0.70,
        "dead_map_at_r_maximum": 0.462,
        "middle_band_action": "inconclusive-not-alive",
        "selected_arm": selected,
        "arms": arms,
    }
