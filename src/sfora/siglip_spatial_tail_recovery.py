"""Evidence authority for class-isolated SigLIP spatial-tail recovery."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import cast

import torch
from torch.nn import functional as F


@dataclass(frozen=True, slots=True)
class SpatialRetrievalEvidence:
    """Exact per-query Recall@1 and AP@R evidence."""

    correct: tuple[bool, ...]
    average_precisions: tuple[float, ...]

    @property
    def map_at_r(self) -> float:
        return math.fsum(self.average_precisions) / len(self.average_precisions)


def spatial_tail_class_split(
    labels: tuple[int, ...],
) -> tuple[frozenset[int], frozenset[int]]:
    """Return the fixed 39-class fit and 10-class development partition."""

    if (
        type(labels) is not tuple
        or len(labels) != 49
        or any(type(label) is not int for label in labels)
        or set(labels) != set(range(49))
    ):
        raise ValueError("spatial tail class authority differs")
    ranked = sorted(
        (
            int.from_bytes(
                hashlib.sha256(
                    b"sfora-spatial-tail-dev-v1\0" + str(label).encode("ascii")
                ).digest()[:8],
                "big",
            ),
            label,
        )
        for label in labels
    )
    development = frozenset(label for _, label in ranked[:10])
    return frozenset(range(49)) - development, development


def spatial_retrieval_evidence(
    query_descriptors: torch.Tensor,
    gallery_descriptors: torch.Tensor,
    *,
    query_ids: tuple[str, ...],
    gallery_ids: tuple[str, ...],
    query_labels: tuple[int, ...],
    gallery_labels: tuple[int, ...],
) -> SpatialRetrievalEvidence:
    """Evaluate identity-bound cosine retrieval with deterministic tie order."""

    rows = len(query_ids)
    if (
        rows < 2
        or len(set(query_ids)) != rows
        or len(gallery_ids) != rows
        or len(set(gallery_ids)) != rows
        or set(query_ids) != set(gallery_ids)
        or len(query_labels) != rows
        or len(gallery_labels) != rows
        or any(type(value) is not str or not value for value in (*query_ids, *gallery_ids))
        or any(type(value) is not int or value < 0 for value in (*query_labels, *gallery_labels))
        or type(query_descriptors) is not torch.Tensor
        or type(gallery_descriptors) is not torch.Tensor
        or query_descriptors.device.type != "cpu"
        or gallery_descriptors.device.type != "cpu"
        or query_descriptors.dtype != torch.float32
        or gallery_descriptors.dtype != torch.float32
        or query_descriptors.ndim != 2
        or query_descriptors.shape != gallery_descriptors.shape
        or query_descriptors.shape[0] != rows
        or query_descriptors.shape[1] < 1
        or not bool(torch.isfinite(query_descriptors).all())
        or not bool(torch.isfinite(gallery_descriptors).all())
    ):
        raise ValueError("spatial retrieval authority differs")
    gallery_by_id = dict(zip(gallery_ids, range(rows), strict=True))
    gallery_label_by_id = dict(zip(gallery_ids, gallery_labels, strict=True))
    if tuple(gallery_label_by_id[value] for value in query_ids) != query_labels:
        raise ValueError("spatial retrieval label binding differs")
    canonical_gallery = gallery_descriptors[
        torch.tensor([gallery_by_id[value] for value in query_ids], dtype=torch.int64)
    ]
    if bool((torch.linalg.vector_norm(query_descriptors, dim=1) <= 0).any()) or bool(
        (torch.linalg.vector_norm(canonical_gallery, dim=1) <= 0).any()
    ):
        raise ValueError("spatial retrieval descriptors must have nonzero norms")
    queries = F.normalize(query_descriptors, dim=1)
    gallery = F.normalize(canonical_gallery, dim=1)
    counts = {label: query_labels.count(label) for label in set(query_labels)}
    if min(counts.values()) < 2:
        raise ValueError("spatial retrieval classes require at least two images")
    label_tensor = torch.tensor(query_labels)
    correct: list[bool] = []
    average_precisions: list[float] = []
    for start in range(0, rows, 128):
        stop = min(start + 128, rows)
        scores = queries[start:stop] @ gallery.T
        scores[torch.arange(stop - start), torch.arange(start, stop)] = -torch.inf
        ranked = torch.argsort(scores, dim=1, descending=True, stable=True)
        for local, row in enumerate(range(start, stop)):
            correct.append(query_labels[int(ranked[local, 0])] == query_labels[row])
            retained = ranked[local, : counts[query_labels[row]] - 1]
            relevant = (label_tensor[retained] == query_labels[row]).tolist()
            hits = 0
            terms: list[float] = []
            for rank, hit in enumerate(relevant, 1):
                if hit:
                    hits += 1
                    terms.append(hits / rank)
            average_precisions.append(math.fsum(terms) / len(relevant))
    return SpatialRetrievalEvidence(tuple(correct), tuple(average_precisions))


def _quality(value: object) -> tuple[int, int, float]:
    if type(value) is not dict or set(value) != {"queries", "correct", "map_at_r"}:
        raise ValueError("spatial tail quality schema differs")
    cell = cast(dict[str, object], value)
    if (
        type(cell["queries"]) is not int
        or cell["queries"] < 2
        or type(cell["correct"]) is not int
        or not 0 <= cast(int, cell["correct"]) <= cast(int, cell["queries"])
        or type(cell["map_at_r"]) is not float
        or not math.isfinite(cast(float, cell["map_at_r"]))
        or not 0.0 <= cast(float, cell["map_at_r"]) <= 1.0
    ):
        raise ValueError("spatial tail quality authority differs")
    return cast(int, cell["queries"]), cast(int, cell["correct"]), cast(float, cell["map_at_r"])


def spatial_tail_decision(cells: dict[str, dict[str, object]]) -> dict[str, object]:
    """Classify treatment quality using the preregistered half-gap rule."""

    names = {"baseline", "tokenwise-control", "latent-interaction", "teacher"}
    if type(cells) is not dict or set(cells) != names:
        raise ValueError("spatial tail cell inventory differs")
    parsed = {name: _quality(cells[name]) for name in names}
    if len({value[0] for value in parsed.values()}) != 1:
        raise ValueError("spatial tail query count differs")
    baseline = parsed["baseline"]
    control = parsed["tokenwise-control"]
    treatment = parsed["latent-interaction"]
    teacher = parsed["teacher"]
    recall_gap = teacher[1] - baseline[1]
    map_gap = teacher[2] - baseline[2]
    if recall_gap <= 0 or map_gap <= 0:
        raise ValueError("spatial tail teacher gap differs")
    recall_closure = (treatment[1] - baseline[1]) / recall_gap
    map_closure = (treatment[2] - baseline[2]) / map_gap
    passed = (
        treatment[1] > control[1]
        and treatment[2] > control[2]
        and recall_closure >= 0.5
        and map_closure >= 0.5
    )
    return {
        "classification": "latency-pending" if passed else "interaction-rejected",
        "recall_gap_closure": recall_closure,
        "map_gap_closure": map_closure,
    }


def _hex_digest(value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("spatial tail digest differs")
    return value


def _retrieval_mapping(value: object, *, summarized: bool) -> dict[str, object]:
    keys = {"hits", "average_precision"}
    if summarized:
        keys |= {"queries", "correct", "map_at_r"}
    if type(value) is not dict or set(value) != keys:
        raise ValueError("spatial tail retrieval evidence differs")
    evidence = cast(dict[str, object], value)
    hits = evidence["hits"]
    average_precision = evidence["average_precision"]
    if (
        type(hits) is not list
        or type(average_precision) is not list
        or len(hits) < 2
        or len(average_precision) != len(hits)
        or any(type(hit) is not bool for hit in hits)
        or any(
            type(score) is not float or not math.isfinite(score) or not 0.0 <= score <= 1.0
            for score in average_precision
        )
    ):
        raise ValueError("spatial tail retrieval evidence differs")
    correct = sum(cast(list[bool], hits))
    scores = cast(list[float], average_precision)
    map_at_r = math.fsum(scores) / len(scores)
    if summarized and (
        evidence["queries"] != len(hits)
        or type(evidence["queries"]) is not int
        or evidence["correct"] != correct
        or type(evidence["correct"]) is not int
        or evidence["map_at_r"] != map_at_r
        or type(evidence["map_at_r"]) is not float
    ):
        raise ValueError("spatial tail retrieval relation differs")
    return {
        "queries": len(hits),
        "correct": correct,
        "map_at_r": map_at_r,
        "hits": cast(list[bool], hits).copy(),
        "average_precision": cast(list[float], average_precision).copy(),
    }


def _training(value: object) -> dict[str, object]:
    keys = {
        "updates",
        "tokenwise_initial_loss",
        "tokenwise_final_loss",
        "interaction_initial_loss",
        "interaction_final_loss",
    }
    if type(value) is not dict or set(value) != keys:
        raise ValueError("spatial tail training evidence differs")
    evidence = cast(dict[str, object], value)
    if type(evidence["updates"]) is not int or evidence["updates"] != 4_000:
        raise ValueError("spatial tail training evidence differs")
    for key in keys - {"updates"}:
        if (
            type(evidence[key]) is not float
            or not math.isfinite(cast(float, evidence[key]))
            or cast(float, evidence[key]) < 0.0
        ):
            raise ValueError("spatial tail training evidence differs")
    return evidence.copy()


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def build_spatial_tail_result(
    *,
    checkpoint_sha256: str,
    optimization_manifest_sha256: str,
    artifact_sha256: str,
    fit_labels: tuple[int, ...],
    development_labels: tuple[int, ...],
    cells: dict[str, dict[str, object]],
    training: dict[str, object],
) -> bytes:
    """Build canonical per-query evidence and recompute the branch decision."""

    fit, development = spatial_tail_class_split(tuple(range(49)))
    if tuple(sorted(fit)) != fit_labels or tuple(sorted(development)) != development_labels:
        raise ValueError("spatial tail class authority differs")
    names = ("baseline", "tokenwise-control", "latent-interaction", "teacher")
    if type(cells) is not dict or set(cells) != set(names):
        raise ValueError("spatial tail cell inventory differs")
    output_cells: dict[str, dict[str, object]] = {}
    for name in names:
        cell = cells[name]
        if type(cell) is not dict or set(cell) != {"self", "cross"}:
            raise ValueError("spatial tail cell schema differs")
        output_cells[name] = {
            "self": _retrieval_mapping(cell["self"], summarized=False),
            "cross": _retrieval_mapping(cell["cross"], summarized=False),
        }
    decision = spatial_tail_decision(
        {
            name: {
                key: output_cells[name]["self"][key]
                for key in ("queries", "correct", "map_at_r")
            }
            for name in names
        }
    )
    result = {
        "schema": "sfora-siglip-spatial-tail-recovery-v1",
        "claim_eligible": False,
        "external_evaluation_access": False,
        "checkpoint_sha256": _hex_digest(checkpoint_sha256),
        "optimization_manifest_sha256": _hex_digest(optimization_manifest_sha256),
        "artifact_sha256": _hex_digest(artifact_sha256),
        "fit_labels": list(fit_labels),
        "development_labels": list(development_labels),
        "cells": output_cells,
        "training": _training(training),
        **decision,
    }
    raw = _canonical_bytes(result)
    validate_spatial_tail_result_bytes(raw)
    return raw


def validate_spatial_tail_result_bytes(raw: bytes) -> dict[str, object]:
    """Validate canonical result bytes and independently recompute summaries."""

    if type(raw) is not bytes:
        raise ValueError("spatial tail result bytes differ")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("spatial tail result is not JSON") from error
    keys = {
        "schema",
        "claim_eligible",
        "external_evaluation_access",
        "checkpoint_sha256",
        "optimization_manifest_sha256",
        "artifact_sha256",
        "fit_labels",
        "development_labels",
        "cells",
        "training",
        "classification",
        "recall_gap_closure",
        "map_gap_closure",
    }
    if type(value) is not dict or set(value) != keys:
        raise ValueError("spatial tail result schema differs")
    result = cast(dict[str, object], value)
    if (
        result["schema"] != "sfora-siglip-spatial-tail-recovery-v1"
        or result["claim_eligible"] is not False
        or result["external_evaluation_access"] is not False
    ):
        raise ValueError("spatial tail result authority differs")
    for key in ("checkpoint_sha256", "optimization_manifest_sha256", "artifact_sha256"):
        _hex_digest(result[key])
    fit, development = spatial_tail_class_split(tuple(range(49)))
    if result["fit_labels"] != sorted(fit) or result["development_labels"] != sorted(development):
        raise ValueError("spatial tail class authority differs")
    _training(result["training"])
    names = ("baseline", "tokenwise-control", "latent-interaction", "teacher")
    if type(result["cells"]) is not dict or set(
        cast(dict[str, object], result["cells"])
    ) != set(names):
        raise ValueError("spatial tail cell inventory differs")
    cells_value = cast(dict[str, object], result["cells"])
    parsed: dict[str, dict[str, object]] = {}
    for name in names:
        cell = cells_value[name]
        if type(cell) is not dict or set(cell) != {"self", "cross"}:
            raise ValueError("spatial tail cell schema differs")
        parsed[name] = {
            "self": _retrieval_mapping(cell["self"], summarized=True),
            "cross": _retrieval_mapping(cell["cross"], summarized=True),
        }
    decision = spatial_tail_decision(
        {
            name: {
                key: parsed[name]["self"][key]
                for key in ("queries", "correct", "map_at_r")
            }
            for name in names
        }
    )
    if any(result[key] != decision[key] for key in decision):
        raise ValueError("spatial tail decision differs")
    try:
        canonical = _canonical_bytes(value)
    except (TypeError, ValueError) as error:
        raise ValueError("spatial tail result is not canonical") from error
    if raw != canonical:
        raise ValueError("spatial tail result is not canonical")
    return result
