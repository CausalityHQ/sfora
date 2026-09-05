"""Deterministic teacher-coordinate readouts for frozen SigLIP depth features."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import cast

import torch
from torch.nn import functional as F


@dataclass(frozen=True, slots=True)
class DirectionalReadoutEvidence:
    """Final-only fixed-budget readout and its optimization-loss evidence."""

    weight: torch.Tensor
    initial_loss: float
    final_loss: float
    final_200_losses: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class AttentionRetrievalEvidence:
    """Exact per-query hit and AP@R evidence in query identity order."""

    correct: tuple[bool, ...]
    average_precisions: tuple[float, ...]

    @property
    def map_at_r(self) -> float:
        return math.fsum(self.average_precisions) / len(self.average_precisions)


def attention_retrieval_evidence(
    query_descriptors: torch.Tensor,
    gallery_descriptors: torch.Tensor,
    *,
    query_ids: tuple[str, ...],
    gallery_ids: tuple[str, ...],
    query_labels: tuple[int, ...],
    gallery_labels: tuple[int, ...],
) -> AttentionRetrievalEvidence:
    """Rank query descriptors against an identity-aligned, potentially permuted gallery."""

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
        or tuple(query_descriptors.shape) != tuple(gallery_descriptors.shape)
        or query_descriptors.shape != (rows, query_descriptors.shape[1])
        or query_descriptors.shape[1] < 1
        or not bool(torch.isfinite(query_descriptors).all())
        or not bool(torch.isfinite(gallery_descriptors).all())
    ):
        raise ValueError("attention retrieval authority differs")
    gallery_by_id = dict(zip(gallery_ids, range(rows), strict=True))
    gallery_label_by_id = dict(zip(gallery_ids, gallery_labels, strict=True))
    if tuple(gallery_label_by_id[value] for value in query_ids) != query_labels:
        raise ValueError("attention retrieval label binding differs")
    canonical_gallery = gallery_descriptors[
        torch.tensor([gallery_by_id[value] for value in query_ids], dtype=torch.int64)
    ]
    query_norms = torch.linalg.vector_norm(query_descriptors, dim=1)
    gallery_norms = torch.linalg.vector_norm(canonical_gallery, dim=1)
    if not bool((query_norms > 0).all()) or not bool((gallery_norms > 0).all()):
        raise ValueError("attention retrieval descriptors must have nonzero norms")
    queries = F.normalize(query_descriptors, dim=1)
    gallery = F.normalize(canonical_gallery, dim=1)
    counts = {label: query_labels.count(label) for label in set(query_labels)}
    if min(counts.values()) < 2:
        raise ValueError("attention retrieval classes require at least two images")
    label_tensor = torch.tensor(query_labels)
    correct: list[bool] = []
    average_precisions: list[float] = []
    for start in range(0, rows, 128):
        stop = min(start + 128, rows)
        scores = queries[start:stop] @ gallery.T
        scores[torch.arange(stop - start), torch.arange(start, stop)] = -torch.inf
        ranked = torch.argsort(scores, dim=1, descending=True, stable=True)
        for local, row in enumerate(range(start, stop)):
            first = int(ranked[local, 0])
            correct.append(query_labels[first] == query_labels[row])
            retained = ranked[local, : counts[query_labels[row]] - 1]
            relevant = (label_tensor[retained] == query_labels[row]).tolist()
            hits = 0
            terms = []
            for rank, hit in enumerate(relevant, 1):
                if hit:
                    hits += 1
                    terms.append(hits / rank)
            average_precisions.append(math.fsum(terms) / len(relevant))
    return AttentionRetrievalEvidence(tuple(correct), tuple(average_precisions))


def _readout_tensors(
    features: torch.Tensor,
    targets: torch.Tensor,
    teacher_weight: torch.Tensor,
) -> tuple[int, int, int]:
    if (
        type(features) is not torch.Tensor
        or type(targets) is not torch.Tensor
        or type(teacher_weight) is not torch.Tensor
        or features.device.type != "cpu"
        or targets.device.type != "cpu"
        or teacher_weight.device.type != "cpu"
        or features.dtype != torch.float32
        or targets.dtype != torch.float32
        or teacher_weight.dtype != torch.float32
        or features.ndim != 2
        or targets.ndim != 2
        or teacher_weight.ndim != 2
        or features.shape[0] < 2
        or features.shape[1] < 1
        or targets.shape[0] != features.shape[0]
        or targets.shape[1] < 1
        or teacher_weight.shape != (targets.shape[1], features.shape[1])
        or not features.is_contiguous()
        or not targets.is_contiguous()
        or not teacher_weight.is_contiguous()
        or not bool(torch.isfinite(features).all())
        or not bool(torch.isfinite(targets).all())
        or not bool(torch.isfinite(teacher_weight).all())
    ):
        raise ValueError("readout tensor authority differs")
    return features.shape[0], features.shape[1], targets.shape[1]


def fit_ridge_readout(
    features: torch.Tensor,
    targets: torch.Tensor,
    teacher_weight: torch.Tensor,
) -> torch.Tensor:
    """Fit the fixed teacher-anchored FP64 ridge readout and return FP32 weight."""

    rows, dimensions, _ = _readout_tensors(features, targets, teacher_weight)
    x = features.double()
    y = targets.double()
    anchor = teacher_weight.T.double()
    scale_squared = torch.sum(x * x) / (dimensions * rows)
    penalty = rows * 1.0e-3 * scale_squared
    gram = x.T @ x
    gram.diagonal().add_(penalty)
    right = x.T @ y + penalty * anchor
    factor = torch.linalg.cholesky(gram)
    solution = torch.cholesky_solve(right, factor)
    if not bool(torch.isfinite(solution).all()):
        raise ValueError("readout ridge solution is nonfinite")
    return solution.T.float().contiguous()


def apply_readout(features: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Apply a finite bias-free readout and return unit FP32 descriptors."""

    if (
        type(features) is not torch.Tensor
        or type(weight) is not torch.Tensor
        or features.device.type != "cpu"
        or weight.device.type != "cpu"
        or features.dtype != torch.float32
        or weight.dtype != torch.float32
        or features.ndim != 2
        or weight.ndim != 2
        or features.shape[0] < 2
        or features.shape[1] != weight.shape[1]
        or weight.shape[0] < 1
        or not features.is_contiguous()
        or not weight.is_contiguous()
        or not bool(torch.isfinite(features).all())
        or not bool(torch.isfinite(weight).all())
    ):
        raise ValueError("readout descriptor authority differs")
    projected = features @ weight.T
    norms = torch.linalg.vector_norm(projected, dim=1)
    if not bool(torch.isfinite(projected).all()) or bool((norms <= 0).any()):
        raise ValueError("readout descriptor authority differs")
    return F.normalize(projected, dim=1).contiguous()


def _directional_loss(
    features: torch.Tensor, targets: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    descriptors = F.normalize(features @ weight.T, dim=1)
    return (1.0 - torch.sum(descriptors * targets, dim=1)).mean()


def refine_directional_readout(
    features: torch.Tensor,
    targets: torch.Tensor,
    initial_weight: torch.Tensor,
) -> DirectionalReadoutEvidence:
    """Run the fixed 2,000-update final-only directional refinement."""

    _readout_tensors(features, targets, initial_weight)
    target_norms = torch.linalg.vector_norm(targets, dim=1)
    if not torch.allclose(target_norms, torch.ones_like(target_norms), atol=1.0e-6, rtol=0.0):
        raise ValueError("readout directional target authority differs")
    weight = torch.nn.Parameter(initial_weight.detach().clone())
    optimizer = torch.optim.AdamW(
        (weight,), lr=1.0e-4, betas=(0.9, 0.999), eps=1.0e-8, weight_decay=0.0, foreach=False
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(17)
    pending = torch.empty(0, dtype=torch.int64)
    losses: list[float] = []
    initial_loss = float(_directional_loss(features, targets, weight.detach()))
    for update in range(1, 2_001):
        while pending.numel() < 256:
            pending = torch.cat((pending, torch.randperm(features.shape[0], generator=generator)))
        indexes, pending = pending[:256], pending[256:]
        if update <= 50:
            learning_rate = 1.0e-4 * update / 50.0
        else:
            progress = (update - 50) / 1_950
            learning_rate = 1.0e-5 + 0.5 * (1.0e-4 - 1.0e-5) * (1.0 + math.cos(math.pi * progress))
        optimizer.param_groups[0]["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        loss = _directional_loss(features[indexes], targets[indexes], weight)
        if not bool(torch.isfinite(loss)):
            raise ValueError("readout directional loss is nonfinite")
        torch.autograd.backward(loss)
        torch.nn.utils.clip_grad_norm_((weight,), 1.0)
        optimizer.step()
        if update > 1_800:
            losses.append(float(loss.detach()))
    final_weight = weight.detach().contiguous()
    final_loss = float(_directional_loss(features, targets, final_weight))
    if not math.isfinite(final_loss):
        raise ValueError("readout directional loss is nonfinite")
    return DirectionalReadoutEvidence(
        weight=final_weight,
        initial_loss=initial_loss,
        final_loss=final_loss,
        final_200_losses=tuple(losses),
    )


def _quality_cell(value: object) -> tuple[int, float]:
    if type(value) is not dict or set(value) != {"queries", "correct", "map_at_r"}:
        raise ValueError("attention readout quality schema differs")
    cell = cast(dict[str, object], value)
    if (
        type(cell["queries"]) is not int
        or cell["queries"] != 2_746
        or type(cell["correct"]) is not int
        or not 0 <= cell["correct"] <= 2_746
        or type(cell["map_at_r"]) is not float
        or not math.isfinite(cell["map_at_r"])
        or not 0.0 <= cell["map_at_r"] <= 1.0
    ):
        raise ValueError("attention readout quality authority differs")
    return cell["correct"], cell["map_at_r"]


def _expected_cells() -> tuple[tuple[int, str], ...]:
    cells = [(depth, fit) for depth in (6, 10, 14, 18, 22, 25, 27) for fit in ("ridge", "refined")]
    cells.insert(8, (18, "learned-attention"))
    return tuple(cells)


def attention_readout_decision(cells: list[dict[str, object]]) -> dict[str, object]:
    """Classify the complete fixed depth/fit inventory using frozen quality gates."""

    expected = _expected_cells()
    if type(cells) is not list or len(cells) != len(expected):
        raise ValueError("attention readout cell inventory differs")
    parsed = []
    for value, identity in zip(cells, expected, strict=True):
        if type(value) is not dict or set(value) != {
            "depth",
            "fit",
            "optimization_limited",
            "self",
            "cross",
        }:
            raise ValueError("attention readout cell schema differs")
        if (
            type(value["depth"]) is not int
            or type(value["fit"]) is not str
            or (value["depth"], value["fit"]) != identity
            or type(value["optimization_limited"]) is not bool
        ):
            raise ValueError("attention readout cell identity differs")
        self_hits, self_map = _quality_cell(value["self"])
        cross_hits, cross_map = _quality_cell(value["cross"])
        parsed.append((identity[0], identity[1], self_hits, self_map, cross_hits, cross_map))

    learned = parsed[8]
    self_promising = learned[2] >= 2_555 and learned[3] >= 0.7713744556922272
    cross_promising = learned[4] >= 2_555 and learned[5] >= 0.7713744556922272
    selected: tuple[int | None, str | None]
    if (
        learned[2] >= 2_591
        and learned[3] >= 0.7893744556922272
        and learned[4] >= 2_591
        and learned[5] >= 0.7893744556922272
    ):
        classification, selected = "quality-qualified", (18, "learned-attention")
    elif self_promising and cross_promising:
        classification, selected = "compression-promising", (18, "learned-attention")
    elif self_promising:
        classification, selected = "coordinate-alignment-needed", (18, "learned-attention")
    else:
        classification, selected = "depth-18-recovery-rejected", (None, None)
    return {
        "classification": classification,
        "selected_depth": selected[0],
        "selected_fit": selected[1],
    }


def _hex_digest(value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("attention readout digest differs")
    return value


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def _retrieval_mapping(value: object, *, summarized: bool) -> dict[str, object]:
    keys = {"hits", "average_precision"}
    if summarized:
        keys |= {"queries", "correct", "map_at_r"}
    if type(value) is not dict or set(value) != keys:
        raise ValueError("attention readout retrieval evidence differs")
    evidence = cast(dict[str, object], value)
    hits_value = evidence["hits"]
    ap_value = evidence["average_precision"]
    if (
        type(hits_value) is not list
        or type(ap_value) is not list
        or len(hits_value) != 2_746
        or len(ap_value) != 2_746
        or any(type(hit) is not bool for hit in hits_value)
        or any(
            type(score) is not float or not math.isfinite(score) or not 0.0 <= score <= 1.0
            for score in ap_value
        )
    ):
        raise ValueError("attention readout retrieval evidence differs")
    hits = cast(list[bool], hits_value)
    average_precision = cast(list[float], ap_value)
    correct = sum(hits)
    map_at_r = math.fsum(average_precision) / len(average_precision)
    if summarized and (
        type(evidence["queries"]) is not int
        or evidence["queries"] != 2_746
        or type(evidence["correct"]) is not int
        or evidence["correct"] != correct
        or type(evidence["map_at_r"]) is not float
        or evidence["map_at_r"] != map_at_r
    ):
        raise ValueError("attention readout retrieval relation differs")
    return {
        "queries": 2_746,
        "correct": correct,
        "map_at_r": map_at_r,
        "hits": hits.copy(),
        "average_precision": average_precision.copy(),
    }


def _decision_cells(cells: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "depth": cell["depth"],
            "fit": cell["fit"],
            "optimization_limited": cell["optimization_limited"],
            "self": {
                key: cast(dict[str, object], cell["self"])[key]
                for key in ("queries", "correct", "map_at_r")
            },
            "cross": {
                key: cast(dict[str, object], cell["cross"])[key]
                for key in ("queries", "correct", "map_at_r")
            },
        }
        for cell in cells
    ]


def _optimization_evidence(value: object) -> dict[str, object]:
    if type(value) is not dict or set(value) != {
        "initial_loss",
        "final_loss",
        "final_200_losses",
    }:
        raise ValueError("attention readout optimization evidence differs")
    evidence = cast(dict[str, object], value)
    losses = evidence["final_200_losses"]
    if (
        type(evidence["initial_loss"]) is not float
        or not math.isfinite(cast(float, evidence["initial_loss"]))
        or type(evidence["final_loss"]) is not float
        or not math.isfinite(cast(float, evidence["final_loss"]))
        or type(losses) is not list
        or len(losses) != 200
        or any(type(loss) is not float or not math.isfinite(loss) for loss in losses)
    ):
        raise ValueError("attention readout optimization evidence differs")
    return {
        "initial_loss": evidence["initial_loss"],
        "final_loss": evidence["final_loss"],
        "final_200_losses": cast(list[float], losses).copy(),
    }


def build_attention_readout_result(
    cells: list[dict[str, object]],
    *,
    checkpoint_sha256: str,
    optimization_manifest_sha256: str,
    evaluation_manifest_sha256: str,
    readout_artifact_sha256: str,
    depth_27_identity: bool,
    learned_attention_optimization: dict[str, object],
) -> bytes:
    """Build canonical per-query evidence and validate it independently."""

    expected = _expected_cells()
    if type(cells) is not list or len(cells) != len(expected):
        raise ValueError("attention readout result cell inventory differs")
    output_cells: list[dict[str, object]] = []
    for cell, identity in zip(cells, expected, strict=True):
        if type(cell) is not dict or set(cell) != {
            "depth",
            "fit",
            "optimization_limited",
            "weight_sha256",
            "self",
            "cross",
        }:
            raise ValueError("attention readout result cell schema differs")
        if (
            type(cell["depth"]) is not int
            or type(cell["fit"]) is not str
            or (cell["depth"], cell["fit"]) != identity
            or type(cell["optimization_limited"]) is not bool
        ):
            raise ValueError("attention readout result cell identity differs")
        output_cells.append(
            {
                "depth": cell["depth"],
                "fit": cell["fit"],
                "optimization_limited": cell["optimization_limited"],
                "weight_sha256": _hex_digest(cell["weight_sha256"]),
                "self": _retrieval_mapping(cell["self"], summarized=False),
                "cross": _retrieval_mapping(cell["cross"], summarized=False),
            }
        )
    if type(depth_27_identity) is not bool or not depth_27_identity:
        raise ValueError("attention readout result authority differs")
    decision = attention_readout_decision(_decision_cells(output_cells))
    result = {
        "schema": "sfora-siglip-attention-readout-recovery-v1",
        "claim_eligible": False,
        "official_test_access": False,
        "checkpoint_sha256": _hex_digest(checkpoint_sha256),
        "optimization_manifest_sha256": _hex_digest(optimization_manifest_sha256),
        "evaluation_manifest_sha256": _hex_digest(evaluation_manifest_sha256),
        "readout_artifact_sha256": _hex_digest(readout_artifact_sha256),
        "depth_27_identity": depth_27_identity,
        "learned_attention_optimization": _optimization_evidence(learned_attention_optimization),
        "cells": output_cells,
        **decision,
    }
    raw = _canonical_bytes(result)
    validate_attention_readout_result_bytes(raw)
    return raw


def validate_attention_readout_result_bytes(raw: bytes) -> dict[str, object]:
    """Validate canonical bytes and recompute every aggregate and decision."""

    if type(raw) is not bytes:
        raise ValueError("attention readout result bytes differ")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("attention readout result is not JSON") from error
    if raw != _canonical_bytes(value):
        raise ValueError("attention readout result is not canonical")
    keys = {
        "schema",
        "claim_eligible",
        "official_test_access",
        "checkpoint_sha256",
        "optimization_manifest_sha256",
        "evaluation_manifest_sha256",
        "readout_artifact_sha256",
        "depth_27_identity",
        "learned_attention_optimization",
        "cells",
        "classification",
        "selected_depth",
        "selected_fit",
    }
    if type(value) is not dict or set(value) != keys:
        raise ValueError("attention readout result schema differs")
    result = cast(dict[str, object], value)
    if (
        result["schema"] != "sfora-siglip-attention-readout-recovery-v1"
        or result["claim_eligible"] is not False
        or result["official_test_access"] is not False
        or result["depth_27_identity"] is not True
    ):
        raise ValueError("attention readout result authority differs")
    _optimization_evidence(result["learned_attention_optimization"])
    for key in (
        "checkpoint_sha256",
        "optimization_manifest_sha256",
        "evaluation_manifest_sha256",
        "readout_artifact_sha256",
    ):
        _hex_digest(result[key])
    if type(result["cells"]) is not list or len(result["cells"]) != 15:
        raise ValueError("attention readout result cell inventory differs")
    expected = _expected_cells()
    cells: list[dict[str, object]] = []
    for value_cell, identity in zip(cast(list[object], result["cells"]), expected, strict=True):
        if type(value_cell) is not dict or set(value_cell) != {
            "depth",
            "fit",
            "optimization_limited",
            "weight_sha256",
            "self",
            "cross",
        }:
            raise ValueError("attention readout result cell schema differs")
        cell = cast(dict[str, object], value_cell)
        if (
            type(cell["depth"]) is not int
            or type(cell["fit"]) is not str
            or (cell["depth"], cell["fit"]) != identity
            or type(cell["optimization_limited"]) is not bool
        ):
            raise ValueError("attention readout result cell identity differs")
        _hex_digest(cell["weight_sha256"])
        cells.append(
            {
                **cell,
                "self": _retrieval_mapping(cell["self"], summarized=True),
                "cross": _retrieval_mapping(cell["cross"], summarized=True),
            }
        )
    decision = attention_readout_decision(_decision_cells(cells))
    if any(result[key] != decision[key] for key in decision):
        raise ValueError("attention readout decision relation differs")
    return {**result, "cells": cells}
