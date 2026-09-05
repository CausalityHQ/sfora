"""Tests for class-isolated SigLIP spatial-tail recovery evidence."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import pytest
import torch
from torch.nn import functional as F

from sfora.siglip_spatial_tail_recovery import (
    build_spatial_tail_result,
    spatial_retrieval_evidence,
    spatial_tail_class_split,
    spatial_tail_decision,
    validate_spatial_tail_result_bytes,
)


def test_class_split_is_exact_disjoint_and_order_independent() -> None:
    fit, development = spatial_tail_class_split(tuple(range(49)))
    reverse_fit, reverse_development = spatial_tail_class_split(tuple(reversed(range(49))))

    assert len(fit) == 39
    assert len(development) == 10
    assert fit.isdisjoint(development)
    assert fit | development == frozenset(range(49))
    assert (fit, development) == (reverse_fit, reverse_development)
    assert development == frozenset({4, 5, 12, 15, 19, 24, 26, 32, 40, 45})


@pytest.mark.parametrize(
    "labels",
    [tuple(range(48)), tuple(range(50)), tuple(range(48)) + (46,)],
)
def test_class_split_rejects_incomplete_extra_or_duplicate_labels(labels: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="class authority differs"):
        spatial_tail_class_split(labels)


def _descriptors() -> tuple[torch.Tensor, tuple[str, ...], tuple[int, ...]]:
    values = F.normalize(
        torch.tensor(
            [
                [1.0, 0.1],
                [1.0, -0.1],
                [0.1, 1.0],
                [-0.1, 1.0],
                [-1.0, 0.1],
                [-1.0, -0.1],
            ],
            dtype=torch.float32,
        ),
        dim=1,
    ).contiguous()
    return values, tuple("abcdef"), (0, 0, 1, 1, 2, 2)


def test_retrieval_evidence_binds_permuted_gallery_identity() -> None:
    descriptors, ids, labels = _descriptors()
    permutation = torch.tensor([2, 4, 0, 5, 3, 1])
    evidence = spatial_retrieval_evidence(
        descriptors,
        descriptors[permutation].contiguous(),
        query_ids=ids,
        gallery_ids=tuple(ids[index] for index in permutation),
        query_labels=labels,
        gallery_labels=tuple(labels[index] for index in permutation),
    )
    assert evidence.correct == (True,) * 6
    assert evidence.average_precisions == (1.0,) * 6
    assert evidence.map_at_r == 1.0


def _summary(correct: int, map_at_r: float) -> dict[str, object]:
    return {"queries": 100, "correct": correct, "map_at_r": map_at_r}


def test_decision_requires_treatment_to_beat_control_and_close_half_both_gaps() -> None:
    cells = {
        "baseline": _summary(70, 0.40),
        "tokenwise-control": _summary(78, 0.55),
        "latent-interaction": _summary(86, 0.71),
        "teacher": _summary(100, 1.0),
    }
    assert spatial_tail_decision(cells) == {
        "classification": "latency-pending",
        "recall_gap_closure": 16 / 30,
        "map_gap_closure": (0.71 - 0.40) / (1.0 - 0.40),
    }
    cells["latent-interaction"] = _summary(84, 0.69)
    assert spatial_tail_decision(cells)["classification"] == "interaction-rejected"
    cells["tokenwise-control"] = _summary(86, 0.69)
    assert spatial_tail_decision(cells)["classification"] == "interaction-rejected"


def _query_evidence(correct: int, ap: float) -> dict[str, object]:
    return {
        "hits": [index < correct for index in range(100)],
        "average_precision": [ap] * 100,
    }


def _result_bytes() -> bytes:
    return build_spatial_tail_result(
        checkpoint_sha256="11" * 32,
        optimization_manifest_sha256="22" * 32,
        artifact_sha256="33" * 32,
        fit_labels=tuple(sorted(set(range(49)) - {4, 5, 12, 15, 19, 24, 26, 32, 40, 45})),
        development_labels=(4, 5, 12, 15, 19, 24, 26, 32, 40, 45),
        cells={
            "baseline": {"self": _query_evidence(70, 0.40), "cross": _query_evidence(50, 0.30)},
            "tokenwise-control": {
                "self": _query_evidence(78, 0.55),
                "cross": _query_evidence(60, 0.40),
            },
            "latent-interaction": {
                "self": _query_evidence(86, 0.71),
                "cross": _query_evidence(75, 0.60),
            },
            "teacher": {"self": _query_evidence(100, 1.0), "cross": _query_evidence(100, 1.0)},
        },
        training={
            "updates": 4000,
            "tokenwise_initial_loss": 2.0,
            "tokenwise_final_loss": 0.2,
            "interaction_initial_loss": 2.0,
            "interaction_final_loss": 0.1,
        },
    )


def test_result_is_canonical_and_recomputes_metrics_and_decision() -> None:
    raw = _result_bytes()
    value = validate_spatial_tail_result_bytes(raw)
    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")
    assert value["classification"] == "latency-pending"
    assert value["claim_eligible"] is False
    assert value["external_evaluation_access"] is False
    assert value["recall_gap_closure"] == 16 / 30
    assert value["map_gap_closure"] == (0.71 - 0.40) / (1.0 - 0.40)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda value: value.update({"claim_eligible": 0}), "authority differs"),
        (lambda value: value.update({"artifact_sha256": "x" * 64}), "digest differs"),
        (
            lambda value: value["cells"]["latent-interaction"]["self"].update({"correct": 85}),
            "retrieval relation differs",
        ),
        (lambda value: value.update({"classification": "quality-qualified"}), "decision differs"),
        (lambda value: value["development_labels"].append(49), "class authority differs"),
        (
            lambda value: value["training"].update({"interaction_final_loss": float("nan")}),
            "training evidence differs",
        ),
    ],
)
def test_result_rejects_authority_digest_metric_decision_and_class_drift(
    mutation: Callable[[dict[str, Any]], None], match: str
) -> None:
    value = json.loads(_result_bytes())
    mutation(value)
    raw = (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=True) + "\n").encode()
    with pytest.raises(ValueError, match=match):
        validate_spatial_tail_result_bytes(raw)
