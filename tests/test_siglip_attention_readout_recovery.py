"""Tests for deterministic teacher-coordinate SigLIP depth readouts."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, cast

import pytest
import torch

from sfora.siglip_attention_readout_recovery import (
    apply_readout,
    attention_readout_decision,
    build_attention_readout_result,
    fit_ridge_readout,
    refine_directional_readout,
    validate_attention_readout_result_bytes,
)


def test_teacher_anchored_ridge_matches_hand_derived_normal_equations() -> None:
    """Changing the anchor term, N scaling, or matrix orientation must fail."""

    features = torch.eye(2, dtype=torch.float32)
    targets = torch.eye(2, dtype=torch.float32)
    teacher_weight = torch.zeros((2, 2), dtype=torch.float32)

    weight = fit_ridge_readout(features, targets, teacher_weight)

    expected = torch.diag(torch.tensor([0.9990010261535645, 0.9990010261535645]))
    assert weight.shape == (2, 2)
    assert weight.dtype == torch.float32
    assert weight.device.type == "cpu"
    assert weight.is_contiguous()
    assert torch.equal(weight, expected)
    assert torch.equal(apply_readout(features, weight), torch.eye(2))


@pytest.mark.parametrize(
    ("features", "targets", "teacher_weight"),
    [
        (
            torch.tensor([[1.0, float("nan")], [0.0, 1.0]]),
            torch.eye(2),
            torch.zeros((2, 2)),
        ),
        (torch.eye(2, dtype=torch.float64), torch.eye(2), torch.zeros((2, 2))),
        (torch.eye(2), torch.ones((1, 2)), torch.zeros((2, 2))),
        (torch.eye(2), torch.eye(2), torch.zeros((2, 3))),
    ],
)
def test_ridge_rejects_nonfinite_dtype_and_shape_drift(
    features: torch.Tensor,
    targets: torch.Tensor,
    teacher_weight: torch.Tensor,
) -> None:
    """Relaxing concrete tensor authority would make sealed heads incomparable."""

    with pytest.raises(ValueError, match="readout tensor authority differs"):
        fit_ridge_readout(features, targets, teacher_weight)


def test_apply_readout_rejects_zero_or_nonfinite_descriptors() -> None:
    """Returning NaN or arbitrary zero-normalized rows would corrupt retrieval."""

    for features, weight in (
        (torch.zeros((2, 2)), torch.eye(2)),
        (torch.eye(2), torch.tensor([[1.0, float("inf")], [0.0, 1.0]])),
    ):
        with pytest.raises(ValueError, match="readout descriptor authority differs"):
            apply_readout(features, weight)


def test_directional_refinement_is_deterministic_and_does_not_consume_global_rng() -> None:
    """Global RNG use or a wrong fixed schedule would make sealed heads irreproducible."""

    features = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, -1.0]], dtype=torch.float32)
    targets = torch.nn.functional.normalize(
        torch.tensor([[1.0, 1.0], [-1.0, 1.0], [0.0, 1.0], [1.0, 0.0]]), dim=1
    )
    initial = torch.eye(2, dtype=torch.float32)
    state_before = torch.random.get_rng_state().clone()

    left = refine_directional_readout(features, targets, initial)
    right = refine_directional_readout(features, targets, initial)

    assert torch.equal(torch.random.get_rng_state(), state_before)
    assert torch.equal(left.weight, right.weight)
    assert left.initial_loss == right.initial_loss
    assert left.final_loss == right.final_loss
    assert left.final_loss < left.initial_loss
    assert len(left.final_200_losses) == 200
    assert torch.equal(initial, torch.eye(2))


def _cell(
    depth: int,
    fit: str,
    *,
    self_hits: int = 2_400,
    self_map: float = 0.70,
    cross_hits: int = 2_400,
    cross_map: float = 0.70,
) -> dict[str, object]:
    return {
        "depth": depth,
        "fit": fit,
        "optimization_limited": False,
        "self": {"queries": 2_746, "correct": self_hits, "map_at_r": self_map},
        "cross": {"queries": 2_746, "correct": cross_hits, "map_at_r": cross_map},
    }


def _cells() -> list[dict[str, object]]:
    return [
        _cell(depth, fit) for depth in (6, 10, 14, 18, 22, 25, 27) for fit in ("ridge", "refined")
    ]


def test_decision_preserves_deployment_gate_and_prefers_smallest_qualifying_depth() -> None:
    """A later or merely promising cell must not replace an earlier deployment-grade one."""

    cells = _cells()
    cells[4] = _cell(14, "ridge", self_hits=2_591, self_map=0.79, cross_hits=2_591, cross_map=0.79)
    cells[6] = _cell(18, "ridge", self_hits=2_600, self_map=0.80, cross_hits=2_600, cross_map=0.80)

    assert attention_readout_decision(cells) == {
        "classification": "deployment-grade",
        "selected_depth": 14,
        "selected_fit": "ridge",
    }


@pytest.mark.parametrize(
    ("self_hits", "self_map", "cross_hits", "cross_map", "classification"),
    [
        (2_555, 0.772, 2_555, 0.772, "compression-promising"),
        (2_555, 0.772, 2_400, 0.70, "nonlinear-alignment-needed"),
        (2_554, 0.90, 2_700, 0.90, "linear-readout-rejected"),
    ],
)
def test_decision_classifies_quality_and_coordinate_failures_separately(
    self_hits: int,
    self_map: float,
    cross_hits: int,
    cross_map: float,
    classification: str,
) -> None:
    """Collapsing self and cross evidence would prescribe the wrong next architecture."""

    cells = _cells()
    cells[6] = _cell(
        18,
        "ridge",
        self_hits=self_hits,
        self_map=self_map,
        cross_hits=cross_hits,
        cross_map=cross_map,
    )
    decision = attention_readout_decision(cells)
    assert decision["classification"] == classification
    selected_depth = 18 if classification != "linear-readout-rejected" else None
    assert decision["selected_depth"] == selected_depth


def _query_evidence(correct: int, average_precision: float) -> dict[str, object]:
    return {
        "hits": [ordinal < correct for ordinal in range(2_746)],
        "average_precision": [average_precision] * 2_746,
    }


def _result_cells() -> list[dict[str, object]]:
    cells: list[dict[str, object]] = []
    for depth in (6, 10, 14, 18, 22, 25, 27):
        for fit in ("ridge", "refined"):
            correct = 2_591 if (depth, fit) == (18, "ridge") else 2_400
            average_precision = 0.79 if (depth, fit) == (18, "ridge") else 0.70
            cells.append(
                {
                    "depth": depth,
                    "fit": fit,
                    "optimization_limited": False,
                    "weight_sha256": f"{depth:02x}" * 32,
                    "self": _query_evidence(correct, average_precision),
                    "cross": _query_evidence(correct, average_precision),
                }
            )
    return cells


def _build_result() -> bytes:
    return build_attention_readout_result(
        _result_cells(),
        checkpoint_sha256="11" * 32,
        optimization_manifest_sha256="22" * 32,
        evaluation_manifest_sha256="33" * 32,
        depth_27_identity=True,
        random_projection_passed=False,
        target_derangement_passed=False,
    )


def test_result_is_canonical_and_recomputes_every_metric_and_decision() -> None:
    """Trusted summaries could hide a retrieval or selection implementation error."""

    raw = _build_result()
    result = validate_attention_readout_result_bytes(raw)

    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")
    assert result["classification"] == "deployment-grade"
    assert result["selected_depth"] == 18
    assert result["selected_fit"] == "ridge"
    result_cells = cast(list[dict[str, object]], result["cells"])
    selected = result_cells[6]
    selected_self = cast(dict[str, object], selected["self"])
    assert selected_self["correct"] == 2_591
    assert selected_self["map_at_r"] == 0.79


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda value: value.update({"claim_eligible": 0}), "result authority differs"),
        (lambda value: value.update({"checkpoint_sha256": "x" * 64}), "digest differs"),
        (
            lambda value: value["cells"][6]["self"].update({"correct": 2_590}),
            "retrieval relation differs",
        ),
        (
            lambda value: value.update({"selected_depth": 14}),
            "decision relation differs",
        ),
        (
            lambda value: value["cells"][6]["self"]["hits"].append(False),
            "retrieval evidence differs",
        ),
    ],
)
def test_result_rejects_schema_type_digest_metric_and_decision_drift(
    mutation: Callable[[dict[str, Any]], None],
    match: str,
) -> None:
    """Every stored summary and authority field must remain independently derivable."""

    value = json.loads(_build_result())
    mutation(value)
    raw = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    with pytest.raises(ValueError, match=match):
        validate_attention_readout_result_bytes(raw)


def test_result_rejects_noncanonical_bytes() -> None:
    """Byte identity must not depend on writer whitespace or key ordering."""

    value = json.loads(_build_result())
    raw = (json.dumps(value, indent=2) + "\n").encode()
    with pytest.raises(ValueError, match="result is not canonical"):
        validate_attention_readout_result_bytes(raw)
