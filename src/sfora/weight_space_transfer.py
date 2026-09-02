"""Pure weight-space transfer interpolation authority."""

from __future__ import annotations

import hashlib
import json
import math
from collections import OrderedDict
from dataclasses import dataclass

import torch

INTERPOLATION_ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)


@dataclass(frozen=True)
class FoldedInferenceState:
    """One deterministic tower fold with trained retrieval-head state."""

    alpha: float
    state: OrderedDict[str, torch.Tensor]
    sha256: str
    tower_squared_displacement: float


@dataclass(frozen=True)
class AlphaEvaluation:
    """One authenticated alpha row with query-level paired evidence."""

    seed: int
    alpha: float
    correct: int
    queries: int
    recall_ppm: int
    mean_margin: float
    correctness: tuple[bool, ...]
    folded_state_sha256: str
    tower_squared_displacement: float

    def __post_init__(self) -> None:
        if type(self.seed) is not int or self.seed not in (17, 29, 43):
            raise ValueError("alpha row seed differs from authority")
        if type(self.alpha) is not float or self.alpha not in INTERPOLATION_ALPHAS:
            raise ValueError("alpha row alpha differs from authority")
        if type(self.correct) is not int or type(self.queries) is not int:
            raise ValueError("alpha row counts must be concrete integers")
        if not 0 <= self.correct <= self.queries or self.queries <= 0:
            raise ValueError("alpha row counts differ")
        if type(self.recall_ppm) is not int or self.recall_ppm != (
            self.correct * 1_000_000 // self.queries
        ):
            raise ValueError("alpha row recall arithmetic differs")
        if type(self.mean_margin) is not float or not math.isfinite(self.mean_margin):
            raise ValueError("alpha row margin must be a concrete finite float")
        if (
            type(self.correctness) is not tuple
            or len(self.correctness) != self.queries
            or any(type(value) is not bool for value in self.correctness)
            or sum(self.correctness) != self.correct
        ):
            raise ValueError("alpha row correctness evidence differs")
        if (
            type(self.folded_state_sha256) is not str
            or len(self.folded_state_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.folded_state_sha256)
        ):
            raise ValueError("alpha row state digest differs")
        if (
            type(self.tower_squared_displacement) is not float
            or not math.isfinite(self.tower_squared_displacement)
            or self.tower_squared_displacement < 0.0
        ):
            raise ValueError("alpha row tower displacement differs")


@dataclass(frozen=True)
class SeedInterpolationCurve:
    """The exact ordered five-alpha curve for one seed."""

    seed: int
    rows: tuple[AlphaEvaluation, ...]

    def __post_init__(self) -> None:
        if type(self.seed) is not int or self.seed not in (17, 29, 43):
            raise ValueError("curve seed differs from authority")
        if (
            type(self.rows) is not tuple
            or tuple(row.alpha for row in self.rows) != INTERPOLATION_ALPHAS
            or any(type(row) is not AlphaEvaluation or row.seed != self.seed for row in self.rows)
            or len({row.queries for row in self.rows}) != 1
        ):
            raise ValueError("seed curve rows differ from authority")


@dataclass(frozen=True)
class PairedInterpolationEvidence:
    """Exact paired disagreements against the trained endpoint."""

    seed: int
    candidate_only: int
    endpoint_only: int
    mcnemar_p_value: float


@dataclass(frozen=True)
class InterpolationDecision:
    """Recomputed multi-seed funding decision."""

    terminal_class: str
    selected_alpha: float | None
    aggregate_delta_ppm: int
    mean_margin_delta: float
    paired_evidence: tuple[PairedInterpolationEvidence, ...]


def _canonical_bytes(value: dict[str, object]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n"
    ).encode()


def _validate_state(state: object, *, role: str) -> OrderedDict[str, torch.Tensor]:
    if type(state) is not OrderedDict or not state:
        raise ValueError(f"{role} state must be a nonempty concrete OrderedDict")
    typed = state
    for name, tensor in typed.items():
        if type(name) is not str or not name or not isinstance(tensor, torch.Tensor):
            raise ValueError(f"{role} state schema differs")
        if tensor.layout != torch.strided:
            raise ValueError(f"{role} state tensors must use strided layout")
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"{role} state tensors must be finite")
    if "projection.weight" not in typed or "proxies" not in typed:
        raise ValueError(f"{role} state lacks the registered retrieval head")
    if not any(name.startswith("tower.") for name in typed):
        raise ValueError(f"{role} state lacks the registered tower")
    return typed


def model_state_sha256(state: object) -> str:
    """Hash tensor names, metadata, and bytes using the control-run authority."""

    typed = _validate_state(state, role="model")
    digest = hashlib.sha256()
    for name, tensor in sorted(typed.items()):
        metadata = _canonical_bytes(
            {"dtype": str(tensor.dtype), "name": name, "shape": list(tensor.shape)}
        )
        digest.update(len(metadata).to_bytes(8, "little"))
        digest.update(metadata)
        raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(len(raw).to_bytes(8, "little"))
        digest.update(raw)
    return digest.hexdigest()


def interpolate_inference_state(
    initial_state: object,
    trained_state: object,
    *,
    alpha: float,
) -> FoldedInferenceState:
    """Interpolate only the tower and carry the trained projection/proxies."""

    if type(alpha) is not float or alpha not in INTERPOLATION_ALPHAS:
        raise ValueError("alpha differs from the registered interpolation grid")
    initial = _validate_state(initial_state, role="initial")
    trained = _validate_state(trained_state, role="trained")
    if set(initial) != set(trained):
        raise ValueError("endpoint state names differ")

    folded: OrderedDict[str, torch.Tensor] = OrderedDict()
    tower_squared_displacement = 0.0
    for name, left in initial.items():
        right = trained[name]
        if left.shape != right.shape or left.dtype != right.dtype:
            raise ValueError("endpoint tensor metadata differs")
        if left.is_floating_point() != right.is_floating_point():
            raise ValueError("endpoint tensor kinds differ")
        if not left.is_floating_point() and not torch.equal(left, right):
            raise ValueError("non-floating endpoint tensors differ")

        if name.startswith("tower.") and left.is_floating_point():
            if alpha == 0.0:
                value = left.detach().cpu().clone()
            elif alpha == 1.0:
                value = right.detach().cpu().clone()
            else:
                value = torch.lerp(left.detach().cpu().float(), right.detach().cpu().float(), alpha)
                value = value.to(left.dtype)
            delta = value.double() - left.detach().cpu().double()
            tower_squared_displacement += float(torch.sum(delta * delta))
        else:
            value = right.detach().cpu().clone()
        folded[name] = value

    if not math.isfinite(tower_squared_displacement):
        raise ValueError("tower displacement must be finite")
    return FoldedInferenceState(
        alpha=alpha,
        state=folded,
        sha256=model_state_sha256(folded),
        tower_squared_displacement=tower_squared_displacement,
    )


def _paired_evidence(
    candidate: AlphaEvaluation, endpoint: AlphaEvaluation
) -> PairedInterpolationEvidence:
    candidate_only = sum(
        left and not right
        for left, right in zip(candidate.correctness, endpoint.correctness, strict=True)
    )
    endpoint_only = sum(
        right and not left
        for left, right in zip(candidate.correctness, endpoint.correctness, strict=True)
    )
    disagreements = candidate_only + endpoint_only
    if disagreements == 0:
        p_value = 1.0
    else:
        tail = sum(
            math.comb(disagreements, value)
            for value in range(min(candidate_only, endpoint_only) + 1)
        ) / (2**disagreements)
        p_value = min(1.0, 2.0 * tail)
    return PairedInterpolationEvidence(
        seed=candidate.seed,
        candidate_only=candidate_only,
        endpoint_only=endpoint_only,
        mcnemar_p_value=p_value,
    )


def classify_interpolation_curves(curves: object) -> InterpolationDecision:
    """Select one common alpha and classify provisional or final evidence."""

    if type(curves) is not tuple or tuple(curve.seed for curve in curves) not in (
        (17, 29),
        (17, 29, 43),
    ):
        raise ValueError("interpolation curves must use the registered seed order")
    if any(type(curve) is not SeedInterpolationCurve for curve in curves):
        raise ValueError("interpolation curve type differs")

    endpoints = tuple(curve.rows[-1] for curve in curves)
    endpoint_hits = sum(row.correct for row in endpoints)
    total_queries = sum(row.queries for row in endpoints)
    endpoint_ppm = endpoint_hits * 1_000_000 // total_queries
    endpoint_margin = sum(row.mean_margin for row in endpoints) / len(endpoints)

    observed: list[tuple[int, float, float, tuple[AlphaEvaluation, ...]]] = []
    passing: list[tuple[int, float, float, tuple[AlphaEvaluation, ...]]] = []
    for index, alpha in enumerate(INTERPOLATION_ALPHAS[1:-1], start=1):
        rows = tuple(curve.rows[index] for curve in curves)
        candidate_ppm = sum(row.correct for row in rows) * 1_000_000 // total_queries
        delta_ppm = candidate_ppm - endpoint_ppm
        mean_margin = sum(row.mean_margin for row in rows) / len(rows)
        item = (delta_ppm, mean_margin, alpha, rows)
        observed.append(item)
        if (
            delta_ppm >= 3_000
            and all(
                row.correct >= endpoint.correct - 1
                for row, endpoint in zip(rows, endpoints, strict=True)
            )
            and mean_margin > endpoint_margin
        ):
            passing.append(item)

    best_observed = max(observed, key=lambda item: (item[0], item[1], item[2]))
    if passing:
        selected = max(passing, key=lambda item: (item[0], item[1], item[2]))
        delta_ppm, mean_margin, alpha, rows = selected
        terminal = (
            "interior-benefit" if len(curves) == 3 else "provisional-interior-benefit"
        )
        pairs = tuple(
            _paired_evidence(row, endpoint) for row, endpoint in zip(rows, endpoints, strict=True)
        )
        return InterpolationDecision(
            terminal_class=terminal,
            selected_alpha=alpha,
            aggregate_delta_ppm=delta_ppm,
            mean_margin_delta=mean_margin - endpoint_margin,
            paired_evidence=pairs,
        )

    delta_ppm, mean_margin, _alpha, _rows = best_observed
    return InterpolationDecision(
        terminal_class=(
            "no-interior-benefit"
            if len(curves) == 3
            else "provisional-no-interior-benefit"
        ),
        selected_alpha=None,
        aggregate_delta_ppm=delta_ppm,
        mean_margin_delta=mean_margin - endpoint_margin,
        paired_evidence=(),
    )


def canonical_interpolation_result_bytes(
    curves: object,
    decision: object,
) -> bytes:
    """Recompute and serialize one claim-ineligible interpolation result."""

    recomputed = classify_interpolation_curves(curves)
    if type(decision) is not InterpolationDecision or decision != recomputed:
        raise ValueError("stored interpolation decision differs from recomputation")
    typed_curves = curves

    curve_payloads: list[dict[str, object]] = []
    for curve in typed_curves:
        rows: list[dict[str, object]] = []
        for row in curve.rows:
            correctness_sha256 = hashlib.sha256(bytes(row.correctness)).hexdigest()
            rows.append(
                {
                    "alpha": row.alpha,
                    "correct": row.correct,
                    "correctness_sha256": correctness_sha256,
                    "folded_state_sha256": row.folded_state_sha256,
                    "mean_margin": row.mean_margin,
                    "queries": row.queries,
                    "recall_ppm": row.recall_ppm,
                    "tower_squared_displacement": row.tower_squared_displacement,
                }
            )
        curve_payloads.append({"rows": rows, "seed": curve.seed})

    payload: dict[str, object] = {
        "claim_eligible": False,
        "curves": curve_payloads,
        "decision": {
            "aggregate_delta_ppm": recomputed.aggregate_delta_ppm,
            "mean_margin_delta": recomputed.mean_margin_delta,
            "paired_evidence": [
                {
                    "candidate_only": item.candidate_only,
                    "endpoint_only": item.endpoint_only,
                    "mcnemar_p_value": item.mcnemar_p_value,
                    "seed": item.seed,
                }
                for item in recomputed.paired_evidence
            ],
            "selected_alpha": recomputed.selected_alpha,
            "terminal_class": recomputed.terminal_class,
        },
        "schema": "sfora-weight-space-transfer-result-v1",
    }
    return _canonical_bytes(payload)
