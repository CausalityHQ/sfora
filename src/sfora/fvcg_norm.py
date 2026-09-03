"""Norm-stabilized FVCG gradient arithmetic and evidence."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


def _finite_positive(value: object, *, name: str) -> float:
    if type(value) is not float or not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"FVCG-Norm {name} differs")
    return value


def _dot(left: tuple[torch.Tensor, ...], right: tuple[torch.Tensor, ...]) -> float:
    return math.fsum(
        float(torch.sum(a.double() * b.double()))
        for a, b in zip(left, right, strict=True)
    )


def _norm(values: tuple[torch.Tensor, ...]) -> float:
    return math.sqrt(max(0.0, _dot(values, values)))


@dataclass(frozen=True, eq=False)
class NormCombinedField:
    """FP32 combined gradients plus independently recomputable scalar evidence."""

    gradients: tuple[torch.Tensor, ...]
    dml_norm: float
    semantic_norm: float
    safe_semantic_norm: float
    raw_dot: float
    projected_dot: float
    applied_semantic_norm: float
    applied_to_dml_ratio_ppm: int
    combined_cosine_distance_ppm: int

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, NormCombinedField):
            return NotImplemented
        return (
            len(self.gradients) == len(other.gradients)
            and all(
                torch.equal(left, right)
                for left, right in zip(self.gradients, other.gradients, strict=True)
            )
            and self.__dict__ | {"gradients": ()}
            == other.__dict__ | {"gradients": ()}
        )


def combine_norm_stabilized_gradients(
    dml_gradients: tuple[torch.Tensor, ...],
    semantic_gradients: tuple[torch.Tensor, ...],
    *,
    rho: float,
) -> NormCombinedField:
    """Remove semantic conflict and inject a fixed-ratio normalized FP32 field."""

    _finite_positive(rho, name="rho")
    if (
        type(dml_gradients) is not tuple
        or type(semantic_gradients) is not tuple
        or not dml_gradients
        or len(dml_gradients) != len(semantic_gradients)
        or any(
            not isinstance(left, torch.Tensor)
            or not isinstance(right, torch.Tensor)
            or left.shape != right.shape
            or not bool(torch.isfinite(left).all())
            or not bool(torch.isfinite(right).all())
            for left, right in zip(dml_gradients, semantic_gradients, strict=True)
        )
    ):
        raise ValueError("FVCG-Norm gradient authority differs")

    dml = tuple(value.detach().float() for value in dml_gradients)
    semantic = tuple(value.detach().float() for value in semantic_gradients)
    dml_squared = _dot(dml, dml)
    dml_norm = math.sqrt(max(0.0, dml_squared))
    semantic_norm = _norm(semantic)
    if dml_norm <= 0.0 or semantic_norm <= 0.0:
        raise ValueError("FVCG-Norm gradient norm differs")

    raw_dot = _dot(dml, semantic)
    conflict_scale = min(0.0, raw_dot) / dml_squared
    safe = tuple(
        semantic_value - conflict_scale * dml_value
        for dml_value, semantic_value in zip(dml, semantic, strict=True)
    )
    safe_norm = _norm(safe)
    if not math.isfinite(safe_norm) or safe_norm / dml_norm < 1.0e-12:
        raise ValueError("FVCG-Norm safe semantic field differs")
    projected_dot = _dot(dml, safe)
    projection_tolerance = 1.0e-10 * dml_norm * safe_norm
    if projected_dot < -projection_tolerance:
        raise ValueError("FVCG-Norm conflict projection differs")

    applied_norm = rho * dml_norm
    applied = tuple(value * (applied_norm / safe_norm) for value in safe)
    combined = tuple(
        dml_value + semantic_value
        for dml_value, semantic_value in zip(dml, applied, strict=True)
    )
    combined_norm = _norm(combined)
    cosine = _dot(dml, combined) / (dml_norm * combined_norm)
    cosine_distance_ppm = max(
        0, round((1.0 - max(-1.0, min(1.0, cosine))) * 1_000_000)
    )
    return NormCombinedField(
        gradients=combined,
        dml_norm=dml_norm,
        semantic_norm=semantic_norm,
        safe_semantic_norm=safe_norm,
        raw_dot=raw_dot,
        projected_dot=projected_dot,
        applied_semantic_norm=applied_norm,
        applied_to_dml_ratio_ppm=round(applied_norm / dml_norm * 1_000_000),
        combined_cosine_distance_ppm=cosine_distance_ppm,
    )
