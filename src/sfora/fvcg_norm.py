"""Norm-stabilized FVCG gradient arithmetic and evidence."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, fields
from typing import Any, cast

import torch

from sfora.fvcg_direct import FvcgPhaseAResult, FvcgStepAuthority, FvcgStepEvidence

FVCG_NORM_PHASE_A_SCHEMA = "sfora-fvcg-norm-phase-a-v1"
FVCG_NORM_RHO = 0.25
FVCG_NORM_RATIO_MIN_PPM = 249_000
FVCG_NORM_RATIO_MAX_PPM = 251_000
FVCG_NORM_DIRECTION_MIN_PPM = 5_000
FVCG_NORM_DIRECTION_MAX_PPM = 50_000


def _finite_positive(value: object, *, name: str) -> float:
    if type(value) is not float or not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"FVCG-Norm {name} differs")
    return value


def _exact_keys(value: object, expected: set[str], *, name: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != expected:
        raise ValueError(f"FVCG-Norm {name} schema differs")
    return cast(dict[str, Any], value)


def _canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


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


@dataclass(frozen=True)
class FvcgNormAuthority:
    """Frozen authority for one FVCG-Norm Phase-A campaign."""

    base: FvcgStepAuthority
    rho: float

    def validated(self) -> FvcgNormAuthority:
        self.base.validated()
        _finite_positive(self.rho, name="rho")
        if self.rho != FVCG_NORM_RHO or self.base.semantic_weight != self.rho:
            raise ValueError("FVCG-Norm authority differs")
        return self

    @classmethod
    def from_mapping(cls, value: object) -> FvcgNormAuthority:
        raw = _exact_keys(value, {"base", "rho"}, name="authority")
        if type(raw["rho"]) is not float:
            raise ValueError("FVCG-Norm authority differs")
        return cls(
            base=FvcgStepAuthority.from_mapping(raw["base"]), rho=raw["rho"]
        ).validated()


@dataclass(frozen=True)
class FvcgNormStepEvidence:
    """One norm-stabilized step and its recomputable field-scale evidence."""

    base: FvcgStepEvidence
    safe_semantic_norm: float
    raw_dot: float
    projected_dot: float
    applied_semantic_norm: float
    applied_to_dml_ratio_ppm: int

    def validated(self, authority: FvcgNormAuthority) -> FvcgNormStepEvidence:
        authority.validated()
        self.base.validated(authority.base)
        safe_norm = _finite_positive(self.safe_semantic_norm, name="safe semantic norm")
        applied = _finite_positive(
            self.applied_semantic_norm, name="applied semantic norm"
        )
        if (
            type(self.raw_dot) is not float
            or not math.isfinite(self.raw_dot)
            or type(self.projected_dot) is not float
            or not math.isfinite(self.projected_dot)
            or type(self.applied_to_dml_ratio_ppm) is not int
            or self.applied_to_dml_ratio_ppm
            != round(applied / self.base.dml_gradient_norm * 1_000_000)
            or not math.isclose(
                applied,
                authority.rho * self.base.dml_gradient_norm,
                rel_tol=1.0e-6,
                abs_tol=1.0e-8,
            )
            or self.projected_dot
            < -1.0e-10 * self.base.dml_gradient_norm * safe_norm
        ):
            raise ValueError("FVCG-Norm step arithmetic differs")
        return self

    @classmethod
    def from_mapping(
        cls, value: object, authority: FvcgNormAuthority
    ) -> FvcgNormStepEvidence:
        raw = dict(
            _exact_keys(value, {field.name for field in fields(cls)}, name="step")
        )
        try:
            return cls(
                base=FvcgStepEvidence.from_mapping(raw.pop("base"), authority.base),
                **raw,
            ).validated(authority)
        except TypeError as error:
            raise ValueError("FVCG-Norm step differs") from error


def _norm_step_passes(step: FvcgNormStepEvidence) -> bool:
    return (
        FVCG_NORM_RATIO_MIN_PPM
        <= step.applied_to_dml_ratio_ppm
        <= FVCG_NORM_RATIO_MAX_PPM
        and FVCG_NORM_DIRECTION_MIN_PPM
        <= step.base.combined_gradient_cosine_distance_ppm
        <= FVCG_NORM_DIRECTION_MAX_PPM
    )


@dataclass(frozen=True)
class FvcgNormPhaseAResult:
    """Canonical FVCG-Norm Phase-A result."""

    authority: FvcgNormAuthority
    steps: tuple[FvcgNormStepEvidence, ...]
    repeated_step_zero: FvcgNormStepEvidence
    initial_language_state_sha256: str
    passed: bool
    result_sha256: str

    @classmethod
    def from_steps(
        cls,
        *,
        authority: FvcgNormAuthority,
        steps: tuple[FvcgNormStepEvidence, ...],
        repeated_step_zero: FvcgNormStepEvidence,
        initial_language_state_sha256: str,
    ) -> FvcgNormPhaseAResult:
        authority.validated()
        if type(steps) is not tuple or len(steps) != 3:
            raise ValueError("FVCG-Norm measured steps differ")
        for step in (*steps, repeated_step_zero):
            step.validated(authority)
        base_result = FvcgPhaseAResult.from_steps(
            authority=authority.base,
            steps=tuple(step.base for step in steps),
            repeated_step_zero=repeated_step_zero.base,
            initial_language_state_sha256=initial_language_state_sha256,
        )
        norm_replay_fields = (
            "safe_semantic_norm",
            "raw_dot",
            "projected_dot",
            "applied_semantic_norm",
            "applied_to_dml_ratio_ppm",
        )
        norm_deterministic = all(
            getattr(steps[0], name) == getattr(repeated_step_zero, name)
            for name in norm_replay_fields
        )
        unsigned = cls(
            authority=authority,
            steps=steps,
            repeated_step_zero=repeated_step_zero,
            initial_language_state_sha256=initial_language_state_sha256,
            passed=base_result.passed
            and norm_deterministic
            and all(_norm_step_passes(step) for step in steps),
            result_sha256="",
        )
        digest = hashlib.sha256(_canonical(unsigned._unsigned_mapping())).hexdigest()
        return cls(**{**unsigned.__dict__, "result_sha256": digest})

    def _unsigned_mapping(self) -> dict[str, object]:
        return {
            "schema": FVCG_NORM_PHASE_A_SCHEMA,
            "claim_eligible": False,
            "official_test_access": False,
            "gates": {
                "rho": FVCG_NORM_RHO,
                "applied_to_dml_ratio_ppm": [
                    FVCG_NORM_RATIO_MIN_PPM,
                    FVCG_NORM_RATIO_MAX_PPM,
                ],
                "combined_cosine_distance_ppm": [
                    FVCG_NORM_DIRECTION_MIN_PPM,
                    FVCG_NORM_DIRECTION_MAX_PPM,
                ],
            },
            "authority": asdict(self.authority),
            "steps": [asdict(step) for step in self.steps],
            "repeated_step_zero": asdict(self.repeated_step_zero),
            "initial_language_state_sha256": self.initial_language_state_sha256,
            "passed": self.passed,
        }

    def to_mapping(self) -> dict[str, object]:
        value = self._unsigned_mapping()
        value["result_sha256"] = self.result_sha256
        return value


def canonical_fvcg_norm_phase_a_result_bytes(result: FvcgNormPhaseAResult) -> bytes:
    """Return strict canonical JSON plus one LF after recomputation."""

    rebuilt = FvcgNormPhaseAResult.from_steps(
        authority=result.authority,
        steps=result.steps,
        repeated_step_zero=result.repeated_step_zero,
        initial_language_state_sha256=result.initial_language_state_sha256,
    )
    if rebuilt != result:
        raise ValueError("FVCG-Norm result differs")
    return _canonical(result.to_mapping())


def validate_fvcg_norm_phase_a_result_bytes(raw: bytes) -> FvcgNormPhaseAResult:
    """Reopen strict canonical Phase-A bytes and recompute all derived evidence."""

    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError("FVCG-Norm result JSON differs") from error
    expected = {
        "schema",
        "claim_eligible",
        "official_test_access",
        "gates",
        "authority",
        "steps",
        "repeated_step_zero",
        "initial_language_state_sha256",
        "passed",
        "result_sha256",
    }
    parsed = _exact_keys(value, expected, name="result")
    expected_gates = {
        "rho": FVCG_NORM_RHO,
        "applied_to_dml_ratio_ppm": [
            FVCG_NORM_RATIO_MIN_PPM,
            FVCG_NORM_RATIO_MAX_PPM,
        ],
        "combined_cosine_distance_ppm": [
            FVCG_NORM_DIRECTION_MIN_PPM,
            FVCG_NORM_DIRECTION_MAX_PPM,
        ],
    }
    if (
        parsed["schema"] != FVCG_NORM_PHASE_A_SCHEMA
        or parsed["claim_eligible"] is not False
        or parsed["official_test_access"] is not False
        or parsed["gates"] != expected_gates
        or type(parsed["steps"]) is not list
        or type(parsed["passed"]) is not bool
        or type(parsed["result_sha256"]) is not str
    ):
        raise ValueError("FVCG-Norm result schema differs")
    authority = FvcgNormAuthority.from_mapping(parsed["authority"])
    steps = tuple(
        FvcgNormStepEvidence.from_mapping(step, authority) for step in parsed["steps"]
    )
    repeated = FvcgNormStepEvidence.from_mapping(
        parsed["repeated_step_zero"], authority
    )
    rebuilt = FvcgNormPhaseAResult.from_steps(
        authority=authority,
        steps=steps,
        repeated_step_zero=repeated,
        initial_language_state_sha256=parsed["initial_language_state_sha256"],
    )
    if rebuilt.to_mapping() != parsed or _canonical(parsed) != raw:
        raise ValueError("FVCG-Norm result differs")
    return rebuilt
