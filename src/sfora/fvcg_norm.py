"""Norm-stabilized FVCG gradient arithmetic and evidence."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, fields, replace
from typing import Any, cast

import torch

from sfora.fvcg_direct import FvcgPhaseAResult, FvcgStepAuthority, FvcgStepEvidence

FVCG_NORM_PHASE_A_SCHEMA = "sfora-fvcg-norm-phase-a-v1"
FVCG_NORM_RHO = 0.25
FVCG_NORM_RATIO_MIN_PPM = 249_000
FVCG_NORM_RATIO_MAX_PPM = 251_000
FVCG_NORM_DIRECTION_MIN_PPM = 5_000
FVCG_NORM_DIRECTION_MAX_PPM = 30_000
FVCG_NORM_FP32_PROJECTION_ERROR_FACTOR = 8.0 * 2.0**-23


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
        float(torch.sum(a.double() * b.double())) for a, b in zip(left, right, strict=True)
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
            and self.__dict__ | {"gradients": ()} == other.__dict__ | {"gradients": ()}
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
    projection_tolerance = (
        FVCG_NORM_FP32_PROJECTION_ERROR_FACTOR * dml_norm * semantic_norm
    )
    if projected_dot < -projection_tolerance:
        raise ValueError("FVCG-Norm conflict projection differs")

    target_applied_norm = rho * dml_norm
    applied = tuple(value * (target_applied_norm / safe_norm) for value in safe)
    combined = tuple(
        dml_value + semantic_value for dml_value, semantic_value in zip(dml, applied, strict=True)
    )
    applied_norm = _norm(
        tuple(
            combined_value - dml_value
            for combined_value, dml_value in zip(combined, dml, strict=True)
        )
    )
    combined_norm = _norm(combined)
    cosine = _dot(dml, combined) / (dml_norm * combined_norm)
    cosine_distance_ppm = max(0, round((1.0 - max(-1.0, min(1.0, cosine))) * 1_000_000))
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


def remeasure_stored_combined_gradients(
    field: NormCombinedField,
    dml_gradients: tuple[torch.Tensor, ...],
    stored_gradients: tuple[torch.Tensor, ...],
) -> NormCombinedField:
    """Measure the increment and direction after the optimizer-dtype store."""

    if (
        not isinstance(field, NormCombinedField)
        or type(dml_gradients) is not tuple
        or type(stored_gradients) is not tuple
        or len(dml_gradients) != len(field.gradients)
        or len(stored_gradients) != len(field.gradients)
        or any(
            not isinstance(dml, torch.Tensor)
            or not isinstance(stored, torch.Tensor)
            or dml.shape != stored.shape
            or not bool(torch.isfinite(dml).all())
            or not bool(torch.isfinite(stored).all())
            for dml, stored in zip(dml_gradients, stored_gradients, strict=True)
        )
    ):
        raise ValueError("FVCG-Norm stored gradient authority differs")
    dml = tuple(value.detach().float() for value in dml_gradients)
    stored = tuple(value.detach().float() for value in stored_gradients)
    increment = tuple(combined - source for combined, source in zip(stored, dml, strict=True))
    applied_norm = _norm(increment)
    dml_norm = _norm(dml)
    combined_norm = _norm(stored)
    if applied_norm <= 0.0 or dml_norm <= 0.0 or combined_norm <= 0.0:
        raise ValueError("FVCG-Norm stored gradient norm differs")
    cosine = _dot(dml, stored) / (dml_norm * combined_norm)
    return replace(
        field,
        gradients=stored,
        dml_norm=dml_norm,
        applied_semantic_norm=applied_norm,
        applied_to_dml_ratio_ppm=round(applied_norm / dml_norm * 1_000_000),
        combined_cosine_distance_ppm=max(0, round((1.0 - max(-1.0, min(1.0, cosine))) * 1_000_000)),
    )


@dataclass(frozen=True)
class FvcgNormAuthority:
    """Frozen authority for one FVCG-Norm Phase-A campaign."""

    base: FvcgStepAuthority
    rho: float
    fixture_source_commit: str

    def validated(self) -> FvcgNormAuthority:
        self.base.validated()
        _finite_positive(self.rho, name="rho")
        if (
            type(self.fixture_source_commit) is not str
            or len(self.fixture_source_commit) != 40
            or any(character not in "0123456789abcdef" for character in self.fixture_source_commit)
        ):
            raise ValueError("FVCG-Norm fixture source commit differs")
        if self.rho != FVCG_NORM_RHO or self.base.semantic_weight != self.rho:
            raise ValueError("FVCG-Norm authority differs")
        return self

    @classmethod
    def from_mapping(cls, value: object) -> FvcgNormAuthority:
        raw = _exact_keys(value, {"base", "rho", "fixture_source_commit"}, name="authority")
        if type(raw["rho"]) is not float:
            raise ValueError("FVCG-Norm authority differs")
        return cls(
            base=FvcgStepAuthority.from_mapping(raw["base"]),
            rho=raw["rho"],
            fixture_source_commit=raw["fixture_source_commit"],
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
        applied = _finite_positive(self.applied_semantic_norm, name="applied semantic norm")
        dml_norm = self.base.dml_gradient_norm
        semantic_norm = self.base.semantic_gradient_norm
        dot_bound = dml_norm * semantic_norm
        scalar_tolerance = 1.0e-6 * max(1.0, dot_bound)
        expected_projected = max(0.0, self.raw_dot)
        conflict = min(0.0, self.raw_dot)
        expected_safe_squared = semantic_norm**2 - conflict**2 / dml_norm**2
        if (
            type(self.raw_dot) is not float
            or not math.isfinite(self.raw_dot)
            or type(self.projected_dot) is not float
            or not math.isfinite(self.projected_dot)
            or type(self.applied_to_dml_ratio_ppm) is not int
            or self.applied_to_dml_ratio_ppm
            != round(applied / self.base.dml_gradient_norm * 1_000_000)
            or abs(self.raw_dot) > dot_bound + scalar_tolerance
            or not math.isclose(
                self.projected_dot,
                expected_projected,
                rel_tol=1.0e-6,
                abs_tol=scalar_tolerance,
            )
            or expected_safe_squared <= 0.0
            or not math.isclose(
                safe_norm**2,
                expected_safe_squared,
                rel_tol=1.0e-5,
                abs_tol=1.0e-8,
            )
            or self.projected_dot
            < -FVCG_NORM_FP32_PROJECTION_ERROR_FACTOR * dml_norm * semantic_norm
        ):
            raise ValueError("FVCG-Norm step arithmetic differs")
        return self

    @classmethod
    def from_mapping(cls, value: object, authority: FvcgNormAuthority) -> FvcgNormStepEvidence:
        raw = dict(_exact_keys(value, {field.name for field in fields(cls)}, name="step"))
        try:
            return cls(
                base=FvcgStepEvidence.from_mapping(raw.pop("base"), authority.base),
                **raw,
            ).validated(authority)
        except TypeError as error:
            raise ValueError("FVCG-Norm step differs") from error


def _norm_step_passes(step: FvcgNormStepEvidence) -> bool:
    return (
        FVCG_NORM_RATIO_MIN_PPM <= step.applied_to_dml_ratio_ppm <= FVCG_NORM_RATIO_MAX_PPM
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
    combined_p90_ns: int
    semantic_p90_ns: int
    peak_cuda_reserved_bytes: int
    peak_rss_bytes: int
    peak_memory_psi_full_avg10_ppm: int
    deterministic_step_zero: bool
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
        ) and all(
            getattr(steps[0].base, name) == getattr(repeated_step_zero.base, name)
            for name in (
                "dml_gradient_norm",
                "semantic_gradient_norm",
                "combined_gradient_cosine_distance_ppm",
            )
        )
        unsigned = cls(
            authority=authority,
            steps=steps,
            repeated_step_zero=repeated_step_zero,
            initial_language_state_sha256=initial_language_state_sha256,
            combined_p90_ns=base_result.combined_p90_ns,
            semantic_p90_ns=base_result.semantic_p90_ns,
            peak_cuda_reserved_bytes=base_result.peak_cuda_reserved_bytes,
            peak_rss_bytes=base_result.peak_rss_bytes,
            peak_memory_psi_full_avg10_ppm=base_result.peak_memory_psi_full_avg10_ppm,
            deterministic_step_zero=base_result.deterministic_step_zero and norm_deterministic,
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
                "combined_p90_ns": 15_000_000_000,
                "semantic_p90_ns": 2_000_000_000,
                "peak_cuda_reserved_bytes": 96 * 1024**3,
                "peak_rss_bytes": 96 * 1024**3,
                "memory_psi_full_avg10_ppm_exclusive": 500_000,
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
            "combined_p90_ns": self.combined_p90_ns,
            "semantic_p90_ns": self.semantic_p90_ns,
            "peak_cuda_reserved_bytes": self.peak_cuda_reserved_bytes,
            "peak_rss_bytes": self.peak_rss_bytes,
            "peak_memory_psi_full_avg10_ppm": self.peak_memory_psi_full_avg10_ppm,
            "deterministic_step_zero": self.deterministic_step_zero,
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
        "combined_p90_ns",
        "semantic_p90_ns",
        "peak_cuda_reserved_bytes",
        "peak_rss_bytes",
        "peak_memory_psi_full_avg10_ppm",
        "deterministic_step_zero",
        "passed",
        "result_sha256",
    }
    parsed = _exact_keys(value, expected, name="result")
    expected_gates = {
        "rho": FVCG_NORM_RHO,
        "combined_p90_ns": 15_000_000_000,
        "semantic_p90_ns": 2_000_000_000,
        "peak_cuda_reserved_bytes": 96 * 1024**3,
        "peak_rss_bytes": 96 * 1024**3,
        "memory_psi_full_avg10_ppm_exclusive": 500_000,
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
        or type(parsed["deterministic_step_zero"]) is not bool
        or type(parsed["result_sha256"]) is not str
    ):
        raise ValueError("FVCG-Norm result schema differs")
    authority = FvcgNormAuthority.from_mapping(parsed["authority"])
    steps = tuple(FvcgNormStepEvidence.from_mapping(step, authority) for step in parsed["steps"])
    repeated = FvcgNormStepEvidence.from_mapping(parsed["repeated_step_zero"], authority)
    rebuilt = FvcgNormPhaseAResult.from_steps(
        authority=authority,
        steps=steps,
        repeated_step_zero=repeated,
        initial_language_state_sha256=parsed["initial_language_state_sha256"],
    )
    if rebuilt.to_mapping() != parsed or _canonical(parsed) != raw:
        raise ValueError("FVCG-Norm result differs")
    return rebuilt
