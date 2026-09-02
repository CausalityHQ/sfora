"""Pure authority and canonical evidence for FVCG-Direct Phase A."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, fields, replace
from typing import Any, cast

from sfora.asgcv_verdict_marginal import (
    collapsed_verdict_coefficient,
    collapsed_verdict_probability,
)

FVCG_PHASE_A_SCHEMA = "sfora-fvcg-direct-phase-a-v1"
FVCG_STEP_COUNT = 3
FVCG_CUDA_CAP_BYTES = 96 * 1024**3
FVCG_RSS_CAP_BYTES = 96 * 1024**3
FVCG_COMBINED_P90_CAP_NS = 15_000_000_000
FVCG_SEMANTIC_P90_CAP_NS = 2_000_000_000
FVCG_PSI_FULL_AVG10_CAP_PPM = 500_000


def _canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _hex(value: object, length: int, *, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != length
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"FVCG {name} differs")
    return value


def _finite_float(value: object, *, name: str, positive: bool = False) -> float:
    if type(value) is not float or not math.isfinite(value) or (positive and value <= 0.0):
        raise ValueError(f"FVCG {name} differs")
    return value


def _exact_keys(value: object, expected: set[str], *, name: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != expected:
        raise ValueError(f"FVCG {name} schema differs")
    return cast(dict[str, Any], value)


def select_stratum_pair(
    stratum: tuple[int, ...], *, seed_sha256: str, step: int
) -> int:
    """Select one of exactly eight registered pairs from framed seed/step bytes."""

    _hex(seed_sha256, 64, name="selection seed")
    if (
        type(stratum) is not tuple
        or len(stratum) != 8
        or len(set(stratum)) != 8
        or any(type(value) is not int or value < 0 for value in stratum)
        or type(step) is not int
        or step < 0
    ):
        raise ValueError("FVCG stratum authority differs")
    digest = hashlib.sha256(
        b"fvcg-stratum-v1\0" + bytes.fromhex(seed_sha256) + step.to_bytes(8, "little")
    ).digest()
    return stratum[int.from_bytes(digest[:8], "little") % len(stratum)]


@dataclass(frozen=True)
class FvcgStepAuthority:
    source_commit: str
    launch_authority_sha256: str
    model_revision: str
    fixture_sha256: str
    selection_seed_sha256: str
    semantic_weight: float
    gradient_clip_norm: float
    direct_vjp_atol: float
    direct_vjp_rtol: float

    def validated(self) -> FvcgStepAuthority:
        _hex(self.source_commit, 40, name="source commit")
        _hex(self.model_revision, 40, name="model revision")
        for name in ("launch_authority_sha256", "fixture_sha256", "selection_seed_sha256"):
            _hex(getattr(self, name), 64, name=name.replace("_", " "))
        for name in (
            "semantic_weight",
            "gradient_clip_norm",
            "direct_vjp_atol",
            "direct_vjp_rtol",
        ):
            _finite_float(getattr(self, name), name=name.replace("_", " "), positive=True)
        return self

    @classmethod
    def from_mapping(cls, value: object) -> FvcgStepAuthority:
        raw = _exact_keys(value, {field.name for field in fields(cls)}, name="authority")
        try:
            return cls(**raw).validated()
        except TypeError as error:
            raise ValueError("FVCG authority differs") from error


@dataclass(frozen=True)
class FvcgStepEvidence:
    ordinal: int
    selected_pair: int
    correct_score: float
    incorrect_score: float
    correct_probability_ppm: int
    coefficient_ppm: int
    loss: float
    generated_tokens: int
    vision_nonzero_gradient_parameters: int
    pooler_nonzero_gradient_parameters: int
    proxy_nonzero_gradient_parameters: int
    language_gradient_parameters: int
    gradients_finite: bool
    dml_gradient_norm: float
    semantic_gradient_norm: float
    combined_gradient_cosine_distance_ppm: int
    clip_activated: bool
    vision_state_changed: bool
    pooler_state_changed: bool
    proxy_state_changed: bool
    combined_elapsed_ns: int
    semantic_elapsed_ns: int
    peak_cuda_reserved_bytes: int
    peak_rss_bytes: int
    memory_psi_full_avg10_ppm: int
    direct_vjp_max_abs_error: float
    direct_vjp_max_rel_error: float
    gradient_sha256: str
    updated_state_sha256: str
    optimizer_state_sha256: str
    language_state_sha256: str

    def validated(self, authority: FvcgStepAuthority) -> FvcgStepEvidence:
        authority.validated()
        if type(self.ordinal) is not int or self.ordinal < 0:
            raise ValueError("FVCG step ordinal differs")
        expected_pair = select_stratum_pair(
            tuple(range(8)), seed_sha256=authority.selection_seed_sha256, step=self.ordinal
        )
        if type(self.selected_pair) is not int or self.selected_pair != expected_pair:
            raise ValueError("FVCG selected pair differs")
        correct = _finite_float(self.correct_score, name="correct score")
        incorrect = _finite_float(self.incorrect_score, name="incorrect score")
        probability = collapsed_verdict_probability(correct, incorrect)
        coefficient = collapsed_verdict_coefficient(probability)
        if (
            type(self.correct_probability_ppm) is not int
            or self.correct_probability_ppm != round(probability * 1_000_000)
            or type(self.coefficient_ppm) is not int
            or self.coefficient_ppm != round(coefficient * 1_000_000)
        ):
            raise ValueError("FVCG collapsed scalar differs")
        loss = _finite_float(self.loss, name="loss")
        if not math.isclose(loss, -coefficient * (correct - incorrect), rel_tol=2e-6, abs_tol=2e-7):
            raise ValueError("FVCG loss differs")
        integer_fields = (
            "generated_tokens",
            "vision_nonzero_gradient_parameters",
            "pooler_nonzero_gradient_parameters",
            "proxy_nonzero_gradient_parameters",
            "language_gradient_parameters",
            "combined_gradient_cosine_distance_ppm",
            "combined_elapsed_ns",
            "semantic_elapsed_ns",
            "peak_cuda_reserved_bytes",
            "peak_rss_bytes",
            "memory_psi_full_avg10_ppm",
        )
        if any(
            type(getattr(self, name)) is not int or getattr(self, name) < 0
            for name in integer_fields
        ):
            raise ValueError("FVCG integer evidence differs")
        if type(self.gradients_finite) is not bool:
            raise ValueError("FVCG gradient finiteness differs")
        for name in ("dml_gradient_norm", "semantic_gradient_norm"):
            _finite_float(getattr(self, name), name=name.replace("_", " "), positive=True)
        for name in ("direct_vjp_max_abs_error", "direct_vjp_max_rel_error"):
            value = _finite_float(getattr(self, name), name=name.replace("_", " "))
            if value < 0.0:
                raise ValueError("FVCG direct VJP error differs")
        for name in (
            "clip_activated",
            "vision_state_changed",
            "pooler_state_changed",
            "proxy_state_changed",
        ):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"FVCG {name.replace('_', ' ')} differs")
        for name in (
            "gradient_sha256",
            "updated_state_sha256",
            "optimizer_state_sha256",
            "language_state_sha256",
        ):
            _hex(getattr(self, name), 64, name=name.replace("_", " "))
        return self

    @classmethod
    def from_mapping(cls, value: object, authority: FvcgStepAuthority) -> FvcgStepEvidence:
        raw = _exact_keys(value, {field.name for field in fields(cls)}, name="step")
        try:
            return cls(**raw).validated(authority)
        except TypeError as error:
            raise ValueError("FVCG step differs") from error


def _nearest_rank_p90(values: tuple[int, ...]) -> int:
    if not values:
        raise ValueError("FVCG timing evidence differs")
    return sorted(values)[math.ceil(0.9 * len(values)) - 1]


@dataclass(frozen=True)
class FvcgPhaseAResult:
    authority: FvcgStepAuthority
    steps: tuple[FvcgStepEvidence, ...]
    repeated_step_zero: FvcgStepEvidence
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
        authority: FvcgStepAuthority,
        steps: tuple[FvcgStepEvidence, ...],
        repeated_step_zero: FvcgStepEvidence,
        initial_language_state_sha256: str,
    ) -> FvcgPhaseAResult:
        authority.validated()
        _hex(initial_language_state_sha256, 64, name="initial language state")
        if type(steps) is not tuple or len(steps) != FVCG_STEP_COUNT:
            raise ValueError("FVCG measured steps differ")
        if tuple(step.ordinal for step in steps) != tuple(range(FVCG_STEP_COUNT)):
            raise ValueError("FVCG step order differs")
        for step in steps:
            step.validated(authority)
        repeated_step_zero.validated(authority)
        deterministic = all(
            getattr(steps[0], name) == getattr(repeated_step_zero, name)
            for name in (
                "selected_pair",
                "correct_score",
                "incorrect_score",
                "correct_probability_ppm",
                "coefficient_ppm",
                "loss",
                "gradient_sha256",
                "updated_state_sha256",
                "optimizer_state_sha256",
            )
        )
        combined_p90 = _nearest_rank_p90(tuple(step.combined_elapsed_ns for step in steps))
        semantic_p90 = _nearest_rank_p90(tuple(step.semantic_elapsed_ns for step in steps))
        peak_cuda = max(step.peak_cuda_reserved_bytes for step in steps)
        peak_rss = max(step.peak_rss_bytes for step in steps)
        peak_psi = max(step.memory_psi_full_avg10_ppm for step in steps)
        passed = (
            deterministic
            and all(step.language_state_sha256 == initial_language_state_sha256 for step in steps)
            and repeated_step_zero.language_state_sha256 == initial_language_state_sha256
            and all(step.generated_tokens == 0 for step in steps)
            and repeated_step_zero.generated_tokens == 0
            and all(step.language_gradient_parameters == 0 for step in steps)
            and repeated_step_zero.language_gradient_parameters == 0
            and all(step.gradients_finite for step in steps)
            and repeated_step_zero.gradients_finite
            and all(
                step.vision_state_changed
                and step.pooler_state_changed
                and step.proxy_state_changed
                for step in (*steps, repeated_step_zero)
            )
            and all(
                step.vision_nonzero_gradient_parameters > 0
                and step.pooler_nonzero_gradient_parameters > 0
                and step.proxy_nonzero_gradient_parameters > 0
                for step in (*steps, repeated_step_zero)
            )
            and all(
                step.direct_vjp_max_abs_error <= authority.direct_vjp_atol
                and step.direct_vjp_max_rel_error <= authority.direct_vjp_rtol
                for step in (*steps, repeated_step_zero)
            )
            and combined_p90 <= FVCG_COMBINED_P90_CAP_NS
            and semantic_p90 <= FVCG_SEMANTIC_P90_CAP_NS
            and peak_cuda <= FVCG_CUDA_CAP_BYTES
            and peak_rss <= FVCG_RSS_CAP_BYTES
            and peak_psi < FVCG_PSI_FULL_AVG10_CAP_PPM
        )
        provisional = cls(
            authority=authority,
            steps=steps,
            repeated_step_zero=repeated_step_zero,
            initial_language_state_sha256=initial_language_state_sha256,
            combined_p90_ns=combined_p90,
            semantic_p90_ns=semantic_p90,
            peak_cuda_reserved_bytes=peak_cuda,
            peak_rss_bytes=peak_rss,
            peak_memory_psi_full_avg10_ppm=peak_psi,
            deterministic_step_zero=deterministic,
            passed=passed,
            result_sha256="",
        )
        digest = hashlib.sha256(_canonical(provisional._unsigned_mapping())).hexdigest()
        return replace(provisional, result_sha256=digest)

    def _unsigned_mapping(self) -> dict[str, object]:
        return {
            "schema": FVCG_PHASE_A_SCHEMA,
            "claim_eligible": False,
            "official_test_access": False,
            "gates": {
                "combined_p90_ns": FVCG_COMBINED_P90_CAP_NS,
                "semantic_p90_ns": FVCG_SEMANTIC_P90_CAP_NS,
                "peak_cuda_reserved_bytes": FVCG_CUDA_CAP_BYTES,
                "peak_rss_bytes": FVCG_RSS_CAP_BYTES,
                "memory_psi_full_avg10_ppm_exclusive": FVCG_PSI_FULL_AVG10_CAP_PPM,
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

    @classmethod
    def from_mapping(cls, value: object) -> FvcgPhaseAResult:
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
        raw = _exact_keys(value, expected, name="result")
        expected_gates = {
            "combined_p90_ns": FVCG_COMBINED_P90_CAP_NS,
            "semantic_p90_ns": FVCG_SEMANTIC_P90_CAP_NS,
            "peak_cuda_reserved_bytes": FVCG_CUDA_CAP_BYTES,
            "peak_rss_bytes": FVCG_RSS_CAP_BYTES,
            "memory_psi_full_avg10_ppm_exclusive": FVCG_PSI_FULL_AVG10_CAP_PPM,
        }
        if (
            raw["schema"] != FVCG_PHASE_A_SCHEMA
            or raw["claim_eligible"] is not False
            or raw["official_test_access"] is not False
            or raw["gates"] != expected_gates
            or type(raw["steps"]) is not list
        ):
            raise ValueError("FVCG result schema differs")
        authority = FvcgStepAuthority.from_mapping(raw["authority"])
        steps = tuple(FvcgStepEvidence.from_mapping(item, authority) for item in raw["steps"])
        repeated = FvcgStepEvidence.from_mapping(raw["repeated_step_zero"], authority)
        recomputed = cls.from_steps(
            authority=authority,
            steps=steps,
            repeated_step_zero=repeated,
            initial_language_state_sha256=raw["initial_language_state_sha256"],
        )
        if raw != recomputed.to_mapping():
            raise ValueError("FVCG result arithmetic differs")
        return recomputed


def canonical_fvcg_phase_a_result_bytes(value: FvcgPhaseAResult) -> bytes:
    """Serialize one fully recomputed Phase-A result."""

    if type(value) is not FvcgPhaseAResult:
        raise ValueError("FVCG result differs")
    reopened = FvcgPhaseAResult.from_mapping(value.to_mapping())
    return _canonical(reopened.to_mapping())


def validate_fvcg_phase_a_result_bytes(raw: object) -> FvcgPhaseAResult:
    """Authenticate canonical bytes and independently recompute every gate."""

    if type(raw) is not bytes:
        raise ValueError("FVCG result bytes differ")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("FVCG result JSON differs") from error
    result = FvcgPhaseAResult.from_mapping(value)
    if _canonical(result.to_mapping()) != raw:
        raise ValueError("FVCG result canonical bytes differ")
    return result
