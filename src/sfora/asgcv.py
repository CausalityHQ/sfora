"""Scalar authority for Amortized Semantic Gradient Control Variates."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from sfora.asgcv_protocol import (
    AsgcvCompletionGroup,
    AsgcvCompletionProtocol,
    AsgcvEligibleSchedule,
    AsgcvPairSchedule,
    AsgcvPartitionAuthority,
    AsgcvRolloutAuthority,
    validate_asgcv_protocol_bundle,
)

Float64Array = NDArray[np.float64]
Float32Array = NDArray[np.float32]

ASGCV_SCHEMA = "sfora-asgcv-authority-v1"
ASGCV_STRATUM_SIZE = 8
ASGCV_PREDICTOR_RANK = 16
ASGCV_SELECTION_POLICY = "one-uniform-index-per-eight-pair-stratum-v1"
ASGCV_E0_SCHEMA = "sfora-asgcv-e0-metrics-v5"
ASGCV_E0_CAPACITY_FLOOR_SCHEMA = "sfora-asgcv-e0-capacity-floor-v1"
ASGCV_E0_RESULT_SCHEMA = "sfora-asgcv-e0-result-v5"
ASGCV_E0_ARRAY_DOMAIN = b"sfora-asgcv-e0-array-v1\0"
ASGCV_GRADIENT_SAMPLE_SCHEMA = "sfora-asgcv-gradient-sample-v4"
ASGCV_GRADIENT_SAMPLE_ARRAY_DOMAIN = b"sfora-asgcv-gradient-sample-array-v1\0"
ASGCV_DENSE_COSINE_GATE_PPM = 850_000
ASGCV_PROJECTED_COSINE_GATE_PPM = 900_000
ASGCV_PATCH_SPEARMAN_GATE_PPM = 800_000
ASGCV_RESIDUAL_ENERGY_GATE_PPM = 350_000
ASGCV_VARIANCE_RATIO_GATE_PPM = 600_000
ASGCV_PRECLIP_P99_RATIO_GATE_PPM = 2_000_000
ASGCV_CLIP_RATE_DELTA_GATE_PPM = 50_000
ASGCV_CLIP_REFERENCE_QUANTILE = 0.90
ASGCV_SEMANTIC_WALL_RATIO_GATE_PPM = 350_000
ASGCV_MEAN_AGREEMENT_GATE_PPM = 150_000
ASGCV_PEAK_CUDA_RESERVED_GATE_BYTES = 96 * 1024**3
ASGCV_E0_MINIMUM_PAIRS = 512
ASGCV_E0_CAPACITY_MINIMUM_PAIRS = 64
ASGCV_SELECTION_DOMAIN = b"sfora-asgcv-selection-v1\0"
ASGCV_SCHEDULE_DOMAIN = b"sfora-asgcv-schedule-v1\0"
ASGCV_MAX_SCHEDULE_SELECTIONS = 10_000_000
ASGCV_SRHT_SCHEMA = "sfora-asgcv-srht-authority-v1"
ASGCV_SRHT_SIGN_DOMAIN = b"sfora-asgcv-srht-sign-v1\0"
ASGCV_SRHT_ROW_DOMAIN = b"sfora-asgcv-srht-row-v1\0"
ASGCV_SRHT_NORMALIZATION = "orthonormal-hadamard-times-sqrt-padded-over-output-v1"
ASGCV_MEAN_BOOTSTRAP_DOMAIN = b"sfora-asgcv-e0-mean-bootstrap-v1\0"
ASGCV_MEAN_BOOTSTRAP_DRAWS = 10_000


def _require_exact_positive_integer(value: object, *, expected: int, name: str) -> int:
    if type(value) is not int or value != expected:
        raise ValueError(f"ASG-CV {name} differs")
    return value


def _sha256_bytes(value: object, *, name: str) -> bytes:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"ASG-CV {name} differs")
    return bytes.fromhex(value)


def _source_commit(value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("ASG-CV source commit differs")
    return value


def _canonical_json_bytes(value: dict[str, object]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")


def _selection_seed_bytes(value: object) -> bytes:
    return _sha256_bytes(value, name="selection seed")


def _u64(value: object, *, name: str) -> int:
    if type(value) is not int or not 0 <= value < 2**64:
        raise ValueError(f"ASG-CV {name} differs")
    return value


def select_stratum_index(
    selection_seed: object,
    *,
    optimizer_step: int,
    stratum_ordinal: int,
) -> int:
    """Select one of eight pairs from a source-bound independent stream."""

    seed = _selection_seed_bytes(selection_seed)
    step = _u64(optimizer_step, name="optimizer step")
    ordinal = _u64(stratum_ordinal, name="stratum ordinal")
    digest = hashlib.sha256(
        ASGCV_SELECTION_DOMAIN + seed + step.to_bytes(8, "big") + ordinal.to_bytes(8, "big")
    ).digest()
    return int.from_bytes(digest[:8], "big") % ASGCV_STRATUM_SIZE


def selection_schedule_sha256(
    selection_seed: object,
    *,
    optimizer_steps: int,
    strata_per_step: int,
) -> str:
    """Digest the exact step-major ASG-CV pair-selection schedule."""

    seed = _selection_seed_bytes(selection_seed)
    steps = _u64(optimizer_steps, name="optimizer step count")
    strata = _u64(strata_per_step, name="strata per step")
    if steps <= 0 or strata <= 0 or steps * strata > ASGCV_MAX_SCHEDULE_SELECTIONS:
        raise ValueError("ASG-CV selection schedule size differs")
    selections = bytearray()
    for step in range(steps):
        for ordinal in range(strata):
            selections.append(
                select_stratum_index(
                    selection_seed,
                    optimizer_step=step,
                    stratum_ordinal=ordinal,
                )
            )
    payload = (
        ASGCV_SCHEDULE_DOMAIN
        + seed
        + steps.to_bytes(8, "big")
        + strata.to_bytes(8, "big")
        + selections
    )
    return hashlib.sha256(payload).hexdigest()


def _require_float64_array(
    value: object,
    *,
    name: str,
    dimensions: int,
) -> Float64Array:
    if type(value) is not np.ndarray:
        raise ValueError(f"ASG-CV {name} differs")
    if value.dtype != np.dtype(np.float64) or value.ndim != dimensions:
        raise ValueError(f"ASG-CV {name} differs")
    if any(size <= 0 for size in value.shape) or not bool(np.isfinite(value).all()):
        raise ValueError(f"ASG-CV {name} differs")
    return value


def _require_float32_array(
    value: object,
    *,
    name: str,
    dimensions: int,
) -> Float32Array:
    if type(value) is not np.ndarray:
        raise ValueError(f"ASG-CV {name} differs")
    if value.dtype != np.dtype(np.float32) or value.ndim != dimensions:
        raise ValueError(f"ASG-CV {name} differs")
    if any(size <= 0 for size in value.shape) or not bool(np.isfinite(value).all()):
        raise ValueError(f"ASG-CV {name} differs")
    return value


@dataclass(frozen=True, slots=True)
class AsgcvAuthority:
    """Frozen scalar estimator parameters."""

    stratum_size: int = ASGCV_STRATUM_SIZE
    predictor_rank: int = ASGCV_PREDICTOR_RANK

    def validated(self) -> AsgcvAuthority:
        _require_exact_positive_integer(
            self.stratum_size,
            expected=ASGCV_STRATUM_SIZE,
            name="stratum size",
        )
        _require_exact_positive_integer(
            self.predictor_rank,
            expected=ASGCV_PREDICTOR_RANK,
            name="predictor rank",
        )
        return self

    def to_mapping(self) -> dict[str, object]:
        self.validated()
        return {
            "schema": ASGCV_SCHEMA,
            "stratum_size": self.stratum_size,
            "predictor_rank": self.predictor_rank,
            "accumulator_dtype": "float64",
            "selection_policy": ASGCV_SELECTION_POLICY,
        }

    @classmethod
    def from_mapping(cls, value: object) -> AsgcvAuthority:
        if type(value) is not dict or set(value) != {
            "schema",
            "stratum_size",
            "predictor_rank",
            "accumulator_dtype",
            "selection_policy",
        }:
            raise ValueError("ASG-CV authority schema differs")
        if (
            value["schema"] != ASGCV_SCHEMA
            or type(value["schema"]) is not str
            or value["accumulator_dtype"] != "float64"
            or type(value["accumulator_dtype"]) is not str
            or value["selection_policy"] != ASGCV_SELECTION_POLICY
            or type(value["selection_policy"]) is not str
        ):
            raise ValueError("ASG-CV authority differs")
        authority = cls(
            stratum_size=value["stratum_size"],
            predictor_rank=value["predictor_rank"],
        )
        return authority.validated()


@dataclass(frozen=True, slots=True)
class AsgcvSrhtAuthority:
    """Frozen CPU-reference authority for the ASG-CV gradient sketch."""

    input_dimensions: int
    padded_dimensions: int
    output_dimensions: int
    seed_sha256: str

    def validated(self) -> AsgcvSrhtAuthority:
        for name in ("input_dimensions", "padded_dimensions", "output_dimensions"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError("ASG-CV SRHT dimension differs")
        expected_padded = 1 << (self.input_dimensions - 1).bit_length()
        if self.padded_dimensions != expected_padded:
            raise ValueError("ASG-CV SRHT padded dimension differs")
        if self.output_dimensions > self.padded_dimensions:
            raise ValueError("ASG-CV SRHT output dimension differs")
        _sha256_bytes(self.seed_sha256, name="SRHT seed")
        return self

    def to_mapping(self) -> dict[str, object]:
        self.validated()
        return {
            "schema": ASGCV_SRHT_SCHEMA,
            "input_dimensions": self.input_dimensions,
            "padded_dimensions": self.padded_dimensions,
            "output_dimensions": self.output_dimensions,
            "seed_sha256": self.seed_sha256,
            "accumulator_dtype": "float64",
            "normalization": ASGCV_SRHT_NORMALIZATION,
        }

    @classmethod
    def from_mapping(cls, value: object) -> AsgcvSrhtAuthority:
        if type(value) is not dict or set(value) != {
            "schema",
            "input_dimensions",
            "padded_dimensions",
            "output_dimensions",
            "seed_sha256",
            "accumulator_dtype",
            "normalization",
        }:
            raise ValueError("ASG-CV SRHT authority schema differs")
        if (
            type(value["schema"]) is not str
            or value["schema"] != ASGCV_SRHT_SCHEMA
            or type(value["accumulator_dtype"]) is not str
            or value["accumulator_dtype"] != "float64"
            or type(value["normalization"]) is not str
            or value["normalization"] != ASGCV_SRHT_NORMALIZATION
        ):
            raise ValueError("ASG-CV SRHT authority differs")
        return cls(
            input_dimensions=value["input_dimensions"],
            padded_dimensions=value["padded_dimensions"],
            output_dimensions=value["output_dimensions"],
            seed_sha256=value["seed_sha256"],
        ).validated()


def srht_signs_and_rows(
    authority: AsgcvSrhtAuthority,
) -> tuple[Float64Array, NDArray[np.int64]]:
    """Materialize source-bound signs and row ordinals for audit evidence."""

    if type(authority) is not AsgcvSrhtAuthority:
        raise ValueError("ASG-CV SRHT authority differs")
    authority.validated()
    seed = _sha256_bytes(authority.seed_sha256, name="SRHT seed")
    signs = np.empty(authority.padded_dimensions, dtype=np.float64)
    row_scores: list[tuple[bytes, int]] = []
    for index in range(authority.padded_dimensions):
        encoded_index = index.to_bytes(8, "big")
        sign_digest = hashlib.sha256(ASGCV_SRHT_SIGN_DOMAIN + seed + encoded_index).digest()
        signs[index] = 1.0 if sign_digest[0] & 1 == 0 else -1.0
        row_scores.append(
            (
                hashlib.sha256(ASGCV_SRHT_ROW_DOMAIN + seed + encoded_index).digest(),
                index,
            )
        )
    rows = np.asarray(
        [index for _, index in sorted(row_scores)[: authority.output_dimensions]],
        dtype=np.int64,
    )
    return signs, rows


def srht_gradient_sketch(
    field: object,
    authority: AsgcvSrhtAuthority,
) -> Float64Array:
    """Apply the fixed-order fp64 SRHT scalar reference to one patch field."""

    if type(authority) is not AsgcvSrhtAuthority:
        raise ValueError("ASG-CV SRHT authority differs")
    authority.validated()
    field_array = _require_float64_array(
        field,
        name="SRHT gradient field",
        dimensions=2,
    )
    if field_array.shape[1] != authority.input_dimensions:
        raise ValueError("ASG-CV SRHT gradient shape differs")
    signs, rows = srht_signs_and_rows(authority)
    work = np.zeros(
        (field_array.shape[0], authority.padded_dimensions),
        dtype=np.float64,
    )
    work[:, : authority.input_dimensions] = field_array * signs[: authority.input_dimensions]
    width = 1
    while width < authority.padded_dimensions:
        for start in range(0, authority.padded_dimensions, width * 2):
            left = work[:, start : start + width].copy()
            right = work[:, start + width : start + 2 * width].copy()
            work[:, start : start + width] = left + right
            work[:, start + width : start + 2 * width] = left - right
        width *= 2
    work /= np.sqrt(float(authority.padded_dimensions))
    result = work[:, rows] * np.sqrt(
        float(authority.padded_dimensions) / authority.output_dimensions
    )
    if not bool(np.isfinite(result).all()):
        raise ValueError("ASG-CV SRHT result is not finite")
    return np.asarray(result, dtype=np.float64)


def _validated_stratum_pair(
    exact: object,
    predicted: object,
) -> tuple[Float64Array, Float64Array]:
    exact_array = _require_float64_array(
        exact,
        name="exact gradient stratum",
        dimensions=4,
    )
    predicted_array = _require_float64_array(
        predicted,
        name="predicted gradient stratum",
        dimensions=4,
    )
    if (
        exact_array.shape != predicted_array.shape
        or exact_array.shape[0] != ASGCV_STRATUM_SIZE
        or exact_array.shape[1] != 2
    ):
        raise ValueError("ASG-CV gradient stratum shape differs")
    return exact_array, predicted_array


def asgcv_stratum_gradient(
    predicted: object,
    exact_selected: object,
    *,
    selected_index: int,
) -> Float64Array:
    """Return the authoritative one-of-eight stratum gradient estimate."""

    predicted_array = _require_float64_array(
        predicted,
        name="predicted gradient stratum",
        dimensions=4,
    )
    exact_array = _require_float64_array(
        exact_selected,
        name="selected exact gradient",
        dimensions=3,
    )
    if predicted_array.shape[0] != ASGCV_STRATUM_SIZE or predicted_array.shape[1] != 2:
        raise ValueError("ASG-CV gradient stratum shape differs")
    if exact_array.shape != predicted_array.shape[1:]:
        raise ValueError("ASG-CV selected gradient shape differs")
    if type(selected_index) is not int or not 0 <= selected_index < ASGCV_STRATUM_SIZE:
        raise ValueError("ASG-CV selected index differs")
    predicted_mean = np.mean(predicted_array, axis=0, dtype=np.float64)
    result = predicted_mean + exact_array - predicted_array[selected_index]
    if not bool(np.isfinite(result).all()):
        raise ValueError("ASG-CV stratum gradient is not finite")
    return np.asarray(result, dtype=np.float64)


def exhaustive_selection_mean(exact: object, predicted: object) -> Float64Array:
    """Average every registered selection to audit estimator unbiasedness."""

    exact_array, predicted_array = _validated_stratum_pair(exact, predicted)
    estimates = np.stack(
        [
            asgcv_stratum_gradient(
                predicted_array,
                exact_array[index],
                selected_index=index,
            )
            for index in range(ASGCV_STRATUM_SIZE)
        ]
    )
    return np.mean(estimates, axis=0, dtype=np.float64)


def normalized_residual_energy(exact: object, predicted: object) -> float:
    """Measure predictor residual energy relative to exact semantic energy."""

    exact_array, predicted_array = _validated_stratum_pair(exact, predicted)
    exact_energy = float(np.square(exact_array).sum(dtype=np.float64))
    if not np.isfinite(exact_energy) or exact_energy <= 0.0:
        raise ValueError("ASG-CV exact gradient energy differs")
    residual_energy = float(np.square(exact_array - predicted_array).sum(dtype=np.float64))
    ratio = residual_energy / exact_energy
    if not np.isfinite(ratio) or ratio < 0.0:
        raise ValueError("ASG-CV residual energy differs")
    return ratio


def _batch_normalized_residual_energy(exact: Float64Array, predicted: Float64Array) -> float:
    numerator = float(np.square(exact - predicted).sum(dtype=np.float64))
    denominator = float(np.square(exact).sum(dtype=np.float64))
    if (
        not math.isfinite(numerator)
        or numerator < 0.0
        or not math.isfinite(denominator)
        or denominator <= 0.0
    ):
        raise ValueError("ASG-CV E0 normalized residual energy differs")
    return numerator / denominator


def selection_variance_ratio(exact: object, predicted: object) -> float:
    """Compare selection variance with a one-exact-pair estimator."""

    exact_array, predicted_array = _validated_stratum_pair(exact, predicted)
    target_mean = np.mean(exact_array, axis=0, dtype=np.float64)
    baseline_variance = float(
        np.square(exact_array - target_mean).sum(dtype=np.float64) / ASGCV_STRATUM_SIZE
    )
    if not np.isfinite(baseline_variance) or baseline_variance <= 0.0:
        raise ValueError("ASG-CV baseline selection variance differs")
    estimates = np.stack(
        [
            asgcv_stratum_gradient(
                predicted_array,
                exact_array[index],
                selected_index=index,
            )
            for index in range(ASGCV_STRATUM_SIZE)
        ]
    )
    estimate_mean = np.mean(estimates, axis=0, dtype=np.float64)
    estimator_variance = float(
        np.square(estimates - estimate_mean).sum(dtype=np.float64) / ASGCV_STRATUM_SIZE
    )
    ratio = estimator_variance / baseline_variance
    if not np.isfinite(ratio) or ratio < 0.0:
        raise ValueError("ASG-CV selection variance ratio differs")
    return ratio


@dataclass(frozen=True, slots=True)
class AsgcvE0Metrics:
    """Outcome-blind gradient fidelity and variance gates for ASG-CV E0."""

    pair_count: int
    dense_gradient_cosine_ppm: int
    projected_gradient_cosine_ppm: int
    patch_salience_spearman_ppm: int
    normalized_residual_energy_ppm: int
    selection_variance_ratio_ppm: int
    mean_agreement_upper_ppm: int
    preclip_p99_ratio_ppm: int
    exact_clip_rate_ppm: int
    asgcv_clip_rate_ppm: int
    clip_rate_delta_ppm: int
    semantic_wall_ratio_ppm: int
    peak_cuda_reserved_bytes: int
    passed: bool

    def validated(self) -> AsgcvE0Metrics:
        if (
            type(self.pair_count) is not int
            or self.pair_count < ASGCV_E0_MINIMUM_PAIRS
            or self.pair_count % ASGCV_STRATUM_SIZE != 0
        ):
            raise ValueError("ASG-CV E0 pair count differs")
        similarity_names = (
            "dense_gradient_cosine_ppm",
            "projected_gradient_cosine_ppm",
            "patch_salience_spearman_ppm",
        )
        ratio_names = (
            "normalized_residual_energy_ppm",
            "selection_variance_ratio_ppm",
            "mean_agreement_upper_ppm",
            "preclip_p99_ratio_ppm",
            "clip_rate_delta_ppm",
            "semantic_wall_ratio_ppm",
        )
        for name in similarity_names + ratio_names:
            if type(getattr(self, name)) is not int:
                raise ValueError("ASG-CV E0 metric type differs")
        if any(not -1_000_000 <= getattr(self, name) <= 1_000_000 for name in similarity_names):
            raise ValueError("ASG-CV E0 similarity metric differs")
        if any(getattr(self, name) < 0 for name in ratio_names):
            raise ValueError("ASG-CV E0 ratio metric differs")
        for name in ("exact_clip_rate_ppm", "asgcv_clip_rate_ppm"):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value <= 1_000_000:
                raise ValueError("ASG-CV E0 clip rate differs")
        if type(self.peak_cuda_reserved_bytes) is not int or self.peak_cuda_reserved_bytes <= 0:
            raise ValueError("ASG-CV E0 peak CUDA memory differs")
        expected_pass = (
            self.dense_gradient_cosine_ppm >= ASGCV_DENSE_COSINE_GATE_PPM
            and self.projected_gradient_cosine_ppm >= ASGCV_PROJECTED_COSINE_GATE_PPM
            and self.patch_salience_spearman_ppm >= ASGCV_PATCH_SPEARMAN_GATE_PPM
            and self.normalized_residual_energy_ppm <= ASGCV_RESIDUAL_ENERGY_GATE_PPM
            and self.selection_variance_ratio_ppm <= ASGCV_VARIANCE_RATIO_GATE_PPM
            and self.mean_agreement_upper_ppm <= ASGCV_MEAN_AGREEMENT_GATE_PPM
            and self.preclip_p99_ratio_ppm <= ASGCV_PRECLIP_P99_RATIO_GATE_PPM
            and self.clip_rate_delta_ppm <= ASGCV_CLIP_RATE_DELTA_GATE_PPM
            and self.semantic_wall_ratio_ppm <= ASGCV_SEMANTIC_WALL_RATIO_GATE_PPM
            and self.peak_cuda_reserved_bytes <= ASGCV_PEAK_CUDA_RESERVED_GATE_BYTES
        )
        if type(self.passed) is not bool or self.passed is not expected_pass:
            raise ValueError("ASG-CV E0 pass gate differs")
        return self

    def to_mapping(self) -> dict[str, object]:
        self.validated()
        return {
            "schema": ASGCV_E0_SCHEMA,
            "pair_count": self.pair_count,
            "dense_gradient_cosine_ppm": self.dense_gradient_cosine_ppm,
            "projected_gradient_cosine_ppm": self.projected_gradient_cosine_ppm,
            "patch_salience_spearman_ppm": self.patch_salience_spearman_ppm,
            "normalized_residual_energy_ppm": self.normalized_residual_energy_ppm,
            "selection_variance_ratio_ppm": self.selection_variance_ratio_ppm,
            "mean_agreement_upper_ppm": self.mean_agreement_upper_ppm,
            "preclip_p99_ratio_ppm": self.preclip_p99_ratio_ppm,
            "exact_clip_rate_ppm": self.exact_clip_rate_ppm,
            "asgcv_clip_rate_ppm": self.asgcv_clip_rate_ppm,
            "clip_rate_delta_ppm": self.clip_rate_delta_ppm,
            "semantic_wall_ratio_ppm": self.semantic_wall_ratio_ppm,
            "peak_cuda_reserved_bytes": self.peak_cuda_reserved_bytes,
            "passed": self.passed,
        }

    @classmethod
    def from_mapping(cls, value: object) -> AsgcvE0Metrics:
        if type(value) is not dict or set(value) != {
            "schema",
            "pair_count",
            "dense_gradient_cosine_ppm",
            "projected_gradient_cosine_ppm",
            "patch_salience_spearman_ppm",
            "normalized_residual_energy_ppm",
            "selection_variance_ratio_ppm",
            "mean_agreement_upper_ppm",
            "preclip_p99_ratio_ppm",
            "exact_clip_rate_ppm",
            "asgcv_clip_rate_ppm",
            "clip_rate_delta_ppm",
            "semantic_wall_ratio_ppm",
            "peak_cuda_reserved_bytes",
            "passed",
        }:
            raise ValueError("ASG-CV E0 metrics schema differs")
        if type(value["schema"]) is not str or value["schema"] != ASGCV_E0_SCHEMA:
            raise ValueError("ASG-CV E0 metrics authority differs")
        return cls(
            pair_count=value["pair_count"],
            dense_gradient_cosine_ppm=value["dense_gradient_cosine_ppm"],
            projected_gradient_cosine_ppm=value["projected_gradient_cosine_ppm"],
            patch_salience_spearman_ppm=value["patch_salience_spearman_ppm"],
            normalized_residual_energy_ppm=value["normalized_residual_energy_ppm"],
            selection_variance_ratio_ppm=value["selection_variance_ratio_ppm"],
            mean_agreement_upper_ppm=value["mean_agreement_upper_ppm"],
            preclip_p99_ratio_ppm=value["preclip_p99_ratio_ppm"],
            exact_clip_rate_ppm=value["exact_clip_rate_ppm"],
            asgcv_clip_rate_ppm=value["asgcv_clip_rate_ppm"],
            clip_rate_delta_ppm=value["clip_rate_delta_ppm"],
            semantic_wall_ratio_ppm=value["semantic_wall_ratio_ppm"],
            peak_cuda_reserved_bytes=value["peak_cuda_reserved_bytes"],
            passed=value["passed"],
        ).validated()


@dataclass(frozen=True, slots=True)
class AsgcvE0CapacityFloor:
    """Query-independent lower bounds on achievable predictor residual energy."""

    pair_count: int
    conditional_variance_floor_ppm: int
    fixed_channel_residual_floor_ppm: int
    per_sample_rank_residual_floor_ppm: int
    passed: bool

    def validated(self) -> AsgcvE0CapacityFloor:
        if (
            type(self.pair_count) is not int
            or self.pair_count < ASGCV_E0_CAPACITY_MINIMUM_PAIRS
            or self.pair_count % ASGCV_STRATUM_SIZE != 0
        ):
            raise ValueError("ASG-CV E0 capacity pair count differs")
        metric_names = (
            "conditional_variance_floor_ppm",
            "fixed_channel_residual_floor_ppm",
            "per_sample_rank_residual_floor_ppm",
        )
        if any(
            type(getattr(self, name)) is not int or getattr(self, name) < 0 for name in metric_names
        ):
            raise ValueError("ASG-CV E0 capacity metric differs")
        expected_pass = all(
            getattr(self, name) <= ASGCV_RESIDUAL_ENERGY_GATE_PPM for name in metric_names
        )
        if type(self.passed) is not bool or self.passed is not expected_pass:
            raise ValueError("ASG-CV E0 capacity pass gate differs")
        return self

    def to_mapping(self) -> dict[str, object]:
        self.validated()
        return {
            "schema": ASGCV_E0_CAPACITY_FLOOR_SCHEMA,
            "pair_count": self.pair_count,
            "conditional_variance_floor_ppm": self.conditional_variance_floor_ppm,
            "fixed_channel_residual_floor_ppm": self.fixed_channel_residual_floor_ppm,
            "per_sample_rank_residual_floor_ppm": self.per_sample_rank_residual_floor_ppm,
            "passed": self.passed,
        }

    @classmethod
    def from_mapping(cls, value: object) -> AsgcvE0CapacityFloor:
        expected_keys = {
            "schema",
            "pair_count",
            "conditional_variance_floor_ppm",
            "fixed_channel_residual_floor_ppm",
            "per_sample_rank_residual_floor_ppm",
            "passed",
        }
        if (
            type(value) is not dict
            or set(value) != expected_keys
            or value["schema"] != ASGCV_E0_CAPACITY_FLOOR_SCHEMA
            or type(value["schema"]) is not str
        ):
            raise ValueError("ASG-CV E0 capacity schema differs")
        return cls(
            pair_count=value["pair_count"],
            conditional_variance_floor_ppm=value["conditional_variance_floor_ppm"],
            fixed_channel_residual_floor_ppm=value["fixed_channel_residual_floor_ppm"],
            per_sample_rank_residual_floor_ppm=value["per_sample_rank_residual_floor_ppm"],
            passed=value["passed"],
        ).validated()


def evaluate_e0_capacity_floor(
    first_seed_gradients: object,
    second_seed_gradients: object,
) -> AsgcvE0CapacityFloor:
    """Measure variance and rank floors before fitting the ASG-CV predictor."""

    first = _require_float64_array(
        first_seed_gradients,
        name="E0 first-seed gradient batch",
        dimensions=4,
    )
    second = _require_float64_array(
        second_seed_gradients,
        name="E0 second-seed gradient batch",
        dimensions=4,
    )
    if (
        first.shape != second.shape
        or first.shape[0] < ASGCV_E0_CAPACITY_MINIMUM_PAIRS
        or first.shape[1] != 2
    ):
        raise ValueError("ASG-CV E0 capacity batch shape differs")
    first_energy = float(np.square(first).sum(dtype=np.float64))
    second_energy = float(np.square(second).sum(dtype=np.float64))
    total_energy = first_energy + second_energy
    if not math.isfinite(total_energy) or total_energy <= 0.0:
        raise ValueError("ASG-CV E0 capacity energy differs")
    conditional_floor = float(np.square(first - second).sum(dtype=np.float64) / total_energy)

    combined = np.concatenate((first, second), axis=0)
    channel_matrix = combined.reshape(-1, combined.shape[-1])
    channel_singular_values = np.linalg.svd(channel_matrix, compute_uv=False)
    fixed_residual_energy = float(
        np.square(channel_singular_values[ASGCV_PREDICTOR_RANK:]).sum(dtype=np.float64)
    )
    fixed_floor = fixed_residual_energy / total_energy

    per_sample_residual_energy = 0.0
    for sample in combined:
        singular_values = np.linalg.svd(
            sample.reshape(-1, sample.shape[-1]),
            compute_uv=False,
        )
        per_sample_residual_energy += float(
            np.square(singular_values[ASGCV_PREDICTOR_RANK:]).sum(dtype=np.float64)
        )
    per_sample_floor = per_sample_residual_energy / total_energy
    metrics = {
        "pair_count": first.shape[0],
        "conditional_variance_floor_ppm": _ratio_ppm(conditional_floor),
        "fixed_channel_residual_floor_ppm": _ratio_ppm(fixed_floor),
        "per_sample_rank_residual_floor_ppm": _ratio_ppm(per_sample_floor),
    }
    passed = all(
        value <= ASGCV_RESIDUAL_ENERGY_GATE_PPM
        for name, value in metrics.items()
        if name != "pair_count"
    )
    return AsgcvE0CapacityFloor(**metrics, passed=passed).validated()


def _median_cosine(exact: Float64Array, predicted: Float64Array) -> float:
    flattened_exact = exact.reshape(exact.shape[0] * exact.shape[1], -1)
    flattened_predicted = predicted.reshape(predicted.shape[0] * predicted.shape[1], -1)
    exact_norms = np.linalg.norm(flattened_exact, axis=1)
    predicted_norms = np.linalg.norm(flattened_predicted, axis=1)
    denominators = exact_norms * predicted_norms
    if not bool(np.isfinite(denominators).all()) or bool((denominators <= 0.0).any()):
        raise ValueError("ASG-CV gradient cosine denominator differs")
    values = np.sum(flattened_exact * flattened_predicted, axis=1) / denominators
    if not bool(np.isfinite(values).all()):
        raise ValueError("ASG-CV gradient cosine differs")
    return float(np.median(np.clip(values, -1.0, 1.0)))


def _average_ranks(values: Float64Array) -> Float64Array:
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    ranks = np.empty(values.shape, dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def _spearman(exact: Float64Array, predicted: Float64Array) -> float:
    exact_ranks = _average_ranks(exact)
    predicted_ranks = _average_ranks(predicted)
    exact_centered = exact_ranks - exact_ranks.mean(dtype=np.float64)
    predicted_centered = predicted_ranks - predicted_ranks.mean(dtype=np.float64)
    denominator = float(np.linalg.norm(exact_centered) * np.linalg.norm(predicted_centered))
    if not np.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("ASG-CV patch salience rank denominator differs")
    value = float(np.dot(exact_centered, predicted_centered) / denominator)
    if not np.isfinite(value):
        raise ValueError("ASG-CV patch salience rank differs")
    return min(1.0, max(-1.0, value))


def _median_patch_salience_spearman(
    exact: Float64Array,
    predicted: Float64Array,
) -> float:
    exact_salience = np.linalg.norm(exact, axis=-1).reshape(-1, exact.shape[-2])
    predicted_salience = np.linalg.norm(predicted, axis=-1).reshape(-1, predicted.shape[-2])
    if exact_salience.shape[1] < 2:
        raise ValueError("ASG-CV patch count differs")
    values = [
        _spearman(exact_salience[index], predicted_salience[index])
        for index in range(exact_salience.shape[0])
    ]
    return float(np.median(np.asarray(values, dtype=np.float64)))


def _batch_selection_variance_ratio(
    exact: Float64Array,
    predicted: Float64Array,
) -> float:
    if np.array_equal(exact, predicted):
        return 0.0
    baseline_sum = 0.0
    estimator_sum = 0.0
    for stratum_index in range(exact.shape[0]):
        exact_stratum = exact[stratum_index]
        predicted_stratum = predicted[stratum_index]
        target_mean = np.mean(exact_stratum, axis=0, dtype=np.float64)
        baseline_sum += float(
            np.square(exact_stratum - target_mean).sum(dtype=np.float64) / ASGCV_STRATUM_SIZE
        )
        estimates = np.stack(
            [
                asgcv_stratum_gradient(
                    predicted_stratum,
                    exact_stratum[index],
                    selected_index=index,
                )
                for index in range(ASGCV_STRATUM_SIZE)
            ]
        )
        estimate_mean = np.mean(estimates, axis=0, dtype=np.float64)
        estimator_sum += float(
            np.square(estimates - estimate_mean).sum(dtype=np.float64) / ASGCV_STRATUM_SIZE
        )
    if not np.isfinite(baseline_sum) or baseline_sum <= 0.0:
        raise ValueError("ASG-CV baseline selection variance differs")
    ratio = estimator_sum / baseline_sum
    if not np.isfinite(ratio) or ratio < 0.0:
        raise ValueError("ASG-CV selection variance ratio differs")
    return ratio


def _similarity_ppm(value: float) -> int:
    return int(round(min(1.0, max(-1.0, value)) * 1_000_000))


def _ratio_ppm(value: float) -> int:
    if not np.isfinite(value) or value < 0.0:
        raise ValueError("ASG-CV E0 ratio differs")
    return int(np.ceil(value * 1_000_000))


def _mean_bootstrap_indices(seed: bytes, *, strata: int) -> NDArray[np.uint16]:
    if strata < ASGCV_E0_MINIMUM_PAIRS // ASGCV_STRATUM_SIZE or strata > 256:
        raise ValueError("ASG-CV E0 mean bootstrap shape differs")
    limit = 256 - (256 % strata)
    output = np.empty((ASGCV_MEAN_BOOTSTRAP_DRAWS, strata), dtype=np.uint16)
    for draw in range(ASGCV_MEAN_BOOTSTRAP_DRAWS):
        filled = 0
        block = 0
        while filled < strata:
            digest = hashlib.sha256(
                ASGCV_MEAN_BOOTSTRAP_DOMAIN
                + seed
                + draw.to_bytes(8, "big")
                + block.to_bytes(8, "big")
            ).digest()
            for byte in digest:
                if byte < limit:
                    output[draw, filled] = byte % strata
                    filled += 1
                    if filled == strata:
                        break
            block += 1
    return output


def _mean_agreement_upper_ppm(
    exact_estimates: Float64Array,
    estimator_deltas: Float64Array,
    *,
    selection_seed: bytes,
) -> int:
    deltas = np.ascontiguousarray(
        estimator_deltas.reshape(exact_estimates.shape[0], -1),
        dtype=np.float64,
    )
    targets = exact_estimates.reshape(exact_estimates.shape[0], -1)
    scale_energy = float(np.mean(np.einsum("ij,ij->i", targets, targets, dtype=np.float64)))
    if not math.isfinite(scale_energy) or scale_energy <= 0.0:
        raise ValueError("ASG-CV E0 mean agreement scale differs")
    gram = deltas @ deltas.T
    if not bool(np.isfinite(gram).all()):
        raise ValueError("ASG-CV E0 mean agreement Gram differs")
    if not bool(np.count_nonzero(gram)):
        return 0
    indices = _mean_bootstrap_indices(selection_seed, strata=deltas.shape[0])
    weights = np.zeros((indices.shape[0], deltas.shape[0]), dtype=np.float64)
    rows = np.repeat(np.arange(indices.shape[0]), indices.shape[1])
    np.add.at(weights, (rows, indices.reshape(-1)), 1.0)
    squared = np.einsum("bi,ij,bj->b", weights, gram, weights, dtype=np.float64)
    squared /= float(deltas.shape[0] ** 2)
    tolerance = (
        np.finfo(np.float64).eps * max(1.0, float(np.max(np.abs(squared)))) * deltas.shape[0]
    )
    if not bool(np.isfinite(squared).all()) or float(np.min(squared)) < -tolerance:
        raise ValueError("ASG-CV E0 mean agreement bootstrap differs")
    squared = np.maximum(squared, 0.0)
    upper = float(np.quantile(squared, 0.95, method="higher"))
    return _ratio_ppm(math.sqrt(upper / scale_energy))


@dataclass(frozen=True, slots=True)
class _E0BlockStatistics:
    patch_salience_spearman: float
    selection_variance_ratio: float
    mean_agreement_upper_ppm: int
    exact_norms: Float64Array
    asgcv_norms: Float64Array


def _e0_block_statistics(
    exact: Float64Array,
    predicted: Float64Array,
    *,
    selected_indices: tuple[int, ...],
    selection_seed: bytes,
) -> _E0BlockStatistics:
    exact_estimates = np.mean(exact, axis=1, dtype=np.float64)
    asgcv_estimates = np.stack(
        [
            asgcv_stratum_gradient(
                predicted[stratum_ordinal],
                exact[stratum_ordinal, selected_index],
                selected_index=selected_index,
            )
            for stratum_ordinal, selected_index in enumerate(selected_indices)
        ]
    )
    residuals = exact - predicted
    estimator_deltas = np.stack(
        [
            residuals[stratum_ordinal, selected_index]
            - np.mean(residuals[stratum_ordinal], axis=0, dtype=np.float64)
            for stratum_ordinal, selected_index in enumerate(selected_indices)
        ]
    )
    exact_norms = np.linalg.norm(exact_estimates.reshape(exact_estimates.shape[0], -1), axis=1)
    asgcv_norms = np.linalg.norm(asgcv_estimates.reshape(asgcv_estimates.shape[0], -1), axis=1)
    return _E0BlockStatistics(
        patch_salience_spearman=_median_patch_salience_spearman(exact, predicted),
        selection_variance_ratio=_batch_selection_variance_ratio(exact, predicted),
        mean_agreement_upper_ppm=_mean_agreement_upper_ppm(
            exact_estimates,
            estimator_deltas,
            selection_seed=selection_seed,
        ),
        exact_norms=exact_norms,
        asgcv_norms=asgcv_norms,
    )


def evaluate_e0(
    exact: object,
    predicted: object,
    srht_authority: AsgcvSrhtAuthority,
    *,
    second_exact: object,
    selection_seed_sha256: object,
    peak_cuda_reserved_bytes: int,
    exact_semantic_wall_ns: int,
    asgcv_semantic_wall_ns: int,
) -> AsgcvE0Metrics:
    """Recompute the registered E0 fidelity and variance evidence."""

    exact_array = _require_float64_array(
        exact,
        name="E0 exact gradient batch",
        dimensions=5,
    )
    predicted_array = _require_float64_array(
        predicted,
        name="E0 predicted gradient batch",
        dimensions=5,
    )
    second_exact_array = _require_float64_array(
        second_exact,
        name="E0 second exact gradient batch",
        dimensions=5,
    )
    selection_seed = _selection_seed_bytes(selection_seed_sha256)
    if type(exact_semantic_wall_ns) is not int or exact_semantic_wall_ns <= 0:
        raise ValueError("ASG-CV E0 exact semantic wall time differs")
    if type(asgcv_semantic_wall_ns) is not int or asgcv_semantic_wall_ns <= 0:
        raise ValueError("ASG-CV E0 semantic wall time differs")
    if type(peak_cuda_reserved_bytes) is not int or peak_cuda_reserved_bytes <= 0:
        raise ValueError("ASG-CV E0 peak CUDA memory differs")
    semantic_wall_ratio_ppm = (
        asgcv_semantic_wall_ns * 1_000_000 + exact_semantic_wall_ns - 1
    ) // exact_semantic_wall_ns
    if (
        exact_array.shape != predicted_array.shape
        or exact_array.shape != second_exact_array.shape
        or exact_array.shape[1] != ASGCV_STRATUM_SIZE
        or exact_array.shape[2] != 2
    ):
        raise ValueError("ASG-CV E0 gradient batch shape differs")
    if type(srht_authority) is not AsgcvSrhtAuthority:
        raise ValueError("ASG-CV E0 SRHT authority differs")
    srht_authority.validated()
    if srht_authority.input_dimensions != exact_array.shape[-1]:
        raise ValueError("ASG-CV E0 SRHT shape differs")
    selected_indices = tuple(
        select_stratum_index(
            selection_seed.hex(),
            optimizer_step=0,
            stratum_ordinal=stratum_ordinal,
        )
        for stratum_ordinal in range(exact_array.shape[0])
    )
    first_statistics = _e0_block_statistics(
        exact_array,
        predicted_array,
        selected_indices=selected_indices,
        selection_seed=selection_seed,
    )
    second_statistics = _e0_block_statistics(
        second_exact_array,
        predicted_array,
        selected_indices=selected_indices,
        selection_seed=selection_seed,
    )
    combined_exact_norms = np.concatenate(
        (first_statistics.exact_norms, second_statistics.exact_norms)
    )
    combined_asgcv_norms = np.concatenate(
        (first_statistics.asgcv_norms, second_statistics.asgcv_norms)
    )
    exact_p99 = float(np.quantile(combined_exact_norms, 0.99, method="higher"))
    asgcv_p99 = float(np.quantile(combined_asgcv_norms, 0.99, method="higher"))
    clip_reference = float(
        np.quantile(combined_exact_norms, ASGCV_CLIP_REFERENCE_QUANTILE, method="higher")
    )
    if (
        not np.isfinite(exact_p99)
        or exact_p99 <= 0.0
        or not np.isfinite(asgcv_p99)
        or not np.isfinite(clip_reference)
        or clip_reference <= 0.0
    ):
        raise ValueError("ASG-CV E0 combined norm authority differs")
    clip_comparison_tolerance = (
        64.0 * np.finfo(np.float64).eps * max(clip_reference, np.finfo(np.float64).tiny)
    )

    def combined_clip_rate(values: Float64Array) -> int:
        return int(
            round(
                float(np.count_nonzero(values > clip_reference + clip_comparison_tolerance))
                / len(values)
                * 1_000_000
            )
        )

    exact_clip_rate_ppm = combined_clip_rate(combined_exact_norms)
    asgcv_clip_rate_ppm = combined_clip_rate(combined_asgcv_norms)

    projected_exact = srht_gradient_sketch(
        exact_array.reshape(-1, exact_array.shape[-1]),
        srht_authority,
    ).reshape(*exact_array.shape[:-1], srht_authority.output_dimensions)
    projected_predicted = srht_gradient_sketch(
        predicted_array.reshape(-1, predicted_array.shape[-1]),
        srht_authority,
    ).reshape(*predicted_array.shape[:-1], srht_authority.output_dimensions)
    projected_second_exact = srht_gradient_sketch(
        second_exact_array.reshape(-1, second_exact_array.shape[-1]),
        srht_authority,
    ).reshape(*second_exact_array.shape[:-1], srht_authority.output_dimensions)
    residual_energy = max(
        _batch_normalized_residual_energy(exact_array, predicted_array),
        _batch_normalized_residual_energy(second_exact_array, predicted_array),
    )
    metrics_without_pass = {
        "pair_count": exact_array.shape[0] * exact_array.shape[1],
        "dense_gradient_cosine_ppm": _similarity_ppm(
            min(
                _median_cosine(exact_array, predicted_array),
                _median_cosine(second_exact_array, predicted_array),
            )
        ),
        "projected_gradient_cosine_ppm": _similarity_ppm(
            min(
                _median_cosine(projected_exact, projected_predicted),
                _median_cosine(projected_second_exact, projected_predicted),
            )
        ),
        "patch_salience_spearman_ppm": _similarity_ppm(
            min(
                first_statistics.patch_salience_spearman,
                second_statistics.patch_salience_spearman,
            )
        ),
        "normalized_residual_energy_ppm": _ratio_ppm(residual_energy),
        "selection_variance_ratio_ppm": _ratio_ppm(
            max(
                first_statistics.selection_variance_ratio,
                second_statistics.selection_variance_ratio,
            )
        ),
        "mean_agreement_upper_ppm": max(
            first_statistics.mean_agreement_upper_ppm,
            second_statistics.mean_agreement_upper_ppm,
        ),
        "preclip_p99_ratio_ppm": _ratio_ppm(asgcv_p99 / exact_p99),
        "exact_clip_rate_ppm": exact_clip_rate_ppm,
        "asgcv_clip_rate_ppm": asgcv_clip_rate_ppm,
        "clip_rate_delta_ppm": max(0, asgcv_clip_rate_ppm - exact_clip_rate_ppm),
        "semantic_wall_ratio_ppm": semantic_wall_ratio_ppm,
        "peak_cuda_reserved_bytes": peak_cuda_reserved_bytes,
    }
    passed = (
        metrics_without_pass["dense_gradient_cosine_ppm"] >= ASGCV_DENSE_COSINE_GATE_PPM
        and metrics_without_pass["projected_gradient_cosine_ppm"] >= ASGCV_PROJECTED_COSINE_GATE_PPM
        and metrics_without_pass["patch_salience_spearman_ppm"] >= ASGCV_PATCH_SPEARMAN_GATE_PPM
        and metrics_without_pass["normalized_residual_energy_ppm"] <= ASGCV_RESIDUAL_ENERGY_GATE_PPM
        and metrics_without_pass["selection_variance_ratio_ppm"] <= ASGCV_VARIANCE_RATIO_GATE_PPM
        and metrics_without_pass["mean_agreement_upper_ppm"] <= ASGCV_MEAN_AGREEMENT_GATE_PPM
        and metrics_without_pass["preclip_p99_ratio_ppm"] <= ASGCV_PRECLIP_P99_RATIO_GATE_PPM
        and metrics_without_pass["clip_rate_delta_ppm"] <= ASGCV_CLIP_RATE_DELTA_GATE_PPM
        and metrics_without_pass["semantic_wall_ratio_ppm"] <= ASGCV_SEMANTIC_WALL_RATIO_GATE_PPM
        and peak_cuda_reserved_bytes <= ASGCV_PEAK_CUDA_RESERVED_GATE_BYTES
    )
    return AsgcvE0Metrics(**metrics_without_pass, passed=passed).validated()


def _gradient_sample_array_authority(
    value: object,
    *,
    role: str,
) -> dict[str, object]:
    array = _require_float32_array(value, name=f"gradient sample {role}", dimensions=3)
    role_bytes = role.encode("ascii")
    shape = tuple(int(size) for size in array.shape)
    frame = bytearray(ASGCV_GRADIENT_SAMPLE_ARRAY_DOMAIN)
    frame.extend(len(role_bytes).to_bytes(8, "big"))
    frame.extend(role_bytes)
    frame.extend(len(shape).to_bytes(8, "big"))
    for size in shape:
        frame.extend(size.to_bytes(8, "big"))
    frame.extend(np.ascontiguousarray(array, dtype="<f4").tobytes(order="C"))
    return {
        "dtype": "float32-le",
        "shape": list(shape),
        "sha256": hashlib.sha256(frame).hexdigest(),
    }


def canonical_gradient_sample_bytes(
    *,
    source_commit: object,
    model_revision: object,
    fixture_sha256: object,
    completion_group_sha256: object,
    completion_protocol_sha256: object,
    eligible_schedule_sha256: object,
    pooler_state_sha256: object,
    eligible_pair_ordinal: object,
    candidate_pair_ordinal: object,
    pair_ordinals: object,
    relation_sign: object,
    grpo_loss: object,
    attention_kl: object,
    generated_tokens: object,
    patch_tokens: object,
    exact_gradient: object,
) -> bytes:
    """Seal one exact Qwen replay-gradient target before predictor fitting."""

    commit = _source_commit(source_commit)
    revision = _source_commit(model_revision)
    fixture_digest = _sha256_bytes(fixture_sha256, name="fixture digest").hex()
    completion_digest = _sha256_bytes(
        completion_group_sha256,
        name="completion group digest",
    ).hex()
    protocol_digest = _sha256_bytes(
        completion_protocol_sha256,
        name="completion protocol digest",
    ).hex()
    eligible_schedule_digest = _sha256_bytes(
        eligible_schedule_sha256,
        name="eligible schedule digest",
    ).hex()
    pooler_state_digest = _sha256_bytes(
        pooler_state_sha256,
        name="pooler state digest",
    ).hex()
    if type(eligible_pair_ordinal) is not int or eligible_pair_ordinal < 0:
        raise ValueError("ASG-CV gradient sample eligible pair ordinal differs")
    if type(candidate_pair_ordinal) is not int or candidate_pair_ordinal < 0:
        raise ValueError("ASG-CV gradient sample candidate pair ordinal differs")
    if (
        type(pair_ordinals) is not tuple
        or len(pair_ordinals) != 2
        or any(type(value) is not int or value < 0 for value in pair_ordinals)
        or pair_ordinals[0] == pair_ordinals[1]
    ):
        raise ValueError("ASG-CV gradient sample pair ordinals differ")
    if type(relation_sign) is not int or relation_sign not in {-1, 1}:
        raise ValueError("ASG-CV gradient sample relation sign differs")
    if type(grpo_loss) is not float or not math.isfinite(grpo_loss):
        raise ValueError("ASG-CV gradient sample GRPO loss differs")
    if type(attention_kl) is not float or not math.isfinite(attention_kl) or attention_kl < 0.0:
        raise ValueError("ASG-CV gradient sample attention KL differs")
    semantic_loss = grpo_loss + attention_kl
    if not math.isfinite(semantic_loss):
        raise ValueError("ASG-CV gradient sample semantic loss differs")
    if type(generated_tokens) is not int or generated_tokens <= 0:
        raise ValueError("ASG-CV gradient sample generated tokens differ")
    token_array = _require_float32_array(
        patch_tokens,
        name="gradient sample patch-tokens",
        dimensions=3,
    )
    gradient_array = _require_float32_array(
        exact_gradient,
        name="gradient sample exact-gradient",
        dimensions=3,
    )
    if token_array.shape != gradient_array.shape or token_array.shape[0] != 2:
        raise ValueError("ASG-CV gradient sample array relation differs")
    token_authority = _gradient_sample_array_authority(token_array, role="patch-tokens")
    gradient_authority = _gradient_sample_array_authority(
        gradient_array,
        role="exact-gradient",
    )
    payload: dict[str, object] = {
        "schema": ASGCV_GRADIENT_SAMPLE_SCHEMA,
        "claim_eligible": False,
        "source_commit": commit,
        "model_revision": revision,
        "fixture_sha256": fixture_digest,
        "completion_group_sha256": completion_digest,
        "completion_protocol_sha256": protocol_digest,
        "eligible_schedule_sha256": eligible_schedule_digest,
        "pooler_state_sha256": pooler_state_digest,
        "eligible_pair_ordinal": eligible_pair_ordinal,
        "candidate_pair_ordinal": candidate_pair_ordinal,
        "pair_ordinals": list(pair_ordinals),
        "relation_sign": relation_sign,
        "replay_branch_count": ASGCV_STRATUM_SIZE,
        "losses": {
            "grpo": grpo_loss,
            "attention_kl": attention_kl,
            "semantic": semantic_loss,
        },
        "generated_tokens": generated_tokens,
        "arrays": {
            "patch_tokens": token_authority,
            "exact_gradient": gradient_authority,
        },
    }
    payload["sample_sha256"] = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
    return _canonical_json_bytes(payload)


def _validate_gradient_sample_array_authority(value: object) -> tuple[int, int, int]:
    if type(value) is not dict or set(value) != {"dtype", "shape", "sha256"}:
        raise ValueError("ASG-CV gradient sample array schema differs")
    if value["dtype"] != "float32-le":
        raise ValueError("ASG-CV gradient sample array dtype differs")
    shape = value["shape"]
    if (
        type(shape) is not list
        or len(shape) != 3
        or any(type(size) is not int or size <= 0 for size in shape)
    ):
        raise ValueError("ASG-CV gradient sample array shape differs")
    _sha256_bytes(value["sha256"], name="gradient sample array digest")
    return shape[0], shape[1], shape[2]


def validate_gradient_sample_bytes(raw: bytes) -> dict[str, object]:
    """Validate one canonical captured-gradient receipt and its relations."""

    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("ASG-CV gradient sample is not canonical JSON") from error
    expected_keys = {
        "schema",
        "claim_eligible",
        "source_commit",
        "model_revision",
        "fixture_sha256",
        "completion_group_sha256",
        "completion_protocol_sha256",
        "eligible_schedule_sha256",
        "pooler_state_sha256",
        "eligible_pair_ordinal",
        "candidate_pair_ordinal",
        "pair_ordinals",
        "relation_sign",
        "replay_branch_count",
        "losses",
        "generated_tokens",
        "arrays",
        "sample_sha256",
    }
    if (
        type(value) is not dict
        or set(value) != expected_keys
        or _canonical_json_bytes(value) != raw
        or value["schema"] != ASGCV_GRADIENT_SAMPLE_SCHEMA
        or value["claim_eligible"] is not False
    ):
        raise ValueError("ASG-CV gradient sample authority differs")
    _source_commit(value["source_commit"])
    _source_commit(value["model_revision"])
    _sha256_bytes(value["fixture_sha256"], name="fixture digest")
    _sha256_bytes(value["completion_group_sha256"], name="completion group digest")
    _sha256_bytes(value["completion_protocol_sha256"], name="completion protocol digest")
    _sha256_bytes(value["eligible_schedule_sha256"], name="eligible schedule digest")
    _sha256_bytes(value["pooler_state_sha256"], name="pooler state digest")
    if type(value["eligible_pair_ordinal"]) is not int or value["eligible_pair_ordinal"] < 0:
        raise ValueError("ASG-CV gradient sample eligible pair ordinal differs")
    if type(value["candidate_pair_ordinal"]) is not int or value["candidate_pair_ordinal"] < 0:
        raise ValueError("ASG-CV gradient sample candidate pair ordinal differs")
    pair_ordinals = value["pair_ordinals"]
    if (
        type(pair_ordinals) is not list
        or len(pair_ordinals) != 2
        or any(type(ordinal) is not int or ordinal < 0 for ordinal in pair_ordinals)
        or pair_ordinals[0] == pair_ordinals[1]
    ):
        raise ValueError("ASG-CV gradient sample pair ordinals differ")
    if type(value["relation_sign"]) is not int or value["relation_sign"] not in {-1, 1}:
        raise ValueError("ASG-CV gradient sample relation sign differs")
    if (
        value["replay_branch_count"] != ASGCV_STRATUM_SIZE
        or type(value["replay_branch_count"]) is not int
    ):
        raise ValueError("ASG-CV gradient sample replay count differs")
    losses = value["losses"]
    if type(losses) is not dict or set(losses) != {"grpo", "attention_kl", "semantic"}:
        raise ValueError("ASG-CV gradient sample loss schema differs")
    grpo_loss = losses["grpo"]
    attention_kl = losses["attention_kl"]
    semantic_loss = losses["semantic"]
    if type(grpo_loss) is not float or not math.isfinite(grpo_loss):
        raise ValueError("ASG-CV gradient sample GRPO loss differs")
    if type(attention_kl) is not float or not math.isfinite(attention_kl) or attention_kl < 0.0:
        raise ValueError("ASG-CV gradient sample attention KL differs")
    if (
        type(semantic_loss) is not float
        or not math.isfinite(semantic_loss)
        or semantic_loss != grpo_loss + attention_kl
    ):
        raise ValueError("ASG-CV gradient sample semantic loss differs")
    if type(value["generated_tokens"]) is not int or value["generated_tokens"] <= 0:
        raise ValueError("ASG-CV gradient sample generated tokens differ")
    arrays = value["arrays"]
    if type(arrays) is not dict or set(arrays) != {"patch_tokens", "exact_gradient"}:
        raise ValueError("ASG-CV gradient sample array schema differs")
    token_shape = _validate_gradient_sample_array_authority(arrays["patch_tokens"])
    gradient_shape = _validate_gradient_sample_array_authority(arrays["exact_gradient"])
    if token_shape != gradient_shape or token_shape[0] != 2:
        raise ValueError("ASG-CV gradient sample array relation differs")
    digest = _sha256_bytes(value["sample_sha256"], name="gradient sample digest").hex()
    unsigned = dict(value)
    del unsigned["sample_sha256"]
    if hashlib.sha256(_canonical_json_bytes(unsigned)).hexdigest() != digest:
        raise ValueError("ASG-CV gradient sample digest differs")
    return value


def validate_gradient_sample_inputs(
    raw: bytes,
    *,
    patch_tokens: object,
    exact_gradient: object,
) -> dict[str, object]:
    """Reopen both dense sample arrays and require byte-identical authority."""

    value = validate_gradient_sample_bytes(raw)
    pair_ordinals = value["pair_ordinals"]
    if type(pair_ordinals) is not list:
        raise ValueError("ASG-CV gradient sample pair ordinals differ")
    losses = value["losses"]
    if type(losses) is not dict:
        raise ValueError("ASG-CV gradient sample loss schema differs")
    rebuilt = canonical_gradient_sample_bytes(
        source_commit=value["source_commit"],
        model_revision=value["model_revision"],
        fixture_sha256=value["fixture_sha256"],
        completion_group_sha256=value["completion_group_sha256"],
        completion_protocol_sha256=value["completion_protocol_sha256"],
        eligible_schedule_sha256=value["eligible_schedule_sha256"],
        pooler_state_sha256=value["pooler_state_sha256"],
        eligible_pair_ordinal=value["eligible_pair_ordinal"],
        candidate_pair_ordinal=value["candidate_pair_ordinal"],
        pair_ordinals=tuple(pair_ordinals),
        relation_sign=value["relation_sign"],
        grpo_loss=losses["grpo"],
        attention_kl=losses["attention_kl"],
        generated_tokens=value["generated_tokens"],
        patch_tokens=patch_tokens,
        exact_gradient=exact_gradient,
    )
    if rebuilt != raw:
        raise ValueError("ASG-CV gradient sample reopened inputs differ")
    return value


def validate_gradient_sample_context(
    raw: bytes,
    *,
    eligible_schedule: AsgcvEligibleSchedule,
    candidate_schedule: AsgcvPairSchedule,
    completion_groups: tuple[AsgcvCompletionGroup, ...],
) -> dict[str, object]:
    """Cross-bind one sample to its eligible row, candidate pair, and completions."""

    value = validate_gradient_sample_bytes(raw)
    if (
        type(eligible_schedule) is not AsgcvEligibleSchedule
        or type(candidate_schedule) is not AsgcvPairSchedule
        or type(completion_groups) is not tuple
    ):
        raise ValueError("ASG-CV gradient sample context differs")
    eligible_schedule.validated()
    candidate_schedule.validated()
    if (
        eligible_schedule.candidate_schedule_sha256 != candidate_schedule.sha256()
        or value["eligible_schedule_sha256"] != eligible_schedule.sha256()
        or len(completion_groups) != candidate_schedule.pair_count
    ):
        raise ValueError("ASG-CV gradient sample context differs")
    eligible_index = value["eligible_pair_ordinal"]
    candidate_index = value["candidate_pair_ordinal"]
    if (
        type(eligible_index) is not int
        or not 0 <= eligible_index < eligible_schedule.target_pair_count
        or type(candidate_index) is not int
        or candidate_index != eligible_schedule.candidate_ordinals[eligible_index]
    ):
        raise ValueError("ASG-CV gradient sample context differs")
    pair = candidate_schedule.pairs[candidate_index]
    group = completion_groups[candidate_index]
    if type(group) is not AsgcvCompletionGroup:
        raise ValueError("ASG-CV gradient sample context differs")
    group.validated()
    if (
        group.nonzero_reward_variance is not True
        or group.candidate_pair_ordinal != candidate_index
        or group.expected_relation_sign != pair.relation_sign
        or value["completion_group_sha256"] != group.sha256()
        or value["completion_protocol_sha256"] != group.protocol_sha256
        or value["pair_ordinals"] != [pair.left_index, pair.right_index]
        or value["relation_sign"] != pair.relation_sign
    ):
        raise ValueError("ASG-CV gradient sample context differs")
    return value


def validate_gradient_sample_bundle(
    raw: bytes,
    *,
    patch_tokens: object,
    exact_gradient: object,
    protocol: AsgcvCompletionProtocol,
    rollout_authority: AsgcvRolloutAuthority,
    eligible_schedule: AsgcvEligibleSchedule,
    candidate_schedule: AsgcvPairSchedule,
    completion_groups: tuple[AsgcvCompletionGroup, ...],
    example_ids: tuple[str, ...],
    labels: tuple[int, ...],
) -> dict[str, object]:
    """Reopen one gradient sample and every semantic object it depends on."""

    validate_asgcv_protocol_bundle(
        protocol,
        rollout_authority,
        candidate_schedule,
        completion_groups,
        eligible_schedule,
        example_ids=example_ids,
        labels=labels,
    )
    value = validate_gradient_sample_inputs(
        raw,
        patch_tokens=patch_tokens,
        exact_gradient=exact_gradient,
    )
    if (
        validate_gradient_sample_context(
            raw,
            eligible_schedule=eligible_schedule,
            candidate_schedule=candidate_schedule,
            completion_groups=completion_groups,
        )
        != value
    ):
        raise ValueError("ASG-CV gradient sample bundle differs")
    return value


def _array_authority(value: object, *, role: str, dimensions: int) -> dict[str, object]:
    array = _require_float64_array(value, name=f"E0 {role}", dimensions=dimensions)
    role_bytes = role.encode("ascii")
    shape = tuple(int(size) for size in array.shape)
    frame = bytearray(ASGCV_E0_ARRAY_DOMAIN)
    frame.extend(len(role_bytes).to_bytes(8, "big"))
    frame.extend(role_bytes)
    frame.extend(len(shape).to_bytes(8, "big"))
    for size in shape:
        frame.extend(size.to_bytes(8, "big"))
    frame.extend(np.ascontiguousarray(array, dtype="<f8").tobytes(order="C"))
    return {
        "dtype": "float64-le",
        "shape": list(shape),
        "sha256": hashlib.sha256(frame).hexdigest(),
    }


def _canonical_exact_block_order(
    first_exact: object,
    second_exact: object,
) -> tuple[Float64Array, Float64Array, bool]:
    first = _require_float64_array(
        first_exact,
        name="E0 first exact gradient batch",
        dimensions=5,
    )
    second = _require_float64_array(
        second_exact,
        name="E0 second exact gradient batch",
        dimensions=5,
    )
    if first.shape != second.shape:
        raise ValueError("ASG-CV E0 exact block shape differs")
    first_digest = hashlib.sha256(np.ascontiguousarray(first, dtype="<f8").tobytes()).digest()
    second_digest = hashlib.sha256(np.ascontiguousarray(second, dtype="<f8").tobytes()).digest()
    if second_digest < first_digest:
        return second, first, True
    return first, second, False


def canonical_e0_result_bytes(
    *,
    source_commit: object,
    dataset_manifest_sha256: object,
    partition_manifest_sha256: object,
    predictor_state_sha256: object,
    selection_seed_sha256: object,
    first_exact: object,
    second_exact: object,
    predicted: object,
    srht_authority: AsgcvSrhtAuthority,
    peak_cuda_reserved_bytes: int,
    exact_semantic_wall_ns: int,
    asgcv_semantic_wall_ns: int,
) -> bytes:
    """Build one canonical claim-ineligible ASG-CV E0 result."""

    commit = _source_commit(source_commit)
    dataset_digest = _sha256_bytes(
        dataset_manifest_sha256,
        name="dataset manifest digest",
    ).hex()
    partition_digest = _sha256_bytes(
        partition_manifest_sha256,
        name="partition manifest digest",
    ).hex()
    predictor_digest = _sha256_bytes(
        predictor_state_sha256,
        name="predictor state digest",
    ).hex()
    ordered_first, ordered_second, _ = _canonical_exact_block_order(first_exact, second_exact)
    metrics = evaluate_e0(
        ordered_first,
        predicted,
        srht_authority,
        second_exact=ordered_second,
        selection_seed_sha256=selection_seed_sha256,
        peak_cuda_reserved_bytes=peak_cuda_reserved_bytes,
        exact_semantic_wall_ns=exact_semantic_wall_ns,
        asgcv_semantic_wall_ns=asgcv_semantic_wall_ns,
    )
    selection_seed = _selection_seed_bytes(selection_seed_sha256).hex()
    selection_digest = selection_schedule_sha256(
        selection_seed,
        optimizer_steps=1,
        strata_per_step=metrics.pair_count // ASGCV_STRATUM_SIZE,
    )
    payload: dict[str, object] = {
        "schema": ASGCV_E0_RESULT_SCHEMA,
        "claim_eligible": False,
        "source_commit": commit,
        "dataset_manifest_sha256": dataset_digest,
        "partition_manifest_sha256": partition_digest,
        "predictor_state_sha256": predictor_digest,
        "selection_seed_sha256": selection_seed,
        "selection_schedule_sha256": selection_digest,
        "srht_authority": srht_authority.to_mapping(),
        "arrays": {
            "first_exact_gradients": _array_authority(
                ordered_first,
                role="first-exact-gradients",
                dimensions=5,
            ),
            "second_exact_gradients": _array_authority(
                ordered_second,
                role="second-exact-gradients",
                dimensions=5,
            ),
            "predicted_gradients": _array_authority(
                predicted,
                role="predicted-gradients",
                dimensions=5,
            ),
        },
        "semantic_wall_ns": {
            "exact": exact_semantic_wall_ns,
            "asgcv": asgcv_semantic_wall_ns,
        },
        "metrics": metrics.to_mapping(),
    }
    payload["result_sha256"] = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
    return _canonical_json_bytes(payload)


def _validate_array_authority(value: object, *, dimensions: int) -> tuple[int, ...]:
    if type(value) is not dict or set(value) != {"dtype", "shape", "sha256"}:
        raise ValueError("ASG-CV E0 array authority schema differs")
    if type(value["dtype"]) is not str or value["dtype"] != "float64-le":
        raise ValueError("ASG-CV E0 array dtype differs")
    shape = value["shape"]
    if (
        type(shape) is not list
        or len(shape) != dimensions
        or any(type(size) is not int or size <= 0 for size in shape)
    ):
        raise ValueError("ASG-CV E0 array shape differs")
    _sha256_bytes(value["sha256"], name="E0 array digest")
    return tuple(shape)


def validate_e0_result_bytes(raw: bytes) -> dict[str, object]:
    """Validate canonical E0 receipt authority and all derivable relations."""

    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("ASG-CV E0 result is not canonical JSON") from error
    expected_keys = {
        "schema",
        "claim_eligible",
        "source_commit",
        "dataset_manifest_sha256",
        "partition_manifest_sha256",
        "predictor_state_sha256",
        "selection_seed_sha256",
        "selection_schedule_sha256",
        "srht_authority",
        "arrays",
        "semantic_wall_ns",
        "metrics",
        "result_sha256",
    }
    if (
        type(value) is not dict
        or set(value) != expected_keys
        or _canonical_json_bytes(value) != raw
    ):
        raise ValueError("ASG-CV E0 result is not canonical JSON")
    if (
        type(value["schema"]) is not str
        or value["schema"] != ASGCV_E0_RESULT_SCHEMA
        or value["claim_eligible"] is not False
    ):
        raise ValueError("ASG-CV E0 result authority differs")
    _source_commit(value["source_commit"])
    for name in (
        "dataset_manifest_sha256",
        "partition_manifest_sha256",
        "predictor_state_sha256",
        "selection_seed_sha256",
        "selection_schedule_sha256",
    ):
        _sha256_bytes(value[name], name=f"E0 {name}")

    arrays = value["arrays"]
    array_roles = {
        "first_exact_gradients": 5,
        "second_exact_gradients": 5,
        "predicted_gradients": 5,
    }
    if type(arrays) is not dict or set(arrays) != set(array_roles):
        raise ValueError("ASG-CV E0 array authority schema differs")
    shapes = {
        role: _validate_array_authority(arrays[role], dimensions=dimensions)
        for role, dimensions in array_roles.items()
    }
    exact_shape = shapes["first_exact_gradients"]
    srht = AsgcvSrhtAuthority.from_mapping(value["srht_authority"])
    if (
        shapes["second_exact_gradients"] != exact_shape
        or shapes["predicted_gradients"] != exact_shape
        or exact_shape[1] != ASGCV_STRATUM_SIZE
        or exact_shape[2] != 2
        or srht.input_dimensions != exact_shape[-1]
    ):
        raise ValueError("ASG-CV E0 array relation differs")

    wall = value["semantic_wall_ns"]
    if type(wall) is not dict or set(wall) != {"exact", "asgcv"}:
        raise ValueError("ASG-CV E0 wall-time schema differs")
    exact_wall = wall["exact"]
    asgcv_wall = wall["asgcv"]
    if type(exact_wall) is not int or exact_wall <= 0:
        raise ValueError("ASG-CV E0 exact semantic wall time differs")
    if type(asgcv_wall) is not int or asgcv_wall <= 0:
        raise ValueError("ASG-CV E0 semantic wall time differs")

    metrics = AsgcvE0Metrics.from_mapping(value["metrics"])
    if metrics.pair_count != exact_shape[0] * exact_shape[1]:
        raise ValueError("ASG-CV E0 pair count relation differs")
    expected_selection_schedule = selection_schedule_sha256(
        value["selection_seed_sha256"],
        optimizer_steps=1,
        strata_per_step=exact_shape[0],
    )
    if value["selection_schedule_sha256"] != expected_selection_schedule:
        raise ValueError("ASG-CV E0 selection schedule relation differs")
    expected_wall_ratio = (asgcv_wall * 1_000_000 + exact_wall - 1) // exact_wall
    if metrics.semantic_wall_ratio_ppm != expected_wall_ratio:
        raise ValueError("ASG-CV E0 wall-time relation differs")

    result_digest = _sha256_bytes(value["result_sha256"], name="E0 result digest").hex()
    unsigned = dict(value)
    del unsigned["result_sha256"]
    if hashlib.sha256(_canonical_json_bytes(unsigned)).hexdigest() != result_digest:
        raise ValueError("ASG-CV E0 result digest differs")
    return value


def validate_e0_result_inputs(
    raw: bytes,
    *,
    first_exact: object,
    second_exact: object,
    predicted: object,
) -> dict[str, object]:
    """Reopen every E0 numeric input and require the same canonical result."""

    value = validate_e0_result_bytes(raw)
    wall = value["semantic_wall_ns"]
    if type(wall) is not dict:
        raise ValueError("ASG-CV E0 wall-time schema differs")
    rebuilt = canonical_e0_result_bytes(
        source_commit=value["source_commit"],
        dataset_manifest_sha256=value["dataset_manifest_sha256"],
        partition_manifest_sha256=value["partition_manifest_sha256"],
        predictor_state_sha256=value["predictor_state_sha256"],
        selection_seed_sha256=value["selection_seed_sha256"],
        first_exact=first_exact,
        second_exact=second_exact,
        predicted=predicted,
        srht_authority=AsgcvSrhtAuthority.from_mapping(value["srht_authority"]),
        peak_cuda_reserved_bytes=AsgcvE0Metrics.from_mapping(
            value["metrics"]
        ).peak_cuda_reserved_bytes,
        exact_semantic_wall_ns=wall["exact"],
        asgcv_semantic_wall_ns=wall["asgcv"],
    )
    if rebuilt != raw:
        raise ValueError("ASG-CV E0 reopened inputs differ")
    return value


def validate_e0_result_context(
    raw: bytes,
    *,
    partition_authority: object,
    predictor_state_sha256: object,
) -> dict[str, object]:
    """Cross-bind an E0 result to its sealed partition and predictor state."""

    value = validate_e0_result_bytes(raw)
    if type(partition_authority) is not AsgcvPartitionAuthority:
        raise ValueError("ASG-CV E0 partition authority differs")
    partition_authority.validated()
    if value["partition_manifest_sha256"] != partition_authority.sha256():
        raise ValueError("ASG-CV E0 partition binding differs")
    predictor_digest = _sha256_bytes(
        predictor_state_sha256,
        name="predictor state digest",
    ).hex()
    if value["predictor_state_sha256"] != predictor_digest:
        raise ValueError("ASG-CV E0 predictor state binding differs")
    return value


def low_rank_gradient_field(
    patch_factors: object,
    channel_factors: object,
    *,
    predictor_rank: int,
) -> Float64Array:
    """Reconstruct the scalar-reference `P x D` low-rank gradient field."""

    patch_array = _require_float64_array(
        patch_factors,
        name="patch factors",
        dimensions=2,
    )
    channel_array = _require_float64_array(
        channel_factors,
        name="channel factors",
        dimensions=2,
    )
    if type(predictor_rank) is not int or predictor_rank <= 0:
        raise ValueError("ASG-CV predictor rank differs")
    if patch_array.shape[1] != predictor_rank or channel_array.shape[1] != predictor_rank:
        raise ValueError("ASG-CV low-rank factor shape differs")
    result = patch_array @ channel_array.T
    if not bool(np.isfinite(result).all()):
        raise ValueError("ASG-CV low-rank field is not finite")
    return np.asarray(result, dtype=np.float64)
