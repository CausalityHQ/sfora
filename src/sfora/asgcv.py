"""Scalar authority for Amortized Semantic Gradient Control Variates."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Float64Array = NDArray[np.float64]
Float32Array = NDArray[np.float32]

ASGCV_SCHEMA = "sfora-asgcv-authority-v1"
ASGCV_STRATUM_SIZE = 8
ASGCV_PREDICTOR_RANK = 16
ASGCV_SELECTION_POLICY = "one-uniform-index-per-eight-pair-stratum-v1"
ASGCV_E0_SCHEMA = "sfora-asgcv-e0-metrics-v1"
ASGCV_E0_RESULT_SCHEMA = "sfora-asgcv-e0-result-v1"
ASGCV_E0_ARRAY_DOMAIN = b"sfora-asgcv-e0-array-v1\0"
ASGCV_GRADIENT_SAMPLE_SCHEMA = "sfora-asgcv-gradient-sample-v1"
ASGCV_GRADIENT_SAMPLE_ARRAY_DOMAIN = b"sfora-asgcv-gradient-sample-array-v1\0"
ASGCV_DENSE_COSINE_GATE_PPM = 850_000
ASGCV_PROJECTED_COSINE_GATE_PPM = 900_000
ASGCV_PATCH_SPEARMAN_GATE_PPM = 800_000
ASGCV_RESIDUAL_ENERGY_GATE_PPM = 350_000
ASGCV_VARIANCE_RATIO_GATE_PPM = 600_000
ASGCV_PRECLIP_P99_RATIO_GATE_PPM = 2_000_000
ASGCV_CLIP_RATE_DELTA_GATE_PPM = 50_000
ASGCV_SEMANTIC_WALL_RATIO_GATE_PPM = 350_000
ASGCV_GLOBAL_CLIP_NORM = 1.0
ASGCV_SELECTION_DOMAIN = b"sfora-asgcv-selection-v1\0"
ASGCV_SCHEDULE_DOMAIN = b"sfora-asgcv-schedule-v1\0"
ASGCV_MAX_SCHEDULE_SELECTIONS = 10_000_000
ASGCV_SRHT_SCHEMA = "sfora-asgcv-srht-authority-v1"
ASGCV_SRHT_SIGN_DOMAIN = b"sfora-asgcv-srht-sign-v1\0"
ASGCV_SRHT_ROW_DOMAIN = b"sfora-asgcv-srht-row-v1\0"
ASGCV_SRHT_NORMALIZATION = "orthonormal-hadamard-times-sqrt-padded-over-output-v1"


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
        dimensions=3,
    )
    predicted_array = _require_float64_array(
        predicted,
        name="predicted gradient stratum",
        dimensions=3,
    )
    if exact_array.shape != predicted_array.shape or exact_array.shape[0] != ASGCV_STRATUM_SIZE:
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
        dimensions=3,
    )
    exact_array = _require_float64_array(
        exact_selected,
        name="selected exact gradient",
        dimensions=2,
    )
    if predicted_array.shape[0] != ASGCV_STRATUM_SIZE:
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
    preclip_p99_ratio_ppm: int
    exact_clip_rate_ppm: int
    asgcv_clip_rate_ppm: int
    clip_rate_delta_ppm: int
    semantic_wall_ratio_ppm: int
    passed: bool

    def validated(self) -> AsgcvE0Metrics:
        if (
            type(self.pair_count) is not int
            or self.pair_count <= 0
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
        expected_pass = (
            self.dense_gradient_cosine_ppm >= ASGCV_DENSE_COSINE_GATE_PPM
            and self.projected_gradient_cosine_ppm >= ASGCV_PROJECTED_COSINE_GATE_PPM
            and self.patch_salience_spearman_ppm >= ASGCV_PATCH_SPEARMAN_GATE_PPM
            and self.normalized_residual_energy_ppm <= ASGCV_RESIDUAL_ENERGY_GATE_PPM
            and self.selection_variance_ratio_ppm <= ASGCV_VARIANCE_RATIO_GATE_PPM
            and self.preclip_p99_ratio_ppm <= ASGCV_PRECLIP_P99_RATIO_GATE_PPM
            and self.clip_rate_delta_ppm <= ASGCV_CLIP_RATE_DELTA_GATE_PPM
            and self.semantic_wall_ratio_ppm <= ASGCV_SEMANTIC_WALL_RATIO_GATE_PPM
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
            "preclip_p99_ratio_ppm": self.preclip_p99_ratio_ppm,
            "exact_clip_rate_ppm": self.exact_clip_rate_ppm,
            "asgcv_clip_rate_ppm": self.asgcv_clip_rate_ppm,
            "clip_rate_delta_ppm": self.clip_rate_delta_ppm,
            "semantic_wall_ratio_ppm": self.semantic_wall_ratio_ppm,
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
            "preclip_p99_ratio_ppm",
            "exact_clip_rate_ppm",
            "asgcv_clip_rate_ppm",
            "clip_rate_delta_ppm",
            "semantic_wall_ratio_ppm",
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
            preclip_p99_ratio_ppm=value["preclip_p99_ratio_ppm"],
            exact_clip_rate_ppm=value["exact_clip_rate_ppm"],
            asgcv_clip_rate_ppm=value["asgcv_clip_rate_ppm"],
            clip_rate_delta_ppm=value["clip_rate_delta_ppm"],
            semantic_wall_ratio_ppm=value["semantic_wall_ratio_ppm"],
            passed=value["passed"],
        ).validated()


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


def evaluate_e0(
    exact: object,
    predicted: object,
    srht_authority: AsgcvSrhtAuthority,
    *,
    exact_preclip_norms: object,
    asgcv_preclip_norms: object,
    exact_semantic_wall_ns: int,
    asgcv_semantic_wall_ns: int,
) -> AsgcvE0Metrics:
    """Recompute the registered E0 fidelity and variance evidence."""

    exact_array = _require_float64_array(
        exact,
        name="E0 exact gradient batch",
        dimensions=4,
    )
    predicted_array = _require_float64_array(
        predicted,
        name="E0 predicted gradient batch",
        dimensions=4,
    )
    exact_norms = _require_float64_array(
        exact_preclip_norms,
        name="E0 exact pre-clip norms",
        dimensions=1,
    )
    asgcv_norms = _require_float64_array(
        asgcv_preclip_norms,
        name="E0 ASG-CV pre-clip norms",
        dimensions=1,
    )
    if type(exact_semantic_wall_ns) is not int or exact_semantic_wall_ns <= 0:
        raise ValueError("ASG-CV E0 exact semantic wall time differs")
    if type(asgcv_semantic_wall_ns) is not int or asgcv_semantic_wall_ns <= 0:
        raise ValueError("ASG-CV E0 semantic wall time differs")
    semantic_wall_ratio_ppm = (
        asgcv_semantic_wall_ns * 1_000_000 + exact_semantic_wall_ns - 1
    ) // exact_semantic_wall_ns
    if exact_array.shape != predicted_array.shape or exact_array.shape[1] != ASGCV_STRATUM_SIZE:
        raise ValueError("ASG-CV E0 gradient batch shape differs")
    if type(srht_authority) is not AsgcvSrhtAuthority:
        raise ValueError("ASG-CV E0 SRHT authority differs")
    srht_authority.validated()
    if srht_authority.input_dimensions != exact_array.shape[-1]:
        raise ValueError("ASG-CV E0 SRHT shape differs")
    if (
        exact_norms.shape != asgcv_norms.shape
        or bool((exact_norms < 0.0).any())
        or bool((asgcv_norms < 0.0).any())
    ):
        raise ValueError("ASG-CV E0 pre-clip norm shape differs")

    exact_p99 = float(np.quantile(exact_norms, 0.99, method="higher"))
    asgcv_p99 = float(np.quantile(asgcv_norms, 0.99, method="higher"))
    if not np.isfinite(exact_p99) or exact_p99 <= 0.0 or not np.isfinite(asgcv_p99):
        raise ValueError("ASG-CV E0 pre-clip p99 differs")
    exact_clip_rate_ppm = int(
        round(
            float(np.count_nonzero(exact_norms > ASGCV_GLOBAL_CLIP_NORM))
            / len(exact_norms)
            * 1_000_000
        )
    )
    asgcv_clip_rate_ppm = int(
        round(
            float(np.count_nonzero(asgcv_norms > ASGCV_GLOBAL_CLIP_NORM))
            / len(asgcv_norms)
            * 1_000_000
        )
    )

    projected_exact = srht_gradient_sketch(
        exact_array.reshape(-1, exact_array.shape[-1]),
        srht_authority,
    ).reshape(*exact_array.shape[:-1], srht_authority.output_dimensions)
    projected_predicted = srht_gradient_sketch(
        predicted_array.reshape(-1, predicted_array.shape[-1]),
        srht_authority,
    ).reshape(*predicted_array.shape[:-1], srht_authority.output_dimensions)
    residual_energy = float(
        np.square(exact_array - predicted_array).sum(dtype=np.float64)
        / np.square(exact_array).sum(dtype=np.float64)
    )
    if not np.isfinite(residual_energy) or residual_energy < 0.0:
        raise ValueError("ASG-CV E0 residual energy differs")
    metrics_without_pass = {
        "pair_count": exact_array.shape[0] * exact_array.shape[1],
        "dense_gradient_cosine_ppm": _similarity_ppm(_median_cosine(exact_array, predicted_array)),
        "projected_gradient_cosine_ppm": _similarity_ppm(
            _median_cosine(projected_exact, projected_predicted)
        ),
        "patch_salience_spearman_ppm": _similarity_ppm(
            _median_patch_salience_spearman(exact_array, predicted_array)
        ),
        "normalized_residual_energy_ppm": _ratio_ppm(residual_energy),
        "selection_variance_ratio_ppm": _ratio_ppm(
            _batch_selection_variance_ratio(exact_array, predicted_array)
        ),
        "preclip_p99_ratio_ppm": _ratio_ppm(asgcv_p99 / exact_p99),
        "exact_clip_rate_ppm": exact_clip_rate_ppm,
        "asgcv_clip_rate_ppm": asgcv_clip_rate_ppm,
        "clip_rate_delta_ppm": max(0, asgcv_clip_rate_ppm - exact_clip_rate_ppm),
        "semantic_wall_ratio_ppm": semantic_wall_ratio_ppm,
    }
    passed = (
        metrics_without_pass["dense_gradient_cosine_ppm"] >= ASGCV_DENSE_COSINE_GATE_PPM
        and metrics_without_pass["projected_gradient_cosine_ppm"] >= ASGCV_PROJECTED_COSINE_GATE_PPM
        and metrics_without_pass["patch_salience_spearman_ppm"] >= ASGCV_PATCH_SPEARMAN_GATE_PPM
        and metrics_without_pass["normalized_residual_energy_ppm"] <= ASGCV_RESIDUAL_ENERGY_GATE_PPM
        and metrics_without_pass["selection_variance_ratio_ppm"] <= ASGCV_VARIANCE_RATIO_GATE_PPM
        and metrics_without_pass["preclip_p99_ratio_ppm"] <= ASGCV_PRECLIP_P99_RATIO_GATE_PPM
        and metrics_without_pass["clip_rate_delta_ppm"] <= ASGCV_CLIP_RATE_DELTA_GATE_PPM
        and metrics_without_pass["semantic_wall_ratio_ppm"] <= ASGCV_SEMANTIC_WALL_RATIO_GATE_PPM
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
    if value["replay_branch_count"] != ASGCV_STRATUM_SIZE or type(
        value["replay_branch_count"]
    ) is not int:
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


def canonical_e0_result_bytes(
    *,
    source_commit: object,
    dataset_manifest_sha256: object,
    partition_manifest_sha256: object,
    predictor_state_sha256: object,
    selection_schedule_sha256: object,
    exact: object,
    predicted: object,
    srht_authority: AsgcvSrhtAuthority,
    exact_preclip_norms: object,
    asgcv_preclip_norms: object,
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
    selection_digest = _sha256_bytes(
        selection_schedule_sha256,
        name="selection schedule digest",
    ).hex()
    metrics = evaluate_e0(
        exact,
        predicted,
        srht_authority,
        exact_preclip_norms=exact_preclip_norms,
        asgcv_preclip_norms=asgcv_preclip_norms,
        exact_semantic_wall_ns=exact_semantic_wall_ns,
        asgcv_semantic_wall_ns=asgcv_semantic_wall_ns,
    )
    payload: dict[str, object] = {
        "schema": ASGCV_E0_RESULT_SCHEMA,
        "claim_eligible": False,
        "source_commit": commit,
        "dataset_manifest_sha256": dataset_digest,
        "partition_manifest_sha256": partition_digest,
        "predictor_state_sha256": predictor_digest,
        "selection_schedule_sha256": selection_digest,
        "srht_authority": srht_authority.to_mapping(),
        "arrays": {
            "exact_gradients": _array_authority(
                exact,
                role="exact-gradients",
                dimensions=4,
            ),
            "predicted_gradients": _array_authority(
                predicted,
                role="predicted-gradients",
                dimensions=4,
            ),
            "exact_preclip_norms": _array_authority(
                exact_preclip_norms,
                role="exact-preclip-norms",
                dimensions=1,
            ),
            "asgcv_preclip_norms": _array_authority(
                asgcv_preclip_norms,
                role="asgcv-preclip-norms",
                dimensions=1,
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
        "selection_schedule_sha256",
    ):
        _sha256_bytes(value[name], name=f"E0 {name}")

    arrays = value["arrays"]
    array_roles = {
        "exact_gradients": 4,
        "predicted_gradients": 4,
        "exact_preclip_norms": 1,
        "asgcv_preclip_norms": 1,
    }
    if type(arrays) is not dict or set(arrays) != set(array_roles):
        raise ValueError("ASG-CV E0 array authority schema differs")
    shapes = {
        role: _validate_array_authority(arrays[role], dimensions=dimensions)
        for role, dimensions in array_roles.items()
    }
    exact_shape = shapes["exact_gradients"]
    srht = AsgcvSrhtAuthority.from_mapping(value["srht_authority"])
    if (
        shapes["predicted_gradients"] != exact_shape
        or exact_shape[1] != ASGCV_STRATUM_SIZE
        or srht.input_dimensions != exact_shape[-1]
        or shapes["exact_preclip_norms"] != shapes["asgcv_preclip_norms"]
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
    exact: object,
    predicted: object,
    exact_preclip_norms: object,
    asgcv_preclip_norms: object,
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
        selection_schedule_sha256=value["selection_schedule_sha256"],
        exact=exact,
        predicted=predicted,
        srht_authority=AsgcvSrhtAuthority.from_mapping(value["srht_authority"]),
        exact_preclip_norms=exact_preclip_norms,
        asgcv_preclip_norms=asgcv_preclip_norms,
        exact_semantic_wall_ns=wall["exact"],
        asgcv_semantic_wall_ns=wall["asgcv"],
    )
    if rebuilt != raw:
        raise ValueError("ASG-CV E0 reopened inputs differ")
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
