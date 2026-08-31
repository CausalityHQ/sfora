"""Scalar authority for Amortized Semantic Gradient Control Variates."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Float64Array = NDArray[np.float64]

ASGCV_SCHEMA = "sfora-asgcv-authority-v1"
ASGCV_STRATUM_SIZE = 8
ASGCV_PREDICTOR_RANK = 16
ASGCV_SELECTION_POLICY = "one-uniform-index-per-eight-pair-stratum-v1"
ASGCV_E0_SCHEMA = "sfora-asgcv-e0-metrics-v1"
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
    projection: object,
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
    projection_array = _require_float64_array(
        projection,
        name="E0 projection",
        dimensions=2,
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
    if projection_array.shape[1] != exact_array.shape[-1]:
        raise ValueError("ASG-CV E0 projection shape differs")
    if bool((np.linalg.norm(projection_array, axis=1) <= 0.0).any()):
        raise ValueError("ASG-CV E0 projection row differs")
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

    projected_exact = np.einsum(
        "sijd,kd->sijk",
        exact_array,
        projection_array,
        dtype=np.float64,
        optimize=False,
    )
    projected_predicted = np.einsum(
        "sijd,kd->sijk",
        predicted_array,
        projection_array,
        dtype=np.float64,
        optimize=False,
    )
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
