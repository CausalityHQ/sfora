"""Scalar authority for Amortized Semantic Gradient Control Variates."""

from __future__ import annotations

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


def _require_exact_positive_integer(value: object, *, expected: int, name: str) -> int:
    if type(value) is not int or value != expected:
        raise ValueError(f"ASG-CV {name} differs")
    return value


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
        )
        for name in similarity_names + ratio_names:
            if type(getattr(self, name)) is not int:
                raise ValueError("ASG-CV E0 metric type differs")
        if any(not -1_000_000 <= getattr(self, name) <= 1_000_000 for name in similarity_names):
            raise ValueError("ASG-CV E0 similarity metric differs")
        if any(getattr(self, name) < 0 for name in ratio_names):
            raise ValueError("ASG-CV E0 ratio metric differs")
        expected_pass = (
            self.dense_gradient_cosine_ppm >= ASGCV_DENSE_COSINE_GATE_PPM
            and self.projected_gradient_cosine_ppm >= ASGCV_PROJECTED_COSINE_GATE_PPM
            and self.patch_salience_spearman_ppm >= ASGCV_PATCH_SPEARMAN_GATE_PPM
            and self.normalized_residual_energy_ppm <= ASGCV_RESIDUAL_ENERGY_GATE_PPM
            and self.selection_variance_ratio_ppm <= ASGCV_VARIANCE_RATIO_GATE_PPM
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
    if exact_array.shape != predicted_array.shape or exact_array.shape[1] != ASGCV_STRATUM_SIZE:
        raise ValueError("ASG-CV E0 gradient batch shape differs")
    if projection_array.shape[1] != exact_array.shape[-1]:
        raise ValueError("ASG-CV E0 projection shape differs")
    if bool((np.linalg.norm(projection_array, axis=1) <= 0.0).any()):
        raise ValueError("ASG-CV E0 projection row differs")

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
    }
    passed = (
        metrics_without_pass["dense_gradient_cosine_ppm"] >= ASGCV_DENSE_COSINE_GATE_PPM
        and metrics_without_pass["projected_gradient_cosine_ppm"] >= ASGCV_PROJECTED_COSINE_GATE_PPM
        and metrics_without_pass["patch_salience_spearman_ppm"] >= ASGCV_PATCH_SPEARMAN_GATE_PPM
        and metrics_without_pass["normalized_residual_energy_ppm"] <= ASGCV_RESIDUAL_ENERGY_GATE_PPM
        and metrics_without_pass["selection_variance_ratio_ppm"] <= ASGCV_VARIANCE_RATIO_GATE_PPM
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
