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
