"""Deterministic finite-sample bias diagnostics for ASG-CV E0."""

from __future__ import annotations

import hashlib
import math

import numpy as np
from numpy.typing import NDArray

from sfora.asgcv import AsgcvSrhtAuthority, srht_gradient_sketch

ASGCV_MEAN_NULL_DOMAIN = b"sfora-asgcv-e0-mean-null-v1\0"
ASGCV_MEAN_RANDOMIZATION_DRAWS = 200_000
ASGCV_MAX_BIAS_STRATA = 512
ASGCV_RANDOMIZATION_BLOCK_DRAWS = 256


def _seed_bytes(value: object) -> bytes:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("ASG-CV bias seed differs")
    return bytes.fromhex(value)


def randomization_selection_indices(
    seed_sha256: object,
    *,
    draw_count: int,
    stratum_count: int,
) -> NDArray[np.uint8]:
    """Generate the registered independent one-of-eight randomization stream."""

    seed = _seed_bytes(seed_sha256)
    if (
        type(draw_count) is not int
        or not 0 < draw_count <= ASGCV_MEAN_RANDOMIZATION_DRAWS
        or type(stratum_count) is not int
        or not 0 < stratum_count <= ASGCV_MAX_BIAS_STRATA
    ):
        raise ValueError("ASG-CV bias randomization shape differs")
    blocks_per_draw = (stratum_count + 31) // 32
    output = np.empty((draw_count, stratum_count), dtype=np.uint8)
    for draw in range(draw_count):
        offset = 0
        for block in range(blocks_per_draw):
            digest = hashlib.sha256(
                ASGCV_MEAN_NULL_DOMAIN
                + seed
                + draw.to_bytes(8, "big")
                + block.to_bytes(8, "big")
            ).digest()
            width = min(len(digest), stratum_count - offset)
            output[draw, offset : offset + width] = np.frombuffer(
                digest[:width],
                dtype=np.uint8,
            ) & np.uint8(7)
            offset += width
    return output


def randomization_mean_p_value_ppm(
    potential_errors: object,
    observed_indices: object,
    null_indices: object,
) -> int:
    """Return the add-one randomization p-value for squared mean error."""

    if (
        type(potential_errors) is not np.ndarray
        or potential_errors.dtype != np.dtype(np.float64)
        or potential_errors.ndim != 3
        or potential_errors.shape[0] <= 0
        or potential_errors.shape[0] > ASGCV_MAX_BIAS_STRATA
        or potential_errors.shape[1] != 8
        or potential_errors.shape[2] <= 0
        or not bool(np.isfinite(potential_errors).all())
    ):
        raise ValueError("ASG-CV bias potential errors differ")
    stratum_count = potential_errors.shape[0]
    if (
        type(observed_indices) is not np.ndarray
        or observed_indices.dtype != np.dtype(np.uint8)
        or observed_indices.shape != (stratum_count,)
        or bool((observed_indices >= 8).any())
        or type(null_indices) is not np.ndarray
        or null_indices.dtype != np.dtype(np.uint8)
        or null_indices.ndim != 2
        or null_indices.shape[0] <= 0
        or null_indices.shape[1] != stratum_count
        or bool((null_indices >= 8).any())
    ):
        raise ValueError("ASG-CV bias selection indices differ")
    centered = potential_errors - np.mean(potential_errors, axis=1, keepdims=True)
    strata = np.arange(stratum_count)
    observed_sum = np.sum(centered[strata, observed_indices], axis=0, dtype=np.float64)
    observed_statistic = float(np.dot(observed_sum, observed_sum))
    if not math.isfinite(observed_statistic):
        raise ValueError("ASG-CV bias observed statistic differs")
    exceedances = 0
    for offset in range(0, null_indices.shape[0], ASGCV_RANDOMIZATION_BLOCK_DRAWS):
        block = null_indices[offset : offset + ASGCV_RANDOMIZATION_BLOCK_DRAWS]
        selected = centered[strata[None, :], block]
        sums = np.sum(selected, axis=1, dtype=np.float64)
        statistics = np.einsum("ij,ij->i", sums, sums, dtype=np.float64)
        exceedances += int(np.count_nonzero(statistics >= observed_statistic))
    return int((exceedances + 1) * 1_000_000 // (null_indices.shape[0] + 1))


def projected_mean_error_potentials(
    exact: object,
    predicted: object,
    srht_authority: object,
) -> NDArray[np.float64]:
    """Reduce all eight estimator errors to centered SRHT channel sketches."""

    if (
        type(exact) is not np.ndarray
        or exact.dtype != np.dtype(np.float64)
        or exact.ndim != 5
        or type(predicted) is not np.ndarray
        or predicted.dtype != np.dtype(np.float64)
        or predicted.shape != exact.shape
        or exact.shape[0] <= 0
        or exact.shape[0] > ASGCV_MAX_BIAS_STRATA
        or exact.shape[1] != 8
        or exact.shape[2] != 2
        or not bool(np.isfinite(exact).all())
        or not bool(np.isfinite(predicted).all())
        or type(srht_authority) is not AsgcvSrhtAuthority
    ):
        raise ValueError("ASG-CV projected bias fields differ")
    srht_authority.validated()
    if srht_authority.input_dimensions != exact.shape[-1]:
        raise ValueError("ASG-CV projected bias SRHT shape differs")
    residual = exact - predicted
    projected = srht_gradient_sketch(
        residual.reshape(-1, residual.shape[-1]),
        srht_authority,
    ).reshape(*residual.shape[:-1], srht_authority.output_dimensions)
    reduced = np.sum(projected, axis=(2, 3), dtype=np.float64)
    return np.asarray(
        reduced - np.mean(reduced, axis=1, keepdims=True, dtype=np.float64),
        dtype=np.float64,
    )
