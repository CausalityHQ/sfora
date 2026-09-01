"""Deterministic finite-sample bias diagnostics for ASG-CV E0."""

from __future__ import annotations

import hashlib
import json
import math

import numpy as np
from numpy.typing import NDArray

from sfora.asgcv import (
    ASGCV_E0_MINIMUM_PAIRS,
    ASGCV_STRATUM_SIZE,
    AsgcvSrhtAuthority,
    select_stratum_index,
    srht_gradient_sketch,
    validate_e0_result_inputs,
)

ASGCV_MEAN_NULL_DOMAIN = b"sfora-asgcv-e0-mean-null-v1\0"
ASGCV_MEAN_RANDOMIZATION_DRAWS = 10_000
ASGCV_MAX_BIAS_STRATA = 512
ASGCV_RANDOMIZATION_BLOCK_DRAWS = 256
ASGCV_PATCH_SIGN_DOMAIN = b"sfora-asgcv-e0-patch-sign-v1\0"
ASGCV_NULL_SEED_DOMAIN = b"sfora-asgcv-e0-mean-null-seed-v1\0"
ASGCV_SELECTION_AUDIT_SCHEMA = "sfora-asgcv-e0-selection-audit-v1"


def _canonical_json_bytes(value: dict[str, object]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")


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
                ASGCV_MEAN_NULL_DOMAIN + seed + draw.to_bytes(8, "big") + block.to_bytes(8, "big")
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
    patch_signs = np.empty(residual.shape[2:4], dtype=np.float64)
    seed = bytes.fromhex(srht_authority.seed_sha256)
    for image_ordinal in range(residual.shape[2]):
        for patch_ordinal in range(residual.shape[3]):
            digest = hashlib.sha256(
                ASGCV_PATCH_SIGN_DOMAIN
                + seed
                + image_ordinal.to_bytes(8, "big")
                + patch_ordinal.to_bytes(8, "big")
            ).digest()
            patch_signs[image_ordinal, patch_ordinal] = 1.0 if digest[0] & 1 else -1.0
    reduced = np.sum(
        projected * patch_signs[None, None, :, :, None],
        axis=(2, 3),
        dtype=np.float64,
    ) / math.sqrt(float(residual.shape[2] * residual.shape[3]))
    return np.asarray(
        reduced - np.mean(reduced, axis=1, keepdims=True, dtype=np.float64),
        dtype=np.float64,
    )


def _projected_selection_independence_p_value_ppm(
    exact: object,
    predicted: object,
    srht_authority: object,
    *,
    selection_seed_sha256: object,
    null_seed_sha256: object,
) -> int:
    """Audit whether the realized selection stream aligns with projected errors."""

    potentials = projected_mean_error_potentials(exact, predicted, srht_authority)
    if potentials.shape[0] < ASGCV_E0_MINIMUM_PAIRS // ASGCV_STRATUM_SIZE:
        raise ValueError("ASG-CV selection-independence sample count differs")
    observed_indices = np.fromiter(
        (
            select_stratum_index(
                selection_seed_sha256,
                optimizer_step=0,
                stratum_ordinal=ordinal,
            )
            for ordinal in range(potentials.shape[0])
        ),
        dtype=np.uint8,
        count=potentials.shape[0],
    )
    null_indices = randomization_selection_indices(
        null_seed_sha256,
        draw_count=ASGCV_MEAN_RANDOMIZATION_DRAWS,
        stratum_count=potentials.shape[0],
    )
    return randomization_mean_p_value_ppm(
        potentials,
        observed_indices,
        null_indices,
    )


def _projected_selection_bias_z_ppm(
    exact: object,
    predicted: object,
    srht_authority: object,
    *,
    selection_seed_sha256: object,
) -> int:
    """Return the diagonal-free selected-error U statistic in z-score ppm."""

    potentials = projected_mean_error_potentials(exact, predicted, srht_authority)
    stratum_count = potentials.shape[0]
    if stratum_count < ASGCV_E0_MINIMUM_PAIRS // ASGCV_STRATUM_SIZE:
        raise ValueError("ASG-CV projected selection-bias sample count differs")
    observed_indices = np.fromiter(
        (
            select_stratum_index(
                selection_seed_sha256,
                optimizer_step=0,
                stratum_ordinal=ordinal,
            )
            for ordinal in range(stratum_count)
        ),
        dtype=np.uint8,
        count=stratum_count,
    )
    selected = potentials[np.arange(stratum_count), observed_indices]
    selected_sum = np.sum(selected, axis=0, dtype=np.float64)
    observed_u = float(
        np.dot(selected_sum, selected_sum)
        - np.einsum("ij,ij->", selected, selected, dtype=np.float64)
    )
    covariance = (
        np.einsum(
            "sip,siq->spq",
            potentials,
            potentials,
            dtype=np.float64,
        )
        / 8.0
    )
    accumulated = np.zeros_like(covariance[0])
    cross_variance = 0.0
    for stratum_covariance in covariance:
        cross_variance += float(np.sum(accumulated * stratum_covariance, dtype=np.float64))
        accumulated += stratum_covariance
    variance = 4.0 * cross_variance
    if not math.isfinite(observed_u) or not math.isfinite(variance) or variance < 0.0:
        raise ValueError("ASG-CV projected selection-bias statistic differs")
    if variance == 0.0:
        if observed_u != 0.0:
            raise ValueError("ASG-CV projected selection-bias variance differs")
        return 0
    z_ppm = int(round(abs(observed_u) / math.sqrt(variance) * 1_000_000))
    if not 0 <= z_ppm < 2**63:
        raise ValueError("ASG-CV projected selection-bias z-score differs")
    return z_ppm


def canonical_e0_selection_audit_bytes(
    e0_result: bytes,
    *,
    exact: object,
    predicted: object,
) -> bytes:
    """Bind the auxiliary selection-independence audit to one complete E0 result."""

    result = validate_e0_result_inputs(e0_result, exact=exact, predicted=predicted)
    metrics = result["metrics"]
    arrays = result["arrays"]
    if type(metrics) is not dict or type(arrays) is not dict:
        raise ValueError("ASG-CV selection audit E0 authority differs")
    pair_count = metrics["pair_count"]
    exact_shape = arrays["exact_gradients"]["shape"]
    if (
        type(pair_count) is not int
        or type(exact_shape) is not list
        or len(exact_shape) != 5
        or pair_count != exact_shape[0] * ASGCV_STRATUM_SIZE
        or exact_shape[0] < ASGCV_E0_MINIMUM_PAIRS // ASGCV_STRATUM_SIZE
    ):
        raise ValueError("ASG-CV selection audit sample authority differs")
    srht = AsgcvSrhtAuthority.from_mapping(result["srht_authority"])
    result_digest = result["result_sha256"]
    if type(result_digest) is not str:
        raise ValueError("ASG-CV selection audit result digest differs")
    null_seed = hashlib.sha256(ASGCV_NULL_SEED_DOMAIN + bytes.fromhex(result_digest)).hexdigest()
    selection_seed = result["selection_seed_sha256"]
    payload: dict[str, object] = {
        "schema": ASGCV_SELECTION_AUDIT_SCHEMA,
        "claim_eligible": False,
        "e0_result_sha256": result_digest,
        "selection_seed_sha256": selection_seed,
        "null_seed_sha256": null_seed,
        "randomization_draws": ASGCV_MEAN_RANDOMIZATION_DRAWS,
        "selection_independence_p_value_ppm": (
            _projected_selection_independence_p_value_ppm(
                exact,
                predicted,
                srht,
                selection_seed_sha256=selection_seed,
                null_seed_sha256=null_seed,
            )
        ),
        "selection_independence_z_ppm": _projected_selection_bias_z_ppm(
            exact,
            predicted,
            srht,
            selection_seed_sha256=selection_seed,
        ),
    }
    payload["audit_sha256"] = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
    return _canonical_json_bytes(payload)


def validate_e0_selection_audit_bytes(raw: bytes) -> dict[str, object]:
    """Validate one canonical, claim-ineligible selection-independence audit."""

    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("ASG-CV selection audit is not canonical JSON") from error
    expected = {
        "schema",
        "claim_eligible",
        "e0_result_sha256",
        "selection_seed_sha256",
        "null_seed_sha256",
        "randomization_draws",
        "selection_independence_p_value_ppm",
        "selection_independence_z_ppm",
        "audit_sha256",
    }
    if (
        type(value) is not dict
        or set(value) != expected
        or _canonical_json_bytes(value) != raw
        or value["schema"] != ASGCV_SELECTION_AUDIT_SCHEMA
        or value["claim_eligible"] is not False
        or type(value["randomization_draws"]) is not int
        or value["randomization_draws"] != ASGCV_MEAN_RANDOMIZATION_DRAWS
        or type(value["selection_independence_p_value_ppm"]) is not int
        or not 0 <= value["selection_independence_p_value_ppm"] <= 1_000_000
        or type(value["selection_independence_z_ppm"]) is not int
        or value["selection_independence_z_ppm"] < 0
    ):
        raise ValueError("ASG-CV selection audit authority differs")
    for name in (
        "e0_result_sha256",
        "selection_seed_sha256",
        "null_seed_sha256",
        "audit_sha256",
    ):
        _seed_bytes(value[name])
    unsigned = dict(value)
    digest = unsigned.pop("audit_sha256")
    if hashlib.sha256(_canonical_json_bytes(unsigned)).hexdigest() != digest:
        raise ValueError("ASG-CV selection audit digest differs")
    return value
