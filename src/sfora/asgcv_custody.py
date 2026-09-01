"""Canonical custody chain for ASG-CV E0 gradient evidence."""

from __future__ import annotations

import hashlib
import json
from typing import cast

import numpy as np

from sfora.asgcv import (
    ASGCV_E0_MINIMUM_PAIRS,
    _array_authority,
    _canonical_exact_block_order,
    _canonical_json_bytes,
    _gradient_sample_array_authority,
    _sha256_bytes,
    _source_commit,
    validate_e0_result_bytes,
    validate_gradient_sample_bytes,
)

ASGCV_E0_CUSTODY_SCHEMA = "sfora-asgcv-e0-custody-v4"


def _validated_sample_block(
    *,
    exact: np.ndarray,
    source_commit: object,
    sample_receipts: object,
) -> list[dict[str, object]]:
    sample_count = exact.shape[0] * exact.shape[1]
    if type(sample_receipts) is not tuple or len(sample_receipts) != sample_count:
        raise ValueError("ASG-CV E0 custody sample receipt block differs")
    flat = exact.astype(np.float32).reshape(sample_count, *exact.shape[2:])
    values: list[dict[str, object]] = []
    for ordinal, raw in enumerate(sample_receipts):
        if type(raw) is not bytes:
            raise ValueError("ASG-CV E0 custody sample receipt differs")
        value = validate_gradient_sample_bytes(raw)
        sample_arrays = value["arrays"]
        if (
            value["eligible_pair_ordinal"] != ordinal
            or value["source_commit"] != source_commit
            or type(sample_arrays) is not dict
            or sample_arrays["exact_gradient"]
            != _gradient_sample_array_authority(flat[ordinal], role="exact-gradient")
        ):
            raise ValueError("ASG-CV E0 custody sample binding differs")
        values.append(value)

    first = values[0]
    identity_names = (
        "source_commit",
        "model_revision",
        "fixture_sha256",
        "pooler_state_sha256",
        "completion_protocol_sha256",
        "eligible_schedule_sha256",
    )
    for value in values[1:]:
        if any(value[name] != first[name] for name in identity_names):
            raise ValueError("ASG-CV E0 custody sample identity differs")
    candidate_ordinals = [cast(int, value["candidate_pair_ordinal"]) for value in values]
    completion_groups = [cast(str, value["completion_group_sha256"]) for value in values]
    pair_ordinals = [
        ordinal for value in values for ordinal in cast(list[int], value["pair_ordinals"])
    ]
    if (
        candidate_ordinals != sorted(set(candidate_ordinals))
        or len(set(completion_groups)) != sample_count
        or len(set(pair_ordinals)) != 2 * sample_count
        or any(
            sorted(cast(int, value["relation_sign"]) for value in values[offset : offset + 8])
            != [-1, -1, -1, -1, 1, 1, 1, 1]
            for offset in range(0, sample_count, 8)
        )
    ):
        raise ValueError("ASG-CV E0 custody schedule differs")
    return values


def canonical_e0_custody_bytes(
    *,
    e0_result: bytes,
    first_exact: object,
    second_exact: object,
    predicted: object,
    first_sample_receipts: object,
    second_sample_receipts: object,
) -> bytes:
    """Bind both independent E0 seed blocks to captured-gradient receipts."""

    result = validate_e0_result_bytes(e0_result)
    if (
        type(first_exact) is not np.ndarray
        or first_exact.dtype != np.dtype(np.float64)
        or first_exact.ndim != 5
        or first_exact.shape[0] * first_exact.shape[1] < ASGCV_E0_MINIMUM_PAIRS
        or first_exact.shape[0] * first_exact.shape[1] % 8 != 0
        or first_exact.shape[1] != 8
        or first_exact.shape[2] != 2
        or not bool(np.isfinite(first_exact).all())
        or not np.array_equal(first_exact, first_exact.astype(np.float32).astype(np.float64))
        or type(second_exact) is not np.ndarray
        or second_exact.dtype != np.dtype(np.float64)
        or second_exact.shape != first_exact.shape
        or not bool(np.isfinite(second_exact).all())
        or not np.array_equal(second_exact, second_exact.astype(np.float32).astype(np.float64))
        or type(predicted) is not np.ndarray
        or predicted.dtype != np.dtype(np.float64)
        or predicted.shape != first_exact.shape
        or not bool(np.isfinite(predicted).all())
    ):
        raise ValueError("ASG-CV E0 custody shape differs")
    ordered_first, ordered_second, swapped = _canonical_exact_block_order(
        first_exact,
        second_exact,
    )
    if swapped:
        first_sample_receipts, second_sample_receipts = (
            second_sample_receipts,
            first_sample_receipts,
        )
    first_exact = ordered_first
    second_exact = ordered_second
    arrays = result["arrays"]
    if (
        type(arrays) is not dict
        or arrays["first_exact_gradients"]
        != _array_authority(first_exact, role="first-exact-gradients", dimensions=5)
        or arrays["second_exact_gradients"]
        != _array_authority(second_exact, role="second-exact-gradients", dimensions=5)
        or arrays["predicted_gradients"]
        != _array_authority(predicted, role="predicted-gradients", dimensions=5)
    ):
        raise ValueError("ASG-CV E0 custody exact-gradient authority differs")

    source_commit = result["source_commit"]
    predictor_state = result["predictor_state_sha256"]
    first_values = _validated_sample_block(
        exact=first_exact,
        source_commit=source_commit,
        sample_receipts=first_sample_receipts,
    )
    second_values = _validated_sample_block(
        exact=second_exact,
        source_commit=source_commit,
        sample_receipts=second_sample_receipts,
    )
    if np.array_equal(first_exact, second_exact) and tuple(
        cast(str, value["completion_group_sha256"]) for value in second_values
    ) < tuple(cast(str, value["completion_group_sha256"]) for value in first_values):
        first_values, second_values = second_values, first_values
    first = first_values[0]
    second = second_values[0]
    identity_names = (
        "source_commit",
        "model_revision",
        "fixture_sha256",
        "pooler_state_sha256",
        "completion_protocol_sha256",
        "eligible_schedule_sha256",
    )
    first_groups = {cast(str, value["completion_group_sha256"]) for value in first_values}
    second_groups = {cast(str, value["completion_group_sha256"]) for value in second_values}
    if any(second[name] != first[name] for name in identity_names) or any(
        second_value[name] != first_value[name]
        for first_value, second_value in zip(first_values, second_values, strict=True)
        for name in (
            "eligible_pair_ordinal",
            "candidate_pair_ordinal",
            "pair_ordinals",
            "relation_sign",
        )
    ):
        raise ValueError("ASG-CV E0 custody seed-block identity differs")
    if not first_groups.isdisjoint(second_groups):
        raise ValueError("ASG-CV E0 custody seed blocks are not distinct")
    if any(
        cast(dict[str, dict[str, object]], first_value["arrays"])["patch_tokens"]
        != cast(dict[str, dict[str, object]], second_value["arrays"])["patch_tokens"]
        for first_value, second_value in zip(first_values, second_values, strict=True)
    ):
        raise ValueError("ASG-CV E0 custody patch-token blocks differ")
    payload: dict[str, object] = {
        "schema": ASGCV_E0_CUSTODY_SCHEMA,
        "claim_eligible": False,
        "e0_result_sha256": result["result_sha256"],
        "source_commit": first["source_commit"],
        "model_revision": first["model_revision"],
        "fixture_sha256": first["fixture_sha256"],
        "pooler_state_sha256": first["pooler_state_sha256"],
        "predictor_state_sha256": predictor_state,
        "completion_protocol_sha256": first["completion_protocol_sha256"],
        "eligible_schedule_sha256": first["eligible_schedule_sha256"],
        "sample_count": len(first_values),
        "first_sample_sha256": [value["sample_sha256"] for value in first_values],
        "second_sample_sha256": [value["sample_sha256"] for value in second_values],
        "first_patch_token_sha256": [
            cast(dict[str, dict[str, object]], value["arrays"])["patch_tokens"]["sha256"]
            for value in first_values
        ],
        "second_patch_token_sha256": [
            cast(dict[str, dict[str, object]], value["arrays"])["patch_tokens"]["sha256"]
            for value in second_values
        ],
    }
    payload["custody_sha256"] = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
    return _canonical_json_bytes(payload)


def validate_e0_custody_bytes(raw: bytes) -> dict[str, object]:
    """Validate the canonical E0 custody receipt and its internal relations."""

    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("ASG-CV E0 custody is not canonical JSON") from error
    expected = {
        "schema",
        "claim_eligible",
        "e0_result_sha256",
        "source_commit",
        "model_revision",
        "fixture_sha256",
        "pooler_state_sha256",
        "predictor_state_sha256",
        "completion_protocol_sha256",
        "eligible_schedule_sha256",
        "sample_count",
        "first_sample_sha256",
        "second_sample_sha256",
        "first_patch_token_sha256",
        "second_patch_token_sha256",
        "custody_sha256",
    }
    if (
        type(value) is not dict
        or set(value) != expected
        or _canonical_json_bytes(value) != raw
        or value["schema"] != ASGCV_E0_CUSTODY_SCHEMA
        or value["claim_eligible"] is not False
        or type(value["sample_count"]) is not int
        or value["sample_count"] < ASGCV_E0_MINIMUM_PAIRS
        or value["sample_count"] % 8 != 0
        or any(
            type(value[name]) is not list or len(value[name]) != value["sample_count"]
            for name in (
                "first_sample_sha256",
                "second_sample_sha256",
                "first_patch_token_sha256",
                "second_patch_token_sha256",
            )
        )
    ):
        raise ValueError("ASG-CV E0 custody authority differs")
    _source_commit(value["source_commit"])
    _source_commit(value["model_revision"])
    for name in (
        "e0_result_sha256",
        "fixture_sha256",
        "pooler_state_sha256",
        "predictor_state_sha256",
        "completion_protocol_sha256",
        "eligible_schedule_sha256",
        "custody_sha256",
    ):
        _sha256_bytes(value[name], name=f"E0 custody {name}")
    for name in ("first_sample_sha256", "second_sample_sha256"):
        for digest in value[name]:
            _sha256_bytes(digest, name="E0 custody sample digest")
    for name in ("first_patch_token_sha256", "second_patch_token_sha256"):
        for digest in value[name]:
            _sha256_bytes(digest, name="E0 custody patch-token digest")
    if not set(value["first_sample_sha256"]).isdisjoint(value["second_sample_sha256"]):
        raise ValueError("ASG-CV E0 custody seed blocks are not distinct")
    unsigned = dict(value)
    digest = unsigned.pop("custody_sha256")
    if hashlib.sha256(_canonical_json_bytes(unsigned)).hexdigest() != digest:
        raise ValueError("ASG-CV E0 custody digest differs")
    return value


def validate_e0_custody_bundle(
    raw: bytes,
    *,
    e0_result: bytes,
    first_exact: object,
    second_exact: object,
    predicted: object,
    first_sample_receipts: object,
    second_sample_receipts: object,
) -> dict[str, object]:
    """Reopen every receipt and require the identical custody artifact."""

    value = validate_e0_custody_bytes(raw)
    rebuilt = canonical_e0_custody_bytes(
        e0_result=e0_result,
        first_exact=first_exact,
        second_exact=second_exact,
        predicted=predicted,
        first_sample_receipts=first_sample_receipts,
        second_sample_receipts=second_sample_receipts,
    )
    if rebuilt != raw:
        raise ValueError("ASG-CV E0 custody reopened inputs differ")
    return value
