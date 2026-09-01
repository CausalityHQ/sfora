"""Canonical custody chain for ASG-CV E0 gradient evidence."""

from __future__ import annotations

import hashlib
import json
from typing import cast

import numpy as np

from sfora.asgcv import (
    ASGCV_E0_MINIMUM_PAIRS,
    _array_authority,
    _canonical_json_bytes,
    _gradient_sample_array_authority,
    _sha256_bytes,
    _source_commit,
    validate_e0_result_bytes,
    validate_gradient_sample_bytes,
)

ASGCV_E0_CUSTODY_SCHEMA = "sfora-asgcv-e0-custody-v3"


def canonical_e0_custody_bytes(
    *,
    e0_result: bytes,
    exact: object,
    predicted: object,
    sample_receipts: object,
) -> bytes:
    """Bind each E0 row to one canonical captured-gradient receipt."""

    result = validate_e0_result_bytes(e0_result)
    if (
        type(exact) is not np.ndarray
        or exact.dtype != np.dtype(np.float64)
        or exact.ndim != 5
        or exact.shape[0] * exact.shape[1] < ASGCV_E0_MINIMUM_PAIRS
        or exact.shape[0] * exact.shape[1] % 8 != 0
        or exact.shape[1] != 8
        or exact.shape[2] != 2
        or not bool(np.isfinite(exact).all())
        or not np.array_equal(exact, exact.astype(np.float32).astype(np.float64))
        or type(predicted) is not np.ndarray
        or predicted.dtype != np.dtype(np.float64)
        or predicted.shape != exact.shape
        or not bool(np.isfinite(predicted).all())
        or type(sample_receipts) is not tuple
        or len(sample_receipts) != exact.shape[0] * exact.shape[1]
    ):
        raise ValueError("ASG-CV E0 custody shape differs")
    arrays = result["arrays"]
    if (
        type(arrays) is not dict
        or arrays["exact_gradients"]
        != _array_authority(exact, role="exact-gradients", dimensions=5)
        or arrays["predicted_gradients"]
        != _array_authority(predicted, role="predicted-gradients", dimensions=5)
    ):
        raise ValueError("ASG-CV E0 custody exact-gradient authority differs")

    source_commit = result["source_commit"]
    predictor_state = result["predictor_state_sha256"]
    sample_count = exact.shape[0] * exact.shape[1]
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
        "sample_count": len(values),
        "sample_sha256": [value["sample_sha256"] for value in values],
        "patch_token_sha256": [
            cast(dict[str, dict[str, object]], value["arrays"])["patch_tokens"]["sha256"]
            for value in values
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
        "sample_sha256",
        "patch_token_sha256",
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
        or type(value["sample_sha256"]) is not list
        or len(value["sample_sha256"]) != value["sample_count"]
        or type(value["patch_token_sha256"]) is not list
        or len(value["patch_token_sha256"]) != value["sample_count"]
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
    for digest in value["sample_sha256"]:
        _sha256_bytes(digest, name="E0 custody sample digest")
    for digest in value["patch_token_sha256"]:
        _sha256_bytes(digest, name="E0 custody patch-token digest")
    unsigned = dict(value)
    digest = unsigned.pop("custody_sha256")
    if hashlib.sha256(_canonical_json_bytes(unsigned)).hexdigest() != digest:
        raise ValueError("ASG-CV E0 custody digest differs")
    return value


def validate_e0_custody_bundle(
    raw: bytes,
    *,
    e0_result: bytes,
    exact: object,
    predicted: object,
    sample_receipts: object,
) -> dict[str, object]:
    """Reopen every receipt and require the identical custody artifact."""

    value = validate_e0_custody_bytes(raw)
    rebuilt = canonical_e0_custody_bytes(
        e0_result=e0_result,
        exact=exact,
        predicted=predicted,
        sample_receipts=sample_receipts,
    )
    if rebuilt != raw:
        raise ValueError("ASG-CV E0 custody reopened inputs differ")
    return value
