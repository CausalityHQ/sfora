from __future__ import annotations

import json

import numpy as np
import pytest

from sfora.asgcv import (
    AsgcvSrhtAuthority,
    canonical_e0_result_bytes,
    canonical_gradient_sample_bytes,
)
from sfora.asgcv_custody import (
    canonical_e0_custody_bytes,
    validate_e0_custody_bundle,
    validate_e0_custody_bytes,
)


def _gradient_receipt(
    exact32: np.ndarray,
    ordinal: int,
    *,
    model_revision: str = "7" * 40,
    pooler_state_sha256: str = "b" * 64,
    completion_group_sha256: str | None = None,
    patch_token_value: float = 1.0,
) -> bytes:
    return canonical_gradient_sample_bytes(
        source_commit="1" * 40,
        model_revision=model_revision,
        fixture_sha256="8" * 64,
        completion_group_sha256=(
            f"{ordinal + 1:064x}" if completion_group_sha256 is None else completion_group_sha256
        ),
        completion_protocol_sha256="9" * 64,
        eligible_schedule_sha256="a" * 64,
        pooler_state_sha256=pooler_state_sha256,
        eligible_pair_ordinal=ordinal,
        candidate_pair_ordinal=ordinal,
        pair_ordinals=(ordinal * 2, ordinal * 2 + 1),
        relation_sign=1 if ordinal % 2 == 0 else -1,
        grpo_loss=0.0,
        attention_kl=0.0,
        generated_tokens=1,
        patch_tokens=np.full((2, 3, 4), patch_token_value, dtype=np.float32),
        exact_gradient=exact32.reshape(-1, 2, 3, 4)[ordinal],
    )


def _fixture() -> tuple[
    bytes,
    np.ndarray,
    np.ndarray,
    tuple[bytes, ...],
    tuple[bytes, ...],
]:
    first32 = np.arange(64 * 8 * 2 * 3 * 4, dtype=np.float32).reshape(64, 8, 2, 3, 4)
    second32 = first32 + np.float32(1.0)
    first = first32.astype(np.float64)
    second = second32.astype(np.float64)
    e0 = canonical_e0_result_bytes(
        source_commit="1" * 40,
        dataset_manifest_sha256="2" * 64,
        partition_manifest_sha256="3" * 64,
        predictor_state_sha256="4" * 64,
        selection_seed_sha256="5" * 64,
        first_exact=first,
        second_exact=second,
        predicted=first.copy(),
        srht_authority=AsgcvSrhtAuthority(
            input_dimensions=4,
            padded_dimensions=4,
            output_dimensions=2,
            seed_sha256="6" * 64,
        ).validated(),
        peak_cuda_reserved_bytes=1,
        exact_semantic_wall_ns=10,
        asgcv_semantic_wall_ns=1,
    )
    first_receipts = tuple(_gradient_receipt(first32, ordinal) for ordinal in range(512))
    second_receipts = tuple(
        _gradient_receipt(
            second32,
            ordinal,
            completion_group_sha256=f"{ordinal + 513:064x}",
        )
        for ordinal in range(512)
    )
    return e0, first, second, first_receipts, second_receipts


def test_e0_custody_binds_every_receipt_and_exact_widened_fp32_row() -> None:
    e0, first, second, first_receipts, second_receipts = _fixture()
    raw = canonical_e0_custody_bytes(
        e0_result=e0,
        first_exact=first,
        second_exact=second,
        predicted=first.copy(),
        first_sample_receipts=first_receipts,
        second_sample_receipts=second_receipts,
    )
    assert (
        canonical_e0_custody_bytes(
            e0_result=e0,
            first_exact=second,
            second_exact=first,
            predicted=first.copy(),
            first_sample_receipts=second_receipts,
            second_sample_receipts=first_receipts,
        )
        == raw
    )
    value = validate_e0_custody_bytes(raw)

    assert value["schema"] == "sfora-asgcv-e0-custody-v4"
    assert value["claim_eligible"] is False
    assert value["sample_count"] == 512
    first_sample_digests = value["first_sample_sha256"]
    second_sample_digests = value["second_sample_sha256"]
    assert isinstance(first_sample_digests, list)
    assert isinstance(second_sample_digests, list)
    assert len(first_sample_digests) == 512
    assert len(second_sample_digests) == 512
    assert set(first_sample_digests).isdisjoint(second_sample_digests)
    assert value["source_commit"] == "1" * 40
    assert value["model_revision"] == "7" * 40
    assert value["fixture_sha256"] == "8" * 64
    assert value["predictor_state_sha256"] == "4" * 64
    assert (
        validate_e0_custody_bundle(
            raw,
            e0_result=e0,
            first_exact=first,
            second_exact=second,
            predicted=first.copy(),
            first_sample_receipts=first_receipts,
            second_sample_receipts=second_receipts,
        )
        == value
    )


def test_e0_custody_rejects_order_identity_and_gradient_drift() -> None:
    e0, first, second, first_receipts, second_receipts = _fixture()
    raw = canonical_e0_custody_bytes(
        e0_result=e0,
        first_exact=first,
        second_exact=second,
        predicted=first.copy(),
        first_sample_receipts=first_receipts,
        second_sample_receipts=second_receipts,
    )

    swapped = list(first_receipts)
    swapped[0], swapped[1] = swapped[1], swapped[0]
    with pytest.raises(ValueError):
        canonical_e0_custody_bytes(
            e0_result=e0,
            first_exact=first,
            second_exact=second,
            predicted=first.copy(),
            first_sample_receipts=tuple(swapped),
            second_sample_receipts=second_receipts,
        )

    changed_identity = list(first_receipts)
    changed_identity[0] = _gradient_receipt(
        first.astype(np.float32),
        0,
        model_revision="c" * 40,
    )
    with pytest.raises(ValueError, match="identity"):
        canonical_e0_custody_bytes(
            e0_result=e0,
            first_exact=first,
            second_exact=second,
            predicted=first.copy(),
            first_sample_receipts=tuple(changed_identity),
            second_sample_receipts=second_receipts,
        )

    changed_pooler = list(first_receipts)
    changed_pooler[0] = _gradient_receipt(
        first.astype(np.float32),
        0,
        pooler_state_sha256="c" * 64,
    )
    with pytest.raises(ValueError, match="identity"):
        canonical_e0_custody_bytes(
            e0_result=e0,
            first_exact=first,
            second_exact=second,
            predicted=first.copy(),
            first_sample_receipts=tuple(changed_pooler),
            second_sample_receipts=second_receipts,
        )

    duplicate_group = list(first_receipts)
    duplicate_group[1] = _gradient_receipt(
        first.astype(np.float32),
        1,
        completion_group_sha256="1".zfill(64),
    )
    with pytest.raises(ValueError, match="schedule"):
        canonical_e0_custody_bytes(
            e0_result=e0,
            first_exact=first,
            second_exact=second,
            predicted=first.copy(),
            first_sample_receipts=tuple(duplicate_group),
            second_sample_receipts=second_receipts,
        )

    predicted_drift = first.copy()
    predicted_drift[0, 0, 0, 0, 0] += 1.0
    with pytest.raises(ValueError, match="authority"):
        canonical_e0_custody_bytes(
            e0_result=e0,
            first_exact=first,
            second_exact=second,
            predicted=predicted_drift,
            first_sample_receipts=first_receipts,
            second_sample_receipts=second_receipts,
        )

    drifted = first.copy()
    drifted[0, 0, 0, 0, 0] = np.nextafter(drifted[0, 0, 0, 0, 0], np.inf)
    with pytest.raises(ValueError):
        canonical_e0_custody_bytes(
            e0_result=e0,
            first_exact=drifted,
            second_exact=second,
            predicted=first.copy(),
            first_sample_receipts=first_receipts,
            second_sample_receipts=second_receipts,
        )

    second_drift = second.copy()
    second_drift[0, 0, 0, 0, 0] = np.nextafter(second_drift[0, 0, 0, 0, 0], np.inf)
    with pytest.raises(ValueError):
        canonical_e0_custody_bytes(
            e0_result=e0,
            first_exact=first,
            second_exact=second_drift,
            predicted=first.copy(),
            first_sample_receipts=first_receipts,
            second_sample_receipts=second_receipts,
        )

    mismatched_second = list(second_receipts)
    mismatched_second[0] = _gradient_receipt(
        first.astype(np.float32),
        0,
        completion_group_sha256=f"{513:064x}",
    )
    with pytest.raises(ValueError, match="binding"):
        canonical_e0_custody_bytes(
            e0_result=e0,
            first_exact=first,
            second_exact=second,
            predicted=first.copy(),
            first_sample_receipts=first_receipts,
            second_sample_receipts=tuple(mismatched_second),
        )

    reused_groups = tuple(
        _gradient_receipt(
            second.astype(np.float32),
            ordinal,
            completion_group_sha256=f"{ordinal + 1:064x}",
        )
        for ordinal in range(512)
    )
    with pytest.raises(ValueError, match="seed block"):
        canonical_e0_custody_bytes(
            e0_result=e0,
            first_exact=first,
            second_exact=second,
            predicted=first.copy(),
            first_sample_receipts=first_receipts,
            second_sample_receipts=reused_groups,
        )

    changed_tokens = list(second_receipts)
    changed_tokens[0] = _gradient_receipt(
        second.astype(np.float32),
        0,
        completion_group_sha256=f"{513:064x}",
        patch_token_value=2.0,
    )
    with pytest.raises(ValueError, match="patch-token"):
        canonical_e0_custody_bytes(
            e0_result=e0,
            first_exact=first,
            second_exact=second,
            predicted=first.copy(),
            first_sample_receipts=first_receipts,
            second_sample_receipts=tuple(changed_tokens),
        )

    receipt_value = json.loads(raw)
    receipt_value["sample_count"] = 511
    noncanonical = json.dumps(receipt_value, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    with pytest.raises(ValueError):
        validate_e0_custody_bytes(noncanonical)
