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


def _fixture() -> tuple[bytes, np.ndarray, tuple[bytes, ...]]:
    exact32 = np.arange(64 * 8 * 2 * 3 * 4, dtype=np.float32).reshape(64, 8, 2, 3, 4)
    exact = exact32.astype(np.float64)
    e0 = canonical_e0_result_bytes(
        source_commit="1" * 40,
        dataset_manifest_sha256="2" * 64,
        partition_manifest_sha256="3" * 64,
        predictor_state_sha256="4" * 64,
        selection_seed_sha256="5" * 64,
        exact=exact,
        predicted=exact.copy(),
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
    patch_tokens = np.ones((2, 3, 4), dtype=np.float32)
    receipts = tuple(
        canonical_gradient_sample_bytes(
            source_commit="1" * 40,
            model_revision="7" * 40,
            fixture_sha256="8" * 64,
            completion_group_sha256=f"{ordinal + 1:064x}",
            completion_protocol_sha256="9" * 64,
            eligible_schedule_sha256="a" * 64,
            pooler_state_sha256="b" * 64,
            predictor_state_sha256="4" * 64,
            eligible_pair_ordinal=ordinal,
            candidate_pair_ordinal=ordinal,
            pair_ordinals=(ordinal * 2, ordinal * 2 + 1),
            relation_sign=1 if ordinal % 2 == 0 else -1,
            grpo_loss=0.0,
            attention_kl=0.0,
            generated_tokens=1,
            patch_tokens=patch_tokens,
            exact_gradient=exact32.reshape(512, 2, 3, 4)[ordinal],
        )
        for ordinal in range(512)
    )
    return e0, exact, receipts


def test_e0_custody_binds_every_receipt_and_exact_widened_fp32_row() -> None:
    e0, exact, receipts = _fixture()
    raw = canonical_e0_custody_bytes(
        e0_result=e0,
        exact=exact,
        sample_receipts=receipts,
    )
    value = validate_e0_custody_bytes(raw)

    assert value["schema"] == "sfora-asgcv-e0-custody-v1"
    assert value["claim_eligible"] is False
    assert value["sample_count"] == 512
    sample_digests = value["sample_sha256"]
    assert isinstance(sample_digests, list)
    assert len(sample_digests) == 512
    assert value["source_commit"] == "1" * 40
    assert value["model_revision"] == "7" * 40
    assert value["fixture_sha256"] == "8" * 64
    assert value["predictor_state_sha256"] == "4" * 64
    assert validate_e0_custody_bundle(
        raw,
        e0_result=e0,
        exact=exact,
        sample_receipts=receipts,
    ) == value


def test_e0_custody_rejects_order_identity_and_gradient_drift() -> None:
    e0, exact, receipts = _fixture()
    raw = canonical_e0_custody_bytes(
        e0_result=e0,
        exact=exact,
        sample_receipts=receipts,
    )

    swapped = list(receipts)
    swapped[0], swapped[1] = swapped[1], swapped[0]
    with pytest.raises(ValueError):
        canonical_e0_custody_bytes(e0_result=e0, exact=exact, sample_receipts=tuple(swapped))

    changed_identity = list(receipts)
    changed_identity[0] = canonical_gradient_sample_bytes(
        source_commit="1" * 40,
        model_revision="c" * 40,
        fixture_sha256="8" * 64,
        completion_group_sha256="1".zfill(64),
        completion_protocol_sha256="9" * 64,
        eligible_schedule_sha256="a" * 64,
        pooler_state_sha256="b" * 64,
        predictor_state_sha256="4" * 64,
        eligible_pair_ordinal=0,
        candidate_pair_ordinal=0,
        pair_ordinals=(0, 1),
        relation_sign=1,
        grpo_loss=0.0,
        attention_kl=0.0,
        generated_tokens=1,
        patch_tokens=np.ones((2, 3, 4), dtype=np.float32),
        exact_gradient=exact.astype(np.float32).reshape(512, 2, 3, 4)[0],
    )
    with pytest.raises(ValueError, match="identity"):
        canonical_e0_custody_bytes(
            e0_result=e0,
            exact=exact,
            sample_receipts=tuple(changed_identity),
        )

    drifted = exact.copy()
    drifted[0, 0, 0, 0, 0] = np.nextafter(drifted[0, 0, 0, 0, 0], np.inf)
    with pytest.raises(ValueError):
        canonical_e0_custody_bytes(e0_result=e0, exact=drifted, sample_receipts=receipts)

    value = json.loads(receipts[0])
    value["predictor_state_sha256"] = "c" * 64
    changed = list(receipts)
    changed[0] = json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    with pytest.raises(ValueError):
        canonical_e0_custody_bytes(e0_result=e0, exact=exact, sample_receipts=tuple(changed))

    receipt_value = json.loads(raw)
    receipt_value["sample_count"] = 511
    noncanonical = json.dumps(receipt_value, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    with pytest.raises(ValueError):
        validate_e0_custody_bytes(noncanonical)
