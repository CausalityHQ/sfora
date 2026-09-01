from __future__ import annotations

import hashlib
import json

import pytest

from sfora.asgcv_e0_capture import (
    canonical_capture_manifest_bytes,
    canonical_phase_receipt_bytes,
    validate_capture_manifest_bytes,
    validate_phase_receipt_bytes,
)
from sfora.asgcv_protocol import AsgcvPartitionAuthority, AsgcvRolloutAuthority


def _canonical(value: dict[str, object]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _rehash(raw: bytes, digest_name: str) -> bytes:
    value = json.loads(raw)
    value.pop(digest_name)
    value[digest_name] = hashlib.sha256(_canonical(value)).hexdigest()
    return _canonical(value)


def _partition() -> AsgcvPartitionAuthority:
    return AsgcvPartitionAuthority(
        source_manifest_sha256="10" * 32,
        partition_seed_sha256="11" * 32,
        predictor_train_class_ids=(0, 1),
        e0_validation_class_ids=(2, 3),
        e1_optimization_class_ids=(4, 5),
    ).validated()


def _rollout() -> AsgcvRolloutAuthority:
    return AsgcvRolloutAuthority(
        master_seed_sha256="12" * 32,
        model_revision="5" * 40,
        temperature=0.7,
        top_p=0.9,
        max_new_tokens=128,
    ).validated()


def _manifest() -> bytes:
    return canonical_capture_manifest_bytes(
        source_commit="1" * 40,
        dataset_manifest_sha256="10" * 32,
        partition_authority=_partition(),
        rollout_authority=_rollout(),
        predictor_train_candidate_schedule_sha256="3" * 64,
        predictor_train_eligible_schedule_sha256="4" * 64,
        e0_validation_candidate_schedule_sha256="6" * 64,
        e0_validation_eligible_schedule_sha256="7" * 64,
        model_revision="5" * 40,
        fixture_sha256="8" * 64,
        pooler_state_sha256="9" * 64,
        official_test_access=False,
    )


def test_capture_manifest_binds_disjoint_phase_inputs_and_forbids_test_access() -> None:
    raw = _manifest()
    value = validate_capture_manifest_bytes(raw)
    assert value["claim_eligible"] is False
    assert value["official_test_access"] is False
    assert value["partition_authority"] == _partition().to_mapping()
    assert value["rollout_authority"] == _rollout().to_mapping()
    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")

    for name, replacement in (
        ("official_test_access", True),
        ("claim_eligible", 0),
        ("model_revision", True),
        ("pooler_state_sha256", True),
    ):
        mutation = json.loads(raw)
        mutation[name] = replacement
        with pytest.raises(ValueError):
            validate_capture_manifest_bytes(_rehash(_canonical(mutation), "manifest_sha256"))

    partition_drift = json.loads(raw)
    partition_drift["partition_authority"]["source_manifest_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="binding"):
        validate_capture_manifest_bytes(_rehash(_canonical(partition_drift), "manifest_sha256"))

    model_drift = json.loads(raw)
    model_drift["model_revision"] = "f" * 40
    with pytest.raises(ValueError, match="model binding"):
        validate_capture_manifest_bytes(_rehash(_canonical(model_drift), "manifest_sha256"))


def test_phase_receipts_require_exact_monotone_transition_chain() -> None:
    manifest = validate_capture_manifest_bytes(_manifest())
    manifest_digest = manifest["manifest_sha256"]
    assert isinstance(manifest_digest, str)
    eligibility = canonical_phase_receipt_bytes(
        manifest_sha256=manifest_digest,
        phase="eligibility",
        input_sha256=(manifest_digest,),
        output_sha256=("a" * 64, "b" * 64),
        elapsed_ns=1,
    )
    capture = canonical_phase_receipt_bytes(
        manifest_sha256=manifest_digest,
        phase="capture",
        input_sha256=("a" * 64, "b" * 64),
        output_sha256=("c" * 64, "d" * 64),
        elapsed_ns=2,
        previous_phase_receipt=eligibility,
    )
    fit = canonical_phase_receipt_bytes(
        manifest_sha256=manifest_digest,
        phase="fit",
        input_sha256=("c" * 64, "d" * 64),
        output_sha256=("e" * 64,),
        elapsed_ns=3,
        previous_phase_receipt=capture,
    )
    evaluate = canonical_phase_receipt_bytes(
        manifest_sha256=manifest_digest,
        phase="evaluate",
        input_sha256=("e" * 64,),
        output_sha256=("f" * 64,),
        elapsed_ns=4,
        previous_phase_receipt=fit,
    )
    assert (
        validate_phase_receipt_bytes(
            evaluate,
            previous_phase_receipt=fit,
        )["phase"]
        == "evaluate"
    )

    with pytest.raises(ValueError):
        canonical_phase_receipt_bytes(
            manifest_sha256=manifest_digest,
            phase="evaluate",
            input_sha256=("0" * 64,),
            output_sha256=("f" * 64,),
            elapsed_ns=4,
            previous_phase_receipt=fit,
        )
    with pytest.raises(ValueError):
        canonical_phase_receipt_bytes(
            manifest_sha256=manifest_digest,
            phase="capture",
            input_sha256=("a" * 64, "b" * 64),
            output_sha256=("c" * 64, "c" * 64),
            elapsed_ns=2,
            previous_phase_receipt=eligibility,
        )
    with pytest.raises(ValueError):
        canonical_phase_receipt_bytes(
            manifest_sha256=manifest_digest,
            phase="evaluate",
            input_sha256=("a" * 64, "b" * 64),
            output_sha256=("f" * 64,),
            elapsed_ns=4,
            previous_phase_receipt=eligibility,
        )

    elapsed_type = json.loads(capture)
    elapsed_type["elapsed_ns"] = True
    with pytest.raises(ValueError):
        validate_phase_receipt_bytes(
            _rehash(_canonical(elapsed_type), "receipt_sha256"),
            previous_phase_receipt=eligibility,
        )
