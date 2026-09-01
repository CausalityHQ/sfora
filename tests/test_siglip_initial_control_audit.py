from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from sfora.siglip_initial_control_audit import (
    SiglipInitialControlAuditAuthority,
    build_siglip_initial_control_audit,
    canonical_siglip_initial_control_audit_bytes,
    validate_siglip_initial_control_audit_bytes,
)
from sfora.substrate_screen import (
    SubstrateRetrievalError,
    SubstrateScreenEvidence,
    SubstrateScreenMetrics,
)


def _canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _examples() -> tuple[SimpleNamespace, ...]:
    return tuple(
        SimpleNamespace(example_id=f"cars/train/{position:04d}.jpg", label=82 + position % 16)
        for position in range(1_345)
    )


def _identity(role: str, values: list[object]) -> str:
    return hashlib.sha256(_canonical({role: values})).hexdigest()


def _authority() -> SiglipInitialControlAuditAuthority:
    examples = _examples()
    return SiglipInitialControlAuditAuthority(
        source_revision="1" * 40,
        source_tree_digest="2" * 64,
        seed_receipt_sha256="3" * 64,
        dataset_revision="9abf6cf7d6dfa7b95152a0d6e791ea9435b47a40",
        dataset_manifest_sha256="4" * 64,
        model_name="google/siglip-so400m-patch14-384",
        model_revision="9fdffc58afc957d1a03a25b10dba0329ab15c2a3",
        config_sha256="5" * 64,
        seed=17,
        initial_state_sha256="6" * 64,
        evaluation_batch_size=32,
        query_block=128,
        ordered_example_ids_sha256=_identity(
            "example_ids", [row.example_id for row in examples]
        ),
        label_vector_sha256=_identity("labels", [row.label for row in examples]),
    )


def _screen(*errors: SubstrateRetrievalError) -> SubstrateScreenEvidence:
    correct = 1_345 - len(errors)
    return SubstrateScreenEvidence(
        metrics=SubstrateScreenMetrics(
            correct=correct,
            queries=1_345,
            recall_at_1=correct / 1_345,
        ),
        errors=tuple(errors),
    )


def _artifact() -> tuple[SiglipInitialControlAuditAuthority, tuple[SimpleNamespace, ...], bytes]:
    examples = _examples()
    authority = _authority()
    raw = _screen(SubstrateRetrievalError(0, 1, 82, 83))
    projected = _screen(
        SubstrateRetrievalError(0, 1, 82, 83),
        SubstrateRetrievalError(2, 3, 84, 85),
    )
    evidence = build_siglip_initial_control_audit(
        authority=authority,
        examples=examples,
        raw=raw,
        projected=projected,
    )
    encoded = canonical_siglip_initial_control_audit_bytes(
        evidence,
        authority=authority,
        expected_example_ids=tuple(row.example_id for row in examples),
        expected_labels=tuple(row.label for row in examples),
    )
    return authority, examples, encoded


def test_initial_control_audit_binds_exact_initial_state_and_ordered_errors() -> None:
    authority, examples, encoded = _artifact()

    payload = json.loads(encoded)
    assert payload["schema"] == "sfora-siglip-initial-control-error-audit-v1"
    assert payload["claim_eligible"] is False
    assert payload["official_test_access"] is False
    assert payload["producer_kind"] == "seeded-initial-control"
    assert payload["raw"]["correct"] == 1_344
    assert payload["projected"]["correct"] == 1_343
    assert payload["projected"]["errors"][1] == {
        "nearest_example_id": examples[3].example_id,
        "nearest_label": 85,
        "nearest_position": 3,
        "query_example_id": examples[2].example_id,
        "query_label": 84,
        "query_position": 2,
    }
    assert "checkpoint" not in encoded.decode()
    assert "clean" not in encoded.decode()
    assert "passed" not in payload
    assert encoded.endswith(b"\n") and not encoded.endswith(b"\n\n")
    validate_siglip_initial_control_audit_bytes(
        encoded,
        expected_authority=authority,
        expected_example_ids=tuple(row.example_id for row in examples),
        expected_labels=tuple(row.label for row in examples),
    )


def test_initial_control_audit_rejects_authority_result_and_canonical_drift() -> None:
    authority, examples, encoded = _artifact()
    example_ids = tuple(row.example_id for row in examples)
    labels = tuple(row.label for row in examples)
    evidence = json.loads(encoded)

    for mutated in (
        replace(authority, seed=True),
        replace(authority, seed=29),
        replace(authority, initial_state_sha256="x" * 64),
        replace(authority, evaluation_batch_size=8),
    ):
        with pytest.raises((TypeError, ValueError)):
            validate_siglip_initial_control_audit_bytes(
                encoded,
                expected_authority=mutated,
                expected_example_ids=example_ids,
                expected_labels=labels,
            )

    for mutate in (
        lambda value: value.update(schema="wrong"),
        lambda value: value.update(claim_eligible=True),
        lambda value: value.update(producer_kind="checkpoint"),
        lambda value: value["raw"].update(correct=1_343),
        lambda value: value["projected"]["errors"][0].update(query_position=2),
        lambda value: value["projected"]["errors"][0].update(query_example_id="wrong"),
        lambda value: value["authority"].update(initial_state_sha256="0" * 64),
    ):
        mutated = copy.deepcopy(evidence)
        mutate(mutated)
        raw = _canonical(mutated)
        with pytest.raises((TypeError, ValueError)):
            validate_siglip_initial_control_audit_bytes(
                raw,
                expected_authority=authority,
                expected_example_ids=example_ids,
                expected_labels=labels,
            )

    with pytest.raises(ValueError, match="canonical"):
        validate_siglip_initial_control_audit_bytes(
            encoded.replace(b'"schema":', b'"schema": '),
            expected_authority=authority,
            expected_example_ids=example_ids,
            expected_labels=labels,
        )


def test_initial_control_audit_rejects_misbound_source_errors() -> None:
    examples = _examples()
    with pytest.raises(ValueError, match="labels"):
        build_siglip_initial_control_audit(
            authority=_authority(),
            examples=examples,
            raw=_screen(SubstrateRetrievalError(0, 1, 83, 82)),
            projected=_screen(),
        )
