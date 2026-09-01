from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from sfora.siglip_checkpoint_audit import (
    SiglipCheckpointAuditAuthority,
    build_siglip_checkpoint_audit,
    canonical_siglip_checkpoint_audit_bytes,
    validate_siglip_checkpoint_audit_bytes,
)
from sfora.substrate_screen import (
    SubstrateRetrievalError,
    SubstrateScreenEvidence,
    SubstrateScreenMetrics,
)


def _authority() -> SiglipCheckpointAuditAuthority:
    examples = _examples()
    example_ids = [row.example_id for row in examples]
    labels = [row.label for row in examples]
    identity = lambda role, values: hashlib.sha256(  # noqa: E731
        (json.dumps({role: values}, sort_keys=True, separators=(",", ":")) + "\n").encode()
    ).hexdigest()
    return SiglipCheckpointAuditAuthority(
        source_revision="1" * 40,
        source_tree_digest="2" * 64,
        aggregate_sha256="3" * 64,
        seed_receipt_sha256="4" * 64,
        dataset_revision="9abf6cf7d6dfa7b95152a0d6e791ea9435b47a40",
        dataset_manifest_sha256="6" * 64,
        model_name="google/siglip-so400m-patch14-384",
        model_revision="9fdffc58afc957d1a03a25b10dba0329ab15c2a3",
        config_sha256="8" * 64,
        seed=17,
        checkpoint_sha256="9" * 64,
        checkpoint_bytes=5_146_653_305,
        checkpoint_epoch=60,
        evaluation_batch_size=32,
        query_block=128,
        ordered_example_ids_sha256=identity("example_ids", example_ids),
        label_vector_sha256=identity("labels", labels),
    )


def _examples() -> tuple[SimpleNamespace, ...]:
    return tuple(
        SimpleNamespace(example_id=f"cars/train/{position:04d}.jpg", label=82 + position % 16)
        for position in range(1_345)
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


def _evidence() -> tuple[
    SiglipCheckpointAuditAuthority,
    tuple[SimpleNamespace, ...],
    object,
]:
    examples = _examples()
    raw_error = SubstrateRetrievalError(
        query_position=0,
        nearest_position=1,
        query_label=82,
        nearest_label=83,
    )
    projected_errors = (
        raw_error,
        SubstrateRetrievalError(
            query_position=2,
            nearest_position=3,
            query_label=84,
            nearest_label=85,
        ),
    )
    authority = _authority()
    evidence = build_siglip_checkpoint_audit(
        authority=authority,
        examples=examples,
        raw=_screen(raw_error),
        projected=_screen(*projected_errors),
    )
    return authority, examples, evidence


def test_checkpoint_audit_builds_ordered_raw_and_projected_errors() -> None:
    authority, examples, evidence = _evidence()

    assert evidence.raw.name == "raw"
    assert evidence.raw.correct == 1_344
    assert evidence.raw.errors[0].query_example_id == examples[0].example_id
    assert evidence.raw.errors[0].nearest_example_id == examples[1].example_id
    assert evidence.projected.name == "projected"
    assert evidence.projected.correct == 1_343
    assert [row.query_position for row in evidence.projected.errors] == [0, 2]

    raw = canonical_siglip_checkpoint_audit_bytes(
        evidence,
        authority=authority,
        expected_example_ids=tuple(row.example_id for row in examples),
        expected_labels=tuple(row.label for row in examples),
    )
    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")
    payload = json.loads(raw)
    assert payload["schema"] == "sfora-siglip-checkpoint-error-audit-v1"
    assert payload["claim_eligible"] is False
    assert payload["official_test_access"] is False
    assert set(payload) == {
        "authority",
        "claim_eligible",
        "official_test_access",
        "projected",
        "raw",
        "schema",
    }
    assert "clean" not in raw.decode()
    assert "passed" not in payload
    validate_siglip_checkpoint_audit_bytes(
        raw,
        expected_authority=authority,
        expected_example_ids=tuple(row.example_id for row in examples),
        expected_labels=tuple(row.label for row in examples),
    )


def test_checkpoint_audit_rejects_authority_and_result_mutations() -> None:
    authority, examples, evidence = _evidence()
    example_ids = tuple(row.example_id for row in examples)
    labels = tuple(row.label for row in examples)
    raw = canonical_siglip_checkpoint_audit_bytes(
        evidence,
        authority=authority,
        expected_example_ids=example_ids,
        expected_labels=labels,
    )
    payload = json.loads(raw)

    authority_mutations = (
        replace(authority, seed=True),
        replace(authority, seed=29),
        replace(authority, checkpoint_epoch=59),
        replace(authority, checkpoint_bytes=True),
        replace(authority, source_revision="0" * 40),
    )
    for mutated in authority_mutations:
        with pytest.raises((TypeError, ValueError)):
            canonical_siglip_checkpoint_audit_bytes(
                evidence,
                authority=mutated,
                expected_example_ids=example_ids,
                expected_labels=labels,
            )

    mutations: list[dict[str, object]] = []
    for mutate in (
        lambda value: value.update(schema="wrong"),
        lambda value: value.update(claim_eligible=True),
        lambda value: value.update(official_test_access=True),
        lambda value: value["raw"].update(name="projected"),
        lambda value: value["raw"].update(correct=1_343),
        lambda value: value["raw"].update(queries=1_344),
        lambda value: value["raw"].update(recall_at_1=0.5),
        lambda value: value["raw"]["errors"][0].update(query_position=2),
        lambda value: value["raw"]["errors"][0].update(query_label=83),
        lambda value: value["raw"]["errors"][0].update(query_example_id="wrong"),
        lambda value: value["authority"].update(checkpoint_sha256="0" * 64),
    ):
        mutated = copy.deepcopy(payload)
        mutate(mutated)
        mutations.append(mutated)
    for mutated in mutations:
        mutated_raw = (
            json.dumps(mutated, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode()
        with pytest.raises((TypeError, ValueError)):
            validate_siglip_checkpoint_audit_bytes(
                mutated_raw,
                expected_authority=authority,
                expected_example_ids=example_ids,
                expected_labels=labels,
            )

    with pytest.raises(ValueError, match="canonical"):
        validate_siglip_checkpoint_audit_bytes(
            raw.replace(b'"schema":', b'"schema": '),
            expected_authority=authority,
            expected_example_ids=example_ids,
            expected_labels=labels,
        )

def test_checkpoint_audit_rejects_unordered_or_misbound_source_errors() -> None:
    examples = _examples()
    errors = (
        SubstrateRetrievalError(2, 3, 84, 85),
        SubstrateRetrievalError(0, 1, 82, 83),
    )
    with pytest.raises(ValueError, match="ordered"):
        build_siglip_checkpoint_audit(
            authority=_authority(),
            examples=examples,
            raw=_screen(*errors),
            projected=_screen(),
        )

    misbound = SubstrateRetrievalError(0, 1, 83, 82)
    with pytest.raises(ValueError, match="labels"):
        build_siglip_checkpoint_audit(
            authority=_authority(),
            examples=examples,
            raw=_screen(misbound),
            projected=_screen(),
        )
