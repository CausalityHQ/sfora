"""Authenticated burned-band error evidence for a seeded initial SigLIP control."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any, cast

from sfora.substrate_screen import (
    SubstrateRetrievalError,
    SubstrateScreenEvidence,
    SubstrateScreenMetrics,
)

_SCHEMA = "sfora-siglip-initial-control-error-audit-v1"
_PRODUCER_KIND = "seeded-initial-control"
_QUERY_COUNT = 1_345
_DATASET_REVISION = "9abf6cf7d6dfa7b95152a0d6e791ea9435b47a40"
_MODEL_NAME = "google/siglip-so400m-patch14-384"
_MODEL_REVISION = "9fdffc58afc957d1a03a25b10dba0329ab15c2a3"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True, slots=True)
class SiglipInitialControlAuditAuthority:
    """Exact source, initial model, and evaluation identity."""

    source_revision: str
    source_tree_digest: str
    seed_receipt_sha256: str
    dataset_revision: str
    dataset_manifest_sha256: str
    model_name: str
    model_revision: str
    config_sha256: str
    seed: int
    initial_state_sha256: str
    evaluation_batch_size: int
    query_block: int
    ordered_example_ids_sha256: str
    label_vector_sha256: str


@dataclass(frozen=True, slots=True)
class SiglipInitialControlErrorRow:
    """One exact misclassified query and its selected gallery row."""

    query_position: int
    query_example_id: str
    query_label: int
    nearest_position: int
    nearest_example_id: str
    nearest_label: int


@dataclass(frozen=True, slots=True)
class SiglipInitialControlRepresentationEvidence:
    """One initial descriptor plane's metrics and ordered errors."""

    name: str
    correct: int
    queries: int
    recall_at_1: float
    errors: tuple[SiglipInitialControlErrorRow, ...]


@dataclass(frozen=True, slots=True)
class SiglipInitialControlAuditEvidence:
    """Raw and projected evidence from one authenticated seeded initial state."""

    authority: SiglipInitialControlAuditAuthority
    raw: SiglipInitialControlRepresentationEvidence
    projected: SiglipInitialControlRepresentationEvidence


def _canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _identity_sha256(role: str, values: Sequence[object]) -> str:
    return hashlib.sha256(_canonical({role: list(values)})).hexdigest()


def _validate_examples(example_ids: Sequence[str], labels: Sequence[int]) -> None:
    if len(example_ids) != _QUERY_COUNT or len(labels) != _QUERY_COUNT:
        raise ValueError("SigLIP initial-control burned cardinality differs")
    if len(set(example_ids)) != _QUERY_COUNT or any(
        type(value) is not str or not value for value in example_ids
    ):
        raise ValueError("SigLIP initial-control example identity differs")
    if any(type(label) is not int or not 82 <= label <= 97 for label in labels):
        raise ValueError("SigLIP initial-control burned labels differ")
    counts = Counter(labels)
    if set(counts) != set(range(82, 98)) or any(count < 2 for count in counts.values()):
        raise ValueError("SigLIP initial-control burned class authority differs")


def _validate_authority(
    authority: SiglipInitialControlAuditAuthority,
    *,
    expected_example_ids: Sequence[str],
    expected_labels: Sequence[int],
) -> None:
    if type(authority) is not SiglipInitialControlAuditAuthority:
        raise TypeError("SigLIP initial-control authority has the wrong concrete type")
    if _COMMIT.fullmatch(authority.source_revision) is None:
        raise ValueError("SigLIP initial-control source revision differs")
    for name in (
        "source_tree_digest",
        "seed_receipt_sha256",
        "dataset_manifest_sha256",
        "config_sha256",
        "initial_state_sha256",
        "ordered_example_ids_sha256",
        "label_vector_sha256",
    ):
        if _SHA256.fullmatch(getattr(authority, name)) is None:
            raise ValueError(f"SigLIP initial-control {name} differs")
    if (
        authority.dataset_revision != _DATASET_REVISION
        or authority.model_name != _MODEL_NAME
        or authority.model_revision != _MODEL_REVISION
        or type(authority.seed) is not int
        or authority.seed != 17
        or type(authority.evaluation_batch_size) is not int
        or authority.evaluation_batch_size != 32
        or type(authority.query_block) is not int
        or authority.query_block != 128
    ):
        raise ValueError("SigLIP initial-control registered authority differs")
    if (
        authority.ordered_example_ids_sha256
        != _identity_sha256("example_ids", expected_example_ids)
        or authority.label_vector_sha256 != _identity_sha256("labels", expected_labels)
    ):
        raise ValueError("SigLIP initial-control burned example authority differs")


def _representation_from_screen(
    name: str,
    evidence: SubstrateScreenEvidence,
    *,
    example_ids: Sequence[str],
    labels: Sequence[int],
) -> SiglipInitialControlRepresentationEvidence:
    if name not in {"raw", "projected"} or type(evidence) is not SubstrateScreenEvidence:
        raise TypeError("SigLIP initial-control representation evidence differs")
    metrics = evidence.metrics
    if (
        type(metrics.correct) is not int
        or type(metrics.queries) is not int
        or type(metrics.recall_at_1) is not float
        or metrics.queries != _QUERY_COUNT
        or metrics.correct != _QUERY_COUNT - len(evidence.errors)
        or not math.isfinite(metrics.recall_at_1)
        or metrics.recall_at_1 != metrics.correct / metrics.queries
    ):
        raise ValueError("SigLIP initial-control retrieval metrics differ")
    rows: list[SiglipInitialControlErrorRow] = []
    previous = -1
    for error in evidence.errors:
        if type(error) is not SubstrateRetrievalError:
            raise TypeError("SigLIP initial-control retrieval error differs")
        if (
            type(error.query_position) is not int
            or type(error.nearest_position) is not int
            or not 0 <= error.query_position < _QUERY_COUNT
            or not 0 <= error.nearest_position < _QUERY_COUNT
            or error.query_position <= previous
            or error.query_position == error.nearest_position
        ):
            raise ValueError("SigLIP initial-control retrieval errors must be ordered")
        if (
            type(error.query_label) is not int
            or type(error.nearest_label) is not int
            or error.query_label != labels[error.query_position]
            or error.nearest_label != labels[error.nearest_position]
            or error.query_label == error.nearest_label
        ):
            raise ValueError("SigLIP initial-control retrieval error labels differ")
        rows.append(
            SiglipInitialControlErrorRow(
                query_position=error.query_position,
                query_example_id=example_ids[error.query_position],
                query_label=error.query_label,
                nearest_position=error.nearest_position,
                nearest_example_id=example_ids[error.nearest_position],
                nearest_label=error.nearest_label,
            )
        )
        previous = error.query_position
    return SiglipInitialControlRepresentationEvidence(
        name=name,
        correct=metrics.correct,
        queries=metrics.queries,
        recall_at_1=metrics.recall_at_1,
        errors=tuple(rows),
    )


def build_siglip_initial_control_audit(
    *,
    authority: SiglipInitialControlAuditAuthority,
    examples: Sequence[object],
    raw: SubstrateScreenEvidence,
    projected: SubstrateScreenEvidence,
) -> SiglipInitialControlAuditEvidence:
    """Bind two score passes to one exact seeded initial model state."""

    example_ids = tuple(getattr(example, "example_id", None) for example in examples)
    labels = tuple(getattr(example, "label", None) for example in examples)
    _validate_examples(cast(tuple[str, ...], example_ids), cast(tuple[int, ...], labels))
    _validate_authority(
        authority,
        expected_example_ids=cast(tuple[str, ...], example_ids),
        expected_labels=cast(tuple[int, ...], labels),
    )
    return SiglipInitialControlAuditEvidence(
        authority=authority,
        raw=_representation_from_screen(
            "raw",
            raw,
            example_ids=cast(tuple[str, ...], example_ids),
            labels=cast(tuple[int, ...], labels),
        ),
        projected=_representation_from_screen(
            "projected",
            projected,
            example_ids=cast(tuple[str, ...], example_ids),
            labels=cast(tuple[int, ...], labels),
        ),
    )


def _validate_representation_payload(
    value: object,
    *,
    expected: SiglipInitialControlRepresentationEvidence,
) -> None:
    expected_value = json.loads(_canonical(asdict(expected)))
    if type(value) is not dict or value != expected_value:
        raise ValueError("SigLIP initial-control representation artifact differs")


def canonical_siglip_initial_control_audit_bytes(
    evidence: SiglipInitialControlAuditEvidence,
    *,
    authority: SiglipInitialControlAuditAuthority,
    expected_example_ids: Sequence[str],
    expected_labels: Sequence[int],
) -> bytes:
    """Serialize one self-validating, claim-ineligible initial-state artifact."""

    if type(evidence) is not SiglipInitialControlAuditEvidence:
        raise TypeError("SigLIP initial-control evidence has the wrong concrete type")
    _validate_examples(expected_example_ids, expected_labels)
    _validate_authority(
        authority,
        expected_example_ids=expected_example_ids,
        expected_labels=expected_labels,
    )
    if evidence.authority != authority:
        raise ValueError("SigLIP initial-control evidence authority differs")
    payload = {
        "authority": asdict(authority),
        "claim_eligible": False,
        "official_test_access": False,
        "producer_kind": _PRODUCER_KIND,
        "projected": asdict(evidence.projected),
        "raw": asdict(evidence.raw),
        "schema": _SCHEMA,
    }
    raw = _canonical(payload)
    validate_siglip_initial_control_audit_bytes(
        raw,
        expected_authority=authority,
        expected_example_ids=expected_example_ids,
        expected_labels=expected_labels,
    )
    return raw


def validate_siglip_initial_control_audit_bytes(
    raw: bytes,
    *,
    expected_authority: SiglipInitialControlAuditAuthority,
    expected_example_ids: Sequence[str],
    expected_labels: Sequence[int],
) -> None:
    """Recompute and validate one canonical initial-control artifact."""

    if type(raw) is not bytes:
        raise TypeError("SigLIP initial-control artifact has the wrong concrete type")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("SigLIP initial-control artifact is invalid") from error
    if _canonical(value) != raw:
        raise ValueError("SigLIP initial-control artifact is not canonical")
    if type(value) is not dict or set(value) != {
        "authority",
        "claim_eligible",
        "official_test_access",
        "producer_kind",
        "projected",
        "raw",
        "schema",
    }:
        raise ValueError("SigLIP initial-control artifact schema differs")
    if (
        value["schema"] != _SCHEMA
        or value["producer_kind"] != _PRODUCER_KIND
        or value["claim_eligible"] is not False
        or value["official_test_access"] is not False
    ):
        raise ValueError("SigLIP initial-control artifact contract differs")
    _validate_examples(expected_example_ids, expected_labels)
    _validate_authority(
        expected_authority,
        expected_example_ids=expected_example_ids,
        expected_labels=expected_labels,
    )
    if value["authority"] != asdict(expected_authority):
        raise ValueError("SigLIP initial-control artifact authority differs")
    try:
        raw_evidence = SiglipInitialControlRepresentationEvidence(
            **cast(dict[str, Any], value["raw"])
        )
        projected_evidence = SiglipInitialControlRepresentationEvidence(
            **cast(dict[str, Any], value["projected"])
        )
    except TypeError as error:
        raise ValueError("SigLIP initial-control representation schema differs") from error
    expected_rows: dict[str, SiglipInitialControlRepresentationEvidence] = {}
    for name, candidate in (("raw", raw_evidence), ("projected", projected_evidence)):
        if type(candidate.errors) is not list:
            raise ValueError("SigLIP initial-control error schema differs")
        try:
            errors = tuple(
                SiglipInitialControlErrorRow(**cast(dict[str, Any], row))
                for row in candidate.errors
            )
        except TypeError as error:
            raise ValueError("SigLIP initial-control error schema differs") from error
        if any(type(row) is not dict for row in candidate.errors):
            raise ValueError("SigLIP initial-control error schema differs")
        screen_errors = tuple(
            SubstrateRetrievalError(
                query_position=row.query_position,
                nearest_position=row.nearest_position,
                query_label=row.query_label,
                nearest_label=row.nearest_label,
            )
            for row in errors
        )
        expected_rows[name] = _representation_from_screen(
            name,
            SubstrateScreenEvidence(
                metrics=SubstrateScreenMetrics(
                    correct=candidate.correct,
                    queries=candidate.queries,
                    recall_at_1=candidate.recall_at_1,
                ),
                errors=screen_errors,
            ),
            example_ids=expected_example_ids,
            labels=expected_labels,
        )
        _validate_representation_payload(value[name], expected=expected_rows[name])
