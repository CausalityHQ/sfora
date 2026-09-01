"""Authenticated burned-band error evidence for a trained SigLIP checkpoint."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any, cast

from sfora.substrate_screen import SubstrateRetrievalError, SubstrateScreenEvidence

_SCHEMA = "sfora-siglip-checkpoint-error-audit-v1"
_QUERY_COUNT = 1_345
_DATASET_REVISION = "9abf6cf7d6dfa7b95152a0d6e791ea9435b47a40"
_MODEL_NAME = "google/siglip-so400m-patch14-384"
_MODEL_REVISION = "9fdffc58afc957d1a03a25b10dba0329ab15c2a3"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True, slots=True)
class SiglipCheckpointAuditAuthority:
    """Exact campaign, model, checkpoint, and evaluation identity."""

    source_revision: str
    source_tree_digest: str
    aggregate_sha256: str
    seed_receipt_sha256: str
    dataset_revision: str
    dataset_manifest_sha256: str
    model_name: str
    model_revision: str
    config_sha256: str
    seed: int
    checkpoint_sha256: str
    checkpoint_bytes: int
    checkpoint_epoch: int
    evaluation_batch_size: int
    query_block: int
    ordered_example_ids_sha256: str
    label_vector_sha256: str


@dataclass(frozen=True, slots=True)
class SiglipCheckpointErrorRow:
    """One exact misclassified query and its selected gallery row."""

    query_position: int
    query_example_id: str
    query_label: int
    nearest_position: int
    nearest_example_id: str
    nearest_label: int


@dataclass(frozen=True, slots=True)
class SiglipCheckpointRepresentationEvidence:
    """One descriptor plane's metrics and ordered errors."""

    name: str
    correct: int
    queries: int
    recall_at_1: float
    errors: tuple[SiglipCheckpointErrorRow, ...]


@dataclass(frozen=True, slots=True)
class SiglipCheckpointAuditEvidence:
    """Raw and projected evidence from one authenticated terminal checkpoint."""

    authority: SiglipCheckpointAuditAuthority
    raw: SiglipCheckpointRepresentationEvidence
    projected: SiglipCheckpointRepresentationEvidence


def _canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _identity_sha256(role: str, values: Sequence[object]) -> str:
    return hashlib.sha256(_canonical({role: list(values)})).hexdigest()


def _validate_authority(
    authority: SiglipCheckpointAuditAuthority,
    *,
    expected_example_ids: Sequence[str],
    expected_labels: Sequence[int],
) -> None:
    if type(authority) is not SiglipCheckpointAuditAuthority:
        raise TypeError("SigLIP checkpoint authority has the wrong concrete type")
    if _COMMIT.fullmatch(authority.source_revision) is None:
        raise ValueError("SigLIP checkpoint source revision differs")
    for name in (
        "source_tree_digest",
        "aggregate_sha256",
        "seed_receipt_sha256",
        "dataset_manifest_sha256",
        "config_sha256",
        "checkpoint_sha256",
        "ordered_example_ids_sha256",
        "label_vector_sha256",
    ):
        if _SHA256.fullmatch(getattr(authority, name)) is None:
            raise ValueError(f"SigLIP checkpoint {name} differs")
    if (
        authority.dataset_revision != _DATASET_REVISION
        or authority.model_name != _MODEL_NAME
        or authority.model_revision != _MODEL_REVISION
        or type(authority.seed) is not int
        or authority.seed != 17
        or type(authority.checkpoint_bytes) is not int
        or authority.checkpoint_bytes < 1
        or type(authority.checkpoint_epoch) is not int
        or authority.checkpoint_epoch != 60
        or type(authority.evaluation_batch_size) is not int
        or authority.evaluation_batch_size != 32
        or type(authority.query_block) is not int
        or authority.query_block != 128
    ):
        raise ValueError("SigLIP checkpoint registered authority differs")
    if (
        authority.ordered_example_ids_sha256
        != _identity_sha256("example_ids", expected_example_ids)
        or authority.label_vector_sha256 != _identity_sha256("labels", expected_labels)
    ):
        raise ValueError("SigLIP checkpoint burned example authority differs")


def _validate_examples(
    example_ids: Sequence[str],
    labels: Sequence[int],
) -> None:
    if len(example_ids) != _QUERY_COUNT or len(labels) != _QUERY_COUNT:
        raise ValueError("SigLIP checkpoint burned cardinality differs")
    if len(set(example_ids)) != _QUERY_COUNT or any(
        type(value) is not str or not value for value in example_ids
    ):
        raise ValueError("SigLIP checkpoint example identity differs")
    if any(type(label) is not int or not 82 <= label <= 97 for label in labels):
        raise ValueError("SigLIP checkpoint burned labels differ")
    counts = Counter(labels)
    if set(counts) != set(range(82, 98)) or any(count < 2 for count in counts.values()):
        raise ValueError("SigLIP checkpoint burned class authority differs")


def _representation_from_screen(
    name: str,
    evidence: SubstrateScreenEvidence,
    *,
    example_ids: Sequence[str],
    labels: Sequence[int],
) -> SiglipCheckpointRepresentationEvidence:
    if name not in {"raw", "projected"} or type(evidence) is not SubstrateScreenEvidence:
        raise TypeError("SigLIP checkpoint representation evidence differs")
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
        raise ValueError("SigLIP checkpoint retrieval metrics differ")
    rows: list[SiglipCheckpointErrorRow] = []
    previous = -1
    for error in evidence.errors:
        if type(error) is not SubstrateRetrievalError:
            raise TypeError("SigLIP checkpoint retrieval error differs")
        if (
            type(error.query_position) is not int
            or type(error.nearest_position) is not int
            or not 0 <= error.query_position < _QUERY_COUNT
            or not 0 <= error.nearest_position < _QUERY_COUNT
            or error.query_position <= previous
            or error.query_position == error.nearest_position
        ):
            raise ValueError("SigLIP checkpoint retrieval errors must be ordered")
        if (
            type(error.query_label) is not int
            or type(error.nearest_label) is not int
            or error.query_label != labels[error.query_position]
            or error.nearest_label != labels[error.nearest_position]
            or error.query_label == error.nearest_label
        ):
            raise ValueError("SigLIP checkpoint retrieval error labels differ")
        rows.append(
            SiglipCheckpointErrorRow(
                query_position=error.query_position,
                query_example_id=example_ids[error.query_position],
                query_label=error.query_label,
                nearest_position=error.nearest_position,
                nearest_example_id=example_ids[error.nearest_position],
                nearest_label=error.nearest_label,
            )
        )
        previous = error.query_position
    return SiglipCheckpointRepresentationEvidence(
        name=name,
        correct=metrics.correct,
        queries=metrics.queries,
        recall_at_1=metrics.recall_at_1,
        errors=tuple(rows),
    )


def build_siglip_checkpoint_audit(
    *,
    authority: SiglipCheckpointAuditAuthority,
    examples: Sequence[object],
    raw: SubstrateScreenEvidence,
    projected: SubstrateScreenEvidence,
) -> SiglipCheckpointAuditEvidence:
    """Bind two exact score passes to ordered burned example identities."""

    example_ids = tuple(getattr(example, "example_id", None) for example in examples)
    labels = tuple(getattr(example, "label", None) for example in examples)
    _validate_examples(cast(tuple[str, ...], example_ids), cast(tuple[int, ...], labels))
    _validate_authority(
        authority,
        expected_example_ids=cast(tuple[str, ...], example_ids),
        expected_labels=cast(tuple[int, ...], labels),
    )
    return SiglipCheckpointAuditEvidence(
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


def _representation_payload(
    evidence: SiglipCheckpointRepresentationEvidence,
) -> dict[str, object]:
    return {
        "name": evidence.name,
        "correct": evidence.correct,
        "queries": evidence.queries,
        "recall_at_1": evidence.recall_at_1,
        "errors": [asdict(row) for row in evidence.errors],
    }


def canonical_siglip_checkpoint_audit_bytes(
    evidence: SiglipCheckpointAuditEvidence,
    *,
    authority: SiglipCheckpointAuditAuthority,
    expected_example_ids: Sequence[str],
    expected_labels: Sequence[int],
) -> bytes:
    """Serialize and self-validate one exact claim-ineligible audit result."""

    if type(evidence) is not SiglipCheckpointAuditEvidence or evidence.authority != authority:
        raise ValueError("SigLIP checkpoint audit authority differs")
    raw = _canonical(
        {
            "schema": _SCHEMA,
            "claim_eligible": False,
            "official_test_access": False,
            "authority": asdict(authority),
            "raw": _representation_payload(evidence.raw),
            "projected": _representation_payload(evidence.projected),
        }
    )
    validate_siglip_checkpoint_audit_bytes(
        raw,
        expected_authority=authority,
        expected_example_ids=expected_example_ids,
        expected_labels=expected_labels,
    )
    return raw


def _parse_representation(
    payload: object,
    *,
    expected_name: str,
    example_ids: Sequence[str],
    labels: Sequence[int],
) -> None:
    if type(payload) is not dict or set(payload) != {
        "name",
        "correct",
        "queries",
        "recall_at_1",
        "errors",
    }:
        raise ValueError("SigLIP checkpoint representation schema differs")
    value = cast(dict[str, Any], payload)
    errors = value["errors"]
    if (
        value["name"] != expected_name
        or type(value["correct"]) is not int
        or type(value["queries"]) is not int
        or type(value["recall_at_1"]) is not float
        or type(errors) is not list
        or value["queries"] != _QUERY_COUNT
        or value["correct"] != _QUERY_COUNT - len(errors)
        or value["recall_at_1"] != value["correct"] / _QUERY_COUNT
    ):
        raise ValueError("SigLIP checkpoint representation metrics differ")
    previous = -1
    for row in errors:
        if type(row) is not dict or set(row) != set(SiglipCheckpointErrorRow.__dataclass_fields__):
            raise ValueError("SigLIP checkpoint error schema differs")
        error = cast(dict[str, Any], row)
        query = error["query_position"]
        nearest = error["nearest_position"]
        if (
            type(query) is not int
            or type(nearest) is not int
            or not 0 <= query < _QUERY_COUNT
            or not 0 <= nearest < _QUERY_COUNT
            or query <= previous
            or query == nearest
            or type(error["query_label"]) is not int
            or type(error["nearest_label"]) is not int
            or error["query_label"] != labels[query]
            or error["nearest_label"] != labels[nearest]
            or error["query_label"] == error["nearest_label"]
            or error["query_example_id"] != example_ids[query]
            or error["nearest_example_id"] != example_ids[nearest]
        ):
            raise ValueError("SigLIP checkpoint error binding differs")
        previous = query


def validate_siglip_checkpoint_audit_bytes(
    raw: bytes,
    *,
    expected_authority: SiglipCheckpointAuditAuthority,
    expected_example_ids: Sequence[str],
    expected_labels: Sequence[int],
) -> None:
    """Reject any byte, schema, type, metric, order, or identity drift."""

    if type(raw) is not bytes:
        raise TypeError("SigLIP checkpoint result must be concrete bytes")
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("SigLIP checkpoint result is not JSON") from error
    if type(payload) is not dict or raw != _canonical(payload):
        raise ValueError("SigLIP checkpoint result is not canonical")
    if set(payload) != {
        "authority",
        "claim_eligible",
        "official_test_access",
        "projected",
        "raw",
        "schema",
    }:
        raise ValueError("SigLIP checkpoint result schema differs")
    if (
        payload["schema"] != _SCHEMA
        or payload["claim_eligible"] is not False
        or payload["official_test_access"] is not False
        or type(payload["authority"]) is not dict
    ):
        raise ValueError("SigLIP checkpoint result claim authority differs")
    try:
        authority = SiglipCheckpointAuditAuthority(**payload["authority"])
    except TypeError as error:
        raise ValueError("SigLIP checkpoint authority schema differs") from error
    _validate_examples(expected_example_ids, expected_labels)
    _validate_authority(
        authority,
        expected_example_ids=expected_example_ids,
        expected_labels=expected_labels,
    )
    if authority != expected_authority:
        raise ValueError("SigLIP checkpoint result authority differs")
    _parse_representation(
        payload["raw"],
        expected_name="raw",
        example_ids=expected_example_ids,
        labels=expected_labels,
    )
    _parse_representation(
        payload["projected"],
        expected_name="projected",
        example_ids=expected_example_ids,
        labels=expected_labels,
    )
