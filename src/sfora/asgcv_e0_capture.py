"""Canonical phase authority for the claim-ineligible ASG-CV E0 capture."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import cast

from sfora.asgcv_protocol import AsgcvPartitionAuthority, AsgcvRolloutAuthority

ASGCV_E0_CAPTURE_MANIFEST_SCHEMA = "sfora-asgcv-e0-capture-manifest-v1"
ASGCV_E0_PHASE_RECEIPT_SCHEMA = "sfora-asgcv-e0-phase-receipt-v1"
ASGCV_E0_PHASES = ("eligibility", "capture", "fit", "evaluate")


def _canonical_json_bytes(value: dict[str, object]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")


def _sha256(value: object, *, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"ASG-CV E0 {name} differs")
    return value


def _commit(value: object, *, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"ASG-CV E0 {name} differs")
    return value


def _digest_list(value: object, *, name: str) -> tuple[str, ...]:
    if type(value) not in {tuple, list} or not value:
        raise ValueError(f"ASG-CV E0 {name} differs")
    sequence = cast(tuple[object, ...] | list[object], value)
    digests = tuple(_sha256(item, name=name) for item in sequence)
    if digests != tuple(sorted(set(digests))):
        raise ValueError(f"ASG-CV E0 {name} differs")
    return digests


@dataclass(frozen=True, slots=True)
class AsgcvE0CaptureManifest:
    """Frozen identities available before any exact-gradient capture."""

    source_commit: str
    dataset_manifest_sha256: str
    partition_authority: AsgcvPartitionAuthority
    rollout_authority: AsgcvRolloutAuthority
    predictor_train_candidate_schedule_sha256: str
    predictor_train_eligible_schedule_sha256: str
    e0_validation_candidate_schedule_sha256: str
    e0_validation_eligible_schedule_sha256: str
    model_revision: str
    fixture_sha256: str
    pooler_state_sha256: str

    def validated(self) -> AsgcvE0CaptureManifest:
        _commit(self.source_commit, name="capture source commit")
        dataset = _sha256(self.dataset_manifest_sha256, name="dataset manifest digest")
        if type(self.partition_authority) is not AsgcvPartitionAuthority:
            raise ValueError("ASG-CV E0 partition authority differs")
        self.partition_authority.validated()
        if self.partition_authority.source_manifest_sha256 != dataset:
            raise ValueError("ASG-CV E0 dataset/partition binding differs")
        if type(self.rollout_authority) is not AsgcvRolloutAuthority:
            raise ValueError("ASG-CV E0 rollout authority differs")
        self.rollout_authority.validated()
        revision = _commit(self.model_revision, name="capture model revision")
        if self.rollout_authority.model_revision != revision:
            raise ValueError("ASG-CV E0 rollout model binding differs")
        schedule_names = (
            "predictor_train_candidate_schedule_sha256",
            "predictor_train_eligible_schedule_sha256",
            "e0_validation_candidate_schedule_sha256",
            "e0_validation_eligible_schedule_sha256",
        )
        schedules = tuple(
            _sha256(getattr(self, name), name=f"{name} digest") for name in schedule_names
        )
        if len(set(schedules)) != len(schedules):
            raise ValueError("ASG-CV E0 phase schedule identities overlap")
        _sha256(self.fixture_sha256, name="fixture digest")
        _sha256(self.pooler_state_sha256, name="pooler-state digest")
        return self

    def to_mapping(self) -> dict[str, object]:
        self.validated()
        return {
            "schema": ASGCV_E0_CAPTURE_MANIFEST_SCHEMA,
            "claim_eligible": False,
            "official_test_access": False,
            "source_commit": self.source_commit,
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "partition_authority": self.partition_authority.to_mapping(),
            "rollout_authority": self.rollout_authority.to_mapping(),
            "predictor_train_candidate_schedule_sha256": (
                self.predictor_train_candidate_schedule_sha256
            ),
            "predictor_train_eligible_schedule_sha256": (
                self.predictor_train_eligible_schedule_sha256
            ),
            "e0_validation_candidate_schedule_sha256": (
                self.e0_validation_candidate_schedule_sha256
            ),
            "e0_validation_eligible_schedule_sha256": (self.e0_validation_eligible_schedule_sha256),
            "model_revision": self.model_revision,
            "fixture_sha256": self.fixture_sha256,
            "pooler_state_sha256": self.pooler_state_sha256,
        }


def canonical_capture_manifest_bytes(
    *,
    source_commit: object,
    dataset_manifest_sha256: object,
    partition_authority: object,
    rollout_authority: object,
    predictor_train_candidate_schedule_sha256: object,
    predictor_train_eligible_schedule_sha256: object,
    e0_validation_candidate_schedule_sha256: object,
    e0_validation_eligible_schedule_sha256: object,
    model_revision: object,
    fixture_sha256: object,
    pooler_state_sha256: object,
    official_test_access: object,
) -> bytes:
    """Serialize one frozen, train-only E0 capture manifest."""

    if official_test_access is not False:
        raise ValueError("ASG-CV E0 official-test access differs")
    if (
        type(partition_authority) is not AsgcvPartitionAuthority
        or type(rollout_authority) is not AsgcvRolloutAuthority
    ):
        raise ValueError("ASG-CV E0 capture authority type differs")
    manifest = AsgcvE0CaptureManifest(
        source_commit=_commit(source_commit, name="capture source commit"),
        dataset_manifest_sha256=_sha256(
            dataset_manifest_sha256,
            name="dataset manifest digest",
        ),
        partition_authority=partition_authority,
        rollout_authority=rollout_authority,
        predictor_train_candidate_schedule_sha256=_sha256(
            predictor_train_candidate_schedule_sha256,
            name="predictor training candidate schedule digest",
        ),
        predictor_train_eligible_schedule_sha256=_sha256(
            predictor_train_eligible_schedule_sha256,
            name="predictor training eligible schedule digest",
        ),
        e0_validation_candidate_schedule_sha256=_sha256(
            e0_validation_candidate_schedule_sha256,
            name="E0 validation candidate schedule digest",
        ),
        e0_validation_eligible_schedule_sha256=_sha256(
            e0_validation_eligible_schedule_sha256,
            name="E0 validation eligible schedule digest",
        ),
        model_revision=_commit(model_revision, name="capture model revision"),
        fixture_sha256=_sha256(fixture_sha256, name="fixture digest"),
        pooler_state_sha256=_sha256(pooler_state_sha256, name="pooler-state digest"),
    ).validated()
    payload = manifest.to_mapping()
    payload["manifest_sha256"] = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
    return _canonical_json_bytes(payload)


def validate_capture_manifest_bytes(raw: bytes) -> dict[str, object]:
    """Validate exact bytes and every cross-binding in an E0 capture manifest."""

    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("ASG-CV E0 capture manifest is not canonical JSON") from error
    expected = {
        "schema",
        "claim_eligible",
        "official_test_access",
        "source_commit",
        "dataset_manifest_sha256",
        "partition_authority",
        "rollout_authority",
        "predictor_train_candidate_schedule_sha256",
        "predictor_train_eligible_schedule_sha256",
        "e0_validation_candidate_schedule_sha256",
        "e0_validation_eligible_schedule_sha256",
        "model_revision",
        "fixture_sha256",
        "pooler_state_sha256",
        "manifest_sha256",
    }
    if (
        type(value) is not dict
        or set(value) != expected
        or _canonical_json_bytes(value) != raw
        or value["schema"] != ASGCV_E0_CAPTURE_MANIFEST_SCHEMA
        or value["claim_eligible"] is not False
        or value["official_test_access"] is not False
    ):
        raise ValueError("ASG-CV E0 capture manifest authority differs")
    manifest = AsgcvE0CaptureManifest(
        source_commit=value["source_commit"],
        dataset_manifest_sha256=value["dataset_manifest_sha256"],
        partition_authority=AsgcvPartitionAuthority.from_mapping(value["partition_authority"]),
        rollout_authority=AsgcvRolloutAuthority.from_mapping(value["rollout_authority"]),
        predictor_train_candidate_schedule_sha256=value[
            "predictor_train_candidate_schedule_sha256"
        ],
        predictor_train_eligible_schedule_sha256=value["predictor_train_eligible_schedule_sha256"],
        e0_validation_candidate_schedule_sha256=value["e0_validation_candidate_schedule_sha256"],
        e0_validation_eligible_schedule_sha256=value["e0_validation_eligible_schedule_sha256"],
        model_revision=value["model_revision"],
        fixture_sha256=value["fixture_sha256"],
        pooler_state_sha256=value["pooler_state_sha256"],
    ).validated()
    unsigned = dict(value)
    digest = unsigned.pop("manifest_sha256")
    if (
        _sha256(digest, name="capture manifest digest")
        != hashlib.sha256(_canonical_json_bytes(unsigned)).hexdigest()
        or manifest.to_mapping() != unsigned
    ):
        raise ValueError("ASG-CV E0 capture manifest digest differs")
    return value


@dataclass(frozen=True, slots=True)
class AsgcvE0PhaseReceipt:
    """One transition in the sealed eligibility→capture→fit→evaluate chain."""

    manifest_sha256: str
    phase: str
    input_sha256: tuple[str, ...]
    output_sha256: tuple[str, ...]
    previous_receipt_sha256: str | None
    elapsed_ns: int

    def validated(self) -> AsgcvE0PhaseReceipt:
        _sha256(self.manifest_sha256, name="phase manifest digest")
        if type(self.phase) is not str or self.phase not in ASGCV_E0_PHASES:
            raise ValueError("ASG-CV E0 phase differs")
        _digest_list(self.input_sha256, name="phase input digest list")
        _digest_list(self.output_sha256, name="phase output digest list")
        if self.previous_receipt_sha256 is not None:
            _sha256(self.previous_receipt_sha256, name="previous phase receipt digest")
        if type(self.elapsed_ns) is not int or self.elapsed_ns <= 0:
            raise ValueError("ASG-CV E0 phase elapsed time differs")
        return self

    def to_mapping(self) -> dict[str, object]:
        self.validated()
        return {
            "schema": ASGCV_E0_PHASE_RECEIPT_SCHEMA,
            "claim_eligible": False,
            "manifest_sha256": self.manifest_sha256,
            "phase": self.phase,
            "phase_ordinal": ASGCV_E0_PHASES.index(self.phase),
            "input_sha256": list(self.input_sha256),
            "output_sha256": list(self.output_sha256),
            "previous_receipt_sha256": self.previous_receipt_sha256,
            "elapsed_ns": self.elapsed_ns,
        }


def _parse_phase_receipt(raw: bytes) -> dict[str, object]:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("ASG-CV E0 phase receipt is not canonical JSON") from error
    expected = {
        "schema",
        "claim_eligible",
        "manifest_sha256",
        "phase",
        "phase_ordinal",
        "input_sha256",
        "output_sha256",
        "previous_receipt_sha256",
        "elapsed_ns",
        "receipt_sha256",
    }
    if (
        type(value) is not dict
        or set(value) != expected
        or _canonical_json_bytes(value) != raw
        or value["schema"] != ASGCV_E0_PHASE_RECEIPT_SCHEMA
        or value["claim_eligible"] is not False
        or type(value["phase"]) is not str
        or value["phase"] not in ASGCV_E0_PHASES
        or type(value["phase_ordinal"]) is not int
        or value["phase_ordinal"] != ASGCV_E0_PHASES.index(value["phase"])
    ):
        raise ValueError("ASG-CV E0 phase receipt authority differs")
    previous_digest = value["previous_receipt_sha256"]
    if previous_digest is not None and type(previous_digest) is not str:
        raise ValueError("ASG-CV E0 previous phase receipt digest differs")
    receipt = AsgcvE0PhaseReceipt(
        manifest_sha256=value["manifest_sha256"],
        phase=value["phase"],
        input_sha256=_digest_list(value["input_sha256"], name="phase input digest list"),
        output_sha256=_digest_list(value["output_sha256"], name="phase output digest list"),
        previous_receipt_sha256=previous_digest,
        elapsed_ns=value["elapsed_ns"],
    ).validated()
    unsigned = dict(value)
    digest = unsigned.pop("receipt_sha256")
    if (
        _sha256(digest, name="phase receipt digest")
        != hashlib.sha256(_canonical_json_bytes(unsigned)).hexdigest()
        or receipt.to_mapping() != unsigned
    ):
        raise ValueError("ASG-CV E0 phase receipt digest differs")
    return value


def _validate_phase_transition(
    current: dict[str, object],
    previous: dict[str, object] | None,
) -> None:
    phase = current["phase"]
    if phase == "eligibility":
        if previous is not None or current["previous_receipt_sha256"] is not None:
            raise ValueError("ASG-CV E0 initial phase transition differs")
        if current["input_sha256"] != [current["manifest_sha256"]]:
            raise ValueError("ASG-CV E0 initial phase inputs differ")
        return
    if previous is None:
        raise ValueError("ASG-CV E0 previous phase receipt is missing")
    if (
        current["manifest_sha256"] != previous["manifest_sha256"]
        or ASGCV_E0_PHASES.index(current["phase"]) != ASGCV_E0_PHASES.index(previous["phase"]) + 1
        or current["input_sha256"] != previous["output_sha256"]
        or current["previous_receipt_sha256"] != previous["receipt_sha256"]
    ):
        raise ValueError("ASG-CV E0 phase transition differs")


def canonical_phase_receipt_bytes(
    *,
    manifest_sha256: object,
    phase: object,
    input_sha256: object,
    output_sha256: object,
    elapsed_ns: object,
    previous_phase_receipt: bytes | None = None,
) -> bytes:
    """Serialize one phase only when it follows the exact sealed predecessor."""

    previous = (
        None if previous_phase_receipt is None else _parse_phase_receipt(previous_phase_receipt)
    )
    previous_receipt_digest = (
        None
        if previous is None
        else _sha256(previous["receipt_sha256"], name="previous phase receipt digest")
    )
    receipt = AsgcvE0PhaseReceipt(
        manifest_sha256=_sha256(manifest_sha256, name="phase manifest digest"),
        phase=phase if type(phase) is str else "",
        input_sha256=_digest_list(input_sha256, name="phase input digest list"),
        output_sha256=_digest_list(output_sha256, name="phase output digest list"),
        previous_receipt_sha256=previous_receipt_digest,
        elapsed_ns=elapsed_ns if type(elapsed_ns) is int else 0,
    ).validated()
    payload = receipt.to_mapping()
    _validate_phase_transition(payload, previous)
    payload["receipt_sha256"] = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
    return _canonical_json_bytes(payload)


def validate_phase_receipt_bytes(
    raw: bytes,
    *,
    previous_phase_receipt: bytes | None = None,
) -> dict[str, object]:
    """Validate one receipt and its exact predecessor transition."""

    current = _parse_phase_receipt(raw)
    previous = (
        None if previous_phase_receipt is None else _parse_phase_receipt(previous_phase_receipt)
    )
    _validate_phase_transition(current, previous)
    return current
