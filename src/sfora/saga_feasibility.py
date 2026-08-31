"""Pure authority and result logic for the claim-ineligible SAGA diagnostic."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import PurePosixPath

from sfora.pass209_m4 import canonical_json_bytes

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_PHASE_NAMES = ("load", "rollout", "replay", "attention", "dml")


class FeasibilityOutcome(StrEnum):
    """Exhaustive outcome of one SAGA GB10 feasibility attempt."""

    FITS = "FITS"
    MEMORY_FAIL = "MEMORY_FAIL"
    ATTENTION_UNAVAILABLE = "ATTENTION_UNAVAILABLE"
    TIME_BUDGET_FAIL = "TIME_BUDGET_FAIL"
    DETERMINISM_FAIL = "DETERMINISM_FAIL"
    BACKEND_INVALID = "BACKEND_INVALID"
    AUTHORITY_INVALID = "AUTHORITY_INVALID"


def _require_nonempty_string(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"SAGA {name} authority differs")
    return value


def _require_positive_integer(value: object, *, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"SAGA {name} authority differs")
    return value


def _require_sha256(value: object, *, name: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"SAGA {name} authority differs")
    return value


def _require_commit(value: object, *, name: str) -> str:
    if type(value) is not str or _COMMIT_RE.fullmatch(value) is None:
        raise ValueError(f"SAGA {name} authority differs")
    return value


@dataclass(frozen=True, slots=True)
class ObjectAuthority:
    """Content and path authority for one local diagnostic object."""

    role: str
    relative_path: str
    byte_length: int
    sha256: str

    def validated(self) -> ObjectAuthority:
        _require_nonempty_string(self.role, name="object authority role")
        path = _require_nonempty_string(
            self.relative_path, name="object authority relative path"
        )
        parsed = PurePosixPath(path)
        if parsed.is_absolute() or ".." in parsed.parts or str(parsed) != path:
            raise ValueError("SAGA object authority relative path differs")
        _require_positive_integer(
            self.byte_length, name="object authority byte length"
        )
        _require_sha256(self.sha256, name="object authority digest")
        return self

    @classmethod
    def from_mapping(cls, value: object) -> ObjectAuthority:
        if type(value) is not dict or set(value) != {
            "role",
            "relative_path",
            "byte_length",
            "sha256",
        }:
            raise ValueError("SAGA object authority schema differs")
        authority = cls(
            role=value["role"],  # type: ignore[arg-type]
            relative_path=value["relative_path"],  # type: ignore[arg-type]
            byte_length=value["byte_length"],  # type: ignore[arg-type]
            sha256=value["sha256"],  # type: ignore[arg-type]
        )
        return authority.validated()

    def to_mapping(self) -> dict[str, object]:
        self.validated()
        return {
            "role": self.role,
            "relative_path": self.relative_path,
            "byte_length": self.byte_length,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class ResourceEnvelope:
    """Frozen controller resource limits."""

    cuda_reserved_limit_bytes: int
    rss_limit_bytes: int
    wall_limit_ns: int
    progress_limit_ns: int

    def to_mapping(self) -> dict[str, object]:
        for name in self.__dataclass_fields__:
            _require_positive_integer(getattr(self, name), name=f"resource {name}")
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }


@dataclass(frozen=True, slots=True)
class PhaseMeasurement:
    """Minimal backend-independent evidence for one completed phase."""

    name: str
    completed: bool
    elapsed_ns: int
    peak_cuda_reserved_bytes: int
    peak_rss_bytes: int

    def to_mapping(self, *, expected_name: str) -> dict[str, object]:
        if self.name != expected_name or self.completed is not True:
            raise ValueError("SAGA phase evidence differs")
        _require_positive_integer(self.elapsed_ns, name="phase elapsed time")
        for name in ("peak_cuda_reserved_bytes", "peak_rss_bytes"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError("SAGA phase evidence differs")
        return {
            "name": self.name,
            "completed": self.completed,
            "elapsed_ns": self.elapsed_ns,
            "peak_cuda_reserved_bytes": self.peak_cuda_reserved_bytes,
            "peak_rss_bytes": self.peak_rss_bytes,
        }


@dataclass(frozen=True, slots=True)
class FeasibilityEvidence:
    """Pure evidence consumed by the canonical feasibility serializer."""

    source_commit: str
    controller_commit: str
    binary_sha256: str
    environment_sha256: str
    host: str
    model: ObjectAuthority
    fixture: ObjectAuthority
    envelope: ResourceEnvelope
    load: PhaseMeasurement
    rollout: PhaseMeasurement
    replay: PhaseMeasurement
    attention: PhaseMeasurement
    dml: PhaseMeasurement
    deterministic: bool
    attention_available: bool
    backend_valid: bool
    authority_valid: bool
    memory_within_envelope: bool
    time_within_envelope: bool
    dataset_reads: int
    label_reads: int
    evaluation_reads: int
    optimizer_steps: int


def parse_canonical_object(raw: bytes, *, role: str) -> dict[str, object]:
    """Parse one exact sorted compact newline-terminated JSON object."""

    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{role} is not canonical JSON") from error
    if type(value) is not dict or canonical_json_bytes(value) != raw:
        raise ValueError(f"{role} is not canonical JSON")
    return value


def project_best_case_step_ns(
    *,
    dml_microbatch_ns: int,
    rollout_group_ns: int,
    replay_pair_ns: int,
    attention_pair_ns: int,
) -> int:
    """Project the frozen one-DML-plus-eight-pair feasibility step."""

    values = (
        dml_microbatch_ns,
        rollout_group_ns,
        replay_pair_ns,
        attention_pair_ns,
    )
    if any(type(value) is not int or value <= 0 for value in values):
        raise ValueError("SAGA feasibility timing authority differs")
    return dml_microbatch_ns + 8 * (
        rollout_group_ns + replay_pair_ns + attention_pair_ns
    )


def _outcome(evidence: FeasibilityEvidence) -> tuple[FeasibilityOutcome, str]:
    flags = (
        (evidence.authority_valid, FeasibilityOutcome.AUTHORITY_INVALID, "authority"),
        (evidence.backend_valid, FeasibilityOutcome.BACKEND_INVALID, "backend"),
        (evidence.deterministic, FeasibilityOutcome.DETERMINISM_FAIL, "determinism"),
        (
            evidence.memory_within_envelope,
            FeasibilityOutcome.MEMORY_FAIL,
            "memory",
        ),
        (
            evidence.attention_available,
            FeasibilityOutcome.ATTENTION_UNAVAILABLE,
            "attention",
        ),
        (
            evidence.time_within_envelope,
            FeasibilityOutcome.TIME_BUDGET_FAIL,
            "time-budget",
        ),
    )
    for value, outcome, clause in flags:
        if type(value) is not bool:
            raise ValueError("SAGA outcome evidence differs")
        if not value:
            return outcome, clause
    return FeasibilityOutcome.FITS, "all-feasibility-gates"


def canonical_feasibility_result_bytes(evidence: FeasibilityEvidence) -> bytes:
    """Validate and serialize one claim-ineligible feasibility result."""

    if type(evidence) is not FeasibilityEvidence:
        raise ValueError("SAGA feasibility evidence schema differs")
    _require_commit(evidence.source_commit, name="source commit")
    _require_commit(evidence.controller_commit, name="controller commit")
    _require_sha256(evidence.binary_sha256, name="binary digest")
    _require_sha256(evidence.environment_sha256, name="environment digest")
    _require_nonempty_string(evidence.host, name="host")

    counters = (
        evidence.dataset_reads,
        evidence.label_reads,
        evidence.evaluation_reads,
        evidence.optimizer_steps,
    )
    if any(type(value) is not int or value != 0 for value in counters):
        raise ValueError("SAGA capability counters differ")

    phases = tuple(
        getattr(evidence, name).to_mapping(expected_name=name) for name in _PHASE_NAMES
    )
    outcome, decisive_clause = _outcome(evidence)
    best_case_step_ns = project_best_case_step_ns(
        dml_microbatch_ns=evidence.dml.elapsed_ns,
        rollout_group_ns=evidence.rollout.elapsed_ns,
        replay_pair_ns=evidence.replay.elapsed_ns,
        attention_pair_ns=evidence.attention.elapsed_ns,
    )
    if not math.isfinite(float(best_case_step_ns)):
        raise ValueError("SAGA projection differs")

    payload: dict[str, object] = {
        "schema": "sfora-saga-gb10-feasibility-result-v1",
        "claim_eligible": False,
        "outcome": outcome.value,
        "decisive_clause": decisive_clause,
        "source_commit": evidence.source_commit,
        "controller_commit": evidence.controller_commit,
        "binary_sha256": evidence.binary_sha256,
        "environment_sha256": evidence.environment_sha256,
        "host": evidence.host,
        "objects": [evidence.model.to_mapping(), evidence.fixture.to_mapping()],
        "resource_envelope": evidence.envelope.to_mapping(),
        "phases": list(phases),
        "best_case_step_ns": best_case_step_ns,
        "dataset_reads": evidence.dataset_reads,
        "label_reads": evidence.label_reads,
        "evaluation_reads": evidence.evaluation_reads,
        "optimizer_steps": evidence.optimizer_steps,
        "quality_metrics": [],
    }
    payload["result_sha256"] = hashlib.sha256(
        canonical_json_bytes(payload)
    ).hexdigest()
    return canonical_json_bytes(payload)
