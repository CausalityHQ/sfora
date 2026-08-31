"""Pure authority and result logic for the claim-ineligible SAGA diagnostic."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath

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


@dataclass(frozen=True, slots=True)
class SnapshotAuthority:
    """Authenticated immutable local model snapshot."""

    repository_id: str
    model_revision: str
    processor_revision: str
    tokenizer_revision: str
    snapshot_tree_sha256: str
    architecture: str
    dtype: str
    attention_backend: str
    files: tuple[ObjectAuthority, ...]


@dataclass(frozen=True, slots=True)
class FixtureAuthority:
    """Authenticated quality-blind synthetic compute fixture."""

    source_commit: str
    model_revision: str
    binary_sha256: str
    environment_sha256: str
    host: str
    group_size: int
    image_count: int
    generation_seeds: tuple[int, ...]
    synthetic_rewards: tuple[int, ...]
    attention_layer: int
    prompt_sha256: str
    message_serialization_sha256: str


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


def _observe_regular_file(root: Path, authority: ObjectAuthority) -> ObjectAuthority:
    authority.validated()
    path = root / authority.relative_path
    if path.is_symlink() or not path.is_file():
        raise ValueError("SAGA snapshot regular-file authority differs")
    payload = path.read_bytes()
    return ObjectAuthority(
        role=authority.role,
        relative_path=authority.relative_path,
        byte_length=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def load_snapshot_authority(*, root: Path, manifest_path: Path) -> SnapshotAuthority:
    """Authenticate every byte and path in one already-local model snapshot."""

    root = root.resolve(strict=True)
    if not root.is_dir() or manifest_path.is_symlink():
        raise ValueError("SAGA snapshot root authority differs")
    manifest = parse_canonical_object(
        manifest_path.read_bytes(), role="SAGA snapshot manifest"
    )
    keys = {
        "schema",
        "repository_id",
        "model_revision",
        "processor_revision",
        "tokenizer_revision",
        "snapshot_tree_sha256",
        "architecture",
        "dtype",
        "attention_backend",
        "trust_remote_code",
        "files",
    }
    if set(manifest) != keys or manifest["schema"] != "sfora-saga-snapshot-v1":
        raise ValueError("SAGA snapshot manifest schema differs")
    if manifest["trust_remote_code"] is not False:
        raise ValueError("SAGA snapshot remote-code authority differs")

    rows = manifest["files"]
    if type(rows) is not list or not rows:
        raise ValueError("SAGA snapshot file authority differs")
    registered = tuple(ObjectAuthority.from_mapping(row) for row in rows)
    relative_paths = tuple(row.relative_path for row in registered)
    if relative_paths != tuple(sorted(relative_paths)) or len(set(relative_paths)) != len(
        relative_paths
    ):
        raise ValueError("SAGA snapshot file ordering differs")
    if tuple(_observe_regular_file(root, row) for row in registered) != registered:
        raise ValueError("SAGA snapshot bytes differ from authority")

    observed_paths: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("SAGA snapshot symbolic links are forbidden")
        if path.is_file():
            observed_paths.add(path.relative_to(root).as_posix())
        elif not path.is_dir():
            raise ValueError("SAGA snapshot special files are forbidden")
    if observed_paths != set(relative_paths):
        raise ValueError("SAGA snapshot file set differs from authority")

    tree_sha256 = hashlib.sha256(
        canonical_json_bytes({"files": [row.to_mapping() for row in registered]})
    ).hexdigest()
    if manifest["snapshot_tree_sha256"] != tree_sha256:
        raise ValueError("SAGA snapshot tree digest differs")

    repository_id = _require_nonempty_string(
        manifest["repository_id"], name="snapshot repository"
    )
    model_revision = _require_commit(manifest["model_revision"], name="model revision")
    processor_revision = _require_commit(
        manifest["processor_revision"], name="processor revision"
    )
    tokenizer_revision = _require_commit(
        manifest["tokenizer_revision"], name="tokenizer revision"
    )
    snapshot_tree_sha256 = _require_sha256(
        manifest["snapshot_tree_sha256"], name="snapshot tree digest"
    )
    architecture = _require_nonempty_string(
        manifest["architecture"], name="snapshot architecture"
    )
    dtype = _require_nonempty_string(manifest["dtype"], name="snapshot dtype")
    attention_backend = _require_nonempty_string(
        manifest["attention_backend"], name="snapshot attention backend"
    )
    if repository_id != "Qwen/Qwen3-VL-8B-Instruct":
        raise ValueError("SAGA snapshot repository differs")
    if architecture != "Qwen3VLForConditionalGeneration":
        raise ValueError("SAGA snapshot architecture differs")
    if dtype != "bfloat16" or attention_backend not in {"eager", "sdpa"}:
        raise ValueError("SAGA snapshot backend authority differs")
    return SnapshotAuthority(
        repository_id=repository_id,
        model_revision=model_revision,
        processor_revision=processor_revision,
        tokenizer_revision=tokenizer_revision,
        snapshot_tree_sha256=snapshot_tree_sha256,
        architecture=architecture,
        dtype=dtype,
        attention_backend=attention_backend,
        files=registered,
    )


def _generated_image_bytes(source_commit: str, ordinal: int) -> bytes:
    output = bytearray()
    counter = 0
    while len(output) < 224 * 224 * 3:
        output.extend(
            hashlib.sha256(
                source_commit.encode("ascii")
                + ordinal.to_bytes(4, "little")
                + counter.to_bytes(4, "little")
            ).digest()
        )
        counter += 1
    return bytes(output[: 224 * 224 * 3])


def _require_exact_integer_list(
    value: object, *, expected: list[int], name: str
) -> tuple[int, ...]:
    if type(value) is not list or any(type(item) is not int for item in value):
        raise ValueError(f"SAGA fixture {name} differs")
    if value != expected:
        raise ValueError(f"SAGA fixture {name} differs")
    return tuple(value)


def load_fixture_authority(path: Path) -> FixtureAuthority:
    """Authenticate the quality-blind synthetic fixture without external files."""

    if path.is_symlink() or not path.is_file():
        raise ValueError("SAGA fixture path differs")
    manifest = parse_canonical_object(path.read_bytes(), role="SAGA fixture")
    keys = {
        "schema",
        "source_commit",
        "model_revision",
        "binary_sha256",
        "environment_sha256",
        "host",
        "image_width",
        "image_height",
        "image_sha256",
        "pair_ordinals",
        "microbatch_ordinals",
        "prompt_utf8",
        "prompt_sha256",
        "message_serialization_sha256",
        "group_size",
        "temperature_ppm",
        "top_p_ppm",
        "max_new_tokens",
        "generation_seeds",
        "synthetic_rewards",
        "attention_layer",
        "pseudo_labels",
    }
    if set(manifest) != keys or manifest["schema"] != "sfora-saga-synthetic-fixture-v1":
        raise ValueError("SAGA fixture schema differs")
    source_commit = _require_commit(manifest["source_commit"], name="fixture source")
    model_revision = _require_commit(
        manifest["model_revision"], name="fixture model revision"
    )
    binary_sha256 = _require_sha256(
        manifest["binary_sha256"], name="fixture binary digest"
    )
    environment_sha256 = _require_sha256(
        manifest["environment_sha256"], name="fixture environment digest"
    )
    host = _require_nonempty_string(manifest["host"], name="fixture host")
    if manifest["image_width"] != 224 or type(manifest["image_width"]) is not int:
        raise ValueError("SAGA fixture image dimensions differ")
    if manifest["image_height"] != 224 or type(manifest["image_height"]) is not int:
        raise ValueError("SAGA fixture image dimensions differ")
    image_sha256 = manifest["image_sha256"]
    if type(image_sha256) is not list or len(image_sha256) != 64:
        raise ValueError("SAGA fixture image digests differ")
    expected_image_sha256 = [
        hashlib.sha256(_generated_image_bytes(source_commit, ordinal)).hexdigest()
        for ordinal in range(64)
    ]
    if image_sha256 != expected_image_sha256:
        raise ValueError("SAGA fixture image digests differ")
    _require_exact_integer_list(
        manifest["pair_ordinals"], expected=[0, 1], name="pair ordinals"
    )
    _require_exact_integer_list(
        manifest["microbatch_ordinals"],
        expected=list(range(64)),
        name="microbatch ordinals",
    )
    prompt = _require_nonempty_string(manifest["prompt_utf8"], name="fixture prompt")
    if manifest["prompt_sha256"] != hashlib.sha256(prompt.encode()).hexdigest():
        raise ValueError("SAGA fixture prompt digest differs")
    prompt_sha256 = _require_sha256(
        manifest["prompt_sha256"], name="fixture prompt digest"
    )
    message_serialization_sha256 = _require_sha256(
        manifest["message_serialization_sha256"],
        name="fixture message serialization digest",
    )
    fixed_integer_fields = {
        "group_size": 8,
        "temperature_ppm": 700_000,
        "top_p_ppm": 950_000,
        "max_new_tokens": 1024,
        "attention_layer": 26,
    }
    for name, expected in fixed_integer_fields.items():
        if type(manifest[name]) is not int or manifest[name] != expected:
            raise ValueError(f"SAGA fixture {name} differs")
    generation_seeds = _require_exact_integer_list(
        manifest["generation_seeds"],
        expected=list(range(8)),
        name="generation seeds",
    )
    synthetic_rewards = _require_exact_integer_list(
        manifest["synthetic_rewards"],
        expected=[0, 1, 0, 1, 0, 1, 0, 1],
        name="synthetic rewards",
    )
    pseudo_labels = _require_exact_integer_list(
        manifest["pseudo_labels"],
        expected=[ordinal % 2 for ordinal in range(64)],
        name="pseudo labels",
    )
    if pseudo_labels.count(0) != 32 or pseudo_labels.count(1) != 32:
        raise ValueError("SAGA fixture pseudo-label balance differs")
    return FixtureAuthority(
        source_commit=source_commit,
        model_revision=model_revision,
        binary_sha256=binary_sha256,
        environment_sha256=environment_sha256,
        host=host,
        group_size=8,
        image_count=64,
        generation_seeds=generation_seeds,
        synthetic_rewards=synthetic_rewards,
        attention_layer=26,
        prompt_sha256=prompt_sha256,
        message_serialization_sha256=message_serialization_sha256,
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
