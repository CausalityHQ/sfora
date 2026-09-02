"""Train-only authority for distilling forced-verdict Qwen gradients."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, fields
from typing import Any, cast

import numpy as np

from sfora.asgcv_protocol import AsgcvPairSchedule, build_asgcv_pair_schedule

ASGCV_FORCED_DISTILL_SHAPE = (2, 196, 2048)
ASGCV_FORCED_DISTILL_TRAIN_PAIRS = 128
ASGCV_FORCED_DISTILL_VALIDATION_PAIRS = 32
ASGCV_FORCED_DISTILL_CAPTURE_SCHEMA = "sfora-asgcv-forced-distill-capture-v1"
_SCHEDULE_DOMAIN = b"sfora-asgcv-forced-distill-schedule-v1\0"


def _canonical(value: dict[str, object]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def _hex(value: object, width: int, *, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != width
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"ASG-CV forced distill {name} differs")
    return value


def _array(value: object, *, name: str) -> np.ndarray:
    if (
        type(value) is not np.ndarray
        or value.dtype != np.dtype(np.float32)
        or value.shape != ASGCV_FORCED_DISTILL_SHAPE
        or not bool(np.isfinite(value).all())
    ):
        raise ValueError(f"ASG-CV forced distill {name} differs")
    return np.ascontiguousarray(value)


def _manifest_bytes(example_ids: tuple[str, ...], labels: tuple[int, ...]) -> bytes:
    if (
        type(example_ids) is not tuple
        or type(labels) is not tuple
        or len(example_ids) != len(labels)
        or any(type(value) is not str or not value for value in example_ids)
        or any(type(value) is not int or value < 0 for value in labels)
    ):
        raise ValueError("ASG-CV forced distill manifest differs")
    return _canonical(
        {
            "examples": [
                {"example_id": example_id, "label": label}
                for example_id, label in zip(example_ids, labels, strict=True)
            ]
        }
    )


def build_forced_distill_schedule(
    example_ids: tuple[str, ...],
    labels: tuple[int, ...],
    *,
    source_commit: str,
    launch_authority_sha256: str,
    role: str,
) -> AsgcvPairSchedule:
    """Build the fixed image-disjoint train or validation pair schedule."""

    _hex(source_commit, 40, name="source commit")
    _hex(launch_authority_sha256, 64, name="launch authority")
    if type(role) is not str or role not in {"train", "validation"}:
        raise ValueError("ASG-CV forced distill role differs")
    manifest = _manifest_bytes(example_ids, labels)
    seed = hashlib.sha256(
        _SCHEDULE_DOMAIN
        + bytes.fromhex(source_commit)
        + bytes.fromhex(launch_authority_sha256)
        + role.encode("ascii")
    ).hexdigest()
    pair_count = (
        ASGCV_FORCED_DISTILL_TRAIN_PAIRS
        if role == "train"
        else ASGCV_FORCED_DISTILL_VALIDATION_PAIRS
    )
    schedule = build_asgcv_pair_schedule(
        example_ids,
        labels,
        schedule_seed_sha256=seed,
        pair_count=pair_count,
    )
    if schedule.example_manifest_sha256 != hashlib.sha256(manifest).hexdigest():
        raise ValueError("ASG-CV forced distill manifest binding differs")
    return schedule


def relation_correct_gradient(value: object, relation_sign: object) -> np.ndarray:
    """Orient a SAME-first symmetric binary-verdict gradient to the true relation."""

    gradient = _array(value, name="gradient")
    if type(relation_sign) is not int or relation_sign not in {-1, 1}:
        raise ValueError("ASG-CV forced distill relation sign differs")
    return np.ascontiguousarray(gradient * np.float32(relation_sign))


@dataclass(frozen=True, slots=True)
class ForcedDistillCapture:
    """One authenticated patch-token and relation-correct-gradient pair."""

    source_commit: str
    launch_authority_sha256: str
    schedule_sha256: str
    role: str
    pair_ordinal: int
    pair_indices: tuple[int, int]
    relation_sign: int
    patch_sha256: str
    gradient_sha256: str
    array_shape: tuple[int, int, int]

    def validated(self) -> ForcedDistillCapture:
        _hex(self.source_commit, 40, name="source commit")
        _hex(self.launch_authority_sha256, 64, name="launch authority")
        _hex(self.schedule_sha256, 64, name="schedule")
        _hex(self.patch_sha256, 64, name="patch digest")
        _hex(self.gradient_sha256, 64, name="gradient digest")
        maximum = (
            ASGCV_FORCED_DISTILL_TRAIN_PAIRS
            if self.role == "train"
            else ASGCV_FORCED_DISTILL_VALIDATION_PAIRS
            if self.role == "validation"
            else -1
        )
        if (
            maximum < 0
            or type(self.pair_ordinal) is not int
            or not 0 <= self.pair_ordinal < maximum
            or type(self.pair_indices) is not tuple
            or len(self.pair_indices) != 2
            or any(type(value) is not int or value < 0 for value in self.pair_indices)
            or self.pair_indices[0] == self.pair_indices[1]
            or type(self.relation_sign) is not int
            or self.relation_sign not in {-1, 1}
            or self.array_shape != ASGCV_FORCED_DISTILL_SHAPE
        ):
            raise ValueError("ASG-CV forced distill capture authority differs")
        return self

    def to_mapping(self) -> dict[str, object]:
        self.validated()
        return {
            "schema": ASGCV_FORCED_DISTILL_CAPTURE_SCHEMA,
            "claim_eligible": False,
            "official_test_access": False,
            **{
                field.name: list(value) if isinstance(value, tuple) else value
                for field in fields(self)
                if (value := getattr(self, field.name)) is not None
            },
        }

    @classmethod
    def from_mapping(cls, value: object) -> ForcedDistillCapture:
        expected = {field.name for field in fields(cls)} | {
            "schema",
            "claim_eligible",
            "official_test_access",
        }
        if (
            type(value) is not dict
            or set(value) != expected
            or value["schema"] != ASGCV_FORCED_DISTILL_CAPTURE_SCHEMA
            or value["claim_eligible"] is not False
            or value["official_test_access"] is not False
        ):
            raise ValueError("ASG-CV forced distill capture schema differs")
        raw = cast(dict[str, Any], value)
        try:
            return cls(
                **{
                    field.name: (
                        tuple(raw[field.name])
                        if field.name in {"pair_indices", "array_shape"}
                        and type(raw[field.name]) is list
                        else raw[field.name]
                    )
                    for field in fields(cls)
                }
            ).validated()
        except (KeyError, TypeError) as error:
            raise ValueError("ASG-CV forced distill capture differs") from error


def canonical_forced_distill_capture_bytes(value: ForcedDistillCapture) -> bytes:
    """Return sorted canonical JSON plus one trailing LF."""

    if type(value) is not ForcedDistillCapture:
        raise ValueError("ASG-CV forced distill capture differs")
    return _canonical(value.to_mapping())


def validate_forced_distill_capture_bytes(
    raw: object,
    *,
    patch_tokens: object,
    exact_gradient: object,
) -> ForcedDistillCapture:
    """Authenticate canonical receipt bytes and their exact float32 arrays."""

    if type(raw) is not bytes:
        raise ValueError("ASG-CV forced distill capture bytes differ")
    try:
        capture = ForcedDistillCapture.from_mapping(json.loads(raw))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("ASG-CV forced distill capture JSON differs") from error
    if canonical_forced_distill_capture_bytes(capture) != raw:
        raise ValueError("ASG-CV forced distill capture bytes differ")
    patches = _array(patch_tokens, name="patch tokens")
    gradient = _array(exact_gradient, name="gradient")
    if hashlib.sha256(patches.tobytes()).hexdigest() != capture.patch_sha256:
        raise ValueError("ASG-CV forced distill patch digest differs")
    if hashlib.sha256(gradient.tobytes()).hexdigest() != capture.gradient_sha256:
        raise ValueError("ASG-CV forced distill gradient digest differs")
    if not bool(np.square(gradient.astype(np.float64)).sum(dtype=np.float64) > 0.0):
        raise ValueError("ASG-CV forced distill gradient energy differs")
    return capture
