"""Train-only authority for distilling forced-verdict Qwen gradients."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from dataclasses import dataclass, fields
from typing import Any, cast

import numpy as np

from sfora.asgcv_protocol import AsgcvPairSchedule, build_asgcv_pair_schedule

ASGCV_FORCED_DISTILL_SHAPE = (2, 256, 4096)
ASGCV_FORCED_DISTILL_TRAIN_PAIRS = 128
ASGCV_FORCED_DISTILL_VALIDATION_PAIRS = 32
ASGCV_FORCED_DISTILL_CAPTURE_SCHEMA = "sfora-asgcv-forced-distill-capture-v1"
ASGCV_FORCED_DISTILL_RESULT_SCHEMA = "sfora-asgcv-forced-distill-result-v1"
ASGCV_FORCED_DISTILL_COSINE_GATE_PPM = 500_000
ASGCV_FORCED_DISTILL_POSITIVE_RATE_GATE_PPM = 750_000
ASGCV_FORCED_DISTILL_NONZERO_RATE_GATE_PPM = 1_000_000
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
    if type(role) is not str or role not in {"train", "validation", "optimization"}:
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


def dense_gradient_cosine(exact: object, predicted: object) -> float:
    """Return one float64-accumulated dense cosine over exact float32 fields."""

    result, live = dense_gradient_cosine_with_liveness(exact, predicted)
    if not live:
        raise ValueError("ASG-CV forced distill gradient energy differs")
    return result


def dense_gradient_cosine_with_liveness(exact: object, predicted: object) -> tuple[float, bool]:
    """Return cosine and record a finite zero prediction as failed liveness."""

    exact_array = _array(exact, name="exact gradient").astype(np.float64)
    predicted_array = _array(predicted, name="predicted gradient").astype(np.float64)
    exact_energy = float(np.square(exact_array).sum(dtype=np.float64))
    predicted_energy = float(np.square(predicted_array).sum(dtype=np.float64))
    if exact_energy <= 0.0:
        raise ValueError("ASG-CV forced distill gradient energy differs")
    if predicted_energy <= 0.0:
        return 0.0, False
    result = float(
        np.multiply(exact_array, predicted_array).sum(dtype=np.float64)
        / math.sqrt(exact_energy * predicted_energy)
    )
    if not math.isfinite(result) or not -1.000001 <= result <= 1.000001:
        raise ValueError("ASG-CV forced distill cosine differs")
    return min(1.0, max(-1.0, result)), True


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
            if self.role in {"validation", "optimization"}
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


@dataclass(frozen=True, slots=True)
class ForcedDistillResult:
    """Recomputed class-disjoint validation result for one frozen student."""

    capture_source_commit: str
    evaluation_source_commit: str
    launch_authority_sha256: str
    train_schedule_sha256: str
    validation_schedule_sha256: str
    predictor_state_sha256: str
    evaluation_role: str
    validation_cosines: tuple[float, ...]
    prediction_nonzero_flags: tuple[bool, ...]
    median_cosine_ppm: int
    positive_cosine_rate_ppm: int
    prediction_nonzero_rate_ppm: int
    passed: bool

    @property
    def gates_ppm(self) -> dict[str, int]:
        return {
            "median_cosine": ASGCV_FORCED_DISTILL_COSINE_GATE_PPM,
            "positive_cosine_rate": ASGCV_FORCED_DISTILL_POSITIVE_RATE_GATE_PPM,
            "prediction_nonzero_rate": ASGCV_FORCED_DISTILL_NONZERO_RATE_GATE_PPM,
        }

    @classmethod
    def from_cosines(
        cls,
        *,
        capture_source_commit: str,
        evaluation_source_commit: str,
        launch_authority_sha256: str,
        train_schedule_sha256: str,
        validation_schedule_sha256: str,
        predictor_state_sha256: str,
        evaluation_role: str,
        validation_cosines: tuple[float, ...],
        prediction_nonzero_flags: tuple[bool, ...],
    ) -> ForcedDistillResult:
        _hex(capture_source_commit, 40, name="capture source commit")
        _hex(evaluation_source_commit, 40, name="evaluation source commit")
        for name, value in (
            ("launch authority", launch_authority_sha256),
            ("train schedule", train_schedule_sha256),
            ("validation schedule", validation_schedule_sha256),
            ("predictor state", predictor_state_sha256),
        ):
            _hex(value, 64, name=name)
        if evaluation_role not in {"e0_validation", "e1_optimization"}:
            raise ValueError("ASG-CV forced distill evaluation role differs")
        if (
            type(validation_cosines) is not tuple
            or len(validation_cosines) != ASGCV_FORCED_DISTILL_VALIDATION_PAIRS
            or any(
                type(value) is not float or not math.isfinite(value) or not -1.0 <= value <= 1.0
                for value in validation_cosines
            )
        ):
            raise ValueError("ASG-CV forced distill validation cosines differ")
        if (
            type(prediction_nonzero_flags) is not tuple
            or len(prediction_nonzero_flags) != ASGCV_FORCED_DISTILL_VALIDATION_PAIRS
            or any(type(value) is not bool for value in prediction_nonzero_flags)
        ):
            raise ValueError("ASG-CV forced distill prediction liveness differs")
        median = int(round(statistics.median(validation_cosines) * 1_000_000))
        positive_rate = int(
            round(
                sum(value > 0.0 for value in validation_cosines)
                * 1_000_000
                / len(validation_cosines)
            )
        )
        nonzero_rate = int(
            round(sum(prediction_nonzero_flags) * 1_000_000 / len(prediction_nonzero_flags))
        )
        return cls(
            capture_source_commit=capture_source_commit,
            evaluation_source_commit=evaluation_source_commit,
            launch_authority_sha256=launch_authority_sha256,
            train_schedule_sha256=train_schedule_sha256,
            validation_schedule_sha256=validation_schedule_sha256,
            predictor_state_sha256=predictor_state_sha256,
            evaluation_role=evaluation_role,
            validation_cosines=validation_cosines,
            prediction_nonzero_flags=prediction_nonzero_flags,
            median_cosine_ppm=median,
            positive_cosine_rate_ppm=positive_rate,
            prediction_nonzero_rate_ppm=nonzero_rate,
            passed=(
                median >= ASGCV_FORCED_DISTILL_COSINE_GATE_PPM
                and positive_rate >= ASGCV_FORCED_DISTILL_POSITIVE_RATE_GATE_PPM
                and nonzero_rate >= ASGCV_FORCED_DISTILL_NONZERO_RATE_GATE_PPM
            ),
        )

    def validated(self) -> ForcedDistillResult:
        for name in (
            "launch_authority_sha256",
            "train_schedule_sha256",
            "validation_schedule_sha256",
            "predictor_state_sha256",
        ):
            _hex(getattr(self, name), 64, name=name.replace("_", " "))
        _hex(self.capture_source_commit, 40, name="capture source commit")
        _hex(self.evaluation_source_commit, 40, name="evaluation source commit")
        recomputed = type(self).from_cosines(
            capture_source_commit=self.capture_source_commit,
            evaluation_source_commit=self.evaluation_source_commit,
            launch_authority_sha256=self.launch_authority_sha256,
            train_schedule_sha256=self.train_schedule_sha256,
            validation_schedule_sha256=self.validation_schedule_sha256,
            predictor_state_sha256=self.predictor_state_sha256,
            evaluation_role=self.evaluation_role,
            validation_cosines=self.validation_cosines,
            prediction_nonzero_flags=self.prediction_nonzero_flags,
        )
        if (
            self.median_cosine_ppm != recomputed.median_cosine_ppm
            or self.positive_cosine_rate_ppm != recomputed.positive_cosine_rate_ppm
            or self.prediction_nonzero_rate_ppm != recomputed.prediction_nonzero_rate_ppm
            or self.passed is not recomputed.passed
        ):
            raise ValueError("ASG-CV forced distill result metrics differ")
        return self

    def to_mapping(self) -> dict[str, object]:
        self.validated()
        return {
            "schema": ASGCV_FORCED_DISTILL_RESULT_SCHEMA,
            "claim_eligible": False,
            "official_test_access": False,
            "gates_ppm": self.gates_ppm,
            **{
                field.name: list(value) if isinstance(value, tuple) else value
                for field in fields(self)
                if (value := getattr(self, field.name)) is not None
            },
        }

    @classmethod
    def from_mapping(cls, value: object) -> ForcedDistillResult:
        expected = {field.name for field in fields(cls)} | {
            "schema",
            "claim_eligible",
            "official_test_access",
            "gates_ppm",
        }
        if (
            type(value) is not dict
            or set(value) != expected
            or value["schema"] != ASGCV_FORCED_DISTILL_RESULT_SCHEMA
            or value["claim_eligible"] is not False
            or value["official_test_access"] is not False
            or value["gates_ppm"]
            != {
                "median_cosine": ASGCV_FORCED_DISTILL_COSINE_GATE_PPM,
                "positive_cosine_rate": ASGCV_FORCED_DISTILL_POSITIVE_RATE_GATE_PPM,
                "prediction_nonzero_rate": ASGCV_FORCED_DISTILL_NONZERO_RATE_GATE_PPM,
            }
        ):
            raise ValueError("ASG-CV forced distill result schema differs")
        raw = cast(dict[str, Any], value)
        try:
            result = cls(
                **{
                    field.name: (
                        tuple(raw[field.name])
                        if field.name in {"validation_cosines", "prediction_nonzero_flags"}
                        and type(raw[field.name]) is list
                        else raw[field.name]
                    )
                    for field in fields(cls)
                }
            )
        except (KeyError, TypeError) as error:
            raise ValueError("ASG-CV forced distill result differs") from error
        expected_result = cls.from_cosines(
            capture_source_commit=result.capture_source_commit,
            evaluation_source_commit=result.evaluation_source_commit,
            launch_authority_sha256=result.launch_authority_sha256,
            train_schedule_sha256=result.train_schedule_sha256,
            validation_schedule_sha256=result.validation_schedule_sha256,
            predictor_state_sha256=result.predictor_state_sha256,
            evaluation_role=result.evaluation_role,
            validation_cosines=result.validation_cosines,
            prediction_nonzero_flags=result.prediction_nonzero_flags,
        )
        if result != expected_result:
            raise ValueError("ASG-CV forced distill result metrics differ")
        return result


def canonical_forced_distill_result_bytes(value: ForcedDistillResult) -> bytes:
    """Serialize one fully recomputed result."""

    if type(value) is not ForcedDistillResult:
        raise ValueError("ASG-CV forced distill result differs")
    return _canonical(value.to_mapping())


def validate_forced_distill_result_bytes(raw: object) -> ForcedDistillResult:
    """Reopen only exact canonical result bytes."""

    if type(raw) is not bytes:
        raise ValueError("ASG-CV forced distill result bytes differ")
    try:
        result = ForcedDistillResult.from_mapping(json.loads(raw))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("ASG-CV forced distill result JSON differs") from error
    if canonical_forced_distill_result_bytes(result) != raw:
        raise ValueError("ASG-CV forced distill result bytes differ")
    return result
