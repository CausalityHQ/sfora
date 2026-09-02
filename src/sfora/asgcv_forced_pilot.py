"""Authority and hard gates for the train-only forced-verdict SAGA pilot."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, fields
from typing import Any, cast

ASGCV_FORCED_PAIR_COUNT = 32
ASGCV_FORCED_ACCURACY_GATE_PPM = 625_000
ASGCV_FORCED_AUC_GATE_PPM = 700_000
ASGCV_FORCED_SIGN_RECALL_GATE_PPM = 500_000
ASGCV_FORCED_OBSERVATION_SCHEMA = "sfora-asgcv-forced-observation-v1"
ASGCV_FORCED_RESULT_SCHEMA = "sfora-asgcv-forced-result-v1"
ASGCV_FORCED_BRANCH_ORDER = ("same", "different")
ASGCV_FORCED_REPEAT_ORDINALS = (0, 31)


def _canonical_bytes(value: dict[str, object]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def _hex(value: object, width: int, *, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != width
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"ASG-CV forced {name} differs")
    return value


def _finite(value: object, *, name: str, positive: bool = False) -> float:
    if type(value) is not float or not math.isfinite(value) or (positive and value <= 0.0):
        raise ValueError(f"ASG-CV forced {name} differs")
    return value


def _positive_int(value: object, *, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"ASG-CV forced {name} differs")
    return value


def _ppm(numerator: float, denominator: int) -> int:
    if denominator <= 0:
        raise ValueError("ASG-CV forced rate denominator differs")
    return int(round(numerator * 1_000_000.0 / denominator))


@dataclass(frozen=True, slots=True)
class AsgcvForcedObservation:
    """One fixed SAME-before-DIFFERENT teacher-forced gradient observation."""

    source_commit: str
    launch_authority_sha256: str
    pilot_schedule_sha256: str
    model_revision: str
    fixture_sha256: str
    candidate_pair_ordinal: int
    pair_ordinals: tuple[int, int]
    relation_sign: int
    same_score: float
    different_score: float
    gradient_sha256: str
    gradient_norm: float
    boundary_norms: tuple[float, float, float, float]
    prepare_elapsed_ns: int
    replay_elapsed_ns: int
    peak_cuda_reserved_bytes: int
    peak_rss_bytes: int

    @property
    def score_gap(self) -> float:
        return self.same_score - self.different_score

    @property
    def correct(self) -> bool:
        return self.score_gap > 0.0 if self.relation_sign == 1 else self.score_gap < 0.0

    def validated(self) -> AsgcvForcedObservation:
        _hex(self.source_commit, 40, name="source commit")
        _hex(self.launch_authority_sha256, 64, name="launch authority")
        _hex(self.pilot_schedule_sha256, 64, name="pilot schedule")
        _hex(self.model_revision, 40, name="model revision")
        _hex(self.fixture_sha256, 64, name="fixture")
        _hex(self.gradient_sha256, 64, name="gradient")
        if (
            type(self.candidate_pair_ordinal) is not int
            or not 0 <= self.candidate_pair_ordinal < ASGCV_FORCED_PAIR_COUNT
            or type(self.pair_ordinals) is not tuple
            or len(self.pair_ordinals) != 2
            or any(type(value) is not int or value < 0 for value in self.pair_ordinals)
            or self.pair_ordinals[0] == self.pair_ordinals[1]
            or type(self.relation_sign) is not int
            or self.relation_sign not in {-1, 1}
        ):
            raise ValueError("ASG-CV forced pair authority differs")
        _finite(self.same_score, name="same score")
        _finite(self.different_score, name="different score")
        _finite(self.gradient_norm, name="gradient norm", positive=True)
        if (
            type(self.boundary_norms) is not tuple
            or len(self.boundary_norms) != 4
            or any(
                type(value) is not float or not math.isfinite(value) or value <= 0.0
                for value in self.boundary_norms
            )
        ):
            raise ValueError("ASG-CV forced boundary norm differs")
        for name in (
            "prepare_elapsed_ns",
            "replay_elapsed_ns",
            "peak_cuda_reserved_bytes",
            "peak_rss_bytes",
        ):
            _positive_int(getattr(self, name), name=name.replace("_", " "))
        return self

    def to_mapping(self) -> dict[str, object]:
        self.validated()
        return {
            "schema": ASGCV_FORCED_OBSERVATION_SCHEMA,
            "claim_eligible": False,
            "official_test_access": False,
            "generated_tokens": 0,
            "branch_order": list(ASGCV_FORCED_BRANCH_ORDER),
            **{
                field.name: list(value) if isinstance(value, tuple) else value
                for field in fields(self)
                if (value := getattr(self, field.name)) is not None
            },
        }

    @classmethod
    def from_mapping(cls, value: object) -> AsgcvForcedObservation:
        expected = {field.name for field in fields(cls)} | {
            "schema",
            "claim_eligible",
            "official_test_access",
            "generated_tokens",
            "branch_order",
        }
        if (
            type(value) is not dict
            or set(value) != expected
            or value["schema"] != ASGCV_FORCED_OBSERVATION_SCHEMA
            or value["claim_eligible"] is not False
            or value["official_test_access"] is not False
            or value["generated_tokens"] != 0
            or value["branch_order"] != list(ASGCV_FORCED_BRANCH_ORDER)
        ):
            raise ValueError("ASG-CV forced observation schema differs")
        raw = cast(dict[str, Any], value)
        try:
            return cls(
                **{
                    field.name: (
                        tuple(raw[field.name])
                        if field.name in {"pair_ordinals", "boundary_norms"}
                        and type(raw[field.name]) is list
                        else raw[field.name]
                    )
                    for field in fields(cls)
                }
            ).validated()
        except (TypeError, KeyError) as error:
            raise ValueError("ASG-CV forced observation differs") from error

    def sha256(self) -> str:
        return hashlib.sha256(canonical_asgcv_forced_observation_bytes(self)).hexdigest()


def canonical_asgcv_forced_observation_bytes(value: AsgcvForcedObservation) -> bytes:
    """Serialize one validated observation as sorted canonical JSON plus LF."""

    if type(value) is not AsgcvForcedObservation:
        raise ValueError("ASG-CV forced observation differs")
    return _canonical_bytes(value.to_mapping())


@dataclass(frozen=True, slots=True)
class AsgcvForcedResult:
    """Recomputed score-separation result over the fixed balanced P32 schedule."""

    observations: tuple[AsgcvForcedObservation, ...]
    repeat_checked_ordinals: tuple[int, int]
    repeat_gradient_sha256s: tuple[str, str]
    accuracy_ppm: int
    same_recall_ppm: int
    different_recall_ppm: int
    auc_ppm: int
    passed: bool

    @classmethod
    def from_observations(
        cls,
        observations: tuple[AsgcvForcedObservation, ...],
        *,
        repeat_checked_ordinals: tuple[int, int],
        repeat_gradient_sha256s: tuple[str, str],
    ) -> AsgcvForcedResult:
        if (
            type(observations) is not tuple
            or len(observations) != ASGCV_FORCED_PAIR_COUNT
            or any(type(row) is not AsgcvForcedObservation for row in observations)
            or tuple(row.candidate_pair_ordinal for row in observations)
            != tuple(range(ASGCV_FORCED_PAIR_COUNT))
        ):
            raise ValueError("ASG-CV forced observation ordinal differs")
        for row in observations:
            row.validated()
        identity = tuple(
            (
                row.source_commit,
                row.launch_authority_sha256,
                row.pilot_schedule_sha256,
                row.model_revision,
                row.fixture_sha256,
            )
            for row in observations
        )
        if len(set(identity)) != 1:
            raise ValueError("ASG-CV forced observation identity differs")
        same = tuple(row for row in observations if row.relation_sign == 1)
        different = tuple(row for row in observations if row.relation_sign == -1)
        if len(same) != 16 or len(different) != 16:
            raise ValueError("ASG-CV forced relation balance differs")
        if repeat_checked_ordinals != ASGCV_FORCED_REPEAT_ORDINALS:
            raise ValueError("ASG-CV forced repeat ordinals differ")
        if (
            type(repeat_gradient_sha256s) is not tuple
            or len(repeat_gradient_sha256s) != 2
            or any(
                _hex(value, 64, name="repeat gradient") != value
                for value in repeat_gradient_sha256s
            )
            or repeat_gradient_sha256s
            != tuple(observations[ordinal].gradient_sha256 for ordinal in repeat_checked_ordinals)
        ):
            raise ValueError("ASG-CV forced repeat evidence differs")
        accuracy = _ppm(sum(row.correct for row in observations), len(observations))
        same_recall = _ppm(sum(row.correct for row in same), len(same))
        different_recall = _ppm(sum(row.correct for row in different), len(different))
        wins = 0.0
        for positive in same:
            for negative in different:
                wins += float(positive.score_gap > negative.score_gap)
                wins += 0.5 * float(positive.score_gap == negative.score_gap)
        auc = _ppm(wins, len(same) * len(different))
        passed = (
            accuracy >= ASGCV_FORCED_ACCURACY_GATE_PPM
            and auc >= ASGCV_FORCED_AUC_GATE_PPM
            and same_recall >= ASGCV_FORCED_SIGN_RECALL_GATE_PPM
            and different_recall >= ASGCV_FORCED_SIGN_RECALL_GATE_PPM
        )
        return cls(
            observations=observations,
            repeat_checked_ordinals=repeat_checked_ordinals,
            repeat_gradient_sha256s=repeat_gradient_sha256s,
            accuracy_ppm=accuracy,
            same_recall_ppm=same_recall,
            different_recall_ppm=different_recall,
            auc_ppm=auc,
            passed=passed,
        )

    def validated(self) -> AsgcvForcedResult:
        rebuilt = type(self).from_observations(
            self.observations,
            repeat_checked_ordinals=self.repeat_checked_ordinals,
            repeat_gradient_sha256s=self.repeat_gradient_sha256s,
        )
        if self != rebuilt:
            raise ValueError("ASG-CV forced result evidence differs")
        return self

    def to_mapping(self) -> dict[str, object]:
        self.validated()
        return {
            "schema": ASGCV_FORCED_RESULT_SCHEMA,
            "claim_eligible": False,
            "official_test_access": False,
            "pair_count": ASGCV_FORCED_PAIR_COUNT,
            "generated_tokens": 0,
            "branch_order": list(ASGCV_FORCED_BRANCH_ORDER),
            "gates_ppm": {
                "accuracy": ASGCV_FORCED_ACCURACY_GATE_PPM,
                "auc": ASGCV_FORCED_AUC_GATE_PPM,
                "per_relation_recall": ASGCV_FORCED_SIGN_RECALL_GATE_PPM,
            },
            "observations": [row.to_mapping() for row in self.observations],
            "repeat_checked_ordinals": list(self.repeat_checked_ordinals),
            "repeat_gradient_sha256s": list(self.repeat_gradient_sha256s),
            "accuracy_ppm": self.accuracy_ppm,
            "same_recall_ppm": self.same_recall_ppm,
            "different_recall_ppm": self.different_recall_ppm,
            "auc_ppm": self.auc_ppm,
            "passed": self.passed,
        }

    @classmethod
    def from_mapping(cls, value: object) -> AsgcvForcedResult:
        expected = {
            "schema",
            "claim_eligible",
            "official_test_access",
            "pair_count",
            "generated_tokens",
            "branch_order",
            "gates_ppm",
            "observations",
            "repeat_checked_ordinals",
            "repeat_gradient_sha256s",
            "accuracy_ppm",
            "same_recall_ppm",
            "different_recall_ppm",
            "auc_ppm",
            "passed",
        }
        gates = {
            "accuracy": ASGCV_FORCED_ACCURACY_GATE_PPM,
            "auc": ASGCV_FORCED_AUC_GATE_PPM,
            "per_relation_recall": ASGCV_FORCED_SIGN_RECALL_GATE_PPM,
        }
        if (
            type(value) is not dict
            or set(value) != expected
            or value["schema"] != ASGCV_FORCED_RESULT_SCHEMA
            or value["claim_eligible"] is not False
            or value["official_test_access"] is not False
            or value["pair_count"] != ASGCV_FORCED_PAIR_COUNT
            or value["generated_tokens"] != 0
            or value["branch_order"] != list(ASGCV_FORCED_BRANCH_ORDER)
            or value["gates_ppm"] != gates
            or type(value["observations"]) is not list
            or type(value["repeat_checked_ordinals"]) is not list
            or type(value["repeat_gradient_sha256s"]) is not list
        ):
            raise ValueError("ASG-CV forced result schema differs")
        raw = cast(dict[str, Any], value)
        rebuilt = cls.from_observations(
            tuple(AsgcvForcedObservation.from_mapping(row) for row in raw["observations"]),
            repeat_checked_ordinals=tuple(raw["repeat_checked_ordinals"]),
            repeat_gradient_sha256s=tuple(raw["repeat_gradient_sha256s"]),
        )
        if rebuilt.to_mapping() != value:
            raise ValueError("ASG-CV forced result evidence differs")
        return rebuilt


def canonical_asgcv_forced_result_bytes(value: AsgcvForcedResult) -> bytes:
    """Serialize one fully recomputed forced-verdict pilot result."""

    if type(value) is not AsgcvForcedResult:
        raise ValueError("ASG-CV forced result differs")
    return _canonical_bytes(value.to_mapping())
