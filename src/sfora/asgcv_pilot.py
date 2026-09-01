"""Canonical authority and hard gates for the training-only ASG-CV P32 pilot."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from dataclasses import dataclass, fields
from typing import Any, cast

import numpy as np

from sfora.asgcv_protocol import (
    AsgcvCompletionGroup,
    AsgcvPairSchedule,
    AsgcvPartitionAuthority,
    AsgcvRolloutAuthority,
    build_asgcv_pair_schedule,
    derive_asgcv_rollout_seeds,
    validate_asgcv_partition_bundle,
)
from sfora.asgcv_verdict_marginal import (
    collapsed_verdict_coefficient,
    collapsed_verdict_probability,
)

ASGCV_P32_PAIR_COUNT = 32
ASGCV_P32_GROUP_SIZE = 8
ASGCV_P32_CANDIDATE_SCHEMA = "sfora-asgcv-pilot-candidate-v1"
ASGCV_P32_RESULT_SCHEMA = "sfora-asgcv-pilot-result-v1"
ASGCV_P32_EXECUTION_BACKEND = "cuda-deterministic-v1"
ASGCV_P32_BRANCH_ENERGY_GATE_PPM = 350_000
ASGCV_P32_DISPERSION_MEDIAN_GATE_PPM = 250_000
ASGCV_P32_DISPERSION_P90_GATE_PPM = 500_000
ASGCV_P32_BRANCH_YIELD_GATE_PPM = 500_000
ASGCV_P32_SIGN_YIELD_GATE_PPM = 375_000
ASGCV_P32_COEFFICIENT_GATE_PPM = 200_000
ASGCV_P32_CALIBRATION_GATE_PPM = 250_000
ASGCV_P32_COMPLETION_VALIDITY_GATE_PPM = 750_000
ASGCV_P32_STEP_WALL_RATIO_GATE_PPM = 250_000
ASGCV_P32_EXCHANGE_EVALUABLE_MINIMUM = 8
ASGCV_P32_PEAK_CUDA_GATE_BYTES = 96 * 1024**3
ASGCV_P32_CANDIDATE_P90_GATE_NS = 300_000_000_000
ASGCV_P32_BOUNDARY_NAMES = ("merger", "deepstack-0", "deepstack-1", "deepstack-2")
ASGCV_P32_FIELD_DOMAIN = b"sfora-asgcv-p32-field-v1\0"
ASGCV_P32_SCHEDULE_DOMAIN = b"sfora-asgcv-p32-pilot-schedule-v1\0"
ASGCV_P32_BRANCH_SCORE_ABS_TOLERANCE = 1e-6
ASGCV_P32_COEFFICIENT_TOLERANCE_PPM = 2


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
        raise ValueError(f"ASG-CV P32 {name} differs")
    return value


def _commit(value: object, *, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"ASG-CV P32 {name} differs")
    return value


def _finite(value: object, *, name: str, positive: bool = False) -> float:
    if type(value) is not float or not math.isfinite(value) or (positive and value <= 0.0):
        raise ValueError(f"ASG-CV P32 {name} differs")
    return value


def _nonnegative_finite(value: object, *, name: str) -> float:
    result = _finite(value, name=name)
    if result < 0.0:
        raise ValueError(f"ASG-CV P32 {name} differs")
    return result


def _branch_scores_match(actual: tuple[float, float], expected: tuple[float, float]) -> bool:
    return all(
        abs(observed - authority) <= ASGCV_P32_BRANCH_SCORE_ABS_TOLERANCE
        for observed, authority in zip(actual, expected, strict=True)
    )


def _ppm(value: float) -> int:
    return int(round(value * 1_000_000.0))


def _rate_ppm(numerator: int, denominator: int) -> int:
    return (numerator * 1_000_000 + denominator // 2) // denominator


def _median(values: tuple[int, ...]) -> int:
    if not values:
        raise ValueError("ASG-CV P32 median evidence differs")
    return int(round(statistics.median(values)))


def _p90(values: tuple[int, ...]) -> int:
    if not values:
        raise ValueError("ASG-CV P32 p90 evidence differs")
    ordered = sorted(values)
    return ordered[math.ceil(0.9 * len(ordered)) - 1]


def _p32_field(value: object, *, name: str) -> np.ndarray:
    if (
        type(value) is not np.ndarray
        or value.dtype != np.dtype(np.float32)
        or value.size == 0
        or not bool(np.isfinite(value).all())
    ):
        raise ValueError(f"ASG-CV P32 {name} field differs")
    return np.ascontiguousarray(value)


def asgcv_p32_field_authority(value: object, *, role: str) -> tuple[str, float]:
    """Return one shape-bound fp32 field digest and fp64 Euclidean norm."""

    if type(role) is not str or not role:
        raise ValueError("ASG-CV P32 field role differs")
    array = _p32_field(value, name=role)
    metadata = _canonical_json_bytes({"dtype": "float32", "role": role, "shape": list(array.shape)})
    digest = hashlib.sha256(ASGCV_P32_FIELD_DOMAIN + metadata + array.tobytes()).hexdigest()
    norm = float(np.sqrt(np.square(array.astype(np.float64)).sum(dtype=np.float64)))
    if not math.isfinite(norm):
        raise ValueError("ASG-CV P32 field norm differs")
    return digest, norm


def asgcv_p32_branch_exchange_energy_ppm(lowest: object, highest: object) -> int:
    """Measure symmetric normalized energy changed by branch exchange."""

    low: np.ndarray = _p32_field(lowest, name="lowest branch").astype(np.float64)
    high: np.ndarray = _p32_field(highest, name="highest branch").astype(np.float64)
    if low.shape != high.shape:
        raise ValueError("ASG-CV P32 branch exchange shape differs")
    denominator = float(
        np.square(low).sum(dtype=np.float64) + np.square(high).sum(dtype=np.float64)
    )
    if denominator <= 0.0 or not math.isfinite(denominator):
        return 1_000_000
    ratio = float(np.square(high - low).sum(dtype=np.float64)) / denominator
    if not math.isfinite(ratio) or ratio < 0.0:
        raise ValueError("ASG-CV P32 branch exchange energy differs")
    return min(1_000_000, _ppm(ratio))


def asgcv_p32_collapsed_exact_cosine(collapsed: object, exact: object) -> float:
    """Measure fp64 cosine between same-shaped collapsed and exact fields."""

    left: np.ndarray = _p32_field(collapsed, name="collapsed cosine").astype(np.float64)
    right: np.ndarray = _p32_field(exact, name="exact cosine").astype(np.float64)
    if left.shape != right.shape:
        raise ValueError("ASG-CV P32 field cosine shape differs")
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 0.0 or not math.isfinite(denominator):
        raise ValueError("ASG-CV P32 field cosine denominator differs")
    cosine = float(np.sum(left * right, dtype=np.float64)) / denominator
    if not math.isfinite(cosine):
        raise ValueError("ASG-CV P32 field cosine differs")
    return max(-1.0, min(1.0, cosine))


def derive_asgcv_p32_schedule_seed(
    *,
    partition_authority: AsgcvPartitionAuthority,
    source_commit: str,
) -> str:
    """Derive the P32-only pair seed from frozen source and partition authority."""

    if type(partition_authority) is not AsgcvPartitionAuthority:
        raise ValueError("ASG-CV P32 partition authority differs")
    partition_authority.validated()
    source = _commit(source_commit, name="schedule source commit")
    return hashlib.sha256(
        ASGCV_P32_SCHEDULE_DOMAIN
        + bytes.fromhex(partition_authority.sha256())
        + bytes.fromhex(source)
    ).hexdigest()


def validate_asgcv_p32_pilot_schedule(
    schedule: AsgcvPairSchedule,
    *,
    partition_authority: AsgcvPartitionAuthority,
    source_commit: str,
    predictor_train: object,
    e0_validation: object,
    e1_optimization: object,
) -> None:
    """Bind P32 pairs to the isolated predictor-training partition."""

    if type(schedule) is not AsgcvPairSchedule:
        raise ValueError("ASG-CV P32 pilot schedule differs")
    validate_asgcv_partition_bundle(
        partition_authority,
        predictor_train=predictor_train,
        e0_validation=e0_validation,
        e1_optimization=e1_optimization,
    )
    if (
        type(predictor_train) is not tuple
        or len(predictor_train) != 2
        or type(predictor_train[0]) is not tuple
        or type(predictor_train[1]) is not tuple
    ):
        raise ValueError("ASG-CV P32 predictor partition differs")
    expected_seed = derive_asgcv_p32_schedule_seed(
        partition_authority=partition_authority,
        source_commit=source_commit,
    )
    rebuilt = build_asgcv_pair_schedule(
        predictor_train[0],
        predictor_train[1],
        schedule_seed_sha256=expected_seed,
        pair_count=ASGCV_P32_PAIR_COUNT,
    )
    if schedule != rebuilt:
        raise ValueError("ASG-CV P32 pilot schedule reconstruction differs")


@dataclass(frozen=True, slots=True)
class AsgcvP32Candidate:
    """One digest-only P32 candidate with derivable semantic and runtime evidence."""

    source_commit: str
    model_revision: str
    fixture_sha256: str
    launch_authority_sha256: str
    predictor_initialization_seed_sha256: str
    partition_authority_sha256: str
    pilot_schedule_sha256: str
    completion_protocol_sha256: str
    rollout_authority_sha256: str
    completion_group_sha256: str
    pooler_state_sha256: str
    candidate_pair_ordinal: int
    pair_ordinals: tuple[int, int]
    relation_sign: int
    generation_seeds: tuple[int, ...]
    rewards: tuple[int, ...]
    valid_flags: tuple[bool, ...]
    verdict_relation_signs: tuple[int | None, ...]
    attribute_span_lengths: tuple[int, ...]
    generated_token_counts: tuple[int, ...]
    completion_scores: tuple[float, ...]
    lowest_branch_indices: tuple[int, int] | None
    highest_branch_indices: tuple[int, int] | None
    branch_exchange_distinct: bool
    collapsed_branch_scores: tuple[float, float] | None
    collapsed_backend_coefficient_ppm: int | None
    highest_branch_scores: tuple[float, float] | None
    highest_backend_coefficient_ppm: int | None
    lowest_gradient_sha256: str | None
    highest_gradient_sha256: str | None
    lowest_gradient_norm: float | None
    highest_gradient_norm: float | None
    branch_exchange_energy_ppm: int | None
    boundary_names: tuple[str, ...]
    boundary_norms: tuple[float, ...] | None
    exact_gradient_sha256: str | None
    exact_gradient_norm: float | None
    exact_replay_generated_tokens: int | None
    collapsed_exact_cosine: float | None
    prepare_elapsed_ns: int
    generate_elapsed_ns: int
    score_elapsed_ns: int
    collapsed_replay_elapsed_ns: int
    branch_exchange_replay_elapsed_ns: int
    exact_replay_elapsed_ns: int
    predictor_forward_elapsed_ns: int
    candidate_total_elapsed_ns: int
    peak_cuda_reserved_bytes: int
    peak_rss_bytes: int

    @property
    def valid_correct_count(self) -> int:
        return sum(
            valid and verdict == self.relation_sign
            for valid, verdict in zip(self.valid_flags, self.verdict_relation_signs, strict=True)
        )

    @property
    def valid_incorrect_count(self) -> int:
        return sum(
            valid and verdict == -self.relation_sign
            for valid, verdict in zip(self.valid_flags, self.verdict_relation_signs, strict=True)
        )

    @property
    def both_verdicts_valid(self) -> bool:
        return self.valid_correct_count >= 1 and self.valid_incorrect_count >= 1

    @property
    def exchange_evaluable(self) -> bool:
        return (
            self.valid_correct_count >= 2
            and self.valid_incorrect_count >= 2
            and self.branch_exchange_distinct
        )

    @property
    def nonzero_reward_variance(self) -> bool:
        return 0 < sum(self.rewards) < ASGCV_P32_GROUP_SIZE

    @property
    def empirical_correct_probability_ppm(self) -> int:
        valid_count = self.valid_correct_count + self.valid_incorrect_count
        return 0 if valid_count == 0 else _rate_ppm(self.valid_correct_count, valid_count)

    def _score_groups(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        correct = tuple(
            score
            for score, valid, verdict in zip(
                self.completion_scores,
                self.valid_flags,
                self.verdict_relation_signs,
                strict=True,
            )
            if valid and verdict == self.relation_sign
        )
        incorrect = tuple(
            score
            for score, valid, verdict in zip(
                self.completion_scores,
                self.valid_flags,
                self.verdict_relation_signs,
                strict=True,
            )
            if valid and verdict == -self.relation_sign
        )
        return correct, incorrect

    @property
    def score_probability(self) -> float | None:
        if self.lowest_branch_indices is None:
            return None
        correct_ordinal, incorrect_ordinal = self.lowest_branch_indices
        return float(
            collapsed_verdict_probability(
                self.completion_scores[correct_ordinal],
                self.completion_scores[incorrect_ordinal],
            )
        )

    @property
    def coefficient_ppm(self) -> int | None:
        probability = self.score_probability
        return None if probability is None else _ppm(collapsed_verdict_coefficient(probability))

    @property
    def probability_calibration_ppm(self) -> int | None:
        probability = self.score_probability
        if probability is None:
            return None
        empirical = self.empirical_correct_probability_ppm / 1_000_000.0
        return _ppm(abs(probability - empirical))

    @property
    def within_verdict_dispersion_ratio_ppm(self) -> int | None:
        correct, incorrect = self._score_groups()
        if not correct or not incorrect:
            return None
        correct_mean = math.fsum(correct) / len(correct)
        incorrect_mean = math.fsum(incorrect) / len(incorrect)
        gap = abs(correct_mean - incorrect_mean)
        if gap == 0.0:
            return 1_000_000

        def population_sd(values: tuple[float, ...], mean: float) -> float:
            return math.sqrt(math.fsum((value - mean) ** 2 for value in values) / len(values))

        ratio = (
            max(
                population_sd(correct, correct_mean),
                population_sd(incorrect, incorrect_mean),
            )
            / gap
        )
        return min(1_000_000, _ppm(ratio))

    @property
    def exact_diagnostic(self) -> bool:
        return self.exact_gradient_sha256 is not None

    def validated(self) -> AsgcvP32Candidate:
        _commit(self.source_commit, name="source commit")
        _commit(self.model_revision, name="model revision")
        for name in (
            "fixture_sha256",
            "launch_authority_sha256",
            "predictor_initialization_seed_sha256",
            "partition_authority_sha256",
            "pilot_schedule_sha256",
            "completion_protocol_sha256",
            "rollout_authority_sha256",
            "completion_group_sha256",
            "pooler_state_sha256",
        ):
            _sha256(getattr(self, name), name=name)
        if type(self.branch_exchange_distinct) is not bool:
            raise ValueError("ASG-CV P32 branch distinctness differs")
        if (
            type(self.candidate_pair_ordinal) is not int
            or not 0 <= self.candidate_pair_ordinal < ASGCV_P32_PAIR_COUNT
            or type(self.pair_ordinals) is not tuple
            or len(self.pair_ordinals) != 2
            or any(type(value) is not int or value < 0 for value in self.pair_ordinals)
            or self.pair_ordinals[0] == self.pair_ordinals[1]
            or type(self.relation_sign) is not int
            or self.relation_sign not in {-1, 1}
        ):
            raise ValueError("ASG-CV P32 candidate relation differs")
        if self.boundary_names != ASGCV_P32_BOUNDARY_NAMES:
            raise ValueError("ASG-CV P32 boundary authority differs")
        sequences = (
            self.generation_seeds,
            self.rewards,
            self.valid_flags,
            self.verdict_relation_signs,
            self.attribute_span_lengths,
            self.generated_token_counts,
            self.completion_scores,
        )
        if any(
            type(value) is not tuple or len(value) != ASGCV_P32_GROUP_SIZE for value in sequences
        ):
            raise ValueError("ASG-CV P32 completion evidence differs")
        if (
            any(type(value) is not int or not 0 <= value < 2**64 for value in self.generation_seeds)
            or len(set(self.generation_seeds)) != ASGCV_P32_GROUP_SIZE
            or any(type(value) is not int or value not in {0, 1} for value in self.rewards)
            or any(type(value) is not bool for value in self.valid_flags)
            or any(
                value is not None and type(value) is not int
                for value in self.verdict_relation_signs
            )
            or any(value not in {-1, 1, None} for value in self.verdict_relation_signs)
            or self.valid_flags != tuple(value is not None for value in self.verdict_relation_signs)
            or self.rewards
            != tuple(
                int(valid and verdict == self.relation_sign)
                for valid, verdict in zip(
                    self.valid_flags,
                    self.verdict_relation_signs,
                    strict=True,
                )
            )
            or any(type(value) is not int or value < 0 for value in self.attribute_span_lengths)
            or any(type(value) is not int or value <= 0 for value in self.generated_token_counts)
            or any(
                type(value) is not float or not math.isfinite(value)
                for value in self.completion_scores
            )
        ):
            raise ValueError("ASG-CV P32 completion relation differs")
        correct = tuple(
            index
            for index, (valid, verdict) in enumerate(
                zip(self.valid_flags, self.verdict_relation_signs, strict=True)
            )
            if valid and verdict == self.relation_sign
        )
        incorrect = tuple(
            index
            for index, (valid, verdict) in enumerate(
                zip(self.valid_flags, self.verdict_relation_signs, strict=True)
            )
            if valid and verdict == -self.relation_sign
        )
        expected_lowest = (correct[0], incorrect[0]) if correct and incorrect else None
        expected_highest = (
            (correct[-1], incorrect[-1])
            if len(correct) >= 2 and len(incorrect) >= 2 and self.branch_exchange_distinct
            else None
        )
        if (
            self.lowest_branch_indices != expected_lowest
            or self.highest_branch_indices != expected_highest
        ):
            raise ValueError("ASG-CV P32 branch selection differs")
        if self.both_verdicts_valid:
            expected_lowest_scores = (
                self.completion_scores[cast(tuple[int, int], expected_lowest)[0]],
                self.completion_scores[cast(tuple[int, int], expected_lowest)[1]],
            )
            if (
                type(self.collapsed_branch_scores) is not tuple
                or len(self.collapsed_branch_scores) != 2
                or any(
                    type(value) is not float or not math.isfinite(value)
                    for value in self.collapsed_branch_scores
                )
                or not _branch_scores_match(
                    self.collapsed_branch_scores,
                    expected_lowest_scores,
                )
                or type(self.collapsed_backend_coefficient_ppm) is not int
                or abs(self.collapsed_backend_coefficient_ppm - cast(int, self.coefficient_ppm))
                > ASGCV_P32_COEFFICIENT_TOLERANCE_PPM
            ):
                raise ValueError("ASG-CV P32 collapsed backend evidence differs")
        elif (
            self.collapsed_branch_scores is not None
            or self.collapsed_backend_coefficient_ppm is not None
        ):
            raise ValueError("ASG-CV P32 collapsed backend eligibility differs")
        lowest_values = (
            self.lowest_gradient_sha256,
            self.lowest_gradient_norm,
            self.boundary_norms,
        )
        if self.both_verdicts_valid:
            _sha256(self.lowest_gradient_sha256, name="lowest gradient digest")
            _nonnegative_finite(self.lowest_gradient_norm, name="lowest gradient norm")
            if (
                type(self.boundary_norms) is not tuple
                or len(self.boundary_norms) != 4
                or any(
                    type(value) is not float or not math.isfinite(value) or value < 0.0
                    for value in self.boundary_norms
                )
            ):
                raise ValueError("ASG-CV P32 boundary norms differ")
        elif any(value is not None for value in lowest_values):
            raise ValueError("ASG-CV P32 lowest branch evidence differs")
        if self.exchange_evaluable:
            _sha256(self.highest_gradient_sha256, name="highest gradient digest")
            _nonnegative_finite(self.highest_gradient_norm, name="highest gradient norm")
            highest = cast(tuple[int, int], expected_highest)
            expected_highest_scores = (
                self.completion_scores[highest[0]],
                self.completion_scores[highest[1]],
            )
            expected_highest_coefficient = _ppm(
                collapsed_verdict_coefficient(
                    collapsed_verdict_probability(*expected_highest_scores)
                )
            )
            if (
                type(self.highest_branch_scores) is not tuple
                or not _branch_scores_match(
                    self.highest_branch_scores,
                    expected_highest_scores,
                )
                or type(self.highest_backend_coefficient_ppm) is not int
                or abs(self.highest_backend_coefficient_ppm - expected_highest_coefficient)
                > ASGCV_P32_COEFFICIENT_TOLERANCE_PPM
                or type(self.branch_exchange_energy_ppm) is not int
                or not 0 <= self.branch_exchange_energy_ppm <= 1_000_000
                or type(self.branch_exchange_replay_elapsed_ns) is not int
                or self.branch_exchange_replay_elapsed_ns <= 0
            ):
                raise ValueError("ASG-CV P32 branch-exchange evidence differs")
        elif (
            any(
                value is not None
                for value in (
                    self.highest_gradient_sha256,
                    self.highest_gradient_norm,
                    self.highest_branch_scores,
                    self.highest_backend_coefficient_ppm,
                    self.branch_exchange_energy_ppm,
                )
            )
            or self.branch_exchange_replay_elapsed_ns != 0
        ):
            raise ValueError("ASG-CV P32 branch-exchange eligibility differs")
        exact_values = (
            self.exact_gradient_sha256,
            self.exact_gradient_norm,
            self.exact_replay_generated_tokens,
            self.collapsed_exact_cosine,
        )
        if self.exact_diagnostic:
            _sha256(self.exact_gradient_sha256, name="exact gradient digest")
            _nonnegative_finite(self.exact_gradient_norm, name="exact gradient norm")
            if (
                type(self.exact_replay_generated_tokens) is not int
                or self.exact_replay_generated_tokens != sum(self.generated_token_counts)
                or type(self.exact_replay_elapsed_ns) is not int
                or self.exact_replay_elapsed_ns <= 0
            ):
                raise ValueError("ASG-CV P32 exact diagnostic differs")
            comparable = (
                self.both_verdicts_valid
                and cast(float, self.lowest_gradient_norm) > 0.0
                and cast(float, self.exact_gradient_norm) > 0.0
            )
            if comparable:
                if (
                    type(self.collapsed_exact_cosine) is not float
                    or not math.isfinite(self.collapsed_exact_cosine)
                    or not -1.0 <= self.collapsed_exact_cosine <= 1.0
                ):
                    raise ValueError("ASG-CV P32 exact diagnostic differs")
            elif self.collapsed_exact_cosine is not None:
                raise ValueError("ASG-CV P32 exact diagnostic differs")
        elif any(value is not None for value in exact_values) or self.exact_replay_elapsed_ns != 0:
            raise ValueError("ASG-CV P32 exact diagnostic eligibility differs")
        positive_timings = (
            self.prepare_elapsed_ns,
            self.generate_elapsed_ns,
            self.score_elapsed_ns,
            self.candidate_total_elapsed_ns,
        )
        optional_timings = (
            self.collapsed_replay_elapsed_ns,
            self.branch_exchange_replay_elapsed_ns,
            self.exact_replay_elapsed_ns,
            self.predictor_forward_elapsed_ns,
        )
        if any(type(value) is not int or value <= 0 for value in positive_timings) or any(
            type(value) is not int or value < 0 for value in optional_timings
        ):
            raise ValueError("ASG-CV P32 timing evidence differs")
        measured_phase_total = (
            self.prepare_elapsed_ns
            + self.generate_elapsed_ns
            + self.score_elapsed_ns
            + self.collapsed_replay_elapsed_ns
            + self.branch_exchange_replay_elapsed_ns
            + self.exact_replay_elapsed_ns
            + self.predictor_forward_elapsed_ns
        )
        if self.candidate_total_elapsed_ns < measured_phase_total:
            raise ValueError("ASG-CV P32 candidate total timing differs")
        if (
            type(self.peak_cuda_reserved_bytes) is not int
            or self.peak_cuda_reserved_bytes <= 0
            or type(self.peak_rss_bytes) is not int
            or self.peak_rss_bytes <= 0
        ):
            raise ValueError("ASG-CV P32 resource evidence differs")
        if self.both_verdicts_valid != (self.collapsed_replay_elapsed_ns > 0):
            raise ValueError("ASG-CV P32 collapsed timing differs")
        if self.both_verdicts_valid != (self.predictor_forward_elapsed_ns > 0):
            raise ValueError("ASG-CV P32 predictor timing differs")
        return self

    def _base_mapping(self) -> dict[str, object]:
        self.validated()
        mapping: dict[str, object] = {
            "schema": ASGCV_P32_CANDIDATE_SCHEMA,
            "claim_eligible": False,
            "official_test_access": False,
            "partition_role": "predictor-training",
            "execution_backend": ASGCV_P32_EXECUTION_BACKEND,
        }
        for field in fields(self):
            value = getattr(self, field.name)
            mapping[field.name] = list(value) if type(value) is tuple else value
        mapping.update(
            {
                "valid_correct_count": self.valid_correct_count,
                "valid_incorrect_count": self.valid_incorrect_count,
                "both_verdicts_valid": self.both_verdicts_valid,
                "exchange_evaluable": self.exchange_evaluable,
                "nonzero_reward_variance": self.nonzero_reward_variance,
                "empirical_correct_probability_ppm": self.empirical_correct_probability_ppm,
                "score_probability_ppm": None
                if self.score_probability is None
                else _ppm(self.score_probability),
                "coefficient_ppm": self.coefficient_ppm,
                "probability_calibration_ppm": self.probability_calibration_ppm,
                "within_verdict_dispersion_ratio_ppm": self.within_verdict_dispersion_ratio_ppm,
            }
        )
        return mapping

    def sha256(self) -> str:
        return hashlib.sha256(_canonical_json_bytes(self._base_mapping())).hexdigest()

    def to_mapping(self) -> dict[str, object]:
        mapping = self._base_mapping()
        mapping["candidate_sha256"] = self.sha256()
        return mapping

    @classmethod
    def from_mapping(cls, value: object) -> AsgcvP32Candidate:
        if type(value) is not dict:
            raise ValueError("ASG-CV P32 candidate schema differs")
        field_names = {field.name for field in fields(cls)}
        expected = field_names | {
            "schema",
            "claim_eligible",
            "official_test_access",
            "partition_role",
            "execution_backend",
            "valid_correct_count",
            "valid_incorrect_count",
            "both_verdicts_valid",
            "exchange_evaluable",
            "nonzero_reward_variance",
            "empirical_correct_probability_ppm",
            "score_probability_ppm",
            "coefficient_ppm",
            "probability_calibration_ppm",
            "within_verdict_dispersion_ratio_ppm",
            "candidate_sha256",
        }
        if (
            set(value) != expected
            or value["schema"] != ASGCV_P32_CANDIDATE_SCHEMA
            or value["claim_eligible"] is not False
            or value["official_test_access"] is not False
            or value["partition_role"] != "predictor-training"
            or value["execution_backend"] != ASGCV_P32_EXECUTION_BACKEND
        ):
            raise ValueError("ASG-CV P32 candidate authority differs")
        tuple_fields = {
            "pair_ordinals",
            "generation_seeds",
            "rewards",
            "valid_flags",
            "verdict_relation_signs",
            "attribute_span_lengths",
            "generated_token_counts",
            "completion_scores",
            "lowest_branch_indices",
            "highest_branch_indices",
            "collapsed_branch_scores",
            "highest_branch_scores",
            "boundary_names",
            "boundary_norms",
        }
        arguments: dict[str, object] = {}
        for name in field_names:
            raw = value[name]
            if name in tuple_fields and raw is not None:
                if type(raw) is not list:
                    raise ValueError("ASG-CV P32 candidate row differs")
                raw = tuple(raw)
            arguments[name] = raw
        candidate = cls(**cast(dict[str, Any], arguments)).validated()
        if candidate.to_mapping() != value:
            raise ValueError("ASG-CV P32 candidate derivation differs")
        return candidate


def canonical_asgcv_p32_candidate_bytes(candidate: AsgcvP32Candidate) -> bytes:
    if type(candidate) is not AsgcvP32Candidate:
        raise ValueError("ASG-CV P32 candidate differs")
    return _canonical_json_bytes(candidate.to_mapping())


def validate_asgcv_p32_candidate_bytes(raw: bytes) -> AsgcvP32Candidate:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("ASG-CV P32 candidate is not canonical JSON") from error
    candidate = AsgcvP32Candidate.from_mapping(value)
    if canonical_asgcv_p32_candidate_bytes(candidate) != raw:
        raise ValueError("ASG-CV P32 candidate bytes differ")
    return candidate


def validate_asgcv_p32_candidate_context(
    candidate: AsgcvP32Candidate,
    *,
    completion_group: AsgcvCompletionGroup,
    rollout_authority: AsgcvRolloutAuthority,
    pilot_schedule: AsgcvPairSchedule,
) -> None:
    """Cross-bind one P32 receipt to its sealed group, sampler, and pair schedule."""

    if (
        type(candidate) is not AsgcvP32Candidate
        or type(completion_group) is not AsgcvCompletionGroup
        or type(rollout_authority) is not AsgcvRolloutAuthority
        or type(pilot_schedule) is not AsgcvPairSchedule
    ):
        raise ValueError("ASG-CV P32 candidate context differs")
    candidate.validated()
    completion_group.validated()
    rollout_authority.validated()
    pilot_schedule.validated()
    ordinal = candidate.candidate_pair_ordinal
    if not 0 <= ordinal < pilot_schedule.pair_count:
        raise ValueError("ASG-CV P32 candidate schedule ordinal differs")
    pair = pilot_schedule.pairs[ordinal]
    expected_spans = tuple(
        0 if span is None else span[1] - span[0] for span in completion_group.attribute_spans
    )
    correct = tuple(
        index
        for index, (valid, verdict) in enumerate(
            zip(completion_group.valid_flags, completion_group.verdict_relation_signs, strict=True)
        )
        if valid and verdict == pair.relation_sign
    )
    incorrect = tuple(
        index
        for index, (valid, verdict) in enumerate(
            zip(completion_group.valid_flags, completion_group.verdict_relation_signs, strict=True)
        )
        if valid and verdict == -pair.relation_sign
    )
    expected_exchange_distinct = (
        len(correct) >= 2
        and len(incorrect) >= 2
        and completion_group.completion_ids[correct[0]]
        != completion_group.completion_ids[correct[-1]]
        and completion_group.completion_ids[incorrect[0]]
        != completion_group.completion_ids[incorrect[-1]]
    )
    if (
        pilot_schedule.pair_count != ASGCV_P32_PAIR_COUNT
        or candidate.pilot_schedule_sha256 != pilot_schedule.sha256()
        or candidate.completion_group_sha256 != completion_group.sha256()
        or candidate.completion_protocol_sha256 != completion_group.protocol_sha256
        or candidate.rollout_authority_sha256 != rollout_authority.sha256()
        or candidate.model_revision != rollout_authority.model_revision
        or completion_group.rollout_authority_sha256 != rollout_authority.sha256()
        or completion_group.candidate_pair_ordinal != ordinal
        or completion_group.expected_relation_sign != pair.relation_sign
        or candidate.pair_ordinals != (pair.left_index, pair.right_index)
        or candidate.relation_sign != pair.relation_sign
        or candidate.generation_seeds
        != derive_asgcv_rollout_seeds(
            rollout_authority,
            candidate_pair_ordinal=ordinal,
        )
        or candidate.generation_seeds != completion_group.generation_seeds
        or candidate.rewards != completion_group.rewards
        or candidate.valid_flags != completion_group.valid_flags
        or candidate.verdict_relation_signs != completion_group.verdict_relation_signs
        or candidate.attribute_span_lengths != expected_spans
        or candidate.generated_token_counts
        != tuple(len(completion) for completion in completion_group.completion_ids)
        or candidate.branch_exchange_distinct is not expected_exchange_distinct
    ):
        raise ValueError("ASG-CV P32 candidate context differs")


@dataclass(frozen=True, slots=True)
class AsgcvP32Result:
    """Aggregate P32 evidence with every gate recomputed from sealed counts."""

    source_commit: str
    model_revision: str
    fixture_sha256: str
    launch_authority_sha256: str
    predictor_initialization_seed_sha256: str
    partition_authority_sha256: str
    pilot_schedule_sha256: str
    completion_protocol_sha256: str
    rollout_authority_sha256: str
    pooler_state_sha256: str
    candidate_sha256s: tuple[str, ...]
    branch_eligible_count: int
    positive_branch_eligible_count: int
    negative_branch_eligible_count: int
    variance_eligible_count: int
    valid_completion_count: int
    exchange_evaluable_candidates: int
    coefficient_evaluable_candidates: int
    calibration_evaluable_candidates: int
    dispersion_evaluable_candidates: int
    collapsed_timing_candidates: int
    exact_timing_candidates: int
    branch_yield_ppm: int
    positive_branch_yield_ppm: int
    negative_branch_yield_ppm: int
    variance_yield_ppm: int
    completion_validity_ppm: int
    median_coefficient_ppm: int
    median_probability_calibration_ppm: int
    median_dispersion_ratio_ppm: int
    p90_dispersion_ratio_ppm: int
    median_branch_exchange_energy_ppm: int | None
    projected_step_wall_ratio_ppm: int
    projected_step_wall_ratio_p90_ppm: int
    projected_exact_capture_wall_ns: int
    projected_collapsed_capture_wall_ns: int
    projected_exact_capture_p90_wall_ns: int
    projected_collapsed_capture_p90_wall_ns: int
    candidate_total_p90_ns: int
    peak_cuda_reserved_bytes: int
    peak_rss_bytes: int
    exact_diagnostic_ordinals: tuple[int, ...]
    branch_exchange_gate_passed: bool
    dispersion_gate_passed: bool
    branch_yield_gate_passed: bool
    coefficient_gate_passed: bool
    calibration_gate_passed: bool
    completion_validity_gate_passed: bool
    step_wall_gate_passed: bool
    cuda_gate_passed: bool
    progress_gate_passed: bool
    passed: bool

    @classmethod
    def from_candidates(cls, candidates: tuple[AsgcvP32Candidate, ...]) -> AsgcvP32Result:
        if (
            type(candidates) is not tuple
            or len(candidates) != ASGCV_P32_PAIR_COUNT
            or any(type(candidate) is not AsgcvP32Candidate for candidate in candidates)
        ):
            raise ValueError("ASG-CV P32 candidate bundle differs")
        for candidate in candidates:
            candidate.validated()
        if tuple(candidate.candidate_pair_ordinal for candidate in candidates) != tuple(
            range(ASGCV_P32_PAIR_COUNT)
        ):
            raise ValueError("ASG-CV P32 candidate order differs")
        identity_names = (
            "source_commit",
            "model_revision",
            "fixture_sha256",
            "launch_authority_sha256",
            "predictor_initialization_seed_sha256",
            "partition_authority_sha256",
            "pilot_schedule_sha256",
            "completion_protocol_sha256",
            "rollout_authority_sha256",
            "pooler_state_sha256",
        )
        identity = tuple(getattr(candidates[0], name) for name in identity_names)
        if any(
            tuple(getattr(candidate, name) for name in identity_names) != identity
            for candidate in candidates[1:]
        ):
            raise ValueError("ASG-CV P32 campaign identity differs")
        branch = sum(candidate.both_verdicts_valid for candidate in candidates)
        positive_rows = tuple(candidate for candidate in candidates if candidate.relation_sign == 1)
        negative_rows = tuple(
            candidate for candidate in candidates if candidate.relation_sign == -1
        )
        if len(positive_rows) != 16 or len(negative_rows) != 16:
            raise ValueError("ASG-CV P32 relation balance differs")
        positive_branch = sum(candidate.both_verdicts_valid for candidate in positive_rows)
        negative_branch = sum(candidate.both_verdicts_valid for candidate in negative_rows)
        variance = sum(candidate.nonzero_reward_variance for candidate in candidates)
        valid = sum(sum(candidate.valid_flags) for candidate in candidates)
        exchanges = tuple(
            cast(int, candidate.branch_exchange_energy_ppm)
            for candidate in candidates
            if candidate.exchange_evaluable
        )
        coefficients = tuple(
            candidate.coefficient_ppm
            for candidate in candidates
            if candidate.coefficient_ppm is not None
        )
        calibrations = tuple(
            candidate.probability_calibration_ppm
            for candidate in candidates
            if candidate.probability_calibration_ppm is not None
        )
        dispersions = tuple(
            candidate.within_verdict_dispersion_ratio_ppm
            for candidate in candidates
            if candidate.exchange_evaluable
            and candidate.within_verdict_dispersion_ratio_ppm is not None
        )
        exact_rows = tuple(candidate for candidate in candidates if candidate.exact_diagnostic)
        exact_ordinals = tuple(candidate.candidate_pair_ordinal for candidate in exact_rows)
        expected_exact = tuple(
            candidate.candidate_pair_ordinal
            for candidate in candidates
            if candidate.nonzero_reward_variance
        )[:4]
        if exact_ordinals != expected_exact:
            raise ValueError("ASG-CV P32 exact diagnostic schedule differs")
        prep = _median(tuple(candidate.prepare_elapsed_ns for candidate in candidates))
        generate = _median(tuple(candidate.generate_elapsed_ns for candidate in candidates))
        score = _median(tuple(candidate.score_elapsed_ns for candidate in candidates))
        prep_p90 = _p90(tuple(candidate.prepare_elapsed_ns for candidate in candidates))
        generate_p90 = _p90(tuple(candidate.generate_elapsed_ns for candidate in candidates))
        score_p90 = _p90(tuple(candidate.score_elapsed_ns for candidate in candidates))
        collapsed_timings = tuple(
            candidate.collapsed_replay_elapsed_ns
            for candidate in candidates
            if candidate.both_verdicts_valid
        )
        predictor_timings = tuple(
            candidate.predictor_forward_elapsed_ns
            for candidate in candidates
            if candidate.both_verdicts_valid
        )
        if collapsed_timings and exact_rows:
            predictor = _median(predictor_timings)
            predictor_p90 = _p90(predictor_timings)
            collapsed = _median(collapsed_timings)
            exact = _median(tuple(candidate.exact_replay_elapsed_ns for candidate in exact_rows))
            collapsed_p90 = _p90(collapsed_timings)
            exact_p90 = _p90(tuple(candidate.exact_replay_elapsed_ns for candidate in exact_rows))
            ratio = _rate_ppm(
                prep + generate + score + collapsed + ASGCV_P32_GROUP_SIZE * predictor,
                ASGCV_P32_GROUP_SIZE * (prep + generate + exact),
            )
            ratio_p90 = _rate_ppm(
                prep_p90
                + generate_p90
                + score_p90
                + collapsed_p90
                + ASGCV_P32_GROUP_SIZE * predictor_p90,
                ASGCV_P32_GROUP_SIZE * (prep_p90 + generate_p90 + exact_p90),
            )
        else:
            ratio = 1_000_000
            ratio_p90 = 1_000_000
            collapsed = 0
            exact = 0
            collapsed_p90 = 0
            exact_p90 = 0
        branch_yield = _rate_ppm(branch, ASGCV_P32_PAIR_COUNT)
        positive_yield = _rate_ppm(positive_branch, len(positive_rows))
        negative_yield = _rate_ppm(negative_branch, len(negative_rows))
        completion_validity = _rate_ppm(valid, ASGCV_P32_PAIR_COUNT * ASGCV_P32_GROUP_SIZE)
        median_exchange = _median(exchanges) if exchanges else None
        branch_gate = (
            len(exchanges) >= ASGCV_P32_EXCHANGE_EVALUABLE_MINIMUM
            and cast(int, median_exchange) <= ASGCV_P32_BRANCH_ENERGY_GATE_PPM
        )
        dispersion_median = _median(dispersions) if dispersions else 1_000_000
        dispersion_p90 = _p90(dispersions) if dispersions else 1_000_000
        gates = {
            "branch_exchange_gate_passed": branch_gate,
            "dispersion_gate_passed": len(dispersions) >= ASGCV_P32_EXCHANGE_EVALUABLE_MINIMUM
            and dispersion_median <= ASGCV_P32_DISPERSION_MEDIAN_GATE_PPM
            and dispersion_p90 <= ASGCV_P32_DISPERSION_P90_GATE_PPM,
            "branch_yield_gate_passed": branch_yield >= ASGCV_P32_BRANCH_YIELD_GATE_PPM
            and positive_yield >= ASGCV_P32_SIGN_YIELD_GATE_PPM
            and negative_yield >= ASGCV_P32_SIGN_YIELD_GATE_PPM,
            "coefficient_gate_passed": bool(coefficients)
            and _median(coefficients) >= ASGCV_P32_COEFFICIENT_GATE_PPM,
            "calibration_gate_passed": bool(calibrations)
            and _median(calibrations) <= ASGCV_P32_CALIBRATION_GATE_PPM,
            "completion_validity_gate_passed": completion_validity
            >= ASGCV_P32_COMPLETION_VALIDITY_GATE_PPM,
            "step_wall_gate_passed": ratio_p90 <= ASGCV_P32_STEP_WALL_RATIO_GATE_PPM,
            "cuda_gate_passed": max(candidate.peak_cuda_reserved_bytes for candidate in candidates)
            <= ASGCV_P32_PEAK_CUDA_GATE_BYTES,
            "progress_gate_passed": _p90(
                tuple(candidate.candidate_total_elapsed_ns for candidate in candidates)
            )
            <= ASGCV_P32_CANDIDATE_P90_GATE_NS,
        }
        return cls(
            source_commit=candidates[0].source_commit,
            model_revision=candidates[0].model_revision,
            fixture_sha256=candidates[0].fixture_sha256,
            launch_authority_sha256=candidates[0].launch_authority_sha256,
            predictor_initialization_seed_sha256=candidates[0].predictor_initialization_seed_sha256,
            partition_authority_sha256=candidates[0].partition_authority_sha256,
            pilot_schedule_sha256=candidates[0].pilot_schedule_sha256,
            completion_protocol_sha256=candidates[0].completion_protocol_sha256,
            rollout_authority_sha256=candidates[0].rollout_authority_sha256,
            pooler_state_sha256=candidates[0].pooler_state_sha256,
            candidate_sha256s=tuple(candidate.sha256() for candidate in candidates),
            branch_eligible_count=branch,
            positive_branch_eligible_count=positive_branch,
            negative_branch_eligible_count=negative_branch,
            variance_eligible_count=variance,
            valid_completion_count=valid,
            exchange_evaluable_candidates=len(exchanges),
            coefficient_evaluable_candidates=len(coefficients),
            calibration_evaluable_candidates=len(calibrations),
            dispersion_evaluable_candidates=len(dispersions),
            collapsed_timing_candidates=len(collapsed_timings),
            exact_timing_candidates=len(exact_rows),
            branch_yield_ppm=branch_yield,
            positive_branch_yield_ppm=positive_yield,
            negative_branch_yield_ppm=negative_yield,
            variance_yield_ppm=_rate_ppm(variance, ASGCV_P32_PAIR_COUNT),
            completion_validity_ppm=completion_validity,
            median_coefficient_ppm=_median(coefficients) if coefficients else 0,
            median_probability_calibration_ppm=_median(calibrations) if calibrations else 1_000_000,
            median_dispersion_ratio_ppm=dispersion_median,
            p90_dispersion_ratio_ppm=dispersion_p90,
            median_branch_exchange_energy_ppm=median_exchange,
            projected_step_wall_ratio_ppm=ratio,
            projected_step_wall_ratio_p90_ppm=ratio_p90,
            projected_exact_capture_wall_ns=ASGCV_P32_PAIR_COUNT * 32 * (prep + generate)
            + variance * 32 * exact,
            projected_collapsed_capture_wall_ns=ASGCV_P32_PAIR_COUNT
            * 32
            * (prep + generate + score)
            + branch * 32 * collapsed,
            projected_exact_capture_p90_wall_ns=ASGCV_P32_PAIR_COUNT
            * 32
            * (prep_p90 + generate_p90)
            + variance * 32 * exact_p90,
            projected_collapsed_capture_p90_wall_ns=ASGCV_P32_PAIR_COUNT
            * 32
            * (prep_p90 + generate_p90 + score_p90)
            + branch * 32 * collapsed_p90,
            candidate_total_p90_ns=_p90(
                tuple(candidate.candidate_total_elapsed_ns for candidate in candidates)
            ),
            peak_cuda_reserved_bytes=max(
                candidate.peak_cuda_reserved_bytes for candidate in candidates
            ),
            peak_rss_bytes=max(candidate.peak_rss_bytes for candidate in candidates),
            exact_diagnostic_ordinals=exact_ordinals,
            passed=all(gates.values()),
            **gates,
        ).validated()

    def validated(self) -> AsgcvP32Result:
        _commit(self.source_commit, name="result source commit")
        _commit(self.model_revision, name="result model revision")
        for name in (
            "fixture_sha256",
            "launch_authority_sha256",
            "predictor_initialization_seed_sha256",
            "partition_authority_sha256",
            "pilot_schedule_sha256",
            "completion_protocol_sha256",
            "rollout_authority_sha256",
            "pooler_state_sha256",
        ):
            _sha256(getattr(self, name), name=f"result {name}")
        if (
            type(self.candidate_sha256s) is not tuple
            or len(self.candidate_sha256s) != ASGCV_P32_PAIR_COUNT
            or len(set(self.candidate_sha256s)) != ASGCV_P32_PAIR_COUNT
        ):
            raise ValueError("ASG-CV P32 candidate digest bundle differs")
        for digest in self.candidate_sha256s:
            _sha256(digest, name="candidate digest")
        integer_fields = tuple(
            getattr(self, field.name)
            for field in fields(self)
            if field.name
            not in {
                "source_commit",
                "model_revision",
                "fixture_sha256",
                "launch_authority_sha256",
                "predictor_initialization_seed_sha256",
                "partition_authority_sha256",
                "pilot_schedule_sha256",
                "completion_protocol_sha256",
                "rollout_authority_sha256",
                "pooler_state_sha256",
                "candidate_sha256s",
                "exact_diagnostic_ordinals",
                "median_branch_exchange_energy_ppm",
                "branch_exchange_gate_passed",
                "dispersion_gate_passed",
                "branch_yield_gate_passed",
                "coefficient_gate_passed",
                "calibration_gate_passed",
                "completion_validity_gate_passed",
                "step_wall_gate_passed",
                "cuda_gate_passed",
                "progress_gate_passed",
                "passed",
            }
        )
        if any(type(value) is not int or value < 0 for value in integer_fields):
            raise ValueError("ASG-CV P32 result metric differs")
        if self.median_branch_exchange_energy_ppm is not None and (
            type(self.median_branch_exchange_energy_ppm) is not int
            or not 0 <= self.median_branch_exchange_energy_ppm <= 1_000_000
        ):
            raise ValueError("ASG-CV P32 exchange metric differs")
        if (
            self.branch_yield_ppm != _rate_ppm(self.branch_eligible_count, ASGCV_P32_PAIR_COUNT)
            or self.positive_branch_yield_ppm != _rate_ppm(self.positive_branch_eligible_count, 16)
            or self.negative_branch_yield_ppm != _rate_ppm(self.negative_branch_eligible_count, 16)
            or self.variance_yield_ppm
            != _rate_ppm(self.variance_eligible_count, ASGCV_P32_PAIR_COUNT)
            or self.completion_validity_ppm
            != _rate_ppm(
                self.valid_completion_count,
                ASGCV_P32_PAIR_COUNT * ASGCV_P32_GROUP_SIZE,
            )
            or self.coefficient_evaluable_candidates != self.branch_eligible_count
            or self.calibration_evaluable_candidates != self.branch_eligible_count
            or self.dispersion_evaluable_candidates != self.exchange_evaluable_candidates
            or self.collapsed_timing_candidates != self.branch_eligible_count
            or self.exact_timing_candidates != len(self.exact_diagnostic_ordinals)
        ):
            raise ValueError("ASG-CV P32 result rate differs")
        expected_gates = (
            self.exchange_evaluable_candidates >= ASGCV_P32_EXCHANGE_EVALUABLE_MINIMUM
            and self.median_branch_exchange_energy_ppm is not None
            and self.median_branch_exchange_energy_ppm <= ASGCV_P32_BRANCH_ENERGY_GATE_PPM,
            self.exchange_evaluable_candidates >= ASGCV_P32_EXCHANGE_EVALUABLE_MINIMUM
            and self.median_dispersion_ratio_ppm <= ASGCV_P32_DISPERSION_MEDIAN_GATE_PPM
            and self.p90_dispersion_ratio_ppm <= ASGCV_P32_DISPERSION_P90_GATE_PPM,
            self.branch_yield_ppm >= ASGCV_P32_BRANCH_YIELD_GATE_PPM
            and self.positive_branch_yield_ppm >= ASGCV_P32_SIGN_YIELD_GATE_PPM
            and self.negative_branch_yield_ppm >= ASGCV_P32_SIGN_YIELD_GATE_PPM,
            self.median_coefficient_ppm >= ASGCV_P32_COEFFICIENT_GATE_PPM,
            self.median_probability_calibration_ppm <= ASGCV_P32_CALIBRATION_GATE_PPM,
            self.completion_validity_ppm >= ASGCV_P32_COMPLETION_VALIDITY_GATE_PPM,
            self.projected_step_wall_ratio_p90_ppm <= ASGCV_P32_STEP_WALL_RATIO_GATE_PPM,
            self.peak_cuda_reserved_bytes <= ASGCV_P32_PEAK_CUDA_GATE_BYTES,
            self.candidate_total_p90_ns <= ASGCV_P32_CANDIDATE_P90_GATE_NS,
        )
        actual_gates = (
            self.branch_exchange_gate_passed,
            self.dispersion_gate_passed,
            self.branch_yield_gate_passed,
            self.coefficient_gate_passed,
            self.calibration_gate_passed,
            self.completion_validity_gate_passed,
            self.step_wall_gate_passed,
            self.cuda_gate_passed,
            self.progress_gate_passed,
        )
        if any(type(value) is not bool for value in actual_gates) or actual_gates != expected_gates:
            raise ValueError("ASG-CV P32 gate relation differs")
        if type(self.passed) is not bool or self.passed is not all(actual_gates):
            raise ValueError("ASG-CV P32 terminal differs")
        if (
            type(self.exact_diagnostic_ordinals) is not tuple
            or len(self.exact_diagnostic_ordinals) > 4
            or self.exact_diagnostic_ordinals != tuple(sorted(set(self.exact_diagnostic_ordinals)))
            or any(
                type(ordinal) is not int or not 0 <= ordinal < ASGCV_P32_PAIR_COUNT
                for ordinal in self.exact_diagnostic_ordinals
            )
        ):
            raise ValueError("ASG-CV P32 exact diagnostic ordinals differ")
        return self

    def to_mapping(self) -> dict[str, object]:
        self.validated()
        mapping: dict[str, object] = {
            "schema": ASGCV_P32_RESULT_SCHEMA,
            "claim_eligible": False,
            "official_test_access": False,
        }
        for field in fields(self):
            value = getattr(self, field.name)
            mapping[field.name] = list(value) if type(value) is tuple else value
        return mapping


def canonical_asgcv_p32_result_bytes(
    candidates: tuple[AsgcvP32Candidate, ...],
) -> bytes:
    """Recompute one canonical P32 result from its ordered candidates."""

    return _canonical_json_bytes(AsgcvP32Result.from_candidates(candidates).to_mapping())


def validate_asgcv_p32_result_bytes(
    raw: bytes,
    candidate_receipts: tuple[bytes, ...],
) -> AsgcvP32Result:
    """Accept a P32 result only after reopening every candidate receipt."""

    if (
        type(candidate_receipts) is not tuple
        or len(candidate_receipts) != ASGCV_P32_PAIR_COUNT
        or any(type(receipt) is not bytes for receipt in candidate_receipts)
    ):
        raise ValueError("ASG-CV P32 candidate receipt bundle differs")
    candidates = tuple(
        validate_asgcv_p32_candidate_bytes(receipt) for receipt in candidate_receipts
    )
    expected = canonical_asgcv_p32_result_bytes(candidates)
    if raw != expected:
        raise ValueError("ASG-CV P32 result bytes differ")
    return AsgcvP32Result.from_candidates(candidates)


def validate_asgcv_p32_result_bundle(
    raw: bytes,
    *,
    candidate_receipts: tuple[bytes, ...],
    completion_groups: tuple[AsgcvCompletionGroup, ...],
    rollout_authority: AsgcvRolloutAuthority,
    pilot_schedule: AsgcvPairSchedule,
) -> AsgcvP32Result:
    """Accept a result only after reopening all outcome-blind P32 context."""

    if (
        type(completion_groups) is not tuple
        or len(completion_groups) != ASGCV_P32_PAIR_COUNT
        or type(candidate_receipts) is not tuple
        or len(candidate_receipts) != ASGCV_P32_PAIR_COUNT
    ):
        raise ValueError("ASG-CV P32 context bundle differs")
    candidates = tuple(
        validate_asgcv_p32_candidate_bytes(raw_row) for raw_row in candidate_receipts
    )
    for candidate, completion_group in zip(candidates, completion_groups, strict=True):
        validate_asgcv_p32_candidate_context(
            candidate,
            completion_group=completion_group,
            rollout_authority=rollout_authority,
            pilot_schedule=pilot_schedule,
        )
    return validate_asgcv_p32_result_bytes(raw, candidate_receipts)
