"""Tokenizer-ID authority for ASG-CV verdict rewards and attribute spans."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

ASGCV_COMPLETION_PROTOCOL_SCHEMA = "sfora-asgcv-completion-protocol-v1"
ASGCV_COMPLETION_PROTOCOL_DOMAIN = b"sfora-asgcv-completion-protocol-v1\0"
ASGCV_PAIR_SCHEDULE_SCHEMA = "sfora-asgcv-pair-schedule-v1"
ASGCV_PAIR_SCHEDULE_DOMAIN = b"sfora-asgcv-pair-schedule-v1\0"
ASGCV_COMPLETION_GROUP_SCHEMA = "sfora-asgcv-completion-group-v2"
ASGCV_COMPLETION_GROUP_DOMAIN = b"sfora-asgcv-completion-group-v1\0"
ASGCV_ELIGIBLE_SCHEDULE_SCHEMA = "sfora-asgcv-eligible-schedule-v1"
ASGCV_ELIGIBLE_SCHEDULE_DOMAIN = b"sfora-asgcv-eligible-schedule-v1\0"
ASGCV_ROLLOUT_AUTHORITY_SCHEMA = "sfora-asgcv-rollout-authority-v1"
ASGCV_ROLLOUT_AUTHORITY_DOMAIN = b"sfora-asgcv-rollout-authority-v1\0"
ASGCV_ROLLOUT_SEED_DOMAIN = b"sfora-asgcv-rollout-seed-v1\0"
ASGCV_PARTITION_AUTHORITY_SCHEMA = "sfora-asgcv-partition-authority-v1"
ASGCV_PARTITION_AUTHORITY_DOMAIN = b"sfora-asgcv-partition-authority-v1\0"
ASGCV_STRATUM_SIZE = 8


def _token_tuple(value: object, *, name: str, allow_empty: bool = False) -> tuple[int, ...]:
    if type(value) is not tuple or (not value and not allow_empty):
        raise ValueError(f"ASG-CV {name} differs")
    if any(type(token) is not int or not 0 <= token < 2**31 for token in value):
        raise ValueError(f"ASG-CV {name} differs")
    return value


def _canonical_json_bytes(value: dict[str, object]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")


@dataclass(frozen=True, slots=True)
class AsgcvCompletionProtocol:
    """Frozen token prefixes and terminal IDs for one exact Qwen tokenizer."""

    same_prefix_ids: tuple[int, ...]
    different_prefix_ids: tuple[int, ...]
    terminal_token_ids: tuple[int, ...]

    def validated(self) -> AsgcvCompletionProtocol:
        same = _token_tuple(self.same_prefix_ids, name="same verdict prefix")
        different = _token_tuple(self.different_prefix_ids, name="different verdict prefix")
        terminals = _token_tuple(
            self.terminal_token_ids,
            name="terminal token IDs",
            allow_empty=True,
        )
        if (
            same == different
            or same[: len(different)] == different
            or different[: len(same)] == same
        ):
            raise ValueError("ASG-CV verdict prefixes are ambiguous")
        if terminals != tuple(sorted(set(terminals))):
            raise ValueError("ASG-CV terminal token IDs differ")
        return self

    def to_mapping(self) -> dict[str, object]:
        self.validated()
        return {
            "schema": ASGCV_COMPLETION_PROTOCOL_SCHEMA,
            "same_prefix_ids": list(self.same_prefix_ids),
            "different_prefix_ids": list(self.different_prefix_ids),
            "terminal_token_ids": list(self.terminal_token_ids),
        }

    @classmethod
    def from_mapping(cls, value: object) -> AsgcvCompletionProtocol:
        if type(value) is not dict or set(value) != {
            "schema",
            "same_prefix_ids",
            "different_prefix_ids",
            "terminal_token_ids",
        }:
            raise ValueError("ASG-CV completion protocol schema differs")
        if value["schema"] != ASGCV_COMPLETION_PROTOCOL_SCHEMA:
            raise ValueError("ASG-CV completion protocol authority differs")
        sequences = []
        for name in ("same_prefix_ids", "different_prefix_ids", "terminal_token_ids"):
            raw = value[name]
            if type(raw) is not list:
                raise ValueError("ASG-CV completion protocol token IDs differ")
            sequences.append(tuple(raw))
        return cls(*sequences).validated()

    def sha256(self) -> str:
        payload = ASGCV_COMPLETION_PROTOCOL_DOMAIN + _canonical_json_bytes(self.to_mapping())
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class AsgcvCompletionClassification:
    """Outcome-blind parse of one generated completion."""

    valid: bool
    verdict_relation_sign: int | None
    reward: int
    attribute_span: tuple[int, int] | None


@dataclass(frozen=True, slots=True)
class AsgcvRolloutAuthority:
    """Source-bound sampler configuration for all candidate completion groups."""

    master_seed_sha256: str
    model_revision: str
    temperature: float
    top_p: float
    max_new_tokens: int

    def validated(self) -> AsgcvRolloutAuthority:
        _sha256_seed(self.master_seed_sha256, name="rollout master seed")
        if (
            type(self.model_revision) is not str
            or len(self.model_revision) != 40
            or any(character not in "0123456789abcdef" for character in self.model_revision)
        ):
            raise ValueError("ASG-CV rollout model revision differs")
        if (
            type(self.temperature) is not float
            or not 0.0 < self.temperature <= 2.0
            or type(self.top_p) is not float
            or not 0.0 < self.top_p <= 1.0
            or type(self.max_new_tokens) is not int
            or not 0 < self.max_new_tokens <= 4096
        ):
            raise ValueError("ASG-CV rollout sampler differs")
        return self

    def to_mapping(self) -> dict[str, object]:
        self.validated()
        return {
            "schema": ASGCV_ROLLOUT_AUTHORITY_SCHEMA,
            "master_seed_sha256": self.master_seed_sha256,
            "model_revision": self.model_revision,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_new_tokens": self.max_new_tokens,
            "rollouts_per_pair": ASGCV_STRATUM_SIZE,
        }

    @classmethod
    def from_mapping(cls, value: object) -> AsgcvRolloutAuthority:
        expected = {
            "schema",
            "master_seed_sha256",
            "model_revision",
            "temperature",
            "top_p",
            "max_new_tokens",
            "rollouts_per_pair",
        }
        if (
            type(value) is not dict
            or set(value) != expected
            or value["schema"] != ASGCV_ROLLOUT_AUTHORITY_SCHEMA
            or value["rollouts_per_pair"] != ASGCV_STRATUM_SIZE
            or type(value["rollouts_per_pair"]) is not int
        ):
            raise ValueError("ASG-CV rollout authority schema differs")
        return cls(
            master_seed_sha256=value["master_seed_sha256"],
            model_revision=value["model_revision"],
            temperature=value["temperature"],
            top_p=value["top_p"],
            max_new_tokens=value["max_new_tokens"],
        ).validated()

    def sha256(self) -> str:
        return hashlib.sha256(
            ASGCV_ROLLOUT_AUTHORITY_DOMAIN + _canonical_json_bytes(self.to_mapping())
        ).hexdigest()


def _class_band(value: object, *, name: str) -> tuple[int, ...]:
    if (
        type(value) is not tuple
        or not value
        or any(type(class_id) is not int or class_id < 0 for class_id in value)
        or value != tuple(sorted(set(value)))
    ):
        raise ValueError(f"ASG-CV {name} class band differs")
    return value


@dataclass(frozen=True, slots=True)
class AsgcvPartitionAuthority:
    """Sealed, disjoint class bands for every ASG-CV scientific phase."""

    source_manifest_sha256: str
    partition_seed_sha256: str
    predictor_train_class_ids: tuple[int, ...]
    e0_validation_class_ids: tuple[int, ...]
    e1_optimization_class_ids: tuple[int, ...]

    def validated(self) -> AsgcvPartitionAuthority:
        _sha256_seed(self.source_manifest_sha256, name="source manifest digest")
        _sha256_seed(self.partition_seed_sha256, name="partition seed")
        bands = (
            _class_band(self.predictor_train_class_ids, name="predictor training"),
            _class_band(self.e0_validation_class_ids, name="E0 validation"),
            _class_band(self.e1_optimization_class_ids, name="E1 optimization"),
        )
        flattened = tuple(class_id for band in bands for class_id in band)
        if len(flattened) != len(set(flattened)):
            raise ValueError("ASG-CV partition class bands overlap")
        return self

    def to_mapping(self) -> dict[str, object]:
        self.validated()
        return {
            "schema": ASGCV_PARTITION_AUTHORITY_SCHEMA,
            "source_manifest_sha256": self.source_manifest_sha256,
            "partition_seed_sha256": self.partition_seed_sha256,
            "predictor_train_class_ids": list(self.predictor_train_class_ids),
            "e0_validation_class_ids": list(self.e0_validation_class_ids),
            "e1_optimization_class_ids": list(self.e1_optimization_class_ids),
            "official_test_accessible": False,
        }

    @classmethod
    def from_mapping(cls, value: object) -> AsgcvPartitionAuthority:
        expected = {
            "schema",
            "source_manifest_sha256",
            "partition_seed_sha256",
            "predictor_train_class_ids",
            "e0_validation_class_ids",
            "e1_optimization_class_ids",
            "official_test_accessible",
        }
        if (
            type(value) is not dict
            or set(value) != expected
            or value["schema"] != ASGCV_PARTITION_AUTHORITY_SCHEMA
            or value["official_test_accessible"] is not False
        ):
            raise ValueError("ASG-CV partition authority schema differs")
        raw_bands = (
            value["predictor_train_class_ids"],
            value["e0_validation_class_ids"],
            value["e1_optimization_class_ids"],
        )
        if any(type(band) is not list for band in raw_bands):
            raise ValueError("ASG-CV partition class bands differ")
        return cls(
            source_manifest_sha256=value["source_manifest_sha256"],
            partition_seed_sha256=value["partition_seed_sha256"],
            predictor_train_class_ids=tuple(raw_bands[0]),
            e0_validation_class_ids=tuple(raw_bands[1]),
            e1_optimization_class_ids=tuple(raw_bands[2]),
        ).validated()

    def sha256(self) -> str:
        return hashlib.sha256(
            ASGCV_PARTITION_AUTHORITY_DOMAIN + _canonical_json_bytes(self.to_mapping())
        ).hexdigest()


def _partition_rows(
    value: object,
    *,
    name: str,
    allowed_classes: tuple[int, ...],
) -> tuple[str, ...]:
    if type(value) is not tuple or len(value) != 2:
        raise ValueError(f"ASG-CV {name} partition differs")
    example_ids, labels = value
    if (
        type(example_ids) is not tuple
        or not example_ids
        or any(type(example_id) is not str or not example_id for example_id in example_ids)
        or len(set(example_ids)) != len(example_ids)
        or type(labels) is not tuple
        or len(labels) != len(example_ids)
        or any(type(label) is not int or label not in allowed_classes for label in labels)
    ):
        raise ValueError(f"ASG-CV {name} partition differs")
    return example_ids


def validate_asgcv_partition_bundle(
    authority: object,
    *,
    predictor_train: object,
    e0_validation: object,
    e1_optimization: object,
) -> None:
    """Require phase class bands and source-image identities to be disjoint."""

    if type(authority) is not AsgcvPartitionAuthority:
        raise ValueError("ASG-CV partition authority differs")
    authority.validated()
    phase_ids = (
        _partition_rows(
            predictor_train,
            name="predictor training",
            allowed_classes=authority.predictor_train_class_ids,
        ),
        _partition_rows(
            e0_validation,
            name="E0 validation",
            allowed_classes=authority.e0_validation_class_ids,
        ),
        _partition_rows(
            e1_optimization,
            name="E1 optimization",
            allowed_classes=authority.e1_optimization_class_ids,
        ),
    )
    flattened = tuple(example_id for ids in phase_ids for example_id in ids)
    if len(flattened) != len(set(flattened)):
        raise ValueError("ASG-CV partition image identities overlap")


def derive_asgcv_rollout_seeds(
    authority: object,
    *,
    candidate_pair_ordinal: int,
) -> tuple[int, ...]:
    """Derive eight pair-unique generation seeds without consulting outcomes."""

    if type(authority) is not AsgcvRolloutAuthority:
        raise ValueError("ASG-CV rollout authority differs")
    authority.validated()
    if (
        type(candidate_pair_ordinal) is not int
        or not 0 <= candidate_pair_ordinal < 2**64
    ):
        raise ValueError("ASG-CV rollout candidate ordinal differs")
    authority_digest = bytes.fromhex(authority.sha256())
    seeds = tuple(
        int.from_bytes(
            hashlib.sha256(
                ASGCV_ROLLOUT_SEED_DOMAIN
                + authority_digest
                + candidate_pair_ordinal.to_bytes(8, "big")
                + rollout_ordinal.to_bytes(8, "big")
            ).digest()[:8],
            "big",
        )
        for rollout_ordinal in range(ASGCV_STRATUM_SIZE)
    )
    if len(set(seeds)) != ASGCV_STRATUM_SIZE:
        raise ValueError("ASG-CV rollout seed collision")
    return seeds


@dataclass(frozen=True, slots=True)
class AsgcvCompletionGroup:
    """Eight classified completions and their exact semantic eligibility."""

    completion_ids: tuple[tuple[int, ...], ...]
    expected_relation_sign: int
    protocol_sha256: str
    rollout_authority_sha256: str
    candidate_pair_ordinal: int
    generation_seeds: tuple[int, ...]
    rewards: tuple[int, ...]
    correct_rollouts: tuple[bool, ...]
    attribute_spans: tuple[tuple[int, int] | None, ...]
    nonzero_reward_variance: bool

    def validated(self) -> AsgcvCompletionGroup:
        if (
            type(self.completion_ids) is not tuple
            or len(self.completion_ids) != ASGCV_STRATUM_SIZE
            or type(self.expected_relation_sign) is not int
            or self.expected_relation_sign not in {-1, 1}
            or type(self.candidate_pair_ordinal) is not int
            or not 0 <= self.candidate_pair_ordinal < 2**64
            or type(self.generation_seeds) is not tuple
            or len(self.generation_seeds) != ASGCV_STRATUM_SIZE
            or any(type(seed) is not int or not 0 <= seed < 2**64 for seed in self.generation_seeds)
            or len(set(self.generation_seeds)) != ASGCV_STRATUM_SIZE
            or type(self.rewards) is not tuple
            or len(self.rewards) != ASGCV_STRATUM_SIZE
            or any(type(value) is not int or value not in {0, 1} for value in self.rewards)
            or type(self.correct_rollouts) is not tuple
            or len(self.correct_rollouts) != ASGCV_STRATUM_SIZE
            or any(type(value) is not bool for value in self.correct_rollouts)
            or self.correct_rollouts != tuple(value == 1 for value in self.rewards)
            or type(self.attribute_spans) is not tuple
            or len(self.attribute_spans) != ASGCV_STRATUM_SIZE
        ):
            raise ValueError("ASG-CV completion group authority differs")
        _sha256_seed(self.protocol_sha256, name="completion protocol digest")
        _sha256_seed(self.rollout_authority_sha256, name="rollout authority digest")
        for completion, correct, span in zip(
            self.completion_ids,
            self.correct_rollouts,
            self.attribute_spans,
            strict=True,
        ):
            _token_tuple(completion, name="completion token IDs")
            if correct:
                if (
                    type(span) is not tuple
                    or len(span) != 2
                    or any(type(value) is not int for value in span)
                    or not 0 <= span[0] < span[1] <= len(completion)
                ):
                    raise ValueError("ASG-CV completion group attribute span differs")
            elif span is not None:
                raise ValueError("ASG-CV completion group teacher span differs")
        expected_variance = 0 < sum(self.rewards) < ASGCV_STRATUM_SIZE
        if (
            type(self.nonzero_reward_variance) is not bool
            or self.nonzero_reward_variance is not expected_variance
        ):
            raise ValueError("ASG-CV completion group variance differs")
        return self

    def to_mapping(self) -> dict[str, object]:
        self.validated()
        return {
            "schema": ASGCV_COMPLETION_GROUP_SCHEMA,
            "completion_ids": [list(completion) for completion in self.completion_ids],
            "expected_relation_sign": self.expected_relation_sign,
            "protocol_sha256": self.protocol_sha256,
            "rollout_authority_sha256": self.rollout_authority_sha256,
            "candidate_pair_ordinal": self.candidate_pair_ordinal,
            "generation_seeds": list(self.generation_seeds),
            "rewards": list(self.rewards),
            "correct_rollouts": list(self.correct_rollouts),
            "attribute_spans": [
                None if span is None else list(span) for span in self.attribute_spans
            ],
            "nonzero_reward_variance": self.nonzero_reward_variance,
        }

    @classmethod
    def from_mapping(cls, value: object) -> AsgcvCompletionGroup:
        expected = {
            "schema",
            "completion_ids",
            "expected_relation_sign",
            "protocol_sha256",
            "rollout_authority_sha256",
            "candidate_pair_ordinal",
            "generation_seeds",
            "rewards",
            "correct_rollouts",
            "attribute_spans",
            "nonzero_reward_variance",
        }
        if type(value) is not dict or set(value) != expected:
            raise ValueError("ASG-CV completion group schema differs")
        if value["schema"] != ASGCV_COMPLETION_GROUP_SCHEMA:
            raise ValueError("ASG-CV completion group authority differs")
        raw_completions = value["completion_ids"]
        raw_rewards = value["rewards"]
        raw_correct = value["correct_rollouts"]
        raw_spans = value["attribute_spans"]
        raw_seeds = value["generation_seeds"]
        rows = (raw_completions, raw_rewards, raw_correct, raw_spans, raw_seeds)
        if any(type(raw) is not list for raw in rows):
            raise ValueError("ASG-CV completion group rows differ")
        completions = tuple(tuple(row) if type(row) is list else () for row in raw_completions)
        spans = tuple(
            tuple(span) if type(span) is list else None for span in raw_spans
        )
        return cls(
            completion_ids=completions,
            expected_relation_sign=value["expected_relation_sign"],
            protocol_sha256=value["protocol_sha256"],
            rollout_authority_sha256=value["rollout_authority_sha256"],
            candidate_pair_ordinal=value["candidate_pair_ordinal"],
            generation_seeds=tuple(raw_seeds),
            rewards=tuple(raw_rewards),
            correct_rollouts=tuple(raw_correct),
            attribute_spans=spans,
            nonzero_reward_variance=value["nonzero_reward_variance"],
        ).validated()

    def sha256(self) -> str:
        return hashlib.sha256(
            ASGCV_COMPLETION_GROUP_DOMAIN + _canonical_json_bytes(self.to_mapping())
        ).hexdigest()


@dataclass(frozen=True, slots=True)
class AsgcvPair:
    """One immutable image-disjoint semantic pair."""

    ordinal: int
    left_index: int
    right_index: int
    relation_sign: int


@dataclass(frozen=True, slots=True)
class AsgcvPairSchedule:
    """Outcome-blind pair schedule over one authenticated example manifest."""

    example_manifest_sha256: str
    schedule_seed_sha256: str
    pairs: tuple[AsgcvPair, ...]

    @property
    def pair_count(self) -> int:
        return len(self.pairs)

    def validated(self) -> AsgcvPairSchedule:
        _sha256_seed(self.example_manifest_sha256, name="example manifest digest")
        _sha256_seed(self.schedule_seed_sha256, name="pair schedule seed")
        if (
            type(self.pairs) is not tuple
            or not self.pairs
            or len(self.pairs) % ASGCV_STRATUM_SIZE != 0
        ):
            raise ValueError("ASG-CV pair schedule count differs")
        used: set[int] = set()
        for ordinal, pair in enumerate(self.pairs):
            if (
                type(pair) is not AsgcvPair
                or type(pair.ordinal) is not int
                or pair.ordinal != ordinal
                or type(pair.left_index) is not int
                or type(pair.right_index) is not int
                or pair.left_index < 0
                or pair.right_index < 0
                or pair.left_index == pair.right_index
                or type(pair.relation_sign) is not int
                or pair.relation_sign not in {-1, 1}
                or pair.left_index in used
                or pair.right_index in used
            ):
                raise ValueError("ASG-CV pair schedule row differs")
            used.update((pair.left_index, pair.right_index))
        for offset in range(0, len(self.pairs), ASGCV_STRATUM_SIZE):
            signs = [pair.relation_sign for pair in self.pairs[offset : offset + 8]]
            if signs.count(1) != 4 or signs.count(-1) != 4:
                raise ValueError("ASG-CV pair stratum balance differs")
        return self

    def to_mapping(self) -> dict[str, object]:
        self.validated()
        return {
            "schema": ASGCV_PAIR_SCHEDULE_SCHEMA,
            "example_manifest_sha256": self.example_manifest_sha256,
            "schedule_seed_sha256": self.schedule_seed_sha256,
            "pairs": [
                {
                    "ordinal": pair.ordinal,
                    "left_index": pair.left_index,
                    "right_index": pair.right_index,
                    "relation_sign": pair.relation_sign,
                }
                for pair in self.pairs
            ],
        }

    @classmethod
    def from_mapping(cls, value: object) -> AsgcvPairSchedule:
        if type(value) is not dict or set(value) != {
            "schema",
            "example_manifest_sha256",
            "schedule_seed_sha256",
            "pairs",
        }:
            raise ValueError("ASG-CV pair schedule schema differs")
        if value["schema"] != ASGCV_PAIR_SCHEDULE_SCHEMA:
            raise ValueError("ASG-CV pair schedule authority differs")
        raw_pairs = value["pairs"]
        if type(raw_pairs) is not list:
            raise ValueError("ASG-CV pair schedule rows differ")
        pairs = []
        for raw in raw_pairs:
            if type(raw) is not dict or set(raw) != {
                "ordinal",
                "left_index",
                "right_index",
                "relation_sign",
            }:
                raise ValueError("ASG-CV pair schedule row schema differs")
            pairs.append(
                AsgcvPair(
                    ordinal=raw["ordinal"],
                    left_index=raw["left_index"],
                    right_index=raw["right_index"],
                    relation_sign=raw["relation_sign"],
                )
            )
        return cls(
            example_manifest_sha256=value["example_manifest_sha256"],
            schedule_seed_sha256=value["schedule_seed_sha256"],
            pairs=tuple(pairs),
        ).validated()

    def sha256(self) -> str:
        return hashlib.sha256(
            ASGCV_PAIR_SCHEDULE_DOMAIN + _canonical_json_bytes(self.to_mapping())
        ).hexdigest()


@dataclass(frozen=True, slots=True)
class AsgcvEligibleSchedule:
    """First eligible candidate rows sealed before exact-gradient replay."""

    candidate_schedule_sha256: str
    target_pair_count: int
    candidate_ordinals: tuple[int, ...]

    def validated(self) -> AsgcvEligibleSchedule:
        _sha256_seed(self.candidate_schedule_sha256, name="candidate schedule digest")
        if (
            type(self.target_pair_count) is not int
            or self.target_pair_count <= 0
            or self.target_pair_count % ASGCV_STRATUM_SIZE != 0
            or type(self.candidate_ordinals) is not tuple
            or len(self.candidate_ordinals) != self.target_pair_count
            or any(type(value) is not int or value < 0 for value in self.candidate_ordinals)
            or len(set(self.candidate_ordinals)) != len(self.candidate_ordinals)
        ):
            raise ValueError("ASG-CV eligible schedule authority differs")
        return self

    def to_mapping(self) -> dict[str, object]:
        self.validated()
        return {
            "schema": ASGCV_ELIGIBLE_SCHEDULE_SCHEMA,
            "candidate_schedule_sha256": self.candidate_schedule_sha256,
            "target_pair_count": self.target_pair_count,
            "candidate_ordinals": list(self.candidate_ordinals),
        }

    @classmethod
    def from_mapping(cls, value: object) -> AsgcvEligibleSchedule:
        if type(value) is not dict or set(value) != {
            "schema",
            "candidate_schedule_sha256",
            "target_pair_count",
            "candidate_ordinals",
        }:
            raise ValueError("ASG-CV eligible schedule schema differs")
        if value["schema"] != ASGCV_ELIGIBLE_SCHEDULE_SCHEMA:
            raise ValueError("ASG-CV eligible schedule authority differs")
        ordinals = value["candidate_ordinals"]
        if type(ordinals) is not list:
            raise ValueError("ASG-CV eligible schedule rows differ")
        return cls(
            candidate_schedule_sha256=value["candidate_schedule_sha256"],
            target_pair_count=value["target_pair_count"],
            candidate_ordinals=tuple(ordinals),
        ).validated()

    def sha256(self) -> str:
        return hashlib.sha256(
            ASGCV_ELIGIBLE_SCHEDULE_DOMAIN + _canonical_json_bytes(self.to_mapping())
        ).hexdigest()


def assemble_asgcv_eligible_schedule(
    candidates: AsgcvPairSchedule,
    groups: tuple[AsgcvCompletionGroup, ...],
    *,
    target_pair_count: int,
) -> AsgcvEligibleSchedule:
    """Take first variance-eligible rows per relation before opening gradients."""

    if type(candidates) is not AsgcvPairSchedule:
        raise ValueError("ASG-CV candidate schedule differs")
    candidates.validated()
    if type(groups) is not tuple or len(groups) != candidates.pair_count:
        raise ValueError("ASG-CV candidate completion groups differ")
    if (
        type(target_pair_count) is not int
        or target_pair_count <= 0
        or target_pair_count % ASGCV_STRATUM_SIZE != 0
        or target_pair_count > candidates.pair_count
    ):
        raise ValueError("ASG-CV eligible target count differs")
    eligible: dict[int, list[int]] = {-1: [], 1: []}
    for pair, group in zip(candidates.pairs, groups, strict=True):
        if type(group) is not AsgcvCompletionGroup:
            raise ValueError("ASG-CV candidate completion group differs")
        group.validated()
        if group.expected_relation_sign != pair.relation_sign:
            raise ValueError("ASG-CV candidate relation binding differs")
        if group.nonzero_reward_variance:
            eligible[pair.relation_sign].append(pair.ordinal)
    needed_per_relation = target_pair_count // 2
    if any(len(eligible[sign]) < needed_per_relation for sign in (-1, 1)):
        raise ValueError("ASG-CV eligible pair capacity differs")
    selected: list[int] = []
    relations_per_stratum = ASGCV_STRATUM_SIZE // 2
    for offset in range(0, needed_per_relation, relations_per_stratum):
        block = (
            eligible[1][offset : offset + relations_per_stratum]
            + eligible[-1][offset : offset + relations_per_stratum]
        )
        selected.extend(sorted(block))
    return AsgcvEligibleSchedule(
        candidate_schedule_sha256=candidates.sha256(),
        target_pair_count=target_pair_count,
        candidate_ordinals=tuple(selected),
    ).validated()


def validate_asgcv_protocol_bundle(
    protocol: AsgcvCompletionProtocol,
    rollout_authority: AsgcvRolloutAuthority,
    candidates: AsgcvPairSchedule,
    groups: tuple[AsgcvCompletionGroup, ...],
    eligible: AsgcvEligibleSchedule,
    *,
    example_ids: tuple[str, ...],
    labels: tuple[int, ...],
) -> None:
    """Rebuild every semantic protocol relation from its primitive authority."""

    if (
        type(protocol) is not AsgcvCompletionProtocol
        or type(rollout_authority) is not AsgcvRolloutAuthority
        or type(candidates) is not AsgcvPairSchedule
        or type(groups) is not tuple
        or type(eligible) is not AsgcvEligibleSchedule
    ):
        raise ValueError("ASG-CV protocol bundle differs")
    protocol.validated()
    rollout_authority.validated()
    candidates.validated()
    eligible.validated()
    rebuilt_candidates = build_asgcv_pair_schedule(
        example_ids,
        labels,
        schedule_seed_sha256=candidates.schedule_seed_sha256,
        pair_count=candidates.pair_count,
    )
    if rebuilt_candidates != candidates:
        raise ValueError("ASG-CV candidate schedule reconstruction differs")
    if len(groups) != candidates.pair_count:
        raise ValueError("ASG-CV completion group count differs")
    for pair, group in zip(candidates.pairs, groups, strict=True):
        if type(group) is not AsgcvCompletionGroup:
            raise ValueError("ASG-CV completion group differs")
        group.validated()
        rebuilt_group = classify_asgcv_completion_group(
            group.completion_ids,
            pair.relation_sign,
            protocol,
            rollout_authority=rollout_authority,
            candidate_pair_ordinal=pair.ordinal,
        )
        if rebuilt_group != group:
            raise ValueError("ASG-CV completion group reconstruction differs")
    rebuilt_eligible = assemble_asgcv_eligible_schedule(
        candidates,
        groups,
        target_pair_count=eligible.target_pair_count,
    )
    if rebuilt_eligible != eligible:
        raise ValueError("ASG-CV eligible schedule reconstruction differs")


def _sha256_seed(value: object, *, name: str) -> bytes:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"ASG-CV {name} differs")
    return bytes.fromhex(value)


def _key(seed: bytes, role: bytes, *values: str) -> bytes:
    payload = bytearray(ASGCV_PAIR_SCHEDULE_DOMAIN)
    payload.extend(seed)
    payload.extend(len(role).to_bytes(8, "big"))
    payload.extend(role)
    for value in values:
        encoded = value.encode("utf-8")
        payload.extend(len(encoded).to_bytes(8, "big"))
        payload.extend(encoded)
    return hashlib.sha256(payload).digest()


def build_asgcv_pair_schedule(
    example_ids: tuple[str, ...],
    labels: tuple[int, ...],
    *,
    schedule_seed_sha256: object,
    pair_count: int,
) -> AsgcvPairSchedule:
    """Build an equal-relation, image-disjoint, stratum-aligned pair schedule."""

    if (
        type(example_ids) is not tuple
        or not example_ids
        or any(type(value) is not str or not value for value in example_ids)
        or len(set(example_ids)) != len(example_ids)
        or type(labels) is not tuple
        or len(labels) != len(example_ids)
        or any(type(value) is not int or value < 0 for value in labels)
        or len(set(labels)) < 2
    ):
        raise ValueError("ASG-CV pair manifest differs")
    if (
        type(pair_count) is not int
        or pair_count <= 0
        or pair_count % ASGCV_STRATUM_SIZE != 0
        or pair_count % 2 != 0
        or 2 * pair_count > len(example_ids)
    ):
        raise ValueError("ASG-CV pair schedule count differs")
    seed = _sha256_seed(schedule_seed_sha256, name="pair schedule seed")
    grouped: dict[int, list[int]] = {}
    for index, label in enumerate(labels):
        grouped.setdefault(label, []).append(index)
    positive_candidates: list[tuple[int, int]] = []
    for label in sorted(grouped):
        ranked = sorted(
            grouped[label],
            key=lambda index: (
                _key(seed, b"positive-example", example_ids[index]),
                example_ids[index],
            ),
        )
        positive_candidates.extend(
            (ranked[offset], ranked[offset + 1])
            for offset in range(0, len(ranked) - 1, 2)
        )
    positive_candidates.sort(
        key=lambda pair: (
            _key(
                seed,
                b"positive-pair",
                example_ids[pair[0]],
                example_ids[pair[1]],
            ),
            pair,
        )
    )
    needed_per_relation = pair_count // 2
    positives = positive_candidates[:needed_per_relation]
    if len(positives) != needed_per_relation:
        raise ValueError("ASG-CV positive pair capacity differs")
    used = {index for pair in positives for index in pair}
    remaining: dict[int, list[int]] = {}
    for index, label in enumerate(labels):
        if index not in used:
            remaining.setdefault(label, []).append(index)
    for label, indices in remaining.items():
        indices.sort(
            key=lambda index: (
                _key(seed, b"negative-example", str(label), example_ids[index]),
                example_ids[index],
            ),
            reverse=True,
        )
    negatives: list[tuple[int, int]] = []
    for negative_ordinal in range(needed_per_relation):
        available = [label for label, indices in remaining.items() if indices]
        if len(available) < 2:
            raise ValueError("ASG-CV negative pair capacity differs")
        ranked_labels = sorted(
            available,
            key=lambda label: (
                -len(remaining[label]),
                _key(seed, b"negative-label", str(negative_ordinal), str(label)),
                label,
            ),
        )
        left_label, right_label = ranked_labels[:2]
        left = remaining[left_label].pop()
        right = remaining[right_label].pop()
        if _key(seed, b"negative-orientation", example_ids[left], example_ids[right])[0] & 1:
            left, right = right, left
        negatives.append((left, right))
    positive_rows = [(left, right, 1) for left, right in positives]
    negative_rows = [(left, right, -1) for left, right in negatives]

    def pair_order_key(pair: tuple[int, int, int]) -> tuple[bytes, tuple[int, int, int]]:
        return (
            _key(
                seed,
                b"pair-order",
                example_ids[pair[0]],
                example_ids[pair[1]],
                str(pair[2]),
            ),
            pair,
        )

    positive_rows.sort(key=pair_order_key)
    negative_rows.sort(key=pair_order_key)
    staged: list[tuple[int, int, int]] = []
    relations_per_stratum = ASGCV_STRATUM_SIZE // 2
    for offset in range(0, needed_per_relation, relations_per_stratum):
        stratum = (
            positive_rows[offset : offset + relations_per_stratum]
            + negative_rows[offset : offset + relations_per_stratum]
        )
        stratum.sort(key=pair_order_key)
        staged.extend(stratum)
    pairs = tuple(
        AsgcvPair(
            ordinal=ordinal,
            left_index=left,
            right_index=right,
            relation_sign=relation,
        )
        for ordinal, (left, right, relation) in enumerate(staged)
    )
    manifest: dict[str, object] = {
        "examples": [
            {"example_id": example_id, "label": label}
            for example_id, label in zip(example_ids, labels, strict=True)
        ]
    }
    manifest_digest = hashlib.sha256(_canonical_json_bytes(manifest)).hexdigest()
    return AsgcvPairSchedule(
        example_manifest_sha256=manifest_digest,
        schedule_seed_sha256=str(schedule_seed_sha256),
        pairs=pairs,
    )


def classify_asgcv_completion(
    completion_ids: tuple[int, ...],
    expected_relation_sign: int,
    protocol: AsgcvCompletionProtocol,
) -> AsgcvCompletionClassification:
    """Classify an exact completion without decoding or regex heuristics."""

    completion = _token_tuple(completion_ids, name="completion token IDs")
    if type(expected_relation_sign) is not int or expected_relation_sign not in {-1, 1}:
        raise ValueError("ASG-CV expected relation sign differs")
    if type(protocol) is not AsgcvCompletionProtocol:
        raise ValueError("ASG-CV completion protocol differs")
    protocol.validated()
    end = len(completion)
    terminals = set(protocol.terminal_token_ids)
    while end and completion[end - 1] in terminals:
        end -= 1
    body = completion[:end]
    verdict: int | None = None
    prefix_length = 0
    if body[: len(protocol.same_prefix_ids)] == protocol.same_prefix_ids:
        verdict = 1
        prefix_length = len(protocol.same_prefix_ids)
    elif body[: len(protocol.different_prefix_ids)] == protocol.different_prefix_ids:
        verdict = -1
        prefix_length = len(protocol.different_prefix_ids)
    if verdict is None or end <= prefix_length:
        return AsgcvCompletionClassification(
            valid=False,
            verdict_relation_sign=None,
            reward=0,
            attribute_span=None,
        )
    return AsgcvCompletionClassification(
        valid=True,
        verdict_relation_sign=verdict,
        reward=int(verdict == expected_relation_sign),
        attribute_span=(prefix_length, end),
    )


def classify_asgcv_completion_group(
    completion_ids: tuple[tuple[int, ...], ...],
    expected_relation_sign: int,
    protocol: AsgcvCompletionProtocol,
    *,
    rollout_authority: AsgcvRolloutAuthority,
    candidate_pair_ordinal: int,
) -> AsgcvCompletionGroup:
    """Classify one exact eight-rollout group and derive DAPO eligibility."""

    if type(completion_ids) is not tuple or len(completion_ids) != ASGCV_STRATUM_SIZE:
        raise ValueError("ASG-CV completion group size differs")
    if type(rollout_authority) is not AsgcvRolloutAuthority:
        raise ValueError("ASG-CV rollout authority differs")
    rollout_authority.validated()
    generation_seeds = derive_asgcv_rollout_seeds(
        rollout_authority,
        candidate_pair_ordinal=candidate_pair_ordinal,
    )
    classifications = tuple(
        classify_asgcv_completion(completion, expected_relation_sign, protocol)
        for completion in completion_ids
    )
    rewards = tuple(value.reward for value in classifications)
    correct = tuple(value.reward == 1 for value in classifications)
    spans = tuple(
        value.attribute_span if value.reward == 1 else None for value in classifications
    )
    correct_count = sum(rewards)
    return AsgcvCompletionGroup(
        completion_ids=completion_ids,
        expected_relation_sign=expected_relation_sign,
        protocol_sha256=protocol.sha256(),
        rollout_authority_sha256=rollout_authority.sha256(),
        candidate_pair_ordinal=candidate_pair_ordinal,
        generation_seeds=generation_seeds,
        rewards=rewards,
        correct_rollouts=correct,
        attribute_spans=spans,
        nonzero_reward_variance=0 < correct_count < ASGCV_STRATUM_SIZE,
    ).validated()
