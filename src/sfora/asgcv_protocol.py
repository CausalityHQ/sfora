"""Tokenizer-ID authority for ASG-CV verdict rewards and attribute spans."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

ASGCV_COMPLETION_PROTOCOL_SCHEMA = "sfora-asgcv-completion-protocol-v1"
ASGCV_COMPLETION_PROTOCOL_DOMAIN = b"sfora-asgcv-completion-protocol-v1\0"
ASGCV_PAIR_SCHEDULE_SCHEMA = "sfora-asgcv-pair-schedule-v1"
ASGCV_PAIR_SCHEDULE_DOMAIN = b"sfora-asgcv-pair-schedule-v1\0"
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
