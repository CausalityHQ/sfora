"""Tokenizer-ID authority for ASG-CV verdict rewards and attribute spans."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

ASGCV_COMPLETION_PROTOCOL_SCHEMA = "sfora-asgcv-completion-protocol-v1"
ASGCV_COMPLETION_PROTOCOL_DOMAIN = b"sfora-asgcv-completion-protocol-v1\0"


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

