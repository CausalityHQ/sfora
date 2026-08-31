from __future__ import annotations

import pytest

from sfora.asgcv_protocol import (
    AsgcvCompletionProtocol,
    classify_asgcv_completion,
)


def _protocol() -> AsgcvCompletionProtocol:
    return AsgcvCompletionProtocol(
        same_prefix_ids=(11, 12),
        different_prefix_ids=(21, 22, 23),
        terminal_token_ids=(0, 99),
    ).validated()


def test_completion_protocol_is_exact_and_content_addressed() -> None:
    protocol = _protocol()
    assert protocol.to_mapping() == {
        "schema": "sfora-asgcv-completion-protocol-v1",
        "same_prefix_ids": [11, 12],
        "different_prefix_ids": [21, 22, 23],
        "terminal_token_ids": [0, 99],
    }
    assert AsgcvCompletionProtocol.from_mapping(protocol.to_mapping()) == protocol
    assert len(protocol.sha256()) == 64
    assert AsgcvCompletionProtocol.from_mapping(protocol.to_mapping()).sha256() == protocol.sha256()


def test_completion_classification_binds_reward_verdict_and_attribute_span() -> None:
    protocol = _protocol()

    same = classify_asgcv_completion((11, 12, 31, 32, 99), 1, protocol)
    assert same.reward == 1
    assert same.verdict_relation_sign == 1
    assert same.attribute_span == (2, 4)
    assert same.valid is True

    wrong = classify_asgcv_completion((21, 22, 23, 41, 0), 1, protocol)
    assert wrong.reward == 0
    assert wrong.verdict_relation_sign == -1
    assert wrong.attribute_span == (3, 4)
    assert wrong.valid is True

    malformed = classify_asgcv_completion((77, 78, 99), -1, protocol)
    assert malformed.reward == 0
    assert malformed.verdict_relation_sign is None
    assert malformed.attribute_span is None
    assert malformed.valid is False


def test_completion_protocol_rejects_ambiguous_empty_and_concrete_type_drift() -> None:
    for kwargs in (
        {"same_prefix_ids": (), "different_prefix_ids": (2,), "terminal_token_ids": (0,)},
        {
            "same_prefix_ids": (1,),
            "different_prefix_ids": (1, 2),
            "terminal_token_ids": (0,),
        },
        {"same_prefix_ids": (True,), "different_prefix_ids": (2,), "terminal_token_ids": (0,)},
        {"same_prefix_ids": (1,), "different_prefix_ids": (2,), "terminal_token_ids": (0, 0)},
    ):
        with pytest.raises(ValueError):
            AsgcvCompletionProtocol(**kwargs).validated()

    mapping = _protocol().to_mapping()
    with pytest.raises(ValueError):
        AsgcvCompletionProtocol.from_mapping({**mapping, "extra": 1})
    with pytest.raises(ValueError):
        AsgcvCompletionProtocol.from_mapping({**mapping, "same_prefix_ids": [11, True]})


def test_completion_classification_rejects_empty_attributes_and_input_drift() -> None:
    protocol = _protocol()
    for completion in ((11, 12, 99), (11, 12, 0, 99), (21, 22, 23)):
        observed = classify_asgcv_completion(completion, 1, protocol)
        assert observed.valid is False
        assert observed.reward == 0
        assert observed.attribute_span is None

    with pytest.raises(ValueError):
        classify_asgcv_completion([11, 12, 31], 1, protocol)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        classify_asgcv_completion((11, True, 31), 1, protocol)
    with pytest.raises(ValueError):
        classify_asgcv_completion((11, 12, 31), 0, protocol)

