from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

from sfora.pass209_m4 import canonical_json_bytes
from sfora.prism_measurement import PRISM_CHANNELS
from sfora.prism_observer import (
    PrismChannelPrompt,
    PrismCompletionBundle,
    PrismCompletionRow,
    PrismObserverAuthority,
    PrismPayloadAuthority,
    PrismPromptBundle,
    canonical_prism_completion_bundle_bytes,
    canonical_prism_observer_authority_bytes,
    canonical_prism_prompt_bundle_bytes,
    derive_prism_token_protocol,
    validate_prism_completion_bundle_bytes,
    validate_prism_observer_authority_bytes,
    validate_prism_prompt_bundle_bytes,
)


def _prompt(channel: str) -> str:
    return (
        f"channel={channel}; compare the two anonymous vehicle images using only "
        "the named visual cue. Return the registered compact completion grammar."
    )


def _bundle() -> PrismPromptBundle:
    rows = tuple(
        PrismChannelPrompt(
            channel=channel,
            prompt_utf8=_prompt(channel),
            prompt_sha256=hashlib.sha256(_prompt(channel).encode("utf-8")).hexdigest(),
            max_new_tokens=192,
            temperature_ppm=1_000_000,
            top_p_ppm=1_000_000,
        )
        for channel in PRISM_CHANNELS
    )
    return PrismPromptBundle(schema="sfora-prism-prompt-bundle-v1", rows=rows)


def test_prompt_bundle_is_canonical_complete_and_round_trips() -> None:
    bundle = _bundle()

    raw = canonical_prism_prompt_bundle_bytes(bundle)

    assert raw.endswith(b"\n")
    assert raw == canonical_json_bytes(json.loads(raw))
    assert validate_prism_prompt_bundle_bytes(raw) == bundle
    assert tuple(row.channel for row in bundle.rows) == PRISM_CHANNELS
    assert all(row.max_new_tokens == 192 for row in bundle.rows)
    assert all(row.temperature_ppm == 1_000_000 for row in bundle.rows)
    assert all(row.top_p_ppm == 1_000_000 for row in bundle.rows)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda value: {**value, "extra": 1}, "schema"),
        (lambda value: {key: item for key, item in value.items() if key != "schema"}, "schema"),
        (
            lambda value: {
                **value,
                "rows": [
                    {**value["rows"][0], "max_new_tokens": True},
                    *value["rows"][1:],
                ],
            },
            "numeric",
        ),
        (
            lambda value: {
                **value,
                "rows": [value["rows"][1], value["rows"][0], *value["rows"][2:]],
            },
            "channel",
        ),
        (
            lambda value: {
                **value,
                "rows": [
                    value["rows"][0],
                    {
                        **value["rows"][1],
                        "prompt_utf8": value["rows"][0]["prompt_utf8"],
                    },
                    *value["rows"][2:],
                ],
            },
            "prompt",
        ),
        (
            lambda value: {
                **value,
                "rows": [
                    {**value["rows"][0], "prompt_utf8": _prompt(PRISM_CHANNELS[1])},
                    *value["rows"][1:],
                ],
            },
            "prompt",
        ),
        (
            lambda value: {
                **value,
                "rows": [
                    {**value["rows"][0], "prompt_sha256": "0" * 64},
                    *value["rows"][1:],
                ],
            },
            "digest",
        ),
    ),
)
def test_prompt_bundle_rejects_schema_type_order_binding_and_digest_drift(
    mutation: object, message: str
) -> None:
    value = json.loads(canonical_prism_prompt_bundle_bytes(_bundle()))
    mutated = mutation(value)  # type: ignore[operator]

    with pytest.raises(ValueError, match=message):
        validate_prism_prompt_bundle_bytes(canonical_json_bytes(mutated))


@pytest.mark.parametrize(
    "forbidden",
    (
        "class names",
        "labels",
        "fold",
        "Cars test",
        "clean",
        "Caliber",
        "Dodge",
        "2007",
        "2012",
    ),
)
def test_prompt_bundle_rejects_semantic_leakage(forbidden: str) -> None:
    bundle = _bundle()
    prompt = f"{bundle.rows[0].prompt_utf8} {forbidden}"
    row = replace(
        bundle.rows[0],
        prompt_utf8=prompt,
        prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
    )

    with pytest.raises(ValueError, match="forbidden"):
        canonical_prism_prompt_bundle_bytes(replace(bundle, rows=(row, *bundle.rows[1:])))


@pytest.mark.parametrize("bad", ("line\nbreak", "nul\x00byte", "tab\tbyte"))
def test_prompt_bundle_rejects_control_characters(bad: str) -> None:
    bundle = _bundle()
    prompt = f"{bundle.rows[0].prompt_utf8} {bad}"
    row = replace(
        bundle.rows[0],
        prompt_utf8=prompt,
        prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
    )

    with pytest.raises(ValueError, match="control"):
        canonical_prism_prompt_bundle_bytes(replace(bundle, rows=(row, *bundle.rows[1:])))


def test_prompt_bundle_rejects_duplicate_json_keys() -> None:
    raw = canonical_prism_prompt_bundle_bytes(_bundle())
    mutated = raw.replace(
        b'{"rows":', b'{"schema":"sfora-prism-prompt-bundle-v1","rows":', 1
    )

    with pytest.raises(ValueError, match="duplicate"):
        validate_prism_prompt_bundle_bytes(mutated)


def _observer_authority() -> PrismObserverAuthority:
    return PrismObserverAuthority(
        schema="sfora-prism-observer-authority-v1",
        source_commit="1" * 40,
        source_tree_sha256="2" * 64,
        dataset_revision="cars-train-revision-1",
        dataset_manifest_sha256="3" * 64,
        model_revision="qwen-snapshot-revision-1",
        observation_manifest_sha256="4" * 64,
        scoring_manifest_sha256="5" * 64,
        prompt_bundle_sha256="6" * 64,
        payload_manifest_sha256="7" * 64,
        row_count=1024,
    )


def test_observer_authority_is_canonical_and_concrete() -> None:
    authority = _observer_authority()

    raw = canonical_prism_observer_authority_bytes(authority)

    assert raw == canonical_json_bytes(json.loads(raw))
    assert validate_prism_observer_authority_bytes(raw) == authority
    assert PrismPayloadAuthority(
        payload_sha256="8" * 64,
        byte_length=101,
        width=8,
        height=6,
        mode="RGB",
    ).mode == "RGB"


@pytest.mark.parametrize(
    "mutation",
    (
        lambda value: {**value, "row_count": True},
        lambda value: {**value, "row_count": 0},
        lambda value: {**value, "prompt_bundle_sha256": "0" * 63},
        lambda value: {**value, "source_commit": "not-a-commit"},
        lambda value: {**value, "extra": "field"},
    ),
)
def test_observer_authority_rejects_type_digest_and_schema_drift(
    mutation: object,
) -> None:
    value = json.loads(canonical_prism_observer_authority_bytes(_observer_authority()))

    with pytest.raises(ValueError):
        validate_prism_observer_authority_bytes(canonical_json_bytes(mutation(value)))  # type: ignore[operator]


def _protocol_literals() -> tuple[str, ...]:
    return (
        *(f"channel={channel};" for channel in PRISM_CHANNELS),
        "left_visible=yes;right_visible=yes;",
        "left_visible=yes;right_visible=no;",
        "left_visible=no;right_visible=yes;",
        "left_visible=no;right_visible=no;",
        "relation=same;",
        "relation=different;",
        "relation=indeterminate;",
        "confidence=low;evidence_left=",
        "confidence=medium;evidence_left=",
        "confidence=high;evidence_left=",
        ";evidence_right=",
        "<PRISM_END>",
    )


class _Tokenizer:
    def __init__(self) -> None:
        self.mapping = {
            literal: (100 + index,) for index, literal in enumerate(_protocol_literals())
        }
        self.all_special_ids: tuple[int, ...] = ()
        self.decode_drift = False

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return list(self.mapping[text])

    def decode(self, token_ids: list[int], **_: object) -> str:
        if self.decode_drift:
            return "drift"
        target = tuple(token_ids)
        return next(text for text, ids in self.mapping.items() if ids == target)


class _Processor:
    def __init__(self, tokenizer: _Tokenizer) -> None:
        self.tokenizer = tokenizer


def test_token_protocol_is_derived_from_exact_literal_encodings() -> None:
    tokenizer = _Tokenizer()

    protocol = derive_prism_token_protocol(_Processor(tokenizer), _bundle())

    assert protocol.channel_prefixes == tuple((100 + index,) for index in range(8))
    assert protocol.visibility_prefixes == tuple((108 + index,) for index in range(4))
    assert protocol.relation_prefixes == tuple((112 + index,) for index in range(3))
    assert protocol.confidence_prefixes == tuple((115 + index,) for index in range(3))
    assert protocol.evidence_separator == (118,)
    assert protocol.terminal_tokens == (119,)
    assert protocol.max_evidence_tokens == 64


@pytest.mark.parametrize(
    "mutation", ("alias", "prefix", "empty", "negative", "bool", "special", "roundtrip")
)
def test_token_protocol_rejects_ambiguous_or_noncanonical_ids(mutation: str) -> None:
    tokenizer = _Tokenizer()
    literals = _protocol_literals()
    if mutation == "alias":
        tokenizer.mapping[literals[1]] = tokenizer.mapping[literals[0]]
    elif mutation == "prefix":
        tokenizer.mapping[literals[0]] = (700,)
        tokenizer.mapping[literals[1]] = (700, 701)
    elif mutation == "empty":
        tokenizer.mapping[literals[0]] = ()
    elif mutation == "negative":
        tokenizer.mapping[literals[0]] = (-1,)
    elif mutation == "bool":
        tokenizer.mapping[literals[0]] = (True,)
    elif mutation == "special":
        tokenizer.mapping[literals[0]] = (999,)
        tokenizer.all_special_ids = (999,)
    elif mutation == "roundtrip":
        tokenizer.decode_drift = True

    with pytest.raises(ValueError):
        derive_prism_token_protocol(_Processor(tokenizer), _bundle())


def _completion_bundle() -> PrismCompletionBundle:
    return PrismCompletionBundle(
        schema="sfora-prism-completion-bundle-v1",
        phase="calibration",
        observer_authority_sha256="1" * 64,
        token_protocol_sha256="2" * 64,
        rows=(
            PrismCompletionRow(
                pair_handle="3" * 64,
                channel=PRISM_CHANNELS[0],
                completion_ids=(101, 102, 103),
            ),
            PrismCompletionRow(
                pair_handle="4" * 64,
                channel=PRISM_CHANNELS[1],
                completion_ids=(201, 202),
            ),
        ),
    )


def test_completion_bundle_round_trips_without_decoded_text() -> None:
    bundle = _completion_bundle()

    raw = canonical_prism_completion_bundle_bytes(bundle)

    assert validate_prism_completion_bundle_bytes(raw) == bundle
    assert b"text" not in raw and b"evidence" not in raw and b"label" not in raw


@pytest.mark.parametrize(
    "mutation",
    (
        lambda value: {**value, "decoded_text": "forbidden"},
        lambda value: {**value, "phase": "unknown"},
        lambda value: {
            **value,
            "rows": [{**value["rows"][0], "completion_ids": [True]}, *value["rows"][1:]],
        },
        lambda value: {**value, "rows": [value["rows"][0], value["rows"][0]]},
    ),
)
def test_completion_bundle_rejects_text_type_phase_and_duplicate_drift(
    mutation: object,
) -> None:
    value = json.loads(canonical_prism_completion_bundle_bytes(_completion_bundle()))

    with pytest.raises(ValueError):
        validate_prism_completion_bundle_bytes(canonical_json_bytes(mutation(value)))  # type: ignore[operator]
