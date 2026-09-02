"""Anonymous prompt and capability authority for the PRISM observer."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass

from sfora.pass209_m4 import canonical_json_bytes
from sfora.prism_measurement import (
    PRISM_CHANNELS,
    PrismChannelCalibration,
    PrismObservationCapabilityRow,
    PrismObservationRow,
    PrismScoringRow,
    PrismTokenProtocol,
    calibrate_prism_channels,
    invalid_prism_observation,
    parse_prism_completion,
    prism_calibration_receipt_sha256,
    release_prism_observation_capability,
)

_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_BUNDLE_KEYS = frozenset(("schema", "rows"))
_ROW_KEYS = frozenset(
    (
        "channel",
        "prompt_utf8",
        "prompt_sha256",
        "max_new_tokens",
        "temperature_ppm",
        "top_p_ppm",
    )
)
_FORBIDDEN_PROMPT_TERMS = (
    "class name",
    "label",
    "fold",
    "cars test",
    "clean",
    "caliber",
    "dodge",
    "2007",
    "2012",
)
_AUTHORITY_KEYS = frozenset(
    (
        "schema",
        "source_commit",
        "source_tree_sha256",
        "dataset_revision",
        "dataset_manifest_sha256",
        "model_revision",
        "observation_manifest_sha256",
        "scoring_manifest_sha256",
        "prompt_bundle_sha256",
        "payload_manifest_sha256",
        "row_count",
    )
)
_COMPLETION_BUNDLE_KEYS = frozenset(
    ("schema", "phase", "observer_authority_sha256", "token_protocol_sha256", "rows")
)
_COMPLETION_ROW_KEYS = frozenset(("pair_handle", "channel", "completion_ids"))


@dataclass(frozen=True, slots=True)
class PrismChannelPrompt:
    """One channel-bound prompt and its fixed sampling authority."""

    channel: str
    prompt_utf8: str
    prompt_sha256: str
    max_new_tokens: int
    temperature_ppm: int
    top_p_ppm: int


@dataclass(frozen=True, slots=True)
class PrismPromptBundle:
    """The complete ordered PRISM prompt bundle."""

    schema: str
    rows: tuple[PrismChannelPrompt, ...]


@dataclass(frozen=True, slots=True)
class PrismPayloadAuthority:
    """One anonymous RGB payload's physical authority."""

    payload_sha256: str
    byte_length: int
    width: int
    height: int
    mode: str


@dataclass(frozen=True, slots=True)
class PrismObserverAuthority:
    """Cross-object authority required by one observer phase."""

    schema: str
    source_commit: str
    source_tree_sha256: str
    dataset_revision: str
    dataset_manifest_sha256: str
    model_revision: str
    observation_manifest_sha256: str
    scoring_manifest_sha256: str
    prompt_bundle_sha256: str
    payload_manifest_sha256: str
    row_count: int


@dataclass(frozen=True, slots=True)
class PrismCompletionRow:
    """One ID-only observer completion with no decoded semantic material."""

    pair_handle: str
    channel: str
    completion_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class PrismCompletionBundle:
    """One phase's complete ordered ID-only observer output."""

    schema: str
    phase: str
    observer_authority_sha256: str
    token_protocol_sha256: str
    rows: tuple[PrismCompletionRow, ...]


@dataclass(frozen=True, slots=True)
class PrismDiagnosticRelease:
    """Calibration evidence and the resulting anonymous diagnostic capability."""

    calibrations: tuple[PrismChannelCalibration, ...]
    calibration_receipt_sha256: str
    capability: tuple[PrismObservationCapabilityRow, ...]


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _validate_prompt_bundle(bundle: PrismPromptBundle) -> None:
    if bundle.schema != "sfora-prism-prompt-bundle-v1":
        raise ValueError("prompt bundle schema differs")
    if tuple(row.channel for row in bundle.rows) != PRISM_CHANNELS:
        raise ValueError("prompt bundle channel order differs")
    prompts = tuple(row.prompt_utf8 for row in bundle.rows)
    if len(set(prompts)) != len(prompts):
        raise ValueError("prompt bundle contains duplicate prompt text")
    for row in bundle.rows:
        if (
            type(row.max_new_tokens) is not int
            or type(row.temperature_ppm) is not int
            or type(row.top_p_ppm) is not int
            or row.max_new_tokens != 192
            or row.temperature_ppm != 1_000_000
            or row.top_p_ppm != 1_000_000
        ):
            raise ValueError("prompt bundle numeric authority differs")
        if type(row.prompt_utf8) is not str or not row.prompt_utf8:
            raise ValueError("prompt text must be a nonempty UTF-8 string")
        if f"channel={row.channel};" not in row.prompt_utf8:
            raise ValueError("prompt channel binding differs")
        if any(ord(character) < 32 or ord(character) == 127 for character in row.prompt_utf8):
            raise ValueError("prompt contains a control character")
        lowered = row.prompt_utf8.casefold()
        if any(term in lowered for term in _FORBIDDEN_PROMPT_TERMS):
            raise ValueError("prompt contains forbidden semantic material")
        if (
            type(row.prompt_sha256) is not str
            or _SHA256.fullmatch(row.prompt_sha256) is None
            or hashlib.sha256(row.prompt_utf8.encode("utf-8")).hexdigest()
            != row.prompt_sha256
        ):
            raise ValueError("prompt digest differs")


def canonical_prism_prompt_bundle_bytes(bundle: PrismPromptBundle) -> bytes:
    """Validate and encode one sorted compact prompt bundle with one LF."""

    _validate_prompt_bundle(bundle)
    return canonical_json_bytes(asdict(bundle))


def validate_prism_prompt_bundle_bytes(raw: bytes) -> PrismPromptBundle:
    """Authenticate canonical bytes and return their typed prompt bundle."""

    if type(raw) is not bytes:
        raise ValueError("prompt bundle bytes must be concrete bytes")
    try:
        value = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("prompt bundle is not valid UTF-8 JSON") from error
    if type(value) is not dict or frozenset(value) != _BUNDLE_KEYS:
        raise ValueError("prompt bundle schema keys differ")
    if raw != canonical_json_bytes(value):
        raise ValueError("prompt bundle bytes are not canonical")
    rows_value = value["rows"]
    if type(rows_value) is not list:
        raise ValueError("prompt bundle rows schema differs")
    rows: list[PrismChannelPrompt] = []
    for row_value in rows_value:
        if type(row_value) is not dict or frozenset(row_value) != _ROW_KEYS:
            raise ValueError("prompt row schema differs")
        rows.append(PrismChannelPrompt(**row_value))
    bundle = PrismPromptBundle(schema=value["schema"], rows=tuple(rows))
    _validate_prompt_bundle(bundle)
    return bundle


def _validate_observer_authority(authority: PrismObserverAuthority) -> None:
    if authority.schema != "sfora-prism-observer-authority-v1":
        raise ValueError("observer authority schema differs")
    if (
        type(authority.source_commit) is not str
        or _COMMIT.fullmatch(authority.source_commit) is None
    ):
        raise ValueError("observer authority source commit differs")
    for name in (
        "source_tree_sha256",
        "dataset_manifest_sha256",
        "observation_manifest_sha256",
        "scoring_manifest_sha256",
        "prompt_bundle_sha256",
        "payload_manifest_sha256",
    ):
        value = getattr(authority, name)
        if type(value) is not str or _SHA256.fullmatch(value) is None:
            raise ValueError(f"observer authority {name} differs")
    for name in ("dataset_revision", "model_revision"):
        value = getattr(authority, name)
        if type(value) is not str or not value or any(ord(char) < 32 for char in value):
            raise ValueError(f"observer authority {name} differs")
    if type(authority.row_count) is not int or authority.row_count not in (256, 1024):
        raise ValueError("observer authority row count differs")


def canonical_prism_observer_authority_bytes(authority: PrismObserverAuthority) -> bytes:
    """Validate and encode one observer authority as canonical JSON."""

    _validate_observer_authority(authority)
    return canonical_json_bytes(asdict(authority))


def validate_prism_observer_authority_bytes(raw: bytes) -> PrismObserverAuthority:
    """Authenticate canonical bytes and return their typed authority."""

    if type(raw) is not bytes:
        raise ValueError("observer authority bytes must be concrete bytes")
    try:
        value = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("observer authority is not valid UTF-8 JSON") from error
    if type(value) is not dict or frozenset(value) != _AUTHORITY_KEYS:
        raise ValueError("observer authority schema keys differ")
    if raw != canonical_json_bytes(value):
        raise ValueError("observer authority bytes are not canonical")
    authority = PrismObserverAuthority(**value)
    _validate_observer_authority(authority)
    return authority


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


def _encode_protocol_literal(tokenizer: object, literal: str) -> tuple[int, ...]:
    encode = getattr(tokenizer, "encode", None)
    decode = getattr(tokenizer, "decode", None)
    if not callable(encode) or not callable(decode):
        raise ValueError("PRISM tokenizer interface differs")
    encoded = encode(literal, add_special_tokens=False)
    if type(encoded) is not list or any(
        type(token) is not int or not 0 <= token <= 0xFFFF_FFFF for token in encoded
    ):
        raise ValueError("PRISM tokenizer token IDs differ")
    sequence = tuple(encoded)
    if not sequence:
        raise ValueError("PRISM tokenizer emitted an empty sequence")
    special_ids = getattr(tokenizer, "all_special_ids", ())
    if type(special_ids) not in (tuple, list) or any(
        type(token) is not int for token in special_ids
    ):
        raise ValueError("PRISM tokenizer special-token authority differs")
    if set(sequence).intersection(special_ids):
        raise ValueError("PRISM tokenizer inserted a special token")
    decoded = decode(
        list(sequence),
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    if type(decoded) is not str or decoded != literal:
        raise ValueError("PRISM tokenizer decode round trip differs")
    if tuple(encode(decoded, add_special_tokens=False)) != sequence:
        raise ValueError("PRISM tokenizer encode round trip differs")
    return sequence


def derive_prism_token_protocol(
    processor: object,
    bundle: PrismPromptBundle,
) -> PrismTokenProtocol:
    """Derive the exact ID-only PRISM grammar from one sealed tokenizer."""

    _validate_prompt_bundle(bundle)
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise ValueError("PRISM processor tokenizer differs")
    sequences = tuple(
        _encode_protocol_literal(tokenizer, literal) for literal in _protocol_literals()
    )
    for left_index, left in enumerate(sequences):
        for right in sequences[left_index + 1 :]:
            shared = min(len(left), len(right))
            if left[:shared] == right[:shared]:
                raise ValueError("PRISM tokenizer protocol is not prefix-free")
    return PrismTokenProtocol(
        channel_prefixes=sequences[:8],
        visibility_prefixes=sequences[8:12],
        relation_prefixes=sequences[12:15],
        confidence_prefixes=sequences[15:18],
        evidence_separator=sequences[18],
        terminal_tokens=sequences[19],
        max_evidence_tokens=64,
    )


def _validate_completion_bundle(bundle: PrismCompletionBundle) -> None:
    if bundle.schema != "sfora-prism-completion-bundle-v1":
        raise ValueError("PRISM completion bundle schema differs")
    if bundle.phase not in ("calibration", "diagnostic"):
        raise ValueError("PRISM completion bundle phase differs")
    for name in ("observer_authority_sha256", "token_protocol_sha256"):
        value = getattr(bundle, name)
        if type(value) is not str or _SHA256.fullmatch(value) is None:
            raise ValueError(f"PRISM completion bundle {name} differs")
    if type(bundle.rows) is not tuple or not bundle.rows:
        raise ValueError("PRISM completion bundle rows differ")
    identities: set[tuple[str, str]] = set()
    for row in bundle.rows:
        if (
            type(row) is not PrismCompletionRow
            or type(row.pair_handle) is not str
            or _SHA256.fullmatch(row.pair_handle) is None
            or row.channel not in PRISM_CHANNELS
            or type(row.completion_ids) is not tuple
            or not row.completion_ids
            or any(
                type(token) is not int or not 0 <= token <= 0xFFFF_FFFF
                for token in row.completion_ids
            )
        ):
            raise ValueError("PRISM completion row authority differs")
        identity = (row.pair_handle, row.channel)
        if identity in identities:
            raise ValueError("PRISM completion bundle contains a duplicate row")
        identities.add(identity)


def canonical_prism_completion_bundle_bytes(bundle: PrismCompletionBundle) -> bytes:
    """Validate and encode an ID-only completion bundle."""

    _validate_completion_bundle(bundle)
    return canonical_json_bytes(asdict(bundle))


def validate_prism_completion_bundle_bytes(raw: bytes) -> PrismCompletionBundle:
    """Authenticate canonical completion bytes without decoding any token IDs."""

    if type(raw) is not bytes:
        raise ValueError("PRISM completion bytes must be concrete bytes")
    try:
        value = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("PRISM completion bundle is not valid UTF-8 JSON") from error
    if type(value) is not dict or frozenset(value) != _COMPLETION_BUNDLE_KEYS:
        raise ValueError("PRISM completion bundle schema keys differ")
    if raw != canonical_json_bytes(value):
        raise ValueError("PRISM completion bundle bytes are not canonical")
    row_values = value["rows"]
    if type(row_values) is not list:
        raise ValueError("PRISM completion rows schema differs")
    rows: list[PrismCompletionRow] = []
    for row_value in row_values:
        if type(row_value) is not dict or frozenset(row_value) != _COMPLETION_ROW_KEYS:
            raise ValueError("PRISM completion row schema differs")
        completion_ids = row_value["completion_ids"]
        if type(completion_ids) is not list:
            raise ValueError("PRISM completion token schema differs")
        rows.append(
            PrismCompletionRow(
                pair_handle=row_value["pair_handle"],
                channel=row_value["channel"],
                completion_ids=tuple(completion_ids),
            )
        )
    bundle = PrismCompletionBundle(
        schema=value["schema"],
        phase=value["phase"],
        observer_authority_sha256=value["observer_authority_sha256"],
        token_protocol_sha256=value["token_protocol_sha256"],
        rows=tuple(rows),
    )
    _validate_completion_bundle(bundle)
    return bundle


def bind_prism_diagnostic_capability(
    schedules: tuple[PrismObservationRow, ...],
    scoring_rows: tuple[PrismScoringRow, ...],
    calibration_capability: tuple[PrismObservationCapabilityRow, ...],
    completion_bundle: PrismCompletionBundle,
    protocol: PrismTokenProtocol,
    *,
    source_identity: str,
) -> PrismDiagnosticRelease:
    """Authenticate calibration completions and release the diagnostic phase."""

    expected_capability = release_prism_observation_capability(
        schedules,
        scoring_rows,
        source_identity=source_identity,
        phase="calibration",
    )
    if calibration_capability != expected_capability:
        raise ValueError("PRISM calibration capability binding differs")
    _validate_completion_bundle(completion_bundle)
    if completion_bundle.phase != "calibration":
        raise ValueError("PRISM calibration completion phase differs")
    expected_schedules = tuple(row for row in schedules if row.fold < 4)
    if len(completion_bundle.rows) != len(expected_schedules):
        raise ValueError("PRISM calibration completion cardinality differs")
    parsed = []
    completion_ids = []
    for private, public, completion in zip(
        expected_schedules,
        expected_capability,
        completion_bundle.rows,
        strict=True,
    ):
        if (
            completion.pair_handle != public.pair_handle
            or completion.channel != public.channel
        ):
            raise ValueError("PRISM calibration completion binding differs")
        completion_ids.append(completion.completion_ids)
        try:
            parsed.append(parse_prism_completion(private, completion.completion_ids, protocol))
        except ValueError:
            parsed.append(invalid_prism_observation(private, completion.completion_ids))
    diagnostic_placeholders = tuple(
        invalid_prism_observation(row, ()) for row in schedules if row.fold == 4
    )
    all_observations = (*parsed, *diagnostic_placeholders)
    calibrations = calibrate_prism_channels(
        all_observations,
        scoring_rows,
        source_identity=source_identity,
    )
    receipt = prism_calibration_receipt_sha256(calibrations, protocol)
    pilot_count = 32 * len(PRISM_CHANNELS)
    diagnostic_capability = release_prism_observation_capability(
        schedules,
        scoring_rows,
        source_identity=source_identity,
        phase="diagnostic",
        calibration_receipt_sha256=receipt,
        calibrations=calibrations,
        pilot_observations=tuple(parsed[:pilot_count]),
        pilot_completion_ids=tuple(completion_ids[:pilot_count]),
        protocol=protocol,
    )
    return PrismDiagnosticRelease(
        calibrations=calibrations,
        calibration_receipt_sha256=receipt,
        capability=diagnostic_capability,
    )
