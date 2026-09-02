"""Anonymous prompt and capability authority for the PRISM observer."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass

from sfora.pass209_m4 import canonical_json_bytes
from sfora.prism_measurement import PRISM_CHANNELS

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
