#!/usr/bin/env python3
"""Run one offline, ID-only PRISM observer phase over anonymous payloads."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image

from sfora.pass209_m4 import canonical_json_bytes
from sfora.prism_measurement import PRISM_CHANNELS, PrismObservationCapabilityRow
from sfora.prism_observer import (
    PrismCompletionBundle,
    PrismCompletionRow,
    PrismPayloadAuthority,
    PrismPromptBundle,
    canonical_prism_completion_bundle_bytes,
    canonical_prism_prompt_bundle_bytes,
    derive_prism_token_protocol,
    validate_prism_prompt_bundle_bytes,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the local-only observer capability surface."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("calibration", "diagnostic"), required=True)
    parser.add_argument("--capability", type=Path, required=True)
    parser.add_argument("--prompt-bundle", type=Path, required=True)
    parser.add_argument("--payload-manifest", type=Path, required=True)
    parser.add_argument("--payload-dir", type=Path, required=True)
    parser.add_argument("--observer-authority-sha256", required=True)
    parser.add_argument("--token-protocol-sha256", required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--snapshot-manifest", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--patch-tokens-per-image", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--progress-output", type=Path, required=True)
    parser.add_argument("--execute-observer", action="store_true", required=True)
    return parser.parse_args(argv)


def _write_progress(
    path: Path,
    *,
    completed_rows: int,
    last_pair_handle: str,
    last_channel: str,
) -> None:
    payload = canonical_json_bytes(
        {
            "completed_rows": completed_rows,
            "last_channel": last_channel,
            "last_pair_handle": last_pair_handle,
            "schema": "sfora-prism-observer-progress-v1",
        }
    )
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _authenticate_payloads(
    authorities: tuple[PrismPayloadAuthority, ...],
    payload_dir: Path,
) -> dict[str, bytes]:
    if type(authorities) is not tuple or not authorities:
        raise ValueError("PRISM payload authority differs")
    expected_names: set[str] = set()
    payloads: dict[str, bytes] = {}
    for authority in authorities:
        if (
            type(authority) is not PrismPayloadAuthority
            or type(authority.payload_sha256) is not str
            or len(authority.payload_sha256) != 64
            or type(authority.byte_length) is not int
            or authority.byte_length <= 0
            or type(authority.width) is not int
            or authority.width <= 0
            or type(authority.height) is not int
            or authority.height <= 0
            or authority.mode != "RGB"
        ):
            raise ValueError("PRISM payload authority differs")
        name = f"{authority.payload_sha256}.png"
        if name in expected_names:
            raise ValueError("PRISM payload authority contains a duplicate")
        expected_names.add(name)
        path = payload_dir / name
        if path.is_symlink() or not path.is_file():
            raise ValueError("PRISM payload path authority differs")
        raw = path.read_bytes()
        if (
            len(raw) != authority.byte_length
            or hashlib.sha256(raw).hexdigest() != authority.payload_sha256
        ):
            raise ValueError("PRISM payload digest differs")
        try:
            with Image.open(io.BytesIO(raw)) as opened:
                opened.load()
                if (
                    opened.mode != authority.mode
                    or opened.width != authority.width
                    or opened.height != authority.height
                ):
                    raise ValueError("PRISM payload image authority differs")
        except OSError as error:
            raise ValueError("PRISM payload image decode differs") from error
        payloads[authority.payload_sha256] = raw
    actual_names = {path.name for path in payload_dir.iterdir()}
    if actual_names != expected_names:
        raise ValueError("PRISM payload directory contains an unregistered file")
    return payloads


def _validate_capability(
    capability: tuple[PrismObservationCapabilityRow, ...], phase: str
) -> None:
    expected_count = 1024 if phase == "calibration" else 256
    if (
        type(capability) is not tuple
        or len(capability) != expected_count
        or any(type(row) is not PrismObservationCapabilityRow for row in capability)
    ):
        raise ValueError("PRISM observer capability cardinality differs")
    identities: set[tuple[str, str]] = set()
    for row in capability:
        if (
            len(row.pair_handle) != 64
            or row.channel not in PRISM_CHANNELS
            or len(row.left_payload_sha256) != 64
            or len(row.right_payload_sha256) != 64
            or type(row.left_first) is not bool
            or type(row.generation_seed) is not int
            or row.generation_seed < 0
        ):
            raise ValueError("PRISM observer capability authority differs")
        identity = (row.pair_handle, row.channel)
        if identity in identities:
            raise ValueError("PRISM observer capability contains a duplicate")
        identities.add(identity)
    for offset in range(0, len(capability), len(PRISM_CHANNELS)):
        group = capability[offset : offset + len(PRISM_CHANNELS)]
        if (
            tuple(row.channel for row in group) != PRISM_CHANNELS
            or len({row.pair_handle for row in group}) != 1
        ):
            raise ValueError("PRISM observer capability order differs")


def _decode_rgb(raw: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(raw)) as opened:
        return np.asarray(opened, dtype=np.uint8).copy(order="C")


def observe_prism_cue_panel(
    output_path: Path,
    progress_path: Path,
    *,
    phase: str,
    capability: tuple[PrismObservationCapabilityRow, ...],
    prompt_bundle: PrismPromptBundle,
    payload_authorities: tuple[PrismPayloadAuthority, ...],
    payload_dir: Path,
    observer_authority_sha256: str,
    token_protocol_sha256: str,
    patch_tokens_per_image: int,
    load_adapter: Callable[[], tuple[object, object]],
) -> None:
    """Authenticate, observe each row once, and publish one completion bundle."""

    output = Path(output_path)
    progress = Path(progress_path)
    if output.exists():
        raise FileExistsError(output)
    if progress.exists():
        raise FileExistsError(progress)
    if phase not in ("calibration", "diagnostic"):
        raise ValueError("PRISM observer phase differs")
    if (
        type(observer_authority_sha256) is not str
        or len(observer_authority_sha256) != 64
        or type(token_protocol_sha256) is not str
        or len(token_protocol_sha256) != 64
        or type(patch_tokens_per_image) is not int
        or patch_tokens_per_image <= 0
    ):
        raise ValueError("PRISM observer authority differs")
    _validate_capability(capability, phase)
    canonical_prism_prompt_bundle_bytes(prompt_bundle)
    payloads = _authenticate_payloads(payload_authorities, Path(payload_dir))
    referenced = {
        digest
        for row in capability
        for digest in (row.left_payload_sha256, row.right_payload_sha256)
    }
    if referenced != set(payloads):
        raise ValueError("PRISM observer payload binding differs")

    adapter, processor = load_adapter()
    protocol = derive_prism_token_protocol(processor, prompt_bundle)
    if hashlib.sha256(canonical_json_bytes(asdict(protocol))).hexdigest() != token_protocol_sha256:
        raise ValueError("PRISM observer token protocol differs")
    prompts = {row.channel: row.prompt_utf8 for row in prompt_bundle.rows}
    completions: list[PrismCompletionRow] = []
    for completed, row in enumerate(capability, start=1):
        left = _decode_rgb(payloads[row.left_payload_sha256])
        right = _decode_rgb(payloads[row.right_payload_sha256])
        images = (left, right) if row.left_first else (right, left)
        pair = adapter.prepare_image_pair(
            images,
            prompts[row.channel],
            (0, 1),
            patch_tokens_per_image,
        )
        completion_ids = adapter.generate(
            pair,
            row.generation_seed,
            temperature=1.0,
            top_p=1.0,
            max_new_tokens=192,
        )
        if type(completion_ids) is not tuple or not completion_ids or any(
            type(token) is not int or not 0 <= token <= 0xFFFF_FFFF
            for token in completion_ids
        ):
            raise ValueError("PRISM observer completion IDs differ")
        completions.append(
            PrismCompletionRow(row.pair_handle, row.channel, completion_ids)
        )
        _write_progress(
            progress,
            completed_rows=completed,
            last_pair_handle=row.pair_handle,
            last_channel=row.channel,
        )
    raw = canonical_prism_completion_bundle_bytes(
        PrismCompletionBundle(
            schema="sfora-prism-completion-bundle-v1",
            phase=phase,
            observer_authority_sha256=observer_authority_sha256,
            token_protocol_sha256=token_protocol_sha256,
            rows=tuple(completions),
        )
    )
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(raw)


def _load_canonical_mapping(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("PRISM observer input path differs")
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("PRISM observer input is not valid JSON") from error
    if type(value) is not dict or raw != canonical_json_bytes(value):
        raise ValueError("PRISM observer input is not canonical")
    return value


def _load_prompt_bundle(path: Path) -> PrismPromptBundle:
    if path.is_symlink() or not path.is_file():
        raise ValueError("PRISM prompt bundle path differs")
    return validate_prism_prompt_bundle_bytes(path.read_bytes())


def _load_capability(
    path: Path, phase: str
) -> tuple[PrismObservationCapabilityRow, ...]:
    value = _load_canonical_mapping(path)
    if frozenset(value) != frozenset(("schema", "rows")) or value["schema"] != (
        f"sfora-prism-{phase}-capability-v1"
    ):
        raise ValueError("PRISM capability schema differs")
    rows_value = value["rows"]
    if type(rows_value) is not list:
        raise ValueError("PRISM capability rows differ")
    keys = frozenset(
        (
            "pair_handle",
            "channel",
            "left_payload_sha256",
            "right_payload_sha256",
            "left_first",
            "generation_seed",
        )
    )
    rows = []
    for row in rows_value:
        if type(row) is not dict or frozenset(row) != keys:
            raise ValueError("PRISM capability row schema differs")
        rows.append(PrismObservationCapabilityRow(**row))
    result = tuple(rows)
    _validate_capability(result, phase)
    return result


def _load_payload_manifest(path: Path) -> tuple[PrismPayloadAuthority, ...]:
    value = _load_canonical_mapping(path)
    if (
        frozenset(value) != frozenset(("schema", "payloads"))
        or value["schema"] != "sfora-prism-payload-manifest-v1"
        or type(value["payloads"]) is not list
    ):
        raise ValueError("PRISM payload manifest schema differs")
    keys = frozenset(("payload_sha256", "byte_length", "width", "height", "mode"))
    rows = []
    for row in value["payloads"]:
        if type(row) is not dict or frozenset(row) != keys:
            raise ValueError("PRISM payload manifest row differs")
        rows.append(PrismPayloadAuthority(**row))
    return tuple(rows)


def _load_real_adapter(
    model_root: Path,
    snapshot_manifest: Path,
    fixture_path: Path,
) -> tuple[object, object, int]:
    from scripts.diagnose_saga_gb10_feasibility import (
        LoadedAuthority,
        TransformersFactory,
        load_qwen_adapter,
    )
    from sfora.saga_feasibility import load_fixture_authority, load_snapshot_authority

    snapshot = load_snapshot_authority(root=model_root, manifest_path=snapshot_manifest)
    fixture = load_fixture_authority(fixture_path)
    adapter = load_qwen_adapter(
        LoadedAuthority(snapshot=snapshot, fixture=fixture),
        factory=TransformersFactory(),
    )
    return adapter, adapter.processor, fixture.patch_tokens_per_image


def main(argv: list[str] | None = None) -> int:
    """Authenticate local inputs, load Qwen once, and observe one phase."""

    args = parse_args(argv)
    prompts = _load_prompt_bundle(args.prompt_bundle)
    capability = _load_capability(args.capability, args.phase)
    payload_authorities = _load_payload_manifest(args.payload_manifest)

    def load() -> tuple[object, object]:
        adapter, processor, patch_tokens = _load_real_adapter(
            args.model_root,
            args.snapshot_manifest,
            args.fixture,
        )
        if patch_tokens != args.patch_tokens_per_image:
            raise ValueError("PRISM observer patch-token authority differs")
        return adapter, processor

    observe_prism_cue_panel(
        args.output,
        args.progress_output,
        phase=args.phase,
        capability=capability,
        prompt_bundle=prompts,
        payload_authorities=payload_authorities,
        payload_dir=args.payload_dir,
        observer_authority_sha256=args.observer_authority_sha256,
        token_protocol_sha256=args.token_protocol_sha256,
        patch_tokens_per_image=args.patch_tokens_per_image,
        load_adapter=load,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
