#!/usr/bin/env python3
"""Publish a burned-only Cars pixel authority for transfer diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path

from scripts.run_siglip_proxy_control import (
    ControlExampleBands,
    control_manifest_artifact_bytes,
    load_control_examples,
)
from sfora.substrate_screen import SUBSTRATE_F0_CLASSES


def _canonical_bytes(value: dict[str, object]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n"
    ).encode()


def _write_new(path: Path, payload: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _absolute_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("path must be absolute")
    return path


def _lower_sha256(value: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise argparse.ArgumentTypeError("digest must be lowercase SHA-256")
    return value


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the explicit model-free preparation interface."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-manifest", required=True, type=_absolute_path)
    parser.add_argument("--control-manifest-sha256", required=True, type=_lower_sha256)
    parser.add_argument("--output", required=True, type=_absolute_path)
    parser.add_argument("--execute-burned-preparation", required=True, action="store_true")
    return parser.parse_args(argv)


def prepare_burned_artifact(
    *,
    bands: ControlExampleBands,
    control_manifest_raw: bytes,
    output: Path,
) -> bytes:
    """Create one deterministic burned-only flat image namespace."""

    if type(bands) is not ControlExampleBands or type(control_manifest_raw) is not bytes:
        raise ValueError("burned preparation authority differs")
    if not isinstance(output, Path):
        raise TypeError("burned output path differs")
    expected_manifest = control_manifest_artifact_bytes(bands)
    if control_manifest_raw != expected_manifest:
        raise ValueError("control manifest bytes differ")
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    if (
        not bands.burned_diagnostic
        or any(example.label not in SUBSTRATE_F0_CLASSES for example in bands.burned_diagnostic)
        or tuple(sorted(bands.burned_diagnostic, key=lambda item: item.example_id))
        != bands.burned_diagnostic
    ):
        raise ValueError("burned example band differs")

    source_ordinals = {
        example.example_id: ordinal for ordinal, example in enumerate(bands.ordered_manifest)
    }
    output.mkdir()
    images = output / "images"
    images.mkdir()
    try:
        rows: list[dict[str, object]] = []
        written: dict[str, bytes] = {}
        for example in bands.burned_diagnostic:
            save = getattr(example.image, "save", None)
            if not callable(save):
                raise TypeError("burned image is not encodable")
            stream = io.BytesIO()
            save(stream, format="PNG", optimize=False, compress_level=9)
            payload = stream.getvalue()
            digest = hashlib.sha256(payload).hexdigest()
            basename = f"{digest}.png"
            prior = written.get(basename)
            if prior is None:
                _write_new(images / basename, payload)
                written[basename] = payload
            elif prior != payload:
                raise RuntimeError("burned image digest collision")
            rows.append(
                {
                    "basename": basename,
                    "byte_length": len(payload),
                    "example_id": example.example_id,
                    "image_sha256": digest,
                    "label": example.label,
                    "source_ordinal": source_ordinals[example.example_id],
                }
            )
        raw = _canonical_bytes(
            {
                "claim_eligible": False,
                "examples": rows,
                "schema": "sfora-weight-space-transfer-burned-input-v1",
                "source_manifest_sha256": hashlib.sha256(control_manifest_raw).hexdigest(),
            }
        )
        _write_new(output / "burned.json", raw)
        return raw
    except BaseException:
        shutil.rmtree(output)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    """Authenticate the control manifest and publish only burned pixels."""

    arguments = parse_arguments(argv)
    if (
        not arguments.control_manifest.is_file()
        or arguments.control_manifest.is_symlink()
    ):
        raise ValueError("control manifest path differs")
    raw = arguments.control_manifest.read_bytes()
    if hashlib.sha256(raw).hexdigest() != arguments.control_manifest_sha256:
        raise ValueError("control manifest digest differs")
    result = prepare_burned_artifact(
        bands=load_control_examples(),
        control_manifest_raw=raw,
        output=arguments.output,
    )
    sys.stdout.buffer.write(
        _canonical_bytes(
            {
                "artifact": str(arguments.output / "burned.json"),
                "artifact_sha256": hashlib.sha256(result).hexdigest(),
                "claim_eligible": False,
                "schema": "sfora-weight-space-transfer-preparation-receipt-v1",
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

