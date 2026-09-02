#!/usr/bin/env python3
"""Prepare truth-separated, anonymous PRISM observer inputs."""

from __future__ import annotations

import hashlib
import io
import os
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path

from PIL import Image

from sfora.data import ImageExample, materialize_image
from sfora.pass209_m4 import canonical_json_bytes
from sfora.prism_measurement import (
    PrismExample,
    build_prism_schedules,
    release_prism_observation_capability,
)
from sfora.prism_observer import (
    PrismPayloadAuthority,
    PrismPromptBundle,
    canonical_prism_prompt_bundle_bytes,
)


def _png_bytes(example: ImageExample) -> bytes:
    if isinstance(example.image, Path) and example.image.is_symlink():
        raise ValueError("PRISM source image must not be a symlink")
    image = materialize_image(example.image)
    if not isinstance(image, Image.Image) or image.mode != "RGB":
        raise ValueError("PRISM source image must materialize as RGB")
    if image.width <= 0 or image.height <= 0:
        raise ValueError("PRISM source image dimensions differ")
    stream = io.BytesIO()
    image.save(stream, format="PNG", compress_level=9, optimize=False)
    return stream.getvalue()


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_bytes(canonical_json_bytes(value))


def prepare_prism_observer_inputs(
    output_dir: Path,
    examples: tuple[ImageExample, ...],
    *,
    source_identity: str,
    prompt_bundle: PrismPromptBundle,
) -> None:
    """Build one create-new anonymous PRISM preparation directory."""

    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(output)
    if type(examples) is not tuple or any(type(row) is not ImageExample for row in examples):
        raise TypeError("PRISM preparation examples differ")
    if type(source_identity) is not str or not source_identity:
        raise ValueError("PRISM preparation source identity differs")
    allowed_labels = frozenset((*range(49), 82, 83))
    if any(type(row.label) is not int or row.label not in allowed_labels for row in examples):
        raise ValueError("PRISM preparation label is outside the registered train-only set")
    if len({row.example_id for row in examples}) != len(examples):
        raise ValueError("PRISM preparation example identity differs")
    prompt_bytes = canonical_prism_prompt_bundle_bytes(prompt_bundle)

    encoded: dict[str, bytes] = {}
    authorities: dict[str, PrismPayloadAuthority] = {}
    prism_examples: list[PrismExample] = []
    seen_payloads: dict[str, str] = {}
    for row in examples:
        payload = _png_bytes(row)
        digest = hashlib.sha256(payload).hexdigest()
        prior = seen_payloads.get(digest)
        if prior is not None and prior != row.example_id:
            raise ValueError("PRISM preparation contains duplicate image bytes")
        seen_payloads[digest] = row.example_id
        encoded[digest] = payload
        image = materialize_image(row.image)
        authorities[digest] = PrismPayloadAuthority(
            payload_sha256=digest,
            byte_length=len(payload),
            width=image.width,
            height=image.height,
            mode=image.mode,
        )
        prism_examples.append(
            PrismExample(
                example_id=row.example_id,
                label=row.label,
                image_sha256=digest,
            )
        )

    optimization = tuple(row for row in prism_examples if 0 <= row.label <= 48)
    diagnostic = tuple(row for row in prism_examples if row.label in (82, 83))
    observations, scoring = build_prism_schedules(
        optimization,
        diagnostic,
        source_identity=source_identity,
    )
    capability = release_prism_observation_capability(
        observations,
        scoring,
        source_identity=source_identity,
        phase="calibration",
    )
    calibration_digests = sorted(
        {
            digest
            for row in observations
            if row.fold < 4
            for digest in (row.left_payload_sha256, row.right_payload_sha256)
        }
    )
    diagnostic_digests = sorted(
        {
            digest
            for row in observations
            if row.fold == 4
            for digest in (row.left_payload_sha256, row.right_payload_sha256)
        }
    )
    if len(calibration_digests) != 256 or len(diagnostic_digests) != 64:
        raise ValueError("PRISM preparation payload cardinality differs")

    output.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.partial-", dir=str(output.parent))
    )
    try:
        private = partial / "private"
        calibration = partial / "calibration"
        diagnostic_sealed = partial / "diagnostic-sealed"
        calibration_payloads = calibration / "payloads"
        diagnostic_payloads = diagnostic_sealed / "payloads"
        for directory in (
            private,
            calibration_payloads,
            diagnostic_payloads,
        ):
            directory.mkdir(parents=True)
        (partial / "prompt-bundle.json").write_bytes(prompt_bytes)
        _write_json(
            partial / "token-protocol-request.json",
            {
                "prompt_bundle_sha256": hashlib.sha256(prompt_bytes).hexdigest(),
                "schema": "sfora-prism-token-protocol-request-v1",
            },
        )
        _write_json(
            private / "observations.json",
            {
                "rows": [asdict(row) for row in observations],
                "schema": "sfora-prism-private-observations-v1",
            },
        )
        _write_json(
            private / "scoring.json",
            {
                "rows": [asdict(row) for row in scoring],
                "schema": "sfora-prism-private-scoring-v1",
            },
        )
        _write_json(
            calibration / "capability.json",
            {
                "rows": [asdict(row) for row in capability],
                "schema": "sfora-prism-calibration-capability-v1",
            },
        )
        for directory, digests in (
            (calibration, calibration_digests),
            (diagnostic_sealed, diagnostic_digests),
        ):
            _write_json(
                directory / "payload-manifest.json",
                {
                    "payloads": [asdict(authorities[digest]) for digest in digests],
                    "schema": "sfora-prism-payload-manifest-v1",
                },
            )
            payload_dir = directory / "payloads"
            for digest in digests:
                (payload_dir / f"{digest}.png").write_bytes(encoded[digest])
        if output.exists():
            raise FileExistsError(output)
        os.rename(partial, output)
    except BaseException:
        if partial.exists():
            shutil.rmtree(partial)
        raise
