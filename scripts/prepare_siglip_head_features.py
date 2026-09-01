#!/usr/bin/env python3
"""Prepare an authenticated Cars-train SigLIP pooler feature cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from sfora.data import ImageExample, materialize_image
from sfora.siglip_proxy_control import SiglipProxyControlConfig

_SCRIPT_DIRECTORY = Path(__file__).resolve().parent
if str(_SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIRECTORY))

from prepare_asgcv_p32_inputs import _authenticated_source_commit  # noqa: E402
from run_siglip_proxy_control import (  # noqa: E402
    ControlExampleBands,
    control_manifest_artifact_bytes,
    load_control_examples,
    load_siglip_control_components,
    preprocess_control_evaluation,
)


def _lower_sha256(value: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise argparse.ArgumentTypeError("digest must be 64 lowercase hexadecimal characters")
    return value


def _absolute_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts or os.path.normpath(value) != value:
        raise argparse.ArgumentTypeError("paths must be normalized absolute paths")
    return path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the local-only extraction boundary."""

    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--control-manifest", required=True, type=_absolute_path)
    parser.add_argument("--control-manifest-sha256", required=True, type=_lower_sha256)
    parser.add_argument("--output", required=True, type=_absolute_path)
    parser.add_argument("--execute-feature-cache", action="store_true", required=True)
    effective = list(sys.argv[1:] if argv is None else argv)
    flags = [value.split("=", 1)[0] for value in effective if value.startswith("--")]
    duplicates = sorted({flag for flag in flags if flags.count(flag) > 1})
    if duplicates:
        parser.error(f"duplicate arguments are forbidden: {duplicates!r}")
    return parser.parse_args(effective)


def _canonical(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _write_npy(path: Path, matrix: np.ndarray) -> str:
    with path.open("xb") as stream:
        np.save(stream, matrix, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_feature_cache(
    *,
    bands: ControlExampleBands,
    control_manifest_raw: bytes,
    output: Path,
    embed_band: Callable[[tuple[ImageExample, ...]], torch.Tensor],
    source_commit: str,
) -> bytes:
    """Materialize all three roles while fitting or selecting nothing."""

    if (
        type(bands) is not ControlExampleBands
        or type(control_manifest_raw) is not bytes
        or not control_manifest_raw
        or not isinstance(output, Path)
        or not callable(embed_band)
        or type(source_commit) is not str
        or len(source_commit) != 40
        or any(character not in "0123456789abcdef" for character in source_commit)
        or control_manifest_raw != control_manifest_artifact_bytes(bands)
    ):
        raise ValueError("control manifest authority differs")
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    try:
        control_path = output / "control-manifest.json"
        with control_path.open("xb") as stream:
            stream.write(control_manifest_raw)
            stream.flush()
            os.fsync(stream.fileno())
        payload: dict[str, object] = {}
        for key, role, examples in (
            ("optimization", "optimization-train", bands.optimization),
            ("clean_validation", "clean-validation", bands.clean_validation),
            ("burned_diagnostic", "burned-diagnostic", bands.burned_diagnostic),
        ):
            features = embed_band(examples)
            if (
                type(features) is not torch.Tensor
                or features.ndim != 2
                or features.shape[0] != len(examples)
                or features.device.type != "cpu"
                or features.dtype != torch.float32
                or not features.is_contiguous()
                or not bool(torch.isfinite(features).all())
                or bool((torch.linalg.vector_norm(features.double(), dim=1) <= 0).any())
            ):
                raise ValueError("SigLIP feature extraction authority differs")
            filename = f"{role}.npy"
            matrix = features.numpy().astype("<f4", copy=False)
            digest = _write_npy(output / filename, matrix)
            payload[key] = {
                "role": role,
                "file": filename,
                "sha256": digest,
                "shape": [int(features.shape[0]), int(features.shape[1])],
                "example_ids": [example.example_id for example in examples],
                "labels": [example.label for example in examples],
            }
        config = SiglipProxyControlConfig()
        raw = _canonical(
            {
                "schema": "sfora-siglip-head-feature-cache-v1",
                "claim_eligible": False,
                "source_manifest_sha256": hashlib.sha256(control_manifest_raw).hexdigest(),
                "control_manifest_file": control_path.name,
                "source_commit": source_commit,
                "model_name": config.model_name,
                "model_revision": config.model_revision,
                "bands": payload,
            }
        )
        manifest = output / "features.json"
        with manifest.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        return raw
    except BaseException:
        shutil.rmtree(output)
        raise


def _default_embedder(device: torch.device) -> Callable[[tuple[ImageExample, ...]], torch.Tensor]:
    config = SiglipProxyControlConfig()
    tower, processor = load_siglip_control_components(config=config)
    tower = tower.to(device).eval()

    def embed(examples: tuple[ImageExample, ...]) -> torch.Tensor:
        batches: list[torch.Tensor] = []
        with torch.inference_mode():
            for start in range(0, len(examples), 64):
                rows = examples[start : start + 64]
                images = [materialize_image(example.image) for example in rows]
                pixels = preprocess_control_evaluation(processor, images).to(device)
                pooled = tower(pixels).float()
                if pooled.ndim != 2 or not bool(torch.isfinite(pooled).all()):
                    raise ValueError("SigLIP pooler output differs")
                batches.append(pooled.cpu())
        return torch.cat(batches).contiguous()

    return embed


def main(argv: Sequence[str] | None = None) -> int:
    """Authenticate the control manifest and extract local pooler features."""

    arguments = parse_args(argv)
    if not arguments.control_manifest.is_file() or arguments.control_manifest.is_symlink():
        raise ValueError("control manifest path differs")
    control_raw = arguments.control_manifest.read_bytes()
    if hashlib.sha256(control_raw).hexdigest() != arguments.control_manifest_sha256:
        raise ValueError("control manifest digest differs")
    bands = load_control_examples()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw = prepare_feature_cache(
        bands=bands,
        control_manifest_raw=control_raw,
        output=arguments.output,
        embed_band=_default_embedder(device),
        source_commit=_authenticated_source_commit(Path(__file__).resolve().parents[1]),
    )
    sys.stdout.write(
        json.dumps(
            {
                "feature_manifest": str(arguments.output / "features.json"),
                "feature_manifest_sha256": hashlib.sha256(raw).hexdigest(),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
