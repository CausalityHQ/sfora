#!/usr/bin/env python3
"""Run the authenticated native-resolution Caliber twin probe offline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from collections import OrderedDict
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from PIL import Image

from scripts.run_siglip_proxy_control import (
    SiglipPooledTower,
    _canonical_bytes,
    _config_sha256,
    load_control_examples,
    load_siglip_control_components,
    preprocess_control_evaluation,
    read_control_seed_receipt,
    require_control_determinism,
)
from sfora.data import materialize_image
from sfora.native_twin_probe import (
    NativeTwinAuthority,
    canonical_native_twin_result_bytes,
    native_descriptor_sha256,
    score_native_twin_probe,
    validate_canonical_native_twin_result_bytes,
    validate_native_twin_result,
)
from sfora.siglip_proxy_control import PooledProxyAnchorModel, SiglipProxyControlConfig
from sfora.token_set_screen import F1_TRAIN_CLASSES

_CHECKPOINT_KEYS = {
    "claim_eligible",
    "completed_epoch",
    "config_sha256",
    "cpu_rng_state",
    "cuda_rng_states",
    "final_objective",
    "initial_snapshot_sha256",
    "maximum_score_disagreement",
    "model_state",
    "optimizer_state",
    "run_authority_sha256",
    "sampler_cycles",
    "sampler_positions",
    "schema",
    "seed",
}


def _validated_checkpoint_model_state(
    payload: object,
    *,
    config: SiglipProxyControlConfig,
    seed: int,
) -> OrderedDict[str, Any]:
    """Return the exact producer-compatible model state from one checkpoint."""

    if (
        type(payload) is not dict
        or set(payload) != _CHECKPOINT_KEYS
        or payload["schema"] != "sfora-siglip-proxy-checkpoint-payload-v1"
        or payload["claim_eligible"] is not False
        or payload["seed"] != seed
        or payload["completed_epoch"] != 60
        or payload["config_sha256"] != _config_sha256(config)
        or type(payload["model_state"]) is not OrderedDict
    ):
        raise ValueError("native checkpoint payload authority differs")
    return cast("OrderedDict[str, Any]", payload["model_state"])


def native_crop_boxes(width: int, height: int) -> tuple[tuple[int, int, int, int], ...]:
    """Return the exact row-major 3x3 grid of overlapping two-thirds crops."""

    if type(width) is not int or type(height) is not int:
        raise TypeError("native image dimensions must be concrete integers")
    if width < 2 or height < 2:
        raise ValueError("native image dimensions must be positive")
    boxes: list[tuple[int, int, int, int]] = []
    for y_index in range(3):
        top = y_index * height // 6
        bottom = ((y_index + 4) * height + 5) // 6
        for x_index in range(3):
            left = x_index * width // 6
            right = ((x_index + 4) * width + 5) // 6
            box = (left, top, min(width, right), min(height, bottom))
            if box[2] <= box[0] or box[3] <= box[1]:
                raise ValueError("native crop has nonpositive area")
            boxes.append(box)
    if len(set(boxes)) != 9:
        raise ValueError("native image dimensions do not support nine distinct crops")
    return tuple(boxes)


def matched_control_crop(crop: object) -> object:
    """Downsample one crop to a 256px long edge without changing its aspect ratio."""

    size = getattr(crop, "size", None)
    resize = getattr(crop, "resize", None)
    if (
        type(size) is not tuple
        or len(size) != 2
        or any(type(value) is not int or value < 2 for value in size)
        or not callable(resize)
    ):
        raise ValueError("matched control crop dimensions differ")
    if max(size) <= 256:
        copy = getattr(crop, "copy", None)
        if not callable(copy):
            raise ValueError("matched control crop lacks copy support")
        return copy()
    scale = 256.0 / max(size)
    target = tuple(max(2, int(round(value * scale))) for value in size)
    reduced = resize(target, resample=Image.Resampling.BICUBIC)
    return reduced.resize(size, resample=Image.Resampling.BICUBIC)


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the single frozen offline execution surface."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-receipt", type=Path, required=True)
    parser.add_argument("--checkpoint-receipt", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--source-tree-digest", required=True)
    parser.add_argument("--probe-revision", required=True)
    parser.add_argument("--probe-tree-digest", required=True)
    parser.add_argument("--seed", type=int, choices=(17, 29, 43), required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--execute-native-twin-probe", action="store_true", required=True)
    parsed = parser.parse_args(arguments)
    if parsed.batch_size < 1:
        parser.error("batch size must be positive")
    return parsed


def _read_regular(path: Path, role: str) -> bytes:
    if path.is_symlink() or not path.exists() or not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError(f"{role} must be one regular file")
    return path.read_bytes()


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _checkpoint_authority(
    *,
    seed_raw: bytes,
    checkpoint_receipt_raw: bytes,
    checkpoint: Path,
    seed: int,
    source_revision: str,
    source_tree_digest: str,
) -> tuple[dict[str, Any], str]:
    seed_receipt = read_control_seed_receipt(seed_raw)
    try:
        receipt = json.loads(checkpoint_receipt_raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("native checkpoint receipt is not JSON") from error
    if (
        type(receipt) is not dict
        or set(receipt)
        != {"bytes", "checkpoint", "claim_eligible", "epoch", "schema", "seed", "sha256"}
        or checkpoint_receipt_raw != _canonical_bytes(cast(dict[str, Any], receipt))
        or receipt["schema"] != "sfora-siglip-proxy-checkpoint-v1"
        or receipt["claim_eligible"] is not False
        or receipt["seed"] != seed
        or receipt["epoch"] != 60
        or receipt["checkpoint"] != checkpoint.name
    ):
        raise ValueError("native checkpoint receipt authority differs")
    source = seed_receipt["source"]
    final_checkpoint = seed_receipt["checkpoint"]
    if (
        seed_receipt["seed"] != seed
        or source != {
            "dirty": False,
            "revision": source_revision,
            "tree_digest": source_tree_digest,
        }
        or final_checkpoint["basename"] != checkpoint.name
        or final_checkpoint["bytes"] != receipt["bytes"]
        or final_checkpoint["epoch"] != receipt["epoch"]
        or final_checkpoint["sha256"] != receipt["sha256"]
        or final_checkpoint["receipt_basename"] == ""
    ):
        raise ValueError("native checkpoint and seed receipt binding differs")
    observed_sha256, observed_bytes = _sha256_file(checkpoint)
    if observed_sha256 != receipt["sha256"] or observed_bytes != receipt["bytes"]:
        raise ValueError("native checkpoint digest or length differs")
    return seed_receipt, observed_sha256


def _decoded_image(example: object) -> tuple[object, str]:
    image = materialize_image(example.image)
    convert = getattr(image, "convert", None)
    if not callable(convert):
        raise ValueError("native source image lacks RGB conversion")
    rgb = convert("RGB")
    size = getattr(rgb, "size", None)
    tobytes = getattr(rgb, "tobytes", None)
    if (
        type(size) is not tuple
        or len(size) != 2
        or any(type(value) is not int or value < 2 for value in size)
        or not callable(tobytes)
    ):
        raise ValueError("native source image shape differs")
    pixels = tobytes()
    if type(pixels) is not bytes or len(pixels) != size[0] * size[1] * 3:
        raise ValueError("native source image pixel bytes differ")
    digest = hashlib.sha256()
    digest.update(b"sfora-native-rgb-v1\0")
    digest.update(size[0].to_bytes(8, "little"))
    digest.update(size[1].to_bytes(8, "little"))
    digest.update(pixels)
    return rgb, digest.hexdigest()


def _encode_images(
    model: PooledProxyAnchorModel,
    processor: object,
    images: list[object],
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    descriptors: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            pixels = preprocess_control_evaluation(processor, images[start : start + batch_size])
            encoded = model.encode(pixels.to(device, non_blocking=True))
            descriptors.append(encoded.detach().cpu().numpy().astype(np.float32, copy=False))
    result = np.concatenate(descriptors, axis=0)
    if result.shape[0] != len(images) or not np.isfinite(result).all():
        raise ValueError("native descriptor extraction differs")
    return result


def run_probe(args: argparse.Namespace) -> bytes:
    """Authenticate, encode, score, and return one canonical result."""

    seed_raw = _read_regular(args.seed_receipt, "native seed receipt")
    checkpoint_receipt_raw = _read_regular(
        args.checkpoint_receipt, "native checkpoint receipt"
    )
    seed_receipt, checkpoint_sha256 = _checkpoint_authority(
        seed_raw=seed_raw,
        checkpoint_receipt_raw=checkpoint_receipt_raw,
        checkpoint=args.checkpoint,
        seed=args.seed,
        source_revision=args.source_revision,
        source_tree_digest=args.source_tree_digest,
    )
    config = SiglipProxyControlConfig()
    if seed_receipt["config_sha256"] != _config_sha256(config):
        raise ValueError("native control config differs")
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("native twin probe requires CUDA")
    require_control_determinism(device)
    tower, processor = load_siglip_control_components(config=config)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model = PooledProxyAnchorModel(
        tower=cast(SiglipPooledTower, tower),
        input_dimensions=config.input_dimensions,
        embedding_dimensions=config.embedding_dimensions,
        class_count=len(F1_TRAIN_CLASSES),
        projection_initialization=config.projection_initialization,
        proxy_initialization=config.proxy_initialization,
    )
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model_state = _validated_checkpoint_model_state(payload, config=config, seed=args.seed)
    model.load_state_dict(model_state, strict=True)
    del payload
    model = model.to(device).eval()
    bands = load_control_examples()
    examples = tuple(
        example for example in bands.burned_diagnostic if example.label in {82, 83}
    )
    if len(examples) < 40:
        raise ValueError("native Caliber population is too small")
    image_sha256: list[str] = []
    crop_long_edges: list[tuple[int, ...]] = []
    global_chunks: list[np.ndarray] = []
    control_chunks: list[np.ndarray] = []
    native_chunks: list[np.ndarray] = []
    image_batch_size = max(1, args.batch_size // 9)
    for start in range(0, len(examples), image_batch_size):
        batch = examples[start : start + image_batch_size]
        decoded: list[object] = []
        control_images: list[object] = []
        native_images: list[object] = []
        for example in batch:
            image, digest = _decoded_image(example)
            decoded.append(image)
            image_sha256.append(digest)
            crop = getattr(image, "crop", None)
            if not callable(crop):
                raise ValueError("native source image lacks crop support")
            crops = [crop(box) for box in native_crop_boxes(*image.size)]
            crop_long_edges.append(tuple(max(value.size) for value in crops))
            native_images.extend(crops)
            control_images.extend(matched_control_crop(value) for value in crops)
        global_chunks.append(
            _encode_images(
                model, processor, decoded, device=device, batch_size=args.batch_size
            )
        )
        control_chunks.append(
            _encode_images(
                model,
                processor,
                control_images,
                device=device,
                batch_size=args.batch_size,
            )
        )
        native_chunks.append(
            _encode_images(
                model,
                processor,
                native_images,
                device=device,
                batch_size=args.batch_size,
            )
        )
    global_plane = np.concatenate(global_chunks, axis=0)
    control_flat = np.concatenate(control_chunks, axis=0)
    native_flat = np.concatenate(native_chunks, axis=0)
    control = control_flat.reshape(len(examples), 9, global_plane.shape[1])
    native = native_flat.reshape(len(examples), 9, global_plane.shape[1])
    source_digest = hashlib.sha256()
    for raw in (
        seed_raw,
        checkpoint_receipt_raw,
        args.source_revision.encode(),
        args.source_tree_digest.encode(),
        args.probe_revision.encode(),
        args.probe_tree_digest.encode(),
    ):
        source_digest.update(len(raw).to_bytes(8, "little"))
        source_digest.update(raw)
    authority = NativeTwinAuthority(
        source_identity=source_digest.hexdigest(),
        checkpoint_sha256=checkpoint_sha256,
        model_revision=config.model_revision,
        probe_revision=args.probe_revision,
        probe_tree_digest=args.probe_tree_digest,
        example_ids=tuple(example.example_id for example in examples),
        image_sha256=tuple(image_sha256),
        labels=tuple(example.label for example in examples),
        crop_long_edges=tuple(crop_long_edges),
        global_descriptor_sha256=native_descriptor_sha256(global_plane),
        control_descriptor_sha256=native_descriptor_sha256(control),
        native_descriptor_sha256=native_descriptor_sha256(native),
    )
    result = score_native_twin_probe(authority, global_plane, control, native)
    validate_native_twin_result(result, authority, global_plane, control, native)
    raw = canonical_native_twin_result_bytes(result)
    validate_canonical_native_twin_result_bytes(raw, expected=result)
    return raw


def main(arguments: Sequence[str] | None = None) -> None:
    """Execute once and publish without overwriting existing evidence."""

    args = parse_args(arguments)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    raw = run_probe(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_name(f"{args.output.name}.partial")
    if partial.exists() or partial.is_symlink():
        raise FileExistsError(partial)
    try:
        with partial.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(partial, args.output, follow_symlinks=False)
    finally:
        partial.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
