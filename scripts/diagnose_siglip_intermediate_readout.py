#!/usr/bin/env python3
"""Run the local-only optimization SigLIP intermediate-readout screen."""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections.abc import Iterable
from pathlib import Path

import torch
from torch.nn import functional as F

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

_DEPTH_COUNT = 27
_TOWER_WIDTH = 1152
_OUTPUT_DIMENSIONS = 512
_BATCH_SIZE = 8


def _absolute_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("path must be absolute")
    return path


def _lower_sha256(value: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise argparse.ArgumentTypeError("digest must be 64 lowercase hexadecimal characters")
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the capability-minimal intermediate-readout command line."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-binding", required=True, type=_absolute_path)
    parser.add_argument("--control-binding-sha256", required=True, type=_lower_sha256)
    parser.add_argument("--checkpoint-seed17", required=True, type=_absolute_path)
    parser.add_argument("--optimization-manifest", required=True, type=_absolute_path)
    parser.add_argument("--optimization-manifest-sha256", required=True, type=_lower_sha256)
    parser.add_argument("--image-root", required=True, type=_absolute_path)
    parser.add_argument("--result", required=True, type=_absolute_path)
    parser.add_argument("--execute-intermediate-readout", required=True, action="store_true")
    effective = list(sys.argv[1:] if argv is None else argv)
    flags = [value.split("=", 1)[0] for value in effective if value.startswith("--")]
    duplicates = sorted({flag for flag in flags if flags.count(flag) > 1})
    if duplicates:
        parser.error(f"duplicate arguments are forbidden: {duplicates!r}")
    return parser.parse_args(effective)


def stream_intermediate_descriptor_planes(
    model: object,
    pixel_batches: Iterable[torch.Tensor],
    *,
    expected_depth_count: int,
    tower_width: int,
    output_dimensions: int,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    """Consume hidden states batchwise and retain only projected descriptor planes."""

    tower = getattr(model, "tower", None)
    vision_model = getattr(tower, "vision_model", None)
    post_layernorm = getattr(vision_model, "post_layernorm", None)
    projection = getattr(model, "projection", None)
    if (
        not isinstance(vision_model, torch.nn.Module)
        or not isinstance(post_layernorm, torch.nn.Module)
        or not isinstance(projection, torch.nn.Linear)
        or projection.bias is not None
        or projection.in_features != tower_width
        or projection.out_features != output_dimensions
        or type(expected_depth_count) is not int
        or expected_depth_count < 1
        or type(tower_width) is not int
        or tower_width < 2
        or type(output_dimensions) is not int
        or output_dimensions < 2
        or type(device) is not torch.device
        or device.type not in {"cpu", "cuda"}
    ):
        raise ValueError("intermediate model authority differs")
    batches = tuple([] for _ in range(expected_depth_count))
    batch_count = 0
    with torch.inference_mode():
        for pixels in pixel_batches:
            if (
                type(pixels) is not torch.Tensor
                or pixels.ndim != 4
                or not pixels.is_floating_point()
                or not bool(torch.isfinite(pixels).all())
                or pixels.shape[0] < 1
            ):
                raise ValueError("intermediate pixel authority differs")
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                output = vision_model(
                    pixel_values=pixels.to(device),
                    output_hidden_states=True,
                    return_dict=True,
                )
            hidden_states = getattr(output, "hidden_states", None)
            if (
                type(hidden_states) not in {tuple, list}
                or len(hidden_states) != expected_depth_count + 1
            ):
                raise ValueError("intermediate hidden-state authority differs")
            for depth, hidden in enumerate(hidden_states[1:]):
                if (
                    type(hidden) is not torch.Tensor
                    or hidden.ndim != 3
                    or hidden.shape[0] != pixels.shape[0]
                    or hidden.shape[1] < 1
                    or hidden.shape[2] != tower_width
                    or hidden.device != device
                    or not hidden.is_floating_point()
                    or not bool(torch.isfinite(hidden).all())
                ):
                    raise ValueError("intermediate hidden-state authority differs")
                normalized = post_layernorm(hidden).float()
                pooled = normalized.mean(dim=1)
                descriptors = F.normalize(projection(pooled).float(), dim=1)
                if (
                    descriptors.shape != (pixels.shape[0], output_dimensions)
                    or not bool(torch.isfinite(descriptors).all())
                    or bool((torch.linalg.vector_norm(descriptors, dim=1) <= 0.0).any())
                ):
                    raise ValueError("intermediate descriptor stream differs")
                batches[depth].append(descriptors.cpu())
            batch_count += 1
    if batch_count < 1 or any(not values for values in batches):
        raise ValueError("intermediate descriptor stream is empty")
    return tuple(torch.cat(values, dim=0).contiguous() for values in batches)


def main(argv: list[str] | None = None) -> int:
    """Authenticate seed 17, stream optimization images, and publish one result."""

    import json
    from collections.abc import Mapping

    from PIL import Image

    from scripts.diagnose_siglip_rsta_stage_a import (
        _load_model_state_checkpoint,
        _load_optimization_manifest,
        _parse_control_binding,
        _read_regular,
        _stage_a_transforms,
        configure_stage_a_determinism,
        load_stage_a_checkpoint_model,
        load_stage_a_siglip_runtime,
    )
    from sfora.siglip_head_screen import build_feature_split_authority
    from sfora.siglip_intermediate_readout import score_intermediate_readout_depths

    arguments = parse_args(argv)
    if arguments.result.exists() or arguments.result.is_symlink():
        raise FileExistsError(arguments.result)
    binding_raw = _read_regular(arguments.control_binding, role="intermediate control binding")
    if hashlib.sha256(binding_raw).hexdigest() != arguments.control_binding_sha256:
        raise ValueError("intermediate control binding digest differs")
    binding = _parse_control_binding(binding_raw)
    if (
        binding.control_complete is not True
        or binding.optimization_manifest_sha256 != arguments.optimization_manifest_sha256
        or tuple(checkpoint.seed for checkpoint in binding.checkpoints) != (17, 29, 43)
    ):
        raise ValueError("intermediate control authority differs")
    seed17_authority = binding.checkpoints[0]
    checkpoint = _load_model_state_checkpoint(
        arguments.checkpoint_seed17,
        seed17_authority,
        binding,
    )
    example_ids, label_values, image_paths = _load_optimization_manifest(
        arguments.optimization_manifest,
        arguments.optimization_manifest_sha256,
        binding,
        arguments.image_root,
    )
    configure_stage_a_determinism()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("intermediate readout requires CUDA bf16")
    device = torch.device("cuda")
    runtime = load_stage_a_siglip_runtime()
    model = load_stage_a_checkpoint_model(
        checkpoint,
        model_factory=runtime.model_factory,
        device=device,
    )
    runtime.disable_checkpointing(model)
    model.eval()
    _graph_transform, evaluation_transform = _stage_a_transforms(runtime.processor)

    def pixel_batches() -> Iterable[torch.Tensor]:
        for start in range(0, len(image_paths), _BATCH_SIZE):
            tensors = []
            for path in image_paths[start : start + _BATCH_SIZE]:
                with Image.open(path) as image:
                    tensor = evaluation_transform(image)
                if not isinstance(tensor, torch.Tensor):
                    raise ValueError("intermediate image transform differs")
                tensors.append(tensor)
            yield torch.stack(tensors)

    planes = stream_intermediate_descriptor_planes(
        model,
        pixel_batches(),
        expected_depth_count=_DEPTH_COUNT,
        tower_width=_TOWER_WIDTH,
        output_dimensions=_OUTPUT_DIMENSIONS,
        device=device,
    )
    labels = torch.tensor(label_values, dtype=torch.int64)
    split_authority = build_feature_split_authority(
        source_manifest_sha256=arguments.optimization_manifest_sha256,
        role="optimization-train",
        official_test_access=False,
        ordered_example_ids=example_ids,
        features=planes[-1],
    )
    raw = score_intermediate_readout_depths(
        planes[-1],
        labels,
        planes,
        split_authority=split_authority,
        checkpoint_sha256=seed17_authority.sha256,
        feature_manifest_sha256=arguments.optimization_manifest_sha256,
        expected_depth_count=_DEPTH_COUNT,
        output_dimensions=_OUTPUT_DIMENSIONS,
        fold_count=4,
    )
    value = json.loads(raw)
    if (
        type(value) is not dict
        or value.get("checkpoint_sha256") != seed17_authority.sha256
        or value.get("feature_manifest_sha256") != arguments.optimization_manifest_sha256
        or value.get("claim_eligible") is not False
        or value.get("official_test_access") is not False
    ):
        raise ValueError("intermediate result binding differs")
    partial = arguments.result.with_name(f"{arguments.result.name}.partial")
    if partial.exists() or partial.is_symlink():
        raise FileExistsError(partial)
    arguments.result.parent.mkdir(parents=True, exist_ok=True)
    partial.write_bytes(raw)
    partial.replace(arguments.result)
    if not isinstance(value, Mapping):
        raise AssertionError("unreachable intermediate result type")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
