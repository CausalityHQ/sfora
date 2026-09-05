#!/usr/bin/env python3
"""Paired full-path latency probe for the sealed tokenwise spatial tail."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import resource
import sys
from collections.abc import Callable, Mapping
from functools import partial
from pathlib import Path
from time import perf_counter_ns
from typing import Any, cast

import torch
from torch import nn
from torch.nn import functional as F

from sfora.siglip_depth_recovery import speed_gate
from sfora.siglip_proxy_control import PooledProxyAnchorModel, SiglipProxyControlConfig

_SCRIPTS = str(Path(__file__).resolve().parent)
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)


def truncate_siglip_vision(vision_model: nn.Module, *, retained_blocks: int = 18) -> nn.Module:
    """Copy a 27-block vision tower and retain the exact leading token field."""

    encoder = cast(Any, getattr(vision_model, "encoder", None))
    layers = cast(Any, getattr(encoder, "layers", None))
    config = cast(Any, getattr(vision_model, "config", None))
    if (
        not isinstance(vision_model, nn.Module)
        or vision_model.training
        or not isinstance(layers, nn.ModuleList)
        or len(layers) != 27
        or type(getattr(config, "num_hidden_layers", None)) is not int
        or config.num_hidden_layers != 27
        or type(retained_blocks) is not int
        or not 0 < retained_blocks < 27
    ):
        raise ValueError("spatial tail latency topology differs")
    student = cast(Any, copy.deepcopy(vision_model))
    student.encoder.layers = nn.ModuleList(student.encoder.layers[:retained_blocks])
    student.config.num_hidden_layers = retained_blocks
    return cast(nn.Module, student.eval().requires_grad_(False))


class SpatialTailInference(nn.Module):
    """Execute the leading vision blocks, sealed tail, and frozen readout."""

    def __init__(self, vision_model: nn.Module, tail: nn.Module, readout: nn.Module) -> None:
        super().__init__()
        if any(module.training for module in (vision_model, tail, readout)):
            raise ValueError("spatial tail latency modules must be frozen and in eval mode")
        self.vision_model = vision_model
        self.tail = tail
        self.readout = readout
        self.requires_grad_(False)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        vision = cast(Any, self.vision_model)
        hidden = vision.embeddings(pixel_values)
        encoded: Any = vision.encoder(inputs_embeds=hidden, return_dict=True)
        tokens = getattr(encoded, "last_hidden_state", None)
        if not isinstance(tokens, torch.Tensor):
            raise ValueError("spatial tail latency encoder output differs")
        return cast(torch.Tensor, self.readout(self.tail(tokens)))


def _absolute_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("path must be absolute")
    return path


def _sha256(value: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise argparse.ArgumentTypeError("digest must be lowercase SHA-256")
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse a strict local-only, evaluation-free latency command."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-root", type=_absolute_path, required=True)
    parser.add_argument("--spatial-artifact", type=_absolute_path, required=True)
    parser.add_argument("--spatial-artifact-sha256", type=_sha256, required=True)
    parser.add_argument("--output", type=_absolute_path, required=True)
    parser.add_argument(
        "--execute-spatial-tail-latency", action="store_true", required=True
    )
    effective = list(sys.argv[1:] if argv is None else argv)
    flags = [value.split("=", 1)[0] for value in effective if value.startswith("--")]
    duplicates = sorted({flag for flag in flags if flags.count(flag) > 1})
    if duplicates:
        parser.error(f"duplicate arguments are forbidden: {duplicates!r}")
    return parser.parse_args(effective)


def _file_sha256(path: Path) -> str:
    if not path.is_file() or path.is_symlink():
        raise ValueError("spatial tail latency artifact must be a regular file")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def spatial_tail_batch_starts() -> tuple[int, ...]:
    """Return the fixed warmup-plus-measurement schedule over all 128 images."""

    return tuple((round_index * 8) % 128 for round_index in range(110))


def teacher_latency_descriptor(
    model: PooledProxyAnchorModel, inputs: torch.Tensor
) -> torch.Tensor:
    """Compute the registered teacher descriptor without validation synchronizations."""

    pooled = model.tower(inputs)
    projected = model.projection(pooled).float()
    return F.normalize(projected, dim=1)


def measure_spatial_tail_pair(
    forwards: dict[str, Callable[[Any], torch.Tensor]],
    bank: Any,
    *,
    window: int,
    synchronize: Callable[[], None],
    clock: Callable[[], int] = perf_counter_ns,
    progress: Callable[[int], None] | None = None,
) -> dict[str, list[int]]:
    """Measure a paired teacher/student window while covering all source images."""

    if set(forwards) != {"full", "student"} or len(bank) != 128 or window not in (0, 1, 2):
        raise ValueError("spatial tail latency pair inventory differs")
    samples: dict[str, list[int]] = {"full": [], "student": []}
    with torch.inference_mode():
        for round_index, start in enumerate(spatial_tail_batch_starts()):
            batch = bank[start : start + 8]
            order = (
                ("full", "student")
                if (round_index + window) % 2 == 0
                else ("student", "full")
            )
            for name in order:
                synchronize()
                began = clock()
                output = forwards[name](batch)
                synchronize()
                elapsed = clock() - began
                if not isinstance(output, torch.Tensor) or not bool(torch.isfinite(output).all()):
                    raise ValueError("spatial tail latency output is nonfinite")
                if type(elapsed) is not int or elapsed <= 0:
                    raise ValueError("spatial tail latency clock failed to advance")
                if torch.cuda.is_available() and torch.cuda.max_memory_reserved() >= 96 * 1024**3:
                    raise RuntimeError("spatial tail latency CUDA memory limit")
                if round_index >= 10:
                    samples[name].append(elapsed)
                del output
            if progress is not None and (round_index + 1) % 10 == 0:
                progress(round_index + 1)
    return samples


def run_probe(arguments: argparse.Namespace) -> dict[str, object]:
    """Measure the frozen full teacher against the frozen leading-18 tokenwise tail."""

    import run_siglip_proxy_control as control
    from probe_siglip_depth_recovery import (
        _release_cpu_pages,
        authenticate_control,
        select_speed_images,
    )
    from probe_siglip_spatial_tail_recovery import (
        FrozenTeacherReadout,
        LatentInteractionTail,
        TokenwiseTailControl,
        _load_spatial_tail_artifact,
    )

    if arguments.output.exists() or arguments.output.is_symlink():
        raise FileExistsError(arguments.output)
    if _file_sha256(arguments.spatial_artifact) != arguments.spatial_artifact_sha256:
        raise ValueError("spatial tail latency artifact digest differs")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("spatial tail latency requires CUDA bf16")
    control.require_control_determinism(torch.device("cuda"))
    import transformers

    if transformers.__version__ != "5.12.1":
        raise RuntimeError("spatial tail latency Transformers version differs from5.12.1")
    began = perf_counter_ns()
    receipt, checkpoint = authenticate_control(arguments.control_root)
    bands = control.load_control_examples()
    images, rows, pixels_sha = select_speed_images(bands)
    del bands
    _release_cpu_pages()
    config = SiglipProxyControlConfig()
    tower, processor = control.load_siglip_control_components(config=config)
    full = PooledProxyAnchorModel(
        tower=tower, input_dimensions=1152, embedding_dimensions=512, class_count=49
    ).float()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    if (
        payload.get("schema") != "sfora-siglip-proxy-checkpoint-payload-v1"
        or payload.get("claim_eligible") is not False
        or payload.get("seed") != 17
        or payload.get("completed_epoch") != 60
        or payload.get("config_sha256") != receipt["config_sha256"]
        or not isinstance(payload.get("model_state"), Mapping)
    ):
        raise ValueError("spatial tail latency checkpoint authority differs")
    full.load_state_dict(payload["model_state"], strict=True)
    del payload, tower
    full.eval().requires_grad_(False)
    vision = cast(Any, full.tower.vision_model)
    readout_template = FrozenTeacherReadout(
        vision.post_layernorm, vision.head, full.projection
    ).eval()
    tokenwise, _interaction, readout = _load_spatial_tail_artifact(
        arguments.spatial_artifact,
        TokenwiseTailControl(1152).eval(),
        LatentInteractionTail(1152).eval(),
        readout_template,
    )
    student = SpatialTailInference(
        truncate_siglip_vision(vision, retained_blocks=18), tokenwise, readout
    ).eval()
    full = full.to("cuda")
    student = student.to("cuda")
    pixels = control.preprocess_control_evaluation(processor, list(images)).to("cuda")

    def encode_full(batch: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            return teacher_latency_descriptor(full, batch)

    def encode_student(batch: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            encoded = student(batch)
        if not isinstance(encoded, torch.Tensor):
            raise ValueError("spatial tail latency student output differs")
        return encoded

    def pipeline(forward: Callable[[torch.Tensor], torch.Tensor], batch: Any) -> torch.Tensor:
        inputs = control.preprocess_control_evaluation(processor, list(batch)).to("cuda")
        return forward(inputs)

    windows: list[dict[str, object]] = []
    for window in range(3):
        scopes: dict[str, object] = {}
        for scope in ("pipeline", "encoder"):
            encoders: dict[str, Callable[[Any], torch.Tensor]] = {
                "full": encode_full,
                "student": encode_student,
            }
            forwards: dict[str, Callable[[Any], torch.Tensor]] = (
                {name: partial(pipeline, forward) for name, forward in encoders.items()}
                if scope == "pipeline"
                else encoders
            )

            def progress(rounds: int, w: int = window, s: str = scope) -> None:
                print(f"spatial-tail-latency:window={w} scope={s} rounds={rounds}/110", flush=True)

            scopes[scope] = measure_spatial_tail_pair(
                forwards,
                images if scope == "pipeline" else pixels,
                window=window,
                synchronize=torch.cuda.synchronize,
                progress=progress,
            )
        windows.append(scopes)
    typed_windows = cast(list[dict[str, Any]], windows)
    return {
        "schema": "sfora-siglip-spatial-tail-latency-v1",
        "claim_eligible": False,
        "quality_measured": False,
        "external_evaluation_access": False,
        "checkpoint_sha256": receipt["checkpoint"]["sha256"],
        "spatial_artifact_sha256": arguments.spatial_artifact_sha256,
        "input_rows": rows,
        "original_rgb_sha256": pixels_sha,
        "retained_leading_blocks": 18,
        "selected_arm": "tokenwise-control",
        "timing": {
            "batch_size": 8,
            "warmups_per_window": 10,
            "samples_per_window": 100,
            "pair_order": "reverse-on-odd-round-plus-window",
            "pipeline_scope": "resident-original-RGB,processor,transfer,encoder,tail,readout",
            "encoder_scope": "resident-device-pixels,encoder,tail,readout",
            "windows": windows,
        },
        "speed_passed": speed_gate(typed_windows),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(),
        },
        "resources": {
            "elapsed_ns": perf_counter_ns() - began,
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
        },
    }


def main(argv: list[str] | None = None) -> int:
    arguments = parse_args(argv)
    result = run_probe(arguments)
    raw = (
        json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    partial = arguments.output.with_name(f"{arguments.output.name}.partial")
    if partial.exists() or partial.is_symlink():
        raise FileExistsError(partial)
    partial.write_bytes(raw)
    partial.replace(arguments.output)
    print(
        f"spatial-tail-latency:COMPLETE speed_passed={result['speed_passed']} "
        f"sha256={hashlib.sha256(raw).hexdigest()}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
