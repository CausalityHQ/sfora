#!/usr/bin/env python3
"""Compare backward precision on one authenticated SOP ArcFace failure state."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import platform
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn

from sfora.sop_compact_training import (
    CompactTrainingArm,
    compact_head_features,
    compact_training_terms,
)

MODEL_SHA256 = {
    "config.json": "172e39dbf0143b8fe22d2f08921730eb8c397967e58cb37d133161e28aa34104",
    "model.safetensors": "fa34f822f016dbb167d8d0e3a8af99b5e199aa28573360d4295252b9ec418e2a",
}
EXPECTED_BATCH_SHA256 = "070ff753eaca317954f39cd6979b6f565828198c6a1bee25c9a482002f214e9f"
EXPECTED_LOSS = 4.480570316314697
EXPECTED_BAD = {
    "vision.embeddings.patch_embedding.weight",
    "vision.embeddings.patch_embedding.bias",
    "vision.embeddings.position_embedding.weight",
    "vision.encoder.layers.0.layer_norm1.weight",
    "vision.encoder.layers.0.layer_norm1.bias",
    "vision.encoder.layers.0.self_attn.k_proj.weight",
    "vision.encoder.layers.0.self_attn.k_proj.bias",
    "vision.encoder.layers.0.self_attn.v_proj.weight",
    "vision.encoder.layers.0.self_attn.v_proj.bias",
    "vision.encoder.layers.0.self_attn.q_proj.weight",
    "vision.encoder.layers.0.self_attn.q_proj.bias",
    "vision.encoder.layers.0.self_attn.out_proj.weight",
    "vision.encoder.layers.0.self_attn.out_proj.bias",
    "vision.encoder.layers.1.self_attn.out_proj.weight",
}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def batch_sha256(batch: dict[str, torch.Tensor], target: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for key in sorted(batch):
        digest.update(key.encode())
        digest.update(batch[key].contiguous().numpy().tobytes())
    digest.update(target.contiguous().numpy().tobytes())
    return digest.hexdigest()


def inspect_gradients(named: list[tuple[str, nn.Parameter]]) -> dict[str, Any]:
    bad = []
    maxima: list[tuple[str, float]] = []
    squared_norm = 0.0
    finite = True
    fp32_grads = []
    for name, parameter in named:
        gradient = parameter.grad
        if gradient is None:
            continue
        values = gradient.detach()
        mask = torch.isfinite(values)
        if not bool(mask.all()):
            bad.append(
                {
                    "name": name,
                    "nan_count": int(torch.isnan(values).sum()),
                    "inf_count": int(torch.isinf(values).sum()),
                }
            )
            finite = False
        else:
            maximum = float(values.abs().max())
            maxima.append((name, maximum))
            squared_norm += float(values.double().square().sum())
            fp32_grads.append(values.abs())
    maxima.sort(key=lambda row: row[1], reverse=True)
    result: dict[str, Any] = {
        "bad_gradients": bad,
        "finite_parameter_gradient_count": len(maxima),
        "maxima_top_12": [{"name": name, "max_abs": maximum} for name, maximum in maxima[:12]],
        "total_norm": math.sqrt(squared_norm) if finite else None,
    }
    layer0 = dict(named)["vision.encoder.layers.0.self_attn.out_proj.weight"].grad
    if layer0 is not None:
        channel_rows = layer0.detach().reshape(layer0.shape[0], -1)
        result["layer0_out_proj_nonfinite_channels"] = (
            torch.nonzero(~torch.isfinite(channel_rows).all(dim=1)).flatten().tolist()
        )
        per_channel = channel_rows.abs().amax(dim=1)
        values, indexes = torch.topk(per_channel, k=min(12, len(per_channel)))
        result["layer0_out_proj_top_channels"] = [
            {
                "channel": int(index),
                "max_abs": float(value) if math.isfinite(float(value)) else None,
                "nonfinite": not math.isfinite(float(value)),
            }
            for value, index in zip(values, indexes, strict=True)
        ]
    if finite and fp32_grads:
        nonzero = sum(int((values > 0).sum()) for values in fp32_grads)
        if nonzero:
            result["fp16_scaled_underflow_fraction_from_parameter_grads"] = {
                str(scale): {
                    "subnormal": sum(
                        int(((values > 0) & (values * scale < 2**-24)).sum())
                        for values in fp32_grads
                    )
                    / nonzero,
                    "normal": sum(
                        int(((values > 0) & (values * scale < 2**-14)).sum())
                        for values in fp32_grads
                    )
                    / nonzero,
                }
                for scale in (128, 32, 8, 1)
            }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--expected-capture-sha256", required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.output.exists()
        or args.output.is_symlink()
        or sha256(args.capture) != args.expected_capture_sha256
        or any(sha256(args.model_snapshot / name) != value for name, value in MODEL_SHA256.items())
        or not torch.cuda.is_available()
        or not torch.cuda.is_bf16_supported()
    ):
        raise ValueError("SOP step543 backward fork authority differs")
    torch.set_num_threads(16)
    torch.backends.cuda.matmul.allow_tf32 = False
    capture = torch.load(args.capture, map_location="cpu", weights_only=True)
    if (
        capture.get("schema") != "sfora-sop-siglip2-arcface-step543-backward-fork-v1"
        or capture.get("seed") != 179020
        or capture.get("step") != 543
        or capture["target"].shape != (64,)
        or capture["batch"]["pixel_values"].shape != (64, 3, 256, 256)
        or batch_sha256(capture["batch"], capture["target"]) != EXPECTED_BATCH_SHA256
    ):
        raise ValueError("SOP step543 captured batch differs")

    from transformers import AutoModel

    full_model = AutoModel.from_pretrained(
        args.model_snapshot, local_files_only=True, use_safetensors=True, dtype=torch.float16
    )
    vision = full_model.vision_model
    del full_model
    vision = vision.float().cuda().train()
    vision.load_state_dict(capture["vision"], strict=True)
    head = nn.Linear(1024, 128).cuda().train()
    head.load_state_dict(capture["head"], strict=True)
    classifier = nn.Parameter(capture["classifier"].cuda())
    named = (
        [(f"vision.{name}", parameter) for name, parameter in vision.named_parameters()]
        + [(f"head.{name}", parameter) for name, parameter in head.named_parameters()]
        + [("classifier", classifier)]
    )
    batch = {key: value.cuda() for key, value in capture["batch"].items()}
    target = capture["target"].cuda()
    masks = torch.arange(128, device="cuda", dtype=torch.int64).unsqueeze(0)
    modes = (
        ("fp16_scale128", torch.float16, 128),
        ("fp32", None, 1),
        ("fp16_scale32", torch.float16, 32),
        ("fp16_scale8", torch.float16, 8),
        ("fp16_scale1", torch.float16, 1),
        ("bf16", torch.bfloat16, 1),
    )
    results = {}
    for name, dtype, scale in modes:
        vision.zero_grad(set_to_none=True)
        head.zero_grad(set_to_none=True)
        classifier.grad = None
        torch.set_rng_state(capture["torch_cpu_rng_state"])
        torch.cuda.set_rng_state(capture["torch_cuda_rng_state"])
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        context = (
            torch.autocast("cuda", dtype=dtype) if dtype is not None else contextlib.nullcontext()
        )
        with context:
            source = vision(**batch).pooler_output
        if source is None:
            raise ValueError("SOP step543 vision output missing")
        features = compact_head_features(source, head)
        loss, _rank = compact_training_terms(
            features,
            classifier,
            target,
            masks,
            arm=CompactTrainingArm.ARCFACE,
            arcface_margin=0.3,
            arcface_scale=64.0,
        )
        (loss * scale).backward()  # type: ignore[no-untyped-call]
        if scale != 1:
            for _parameter_name, parameter in named:
                if parameter.grad is not None:
                    parameter.grad.div_(scale)
        torch.cuda.synchronize()
        results[name] = {
            "loss": float(loss.detach()),
            "source_finite": bool(torch.isfinite(source).all()),
            "features_finite": bool(torch.isfinite(features).all()),
            "elapsed_seconds": time.perf_counter() - started,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            **inspect_gradients(named),
        }
        print(
            json.dumps(
                {
                    "mode": name,
                    "loss": results[name]["loss"],
                    "bad_count": len(results[name]["bad_gradients"]),
                    "total_norm": results[name]["total_norm"],
                }
            ),
            flush=True,
        )
    observed_bad = {row["name"] for row in results["fp16_scale128"]["bad_gradients"]}
    if abs(results["fp16_scale128"]["loss"] - EXPECTED_LOSS) > 1e-4 or observed_bad != EXPECTED_BAD:
        raise ValueError("SOP step543 backward fork failed to reproduce original gradient failure")
    result = {
        "schema": "sfora-sop-siglip2-arcface-step543-backward-fork-v1",
        "claim_eligible": False,
        "split": "SOP official TRAIN fit batch only, seed 179020, step 543",
        "batch_sha256": EXPECTED_BATCH_SHA256,
        "capture_sha256": args.expected_capture_sha256,
        "source_sha256": sha256(Path(__file__)),
        "model_file_sha256": MODEL_SHA256,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "python_version": platform.python_version(),
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "modes": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write((json.dumps(result, sort_keys=True, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())


if __name__ == "__main__":
    main()
