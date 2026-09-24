#!/usr/bin/env python3
"""Reject-only timing and fidelity screen of an exact UNICOM L/14 head fold."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
import time
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from evaluate_sop_cub_transfer import gpu_compute_pids
from evaluate_sop_reference_checkpoint import EXPECTED_SOP_RECORD_SHA256
from export_unicom_sop_embeddings import ordered_record_sha256, parse_sop_records
from PIL import Image
from probe_l14_parallel_fold import (
    assert_no_foreign_gpu_processes,
    load_authenticated_l14,
    timed_call,
    timing_summary,
)
from torch import nn
from torch.nn import functional as F
from train_sop_compact_backbone import publish_file_noreplace, sha256

from sfora.inference_head import fold_eval_affine_head


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("unicom-checkout", "checkpoint", "sop-root", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--blocks", type=int, default=10)
    parser.add_argument("--calls-per-block", type=int, default=10)
    parser.add_argument("--execute-l14-exact-head-fold", action="store_true", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (
        args.output.exists()
        or args.output.is_symlink()
        or args.blocks < 2
        or args.calls_per_block < 2
        or not torch.cuda.is_available()
        or gpu_compute_pids()
    ):
        raise ValueError("L/14 exact head fold invocation differs")
    torch.set_num_threads(16)
    torch.backends.cuda.matmul.allow_tf32 = False
    started = time.perf_counter()
    records = parse_sop_records(args.sop_root)
    if (
        ordered_record_sha256(records) != EXPECTED_SOP_RECORD_SHA256
        or sha256(args.sop_root / "Ebay_train.txt")
        != "77abb1e82af49f2f1f272dc6bd0b8480904f58742091764f9511b152cad1824e"
    ):
        raise ValueError("L/14 exact head fold SOP source differs")
    train = tuple(row for row in records if row.split == "train")
    indexes = np.linspace(0, len(train) - 1, 32, dtype=np.int64)
    selected = tuple(train[int(index)] for index in indexes)
    manifest = b"".join(hashlib.sha256(row.image_path.read_bytes()).digest() for row in selected)
    model, transform = load_authenticated_l14(args.unicom_checkout, args.checkpoint)
    images = []
    for index, row in enumerate(selected):
        data = row.image_path.read_bytes()
        if hashlib.sha256(data).digest() != manifest[index * 32 : (index + 1) * 32]:
            raise ValueError("L/14 exact head fold SOP image changed")
        with Image.open(BytesIO(data)) as image:
            images.append(transform(image.convert("RGB")))
    batch = torch.stack(images).cuda()
    model = model.cuda().eval()
    source_head = model.feature
    assert_no_foreign_gpu_processes()
    fused_head = fold_eval_affine_head(source_head)
    with torch.inference_mode():
        flattened = model.forward_features(batch)
        source = source_head(flattened)
        fused = fused_head(flattened)
        difference = (source - fused).abs()
        cosine = F.cosine_similarity(source, fused)
        if not bool(torch.isfinite(difference).all()):
            raise ValueError("L/14 exact head fold descriptor differs")
        diagnostics = {
            "max_abs_descriptor_difference": float(difference.max()),
            "mean_abs_descriptor_difference": float(difference.mean()),
            "mean_descriptor_cosine": float(cosine.mean()),
            "max_relative_l2_descriptor_difference": float(
                (torch.linalg.vector_norm(source - fused, dim=1)
                / torch.linalg.vector_norm(source, dim=1)).max()
            ),
        }
    timing: dict[str, object] = {}
    arms = {"original": source_head, "exact_fused": fused_head}
    for region in ("full_encoder", "head_only"):
        timing[region] = {}
        for size in (1, 32):
            inputs = batch[:size] if region == "full_encoder" else flattened[:size]
            expected = (size, 768)
            for head in arms.values():
                if region == "full_encoder":
                    model.feature = head
                    target: nn.Module = model
                else:
                    target = head
                for _ in range(5):
                    timed_call(target, inputs, expected)
            samples = {name: {"gpu_ms": [], "wall_ms": []} for name in arms}
            order = []
            for block in range(args.blocks):
                names = tuple(arms) if block % 2 == 0 else tuple(reversed(arms))
                for name in names:
                    order.append(name)
                    head = arms[name]
                    if region == "full_encoder":
                        model.feature = head
                        target = model
                    else:
                        target = head
                    for _ in range(args.calls_per_block):
                        gpu_ms, wall_ms = timed_call(target, inputs, expected)
                        samples[name]["gpu_ms"].append(gpu_ms)
                        samples[name]["wall_ms"].append(wall_ms)
            timing[region][str(size)] = {
                "samples": samples,
                "interleaved_order": order,
                "summary": {
                    name: {kind: timing_summary(values) for kind, values in columns.items()}
                    for name, columns in samples.items()
                },
            }
            print(
                json.dumps(
                    {
                        "region": region,
                        "batch": size,
                        "summary": timing[region][str(size)]["summary"],
                    }
                ),
                flush=True,
            )
    assert_no_foreign_gpu_processes()
    result = {
        "schema": "sfora-l14-exact-affine-head-f0-v1",
        "claim_eligible": False,
        "quality_unmeasured": True,
        "p99_contract_satisfied": False,
        "split": "32 sampled SOP training images only",
        "sampled_train_image_ids": [row.image_id for row in selected],
        "sampled_train_image_manifest_sha256": hashlib.sha256(manifest).hexdigest(),
        "diagnostics": diagnostics,
        "timing": timing,
        "timing_scope": (
            "preprocessed SOP train tensors resident on GPU; fp32 model and head weights"
        ),
        "hardware": {
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "python": platform.python_version(),
        },
        "inputs": {
            "checkpoint_sha256": sha256(args.checkpoint),
            "ordered_sop_records_sha256": EXPECTED_SOP_RECORD_SHA256,
            "source_sha256": sha256(Path(__file__)),
        },
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_host_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    publish_file_noreplace(
        args.output,
        lambda stream: stream.write(
            (json.dumps(result, sort_keys=True, allow_nan=False) + "\n").encode()
        ),
    )
    print(json.dumps({"output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
