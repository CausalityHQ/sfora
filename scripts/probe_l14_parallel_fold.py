#!/usr/bin/env python3
"""Falsify the latency premise of a parallel residual fold of UNICOM L/14."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
from io import BytesIO
from pathlib import Path
from typing import BinaryIO

import numpy as np
import torch
from evaluate_sop_cub_transfer import gpu_compute_pids
from evaluate_sop_reference_checkpoint import EXPECTED_SOP_RECORD_SHA256
from export_unicom_sop_embeddings import model_authority, ordered_record_sha256, parse_sop_records
from PIL import Image
from torch import nn
from torch.nn import functional as F
from train_sop_compact_backbone import publish_file_noreplace, sha256


class ParallelFoldBlock(nn.Module):
    """Fuse two blocks' attention and MLP branches evaluated at the same input.

    This is an approximation to their sequential composition. It preserves
    both blocks' parameters and branch sums but omits sequential interactions.
    """

    def __init__(self, first: nn.Module, second: nn.Module) -> None:
        super().__init__()
        if not all(
            isinstance(block.norm1, nn.LayerNorm)
            and isinstance(block.norm2, nn.LayerNorm)
            and isinstance(block.attn.qkv, nn.Linear)
            and isinstance(block.attn.proj, nn.Linear)
            and isinstance(block.mlp.fc1, nn.Linear)
            and isinstance(block.mlp.fc2, nn.Linear)
            and isinstance(block.mlp.act, nn.ReLU6)
            for block in (first, second)
        ):
            raise ValueError("L/14 fold block architecture differs")
        dim = first.norm1.normalized_shape[0]
        heads = first.attn.num_heads
        hidden = first.mlp.fc1.out_features
        epsilon = first.norm1.eps
        if (
            dim < 2
            or heads < 1
            or dim % heads
            or any(
                block.norm1.normalized_shape != (dim,)
                or block.norm2.normalized_shape != (dim,)
                or block.norm1.eps != epsilon
                or block.norm2.eps != epsilon
                or block.attn.num_heads != heads
                or block.attn.qkv.weight.shape != (3 * dim, dim)
                or block.attn.qkv.bias is not None
                or block.attn.proj.weight.shape != (dim, dim)
                or block.mlp.fc1.weight.shape != (hidden, dim)
                or block.mlp.fc2.weight.shape != (dim, hidden)
                or block.training
                for block in (first, second)
            )
        ):
            raise ValueError("L/14 fold block geometry differs")
        self.dim = dim
        self.heads = 2 * heads
        self.epsilon = epsilon
        self.qkv = nn.Linear(dim, 6 * dim, bias=True)
        self.attn_proj = nn.Linear(2 * dim, dim, bias=True)
        self.mlp_fc1 = nn.Linear(dim, 2 * hidden, bias=True)
        self.act = nn.ReLU6()
        self.mlp_fc2 = nn.Linear(2 * hidden, dim, bias=True)
        with torch.no_grad():
            qkv_weights = []
            qkv_biases = []
            for part in range(3):
                for block in (first, second):
                    weight = block.attn.qkv.weight[part * dim : (part + 1) * dim]
                    qkv_weights.append(weight * block.norm1.weight)
                    qkv_biases.append(weight @ block.norm1.bias)
            self.qkv.weight.copy_(torch.cat(qkv_weights, dim=0))
            self.qkv.bias.copy_(torch.cat(qkv_biases))
            self.attn_proj.weight.copy_(
                torch.cat((first.attn.proj.weight, second.attn.proj.weight), dim=1)
            )
            self.attn_proj.bias.copy_(first.attn.proj.bias + second.attn.proj.bias)
            fc1_weights = []
            fc1_biases = []
            for block in (first, second):
                weight = block.mlp.fc1.weight
                fc1_weights.append(weight * block.norm2.weight)
                fc1_biases.append(block.mlp.fc1.bias + weight @ block.norm2.bias)
            self.mlp_fc1.weight.copy_(torch.cat(fc1_weights, dim=0))
            self.mlp_fc1.bias.copy_(torch.cat(fc1_biases))
            self.mlp_fc2.weight.copy_(
                torch.cat((first.mlp.fc2.weight, second.mlp.fc2.weight), dim=1)
            )
            self.mlp_fc2.bias.copy_(first.mlp.fc2.bias + second.mlp.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, rows, dim = x.shape
        if dim != self.dim:
            raise ValueError("L/14 fold input geometry differs")
        with torch.autocast(device_type="cuda", enabled=x.is_cuda):
            base = F.layer_norm(x, (dim,), None, None, self.epsilon)
            qkv = self.qkv(base).reshape(batch, rows, 3, self.heads, dim // (self.heads // 2))
            q, k, v = qkv.permute(2, 0, 3, 1, 4)
            attended = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
            attended = attended.transpose(1, 2).reshape(batch, rows, 2 * dim)
            attention_update = self.attn_proj(attended)
            mlp_update = self.mlp_fc2(self.act(self.mlp_fc1(base)))
            return x + attention_update + mlp_update


class BlocksOnly(nn.Module):
    """Time only transformer blocks, excluding the shared patch and output head."""

    def __init__(self, blocks: nn.ModuleList) -> None:
        super().__init__()
        self.blocks = blocks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("unicom-checkout", "checkpoint", "sop-root", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--blocks", type=int, default=10)
    parser.add_argument("--calls-per-block", type=int, default=10)
    parser.add_argument("--execute-l14-fold-falsifier", action="store_true", required=True)
    return parser.parse_args()


def load_authenticated_l14(checkout: Path, checkpoint: Path) -> tuple[nn.Module, object]:
    authority = model_authority("l14-336")
    if (
        checkpoint.name != authority.checkpoint_filename
        or sha256(checkpoint) != authority.checkpoint_sha256
        or checkout.is_symlink()
        or checkpoint.is_symlink()
    ):
        raise ValueError("L/14 fold source authority differs")
    revision = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(checkout), "status", "--porcelain", "--untracked-files=no"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != authority.revision or status:
        raise ValueError("L/14 fold source checkout differs")
    package_root = (checkout / "unicom").resolve()
    sys.path.insert(0, str(package_root))
    try:
        unicom = importlib.import_module("unicom")
    finally:
        sys.path.pop(0)
    if Path(unicom.__file__).resolve() != package_root / "unicom" / "__init__.py":
        raise ValueError("L/14 fold imported source differs")
    # The checkpoint hash was checked above, so unicom.load uses the local
    # verified file and cannot take its download branch.
    model, transform = unicom.load(authority.load_name, download_root=str(checkpoint.parent))
    return model.eval(), transform


@torch.inference_mode()
def timed_call(
    model: nn.Module, inputs: torch.Tensor, expected_shape: tuple[int, ...]
) -> tuple[float, float]:
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter_ns()
    start.record()
    result = model(inputs)
    end.record()
    end.synchronize()
    wall_ms = (time.perf_counter_ns() - wall_start) / 1e6
    if result.shape != expected_shape or not bool(torch.isfinite(result).all()):
        raise ValueError("L/14 fold encoder output differs")
    return start.elapsed_time(end), wall_ms


def timing_summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "p50_ms": float(np.quantile(array, 0.50)),
        "p95_ms": float(np.quantile(array, 0.95)),
        "p99_ms_diagnostic_only": float(np.quantile(array, 0.99)),
        "mean_ms": float(np.mean(array)),
    }


def assert_no_foreign_gpu_processes() -> None:
    if gpu_compute_pids() - {os.getpid()}:
        raise ValueError("L/14 fold GPU process overlap")


def main() -> None:
    args = parse_args()
    if (
        args.output.exists()
        or args.output.is_symlink()
        or args.blocks < 2
        or args.calls_per_block < 2
    ):
        raise ValueError("L/14 fold invocation differs")
    started = time.perf_counter()
    torch.set_num_threads(16)
    torch.backends.cuda.matmul.allow_tf32 = False
    if not torch.cuda.is_available() or gpu_compute_pids():
        raise ValueError("L/14 fold GPU unavailable or occupied")
    authority = model_authority("l14-336")
    records = parse_sop_records(args.sop_root)
    if (
        ordered_record_sha256(records) != EXPECTED_SOP_RECORD_SHA256
        or sha256(args.sop_root / "Ebay_train.txt")
        != "77abb1e82af49f2f1f272dc6bd0b8480904f58742091764f9511b152cad1824e"
    ):
        raise ValueError("L/14 fold SOP record authority differs")
    train = tuple(row for row in records if row.split == "train")
    indexes = np.linspace(0, len(train) - 1, 32, dtype=np.int64)
    selected = tuple(train[int(index)] for index in indexes)
    image_manifest = b"".join(
        hashlib.sha256(row.image_path.read_bytes()).digest() for row in selected
    )
    reference, transform = load_authenticated_l14(args.unicom_checkout, args.checkpoint)
    with torch.inference_mode():
        images = []
        for index, row in enumerate(selected):
            image_bytes = row.image_path.read_bytes()
            if (
                hashlib.sha256(image_bytes).digest()
                != image_manifest[index * 32 : (index + 1) * 32]
            ):
                raise ValueError("L/14 fold SOP image changed")
            with Image.open(BytesIO(image_bytes)) as image:
                images.append(transform(image.convert("RGB")))
        batch = torch.stack(images).cuda()
    folded = copy.deepcopy(reference)
    old_blocks = folded.blocks
    if len(old_blocks) != 24:
        raise ValueError("L/14 fold source depth differs")
    folded.blocks = nn.ModuleList(
        ParallelFoldBlock(old_blocks[index], old_blocks[index + 1]) for index in range(0, 24, 2)
    )
    del old_blocks
    assert_no_foreign_gpu_processes()
    reference = reference.cuda().eval()
    folded = folded.cuda().eval()
    block_arms = {
        "sequential": BlocksOnly(reference.blocks).eval(),
        "folded_parallel": BlocksOnly(folded.blocks).eval(),
    }
    with torch.inference_mode():
        tokens = reference.patch_embed(batch) + reference.pos_embed
        first, second = reference.blocks[:2]
        probe = tokens[:1]
        with torch.autocast(device_type="cuda"):
            branch_oracle = probe.clone()
            for block in (first, second):
                branch_oracle = branch_oracle + block.attn(block.norm1(probe))
                branch_oracle = branch_oracle + block.mlp(block.norm2(probe))
            fused_probe = folded.blocks[0](probe)
        branch_error = (fused_probe.float() - branch_oracle.float()).abs()
        full_reference = reference(batch)
        full_folded = folded(batch)
        embedding_cosine = F.cosine_similarity(full_reference.float(), full_folded.float())
        if not (
            bool(torch.isfinite(branch_error).all())
            and bool(torch.isfinite(embedding_cosine).all())
        ):
            raise ValueError("L/14 fold real-weight diagnostic differs")
        diagnostics = {
            "first_pair_parallel_oracle_max_abs_error": float(branch_error.max()),
            "first_pair_parallel_oracle_mean_abs_error": float(branch_error.mean()),
            "reference_vs_folded_embedding_cosine_mean": float(embedding_cosine.mean()),
            "reference_vs_folded_embedding_cosine_min": float(embedding_cosine.min()),
            "quality_claim": False,
        }
    resident_cuda_bytes = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    preflight_seconds = time.perf_counter() - started
    arms = {"sequential": reference, "folded_parallel": folded}
    results: dict[str, dict[str, dict[str, object]]] = {}
    for region, region_arms in (("full_encoder", arms), ("blocks_only", block_arms)):
        results[region] = {}
        for size in (1, 32):
            inputs = (batch if region == "full_encoder" else tokens)[:size]
            expected_shape = (
                (size, 768)
                if region == "full_encoder"
                else (size, reference.patch_embed.num_patches, reference.dim)
            )
            for model in region_arms.values():
                for _ in range(5):
                    timed_call(model, inputs, expected_shape)
            samples = {name: {"gpu_ms": [], "wall_ms": []} for name in region_arms}
            order: list[str] = []
            for block in range(args.blocks):
                names = (
                    ("sequential", "folded_parallel")
                    if block % 2 == 0
                    else ("folded_parallel", "sequential")
                )
                for name in names:
                    order.append(name)
                    for _ in range(args.calls_per_block):
                        gpu_ms, wall_ms = timed_call(region_arms[name], inputs, expected_shape)
                        samples[name]["gpu_ms"].append(gpu_ms)
                        samples[name]["wall_ms"].append(wall_ms)
            results[region][str(size)] = {
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
                        "summary": results[region][str(size)]["summary"],
                    }
                ),
                flush=True,
            )
    assert_no_foreign_gpu_processes()
    result = {
        "schema": "sfora-unicom-l14-parallel-fold-encoder-timing-f0-v1",
        "claim_eligible": False,
        "quality_unmeasured": True,
        "timing_scope": (
            "preprocessed SOP train tensors resident on GPU; full encoder and blocks only"
        ),
        "gate_scope": "short paired reject-only screen; cannot advance a folded model",
        "omitted_from_full_gate": [
            "image decode and preprocessing",
            "packing and top-k search",
            "B/16 and OML efficient controls",
            "10000 calls per cell and p99 confidence interval",
            "quality retention and healing",
        ],
        "fold_variant": "intra- and inter-block parallel residual optimistic timing bound",
        "p99_contract_satisfied": False,
        "fold_semantics": (
            "two source blocks approximated by sum of two attention and MLP branches at same input"
        ),
        "blocks": args.blocks,
        "calls_per_block": args.calls_per_block,
        "batch_sizes": [1, 32],
        "timing": results,
        "diagnostics": diagnostics,
        "preflight_seconds": preflight_seconds,
        "total_seconds": time.perf_counter() - started,
        "resident_cuda_allocated_bytes_before_timing": resident_cuda_bytes,
        "peak_cuda_allocated_bytes_during_timing": torch.cuda.max_memory_allocated(),
        "peak_host_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "hardware": {
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "python": platform.python_version(),
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "weight_dtype": str(next(reference.parameters()).dtype),
        },
        "inputs": {
            "unicom_revision": authority.revision,
            "checkpoint_sha256": authority.checkpoint_sha256,
            "ordered_sop_records_sha256": EXPECTED_SOP_RECORD_SHA256,
            "train_metadata_sha256": sha256(args.sop_root / "Ebay_train.txt"),
            "sampled_train_image_ids": [row.image_id for row in selected],
            "sampled_train_image_manifest_sha256": hashlib.sha256(image_manifest).hexdigest(),
            "source_sha256": sha256(Path(__file__)),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def write_receipt(stream: BinaryIO) -> None:
        stream.write((json.dumps(result, sort_keys=True, allow_nan=False) + "\n").encode())

    publish_file_noreplace(args.output, write_receipt)
    print(json.dumps({"output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
