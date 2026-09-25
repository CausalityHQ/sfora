#!/usr/bin/env python3
"""Screen the isolated cost of a live 1024-to-128 head over a full SOP fit bank."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from sfora.deployed_code_rank import smooth_ap_bank_loss
from sfora.live_head_bank import live_head_bank_loss


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def benchmark(args: argparse.Namespace) -> dict[str, Any]:
    if (
        args.rows < args.anchors * (args.positives + 1)
        or args.anchors < 1
        or args.positives < 1
        or args.warmup < 0
        or args.blocks < 1
        or args.repeats < 1
    ):
        raise ValueError("live-head cost geometry differs")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("live-head cost requires CUDA")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(179023)
    source = F.normalize(torch.randn(args.rows, 1024, device=device), dim=1)
    head = nn.Linear(1024, 128, device=device)
    with torch.no_grad():
        projected = F.normalize(head(source), dim=1).detach()
    self_rows = torch.arange(args.anchors, device=device, dtype=torch.long) * (args.positives + 1)
    positives = (
        self_rows[:, None]
        + torch.arange(1, args.positives + 1, device=device, dtype=torch.long)[None, :]
    )
    anchors = projected[self_rows].clone().detach().requires_grad_()

    def old_loss() -> torch.Tensor:
        return smooth_ap_bank_loss(anchors, projected, positives, self_rows)

    def new_loss() -> torch.Tensor:
        return live_head_bank_loss(anchors, source, head, positives, self_rows)

    with torch.no_grad():
        old_value = float(old_loss())
        new_value = float(new_loss())
    if abs(old_value - new_value) > 1e-6:
        raise ValueError("live-head cost forward parity differs")

    def one_call(name: str) -> tuple[float, int]:
        anchors.grad = None
        head.weight.grad = None
        head.bias.grad = None
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            baseline_bytes = torch.cuda.memory_allocated(device)
        else:
            baseline_bytes = 0
        synchronize(device)
        started = time.perf_counter()
        (old_loss() if name == "detached_bank" else new_loss()).backward()  # type: ignore[no-untyped-call]
        synchronize(device)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if name == "live_head" and (
            head.weight.grad is None or not bool(torch.isfinite(head.weight.grad).all())
        ):
            raise ValueError("live-head cost candidate gradient differs")
        peak_extra_bytes = (
            torch.cuda.max_memory_allocated(device) - baseline_bytes if device.type == "cuda" else 0
        )
        return elapsed_ms, peak_extra_bytes

    for _ in range(args.warmup):
        one_call("detached_bank")
        one_call("live_head")
    timings: dict[str, list[float]] = {"detached_bank": [], "live_head": []}
    peaks: dict[str, list[int]] = {"detached_bank": [], "live_head": []}
    for block in range(args.blocks):
        order = (
            ("detached_bank", "live_head")
            if block % 2 == 0
            else (
                "live_head",
                "detached_bank",
            )
        )
        for name in order:
            for _ in range(args.repeats):
                elapsed_ms, peak_extra_bytes = one_call(name)
                timings[name].append(elapsed_ms)
                peaks[name].append(peak_extra_bytes)
    return {
        "schema": "sfora-sop-siglip2-live-head-isolated-cost-v1",
        "claim_eligible": False,
        "inventory": "synthetic unit-normalized source rows; SOP fit-bank dimensions",
        "measurement": "isolated rank loss forward plus backward; no encoder, optimizer, or export",
        "hardware": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU smoke",
        "device": args.device,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "python_version": platform.python_version(),
        "rows": args.rows,
        "source_width": 1024,
        "output_width": 128,
        "anchors": args.anchors,
        "positives_per_anchor": args.positives,
        "seed": 179023,
        "warmup_calls_per_arm": args.warmup,
        "timed_blocks": args.blocks,
        "calls_per_block_per_arm": args.repeats,
        "forward_loss_difference": new_value - old_value,
        "persistent_bank_bytes": {
            "detached_bank": projected.numel() * projected.element_size(),
            "live_head": source.numel() * source.element_size(),
        },
        "arms": {
            name: {
                "calls": len(timings[name]),
                "median_ms": statistics.median(timings[name]),
                "p95_ms": percentile(timings[name], 0.95),
                "max_incremental_allocated_bytes": max(peaks[name]),
            }
            for name in timings
        },
        "source_sha256": {
            "script": sha256(Path(__file__)),
            "detached_loss": sha256(Path(smooth_ap_bank_loss.__code__.co_filename)),
            "live_loss": sha256(Path(live_head_bank_loss.__code__.co_filename)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--rows", type=int, default=53_700)
    parser.add_argument("--anchors", type=int, default=64)
    parser.add_argument("--positives", type=int, default=11)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise ValueError("live-head cost output already exists")
    result = benchmark(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{args.output.name}.", dir=args.output.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(result, stream, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, args.output)
    finally:
        Path(temporary).unlink(missing_ok=True)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
