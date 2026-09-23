#!/usr/bin/env python3
"""Paired DGX training-step cost screen for float and deployed-code rank loss."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import time
from pathlib import Path

import torch

from sfora.deployed_code_rank import smooth_ap_float_loss, smooth_ap_packed_loss


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _percentile(samples: list[int], fraction: float) -> int:
    return sorted(samples)[math.ceil(len(samples) * fraction) - 1]


def _run(arm: str, base: torch.Tensor, labels: tuple[int, ...], calls: int) -> dict[str, object]:
    device = base.device
    torch.cuda.reset_peak_memory_stats(device)
    samples: list[int] = []
    losses: list[float] = []
    for call in range(calls + 5):
        value = base.clone().detach().requires_grad_(True)
        torch.cuda.synchronize(device)
        started = time.perf_counter_ns()
        if arm == "packed":
            loss = smooth_ap_packed_loss(value, labels)
        else:
            loss = smooth_ap_float_loss(value, labels)
        loss.backward()
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter_ns() - started
        if not bool(torch.isfinite(value.grad).all()):
            raise ValueError("rank step gradient is nonfinite")
        if call >= 5:
            samples.append(elapsed)
            losses.append(float(loss.detach()))
    return {
        "arm": arm,
        "calls": calls,
        "samples_ns": samples,
        "p50_ns": _percentile(samples, 0.50),
        "p95_ns": _percentile(samples, 0.95),
        "p99_ns": _percentile(samples, 0.99),
        "mean_loss": math.fsum(losses) / len(losses),
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--calls", type=int, default=50)
    args = parser.parse_args()
    if args.calls < 20 or args.output.exists() or not torch.cuda.is_available():
        raise ValueError("rank step benchmark authority differs")
    torch.manual_seed(179019)
    torch.cuda.manual_seed_all(179019)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")
    rows = []
    for batch in (32, 128):
        base = torch.randn(batch, 128, device=device)
        labels = tuple(index // 4 for index in range(batch))
        for arm in ("float", "packed", "packed", "float"):
            rows.append({"batch": batch, **_run(arm, base, labels, args.calls)})
    receipt = {
        "schema": "sfora-deployed-code-rank-step-v1",
        "scope": (
            "synthetic fixed-feature forward/backward cost only; "
            "no optimizer, image encoder, or recall"
        ),
        "seed": 179019,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
        "python": platform.python_version(),
        "script_sha256": _sha256(Path(__file__)),
        "packed_module_sha256": _sha256(
            Path(__file__).parents[1] / "src/sfora/deployed_code_rank.py"
        ),
        "rows": rows,
    }
    payload = (json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n").encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"output": str(args.output), "rows": len(rows)}), flush=True)


if __name__ == "__main__":
    main()
