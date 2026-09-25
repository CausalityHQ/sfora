#!/usr/bin/env python3
"""GB10 timing gate for a full-fit, worst-positive-count SmoothAP bank step."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import time
from pathlib import Path

import torch
from torch.nn import functional as F

import sfora.deployed_code_rank as rank_module
from sfora.deployed_code_rank import smooth_ap_bank_loss


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--expected-preflight-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.output.exists()
        or args.output.is_symlink()
        or not torch.cuda.is_available()
        or sha256(args.preflight) != args.expected_preflight_sha256
    ):
        raise ValueError("bank timing authority differs")
    preflight = json.loads(args.preflight.read_text())
    if preflight["decision"] != "raw_bank_allowed" or preflight["errors"] != 4_296:
        raise ValueError("raw-bank timing branch is ineligible")
    torch.manual_seed(179019)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda:0")
    bank = F.normalize(torch.randn((53_700, 128), device=device), dim=1)
    anchors = F.normalize(torch.randn((64, 128), device=device), dim=1).requires_grad_()
    self_ordinals = torch.arange(64, device=device)
    positives = (torch.arange(11, device=device)[None, :] + 64).expand(64, -1)

    def one_step() -> float:
        anchors.grad = None
        started = time.perf_counter()
        loss = smooth_ap_bank_loss(anchors, bank, positives, self_ordinals)
        loss.backward()
        bank[self_ordinals] = anchors.detach()
        torch.cuda.synchronize(device)
        return time.perf_counter() - started

    for _ in range(5):
        one_step()
    torch.cuda.reset_peak_memory_stats(device)
    samples = [one_step() for _ in range(20)]
    median = statistics.median(samples)
    result = {
        "schema": "sfora-sop-siglip2-member-bank-step-cost-v1",
        "claim_eligible": False,
        "preflight_sha256": args.expected_preflight_sha256,
        "source_sha256": sha256(Path(__file__)),
        "loss_source_sha256": sha256(Path(rank_module.__file__)),
        "rows": 53_700,
        "anchors": 64,
        "positives_per_anchor": 11,
        "dimension": 128,
        "warmups": 5,
        "timed_steps": 20,
        "wall_seconds": samples,
        "median_wall_seconds": median,
        "gate_wall_seconds": 0.06,
        "gate_pass": median <= 0.06,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "hardware": {"gpu": torch.cuda.get_device_name(device), "torch": torch.__version__},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write((json.dumps(result, sort_keys=True, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("median_wall_seconds", "gate_pass", "peak_cuda_allocated_bytes")
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
