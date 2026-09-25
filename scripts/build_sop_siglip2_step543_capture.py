#!/usr/bin/env python3
"""Capture the exact frozen ArcFace state and batch before failed step 543."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
from pathlib import Path

ORIGINAL_SHA256 = "32cce40aa7133bfc602c41b8d5fdb6e76609c71336b907a87c964cbb58348ade"
NEEDLE = """    for step, (batch, target) in enumerate(loader, start=1):
        step_started = time.perf_counter()"""
REPLACEMENT = """    for step, (batch, target) in enumerate(loader, start=1):
        if step == 543:
            capture_path = args.output_dir / "step543_state.pt"
            torch.save({
                "schema": "sfora-sop-siglip2-arcface-step543-backward-fork-v1",
                "seed": args.seed,
                "step": step,
                "schedule_sha256": schedule_sha,
                "vision": {key: value.detach().cpu() for key, value in vision.state_dict().items()},
                "head": {key: value.detach().cpu() for key, value in head.state_dict().items()},
                "classifier": classifier.detach().cpu(),
                "batch": {key: value.detach().cpu().clone() for key, value in batch.items()},
                "target": target.detach().cpu().clone(),
                "torch_cpu_rng_state": torch.get_rng_state(),
                "torch_cuda_rng_state": torch.cuda.get_rng_state(),
            }, capture_path)
            print(json.dumps({
                "captured_step": step,
                "capture_sha256": sha256(capture_path),
            }), flush=True)
        step_started = time.perf_counter()"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--trainer", type=Path, required=True)
    parser.add_argument("--diff", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    original = args.trainer.read_text()
    original_sha = hashlib.sha256(original.encode()).hexdigest()
    if (
        original_sha != ORIGINAL_SHA256
        or original.count(NEEDLE) != 1
        or args.diff.exists()
        or args.receipt.exists()
    ):
        raise ValueError("ArcFace capture source differs")
    instrumented = original.replace(NEEDLE, REPLACEMENT)
    instrumented_sha = hashlib.sha256(instrumented.encode()).hexdigest()
    patch = "".join(
        difflib.unified_diff(
            original.splitlines(keepends=True),
            instrumented.splitlines(keepends=True),
            fromfile="frozen/train_sop_siglip2_compact.py",
            tofile="capture/train_sop_siglip2_compact.py",
        )
    )
    args.trainer.write_text(instrumented)
    args.diff.write_text(patch)
    args.receipt.write_text(
        json.dumps(
            {
                "schema": "sfora-sop-siglip2-step543-capture-instrumentation-v1",
                "original_sha256": original_sha,
                "instrumented_sha256": instrumented_sha,
                "diff_sha256": hashlib.sha256(patch.encode()).hexdigest(),
                "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            },
            sort_keys=True,
        )
        + "\n"
    )
    print(instrumented_sha, flush=True)


if __name__ == "__main__":
    main()
