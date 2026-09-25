#!/usr/bin/env python3
"""Instrument the frozen failed ArcFace trainer without changing its RNG path."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
from pathlib import Path

ORIGINAL_SHA256 = "32cce40aa7133bfc602c41b8d5fdb6e76609c71336b907a87c964cbb58348ade"
NEEDLE = """        torch.nn.utils.clip_grad_norm_(
            list(vision.parameters()) + list(head.parameters()) + [classifier],
            1.0,
            error_if_nonfinite=True,
        )"""
REPLACEMENT = """        if step >= 500:
            bad_gradients = []
            named = (
                [(f"vision.{name}", parameter) for name, parameter in vision.named_parameters()]
                + [(f"head.{name}", parameter) for name, parameter in head.named_parameters()]
                + [("classifier", classifier)]
            )
            for name, parameter in named:
                gradient = parameter.grad
                if gradient is not None and not bool(torch.isfinite(gradient).all()):
                    bad_gradients.append({
                        "name": name,
                        "dtype": str(gradient.dtype),
                        "nan_count": int(torch.isnan(gradient).sum()),
                        "inf_count": int(torch.isinf(gradient).sum()),
                    })
            if bad_gradients:
                digest = hashlib.sha256()
                for key in sorted(batch):
                    digest.update(key.encode())
                    digest.update(batch[key].contiguous().numpy().tobytes())
                digest.update(target.detach().cpu().contiguous().numpy().tobytes())
                print(json.dumps({
                    "gradient_failure_step": step,
                    "loss": float(loss.detach()),
                    "grad_scaler": scaler.get_scale(),
                    "input_batch_sha256": digest.hexdigest(),
                    "bad_gradients": bad_gradients,
                }, sort_keys=True), flush=True)
                raise RuntimeError("diagnostic nonfinite gradient")
        torch.nn.utils.clip_grad_norm_(
            list(vision.parameters()) + list(head.parameters()) + [classifier],
            1.0,
            error_if_nonfinite=True,
        )"""


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
        raise ValueError("ArcFace gradient diagnostic source differs")
    instrumented = original.replace(NEEDLE, REPLACEMENT)
    instrumented_sha = hashlib.sha256(instrumented.encode()).hexdigest()
    patch = "".join(
        difflib.unified_diff(
            original.splitlines(keepends=True),
            instrumented.splitlines(keepends=True),
            fromfile="frozen/train_sop_siglip2_compact.py",
            tofile="diagnostic/train_sop_siglip2_compact.py",
        )
    )
    args.trainer.write_text(instrumented)
    args.diff.write_text(patch)
    args.receipt.write_text(
        json.dumps(
            {
                "schema": "sfora-sop-siglip2-arcface-gradient-instrumentation-v1",
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
