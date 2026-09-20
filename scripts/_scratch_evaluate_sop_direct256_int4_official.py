#!/usr/bin/env python3
"""Score the frozen learned 256-D int4 head once on observed SOP test data."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path

import torch
from _scratch_sop_direct256_int4 import (
    DIRECT128_SHA256,
    SOURCE_SHA256,
    TEACHER_SHA256,
    file_sha256,
    fit_pack_decode_int4,
)
from probe_sop_relational_linear import load_paired_archives, score_symmetric
from torch.nn import functional as F

from sfora.deterministic_similarity_runtime import configure_deterministic_similarity_runtime

CHECKPOINT_SHA256 = "9094d27813e520af57b46e56046f3d9c9e23c519111b9bde2d3aef57fd778647"
PARENT_RESULT_SHA256 = "b0b2922a1d9f1e01cb57823961d5c60bf37e712597a4e31b57b1078f3c97b9dd"
DIRECT128_FLOAT_MAP = 0.4880782043774452
DIRECT128_INT8_MAP = 0.4879665058036047
TARGET_MAP = 0.496


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--source-snapshot", type=Path, required=True)
    parser.add_argument("--teacher-snapshot", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--parent-result", type=Path, required=True)
    parser.add_argument("--script-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute-official-positioning", action="store_true", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if (
        args.output.exists()
        or file_sha256(Path(__file__)) != args.script_sha256
        or file_sha256(args.source_snapshot) != SOURCE_SHA256
        or file_sha256(args.teacher_snapshot) != TEACHER_SHA256
        or file_sha256(args.checkpoint) != CHECKPOINT_SHA256
        or file_sha256(args.parent_result) != PARENT_RESULT_SHA256
    ):
        raise ValueError("official positioning authority differs")
    configure_deterministic_similarity_runtime(0, cpu_threads=2)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    started = time.monotonic()
    pair = load_paired_archives(
        args.source_snapshot, SOURCE_SHA256, args.teacher_snapshot, TEACHER_SHA256
    )
    train = F.normalize(pair["teacher_train"].float(), dim=1).contiguous()
    test = F.normalize(pair["teacher_test"].float(), dim=1).contiguous()
    labels = pair["test_labels"]
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if (
        type(state) is not dict
        or set(state) != {"weight", "bias"}
        or state["weight"].shape != (256, 768)
        or state["bias"].shape != (256,)
    ):
        raise ValueError("learned direct256 checkpoint schema differs")
    with torch.inference_mode():
        train_codes = F.normalize(
            F.linear(train, state["weight"].float(), state["bias"].float()), dim=1
        ).contiguous()
        test_codes = F.normalize(
            F.linear(test, state["weight"].float(), state["bias"].float()), dim=1
        ).contiguous()
    int4_codes, packing = fit_pack_decode_int4(train_codes, test_codes)
    candidate_width = max(Counter(labels).values()) - 1
    float_raw = score_symmetric(
        test_codes,
        labels,
        candidate_width=candidate_width,
        device=torch.device("cuda"),
    )
    int4_raw = score_symmetric(
        int4_codes,
        labels,
        candidate_width=candidate_width,
        device=torch.device("cuda"),
    )
    float_score = {"map_at_r": float(float_raw["map_at_r"]), "r1": float(float_raw["r1"])}
    int4_score = {"map_at_r": float(int4_raw["map_at_r"]), "r1": float(int4_raw["r1"])}
    result = {
        "schema": "scratch-sop-learned-direct256-int4-official-positioning-v1",
        "claim_eligible": False,
        "dataset": "stanford-online-products-official-test-already-observed",
        "split_status": "already-observed-adaptive-development-surface",
        "rows": len(labels),
        "inputs": {
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "direct128_checkpoint_sha256": DIRECT128_SHA256,
            "parent_result_sha256": PARENT_RESULT_SHA256,
            "source_sha256": SOURCE_SHA256,
            "teacher_sha256": TEACHER_SHA256,
        },
        "direct128_reference": {
            "float_map_at_r": DIRECT128_FLOAT_MAP,
            "packed_map_at_r": DIRECT128_INT8_MAP,
        },
        "direct256_float": float_score,
        "direct256_int4": {**int4_score, **packing},
        "observed": {
            "float_gain_over_direct128": float_score["map_at_r"] - DIRECT128_FLOAT_MAP,
            "packed_gain_over_direct128": int4_score["map_at_r"] - DIRECT128_INT8_MAP,
            "quantization_map_loss": float_score["map_at_r"] - int4_score["map_at_r"],
            "target_margin": int4_score["map_at_r"] - TARGET_MAP,
        },
        "target_map_at_r": TARGET_MAP,
        "passed": int4_score["map_at_r"] >= TARGET_MAP,
        "elapsed_seconds": time.monotonic() - started,
        "script_sha256": args.script_sha256,
    }
    wire = json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    partial = args.output.with_suffix(args.output.suffix + ".partial")
    partial.write_text(wire)
    os.replace(partial, args.output)
    print(wire, end="")


if __name__ == "__main__":
    main()
