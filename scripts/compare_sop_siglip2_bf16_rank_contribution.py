#!/usr/bin/env python3
"""Choose a fit-only BF16 float-rank coefficient from first-step gradients."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
from pathlib import Path

from compare_sop_siglip2_bf16_member_bank_multiseed import receipt_paths

SEEDS = (179023, 179024, 179025)
TRAINER_SHA256 = "ad66b1613f0c1c8373d69a689d1556c9b250527a8045a6b85bec523f99466d23"
COMMON = (
    "seed",
    "source_sha256",
    "source_files_sha256",
    "train_vision_dtype",
    "initial_head_sha256",
    "initial_classifier_sha256",
    "model_file_sha256",
    "source_archive_sha256",
    "query_image_ids_sha256",
    "batch_size",
    "rank_coefficient",
)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_diagnostic_pair(
    seed: int, floating: dict, bank: dict, full_float: dict, full_bank: dict
) -> tuple[float, float]:
    arms = (floating, bank, full_float, full_bank)
    if (
        seed not in SEEDS
        or tuple(row.get("arm") for row in arms)
        != ("float_rank", "float_rank_member_bank", "float_rank", "float_rank_member_bank")
        or any(row.get("seed") != seed for row in arms)
        or any(row.get("source_sha256") != TRAINER_SHA256 for row in arms)
        or any(
            row.get("source_files_sha256", {}).get("scripts/train_sop_siglip2_compact.py")
            != TRAINER_SHA256
            for row in arms
        )
        or any(row.get("train_vision_dtype") != "bf16" for row in arms)
        or any(row.get("updates") != 1 or row.get("quality") is not None for row in arms[:2])
        or any(row.get("updates") != 1000 for row in arms[2:])
        or any(row.get(key) != floating.get(key) for row in arms[1:] for key in COMMON)
        or any(len(row.get("first_input_batch_sha256", [])) < 1 for row in arms)
        or any(
            row["first_input_batch_sha256"][0] != floating["first_input_batch_sha256"][0]
            for row in arms[1:]
        )
        or any(len(row.get("rank_to_arcface_head_gradient_ratio", [])) != 1 for row in arms[:2])
    ):
        raise ValueError("SOP BF16 rank diagnostic authority differs")
    ratios = tuple(float(row["rank_to_arcface_head_gradient_ratio"][0]) for row in arms[:2])
    if any(not math.isfinite(value) or value <= 0 for value in ratios):
        raise ValueError("SOP BF16 rank diagnostic gradient differs")
    return ratios


def coefficient_from_ratios(pairs: list[tuple[float, float]]) -> float:
    if len(pairs) != 3 or any(
        not math.isfinite(floating) or not math.isfinite(bank) or floating <= 0 or bank <= 0
        for floating, bank in pairs
    ):
        raise ValueError("SOP BF16 rank coefficient ratios differ")
    coefficient = float(
        f"{8.0 * statistics.median(bank / floating for floating, bank in pairs):.4g}"
    )
    if not math.isfinite(coefficient) or not 0.5 <= coefficient <= 256:
        raise ValueError("SOP BF16 rank coefficient is outside the frozen range")
    return coefficient


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--expected-gate-sha256", required=True)
    parser.add_argument("--run-base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.output.exists()
        or args.output.is_symlink()
        or sha256(args.gate) != args.expected_gate_sha256
    ):
        raise ValueError("SOP BF16 rank diagnostic gate or output differs")
    gate = json.loads(args.gate.read_text())
    if (
        gate.get("schema") != "sfora-sop-siglip2-bf16-member-bank-multiseed-v1"
        or gate.get("continuation_gate_pass") is not True
        or gate.get("seeds") != list(SEEDS)
    ):
        raise ValueError("SOP BF16 rank diagnostic needs a passing three-seed gate")
    full_paths = receipt_paths(args.run_base)
    results = {}
    pairs = []
    for seed in SEEDS:
        diagnostic_paths = {
            arm: args.run_base
            / f"sfora-siglip2-bf16-rank-contribution-{seed}-{arm}-v1/receipt.json"
            for arm in ("float_rank", "bank")
        }
        rows = {arm: json.loads(path.read_text()) for arm, path in diagnostic_paths.items()}
        full = {
            arm: json.loads(full_paths[seed][arm].read_text()) for arm in ("float_rank", "bank")
        }
        floating, bank = validate_diagnostic_pair(
            seed, rows["float_rank"], rows["bank"], full["float_rank"], full["bank"]
        )
        pairs.append((floating, bank))
        results[str(seed)] = {
            "float_ratio": floating,
            "bank_ratio": bank,
            "bank_over_float": bank / floating,
            "float_receipt_sha256": sha256(diagnostic_paths["float_rank"]),
            "bank_receipt_sha256": sha256(diagnostic_paths["bank"]),
            "full_float_receipt_sha256": sha256(full_paths[seed]["float_rank"]),
            "full_bank_receipt_sha256": sha256(full_paths[seed]["bank"]),
        }
    output = {
        "schema": "sfora-sop-siglip2-bf16-rank-contribution-v1",
        "split": "SOP official TRAIN fit products; first augmented batch only",
        "source_sha256": sha256(Path(__file__)),
        "gate_sha256": sha256(args.gate),
        "trainer_sha256": TRAINER_SHA256,
        "rank_coefficient_bank": 8.0,
        "rank_coefficient_float_control": coefficient_from_ratios(pairs),
        "first_step_ratios": results,
        "scope": "initial 128-feature gradient norm diagnostic, not whole-backbone equivalence",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write((json.dumps(output, sort_keys=True, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(output, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
