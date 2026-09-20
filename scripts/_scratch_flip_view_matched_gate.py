#!/usr/bin/env python3
"""Matched-process fit-only gate for horizontal-flip gallery aggregation."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from _scratch_flip_view_information_gate import (
    clustered_lower_bound,
    encode_views,
    food_fit,
    load_model,
    pet_fit,
    sha256,
    top1_correct,
)
from torch.nn import functional as F

from sfora.deterministic_similarity_runtime import configure_deterministic_similarity_runtime


def _validate_matched_features(
    identity: np.ndarray,
    flipped: np.ndarray,
    *,
    expected_rows: int,
) -> dict[str, float | int]:
    if (
        identity.ndim != 2
        or flipped.ndim != 2
        or identity.shape != flipped.shape
        or identity.shape[0] != expected_rows
    ):
        raise ValueError("matched feature shape differs")
    if not np.isfinite(identity).all() or not np.isfinite(flipped).all():
        raise ValueError("matched features must be finite")
    identity_error = float(np.max(np.abs(np.linalg.norm(identity, axis=1) - 1.0)))
    flipped_error = float(np.max(np.abs(np.linalg.norm(flipped, axis=1) - 1.0)))
    if identity_error > 2e-5 or flipped_error > 2e-5:
        raise ValueError("matched features must have unit norm")
    return {
        "dimensions": int(identity.shape[1]),
        "identity_max_unit_norm_error": identity_error,
        "flipped_max_unit_norm_error": flipped_error,
    }


def _decision(
    datasets: dict[str, dict[str, object]],
    *,
    bootstrap_lower_95: float,
) -> dict[str, object]:
    total_rows = sum(int(row["rows"]) for row in datasets.values())
    if total_rows <= 0:
        raise ValueError("flip-view decision requires rows")
    pooled_delta = math.fsum(
        int(row["rows"]) * float(row["recall_at_1_delta"])
        for row in datasets.values()
    ) / total_rows
    passed = (
        max(float(row["mean_view_cosine"]) for row in datasets.values()) < 0.98
        and pooled_delta >= 0.003
        and bootstrap_lower_95 > 0.0
    )
    return {
        "passed": passed,
        "pooled_recall_at_1_delta": pooled_delta,
        "class_clustered_bootstrap_lower_95": bootstrap_lower_95,
        "next": (
            "expand exact view to the frozen panel"
            if passed
            else "close horizontal-flip view aggregation"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--preregistration-sha256", required=True)
    parser.add_argument("--script-sha256", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--unicom-checkout", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--food-fit", type=Path, required=True)
    parser.add_argument("--pet-fit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute-fit-only-gate", action="store_true", required=True)
    args = parser.parse_args()

    preregistration = json.loads(args.preregistration.read_text())
    if (
        args.output.exists()
        or sha256(Path(__file__)) != args.script_sha256
        or sha256(args.preregistration) != args.preregistration_sha256
        or preregistration["schema"] != "sfora-flip-view-matched-preregistration-v2"
        or preregistration["script_sha256"] != args.script_sha256
        or preregistration["source_commit"] != args.source_commit
        or sha256(args.checkpoint) != preregistration["checkpoint_sha256"]
        or sha256(args.food_fit) != preregistration["fit_archives"]["food101"]
        or sha256(args.pet_fit) != preregistration["fit_archives"]["pet"]
    ):
        raise ValueError("matched flip-view gate authority differs")

    configure_deterministic_similarity_runtime(17, cpu_threads=2)
    started = time.monotonic()
    model, transform = load_model(args.unicom_checkout, args.checkpoint)
    inputs = {
        "food101": food_fit(args.raw_root, args.food_fit),
        "pet": pet_fit(args.raw_root, args.pet_fit),
    }
    internal: dict[str, dict[str, object]] = {}
    reported: dict[str, dict[str, object]] = {}
    for name, (paths, labels_array, authenticated_reference) in inputs.items():
        identity_tensor, flipped_tensor = encode_views(
            model,
            transform,
            paths,
            batch_size=64,
        )
        identity = np.ascontiguousarray(identity_tensor.numpy(), dtype=np.float32)
        flipped = np.ascontiguousarray(flipped_tensor.numpy(), dtype=np.float32)
        validation = _validate_matched_features(identity, flipped, expected_rows=len(paths))
        if authenticated_reference.shape != identity.shape:
            raise ValueError(f"{name} authenticated feature shape differs")

        identity_values = torch.from_numpy(identity)
        flipped_values = torch.from_numpy(flipped)
        averaged = F.normalize(identity_values + flipped_values, dim=1)
        labels = torch.from_numpy(labels_array)
        identity_correct = top1_correct(identity_values, labels)
        average_correct = top1_correct(averaged, labels)
        internal[name] = {
            "labels": labels_array,
            "identity_correct": identity_correct,
            "average_correct": average_correct,
        }
        reported[name] = {
            "rows": len(paths),
            "classes": len(np.unique(labels_array)),
            "mean_view_cosine": float(np.mean(np.sum(identity * flipped, axis=1))),
            "identity_recall_at_1": float(identity_correct.mean()),
            "averaged_recall_at_1": float(average_correct.mean()),
            "recall_at_1_delta": float(average_correct.mean() - identity_correct.mean()),
            "authenticated_reference_rows": int(authenticated_reference.shape[0]),
            "comparison_authority": "same-process matched extraction",
            **validation,
        }

    lower = clustered_lower_bound(internal)
    decision = _decision(reported, bootstrap_lower_95=lower)
    result = {
        "schema": "sfora-flip-view-matched-gate-v2",
        "claim_eligible": False,
        "source_commit": args.source_commit,
        "script_sha256": args.script_sha256,
        "preregistration_sha256": args.preregistration_sha256,
        "datasets": reported,
        "decision": decision,
        "elapsed_seconds": time.monotonic() - started,
    }
    payload = json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    args.output.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
