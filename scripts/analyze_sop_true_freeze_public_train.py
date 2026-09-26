#!/usr/bin/env python3
"""Apply the frozen paired product gate to public SOP TRAIN receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from compare_sop_siglip2_member_bank_arms import product_bootstrap


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--seed", type=int, choices=(179024, 179026, 179027), required=True)
    parser.add_argument("--precision", choices=("fp32", "fp16"), required=True)
    parser.add_argument("--held-labels", type=Path, required=True)
    parser.add_argument("--receipt-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("SOP public TRAIN decision already exists")
    labels = np.load(args.held_labels, allow_pickle=False)
    if labels.shape != (5_851,) or labels.dtype != np.int64:
        raise ValueError("SOP public TRAIN held labels differ")
    rows: dict[str, dict[str, Any]] = {}
    paths: dict[str, Path] = {}
    for arm in ("control", "freeze"):
        path = (
            args.receipt_dir
            / f"sfora-sop-true-freeze-public-train-{args.seed}-{arm}-{args.precision}-v1.json"
        )
        receipt = json.loads(path.read_text())
        if (
            receipt.get("schema") != "sfora-sop-true-freeze-public-train-v1"
            or receipt.get("native_top10_exact") is not True
            or receipt.get("queries") != len(labels)
            or receipt.get("gallery_images") != 59_551
            or receipt.get("precision")
            != ("fp32_autocast" if args.precision == "fp32" else "fp16_native")
            or receipt.get("seed") != args.seed
            or receipt.get("arm") != arm
            or receipt.get("query_image_manifest_sha256")
            != rows.get("control", receipt).get("query_image_manifest_sha256")
        ):
            raise ValueError(f"SOP public TRAIN {arm} receipt differs")
        for metric, key in (("packed_r1", "per_query_r1"), ("packed_map_at_r", "per_query_ap")):
            values = np.asarray(receipt[key], dtype=np.float64)
            if (
                values.shape != labels.shape
                or not np.isfinite(values).all()
                or abs(float(values.mean()) - receipt[metric]) > 1e-12
            ):
                raise ValueError(f"SOP public TRAIN {arm} {metric} differs")
        rows[arm], paths[arm] = receipt, path
    intervals = {
        metric: product_bootstrap(
            np.asarray(rows["freeze"][key]) - np.asarray(rows["control"][key]), labels
        )
        for metric, key in (("r1", "per_query_r1"), ("map_at_r", "per_query_ap"))
    }
    passed = (
        intervals["r1"]["point"] >= 0
        and intervals["r1"]["lower_95"] > -0.005
        and intervals["map_at_r"]["point"] >= 0
    )
    result = {
        "schema": "sfora-sop-true-freeze-public-train-paired-v1",
        "claim_eligible": False,
        "seed": args.seed,
        "precision": rows["control"]["precision"],
        "quality_pass": bool(passed),
        "control_r1": rows["control"]["packed_r1"],
        "freeze_r1": rows["freeze"]["packed_r1"],
        "control_map_at_r": rows["control"]["packed_map_at_r"],
        "freeze_map_at_r": rows["freeze"]["packed_map_at_r"],
        "paired_product_bootstrap": intervals,
        "source_receipt_sha256": {arm: sha256(path) for arm, path in paths.items()},
        "held_labels_sha256": sha256(args.held_labels),
        "source_sha256": sha256(Path(__file__)),
    }
    args.output.write_text(json.dumps(result, sort_keys=True, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "seed": args.seed,
                "precision": args.precision,
                "quality_pass": passed,
                "r1_delta": intervals["r1"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
