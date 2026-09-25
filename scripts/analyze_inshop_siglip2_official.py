#!/usr/bin/env python3
"""Authenticate and summarize the frozen three-seed official In-Shop readout."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import fmean

SEEDS = (179023, 179024, 179025)
DECISION_SHA = "43036079d17aa6a7010fc70c15716c5b939dd8e82629eb1187763395a386526f"
EVALUATOR_SHA = "6fee591bffdb2af8d77355bd4f0c282ac52979fa2125ff46bfb31df0ea985411"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--receipts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or sha256(args.decision) != DECISION_SHA:
        raise ValueError("official In-Shop decision authority differs")
    decision = json.loads(args.decision.read_text())
    if decision.get("selected_arm") != "bank" or decision.get("seeds") != list(SEEDS):
        raise ValueError("official In-Shop selected arm differs")
    rows = {}
    for seed in SEEDS:
        path = args.receipts / f"inshop-official-bank-{seed}-v1.json"
        r = json.loads(path.read_text())
        quality = r.get("packed_quality", {})
        if (
            r.get("schema") != "sfora-inshop-siglip2-official-query-gallery-v1"
            or r.get("claim_eligible") is not False
            or r.get("seed") != seed
            or r.get("arm") != "bank"
            or r.get("queries") != 14_218
            or r.get("gallery_images") != 12_612
            or r.get("gallery_wire_bytes_per_row") != 130
            or r.get("decision_sha256") != DECISION_SHA
            or r.get("source_sha256") != EVALUATOR_SHA
            or r.get("native_top10_exact") is not True
            or r.get("training_receipt_sha256")
            != decision["arms"][str(seed)]["bank"]["receipt_sha256"]
            or len(quality.get("per_query_r1", ())) != 14_218
            or len(quality.get("per_query_ap", ())) != 14_218
        ):
            raise ValueError(f"official In-Shop seed {seed} authority differs")
        for metric, key in (("recall_at_1", "per_query_r1"), ("map_at_r", "per_query_ap")):
            values = quality[key]
            if (
                not all(
                    isinstance(v, (int, float)) and math.isfinite(v) and 0 <= v <= 1 for v in values
                )
                or abs(fmean(values) - quality[metric]) > 1e-8
            ):
                raise ValueError(f"official In-Shop seed {seed} {metric} differs")
        rows[str(seed)] = {
            "receipt_sha256": sha256(path),
            "packed_r1": quality["recall_at_1"],
            "packed_map_at_r": quality["map_at_r"],
            "float_r1": r["float_quality"]["recall_at_1"],
            "export_seconds": r["export_seconds"],
            "native_top10_max_score_abs_delta": r["native_top10_max_score_abs_delta"],
        }
    report = {
        "schema": "sfora-inshop-siglip2-official-selected-bank-v1",
        "claim_eligible": False,
        "decision_sha256": DECISION_SHA,
        "seeds": SEEDS,
        "rows": rows,
        "mean_packed_r1": fmean(row["packed_r1"] for row in rows.values()),
        "mean_packed_map_at_r": fmean(row["packed_map_at_r"] for row in rows.values()),
        "source_sha256": sha256(Path(__file__)),
    }
    with args.output.open("xb") as stream:
        stream.write((json.dumps(report, sort_keys=True, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"mean_packed_r1": report["mean_packed_r1"]}), flush=True)


if __name__ == "__main__":
    main()
