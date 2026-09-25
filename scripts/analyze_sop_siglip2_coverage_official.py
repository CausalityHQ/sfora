#!/usr/bin/env python3
"""Verify the three coverage-bank SOP TEST receipts and compare the old bank."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from compare_sop_siglip2_member_bank_arms import product_bootstrap

DECISION_SHA256 = "ccfdb7055643f2a16124ecab62b30d31a2b025f380319e950cb4a481066f1852"
OLD_REPORT_SHA256 = "dd7bc98099c67e86cc2d8a074dc4ceed19b65cb162687908335beb1d54d6bd97"
EVALUATOR_SHA256 = "2324a3e4a13ba1758f25882d28cb4bec430a402a2a630eebac6aedeb09f62b1e"
ARCHIVE_SHA256 = "1ba27b2d6b9db39067aa6facd0ef8aafc303c4527f6feabed859b0512c7d921a"
MANIFEST_SHA256 = "28a3ec0561cd83ee426f3d1c301c70799316af91c1e9083a5a1ffdf3414327c1"
SCORER_SHA256 = "5d62354698ad84ff4a419e86e6dfe04d8f1eb27f9adf11a89a1c8e13c27aff63"
SEEDS = (179023, 179024, 179025)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--source-archive", required=True, type=Path)
    parser.add_argument("--decision", required=True, type=Path)
    parser.add_argument("--old-report", required=True, type=Path)
    parser.add_argument("--run-base", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if (
        args.output.exists()
        or sha256(args.source_archive) != ARCHIVE_SHA256
        or sha256(args.decision) != DECISION_SHA256
        or sha256(args.old_report) != OLD_REPORT_SHA256
    ):
        raise ValueError("coverage official source authority differs")
    decision = json.loads(args.decision.read_text())
    old_report = json.loads(args.old_report.read_text())
    if (
        decision.get("schema") != "sfora-sop-siglip2-bf16-coverage-replication-v1"
        or decision.get("replication_gate_pass") is not True
        or decision.get("seeds") != list(SEEDS)
    ):
        raise ValueError("coverage official training decision differs")
    if (
        old_report.get("schema") != "sfora-sop-siglip2-bf16-official-paired-report-v1"
        or old_report.get("test_images") != 60_502
        or old_report.get("seeds") != list(SEEDS)
    ):
        raise ValueError("old official report differs")
    with np.load(args.source_archive, allow_pickle=False) as archive:
        labels = np.asarray(archive["test_labels"], dtype=np.int64)
        ids = np.asarray(archive["test_image_ids"], dtype=np.int64)
    if labels.shape != (60_502,) or ids.shape != labels.shape or len(np.unique(labels)) != 11_316:
        raise ValueError("coverage official TEST inventory differs")
    ids_sha = hashlib.sha256(ids.tobytes()).hexdigest()
    rows = {}
    deltas = []
    for seed in SEEDS:
        training = decision["arms"][str(seed)]["bank"]
        source = args.run_base / f"sfora-siglip2-bf16-coverage-{seed}-bank-v1/receipt.json"
        new_path = (
            args.run_base / f"sfora-siglip2-bf16-coverage-official-{seed}-bank-v1/receipt.json"
        )
        old_path = args.run_base / f"sfora-siglip2-bf16-official-{seed}-bank-v1/receipt.json"
        if sha256(source) != training["receipt_sha256"]:
            raise ValueError(f"seed {seed} training receipt differs")
        if sha256(old_path) != old_report["arms"][str(seed)]["bank"]["official_receipt_sha256"]:
            raise ValueError(f"seed {seed} old official comparator differs")
        new, old = (json.loads(path.read_text()) for path in (new_path, old_path))
        source_files = new.get("evaluator_source_files_sha256", {})
        if (
            new.get("schema") != "sfora-sop-siglip2-official-test-v1"
            or new.get("claim_eligible") is not False
            or new.get("seed") != seed
            or new.get("arm") != "bank"
            or new.get("queries") != len(labels)
            or new.get("products") != 11_316
            or new.get("test_image_ids_sha256") != ids_sha
            or new.get("source_archive_sha256") != ARCHIVE_SHA256
            or new.get("test_image_manifest_sha256") != MANIFEST_SHA256
            or new.get("decision_sha256") != DECISION_SHA256
            or new.get("training_receipt_sha256") != training["receipt_sha256"]
            or new.get("training_checkpoint_sha256")
            != json.loads(source.read_text())["checkpoint_sha256"]
            or new.get("native_top10_exact") is not True
            or new.get("native_per_query_r1_equal") is not True
            or new.get("gallery_wire_bytes_per_row") != 130
            or new.get("source_sha256") != EVALUATOR_SHA256
            or source_files.get("scripts/evaluate_sop_siglip2_official.py") != EVALUATOR_SHA256
            or source_files.get("src/sfora/sop_evaluation.py") != SCORER_SHA256
            or old.get("schema") != "sfora-sop-siglip2-official-test-v1"
            or old.get("seed") != seed
            or old.get("arm") != "bank"
            or old.get("test_image_ids_sha256") != ids_sha
            or old.get("native_top10_exact") is not True
        ):
            raise ValueError(f"seed {seed} official receipt authority differs")
        metrics = {}
        for name, key in (("r1", "per_query_r1"), ("map_at_r", "per_query_ap")):
            fresh = np.asarray(new["packed_quality"][key], dtype=np.float64)
            previous = np.asarray(old["packed_quality"][key], dtype=np.float64)
            stated = "recall_at_1" if name == "r1" else "map_at_r"
            if (
                any(
                    v.shape != labels.shape or not np.isfinite(v).all() or np.any((v < 0) | (v > 1))
                    for v in (fresh, previous)
                )
                or abs(fresh.mean() - new["packed_quality"][stated]) > 1e-6
                or abs(previous.mean() - old["packed_quality"][stated]) > 1e-6
            ):
                raise ValueError(f"seed {seed} {name} quality differs")
            metrics[name] = {"new": float(fresh.mean()), "old": float(previous.mean())}
            if name == "r1":
                deltas.append(fresh - previous)
        rows[str(seed)] = {
            "new_receipt_sha256": sha256(new_path),
            "old_receipt_sha256": sha256(old_path),
            "quality": metrics,
            "training_wall_seconds": training["training_wall_seconds"],
            "official_export_seconds": new["export_seconds"],
            "official_total_wall_seconds": new["total_wall_seconds"],
            "peak_cuda_allocated_bytes": new["peak_cuda_allocated_bytes"],
        }
    delta = np.mean(deltas, axis=0)
    result = {
        "schema": "sfora-sop-siglip2-coverage-official-report-v2",
        "claim_eligible": False,
        "split": "already-observed SOP official TEST, symmetric full gallery, self excluded",
        "source_sha256": sha256(Path(__file__)),
        "decision_sha256": DECISION_SHA256,
        "old_report_sha256": OLD_REPORT_SHA256,
        "seeds": SEEDS,
        "rows": rows,
        "mean_packed_r1": float(
            np.mean([rows[str(seed)]["quality"]["r1"]["new"] for seed in SEEDS])
        ),
        "mean_packed_map_at_r": float(
            np.mean([rows[str(seed)]["quality"]["map_at_r"]["new"] for seed in SEEDS])
        ),
        "old_mean_packed_r1": float(
            np.mean([rows[str(seed)]["quality"]["r1"]["old"] for seed in SEEDS])
        ),
        "paired_seed_mean_r1_delta": float(delta.mean()),
        "paired_product_bootstrap_r1_delta": product_bootstrap(delta, labels),
        "interval_scope": "product bootstrap conditional on trained seeds and reused official TEST",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "mean_packed_r1",
                    "mean_packed_map_at_r",
                    "old_mean_packed_r1",
                    "paired_product_bootstrap_r1_delta",
                )
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
