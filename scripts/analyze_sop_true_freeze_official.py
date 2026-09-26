#!/usr/bin/env python3
"""Apply the frozen SOP official TEST gate to one existing control/freeze pair."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
from pathlib import Path

import numpy as np
from compare_sop_siglip2_member_bank_arms import product_bootstrap

ARCHIVE_SHA = "1ba27b2d6b9db39067aa6facd0ef8aafc303c4527f6feabed859b0512c7d921a"
MANIFEST_SHA = "28a3ec0561cd83ee426f3d1c301c70799316af91c1e9083a5a1ffdf3414327c1"
DECISION_SHA = {
    179024: "31ebbe4ae71c5f1c7922ef9140721957ccc408eab165df1570678a8843988ed5",
    179026: "e12af7bc484e140e1bde577ff713c0eee711fa6854ec8babbf0607afa74cf364",
    179027: "20d0e3aeaeb24093cf05cd87feb692c878f28bac85c9abd88f2bba5fde2ca894",
}
SOURCE_FILES = {
    "scripts/evaluate_sop_siglip2_official.py",
    "scripts/train_sop_siglip2_compact.py",
    "src/sfora/cutile_int8.py",
    "src/sfora/joint_relational_compaction.py",
    "src/sfora/sop_compact_training.py",
    "src/sfora/sop_evaluation.py",
}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--seed", type=int, choices=tuple(DECISION_SHA), required=True)
    parser.add_argument("--mode", choices=("offline", "public"), default="offline")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--run-base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.output.exists()
        or sha256(args.source_archive) != ARCHIVE_SHA
        or sha256(args.decision) != DECISION_SHA[args.seed]
        or Path(inspect.getfile(product_bootstrap)).resolve()
        != (args.source_root / "scripts/compare_sop_siglip2_member_bank_arms.py").resolve()
    ):
        raise ValueError("SOP true-freeze official analyzer authority differs")
    decision = json.loads(args.decision.read_text())
    if (
        decision.get("schema") != "sfora-sop-true-freeze-paired-train-only-v1"
        or decision.get("seed") != args.seed
        or decision.get("advance_fresh_seeds") is not True
    ):
        raise ValueError("SOP true-freeze TRAIN decision differs")
    with np.load(args.source_archive, allow_pickle=False) as archive:
        labels = np.asarray(archive["test_labels"], dtype=np.int64)
        ids = np.asarray(archive["test_image_ids"], dtype=np.int64)
    if labels.shape != (60_502,) or ids.shape != labels.shape or len(np.unique(labels)) != 11_316:
        raise ValueError("SOP official TEST inventory differs")
    ids_sha = hashlib.sha256(ids.tobytes()).hexdigest()
    rows = {}
    vectors = {}
    source_files_expected = SOURCE_FILES | (
        {"src/sfora/siglip2_compact_serving.py"} if args.mode == "public" else set()
    )
    run_name = (
        "sfora-sop-true-freeze-public-official"
        if args.mode == "public"
        else "sfora-sop-true-freeze-official"
    )
    for arm in ("control", "freeze"):
        training_path = (
            args.run_base / f"sfora-sop-true-freeze-{arm}-{args.seed}-1000-v1/receipt.json"
        )
        official_path = args.run_base / f"{run_name}-{args.seed}-{arm}-v1/receipt.json"
        if sha256(training_path) != decision["arms"][arm]["receipt_sha256"]:
            raise ValueError(f"SOP seed {args.seed} {arm} training receipt differs")
        training = json.loads(training_path.read_text())
        official = json.loads(official_path.read_text())
        source_files = official.get("evaluator_source_files_sha256")
        if (
            official.get("schema") != "sfora-sop-siglip2-official-test-v1"
            or official.get("claim_eligible") is not False
            or official.get("seed") != args.seed
            or official.get("arm") != arm
            or official.get("queries") != len(labels)
            or official.get("products") != 11_316
            or official.get("test_image_ids_sha256") != ids_sha
            or official.get("source_archive_sha256") != ARCHIVE_SHA
            or official.get("test_image_manifest_sha256") != MANIFEST_SHA
            or official.get("decision_sha256") != DECISION_SHA[args.seed]
            or official.get("training_receipt_sha256") != sha256(training_path)
            or official.get("training_checkpoint_sha256") != training["checkpoint_sha256"]
            or official.get("native_top10_exact") is not True
            or official.get("native_per_query_r1_equal") is not True
            or official.get("gallery_wire_bytes_per_row") != 130
            or not isinstance(source_files, dict)
            or set(source_files) != source_files_expected
            or any(
                sha256(args.source_root / name) != digest for name, digest in source_files.items()
            )
            or official.get("source_sha256")
            != source_files["scripts/evaluate_sop_siglip2_official.py"]
            or (
                args.mode == "public"
                and (
                    official.get("inference_precision") != "fp16_native"
                    or official.get("export_batch_size") != 32
                    or official.get("public_first_batch_packed_exact") is not True
                )
            )
        ):
            raise ValueError(f"SOP seed {args.seed} {arm} official receipt differs")
        quality = official["packed_quality"]
        for metric, key in (("r1", "per_query_r1"), ("map_at_r", "per_query_ap")):
            values = np.asarray(quality[key], dtype=np.float64)
            stated = "recall_at_1" if metric == "r1" else "map_at_r"
            if (
                values.shape != labels.shape
                or not np.isfinite(values).all()
                or np.any((values < 0) | (values > 1))
                or abs(float(values.mean()) - quality[stated]) > 1e-6
            ):
                raise ValueError(f"SOP seed {args.seed} {arm} {metric} vector differs")
            vectors[(arm, metric)] = values
        rows[arm] = {
            "official_receipt_sha256": sha256(official_path),
            "training_receipt_sha256": sha256(training_path),
            "packed_r1": quality["recall_at_1"],
            "packed_map_at_r": quality["map_at_r"],
            "export_seconds": official["export_seconds"],
        }
    intervals = {
        metric: product_bootstrap(
            vectors[("freeze", metric)] - vectors[("control", metric)], labels
        )
        for metric in ("r1", "map_at_r")
    }
    passed = (
        intervals["r1"]["point"] >= 0
        and intervals["r1"]["lower_95"] > -0.005
        and intervals["map_at_r"]["point"] >= 0
    )
    result = {
        "schema": (
            "sfora-sop-true-freeze-public-official-paired-v1"
            if args.mode == "public"
            else "sfora-sop-true-freeze-official-paired-v1"
        ),
        "claim_eligible": False,
        "seed": args.seed,
        "mode": args.mode,
        "split": "already-observed SOP official TEST, symmetric full gallery, self excluded",
        "decision_sha256": DECISION_SHA[args.seed],
        "source_archive_sha256": ARCHIVE_SHA,
        "bootstrap_helper_sha256": sha256(Path(inspect.getfile(product_bootstrap))),
        "source_sha256": sha256(Path(__file__)),
        "arms": rows,
        "paired_product_bootstrap": intervals,
        "quality_pass": bool(passed),
    }
    if not all(
        math.isfinite(value)
        for row in rows.values()
        for value in row.values()
        if isinstance(value, float)
    ):
        raise ValueError("SOP official resource receipt differs")
    with args.output.open("xb") as stream:
        stream.write((json.dumps(result, sort_keys=True, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"seed": args.seed, "quality_pass": passed, "r1_delta": intervals["r1"]}))


if __name__ == "__main__":
    main()
