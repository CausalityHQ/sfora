#!/usr/bin/env python3
"""Source-bound report for the frozen nine-arm SOP official TEST panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
from compare_sop_siglip2_member_bank_arms import product_bootstrap
from evaluate_sop_siglip2_official import BF16_SEEDS, validate_decision_arm

DECISION_SHA256 = "6646adc6a0c72838f398268876d41d317e94c1c0c1bde47b09eb424a251ac097"
ARCHIVE_SHA256 = "1ba27b2d6b9db39067aa6facd0ef8aafc303c4527f6feabed859b0512c7d921a"
MANIFEST_SHA256 = "28a3ec0561cd83ee426f3d1c301c70799316af91c1e9083a5a1ffdf3414327c1"
EVALUATOR_SHA256 = "5469fe11fb50b96f5088c94979039052dae1bec6334ca87b6cc7319f24b83578"
SCORER_SHA256 = "5d62354698ad84ff4a419e86e6dfe04d8f1eb27f9adf11a89a1c8e13c27aff63"
ARMS = ("bank", "original_float", "matched_float")
METRICS = {
    "r1": ("per_query_r1", "recall_at_1"),
    "r10": ("per_query_r10", "recall_at_10"),
    "r100": ("per_query_r100", "recall_at_100"),
    "r1000": ("per_query_r1000", "recall_at_1000"),
    "map_at_r": ("per_query_ap", "map_at_r"),
}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def training_dir(base: Path, seed: int, arm: str) -> Path:
    if arm == "matched_float":
        return base / f"sfora-siglip2-bf16-rankmatched-{seed}-float-v1"
    suffix = "bank" if arm == "bank" else "float_rank"
    return base / f"sfora-siglip2-bf16-member-bank-{seed}-{suffix}-v1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--source-cache-receipt", type=Path, required=True)
    parser.add_argument("--run-base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.output.exists()
        or args.output.is_symlink()
        or sha256(args.source_archive) != ARCHIVE_SHA256
        or sha256(args.decision) != DECISION_SHA256
    ):
        raise ValueError("SOP official report input authority differs")
    decision = json.loads(args.decision.read_text())
    cache = json.loads(args.source_cache_receipt.read_text())
    cache_sha = sha256(args.source_cache_receipt)
    cache_seconds = cache.get("encode_wall_seconds")
    if (
        cache.get("schema") != "sfora-sop-siglip2-train-feature-export-v1"
        or cache.get("rows") != 59_551
        or cache_sha != "3d49e039c72c0677591835752133caf1cbfd0ea483c23a901773748b44d843fb"
        or not isinstance(cache_seconds, float)
        or not math.isfinite(cache_seconds)
        or cache_seconds <= 0
    ):
        raise ValueError("SOP official source-cache cost authority differs")
    with np.load(args.source_archive, allow_pickle=False) as archive:
        labels = np.asarray(archive["test_labels"], dtype=np.int64)
        ids = np.asarray(archive["test_image_ids"], dtype=np.int64)
    if labels.shape != ids.shape or labels.shape != (60_502,) or len(np.unique(labels)) != 11_316:
        raise ValueError("SOP official TEST inventory differs")
    test_ids_sha = hashlib.sha256(ids.tobytes()).hexdigest()
    rows: dict[str, dict] = {}
    per_query: dict[str, dict[str, list[np.ndarray]]] = {
        arm: {metric: [] for metric in METRICS} for arm in ARMS
    }
    evaluator_sources: dict | None = None
    for seed in BF16_SEEDS:
        rows[str(seed)] = {}
        for arm in ARMS:
            train_path = training_dir(args.run_base, seed, arm) / "receipt.json"
            train_sha = sha256(train_path)
            train = json.loads(train_path.read_text())
            validate_decision_arm(decision, seed, arm, train_sha, train)
            official_path = (
                args.run_base / f"sfora-siglip2-bf16-official-{seed}-{arm}-v1/receipt.json"
            )
            official_sha = sha256(official_path)
            official = json.loads(official_path.read_text())
            source_files = official.get("evaluator_source_files_sha256")
            if (
                official.get("schema") != "sfora-sop-siglip2-official-test-v1"
                or official.get("claim_eligible") is not False
                or official.get("seed") != seed
                or official.get("arm") != arm
                or official.get("queries") != len(labels)
                or official.get("products") != 11_316
                or official.get("source_archive_sha256") != ARCHIVE_SHA256
                or official.get("test_image_manifest_sha256") != MANIFEST_SHA256
                or official.get("test_image_ids_sha256") != test_ids_sha
                or official.get("decision_sha256") != DECISION_SHA256
                or official.get("training_receipt_sha256") != train_sha
                or official.get("training_checkpoint_sha256") != train.get("checkpoint_sha256")
                or official.get("native_library_sha256") != train.get("native_library_sha256")
                or official.get("gallery_wire_bytes_per_row") != 130
                or official.get("native_top10_exact") is not True
                or official.get("native_per_query_r1_equal") is not True
                or not isinstance(source_files, dict)
                or official.get("source_sha256") != EVALUATOR_SHA256
                or source_files.get("scripts/evaluate_sop_siglip2_official.py") != EVALUATOR_SHA256
                or source_files.get("src/sfora/sop_evaluation.py") != SCORER_SHA256
                or (evaluator_sources is not None and source_files != evaluator_sources)
                or train.get("source_export_receipt_sha256") != cache_sha
            ):
                raise ValueError("SOP official evaluation receipt authority differs")
            evaluator_sources = source_files
            quality = official["packed_quality"]
            for metric, (per_key, mean_key) in METRICS.items():
                values = np.asarray(quality[per_key], dtype=np.float64)
                stated = quality[mean_key]
                if (
                    values.shape != (len(labels),)
                    or not np.isfinite(values).all()
                    or np.any((values < 0) | (values > 1))
                    or abs(float(values.mean()) - stated) > 1e-6
                    or (metric != "map_at_r" and not np.isin(values, (0, 1)).all())
                ):
                    raise ValueError("SOP official per-query quality differs")
                per_query[arm][metric].append(values)
            bank_wall = train.get("training_wall_including_member_bank_init_seconds")
            train_seconds = bank_wall if arm == "bank" else train.get("training_wall_seconds")
            phases = {
                key: official.get(key)
                for key in (
                    "export_seconds",
                    "packing_seconds",
                    "float_score_seconds",
                    "packed_score_seconds",
                    "native_verify_seconds",
                    "score_seconds",
                    "total_wall_seconds",
                )
            }
            if not all(
                isinstance(value, (float, int)) and math.isfinite(value) and value > 0
                for value in (train_seconds, *phases.values())
            ):
                raise ValueError("SOP official phase cost differs")
            rows[str(seed)][arm] = {
                "official_receipt_sha256": official_sha,
                "training_receipt_sha256": train_sha,
                "packed_quality": {name: quality[mean] for name, (_per, mean) in METRICS.items()},
                "source_cache_seconds": cache_seconds,
                "training_seconds": train_seconds,
                "cache_plus_training_seconds": cache_seconds + train_seconds,
                "training_peak_cuda_allocated_bytes": train["training_peak_cuda_allocated_bytes"],
                "official_peak_cuda_allocated_bytes": official["peak_cuda_allocated_bytes"],
                "gallery_bytes": 130 * len(labels),
                **phases,
            }
    comparisons = {
        f"bank_minus_{control}": {
            metric: {
                "seedwise": [
                    float((bank - other).mean())
                    for bank, other in zip(
                        per_query["bank"][metric], per_query[control][metric], strict=True
                    )
                ],
                "paired_product_bootstrap": product_bootstrap(
                    np.mean(np.stack(per_query["bank"][metric]), axis=0)
                    - np.mean(np.stack(per_query[control][metric]), axis=0),
                    labels,
                ),
            }
            for metric in METRICS
        }
        for control in ("original_float", "matched_float")
    }
    result = {
        "schema": "sfora-sop-siglip2-bf16-official-paired-report-v1",
        "claim_eligible": False,
        "status": "exploratory official SOP TEST; no SOTA claim",
        "split": "SOP official TEST; symmetric full TEST gallery; self excluded",
        "source_sha256": sha256(Path(__file__)),
        "source_archive_sha256": ARCHIVE_SHA256,
        "decision_sha256": DECISION_SHA256,
        "source_cache_receipt_sha256": cache_sha,
        "evaluator_source_files_sha256": evaluator_sources,
        "test_images": len(labels),
        "test_products": 11_316,
        "seeds": BF16_SEEDS,
        "arms": rows,
        "comparisons": comparisons,
        "interval_scope": (
            "product bootstrap conditional on selected method, official protocol reuse, "
            "and three seeds"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write((json.dumps(result, sort_keys=True, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"comparisons": comparisons}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
