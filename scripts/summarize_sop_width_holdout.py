"""Authenticate and compare matched SOP compact/full-width train holdout receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import cast

import numpy as np

BOOTSTRAPS = 10_000
SEED = 179019
SAME_FIELDS = (
    "schema",
    "arm",
    "recipe",
    "seed",
    "step",
    "schedule_sha256",
    "fit_row_indexes_sha256",
    "validation_row_indexes_sha256",
    "input_checkpoint_sha256",
    "features_archive_sha256",
    "sop_train_metadata_sha256",
    "validation_image_ids",
    "validation_labels",
)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def score_array(score: dict[str, object], key: str, length: int) -> np.ndarray:
    result = np.asarray(score[key], dtype=np.float64)
    if result.shape != (length,) or not bool(np.isfinite(result).all()):
        raise ValueError("SOP per-query score inventory differs")
    if key == "per_query_r1" and not bool(np.isin(result, (0, 1)).all()):
        raise ValueError("SOP per-query Recall@1 differs")
    if key == "per_query_ap" and not bool(((result >= 0) & (result <= 1)).all()):
        raise ValueError("SOP per-query AP differs")
    metric = "recall_at_1" if key == "per_query_r1" else "map_at_r"
    if abs(float(result.mean()) - cast(float, score[metric])) > 1e-7:
        raise ValueError("SOP aggregate/per-query score differs")
    return result


def cluster_interval(labels: np.ndarray, deltas: np.ndarray) -> list[float]:
    _, inverse = np.unique(labels, return_inverse=True)
    classes = int(inverse.max()) + 1
    counts = np.bincount(inverse, minlength=classes)
    sums = np.bincount(inverse, weights=deltas, minlength=classes)
    rng = np.random.default_rng(SEED)
    samples = np.empty(BOOTSTRAPS, dtype=np.float64)
    for index in range(BOOTSTRAPS):
        chosen = rng.integers(0, classes, size=classes)
        samples[index] = sums[chosen].sum() / counts[chosen].sum()
    return [float(x) for x in np.quantile(samples, (0.025, 0.975))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--compact-receipt", type=Path, required=True)
    parser.add_argument("--fullwidth-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("SOP width comparison destination already exists")
    compact = json.loads(args.compact_receipt.read_text())
    full = json.loads(args.fullwidth_receipt.read_text())
    if (
        any(compact.get(field) != full.get(field) for field in SAME_FIELDS)
        or compact.get("schema") != "sfora-sop-compact-training-diagnostic-v1"
        or compact.get("arm") != "arcface"
        or compact.get("recipe") != "reference"
        or compact.get("seed") != SEED
        or compact.get("embedding_width", 128) != 128
        or full.get("embedding_width") != 768
        or type(compact.get("step")) is not int
        or compact["step"] < 1
        or len(compact.get("validation_labels", [])) != 5_851
    ):
        raise ValueError("SOP matched width authorities differ")
    labels = np.asarray(compact["validation_labels"], dtype=np.int64)
    summary: dict[str, object] = {
        "schema": "sfora-sop-matched-width-holdout-comparison-v1",
        "claim_eligible": False,
        "dataset": "Stanford Online Products",
        "split": "official train class-disjoint holdout, same 5851 queries and gallery images",
        "seed": SEED,
        "step": compact["step"],
        "compact_width": 128,
        "full_width": 768,
        "compact_bytes_per_gallery_item": 130,
        "full_bytes_per_gallery_item": 770,
        "compact_receipt_sha256": sha256(args.compact_receipt),
        "fullwidth_receipt_sha256": sha256(args.fullwidth_receipt),
        "compact_checkpoint_sha256": compact["checkpoint_sha256"],
        "fullwidth_checkpoint_sha256": full["checkpoint_sha256"],
        "script_sha256": sha256(Path(__file__)),
        "validation_row_indexes_sha256": compact["validation_row_indexes_sha256"],
        "validation_labels_sha256": hashlib.sha256(labels.astype("<i8").tobytes()).hexdigest(),
        "bootstrap": "10000 product-cluster resamples with replacement, seed 179019",
    }
    for arm in ("float", "packed"):
        c = compact["validation"][arm]
        f = full["validation"][arm]
        c_r1 = score_array(c, "per_query_r1", len(labels))
        f_r1 = score_array(f, "per_query_r1", len(labels))
        c_ap = score_array(c, "per_query_ap", len(labels))
        f_ap = score_array(f, "per_query_ap", len(labels))
        summary[arm] = {
            "compact_recall_at_1": c["recall_at_1"],
            "fullwidth_recall_at_1": f["recall_at_1"],
            "fullwidth_minus_compact_recall_at_1": float((f_r1 - c_r1).mean()),
            "recall_at_1_product_bootstrap_ci95": cluster_interval(labels, f_r1 - c_r1),
            "compact_map_at_r": c["map_at_r"],
            "fullwidth_map_at_r": f["map_at_r"],
            "fullwidth_minus_compact_map_at_r": float((f_ap - c_ap).mean()),
            "map_at_r_product_bootstrap_ci95": cluster_interval(labels, f_ap - c_ap),
            "fullwidth_only_hits": int(np.count_nonzero((f_r1 == 1) & (c_r1 == 0))),
            "compact_only_hits": int(np.count_nonzero((c_r1 == 1) & (f_r1 == 0))),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(summary, stream, sort_keys=True, indent=2)
        stream.write("\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
