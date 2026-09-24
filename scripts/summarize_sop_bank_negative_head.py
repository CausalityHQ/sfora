"""Recompute product-clustered uncertainty from the SOP bank-head raw receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import cast

import numpy as np

from sfora.representation_ceiling import deterministic_class_partition

ARCHIVE_SHA = "16b4554d3868363905f1e1cd385783a8033513835723a7b89f4b762d893d757f"
SEED = 179019
BOOTSTRAPS = 10_000


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def product_interval(labels: np.ndarray, differences: np.ndarray) -> list[float]:
    """Resample complete product classes with replacement, preserving query counts."""

    if differences.shape != labels.shape or not bool(np.isfinite(differences).all()):
        raise ValueError("SOP paired row inventory differs")
    _, inverse = np.unique(labels, return_inverse=True)
    classes = int(inverse.max()) + 1
    counts = np.bincount(inverse, minlength=classes)
    sums = np.bincount(inverse, weights=differences, minlength=classes)
    rng = np.random.default_rng(SEED)
    results = np.empty(BOOTSTRAPS, dtype=np.float64)
    for index in range(BOOTSTRAPS):
        sampled = rng.integers(0, classes, size=classes)
        results[index] = sums[sampled].sum() / counts[sampled].sum()
    return [float(value) for value in np.quantile(results, [0.025, 0.975])]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--raw-receipt", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or sha256(args.source_archive) != ARCHIVE_SHA:
        raise ValueError("SOP bank summary input or output authority differs")
    raw = json.loads(args.raw_receipt.read_text())
    if (
        raw.get("schema") != "sfora-sop-frozen-b16-bank-negative-head-screen-v1"
        or raw.get("source_archive_sha256") != ARCHIVE_SHA
        or raw.get("seed") != SEED
        or raw.get("steps") != 256
        or raw.get("holdout_rows") != 5_851
        or raw.get("gallery_bytes_per_item") != 770
    ):
        raise ValueError("SOP bank raw receipt authority differs")
    with np.load(args.source_archive, allow_pickle=False) as archive:
        labels = np.asarray(archive["train_labels"], dtype=np.int64)
    partition = deterministic_class_partition(tuple(map(int, labels)), fit_fraction=0.9, seed=SEED)
    held_labels = labels[list(partition.validation_row_indexes)]
    arms = cast(dict[str, dict[str, object]], raw["arms"])
    inbatch = cast(dict[str, object], cast(dict[str, object], arms["inbatch"]["score"])["packed"])
    bank = cast(dict[str, object], cast(dict[str, object], arms["fit_bank"]["score"])["packed"])
    base = cast(dict[str, object], cast(dict[str, object], raw["baseline"])["packed"])
    if not all(isinstance(item, dict) for item in (inbatch, bank, base)):
        raise ValueError("SOP packed score authority differs")
    inbatch_r1 = np.asarray(inbatch["per_query_r1"], dtype=np.float64)
    bank_r1 = np.asarray(bank["per_query_r1"], dtype=np.float64)
    ap_delta = np.asarray(bank["per_query_ap"], dtype=np.float64) - np.asarray(
        inbatch["per_query_ap"], dtype=np.float64
    )
    if (
        inbatch_r1.shape != (5_851,)
        or bank_r1.shape != (5_851,)
        or ap_delta.shape != (5_851,)
        or not bool(np.isin(inbatch_r1, (0, 1)).all())
        or not bool(np.isin(bank_r1, (0, 1)).all())
        or abs(inbatch_r1.mean() - cast(float, inbatch["recall_at_1"])) > 1e-12
        or abs(bank_r1.mean() - cast(float, bank["recall_at_1"])) > 1e-12
    ):
        raise ValueError("SOP packed per-query score authority differs")
    summary = {
        "schema": "sfora-sop-frozen-b16-bank-negative-head-screen-summary-v1",
        "claim_eligible": False,
        "dataset": "Stanford Online Products",
        "split": "official train, 5851 class-disjoint held-out images",
        "raw_receipt_sha256": sha256(args.raw_receipt),
        "source_archive_sha256": ARCHIVE_SHA,
        "holdout_labels_sha256": hashlib.sha256(
            held_labels.astype("<i8", copy=False).tobytes(order="C")
        ).hexdigest(),
        "summary_script_sha256": sha256(Path(__file__)),
        "seed": SEED,
        "steps": 256,
        "gallery_bytes_per_item": 770,
        "mining_seconds": raw["mining_seconds"],
        "arm_training_seconds": {
            name: arms[name]["training_seconds"] for name in ("inbatch", "fit_bank")
        },
        "bootstrap": "10000 product-cluster resamples with replacement, seed 179019",
        "baseline_packed": {metric: base[metric] for metric in ("recall_at_1", "map_at_r")},
        "packed": {
            name: {metric: arm[metric] for metric in ("recall_at_1", "map_at_r")}
            for name, arm in (("inbatch", inbatch), ("fit_bank", bank))
        },
        "bank_minus_inbatch_packed_r1": float((bank_r1 - inbatch_r1).mean()),
        "bank_minus_inbatch_packed_r1_product_bootstrap_ci95": product_interval(
            held_labels, bank_r1 - inbatch_r1
        ),
        "bank_minus_inbatch_packed_map_at_r": float(ap_delta.mean()),
        "bank_minus_inbatch_packed_map_at_r_product_bootstrap_ci95": product_interval(
            held_labels, ap_delta
        ),
        "inbatch_only_hits": int(np.count_nonzero((inbatch_r1 == 1) & (bank_r1 == 0))),
        "bank_only_hits": int(np.count_nonzero((bank_r1 == 1) & (inbatch_r1 == 0))),
        "decision": (
            "Reject naive static full-fit-bank single-hardest-negative head loss for "
            "the next full-backbone run; other bank methods are untested"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(summary, stream, sort_keys=True, indent=2)
        stream.write("\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
