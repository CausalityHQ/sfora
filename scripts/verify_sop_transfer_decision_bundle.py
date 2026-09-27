#!/usr/bin/env python3
"""Replay CUB/Cars paired intervals from committed receipts and ordered labels."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from verify_sop_siglip2_cub_transfer import CHECKPOINTS

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/compact_metric/sop-siglip2-substrate-v1"
SEEDS = (179024, 179026, 179027)


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify(dataset: str) -> None:
    folder = EVIDENCE / f"sop-{dataset}-transfer-public-v1"
    labels_receipt = json.loads((folder / "labels.json").read_text())
    labels = np.asarray(labels_receipt["labels"], dtype=np.int64)
    decision = json.loads((folder / "decision.json").read_text())
    verifier = ROOT / f"scripts/verify_sop_siglip2_{dataset}_transfer.py"
    analyzer = ROOT / f"scripts/analyze_sop_siglip2_{dataset}_transfer.py"
    if (
        labels_receipt["source_sha256"] != sha(ROOT / "scripts/archive_sop_transfer_labels.py")
        or decision["source_sha256"] != sha(analyzer)
        or decision["claim_eligible"] is not False
        or tuple(decision["seed_order"]) != SEEDS
        or len(labels) != (5924 if dataset == "cub" else 8131)
        or len(np.unique(labels)) != (100 if dataset == "cub" else 98)
    ):
        raise ValueError(f"{dataset} transfer bundle authority differs")
    if dataset == "cub":
        if labels_receipt["cub_test_records_sha256"] != decision["cub_test_records_sha256"]:
            raise ValueError("CUB ordered labels differ")
    else:
        if labels_receipt["dataset_revision"] != decision["dataset_revision"] or labels_receipt[
            "source_fingerprints"
        ] != {"train": "c027c63962212a03", "test": "59e3e308b987bbb7"}:
            raise ValueError("Cars ordered labels differ")
    groups = [np.flatnonzero(labels == value) for value in np.unique(labels)]
    per_seed = {}
    for seed, arm, receipt_sha, checkpoint_sha in CHECKPOINTS:
        path = folder / f"{seed}-{arm}.json"
        receipt = json.loads(path.read_text())
        if (
            sha(path) != decision["input_receipt_sha256"][f"{seed}-{arm}"]
            or receipt["source_sha256"] != sha(verifier)
            or receipt["serving_sha256"]
            != (
                "5fcf261053136c916e0fe6c51119036b8114ea3f599d69c1080d7225d3ebe73a"
                if dataset == "cub"
                else sha(ROOT / "src/sfora/siglip2_compact_serving.py")
            )
            or receipt["training_receipt_sha256"] != receipt_sha
            or receipt["checkpoint_sha256"] != checkpoint_sha
            or receipt["seed"] != seed
            or receipt["arm"] != arm
            or receipt["claim_eligible"] is not False
            or receipt["rows"] != len(labels)
            or receipt["gallery_wire_bytes"] != 130 * len(labels)
            or (
                receipt["cub_test_records_sha256"]
                if dataset == "cub"
                else receipt["cars_image_manifest_sha256"]
            )
            != (
                decision["cub_test_records_sha256"]
                if dataset == "cub"
                else decision["cars_image_manifest_sha256"]
            )
            or (
                dataset == "cars"
                and (
                    receipt["cache_file_sha256"] != labels_receipt["cache_file_sha256"]
                    or receipt["source_fingerprints"] != labels_receipt["source_fingerprints"]
                )
            )
        ):
            raise ValueError(f"{dataset} {seed} {arm} receipt differs")
        quality = receipt["quality"]
        values = tuple(
            np.asarray(quality[key], dtype=np.float64) for key in ("per_query_r1", "per_query_ap")
        )
        if any(value.shape != labels.shape or not np.isfinite(value).all() for value in values):
            raise ValueError(f"{dataset} per-query inventory differs")
        if not np.isclose(values[0].mean(), quality["recall_at_1"]) or not np.isclose(
            values[1].mean(), quality["map_at_r"]
        ):
            raise ValueError(f"{dataset} per-query metrics differ")
        per_seed[seed, arm] = values
    rng = np.random.default_rng(decision["bootstrap_seed"])
    draws = rng.integers(0, len(groups), size=(decision["bootstrap_draws"], len(groups)))
    for metric, mean_key, interval_key in (
        (0, "mean_r1_delta", "r1_class_bootstrap_95"),
        (1, "mean_mapr_delta", "mapr_class_bootstrap_95"),
    ):
        delta = np.mean(
            [
                per_seed[seed, "freeze"][metric] - per_seed[seed, "control"][metric]
                for seed in SEEDS
            ],
            axis=0,
        )
        sums = np.asarray([delta[group].sum() for group in groups])
        counts = np.asarray([len(group) for group in groups])
        interval = np.quantile(sums[draws].sum(axis=1) / counts[draws].sum(axis=1), (0.025, 0.975))
        if not np.isclose(delta.mean(), decision[mean_key]) or not np.allclose(
            interval, decision[interval_key], rtol=0, atol=1e-12
        ):
            raise ValueError(f"{dataset} paired interval differs")
    print(dataset, "paired decision replay verified", sha(folder / "decision.json"))


if __name__ == "__main__":
    verify("cub")
    verify("cars")
