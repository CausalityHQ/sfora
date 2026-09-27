#!/usr/bin/env python3
"""Paired class-bootstrap summary of fixed SOP-to-Cars checkpoint transfer."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from datasets import load_dataset

RUN = Path("/home/riomus/runs/sfora-sop-cars-transfer-public-v1")
SEEDS = (179024, 179026, 179027)
REVISION = "9abf6cf7d6dfa7b95152a0d6e791ea9435b47a40"


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    output = RUN / "decision.json"
    if output.exists():
        raise ValueError("Cars transfer decision already exists")
    source = load_dataset("tanganke/stanford_cars", revision=REVISION)
    if {split: source[split]._fingerprint for split in ("train", "test")} != {
        "train": "c027c63962212a03",
        "test": "59e3e308b987bbb7",
    }:
        raise ValueError("Cars transfer source inventory differs")
    labels = np.asarray(
        [
            int(label)
            for split in ("train", "test")
            for label in source[split]["label"]
            if int(label) >= 98
        ],
        dtype=np.int64,
    )
    groups = [np.flatnonzero(labels == label) for label in np.unique(labels)]
    if len(labels) != 8131 or len(groups) != 98:
        raise ValueError("Cars transfer class inventory differs")
    paired, deltas, input_hashes = {}, {}, {}
    inventory = None
    for seed in SEEDS:
        arms = {}
        for arm in ("control", "freeze"):
            path = RUN / f"{seed}-{arm}.json"
            receipt = json.loads(path.read_text())
            same = (receipt["cars_image_manifest_sha256"], receipt["source_fingerprints"])
            if inventory is None:
                inventory = same
            if (
                receipt["seed"] != seed
                or receipt["arm"] != arm
                or receipt["rows"] != len(labels)
                or same != inventory
                or receipt["dataset_revision"] != REVISION
                or receipt["claim_eligible"] is not False
            ):
                raise ValueError("Cars transfer receipt authority differs")
            quality = receipt["quality"]
            hits = np.asarray(quality["per_query_r1"], dtype=np.float64)
            ap = np.asarray(quality["per_query_ap"], dtype=np.float64)
            if (
                hits.shape != (len(labels),)
                or ap.shape != hits.shape
                or not np.isfinite(ap).all()
                or not np.isin(hits, (0, 1)).all()
                or not np.isclose(hits.mean(), quality["recall_at_1"])
                or not np.isclose(ap.mean(), quality["map_at_r"])
            ):
                raise ValueError("Cars transfer query scores differ")
            arms[arm] = (hits, ap)
            input_hashes[f"{seed}-{arm}"] = sha(path)
        paired[str(seed)] = {
            f"{arm}_{metric}": float(arms[arm][idx].mean())
            for arm in ("control", "freeze")
            for idx, metric in enumerate(("r1", "mapr"))
        }
        deltas[seed] = tuple(arms["freeze"][i] - arms["control"][i] for i in (0, 1))
    mean_delta = [np.mean([deltas[s][i] for s in SEEDS], axis=0) for i in (0, 1)]
    rng = np.random.default_rng(179027)
    draws = rng.integers(0, len(groups), size=(5000, len(groups)))
    intervals = []
    for delta in mean_delta:
        sums = np.asarray([delta[group].sum() for group in groups])
        counts = np.asarray([len(group) for group in groups])
        boot = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
        intervals.append([float(x) for x in np.quantile(boot, (0.025, 0.975))])
    result = {
        "schema": "sfora-sop-cars-transfer-decision-v1",
        "claim_eligible": False,
        "source_sha256": sha(Path(__file__)),
        "dataset_revision": REVISION,
        "cars_image_manifest_sha256": inventory[0],
        "input_receipt_sha256": input_hashes,
        "seed_order": SEEDS,
        "paired": paired,
        "mean_r1_delta": float(mean_delta[0].mean()),
        "mean_mapr_delta": float(mean_delta[1].mean()),
        "r1_class_bootstrap_95": intervals[0],
        "mapr_class_bootstrap_95": intervals[1],
        "bootstrap_seed": 179027,
        "bootstrap_draws": len(draws),
    }
    output.write_text(json.dumps(result, sort_keys=True, allow_nan=False) + "\n")
    print(
        json.dumps(
            {k: result[k] for k in ("mean_r1_delta", "mean_mapr_delta", "r1_class_bootstrap_95")}
        )
    )


if __name__ == "__main__":
    main()
