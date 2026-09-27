#!/usr/bin/env python3
"""Paired class-bootstrap summary of fixed SOP-to-CUB checkpoint transfer."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from export_unicom_cub_embeddings import ordered_record_sha256, parse_cub_records

RUN = Path("/home/riomus/runs/sfora-sop-cub-transfer-public-v1")
DATA = Path("/home/riomus/datasets/CUB_200_2011_official/extracted/CUB_200_2011")
SEEDS = (179024, 179026, 179027)


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    output = RUN / "decision.json"
    if output.exists():
        raise ValueError("CUB transfer decision already exists")
    records = tuple(row for row in parse_cub_records(DATA) if row.split == "test")
    labels = np.asarray([row.label for row in records], dtype=np.int64)
    groups = [np.flatnonzero(labels == label) for label in np.unique(labels)]
    if len(groups) != 100 or len(records) != 5924:
        raise ValueError("CUB transfer class inventory differs")
    paired = {}
    deltas = {}
    input_hashes = {}
    for seed in SEEDS:
        arms = {}
        for arm in ("control", "freeze"):
            path = RUN / f"{seed}-{arm}.json"
            receipt = json.loads(path.read_text())
            if (
                receipt["seed"] != seed
                or receipt["arm"] != arm
                or receipt["rows"] != len(records)
                or receipt["cub_test_records_sha256"] != ordered_record_sha256(records)
                or receipt["claim_eligible"] is not False
            ):
                raise ValueError("CUB transfer receipt authority differs")
            quality = receipt["quality"]
            hits = np.asarray(quality["per_query_r1"], dtype=np.float64)
            ap = np.asarray(quality["per_query_ap"], dtype=np.float64)
            if (
                hits.shape != (len(records),)
                or ap.shape != hits.shape
                or not np.isfinite(ap).all()
                or not np.isin(hits, (0, 1)).all()
                or not np.isclose(hits.mean(), quality["recall_at_1"])
                or not np.isclose(ap.mean(), quality["map_at_r"])
            ):
                raise ValueError("CUB transfer query scores differ")
            arms[arm] = (hits, ap)
            input_hashes[f"{seed}-{arm}"] = sha(path)
        paired[str(seed)] = {
            "control_r1": float(arms["control"][0].mean()),
            "freeze_r1": float(arms["freeze"][0].mean()),
            "control_mapr": float(arms["control"][1].mean()),
            "freeze_mapr": float(arms["freeze"][1].mean()),
        }
        deltas[seed] = tuple(arms["freeze"][metric] - arms["control"][metric] for metric in (0, 1))
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
        "schema": "sfora-sop-cub-transfer-decision-v1",
        "claim_eligible": False,
        "source_sha256": sha(Path(__file__)),
        "cub_test_records_sha256": ordered_record_sha256(records),
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
