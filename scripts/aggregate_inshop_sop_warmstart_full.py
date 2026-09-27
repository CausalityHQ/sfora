#!/usr/bin/env python3
"""Apply the frozen three-seed TRAIN-held warm-start promotion gate."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
from preflight_inshop_siglip2_unseen_gallery import digest_rows, split
from score_inshop_crop_view_pair import PARTITION_SHA, QUERY_SHA, roles, sha256

from sfora.unicom_inshop import parse_inshop_partition

BASE = Path("/home/riomus/runs")
DATA = Path("/home/riomus/datasets/In-shop Clothes Retrieval Benchmark")
SEEDS = (179024, 179025, 179026)
OUTPUT = BASE / "sfora-inshop-sop-warmstart-full-aggregate-v1.json"


def product_seed_bootstrap(deltas: np.ndarray, products: np.ndarray) -> tuple[float, float]:
    if deltas.shape != (3, len(products)) or len(products) == 0 or not np.isfinite(deltas).all():
        raise ValueError("In-Shop product/seed bootstrap geometry differs")
    names, inverse = np.unique(products, return_inverse=True)
    counts = np.bincount(inverse)
    totals = np.stack([np.bincount(inverse, weights=row, minlength=len(names)) for row in deltas])
    rng = np.random.default_rng(179019)
    draws = np.empty(5_000)
    for index in range(len(draws)):
        seeds = rng.integers(0, 3, 3)
        identities = rng.integers(0, len(names), len(names))
        draws[index] = totals[seeds][:, identities].sum() / (3 * counts[identities].sum())
    return tuple(float(value) for value in np.quantile(draws, [0.025, 0.975]))


def main() -> None:
    if OUTPUT.exists() or sha256(DATA / "Eval/list_eval_partition.txt") != PARTITION_SHA:
        raise ValueError("In-Shop full aggregate authority differs")
    paths = {seed: BASE / f"sfora-inshop-sop-warmstart-{seed}-1000-pair-v1.json" for seed in SEEDS}
    reports = {seed: json.loads(path.read_text()) for seed, path in paths.items()}
    source_shas = {report["source_sha256"] for report in reports.values()}
    if len(source_shas) != 1 or any(
        report["seed"] != seed or report["schema"] != "sfora-inshop-sop-warmstart-full-pair-v1"
        for seed, report in reports.items()
    ):
        raise ValueError("In-Shop paired seed reports differ")
    train = tuple(row for row in parse_inshop_partition(DATA) if row.split == "train")
    _, held = split(tuple(row.label for row in train))
    labels = tuple(train[row].label for row in held)
    query, gallery = roles(labels, tuple(train[row].image_path for row in held), DATA)
    if (
        len(train) != 25_882
        or len(held) != 12_599
        or digest_rows(held) != "9b1151e8cf65343682bd10b92885e79447d418ee6c664e2c005efccf7ebb5d1b"
        or len(query) != 6_354
        or len(gallery) != 6_245
        or hashlib.sha256(np.asarray(query, dtype="<i4").tobytes()).hexdigest() != QUERY_SHA
    ):
        raise ValueError("In-Shop aggregate held roles differ")
    deltas = np.stack(
        [
            np.asarray(reports[seed]["asymmetric_packed"]["treatment"]["per_query_r1"])
            - np.asarray(reports[seed]["asymmetric_packed"]["control"]["per_query_r1"])
            for seed in SEEDS
        ]
    )
    products = np.asarray([labels[row] for row in query])
    lower, upper = product_seed_bootstrap(deltas, products)
    mean = {
        arm: {
            metric: float(
                np.mean([reports[seed]["asymmetric_packed"][arm][metric] for seed in SEEDS])
            )
            for metric in ("recall_at_1", "map_at_r")
        }
        for arm in ("control", "treatment")
    }
    wall = {
        arm: float(np.mean([reports[seed]["training_wall_seconds"][arm] for seed in SEEDS]))
        for arm in ("control", "treatment")
    }
    cuda = {
        arm: float(
            np.mean([reports[seed]["training_peak_cuda_allocated_bytes"][arm] for seed in SEEDS])
        )
        for arm in ("control", "treatment")
    }
    gain = float(deltas.mean())
    passed = (
        gain >= 0.003
        and lower > 0
        and mean["treatment"]["map_at_r"] >= mean["control"]["map_at_r"]
        and all(reports[seed]["r1_gain_percentage_points"] > -0.5 for seed in SEEDS)
        and wall["treatment"] / wall["control"] <= 1.2
        and cuda["treatment"] / cuda["control"] <= 1.2
    )
    output = {
        "schema": "sfora-inshop-sop-warmstart-full-aggregate-v1",
        "claim_eligible": False,
        "split": (
            "official In-Shop TRAIN held products; three paired seeds, "
            "packed 6354 query / 6245 gallery"
        ),
        "source_sha256": sha256(Path(__file__)),
        "seed_receipt_sha256": {str(seed): sha256(path) for seed, path in paths.items()},
        "seeds": list(SEEDS),
        "mean_asymmetric_packed": mean,
        "mean_training_wall_seconds": wall,
        "mean_training_peak_cuda_allocated_bytes": cuda,
        "r1_gain_percentage_points": 100 * gain,
        "r1_product_seed_bootstrap_95_pp": [100 * lower, 100 * upper],
        "gate_passed": passed,
    }
    with OUTPUT.open("x") as stream:
        json.dump(output, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(output, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
