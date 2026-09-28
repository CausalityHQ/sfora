#!/usr/bin/env python3
"""CPU-only full-gallery packing attribution on six pinned TRAIN exports."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from aggregate_inshop_sop_warmstart_full import SEEDS, product_seed_bootstrap, validate_quality
from compare_inshop_sop_warmstart_100 import packed_quality
from preflight_inshop_siglip2_unseen_gallery import digest_rows, split
from probe_inshop_sop_product_prior import score
from score_inshop_crop_view_pair import (
    GALLERY_SHA,
    PARTITION_SHA,
    QUERY_SHA,
    bootstrap_lower,
    roles,
    sha256,
)

EVIDENCE = (
    Path(__file__).resolve().parents[1] / "docs/evidence/compact_metric/sop-siglip2-substrate-v1"
)
AGGREGATE = EVIDENCE / "inshop-sop-warmstart-full-aggregate-v1.json"
AGGREGATE_SHA = "23098dedb1d45427427676cb6da2bce3764e445d986ce287fbb89027bac1e18e"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("partition", "values-directory", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.output.exists()
        or sha256(args.partition) != PARTITION_SHA
        or sha256(AGGREGATE) != AGGREGATE_SHA
    ):
        raise ValueError("packing diagnostic authority differs")
    # No images are needed: the exact partition bytes and role digests bind metadata.
    train = [
        line.split()
        for line in args.partition.read_text().splitlines()[2:]
        if line.split()[-1] == "train"
    ]
    _, held = split(tuple(row[1] for row in train))
    labels = tuple(train[row][1] for row in held)
    root = Path("/dataset")
    query, gallery = roles(labels, tuple(root / "Img" / train[row][0] for row in held), root)
    if (
        len(train) != 25_882
        or digest_rows(held) != "9b1151e8cf65343682bd10b92885e79447d418ee6c664e2c005efccf7ebb5d1b"
    ):
        raise ValueError("packing held split differs")
    for rows, digest in ((query, QUERY_SHA), (gallery, GALLERY_SHA)):
        if hashlib.sha256(np.asarray(rows, dtype="<i4").tobytes()).hexdigest() != digest:
            raise ValueError("packing roles differ")
    aggregate = json.loads(AGGREGATE.read_text())
    device = torch.device("cpu")
    torch.set_num_threads(8)
    products = np.asarray([labels[row] for row in query])
    started = time.perf_counter()
    results, deltas, authority = {}, {arm: [] for arm in ("control", "treatment")}, {}
    for seed in SEEDS:
        directory = EVIDENCE / f"inshop-sop-warmstart-full-{seed}-v1"
        pair_path = directory / "paired_quality.json"
        if sha256(pair_path) != aggregate["seed_receipt_sha256"][str(seed)]:
            raise ValueError("packing paired receipt differs")
        pair = json.loads(pair_path.read_text())
        results[str(seed)] = {}
        for arm in deltas:
            receipt_path = directory / f"{arm}.json"
            receipt = json.loads(receipt_path.read_text())
            values_path = args.values_directory / f"{arm}-{seed}.npy"
            if (
                sha256(receipt_path) != pair["receipt_sha256"][arm]
                or sha256(values_path) != receipt["held_values_sha256"]
            ):
                raise ValueError("packing training/embedding receipt differs")
            values = np.load(values_path, allow_pickle=False)
            if (
                values.shape != (12_599, 128)
                or values.dtype != np.float32
                or not np.isfinite(values).all()
            ):
                raise ValueError("packing embedding geometry differs")
            packed = packed_quality(values, labels, query, gallery, device=device)
            archived = pair["asymmetric_packed"][arm]
            validate_quality(archived, len(query))
            if packed["per_query_r1"] != archived["per_query_r1"] or not np.allclose(
                packed["per_query_ap"], archived["per_query_ap"], rtol=0, atol=2e-7
            ):
                raise ValueError("packing CPU scorer does not replay")
            floating = score(torch.from_numpy(values), labels, query, gallery, device=device)
            delta = np.asarray(floating["per_query_r1"]) - np.asarray(packed["per_query_r1"])
            deltas[arm].append(delta)
            lower, upper = bootstrap_lower(delta, products), -bootstrap_lower(-delta, products)
            results[str(seed)][arm] = {
                "float": floating,
                "packed": packed,
                "rescues": int((delta > 0).sum()),
                "regressions": int((delta < 0).sum()),
                "r1_gain_pp": float(100 * delta.mean()),
                "product_bootstrap_95_pp": [100 * lower, 100 * upper],
            }
            authority[f"{arm}-{seed}"] = {
                "training_receipt_sha256": sha256(receipt_path),
                "values_sha256": sha256(values_path),
            }
        print(
            json.dumps(
                {
                    "seed": seed,
                    **{
                        arm: {
                            key: results[str(seed)][arm][key]
                            for key in ("rescues", "regressions", "r1_gain_pp")
                        }
                        for arm in deltas
                    },
                }
            ),
            flush=True,
        )
    treatment = np.stack(deltas["treatment"])
    lower, upper = product_seed_bootstrap(treatment, products)
    gain = float(treatment.mean())
    persistent = np.all(
        [
            np.asarray(results[str(seed)]["treatment"]["packed"]["per_query_r1"]) == 0
            for seed in SEEDS
        ],
        axis=0,
    )
    core = np.all(
        [
            np.asarray(results[str(seed)][arm]["packed"]["per_query_r1"]) == 0
            for seed in SEEDS
            for arm in deltas
        ],
        axis=0,
    )
    map_gain = float(
        np.mean(
            [
                results[str(seed)]["treatment"]["float"]["map_at_r"]
                - results[str(seed)]["treatment"]["packed"]["map_at_r"]
                for seed in SEEDS
            ]
        )
    )
    report = {
        "schema": "sfora-inshop-warmstart-packing-diagnostic-v1",
        "claim_eligible": False,
        "split": "official In-Shop TRAIN held products; 6354 query / 6245 gallery",
        "source_sha256": sha256(Path(__file__)),
        "authority": authority,
        "results": results,
        "mean_treatment_float_minus_packed_r1_pp": 100 * gain,
        "treatment_seed_product_bootstrap_95_pp": [100 * lower, 100 * upper],
        "mean_treatment_float_minus_packed_map": map_gain,
        "treatment_persistent_misses": np.flatnonzero(persistent).tolist(),
        "all_six_persistent_misses": np.flatnonzero(core).tolist(),
        "packing_advance": gain >= 0.003 and lower > 0 and map_gain >= 0,
        "cpu_diagnostic_wall_seconds": time.perf_counter() - started,
        "serving_latency_measured": False,
        "encoder_export_or_training": False,
    }
    args.output.write_text(json.dumps(report, sort_keys=True, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "mean_treatment_float_minus_packed_r1_pp",
                    "treatment_seed_product_bootstrap_95_pp",
                    "packing_advance",
                    "cpu_diagnostic_wall_seconds",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
