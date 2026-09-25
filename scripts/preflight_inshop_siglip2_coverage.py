#!/usr/bin/env python3
"""Freeze an In-Shop train-only class split and paired coverage schedules."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from sfora.unicom_inshop import parse_inshop_partition
from sfora.unicom_rank_finish import identity_balanced_batches

PARTITION_SHA256 = "cfada103c44df866db5e2ee9ecc2301ca691a4d0cdb3c875fe4051b62570894c"
SEEDS = (179023, 179024, 179025)
SPLIT_SEED = 179019


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fit_and_holdout(labels: tuple[str, ...]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    counts = Counter(labels)
    singleton = {name for name, count in counts.items() if count == 1}
    candidates = sorted(
        set(counts) - singleton,
        key=lambda name: (
            hashlib.sha256(SPLIT_SEED.to_bytes(8, "little") + name.encode()).digest(),
            name,
        ),
    )
    held_count = int(math.floor(len(candidates) * 0.1 + 0.5))
    held_names = set(candidates[:held_count])
    fit = tuple(index for index, name in enumerate(labels) if name not in held_names)
    held = tuple(index for index, name in enumerate(labels) if name in held_names)
    if not fit or not held or any(name in held_names for name in singleton):
        raise ValueError("In-Shop train-only class split differs")
    return fit, held


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if (
        args.output.exists()
        or sha256(args.dataset_root / "Eval/list_eval_partition.txt") != PARTITION_SHA256
    ):
        raise ValueError("In-Shop preflight source differs")
    records = parse_inshop_partition(args.dataset_root)
    train = tuple(row for row in records if row.split == "train")
    labels = tuple(row.label for row in train)
    if len(train) != 25_882 or len(set(labels)) != 3_997:
        raise ValueError("In-Shop TRAIN inventory differs")
    fit, held = fit_and_holdout(labels)
    fit_labels = tuple(labels[index] for index in fit)
    counts = Counter(fit_labels)
    singleton = {name for name, count in counts.items() if count == 1}
    schedules = {}
    for seed in SEEDS:
        batches = identity_balanced_batches(
            fit_labels,
            batch_size=64,
            images_per_identity=4,
            seed=seed,
            epoch=1,
            steps=1_000,
            coverage_first=True,
        )
        singleton_steps = [
            step
            for step, batch in enumerate(batches, start=1)
            if any(fit_labels[index] in singleton for index in batch)
        ]
        schedules[str(seed)] = {
            "sha256": hashlib.sha256(np.asarray(batches, dtype="<i4").tobytes()).hexdigest(),
            "unique_fit_rows_touched": len({index for batch in batches for index in batch}),
            "singleton_steps": singleton_steps,
            "rank_active_updates": len(batches) - len(singleton_steps),
        }
    result = {
        "schema": "sfora-inshop-siglip2-coverage-preflight-v1",
        "claim_eligible": False,
        "split": (
            "official In-Shop TRAIN; class-disjoint 90/10 non-singleton holdout; "
            "all singleton classes in fit"
        ),
        "partition_sha256": PARTITION_SHA256,
        "split_seed": SPLIT_SEED,
        "train_rows": len(train),
        "train_products": len(set(labels)),
        "fit_rows": len(fit),
        "fit_products": len(counts),
        "holdout_rows": len(held),
        "holdout_products": len({labels[index] for index in held}),
        "singleton_fit_products": len(singleton),
        "fit_row_indexes_sha256": hashlib.sha256(
            np.asarray(fit, dtype="<i8").tobytes()
        ).hexdigest(),
        "holdout_row_indexes_sha256": hashlib.sha256(
            np.asarray(held, dtype="<i8").tobytes()
        ).hexdigest(),
        "sampler_source_sha256": sha256(Path(identity_balanced_batches.__code__.co_filename)),
        "source_sha256": sha256(Path(__file__)),
        "singleton_rule": (
            "all images stay in ArcFace; rank term disabled for entire "
            "singleton-containing batch in every paired arm; bank refresh remains active"
        ),
        "schedules": schedules,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "fit_rows": len(fit),
                "holdout_rows": len(held),
                "singleton_fit_products": len(singleton),
                "schedules": {
                    seed: {
                        key: row[key] for key in ("unique_fit_rows_touched", "rank_active_updates")
                    }
                    for seed, row in schedules.items()
                },
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
