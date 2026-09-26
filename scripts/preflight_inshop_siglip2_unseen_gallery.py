#!/usr/bin/env python3
"""Freeze a half-product TRAIN split with an unseen gallery near official scale."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path

import numpy as np

from sfora.unicom_inshop import parse_inshop_partition
from sfora.unicom_rank_finish import identity_balanced_batches

PARTITION_SHA = "cfada103c44df866db5e2ee9ecc2301ca691a4d0cdb3c875fe4051b62570894c"
SEED = 179023
BATCH = 64


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def split(labels: tuple[str, ...]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    counts = Counter(labels)
    eligible = sorted(
        (name for name, count in counts.items() if count > 1),
        key=lambda name: (
            hashlib.sha256(b"inshop-unseen-gallery-v1\0" + name.encode()).digest(),
            name,
        ),
    )
    held_names = set(eligible[: math.ceil(len(eligible) / 2)])
    return (
        tuple(i for i, name in enumerate(labels) if name not in held_names),
        tuple(i for i, name in enumerate(labels) if name in held_names),
    )


def schedule(
    labels: tuple[str, ...], updates: int, *, seed: int = SEED
) -> tuple[tuple[int, ...], ...]:
    return identity_balanced_batches(
        labels,
        batch_size=BATCH,
        images_per_identity=4,
        seed=seed,
        epoch=1,
        steps=updates,
        coverage_first=True,
    )


def digest_rows(rows: tuple[int, ...]) -> str:
    return hashlib.sha256(np.asarray(rows, dtype="<i8").tobytes()).hexdigest()


def digest_schedule(batches: tuple[tuple[int, ...], ...]) -> str:
    return hashlib.sha256(np.asarray(batches, dtype="<i4").tobytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(179023, 179024, 179025), default=SEED)
    args = parser.parse_args()
    if (
        args.output.exists()
        or sha256(args.dataset_root / "Eval/list_eval_partition.txt") != PARTITION_SHA
    ):
        raise ValueError("In-Shop unseen-gallery partition authority differs")
    train = tuple(row for row in parse_inshop_partition(args.dataset_root) if row.split == "train")
    labels = tuple(row.label for row in train)
    fit, held = split(labels)
    fit_labels = tuple(labels[i] for i in fit)
    held_labels = tuple(labels[i] for i in held)
    counts = Counter(fit_labels)
    singletons = {name for name, count in counts.items() if count == 1}
    if (
        len(train) != 25_882
        or len(set(labels)) != 3_997
        or len(singletons) != 12
        or not (12_000 <= len(held) <= 14_000)
        or not set(fit_labels).isdisjoint(held_labels)
        or min(Counter(held_labels).values()) < 2
    ):
        raise ValueError("In-Shop unseen-gallery split differs")
    schedules = {}
    for updates in (1_000, 3_000):
        batches = schedule(fit_labels, updates, seed=args.seed)
        inactive = [
            step
            for step, batch in enumerate(batches, 1)
            if any(fit_labels[i] in singletons for i in batch)
        ]
        if len({i for batch in batches for i in batch}) != len(fit):
            raise ValueError("In-Shop unseen-gallery coverage differs")
        schedules[str(updates)] = {
            "sha256": digest_schedule(batches),
            "rank_inactive_steps": inactive,
            "rank_active_updates": updates - len(inactive),
            "first_ten_batches_sha256": digest_schedule(batches[:10]),
        }
    if (
        schedules["1000"]["first_ten_batches_sha256"]
        != schedules["3000"]["first_ten_batches_sha256"]
    ):
        raise ValueError("In-Shop unseen-gallery paired prefix differs")
    report = {
        "schema": "sfora-inshop-siglip2-unseen-gallery-preflight-v1",
        "partition_sha256": PARTITION_SHA,
        "fit_rows": len(fit),
        "held_rows": len(held),
        "fit_products": len(set(fit_labels)),
        "held_products": len(set(held_labels)),
        "fit_sha256": digest_rows(fit),
        "held_sha256": digest_rows(held),
        "seed": args.seed,
        "schedules": schedules,
        "source_sha256": sha256(Path(__file__)),
    }
    with args.output.open("xb") as stream:
        stream.write((json.dumps(report, sort_keys=True, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    print(
        json.dumps(
            {k: report[k] for k in ("fit_rows", "held_rows", "fit_products", "held_products")}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
