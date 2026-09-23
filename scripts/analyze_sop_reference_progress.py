#!/usr/bin/env python3
"""Measure paired SOP train-holdout progress without reading official test data."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

PAIRED_FIELDS = (
    "arm",
    "recipe",
    "seed",
    "total_updates",
    "input_checkpoint_sha256",
    "features_archive_sha256",
    "sop_train_metadata_sha256",
    "source_sha256",
    "upstream_retrieval_sha256",
    "upstream_launch_sha256",
    "train_transform_sha256",
    "schedule_sha256",
    "fit_row_indexes_sha256",
    "validation_row_indexes_sha256",
    "validation_image_ids",
    "validation_labels",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def product_bootstrap(
    delta: np.ndarray, labels: np.ndarray, *, seed: int, replicates: int
) -> dict[str, float]:
    """Resample whole products, preserving each product's number of queries."""
    delta = np.asarray(delta, dtype=np.float64)
    labels = np.asarray(labels)
    if (
        delta.ndim != 1
        or labels.ndim != 1
        or len(delta) != len(labels)
        or len(delta) == 0
        or not np.all(np.isfinite(delta))
        or not np.issubdtype(labels.dtype, np.integer)
        or type(seed) is not int
        or type(replicates) is not int
        or replicates < 1
    ):
        raise ValueError("paired product inputs differ")
    classes, inverse = np.unique(labels, return_inverse=True)
    counts = np.bincount(inverse)
    totals = np.bincount(inverse, weights=delta, minlength=len(classes))
    generator = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=np.float64)
    for start in range(0, replicates, 512):
        stop = min(start + 512, replicates)
        selected = generator.integers(0, len(classes), size=(stop - start, len(classes)))
        draws[start:stop] = totals[selected].sum(axis=1) / counts[selected].sum(axis=1)
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return {
        "point": float(delta.mean()),
        "lower_95": float(lower),
        "upper_95": float(upper),
    }


def _metric(receipt: dict[str, object], name: str, n: int) -> np.ndarray:
    packed = receipt["validation"]["packed"]
    values = np.asarray(packed[f"per_query_{name}"], dtype=np.float64)
    aggregate = packed["map_at_r" if name == "ap" else "recall_at_1"]
    if (
        values.shape != (n,)
        or not np.all(np.isfinite(values))
        or np.any((values < 0) | (values > 1))
        or not isinstance(aggregate, (int, float))
        or not math.isclose(float(values.mean()), aggregate, abs_tol=1e-10)
    ):
        raise ValueError("SOP holdout metric differs")
    if name == "r1" and not np.all((values == 0) | (values == 1)):
        raise ValueError("SOP holdout metric differs")
    return values


def analyze(
    earlier: dict[str, object], later: dict[str, object], *, seed: int, replicates: int
) -> dict[str, object]:
    """Require one matched diagnostic run before computing paired deltas."""
    if (
        earlier.get("schema") != "sfora-sop-compact-training-diagnostic-v1"
        or later.get("schema") != earlier["schema"]
        or earlier.get("claim_eligible") is not False
        or later.get("claim_eligible") is not False
        or earlier.get("resumable") is not False
        or later.get("resumable") is not False
        or earlier.get("arm") != "arcface"
        or earlier.get("recipe") != "reference"
        or type(earlier.get("step")) is not int
        or type(later.get("step")) is not int
        or earlier["step"] >= later["step"]
        or any(earlier.get(key) != later.get(key) for key in PAIRED_FIELDS)
    ):
        raise ValueError("SOP holdout pairing differs")
    labels = np.asarray(earlier["validation_labels"], dtype=np.int64)
    if len(labels) != 5851 or len(np.unique(labels)) != 1132:
        raise ValueError("SOP holdout inventory differs")
    metrics: dict[str, object] = {}
    for name, label in (("r1", "recall_at_1"), ("ap", "map_at_r")):
        delta = _metric(later, name, len(labels)) - _metric(earlier, name, len(labels))
        metrics[label] = {
            **product_bootstrap(delta, labels, seed=seed, replicates=replicates),
            "queries_gained": int(np.count_nonzero(delta > 0)),
            "queries_lost": int(np.count_nonzero(delta < 0)),
            "queries_unchanged": int(np.count_nonzero(delta == 0)),
        }
    return {
        "earlier_step": earlier["step"],
        "later_step": later["step"],
        "query_count": len(labels),
        "product_count": len(np.unique(labels)),
        "bootstrap_seed": seed,
        "bootstrap_replicates": replicates,
        "resampling_unit": "product identity with all original queries",
        "metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--earlier", type=Path, required=True)
    parser.add_argument("--later", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=179019)
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--execute-sop-reference-progress", action="store_true", required=True)
    args = parser.parse_args()
    result = analyze(
        json.loads(args.earlier.read_text()),
        json.loads(args.later.read_text()),
        seed=args.seed,
        replicates=args.replicates,
    )
    receipt = {
        "schema": "sfora-sop-reference-holdout-progress-v1",
        "claim_eligible": False,
        "inputs": {
            "earlier_sha256": sha256(args.earlier),
            "later_sha256": sha256(args.later),
            "script_sha256": sha256(Path(__file__)),
        },
        **result,
    }
    payload = (
        json.dumps(receipt, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()
    with args.output.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(result["metrics"], sort_keys=True))


if __name__ == "__main__":
    main()
