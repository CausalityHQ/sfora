#!/usr/bin/env python3
"""Frozen native packed TRAIN gate for the paired In-Shop crop-scale screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from preflight_inshop_siglip2_unseen_gallery import split

from sfora.cutile_int8 import CutilePackedInt8Gallery
from sfora.joint_relational_compaction import PackedInt8Embeddings, pack_int8_unit_embeddings
from sfora.unicom_inshop import parse_inshop_partition

PARTITION_SHA = "cfada103c44df866db5e2ee9ecc2301ca691a4d0cdb3c875fe4051b62570894c"
NATIVE_SHA = "39602d0e4e8b0d5ec441be460ad7f18e288241bef19fb6e6c5df14f4033ac73c"
QUERY_SHA = "89f1dacd6dd94147578c46b2a6830655a17d1979bf58f021ddd49ecf4549ac68"
GALLERY_SHA = "e7114b2c24bfe9625a47d698729c8c4e09de18b16541e7919e7983baed7290c3"
TARGET = {"additional", "full", "flat"}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def roles(
    labels: tuple[str, ...], paths: tuple[Path, ...], root: Path
) -> tuple[list[int], list[int]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for row, label in enumerate(labels):
        grouped[label].append(row)
    query, gallery = [], []
    for label in sorted(grouped):
        rows = sorted(
            grouped[label],
            key=lambda row: hashlib.sha256(str(paths[row].relative_to(root)).encode()).digest(),
        )
        count = max(1, min(len(rows) - 1, round(len(rows) / 2)))
        gallery.extend(rows[:count])
        query.extend(rows[count:])
    return sorted(query), sorted(gallery)


def packed_hits(
    values: np.ndarray,
    labels: tuple[str, ...],
    query: list[int],
    gallery: list[int],
    library: Path,
) -> np.ndarray:
    packed = pack_int8_unit_embeddings(torch.from_numpy(values.copy()))
    gallery_pack = PackedInt8Embeddings(
        packed.codes[gallery].contiguous(), packed.inverse_norms[gallery].contiguous()
    )
    query_pack = PackedInt8Embeddings(
        packed.codes[query].contiguous(), packed.inverse_norms[query].contiguous()
    )
    with CutilePackedInt8Gallery.open_packed(library, gallery_pack) as index:
        ordinals, scores = index.search_packed(query_pack)
    if ordinals.shape != (len(query), 10) or not np.isfinite(scores).all():
        raise ValueError("In-Shop crop-view native top-10 differs")
    return np.asarray(
        [
            labels[row] == labels[gallery[int(other)]]
            for row, other in zip(query, ordinals[:, 0], strict=True)
        ],
        dtype=np.int8,
    )


def bootstrap_lower(delta: np.ndarray, labels: np.ndarray) -> float:
    classes, inverse = np.unique(labels, return_inverse=True)
    counts = np.bincount(inverse)
    totals = np.bincount(inverse, weights=delta, minlength=len(classes))
    rng = np.random.default_rng(179019)
    draws = np.empty(5_000)
    for start in range(0, len(draws), 512):
        stop = min(start + 512, len(draws))
        picked = rng.integers(0, len(classes), size=(stop - start, len(classes)))
        draws[start:stop] = totals[picked].sum(axis=1) / counts[picked].sum(axis=1)
    return float(np.quantile(draws, 0.025))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--native-library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.output.exists()
        or sha256(args.dataset_root / "Eval/list_eval_partition.txt") != PARTITION_SHA
        or sha256(args.native_library) != NATIVE_SHA
    ):
        raise ValueError("In-Shop crop-view score authority differs")
    receipts = {
        arm: json.loads((args.runs_root / f"{arm}-100/receipt.json").read_text())
        for arm in ("control", "treatment")
    }
    control, treatment = receipts.values()
    paired_fields = (
        "source_sha256",
        "source_files_sha256",
        "schedule_sha256",
        "pca_sha256",
        "fit_rows_sha256",
        "held_rows_sha256",
        "features_sha256",
        "model_file_sha256",
        "preflight_sha256",
        "training_coordinates",
        "rank_inactive_steps",
    )
    if (
        any(control[key] != treatment[key] for key in paired_fields)
        or any(
            item.get("seed") != 179026 or item.get("updates") != 100 for item in receipts.values()
        )
        or (control.get("crop_scale_min"), treatment.get("crop_scale_min")) != (0.8, 0.25)
        or control["first_input_batch_sha256"] == treatment["first_input_batch_sha256"]
    ):
        raise ValueError("In-Shop crop-view paired training differs")
    train = tuple(row for row in parse_inshop_partition(args.dataset_root) if row.split == "train")
    _, held = split(tuple(row.label for row in train))
    labels = tuple(train[row].label for row in held)
    paths = tuple(train[row].image_path for row in held)
    query, gallery = roles(labels, paths, args.dataset_root)
    if (len(query), len(gallery)) != (6354, 6245) or any(
        hashlib.sha256(np.asarray(rows, dtype="<i4").tobytes()).hexdigest() != expected
        for rows, expected in ((query, QUERY_SHA), (gallery, GALLERY_SHA))
    ):
        raise ValueError("In-Shop crop-view score roles differ")
    hits = {}
    for arm, receipt in receipts.items():
        path = args.runs_root / f"{arm}-100/held_embeddings.npy"
        values = np.load(path, allow_pickle=False)
        if (
            sha256(path) != receipt["held_embeddings_sha256"]
            or values.shape != (12599, 128)
            or values.dtype != np.float32
            or not np.isfinite(values).all()
        ):
            raise ValueError("In-Shop crop-view embedding export differs")
        hits[arm] = packed_hits(values, labels, query, gallery, args.native_library)
    delta = hits["treatment"].astype(np.float64) - hits["control"]
    target = np.asarray([paths[row].stem.rsplit("_", 1)[-1] in TARGET for row in query])
    if int(target.sum()) != 2223:
        raise ValueError("In-Shop crop-view target framing count differs")
    other = ~target
    query_labels = np.asarray(labels, dtype=object)[query]
    target_delta = float(delta[target].mean())
    target_lower = bootstrap_lower(delta[target], query_labels[target])
    other_delta = float(delta[other].mean())
    whole_delta = float(delta.mean())
    map_delta = treatment["quality"]["map_at_r"] - control["quality"]["map_at_r"]
    wall_ratio = (
        treatment["training_wall_including_member_bank_init_seconds"]
        / control["training_wall_including_member_bank_init_seconds"]
    )
    peak_ratio = (
        treatment["training_peak_cuda_allocated_bytes"]
        / control["training_peak_cuda_allocated_bytes"]
    )
    gates = {
        "target_r1": target_delta >= 0.0045 and target_lower > 0,
        "other_r1": other_delta >= -0.001,
        "whole_r1": whole_delta >= -0.0015,
        "symmetric_mapr": map_delta >= -0.002,
        "training_wall": wall_ratio <= 1.02,
        "peak_cuda": peak_ratio <= 1.005,
    }
    report = {
        "schema": "sfora-inshop-crop-view-paired-100-train-v1",
        "claim_eligible": False,
        "source_sha256": sha256(Path(__file__)),
        "receipt_sha256": {
            arm: sha256(args.runs_root / f"{arm}-100/receipt.json") for arm in receipts
        },
        "native_library_sha256": NATIVE_SHA,
        "query_sha256": QUERY_SHA,
        "gallery_sha256": GALLERY_SHA,
        "target_query_rows": int(target.sum()),
        "control_hits": hits["control"].tolist(),
        "treatment_hits": hits["treatment"].tolist(),
        "target_r1_delta": target_delta,
        "target_product_bootstrap_lower95": target_lower,
        "other_r1_delta": other_delta,
        "whole_r1_delta": whole_delta,
        "symmetric_mapr_delta": map_delta,
        "training_wall_ratio": wall_ratio,
        "training_peak_cuda_ratio": peak_ratio,
        "gates": gates,
        "advance": all(gates.values()),
    }
    with args.output.open("xb") as stream:
        stream.write((json.dumps(report, sort_keys=True, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ("control_hits", "treatment_hits")}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
