#!/usr/bin/env python3
"""Frozen TRAIN-only packed readout for the paired worst-positive bank loss."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from preflight_inshop_siglip2_unseen_gallery import split
from score_inshop_crop_view_pair import (
    GALLERY_SHA,
    NATIVE_SHA,
    PARTITION_SHA,
    QUERY_SHA,
    bootstrap_lower,
    packed_hits,
    roles,
    sha256,
)

from sfora.joint_relational_compaction import pack_int8_unit_embeddings
from sfora.unicom_inshop import parse_inshop_partition


@torch.inference_mode()
def coverage_counts(values: np.ndarray, labels: tuple[str, ...], queries: list[int]) -> list[int]:
    packed = pack_int8_unit_embeddings(torch.from_numpy(values.copy()))
    codes = packed.codes.float().cuda()
    inverse_norms = packed.inverse_norms.float().cuda()
    names = {name: i for i, name in enumerate(sorted(set(labels)))}
    ids = torch.tensor([names[name] for name in labels], device="cuda")
    counts: list[int] = []
    for start in range(0, len(queries), 64):
        rows = torch.tensor(queries[start : start + 64], device="cuda")
        scores = (codes[rows] @ codes.T) * inverse_norms[rows, None] * inverse_norms[None, :]
        same = ids[rows, None] == ids[None, :]
        scores[torch.arange(len(rows), device="cuda"), rows] = -torch.inf
        best_negative = scores.masked_fill(same, -torch.inf).max(dim=1).values
        counts.extend(((scores > best_negative[:, None]) & same).sum(dim=1).cpu().tolist())
    return counts


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
        raise ValueError("In-Shop positive-tail paired score authority differs")
    receipts = {
        arm: json.loads((args.runs_root / f"{arm}-100/receipt.json").read_text())
        for arm in ("control", "treatment")
    }
    control, treatment = receipts.values()
    paired = (
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
        "first_input_batch_sha256",
        "rank_active_updates",
    )
    if (
        any(control[key] != treatment[key] for key in paired)
        or (control.get("arm"), treatment.get("arm")) != ("freeze_emb", "freeze_emb_tail")
        or (control.get("worst_positive_hinge"), treatment.get("worst_positive_hinge"))
        != (False, True)
        or any(
            item.get("seed") != 179026 or item.get("updates") != 100 for item in receipts.values()
        )
    ):
        raise ValueError("In-Shop positive-tail paired training differs")
    train = tuple(row for row in parse_inshop_partition(args.dataset_root) if row.split == "train")
    _, held = split(tuple(row.label for row in train))
    labels = tuple(train[row].label for row in held)
    paths = tuple(train[row].image_path for row in held)
    query, gallery = roles(labels, paths, args.dataset_root)
    if (len(query), len(gallery)) != (6354, 6245) or any(
        hashlib.sha256(np.asarray(rows, dtype="<i4").tobytes()).hexdigest() != expected
        for rows, expected in ((query, QUERY_SHA), (gallery, GALLERY_SHA))
    ):
        raise ValueError("In-Shop positive-tail role inventory differs")
    hits: dict[str, np.ndarray] = {}
    above: dict[str, list[int]] = {}
    for arm, receipt in receipts.items():
        path = args.runs_root / f"{arm}-100/held_embeddings.npy"
        values = np.load(path, allow_pickle=False)
        if (
            sha256(path) != receipt.get("held_embeddings_sha256")
            or values.shape != (12599, 128)
            or values.dtype != np.float32
            or not np.isfinite(values).all()
        ):
            raise ValueError("In-Shop positive-tail held export differs")
        hits[arm] = packed_hits(values, labels, query, gallery, args.native_library)
        above[arm] = coverage_counts(values, labels, query)
    query_labels = np.asarray(labels, dtype=object)[query]
    mechanism = (np.asarray(above["treatment"]) >= 3).astype(np.float64) - (
        np.asarray(above["control"]) >= 3
    ).astype(np.float64)
    hit_delta = hits["treatment"].astype(np.float64) - hits["control"]
    wall_ratio = (
        treatment["training_wall_including_member_bank_init_seconds"]
        / control["training_wall_including_member_bank_init_seconds"]
    )
    peak_ratio = (
        treatment["training_peak_cuda_allocated_bytes"]
        / control["training_peak_cuda_allocated_bytes"]
    )
    map_delta = treatment["quality"]["map_at_r"] - control["quality"]["map_at_r"]
    gates = {
        "coverage": float(mechanism.mean()) >= 0.01
        and bootstrap_lower(mechanism, query_labels) > 0,
        "role_r1": float(hit_delta.mean()) >= -0.0015,
        "symmetric_mapr": map_delta >= -0.002,
        "training_wall": wall_ratio <= 1.02,
        "peak_cuda": peak_ratio <= 1.005,
    }
    report = {
        "schema": "sfora-inshop-positive-tail-paired-100-train-v1",
        "claim_eligible": False,
        "source_sha256": sha256(Path(__file__)),
        "score_helper_sha256": sha256(Path(packed_hits.__code__.co_filename)),
        "receipt_sha256": {
            arm: sha256(args.runs_root / f"{arm}-100/receipt.json") for arm in receipts
        },
        "native_library_sha256": NATIVE_SHA,
        "query_sha256": QUERY_SHA,
        "gallery_sha256": GALLERY_SHA,
        "control_hits": hits["control"].tolist(),
        "treatment_hits": hits["treatment"].tolist(),
        "control_above": above["control"],
        "treatment_above": above["treatment"],
        "coverage_delta": float(mechanism.mean()),
        "coverage_product_bootstrap_lower95": bootstrap_lower(mechanism, query_labels),
        "role_r1_delta": float(hit_delta.mean()),
        "symmetric_mapr_delta": map_delta,
        "training_wall_ratio": wall_ratio,
        "peak_cuda_ratio": peak_ratio,
        "gates": gates,
        "advance": all(gates.values()),
    }
    with args.output.open("xb") as stream:
        stream.write((json.dumps(report, sort_keys=True, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in ("control_hits", "treatment_hits", "control_above", "treatment_above")
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
