#!/usr/bin/env python3
"""Kill-only paired packed quality gate for SOP warm-start at 100 updates."""

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from preflight_inshop_siglip2_unseen_gallery import digest_rows, split
from score_inshop_crop_view_pair import (
    GALLERY_SHA,
    PARTITION_SHA,
    QUERY_SHA,
    bootstrap_lower,
    roles,
    sha256,
)

from sfora.joint_relational_compaction import pack_int8_unit_embeddings
from sfora.unicom_inshop import parse_inshop_partition

BASE = Path("/home/riomus/runs")
DATA = Path("/home/riomus/datasets/In-shop Clothes Retrieval Benchmark")
RUNS = {
    "control": BASE / "sfora-inshop-sop-warmstart-control-179024-100-v1",
    "treatment": BASE / "sfora-inshop-sop-warmstart-treatment-179024-100-v1",
}
OUTPUT = BASE / "sfora-inshop-sop-warmstart-smoke-v1/packed_100_quality.json"


@torch.inference_mode()
def packed_quality(
    values: np.ndarray,
    labels: tuple[str, ...],
    query: list[int],
    gallery: list[int],
    *,
    device: torch.device | None = None,
) -> dict:
    device = device or torch.device("cuda")
    packed = pack_int8_unit_embeddings(torch.from_numpy(values.copy()))
    code = packed.codes.float().to(device)
    inverse = packed.inverse_norms.float().to(device)
    classes = {name: index for index, name in enumerate(sorted(set(labels)))}
    ids = torch.tensor([classes[label] for label in labels], device=device)
    relevant = torch.bincount(ids[gallery])[ids[query]]
    if int(relevant.min()) < 1:
        raise ValueError("In-Shop packed positive inventory differs")
    width = int(relevant.max())
    ranks = torch.arange(1, width + 1, device=device)
    hits: list[int] = []
    aps: list[float] = []
    for start in range(0, len(query), 128):
        rows = query[start : start + 128]
        scores = (code[rows] @ code[gallery].T) * inverse[rows, None] * inverse[None, gallery]
        order = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :width]
        matches = ids[gallery][order] == ids[rows, None]
        counts = relevant[start : start + len(rows)]
        precision = matches.cumsum(dim=1) / ranks[None, :]
        ap = (precision * matches * (ranks[None, :] <= counts[:, None])).sum(dim=1) / counts
        hits.extend(int(value) for value in matches[:, 0].cpu().tolist())
        aps.extend(float(value) for value in ap.cpu().tolist())
    return {
        "recall_at_1": float(np.mean(hits)),
        "map_at_r": float(np.mean(aps)),
        "per_query_r1": hits,
        "per_query_ap": aps,
    }


def main() -> None:
    started = time.perf_counter()
    if OUTPUT.exists() or sha256(DATA / "Eval/list_eval_partition.txt") != PARTITION_SHA:
        raise ValueError("In-Shop packed gate authority differs")
    receipts = {
        name: json.loads((path / "receipt.json").read_text()) for name, path in RUNS.items()
    }
    control, treatment = receipts.values()
    paired = (
        "source_sha256",
        "source_files_sha256",
        "fit_rows_sha256",
        "held_rows_sha256",
        "executed_schedule_sha256",
        "first_input_batch_sha256",
        "preflight_sha256",
        "model_file_sha256",
    )
    if (
        any(control[key] != treatment[key] for key in paired)
        or any(
            r["arm"] != "freeze_emb" or r["seed"] != 179024 or r["updates"] != 100
            for r in receipts.values()
        )
        or control["vision_init_sha256"] is not None
        or treatment["vision_init_sha256"]
        != "2c838561b6c23242d74eb29329fd026cc8fba9bf965dcc4529348028dfe6d172"
    ):
        raise ValueError("In-Shop packed pair differs")
    train = tuple(row for row in parse_inshop_partition(DATA) if row.split == "train")
    _, held = split(tuple(row.label for row in train))
    if len(train) != 25_882 or len(held) != 12_599:
        raise ValueError("In-Shop held split differs")
    labels = tuple(train[row].label for row in held)
    query, gallery = roles(labels, tuple(train[row].image_path for row in held), DATA)
    if (
        len(query) != 6_354
        or len(gallery) != 6_245
        or digest_rows(held) != "9b1151e8cf65343682bd10b92885e79447d418ee6c664e2c005efccf7ebb5d1b"
        or hashlib.sha256(np.asarray(query, dtype="<i4").tobytes()).hexdigest() != QUERY_SHA
        or hashlib.sha256(np.asarray(gallery, dtype="<i4").tobytes()).hexdigest() != GALLERY_SHA
    ):
        raise ValueError("In-Shop packed roles differ")
    quality = {}
    for name, path in RUNS.items():
        receipt = receipts[name]
        values_path = path / "held_values.npy"
        values = np.load(values_path, allow_pickle=False)
        if (
            sha256(values_path) != receipt["held_values_sha256"]
            or values.shape != (12_599, 128)
            or values.dtype != np.float32
            or not np.isfinite(values).all()
            or receipt["checkpoint_sha256"] != sha256(path / "checkpoint.pt")
            or len(receipt["preclip_grad_norms"]) != 100
            or not np.isfinite(receipt["preclip_grad_norms"]).all()
            or not np.isfinite([receipt["first_loss"], receipt["last_loss"]]).all()
        ):
            raise ValueError(f"{name} packed checkpoint or gradients differ")
        quality[name] = packed_quality(values, labels, query, gallery)
    delta = np.asarray(quality["treatment"]["per_query_r1"]) - np.asarray(
        quality["control"]["per_query_r1"]
    )
    products = np.asarray([labels[row] for row in query])
    gain = float(delta.mean())
    lower = bootstrap_lower(delta, products)
    upper = -bootstrap_lower(-delta, products)
    wall_ratio = treatment["training_wall_seconds"] / control["training_wall_seconds"]
    cuda_ratio = (
        treatment["training_peak_cuda_allocated_bytes"]
        / control["training_peak_cuda_allocated_bytes"]
    )
    symmetric = {name: receipt["quality"] for name, receipt in receipts.items()}
    passed = (
        gain >= 0.003
        and lower > 0
        and quality["treatment"]["map_at_r"] >= quality["control"]["map_at_r"]
        and symmetric["treatment"]["recall_at_1"] >= symmetric["control"]["recall_at_1"] - 0.003
        and symmetric["treatment"]["map_at_r"] >= symmetric["control"]["map_at_r"] - 0.003
        and wall_ratio <= 1.2
        and cuda_ratio <= 1.2
    )
    report = {
        "schema": "sfora-inshop-sop-warmstart-packed-100-v1",
        "claim_eligible": False,
        "split": "official In-Shop TRAIN held products; packed 6354 query / 6245 gallery",
        "source_sha256": sha256(Path(__file__)),
        "receipt_sha256": {name: sha256(path / "receipt.json") for name, path in RUNS.items()},
        "asymmetric_packed": quality,
        "symmetric_packed": symmetric,
        "r1_gain_percentage_points": 100 * gain,
        "r1_paired_product_bootstrap_95_pp": [100 * lower, 100 * upper],
        "training_wall_ratio": wall_ratio,
        "training_cuda_ratio": cuda_ratio,
        "gate_passed": passed,
        "score_wall_seconds": time.perf_counter() - started,
    }
    with OUTPUT.open("x") as stream:
        json.dump(report, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "r1_gain_percentage_points",
                    "r1_paired_product_bootstrap_95_pp",
                    "training_wall_ratio",
                    "gate_passed",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
