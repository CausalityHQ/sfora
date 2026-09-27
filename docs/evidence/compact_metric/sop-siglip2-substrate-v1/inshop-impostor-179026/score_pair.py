#!/usr/bin/env python3
"""Frozen TRAIN-only packed In-Shop query/gallery gate for the impostor screen."""

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from preflight_inshop_siglip2_unseen_gallery import digest_rows, split

from sfora.joint_relational_compaction import pack_int8_unit_embeddings
from sfora.unicom_inshop import parse_inshop_partition

RUN = Path("/home/riomus/runs/sfora-inshop-impostor-179026-v1")
DATA = Path("/home/riomus/datasets/inshop_official_standard")
QUERY_SHA = "89f1dacd6dd94147578c46b2a6830655a17d1979bf58f021ddd49ecf4549ac68"
GALLERY_SHA = "e7114b2c24bfe9625a47d698729c8c4e09de18b16541e7919e7983baed7290c3"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def role_rows(labels: tuple[str, ...], paths: tuple[Path, ...]) -> tuple[list[int], list[int]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        grouped[label].append(index)
    query: list[int] = []
    gallery: list[int] = []
    for label in sorted(grouped):
        rows = sorted(
            grouped[label],
            key=lambda index: hashlib.sha256(str(paths[index].relative_to(DATA)).encode()).digest(),
        )
        count = max(1, min(len(rows) - 1, round(len(rows) / 2)))
        gallery.extend(rows[:count])
        query.extend(rows[count:])
    query.sort()
    gallery.sort()
    return query, gallery


@torch.inference_mode()
def hits(
    values: np.ndarray, labels: tuple[str, ...], query: list[int], gallery: list[int]
) -> np.ndarray:
    packed = pack_int8_unit_embeddings(torch.from_numpy(values.copy()))
    codes = packed.codes.float().cuda()
    norm = packed.inverse_norms.float().cuda()
    gallery_rows = torch.tensor(gallery, device="cuda")
    nearest: list[torch.Tensor] = []
    for rows in torch.tensor(query, device="cuda").split(64):
        score = (
            (codes[rows] @ codes[gallery_rows].T) * norm[rows, None] * norm[gallery_rows][None, :]
        )
        nearest.append(gallery_rows[torch.argmax(score, dim=1)].cpu())
    chosen = torch.cat(nearest).tolist()
    return np.asarray(
        [labels[row] == labels[other] for row, other in zip(query, chosen, strict=True)],
        dtype=np.int8,
    )


def bootstrap(delta: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    classes, inverse = np.unique(labels, return_inverse=True)
    counts = np.bincount(inverse)
    totals = np.bincount(inverse, weights=delta, minlength=len(classes))
    rng = np.random.default_rng(179019)
    draws = np.empty(5_000)
    for start in range(0, len(draws), 512):
        stop = min(start + 512, len(draws))
        picked = rng.integers(0, len(classes), size=(stop - start, len(classes)))
        draws[start:stop] = totals[picked].sum(axis=1) / counts[picked].sum(axis=1)
    return tuple(float(v) for v in np.quantile(draws, [0.025, 0.975]))


def main() -> None:
    if sys.argv[1:] != ["--execute"] or (RUN / "pair-score.json").exists():
        raise ValueError("pair score execution differs")
    receipts = {
        arm: json.loads((RUN / f"{arm}-100/receipt.json").read_text())
        for arm in ("freeze_emb", "freeze_emb_impostor")
    }
    control, treatment = receipts.values()
    common = (
        "seed",
        "updates",
        "schedule_sha256",
        "fit_rows_sha256",
        "held_rows_sha256",
        "first_input_batch_sha256",
        "preflight_sha256",
        "features_sha256",
        "model_file_sha256",
        "source_files_sha256",
        "frozen_encoder_blocks",
        "frozen_embeddings",
        "rank_inactive_steps",
    )
    if (
        any(control[key] != treatment[key] for key in common)
        or control["seed"] != 179026
        or control["updates"] != 100
    ):
        raise ValueError("paired training authority differs")
    train = tuple(row for row in parse_inshop_partition(DATA) if row.split == "train")
    _, held = split(tuple(row.label for row in train))
    if digest_rows(held) != control["held_rows_sha256"]:
        raise ValueError("held product split differs")
    labels = tuple(train[index].label for index in held)
    paths = tuple(train[index].image_path for index in held)
    query, gallery = role_rows(labels, paths)
    if (len(query), len(gallery)) != (6354, 6245):
        raise ValueError("query/gallery counts differ")
    for rows, expected in ((query, QUERY_SHA), (gallery, GALLERY_SHA)):
        if hashlib.sha256(np.asarray(rows, dtype="<i4").tobytes()).hexdigest() != expected:
            raise ValueError("fixed query/gallery roles differ")
    arm_hits = {}
    for arm, receipt in receipts.items():
        path = RUN / f"{arm}-100/held_embeddings.npy"
        values = np.load(path, allow_pickle=False)
        if (
            sha(path) != receipt["held_embeddings_sha256"]
            or values.shape != (12599, 128)
            or values.dtype != np.float32
        ):
            raise ValueError("held embedding artifact differs")
        arm_hits[arm] = hits(values, labels, query, gallery)
    a, b = arm_hits.values()
    delta = b.astype(np.float64) - a
    lower, upper = bootstrap(delta, np.asarray(labels, dtype=object)[query])
    wall_ratio = treatment["training_wall_seconds"] / control["training_wall_seconds"]
    vram_ratio = (
        treatment["training_peak_cuda_allocated_bytes"]
        / control["training_peak_cuda_allocated_bytes"]
    )
    map_delta = treatment["quality"]["map_at_r"] - control["quality"]["map_at_r"]
    result = {
        "schema": "sfora-inshop-impostor-paired-100-train-v1",
        "claim_eligible": False,
        "source_sha256": sha(Path(__file__)),
        "receipt_sha256": {arm: sha(RUN / f"{arm}-100/receipt.json") for arm in receipts},
        "query_sha256": QUERY_SHA,
        "gallery_sha256": GALLERY_SHA,
        "query_rows": len(query),
        "gallery_rows": len(gallery),
        "control_r1": float(a.mean()),
        "treatment_r1": float(b.mean()),
        "r1_delta": float(delta.mean()),
        "r1_delta_product_bootstrap_95": [lower, upper],
        "control_hits": a.tolist(),
        "treatment_hits": b.tolist(),
        "symmetric_map_at_r_delta": map_delta,
        "training_wall_ratio": wall_ratio,
        "training_peak_cuda_ratio": vram_ratio,
        "gate_pass": bool(
            delta.mean() >= 0.003
            and lower > 0
            and map_delta >= -0.002
            and wall_ratio <= 1.05
            and vram_ratio <= 1.05
        ),
    }
    (RUN / "pair-score.json").write_text(json.dumps(result, sort_keys=True, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key not in ("control_hits", "treatment_hits")
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
