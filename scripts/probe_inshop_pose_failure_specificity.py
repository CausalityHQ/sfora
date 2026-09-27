#!/usr/bin/env python3
"""Compare same-pose top impostors on TRAIN held misses versus hits."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from export_inshop_siglip2_train_features import MODEL_HASHES
from export_sop_siglip2_train import MODEL_REVISION
from preflight_inshop_siglip2_unseen_gallery import PARTITION_SHA, digest_rows, sha256, split
from train_sop_siglip2_compact import export_all

from sfora.joint_relational_compaction import pack_int8_unit_embeddings
from sfora.unicom_inshop import parse_inshop_partition

RECEIPT_SHA = "76f1293ac158f64f1e98a9412033ca37667a2baaadb592bf6051d92b61c9c8cc"
MISS_RECEIPT_SHA = "8dfe23577547b188ae37aa0b83f564bd1dd8c411edb066f474724ccd2e89667c"
PREFLIGHT_SHA = "4a621d82688d2a6c1bfd78c0c0cd2d3f5d7b810ffcb34f9fbb200e3f8d64a254"
HELPER_SHA = "e2f7f8d16e2850a51aa85a3d3f4e04a80ee2a48681dcac55306fd792c8f19ba6"
QUERY_SHA = "89f1dacd6dd94147578c46b2a6830655a17d1979bf58f021ddd49ecf4549ac68"
GALLERY_SHA = "e7114b2c24bfe9625a47d698729c8c4e09de18b16541e7919e7983baed7290c3"


def correct(
    pos_score: torch.Tensor,
    pos_index: torch.Tensor,
    neg_score: torch.Tensor,
    neg_index: torch.Tensor,
) -> torch.Tensor:
    return (pos_score > neg_score) | ((pos_score == neg_score) & (pos_index < neg_index))


def odds_ratio(hit: np.ndarray, same: np.ndarray, poses: np.ndarray, weights: np.ndarray) -> float:
    """Mantel-Haenszel miss/same-pose odds ratio stratified by query pose."""

    numerator = denominator = 0.0
    for pose in sorted(set(poses) - {"flat"}):
        group = poses == pose
        a = weights[group & ~hit & same].sum()
        b = weights[group & ~hit & ~same].sum()
        c = weights[group & hit & same].sum()
        d = weights[group & hit & ~same].sum()
        total = a + b + c + d
        if total:
            numerator += a * d / total
            denominator += b * c / total
    return numerator / denominator if denominator else float("inf")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("dataset-root", "model-snapshot", "preflight", "receipt", "checkpoint", "miss-receipt", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    helper = sys.modules[export_all.__module__].__file__
    if (
        args.output.exists()
        or sha256(args.receipt) != RECEIPT_SHA
        or sha256(args.miss_receipt) != MISS_RECEIPT_SHA
        or sha256(args.preflight) != PREFLIGHT_SHA
        or sha256(args.dataset_root / "Eval/list_eval_partition.txt") != PARTITION_SHA
        or args.model_snapshot.name != MODEL_REVISION
        or any(
            sha256(args.model_snapshot / name) != digest for name, digest in MODEL_HASHES.items()
        )
        or helper is None
        or sha256(Path(helper)) != HELPER_SHA
        or not torch.cuda.is_available()
    ):
        raise ValueError("In-Shop pose-gap authority differs")
    receipt = json.loads(args.receipt.read_text())
    if sha256(args.checkpoint) != receipt["checkpoint_sha256"]:
        raise ValueError("In-Shop pose-gap checkpoint differs")
    train = tuple(row for row in parse_inshop_partition(args.dataset_root) if row.split == "train")
    _, held = split(tuple(row.label for row in train))
    if digest_rows(held) != json.loads(args.preflight.read_text())["held_sha256"]:
        raise ValueError("In-Shop pose-gap held split differs")
    labels = tuple(train[index].label for index in held)
    paths = tuple(train[index].image_path for index in held)
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        grouped[label].append(index)
    query: list[int] = []
    gallery: list[int] = []
    for label in sorted(grouped):
        rows = sorted(
            grouped[label],
            key=lambda index: hashlib.sha256(
                str(paths[index].relative_to(args.dataset_root)).encode()
            ).digest(),
        )
        count = max(1, min(len(rows) - 1, round(len(rows) / 2)))
        gallery.extend(rows[:count])
        query.extend(rows[count:])
    query.sort()
    gallery.sort()
    query_sha = hashlib.sha256(np.asarray(query, dtype="<i4").tobytes()).hexdigest()
    gallery_sha = hashlib.sha256(np.asarray(gallery, dtype="<i4").tobytes()).hexdigest()
    if query_sha != QUERY_SHA or gallery_sha != GALLERY_SHA:
        raise ValueError("In-Shop pose-gap asymmetric role split differs")
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(  # type: ignore[no-untyped-call]
        args.model_snapshot, local_files_only=True, backend="torchvision"
    )
    full = AutoModel.from_pretrained(
        args.model_snapshot, local_files_only=True, use_safetensors=True, dtype=torch.float16
    )
    vision = full.vision_model.float().cuda().eval()
    del full
    head = torch.nn.Linear(1024, 128).cuda().eval()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    vision.load_state_dict(checkpoint["vision"], strict=True)
    head.load_state_dict(checkpoint["head"], strict=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    started = time.perf_counter()
    values = export_all(
        vision, head, paths, tuple(range(len(held))), processor, workers=4, batch_size=64
    )
    export_wall = time.perf_counter() - started
    packed = pack_int8_unit_embeddings(values)
    codes = packed.codes.float().cuda()
    inverse = packed.inverse_norms.float().cuda()
    classes = {name: i for i, name in enumerate(sorted(grouped))}
    label_ids = torch.tensor([classes[name] for name in labels], device="cuda")
    gallery_mask = torch.zeros(len(held), dtype=torch.bool, device="cuda")
    gallery_mask[gallery] = True
    pose = tuple(path.stem.rsplit("_", 1)[-1] for path in paths)
    full_hits: list[bool] = []
    proxy_hits: list[bool] = []
    same_pose: list[bool] = []
    removed_best: list[bool] = []
    proxy_impostors: list[int] = []
    impostor_same_pose: list[bool] = []
    for rows in torch.tensor(query, device="cuda").split(64):
        score = (codes[rows] @ codes.T) * inverse[rows, None] * inverse[None, :]
        score[torch.arange(len(rows), device="cuda"), rows] = -torch.inf
        positive = label_ids[rows, None] == label_ids[None, :]
        negative = ~positive
        full_p_score, full_p = score.masked_fill(~positive, -torch.inf).max(dim=1)
        full_n_score, full_n = score.masked_fill(~negative, -torch.inf).max(dim=1)
        proxy_p_score, proxy_p = score.masked_fill(~(positive & gallery_mask), -torch.inf).max(
            dim=1
        )
        proxy_n_score, proxy_n = score.masked_fill(~(negative & gallery_mask), -torch.inf).max(
            dim=1
        )
        f = correct(full_p_score, full_p, full_n_score, full_n)
        p = correct(proxy_p_score, proxy_p, proxy_n_score, proxy_n)
        full_hits.extend(f.cpu().tolist())
        proxy_hits.extend(p.cpu().tolist())
        proxy_impostors.extend(proxy_n.cpu().tolist())
        impostor_same_pose.extend(
            pose[q] == pose[j]
            for q, j in zip(rows.cpu().tolist(), proxy_n.cpu().tolist(), strict=True)
        )
        same_pose.extend(
            pose[q] == pose[j]
            for q, j in zip(rows.cpu().tolist(), full_p.cpu().tolist(), strict=True)
        )
        removed_best.extend((~gallery_mask[full_p]).cpu().tolist())
    if full_hits != [bool(receipt["quality"]["per_query_r1"][index]) for index in query]:
        raise ValueError("In-Shop pose-gap full-gallery packed parity differs")
    induced = [i for i, (f, p) in enumerate(zip(full_hits, proxy_hits, strict=True)) if f and not p]
    if not all(removed_best[i] for i in induced):
        raise ValueError("In-Shop pose-gap subset monotonicity differs")
    miss_receipt = json.loads(args.miss_receipt.read_text())
    measured_misses = [
        (q, j) for q, j, hit in zip(query, proxy_impostors, proxy_hits, strict=True) if not hit
    ]
    archived_misses = [
        (int(row["query_held_row"]), int(row["best_impostor_held_row"]))
        for row in miss_receipt["misses"]
    ]
    if measured_misses != archived_misses or len(measured_misses) != 151:
        raise ValueError("In-Shop top-impostor replay differs")
    hits = np.asarray(proxy_hits, dtype=bool)
    same = np.asarray(impostor_same_pose, dtype=bool)
    query_pose = np.asarray([pose[i] for i in query])
    product_names = sorted({labels[i] for i in query})
    product_id = np.asarray([product_names.index(labels[i]) for i in query], dtype=np.int32)
    unit_weights = np.ones(len(query), dtype=np.int32)
    observed_or = odds_ratio(hits, same, query_pose, unit_weights)
    rng = np.random.default_rng(179026)
    bootstrap = []
    for _ in range(2000):
        sampled = rng.integers(0, len(product_names), size=len(product_names))
        product_weights = np.bincount(sampled, minlength=len(product_names))
        bootstrap.append(odds_ratio(hits, same, query_pose, product_weights[product_id]))
    lower, upper = np.quantile(bootstrap, [0.025, 0.975]).tolist()
    by_pose = {}
    for tag in sorted(set(query_pose)):
        group = query_pose == tag
        by_pose[tag] = {
            "miss_same": int((group & ~hits & same).sum()),
            "miss_other": int((group & ~hits & ~same).sum()),
            "hit_same": int((group & hits & same).sum()),
            "hit_other": int((group & hits & ~same).sum()),
        }
    report = {
        "schema": "sfora-inshop-pose-failure-specificity-train-only-v1",
        "claim_eligible": False,
        "source_sha256": sha256(Path(__file__)),
        "receipt_sha256": RECEIPT_SHA,
        "checkpoint_sha256": receipt["checkpoint_sha256"],
        "miss_receipt_sha256": MISS_RECEIPT_SHA,
        "query_rows": len(query),
        "gallery_rows": len(gallery),
        "full_r1": sum(full_hits) / len(query),
        "proxy_r1": sum(proxy_hits) / len(query),
        "induced_misses": len(induced),
        "rescued_misses": sum(not f and p for f, p in zip(full_hits, proxy_hits, strict=True)),
        "induced_with_same_pose_best_positive_removed": sum(same_pose[i] for i in induced),
        "same_pose_fraction_of_induced": (
            sum(same_pose[i] for i in induced) / len(induced) if induced else 0.0
        ),
        "miss_same_pose_top_impostors": int((~hits & same).sum()),
        "hit_same_pose_top_impostors": int((hits & same).sum()),
        "flat_excluded_pose_stratified_miss_odds_ratio": observed_or,
        "product_cluster_bootstrap_95_interval": [lower, upper],
        "advance_to_training": bool(lower > 1.3),
        "by_pose": by_pose,
        "export_wall_seconds": export_wall,
    }
    with args.output.open("xb") as stream:
        stream.write((json.dumps(report, sort_keys=True, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    assert odds_ratio(
        np.array([False, False, True, True]),
        np.array([True, False, True, False]),
        np.array(["front"] * 4),
        np.ones(4),
    ) == 1.0
    main()
