#!/usr/bin/env python3
"""Falsify a fixed-negative positive-subset R@1 surrogate on In-Shop TRAIN."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
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
PREFLIGHT_SHA = "4a621d82688d2a6c1bfd78c0c0cd2d3f5d7b810ffcb34f9fbb200e3f8d64a254"
HELPER_SHA = "e2f7f8d16e2850a51aa85a3d3f4e04a80ee2a48681dcac55306fd792c8f19ba6"
QUERY_SHA = "89f1dacd6dd94147578c46b2a6830655a17d1979bf58f021ddd49ecf4549ac68"
GALLERY_SHA = "e7114b2c24bfe9625a47d698729c8c4e09de18b16541e7919e7983baed7290c3"
POSE_RECEIPT_SHA = "a64b428572c1cd94ce24e2d6dd9c5681af9aa971057cb0302fe9b3768b4b4f8a"


def correct(
    pos_score: torch.Tensor,
    pos_index: torch.Tensor,
    neg_score: torch.Tensor,
    neg_index: torch.Tensor,
) -> torch.Tensor:
    return (pos_score > neg_score) | ((pos_score == neg_score) & (pos_index < neg_index))


def fixed_negative_miss(n: int, m: int, k: int) -> float:
    if not 0 <= m <= n or not 1 <= k <= n:
        raise ValueError("In-Shop expected-gallery positive counts differ")
    return math.comb(m, k) / math.comb(n, k) if m >= k else 0.0


def main() -> None:
    if sys.argv[1:] == ["--self-test"]:
        assert fixed_negative_miss(4, 2, 2) == 1 / 6
        assert fixed_negative_miss(4, 1, 2) == 0.0
        assert correct(
            torch.tensor([0.5]), torch.tensor([2]), torch.tensor([0.5]), torch.tensor([3])
        ).item()
        print("expected-gallery self-test passed")
        return
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in (
        "dataset-root",
        "model-snapshot",
        "preflight",
        "receipt",
        "checkpoint",
        "pose-receipt",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    helper = sys.modules[export_all.__module__].__file__
    if (
        args.output.exists()
        or sha256(args.receipt) != RECEIPT_SHA
        or sha256(args.pose_receipt) != POSE_RECEIPT_SHA
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
        raise ValueError("In-Shop expected-gallery authority differs")
    receipt = json.loads(args.receipt.read_text())
    if sha256(args.checkpoint) != receipt["checkpoint_sha256"]:
        raise ValueError("In-Shop expected-gallery checkpoint differs")
    train = tuple(row for row in parse_inshop_partition(args.dataset_root) if row.split == "train")
    _, held = split(tuple(row.label for row in train))
    if digest_rows(held) != json.loads(args.preflight.read_text())["held_sha256"]:
        raise ValueError("In-Shop expected-gallery held split differs")
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
        raise ValueError("In-Shop expected-gallery asymmetric role split differs")
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
    torch.cuda.reset_peak_memory_stats()
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
    gallery_counts = torch.bincount(label_ids[gallery], minlength=len(classes))
    full_hits: list[bool] = []
    proxy_hits: list[bool] = []
    positive_counts: list[int] = []
    below_counts: list[int] = []
    gallery_positive_counts: list[int] = []
    near_fragile = 0
    below_fragile = 0
    score_started = time.perf_counter()
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
        ordinal = torch.arange(len(held), device="cuda")
        above = positive & (
            (score > full_n_score[:, None])
            | ((score == full_n_score[:, None]) & (ordinal[None, :] < full_n[:, None]))
        )
        n = positive.sum(dim=1) - 1
        good = above.sum(dim=1)
        m = n - good
        k = gallery_counts[label_ids[rows]]
        fragile = (good >= 1) & (good <= 2)
        below = positive & ~above & (ordinal[None, :] != rows[:, None])
        near_fragile += int(
            (below & fragile[:, None] & (full_n_score[:, None] - score <= 0.05)).sum()
        )
        below_fragile += int((below & fragile[:, None]).sum())
        positive_counts.extend(n.cpu().tolist())
        below_counts.extend(m.cpu().tolist())
        gallery_positive_counts.extend(k.cpu().tolist())
    score_wall = time.perf_counter() - score_started
    if full_hits != [bool(receipt["quality"]["per_query_r1"][index]) for index in query]:
        raise ValueError("In-Shop expected-gallery full-gallery packed parity differs")
    pose = json.loads(args.pose_receipt.read_text())
    actual_r1 = sum(proxy_hits) / len(query)
    if actual_r1 != pose["proxy_r1"] or sum(full_hits) / len(query) != pose["full_r1"]:
        raise ValueError("In-Shop expected-gallery proxy parity differs")
    miss = [
        fixed_negative_miss(n, m, k)
        for n, m, k in zip(positive_counts, below_counts, gallery_positive_counts, strict=True)
    ]
    predicted_r1 = 1.0 - float(np.mean(miss))
    fragile_rows = sum(1 <= n - m <= 2 for n, m in zip(positive_counts, below_counts, strict=True))
    near_fraction = near_fragile / below_fragile if below_fragile else 0.0
    advance = (
        abs(predicted_r1 - actual_r1) <= 0.004
        and fragile_rows / len(query) >= 0.03
        and near_fraction >= 0.30
    )
    report = {
        "schema": "sfora-inshop-expected-gallery-recall-preflight-v1",
        "claim_eligible": False,
        "advance_other_seeds": advance,
        "source_sha256": sha256(Path(__file__)),
        "receipt_sha256": RECEIPT_SHA,
        "checkpoint_sha256": receipt["checkpoint_sha256"],
        "pose_receipt_sha256": POSE_RECEIPT_SHA,
        "preflight_sha256": PREFLIGHT_SHA,
        "partition_sha256": PARTITION_SHA,
        "query_rows": len(query),
        "gallery_rows": len(gallery),
        "full_r1": sum(full_hits) / len(query),
        "actual_proxy_r1": actual_r1,
        "predicted_fixed_negative_r1": predicted_r1,
        "absolute_prediction_error": abs(predicted_r1 - actual_r1),
        "fragile_query_rows": fragile_rows,
        "fragile_query_fraction": fragile_rows / len(query),
        "near_below_positive_count": near_fragile,
        "below_positive_count_on_fragile_rows": below_fragile,
        "near_fraction_on_fragile_rows": near_fraction,
        "positive_counts": positive_counts,
        "below_impostor_counts": below_counts,
        "gallery_positive_counts": gallery_positive_counts,
        "predicted_miss_probability": miss,
        "actual_proxy_per_query_r1": proxy_hits,
        "export_wall_seconds": export_wall,
        "score_wall_seconds": score_wall,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_parent_host_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    }
    with args.output.open("xb") as stream:
        stream.write((json.dumps(report, sort_keys=True, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "advance_other_seeds",
                    "actual_proxy_r1",
                    "predicted_fixed_negative_r1",
                    "fragile_query_fraction",
                    "near_fraction_on_fragile_rows",
                    "export_wall_seconds",
                )
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
