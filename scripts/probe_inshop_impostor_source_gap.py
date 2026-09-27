#!/usr/bin/env python3
"""Census hard impostors on the pinned In-Shop TRAIN held gallery."""

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
import probe_inshop_expected_gallery_recall as prior
import torch
from torch.nn import functional as F

import sfora.joint_relational_compaction as packing_module
import sfora.unicom_inshop as partition_module
from sfora.joint_relational_compaction import pack_int8_unit_embeddings
from sfora.unicom_inshop import parse_inshop_partition

PRIOR_SOURCE_SHA = "6f0d3c76ca91d34c734c57acb23db8b9c4279957302443ad20c11cf1263e5106"
PRIOR_RESULT_SHA = "d48e2c382fcfb7aa4807b00cc8b2a76fe098aebfa77334dbc2fd8402d36b46a4"
SOURCE_CACHE_RECEIPT_SHA = "da3a2cdfb3d4fc90bfb562c15c9eb0e3b1747b6547883501b596f00ef8ec6e1b"


def census(margins: list[float], cosines: list[float]) -> dict[str, float | bool]:
    if not margins or len(margins) != len(cosines):
        raise ValueError("impostor census rows differ")
    close = sum(-0.05 <= value <= 0 for value in margins) / len(margins)
    near_duplicate = sum(value >= 0.97 for value in cosines) / len(cosines)
    return {
        "close_margin_fraction": close,
        "source_cosine_ge_097_fraction": near_duplicate,
        "reopen_specific_impostor_method": close >= 0.40 and near_duplicate <= 0.25,
    }


def main() -> None:
    if sys.argv[1:] == ["--self-test"]:
        assert census([-0.01, -0.08], [0.5, 0.98]) == {
            "close_margin_fraction": 0.5,
            "source_cosine_ge_097_fraction": 0.5,
            "reopen_specific_impostor_method": False,
        }
        print("impostor census self-test passed")
        return
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in (
        "dataset-root",
        "model-snapshot",
        "preflight",
        "receipt",
        "checkpoint",
        "prior-result",
        "source-cache-receipt",
        "source-cache",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.output.exists()
        or prior.sha256(Path(prior.__file__)) != PRIOR_SOURCE_SHA
        or prior.sha256(args.prior_result) != PRIOR_RESULT_SHA
        or prior.sha256(args.source_cache_receipt) != SOURCE_CACHE_RECEIPT_SHA
        or prior.sha256(Path(sys.modules[prior.export_all.__module__].__file__)) != prior.HELPER_SHA
        or prior.sha256(Path(packing_module.__file__)) != prior.PACKING_SHA
        or prior.sha256(Path(partition_module.__file__)) != prior.PARSER_SHA
        or prior.sha256(args.receipt) != prior.RECEIPT_SHA
        or prior.sha256(args.preflight) != prior.PREFLIGHT_SHA
        or prior.sha256(args.dataset_root / "Eval/list_eval_partition.txt") != prior.PARTITION_SHA
        or args.model_snapshot.name != prior.MODEL_REVISION
        or any(
            prior.sha256(args.model_snapshot / name) != digest
            for name, digest in prior.MODEL_HASHES.items()
        )
        or not torch.cuda.is_available()
    ):
        raise ValueError("In-Shop impostor census authority differs")
    receipt = json.loads(args.receipt.read_text())
    source_receipt = json.loads(args.source_cache_receipt.read_text())
    prior_result = json.loads(args.prior_result.read_text())
    if (
        prior.sha256(args.checkpoint) != receipt["checkpoint_sha256"]
        or prior.sha256(args.source_cache) != source_receipt["features_sha256"]
        or source_receipt["partition_sha256"] != prior.PARTITION_SHA
        or source_receipt["model_file_sha256"] != prior.MODEL_HASHES
        or prior_result["checkpoint_sha256"] != receipt["checkpoint_sha256"]
        or prior_result["query_rows"] != 6354
        or prior_result["gallery_rows"] != 6245
    ):
        raise ValueError("In-Shop impostor census inputs differ")
    train = tuple(row for row in parse_inshop_partition(args.dataset_root) if row.split == "train")
    labels = tuple(row.label for row in train)
    _, held = prior.split(labels)
    if prior.digest_rows(held) != json.loads(args.preflight.read_text())["held_sha256"]:
        raise ValueError("In-Shop impostor census held split differs")
    ordered_sha = hashlib.sha256(
        "\n".join(
            f"{row.label}\0{row.image_path.relative_to(args.dataset_root)}" for row in train
        ).encode()
    ).hexdigest()
    if source_receipt["ordered_rows_sha256"] != ordered_sha:
        raise ValueError("In-Shop source cache row order differs")
    paths = tuple(train[row].image_path for row in held)
    if [prior.sha256(path) for path in paths] != prior_result["held_image_sha256"]:
        raise ValueError("In-Shop held image bytes differ from prior export")
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(held):
        grouped[labels[row]].append(index)
    query: list[int] = []
    gallery: list[int] = []
    for label in sorted(grouped):
        ordered = sorted(
            grouped[label],
            key=lambda index: hashlib.sha256(
                str(paths[index].relative_to(args.dataset_root)).encode()
            ).digest(),
        )
        count = max(1, min(len(ordered) - 1, round(len(ordered) / 2)))
        gallery.extend(ordered[:count])
        query.extend(ordered[count:])
    query.sort()
    gallery.sort()
    if (
        hashlib.sha256(np.asarray(query, dtype="<i4").tobytes()).hexdigest() != prior.QUERY_SHA
        or hashlib.sha256(np.asarray(gallery, dtype="<i4").tobytes()).hexdigest()
        != prior.GALLERY_SHA
    ):
        raise ValueError("In-Shop impostor census role split differs")
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(
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
    values = prior.export_all(
        vision, head, paths, tuple(range(len(held))), processor, workers=4, batch_size=64
    )
    export_wall = time.perf_counter() - started
    packed = pack_int8_unit_embeddings(values)
    codes = packed.codes.float().cuda()
    inverse = packed.inverse_norms.float().cuda()
    classes = {name: i for i, name in enumerate(sorted(grouped))}
    label_ids = torch.tensor([classes[labels[row]] for row in held], device="cuda")
    gallery_mask = torch.zeros(len(held), dtype=torch.bool, device="cuda")
    gallery_mask[gallery] = True
    misses = []
    hits = []
    for rows in torch.tensor(query, device="cuda").split(64):
        scores = (codes[rows] @ codes.T) * inverse[rows, None] * inverse[None, :]
        scores[torch.arange(len(rows), device="cuda"), rows] = -torch.inf
        positive = label_ids[rows, None] == label_ids[None, :]
        best_p_score, best_p = scores.masked_fill(~(positive & gallery_mask), -torch.inf).max(dim=1)
        best_n_score, best_n = scores.masked_fill(positive | ~gallery_mask, -torch.inf).max(dim=1)
        correct = prior.correct(best_p_score, best_p, best_n_score, best_n)
        hits.extend(correct.cpu().tolist())
        for position in torch.where(~correct)[0].tolist():
            misses.append(
                {
                    "query_held_row": int(rows[position]),
                    "best_positive_held_row": int(best_p[position]),
                    "best_impostor_held_row": int(best_n[position]),
                    "packed_margin": float(best_p_score[position] - best_n_score[position]),
                }
            )
    if hits != prior_result["actual_proxy_per_query_r1"] or len(misses) != 151:
        raise ValueError("In-Shop impostor census asymmetric packed parity differs")
    source = np.load(args.source_cache, mmap_mode="r", allow_pickle=False)
    if source.shape != (len(train), 1024) or source.dtype != np.float32:
        raise ValueError("In-Shop source cache geometry differs")
    for miss in misses:
        query_source = torch.from_numpy(np.asarray(source[held[miss["query_held_row"]]]).copy())
        impostor_source = torch.from_numpy(
            np.asarray(source[held[miss["best_impostor_held_row"]]]).copy()
        )
        miss["source_cosine_to_impostor"] = float(
            (
                F.normalize(query_source.float(), dim=0)
                * F.normalize(impostor_source.float(), dim=0)
            ).sum()
        )
    result = census(
        [miss["packed_margin"] for miss in misses],
        [miss["source_cosine_to_impostor"] for miss in misses],
    )
    report = {
        "schema": "sfora-inshop-impostor-source-gap-train-only-v1",
        "claim_eligible": False,
        "source_sha256": prior.sha256(Path(__file__)),
        "prior_source_sha256": PRIOR_SOURCE_SHA,
        "prior_result_sha256": PRIOR_RESULT_SHA,
        "source_cache_receipt_sha256": SOURCE_CACHE_RECEIPT_SHA,
        "checkpoint_sha256": receipt["checkpoint_sha256"],
        "query_rows": len(query),
        "gallery_rows": len(gallery),
        "misses": misses,
        "miss_count": len(misses),
        "export_wall_seconds": export_wall,
        **result,
    }
    with args.output.open("xb") as stream:
        stream.write((json.dumps(report, sort_keys=True, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    print(
        json.dumps(
            {key: report[key] for key in (*result, "miss_count", "export_wall_seconds")},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
