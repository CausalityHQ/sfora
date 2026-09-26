#!/usr/bin/env python3
"""Score an authenticated UNICOM source on the Sfora In-Shop TRAIN-only split."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from compare_sop_siglip2_member_bank_arms import product_bootstrap
from diagnose_inshop_siglip2_trained_width import float_top1_hits
from preflight_inshop_siglip2_unseen_gallery import PARTITION_SHA, digest_rows, sha256, split
from probe_l14_inshop_train_heads import VerifiedTrainImages
from probe_l14_parallel_fold import load_authenticated_l14
from torch.nn import functional as F
from torch.utils.data import DataLoader
from train_sop_siglip2_compact import score_packed_full_gallery

from sfora.joint_relational_compaction import pack_int8_unit_embeddings
from sfora.representation_ceiling import fit_centered_pca
from sfora.unicom_inshop import parse_inshop_partition

PREFLIGHT_SHA = "d5e22c6a331acbdf3b143b9836597593c2317f9488b99b72f61a4c54baf18eb8"
SFORA_RECEIPT_SHA = "d97a2cd402f29fe81d292880b38ae3fb1fddf9ca162c67b96d9a5b0528eb8f5c"
SFORA_WIDTH_SHA = "9f98ddf5bcced51b20e6f31aa752f61642d4bc7c4352872b15af8566502e65b1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--unicom-checkout", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--sfora-receipt", type=Path, required=True)
    parser.add_argument("--sfora-width", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.output.exists()
        or not torch.cuda.is_available()
        or sha256(args.dataset_root / "Eval/list_eval_partition.txt") != PARTITION_SHA
        or sha256(args.preflight) != PREFLIGHT_SHA
        or sha256(args.sfora_receipt) != SFORA_RECEIPT_SHA
        or sha256(args.sfora_width) != SFORA_WIDTH_SHA
    ):
        raise ValueError("In-Shop UNICOM reference authority differs")
    train = tuple(row for row in parse_inshop_partition(args.dataset_root) if row.split == "train")
    fit, held = split(tuple(row.label for row in train))
    preflight = json.loads(args.preflight.read_text())
    sfora = json.loads(args.sfora_receipt.read_text())
    if (
        len(train) != 25_882
        or preflight.get("fit_sha256") != digest_rows(fit)
        or preflight.get("held_sha256") != digest_rows(held)
        or sfora.get("held_rows_sha256") != digest_rows(held)
    ):
        raise ValueError("In-Shop UNICOM reference split differs")
    torch.set_num_threads(16)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model, transform = load_authenticated_l14(args.unicom_checkout, args.checkpoint)
    model = model.cuda().eval()
    manifest = b"".join(hashlib.sha256(row.image_path.read_bytes()).digest() for row in train)
    loader = DataLoader(
        VerifiedTrainImages(train, transform, manifest),
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    parts = []
    encode_started = time.perf_counter()
    with torch.inference_mode():
        for images, _labels in loader:
            values = model.feature(  # type: ignore[operator]
                model.forward_features(images.cuda(non_blocking=True))  # type: ignore[operator]
            )
            parts.append(values.float().cpu())
    encode_seconds = time.perf_counter() - encode_started
    values = torch.cat(parts).contiguous()
    if values.shape != (len(train), 768) or not bool(torch.isfinite(values).all()):
        raise ValueError("In-Shop UNICOM reference feature geometry differs")
    fit_values = F.normalize(values[list(fit)], dim=1)
    pca = fit_centered_pca(fit_values, dimensions=128)
    held_values = F.normalize(values[list(held)], dim=1)
    packed = pack_int8_unit_embeddings(pca.apply(held_values))
    held_labels = np.asarray([train[index].label for index in held])
    encoded = {name: index for index, name in enumerate(sorted(set(held_labels)))}
    label_ids = torch.tensor([encoded[name] for name in held_labels], dtype=torch.int64)
    full_hits = float_top1_hits(held_values.cuda(), label_ids.cuda())
    packed_quality = score_packed_full_gallery(
        packed.codes.float(),
        packed.inverse_norms,
        label_ids,
        torch.arange(len(held), dtype=torch.int64),
        device=torch.device("cuda"),
    )
    sfora_hits = np.asarray(sfora["quality"]["per_query_r1"], dtype=np.float64)
    delta = product_bootstrap(
        np.asarray(packed_quality["per_query_r1"], dtype=np.float64) - sfora_hits,
        held_labels,
    )
    report = {
        "schema": "sfora-inshop-unicom-reference-train-only-v1",
        "claim_eligible": False,
        "split": "official TRAIN product-disjoint held-only symmetric gallery",
        "fit_images": len(fit),
        "held_queries_and_gallery": len(held),
        "unicom_full768_float_r1": float(full_hits.mean()),
        "unicom_pca128_packed_r1": packed_quality["recall_at_1"],
        "unicom_pca128_packed_map_at_r": packed_quality["map_at_r"],
        "sfora_trained128_packed_r1": sfora["quality"]["recall_at_1"],
        "sfora_trained128_packed_map_at_r": sfora["quality"]["map_at_r"],
        "unicom_packed_minus_sfora_packed_product_bootstrap": delta,
        "unicom_full_per_query_r1": full_hits.tolist(),
        "unicom_packed_per_query_r1": packed_quality["per_query_r1"],
        "unicom_packed_per_query_ap": packed_quality["per_query_ap"],
        "fit_source_features_sha256": hashlib.sha256(
            values[list(fit)].numpy().tobytes()
        ).hexdigest(),
        "held_image_manifest_sha256": hashlib.sha256(manifest).hexdigest(),
        "checkpoint_sha256": sha256(args.checkpoint),
        "preflight_sha256": PREFLIGHT_SHA,
        "sfora_receipt_sha256": SFORA_RECEIPT_SHA,
        "sfora_width_receipt_sha256": SFORA_WIDTH_SHA,
        "source_sha256": sha256(Path(__file__)),
        "encode_seconds": encode_seconds,
        "wall_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "gpu": torch.cuda.get_device_name(),
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
                    "unicom_full768_float_r1",
                    "unicom_pca128_packed_r1",
                    "unicom_pca128_packed_map_at_r",
                    "unicom_packed_minus_sfora_packed_product_bootstrap",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
