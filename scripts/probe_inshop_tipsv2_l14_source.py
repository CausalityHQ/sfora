#!/usr/bin/env python3
"""Frozen TIPSv2-L/14@224 source screen on product-disjoint In-Shop TRAIN."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import resource
import time
from pathlib import Path

import numpy as np
import torch
from compare_sop_siglip2_member_bank_arms import product_bootstrap
from export_sop_siglip2_train import export_features
from preflight_inshop_siglip2_unseen_gallery import PARTITION_SHA, digest_rows, sha256, split
from torch.nn import functional as F
from torchvision import transforms
from train_sop_siglip2_compact import score_packed_full_gallery

from sfora.joint_relational_compaction import pack_int8_unit_embeddings
from sfora.representation_ceiling import fit_centered_pca
from sfora.unicom_inshop import parse_inshop_partition

UPSTREAM_REVISION = "bf10a73765b048edfbfcdf1f7e6ca8292542baae"
UPSTREAM_SOURCE_SHA = "7e8194f1c93d002cdd96c60bdb3ac71443dc669eb172478a6178992c83828c53"
CHECKPOINT_SHA = "46e5ac008f02191020b3fe908a60c2d237268b43a864251af2f5354058d3cd77"
BASELINE_SHA = "943faa33e74a1bbf769e6b39d86482ac024a85f574ff6b829dc7328a034c84d7"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("dataset-root", "upstream-source", "checkpoint", "baseline", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.output_dir.exists()
        or not torch.cuda.is_available()
        or sha256(args.dataset_root / "Eval/list_eval_partition.txt") != PARTITION_SHA
        or sha256(args.upstream_source) != UPSTREAM_SOURCE_SHA
        or sha256(args.checkpoint) != CHECKPOINT_SHA
        or sha256(args.baseline) != BASELINE_SHA
    ):
        raise ValueError("TIPSv2 In-Shop source authority differs")
    baseline = json.loads(args.baseline.read_text())
    rows = tuple(row for row in parse_inshop_partition(args.dataset_root) if row.split == "train")
    labels = tuple(row.label for row in rows)
    fit, held = split(labels)
    if (
        len(rows) != 25_882
        or len(fit) != 13_283
        or len(held) != 12_599
        or digest_rows(fit) != baseline["fit_rows_sha256"]
        or digest_rows(held) != baseline["held_rows_sha256"]
    ):
        raise ValueError("TIPSv2 In-Shop held split differs")
    spec = importlib.util.spec_from_file_location("tipsv2_image_encoder", args.upstream_source)
    if spec is None or spec.loader is None:
        raise ValueError("TIPSv2 upstream source cannot load")
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    torch.set_num_threads(16)
    torch.backends.cuda.matmul.allow_tf32 = False
    weights = np.load(args.checkpoint, allow_pickle=False)
    model = upstream.vit_large(
        img_size=448,
        patch_size=14,
        ffn_layer="mlp",
        block_chunks=0,
        init_values=1.0,
        interpolate_antialias=True,
        interpolate_offset=0.0,
    )
    model.load_state_dict({name: torch.from_numpy(weights[name]) for name in weights.files}, strict=True)
    del weights
    model = model.half().cuda().eval()
    transform = transforms.Compose((transforms.Resize((224, 224)), transforms.ToTensor()))
    from PIL import Image

    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()

    @torch.inference_mode()
    def encode(batch: object) -> np.ndarray:
        pixels = []
        for row in batch:
            with Image.open(row.image_path) as image:
                pixels.append(transform(image.convert("RGB")))
        _, synthetic_caption_cls, _ = model(torch.stack(pixels).cuda().half())
        if synthetic_caption_cls.shape != (len(pixels), 1024):
            raise ValueError("TIPSv2 second CLS geometry differs")
        return synthetic_caption_cls.float().cpu().numpy()

    export_features(rows, encode, args.output_dir / "train_features.npy", width=1024, batch_size=32)
    torch.cuda.synchronize()
    encode_seconds = time.perf_counter() - started
    features = np.load(args.output_dir / "train_features.npy", mmap_mode="r")
    fit_values = F.normalize(torch.from_numpy(np.asarray(features[list(fit)]).copy()), dim=1)
    held_values = F.normalize(torch.from_numpy(np.asarray(features[list(held)]).copy()), dim=1)
    pca = fit_centered_pca(fit_values, dimensions=128)
    packed = pack_int8_unit_embeddings(pca.apply(held_values))
    names = tuple(labels[i] for i in held)
    classes = {name: i for i, name in enumerate(sorted(set(names)))}
    quality = score_packed_full_gallery(
        packed.codes.float(),
        packed.inverse_norms,
        torch.tensor([classes[name] for name in names]),
        torch.arange(len(held)),
        device=torch.device("cuda"),
    )
    old = baseline["quality"]["short22_own_pca"]
    delta = quality["map_at_r"] - old["map_at_r"]
    interval = product_bootstrap(
        np.asarray(quality["per_query_ap"]) - np.asarray(old["per_query_ap"]),
        np.asarray(names),
    )
    peak = torch.cuda.max_memory_allocated()
    report = {
        "schema": "sfora-inshop-tipsv2-l14-source-preflight-v1",
        "claim_eligible": False,
        "source_sha256": sha256(Path(__file__)),
        "upstream_revision": UPSTREAM_REVISION,
        "upstream_source_sha256": UPSTREAM_SOURCE_SHA,
        "checkpoint_sha256": CHECKPOINT_SHA,
        "baseline_sha256": BASELINE_SHA,
        "partition_sha256": PARTITION_SHA,
        "fit_rows_sha256": digest_rows(fit),
        "held_rows_sha256": digest_rows(held),
        "features_sha256": sha256(args.output_dir / "train_features.npy"),
        "pca_components_sha256": hashlib.sha256(pca.components.numpy().tobytes()).hexdigest(),
        "input_pixels": 224,
        "selected_output": "second_synthetic_caption_cls",
        "quality": quality,
        "baseline": {"r1": old["recall_at_1"], "map_at_r": old["map_at_r"]},
        "map_at_r_delta": delta,
        "map_at_r_delta_product_bootstrap": interval,
        "encode_wall_seconds": encode_seconds,
        "total_wall_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": peak,
        "peak_parent_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "advance_training": (
            delta >= 0.005
            and interval["lower_95"] > 0
            and quality["recall_at_1"] >= old["recall_at_1"]
            and encode_seconds <= 216.163
            and peak < 10_000_000_000
        ),
        "hardware": {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__},
    }
    if sha256(Path(__file__)) != report["source_sha256"]:
        raise ValueError("TIPSv2 preflight source changed during execution")
    with (args.output_dir / "receipt.json").open("xb") as stream:
        stream.write((json.dumps(report, sort_keys=True, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"r1": quality["recall_at_1"], "mapr": quality["map_at_r"], "encode_s": encode_seconds, "advance": report["advance_training"]}), flush=True)


if __name__ == "__main__":
    main()
