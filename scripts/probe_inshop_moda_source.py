#!/usr/bin/env python3
"""Frozen MODA vision source screen on product-disjoint In-Shop TRAIN."""

from __future__ import annotations

import argparse
import hashlib
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
from train_sop_siglip2_compact import score_packed_full_gallery

from sfora.joint_relational_compaction import pack_int8_unit_embeddings
from sfora.representation_ceiling import fit_centered_pca
from sfora.unicom_inshop import parse_inshop_partition

REVISION = "9f3358c2257e86c35d0525e331e0ff1cc8872911"
MODEL_SHA = "41be1d42956162309c6be661702d9d8e444412395fe32c760f3b19c0c159251d"
INFERENCE_SHA = "53d2f73c5c4c637abdc3e9777094637b2fe92c63d9b1062c05bf602545fc0d11"
BASELINE_SHA = "943faa33e74a1bbf769e6b39d86482ac024a85f574ff6b829dc7328a034c84d7"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("dataset-root", "model-snapshot", "baseline", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.output_dir.exists()
        or not torch.cuda.is_available()
        or sha256(args.dataset_root / "Eval/list_eval_partition.txt") != PARTITION_SHA
        or args.model_snapshot.resolve().name != "moda-fashion-vision-fp16-9f3358c2"
        or sha256(args.model_snapshot / "vision_encoder.safetensors") != MODEL_SHA
        or sha256(args.model_snapshot / "inference.py") != INFERENCE_SHA
        or sha256(args.baseline) != BASELINE_SHA
    ):
        raise ValueError("MODA In-Shop source authority differs")
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
        raise ValueError("MODA In-Shop held split differs")
    from PIL import Image
    import open_clip
    from safetensors.torch import load_file

    torch.set_num_threads(16)
    torch.backends.cuda.matmul.allow_tf32 = False
    if open_clip.__version__ != "3.3.0":
        raise ValueError("MODA OpenCLIP runtime differs")
    model, _, processor = open_clip.create_model_and_transforms(
        "ViT-B-16-SigLIP", pretrained=None
    )
    weights = load_file(str(args.model_snapshot / "vision_encoder.safetensors"))
    model.visual.load_state_dict(
        {key.removeprefix("visual."): value for key, value in weights.items()}, strict=True
    )
    model = model.visual.half().cuda().eval()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()

    @torch.inference_mode()
    def encode(batch: object) -> np.ndarray:
        images = []
        for row in batch:
            with Image.open(row.image_path) as image:
                images.append(image.convert("RGB"))
        pixels = torch.stack([processor(image) for image in images]).cuda().half()
        result = model(pixels)
        if result.shape != (len(images), 768) or not bool(torch.isfinite(result).all()):
            raise ValueError("MODA image embedding differs")
        return result.float().cpu().numpy()

    export_features(rows, encode, args.output_dir / "train_features.npy", width=768, batch_size=32)
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
        "schema": "sfora-inshop-moda-vision-source-preflight-v1",
        "claim_eligible": False,
        "source_sha256": sha256(Path(__file__)),
        "model_revision": REVISION,
        "model_sha256": MODEL_SHA,
        "inference_sha256": INFERENCE_SHA,
        "preprocess": "pinned inference.py: OpenCLIP 3.3.0 default ViT-B-16-SigLIP validation transform",
        "baseline_sha256": BASELINE_SHA,
        "partition_sha256": PARTITION_SHA,
        "fit_rows_sha256": digest_rows(fit),
        "held_rows_sha256": digest_rows(held),
        "features_sha256": sha256(args.output_dir / "train_features.npy"),
        "pca_components_sha256": hashlib.sha256(pca.components.numpy().tobytes()).hexdigest(),
        "input_pixels": 224,
        "selected_output": "visual_768",
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
        raise ValueError("MODA preflight source changed during execution")
    with (args.output_dir / "receipt.json").open("xb") as stream:
        stream.write((json.dumps(report, sort_keys=True, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    print(
        json.dumps(
            {
                "r1": quality["recall_at_1"],
                "mapr": quality["map_at_r"],
                "encode_s": encode_seconds,
                "advance": report["advance_training"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
