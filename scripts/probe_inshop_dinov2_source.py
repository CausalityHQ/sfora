#!/usr/bin/env python3
"""TRAIN-only native DINOv2-L/224 source feasibility for packed In-Shop retrieval."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch
from export_sop_siglip2_train import export_features
from preflight_inshop_siglip2_unseen_gallery import PARTITION_SHA, digest_rows, sha256, split
from torch.nn import functional as F
from train_sop_siglip2_compact import score_packed_full_gallery

from sfora.joint_relational_compaction import pack_int8_unit_embeddings
from sfora.representation_ceiling import fit_centered_pca
from sfora.unicom_inshop import parse_inshop_partition

MODEL_REVISION = "47b73eefe95e8d44ec3623f8890bd894b6ea2d6c"
MODEL_HASHES = {
    "config.json": "12df51c069a2dc1305e34ba71ef58bc2407ea553b75f4722a1715c1bce3bbed0",
    "preprocessor_config.json": "14e780d86fa1861f8751f868d7f45425b5feb55c38ca26f152ca5097ab30f828",
    "model.safetensors": "399fba97a95f22c36834418bc69373364a99af3a1153da1c0fb31db567c92e23",
}
HELPER_HASHES = {
    "export_sop_siglip2_train.py": (
        "e407b393ea94f1abd2cffffdc9cd61081da82de232e9b8037af7cd7cca32db2c"
    ),
    "preflight_inshop_siglip2_unseen_gallery.py": (
        "b6556df2d2011cb23c6b5ce25458c54679646b1ea7989fc47cb67e6a5142906e"
    ),
    "train_sop_siglip2_compact.py": (
        "ebc0986f112eb8ba72943595ac6ef93f5328c0bed105c2990cb8c70d7b3b0495"
    ),
    "representation_ceiling.py": "1608181a9c7ba18d1ae1016804898551b2bba9ab40fec9bd4cc2eaf27981771c",
    "joint_relational_compaction.py": (
        "4ca0de1b0579ea6165c81e9057e9afe77e6dd4141f0b4a0adb281de25300de67"
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    helper_functions = (
        export_features,
        split,
        score_packed_full_gallery,
        fit_centered_pca,
        pack_int8_unit_embeddings,
    )
    imported = {
        Path(sys.modules[function.__module__].__file__).name: Path(
            sys.modules[function.__module__].__file__
        )
        for function in helper_functions
    }
    if (
        args.output_dir.exists()
        or not torch.cuda.is_available()
        or args.model_snapshot.resolve().name != MODEL_REVISION
        or sha256(args.dataset_root / "Eval/list_eval_partition.txt") != PARTITION_SHA
        or any(
            sha256(args.model_snapshot / name) != digest for name, digest in MODEL_HASHES.items()
        )
        or any(
            name not in imported or sha256(imported[name]) != digest
            for name, digest in HELPER_HASHES.items()
        )
    ):
        raise ValueError("DINOv2 In-Shop source authority differs")
    rows = tuple(row for row in parse_inshop_partition(args.dataset_root) if row.split == "train")
    labels = tuple(row.label for row in rows)
    fit, held = split(labels)
    if len(rows) != 25_882 or len(fit) != 13_283 or len(held) != 12_599:
        raise ValueError("DINOv2 In-Shop split differs")
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel

    torch.set_num_threads(16)
    torch.backends.cuda.matmul.allow_tf32 = False
    processor = AutoImageProcessor.from_pretrained(args.model_snapshot, local_files_only=True)
    model = (
        AutoModel.from_pretrained(
            args.model_snapshot, local_files_only=True, use_safetensors=True, dtype=torch.float16
        )
        .cuda()
        .eval()
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()

    @torch.inference_mode()
    def encode(batch: object) -> np.ndarray:
        images = []
        for row in batch:
            with Image.open(row.image_path) as image:
                images.append(image.convert("RGB"))
        pixels = processor(images=images, return_tensors="pt")["pixel_values"].to(
            device="cuda", dtype=torch.float16
        )
        output = model(pixel_values=pixels)
        features = output.pooler_output
        if features is None:
            raise ValueError("DINOv2 pooler output missing")
        return features.float().cpu().numpy()

    export_features(rows, encode, args.output_dir / "train_features.npy", width=1024, batch_size=32)
    torch.cuda.synchronize()
    encode_seconds = time.perf_counter() - started
    features = np.load(args.output_dir / "train_features.npy", mmap_mode="r")
    fit_values = F.normalize(torch.from_numpy(np.asarray(features[list(fit)]).copy()), dim=1)
    held_values = F.normalize(torch.from_numpy(np.asarray(features[list(held)]).copy()), dim=1)
    transform = fit_centered_pca(fit_values, dimensions=128)
    packed = pack_int8_unit_embeddings(transform.apply(held_values))
    held_labels = tuple(labels[index] for index in held)
    classes = {label: index for index, label in enumerate(sorted(set(held_labels)))}
    quality = score_packed_full_gallery(
        packed.codes.float(),
        packed.inverse_norms,
        torch.tensor([classes[label] for label in held_labels]),
        torch.arange(len(held)),
        device=torch.device("cuda"),
    )
    peak = torch.cuda.max_memory_allocated()
    report = {
        "schema": "sfora-inshop-dinov2-source-preflight-v1",
        "claim_eligible": False,
        "split": "official In-Shop TRAIN; fit products disjoint from held products",
        "fit_rows_sha256": digest_rows(fit),
        "held_rows_sha256": digest_rows(held),
        "partition_sha256": PARTITION_SHA,
        "model_revision": MODEL_REVISION,
        "model_file_sha256": MODEL_HASHES,
        "helper_sha256": HELPER_HASHES,
        "source_sha256": sha256(Path(__file__)),
        "features_sha256": sha256(args.output_dir / "train_features.npy"),
        "pca_components_sha256": hashlib.sha256(transform.components.numpy().tobytes()).hexdigest(),
        "quality": quality,
        "encode_wall_seconds": encode_seconds,
        "peak_cuda_allocated_bytes": peak,
        "peak_parent_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "advance": (
            quality["recall_at_1"] >= 0.836652
            and quality["map_at_r"] >= 0.480749
            and encode_seconds <= 216.163
            and peak < 10_000_000_000
        ),
        "hardware": {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__},
    }
    if sha256(Path(__file__)) != report["source_sha256"]:
        raise ValueError("DINOv2 preflight source changed during execution")
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
                "advance": report["advance"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
