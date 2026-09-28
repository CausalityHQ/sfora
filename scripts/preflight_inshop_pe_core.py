#!/usr/bin/env python3
"""Native PE CPU import/config/processor and fit-only inventory; no weights or scores."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from core.vision_encoder.pe import VisionTransformer
from core.vision_encoder.transforms import get_image_transform
from preflight_inshop_siglip2_unseen_gallery import PARTITION_SHA, digest_rows, split
from sfora.joint_relational_compaction import pack_int8_unit_embeddings
from sfora.unicom_inshop import parse_inshop_partition


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--native-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not args.output.exists()
    authority = json.loads((args.native_root / "source-manifest.json").read_text())
    assert all(sha(args.native_root / name) == expected for name, expected in authority.items())
    assert sha(args.dataset_root / "Eval/list_eval_partition.txt") == PARTITION_SHA
    torch.set_num_threads(8)
    torch.manual_seed(179031)
    model = VisionTransformer.from_config("PE-Core-B16-224", pretrained=False).eval()
    assert (model.layers, model.width, model.image_size, model.output_dim) == (12, 768, 224, 1024)
    native = get_image_transform(model.image_size)
    assert torch.equal(native(Image.new("RGB", (31, 17), "black")), torch.full((3, 224, 224), -1.0))
    assert torch.equal(native(Image.new("RGB", (17, 31), "white")), torch.ones(3, 224, 224))
    image = Image.fromarray(np.arange(17 * 31 * 3, dtype=np.uint8).reshape(17, 31, 3))
    pixels = torch.stack([native(image)])
    assert torch.equal(pixels[0], native(image)) and pixels.shape == (1, 3, 224, 224)
    with torch.no_grad():
        values = model(pixels)
        assert values.shape == (1, 1024) and torch.isfinite(values).all()
        packed = pack_int8_unit_embeddings(torch.nn.functional.normalize(values[:, :128], dim=1))
        assert packed.codes.shape == (1, 128)
    # Two identical gallery descriptors must preserve lower ordinal on a tied score.
    scores = torch.tensor([[1.0, 1.0, -1.0]])
    assert torch.argsort(scores, descending=True, stable=True).tolist() == [[0, 1, 2]]
    rows = tuple(row for row in parse_inshop_partition(args.dataset_root) if row.split == "train")
    labels = tuple(row.label for row in rows)
    fit, _ = split(labels)
    assert (
        len(fit) == 13283
        and digest_rows(fit) == "f23783a513f0bce23c4ea6126f8a868ee600fe88a23a5e797778a2a508ca89be"
    )
    counts = Counter(labels[i] for i in fit)
    products = sorted(
        (p for p, n in counts.items() if n >= 2),
        key=lambda p: (hashlib.sha256(b"inshop-pe-core-pilot-v1\0" + p.encode()).digest(), p),
    )[:512]
    selected = [
        [
            i
            for i in sorted(
                (i for i in fit if labels[i] == p),
                key=lambda i: (
                    hashlib.sha256(
                        str(rows[i].image_path.relative_to(args.dataset_root)).encode()
                    ).digest(),
                    i,
                ),
            )[:2]
        ]
        for p in products
    ]
    ordered = [pair[0] for pair in selected] + [pair[1] for pair in selected]
    assert len(products) == 512 and len(ordered) == len(set(ordered)) == 1024
    manifest = [
        {
            "train_row": i,
            "product": labels[i],
            "relative_path": str(rows[i].image_path.relative_to(args.dataset_root)),
            "image_sha256": sha(rows[i].image_path),
        }
        for i in ordered
    ]
    result = {
        "source_sha256": authority,
        "script_sha256": sha(__file__),
        "fit_rows_sha256": digest_rows(fit),
        "image_manifest": manifest,
        "torch": torch.__version__,
        "vision_parameters": sum(p.numel() for p in model.parameters()),
        "claim_eligible": False,
        "weights_loaded": False,
        "quality_read": False,
        "versions": {
            name: importlib.metadata.version(name)
            for name in (
                "torchvision",
                "Pillow",
                "timm",
                "einops",
                "ftfy",
                "wcwidth",
                "huggingface_hub",
            )
        },
    }
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print("PASS native CPU import/config/processor/finite geometry; frozen1024 TRAIN-fit images")


if __name__ == "__main__":
    main()
