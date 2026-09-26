#!/usr/bin/env python3
"""TRAIN-only kill tests for longer training and lower-block freezing."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path

import torch
from export_inshop_siglip2_train_features import MODEL_HASHES
from export_sop_siglip2_train import MODEL_REVISION
from preflight_inshop_siglip2_coverage import PARTITION_SHA256, fit_and_holdout
from torch import nn
from train_sop_siglip2_compact import export_all, score_packed_full_gallery, sha256

from sfora.joint_relational_compaction import pack_int8_unit_embeddings
from sfora.unicom_inshop import parse_inshop_partition

TRAIN_RECEIPT_SHA = "1df5be4c1789041d1fa2868d877a5471f6acdbf3a7dcef5b867a59a04c6d2c39"


def block_drift(
    pretrained: dict[str, torch.Tensor], trained: dict[str, torch.Tensor]
) -> dict[str, float | list[float]]:
    values = []
    for block in range(24):
        prefix = f"encoder.layers.{block}."
        names = [name for name in pretrained if name.startswith(prefix)]
        if not names or any(name not in trained for name in names):
            raise ValueError(f"In-Shop encoder block {block} differs")
        numerator = sum(
            float(torch.sum((trained[name].float() - pretrained[name].float()) ** 2))
            for name in names
        )
        denominator = sum(float(torch.sum(pretrained[name].float() ** 2)) for name in names)
        values.append((numerator / denominator) ** 0.5)
    return {
        "relative_l2_per_block": values,
        "lower_0_11_mean": sum(values[:12]) / 12,
        "upper_12_23_mean": sum(values[12:]) / 12,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt_path = args.training_dir / "receipt.json"
    checkpoint_path = args.training_dir / "checkpoint.pt"
    if (
        args.output.exists()
        or not torch.cuda.is_available()
        or sha256(receipt_path) != TRAIN_RECEIPT_SHA
        or sha256(args.dataset_root / "Eval/list_eval_partition.txt") != PARTITION_SHA256
        or args.model_snapshot.resolve().name != MODEL_REVISION
        or any(
            sha256(args.model_snapshot / name) != digest for name, digest in MODEL_HASHES.items()
        )
    ):
        raise ValueError("In-Shop fit saturation authority differs")
    receipt = json.loads(receipt_path.read_text())
    if receipt["checkpoint_sha256"] != sha256(checkpoint_path):
        raise ValueError("In-Shop fit saturation checkpoint differs")
    train = tuple(row for row in parse_inshop_partition(args.dataset_root) if row.split == "train")
    labels = tuple(row.label for row in train)
    fit, _ = fit_and_holdout(labels)
    fit_labels = tuple(labels[index] for index in fit)
    counts = Counter(fit_labels)
    queries = tuple(i for i, name in enumerate(fit_labels) if counts[name] > 1)
    if len(fit) != 23_342 or len(queries) != 23_330:
        raise ValueError("In-Shop fit saturation inventory differs")
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(  # type: ignore[no-untyped-call]
        args.model_snapshot, local_files_only=True, backend="torchvision"
    )
    full = AutoModel.from_pretrained(
        args.model_snapshot, local_files_only=True, use_safetensors=True, dtype=torch.float16
    )
    vision = full.vision_model.float().cuda().eval()
    del full
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    drift = block_drift(
        {name: value.cpu() for name, value in vision.state_dict().items()}, checkpoint["vision"]
    )
    vision.load_state_dict(checkpoint["vision"], strict=True)
    head = nn.Linear(1024, 128).cuda().eval()
    head.load_state_dict(checkpoint["head"], strict=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    started = time.perf_counter()
    values = export_all(
        vision,
        head,
        tuple(train[index].image_path for index in fit),
        tuple(range(len(fit))),
        processor,
        workers=4,
        batch_size=64,
    )
    packed = pack_int8_unit_embeddings(values)
    encoded = {name: index for index, name in enumerate(sorted(set(fit_labels)))}
    label_ids = torch.tensor([encoded[name] for name in fit_labels], dtype=torch.int64)
    quality = score_packed_full_gallery(
        packed.codes.float(),
        packed.inverse_norms,
        label_ids,
        torch.tensor(queries),
        device=torch.device("cuda"),
    )
    report = {
        "schema": "sfora-inshop-siglip2-fit-saturation-v1",
        "claim_eligible": False,
        "split": "official TRAIN fit rows; non-singleton queries, full fit gallery, self excluded",
        "fit_gallery": len(fit),
        "fit_queries": len(queries),
        "seed": receipt["seed"],
        "arm": receipt["arm"],
        "training_receipt_sha256": TRAIN_RECEIPT_SHA,
        "checkpoint_sha256": receipt["checkpoint_sha256"],
        "source_sha256": sha256(Path(__file__)),
        "fit_packed_quality": quality,
        "drift": drift,
        "export_and_score_seconds": time.perf_counter() - started,
    }
    with args.output.open("xb") as stream:
        stream.write((json.dumps(report, sort_keys=True, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    print(
        json.dumps(
            {
                "fit_r1": quality["recall_at_1"],
                "lower_drift": drift["lower_0_11_mean"],
                "upper_drift": drift["upper_12_23_mean"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
