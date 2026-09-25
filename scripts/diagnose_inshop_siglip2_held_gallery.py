#!/usr/bin/env python3
"""TRAIN-only diagnostic: score held identities against held identities only."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
from export_inshop_siglip2_train_features import MODEL_HASHES
from export_sop_siglip2_train import MODEL_REVISION
from preflight_inshop_siglip2_coverage import PARTITION_SHA256, fit_and_holdout
from torch import nn
from train_sop_siglip2_compact import export_all, score_packed_full_gallery, sha256

from sfora.joint_relational_compaction import pack_int8_unit_embeddings
from sfora.unicom_inshop import parse_inshop_partition

SEED = 179023
DECISION_SHA = "43036079d17aa6a7010fc70c15716c5b939dd8e82629eb1187763395a386526f"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--run-base", type=Path, required=True)
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.output.exists()
        or not torch.cuda.is_available()
        or sha256(args.decision) != DECISION_SHA
        or sha256(args.dataset_root / "Eval/list_eval_partition.txt") != PARTITION_SHA256
        or args.model_snapshot.resolve().name != MODEL_REVISION
        or any(
            sha256(args.model_snapshot / name) != digest for name, digest in MODEL_HASHES.items()
        )
    ):
        raise ValueError("In-Shop held-gallery diagnostic authority differs")
    decision = json.loads(args.decision.read_text())
    if decision.get("selected_arm") != "bank" or SEED not in decision.get("seeds", ()):
        raise ValueError("In-Shop TRAIN decision differs")
    train = tuple(row for row in parse_inshop_partition(args.dataset_root) if row.split == "train")
    labels = tuple(row.label for row in train)
    _, held = fit_and_holdout(labels)
    if len(train) != 25_882 or len(held) != 2_540:
        raise ValueError("In-Shop held-gallery inventory differs")
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(  # type: ignore[no-untyped-call]
        args.model_snapshot, local_files_only=True, backend="torchvision"
    )
    full = AutoModel.from_pretrained(
        args.model_snapshot, local_files_only=True, use_safetensors=True, dtype=torch.float16
    )
    vision = full.vision_model.float().cuda().eval()
    del full
    head = nn.Linear(1024, 128).cuda().eval()
    torch.backends.cuda.matmul.allow_tf32 = False
    held_labels = tuple(labels[index] for index in held)
    encoded = {name: index for index, name in enumerate(sorted(set(held_labels)))}
    label_ids = torch.tensor([encoded[name] for name in held_labels], dtype=torch.int64)
    rows = {}
    for arm in ("bank", "float"):
        run = args.run_base / f"sfora-inshop-siglip2-{arm}-{SEED}-v1"
        receipt_path = run / "receipt.json"
        receipt = json.loads(receipt_path.read_text())
        checkpoint_path = run / "checkpoint.pt"
        if (
            sha256(receipt_path) != decision["arms"][str(SEED)][arm]["receipt_sha256"]
            or receipt["checkpoint_sha256"] != sha256(checkpoint_path)
            or receipt["updates"] != 1000
            or receipt["seed"] != SEED
            or receipt["arm"] != arm
        ):
            raise ValueError(f"In-Shop {arm} checkpoint authority differs")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        vision.load_state_dict(checkpoint["vision"], strict=True)
        head.load_state_dict(checkpoint["head"], strict=True)
        started = time.perf_counter()
        values = export_all(
            vision,
            head,
            tuple(row.image_path for row in train),
            held,
            processor,
            workers=4,
            batch_size=64,
        )
        packed = pack_int8_unit_embeddings(values)
        quality = score_packed_full_gallery(
            packed.codes.float(),
            packed.inverse_norms,
            label_ids,
            torch.arange(len(held)),
            device=torch.device("cuda"),
        )
        rows[arm] = {
            "training_receipt_sha256": sha256(receipt_path),
            "checkpoint_sha256": sha256(checkpoint_path),
            "full_train_gallery_r1": receipt["quality"]["recall_at_1"],
            "full_train_gallery_map_at_r": receipt["quality"]["map_at_r"],
            "held_only_gallery_quality": quality,
            "export_and_score_seconds": time.perf_counter() - started,
        }
    report = {
        "schema": "sfora-inshop-siglip2-held-only-gallery-diagnostic-v1",
        "claim_eligible": False,
        "cannot_change_prior_selection": True,
        "split": "official TRAIN held identities only; self excluded",
        "seed": SEED,
        "query_and_gallery_images": len(held),
        "held_products": len(encoded),
        "decision_sha256": DECISION_SHA,
        "source_sha256": sha256(Path(__file__)),
        "rows": rows,
        "bank_minus_float_held_only_r1": (
            rows["bank"]["held_only_gallery_quality"]["recall_at_1"]
            - rows["float"]["held_only_gallery_quality"]["recall_at_1"]
        ),
    }
    with args.output.open("xb") as stream:
        stream.write((json.dumps(report, sort_keys=True, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    print(
        json.dumps(
            {"held_only_r1": {a: rows[a]["held_only_gallery_quality"]["recall_at_1"] for a in rows}}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
