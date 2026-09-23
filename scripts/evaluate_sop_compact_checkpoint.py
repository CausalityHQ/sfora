#!/usr/bin/env python3
"""Score one frozen compact checkpoint on the official SOP test inventory."""

from __future__ import annotations

import argparse
import json
import math
import platform
import resource
import sys
import tempfile
import time
from pathlib import Path

import torch
from export_unicom_sop_embeddings import ordered_record_sha256
from sop_teacher_anchored_runtime import load_authenticated_source_model
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from train_sop_compact_backbone import (
    CHECKPOINT_SHA256,
    EVAL_BATCH_SIZE,
    IndexedImages,
    parse_sop_records,
    publish_file_noreplace,
    sha256,
    source_manifest,
)

from sfora.joint_relational_compaction import pack_int8_unit_embeddings
from sfora.sop_compact_training import CompactTrainingArm, compact_head_features
from sfora.sop_evaluation import score_symmetric

EXPECTED_SOP_RECORD_SHA256 = "ea323eb87568d3f6ab88372ca5d1bf9a7033811cc958f899c532886f526fe992"


def validate_training_receipt(
    receipt: dict[str, object], checkpoint_sha256: str
) -> tuple[str, int, int]:
    """Bind a frozen checkpoint to the train-only selection run."""

    inputs = receipt.get("inputs")
    arm = receipt.get("arm")
    seed = receipt.get("seed")
    updates = receipt.get("updates")
    if (
        receipt.get("schema") != "sfora-sop-compact-full-backbone-v1"
        or arm not in tuple(item.value for item in CompactTrainingArm)
        or type(seed) is not int
        or seed < 0
        or type(updates) is not int
        or updates < 1
        or receipt.get("fit_images") != 53_700
        or receipt.get("validation_images") != 5_851
        or not isinstance(inputs, dict)
        or inputs.get("checkpoint_output_sha256") != checkpoint_sha256
        or inputs.get("checkpoint_sha256") != CHECKPOINT_SHA256
        or inputs.get("source_sha256") != source_manifest()
    ):
        raise ValueError("SOP compact training receipt differs")
    return arm, seed, updates


def select_training_arm(receipts: list[dict[str, object]]) -> str:
    """Select once by train-holdout packed mAP, Recall@1, then arm name."""

    if type(receipts) is not list or len(receipts) != len(CompactTrainingArm):
        raise ValueError("SOP compact selection receipts differ")
    for receipt in receipts:
        if not isinstance(receipt, dict) or not isinstance(receipt.get("inputs"), dict):
            raise ValueError("SOP compact selection receipts differ")
        digest = receipt["inputs"].get("checkpoint_output_sha256")
        if type(digest) is not str:
            raise ValueError("SOP compact selection receipts differ")
        validate_training_receipt(receipt, digest)
    if {receipt["arm"] for receipt in receipts} != {
        arm.value for arm in CompactTrainingArm
    }:
        raise ValueError("SOP compact selection receipts differ")
    shared = (
        "seed", "updates", "batch_size", "images_per_identity", "split_seed",
        "fit_images", "fit_classes", "validation_images", "validation_classes",
        "schedule_sha256", "fit_row_indexes_sha256", "validation_row_indexes_sha256",
        "validation_image_ids", "validation_labels", "rank_coefficient",
    )
    shared_inputs = (
        "checkpoint_sha256", "features_archive_sha256", "sop_train_metadata_sha256",
        "source_sha256", "initial_head_sha256", "initial_classifier_sha256",
    )
    baseline = receipts[0]
    for receipt in receipts[1:]:
        if any(receipt.get(key) != baseline.get(key) for key in shared) or any(
            receipt["inputs"].get(key) != baseline["inputs"].get(key)
            for key in shared_inputs
        ):
            raise ValueError("SOP compact selection receipts differ")
    metrics: dict[str, tuple[float, float]] = {}
    for receipt in receipts:
        try:
            packed = receipt["validation"]["packed"]
            value = packed["map_at_r"]
            recall = packed["recall_at_1"]
        except (KeyError, TypeError):
            raise ValueError("SOP compact selection receipts differ") from None
        if (
            type(value) not in (int, float)
            or type(recall) not in (int, float)
            or not math.isfinite(value)
            or not math.isfinite(recall)
            or not 0.0 <= value <= 1.0
            or not 0.0 <= recall <= 1.0
        ):
            raise ValueError("SOP compact selection receipts differ")
        metrics[receipt["arm"]] = (float(value), float(recall))
    return min(metrics, key=lambda arm: (-metrics[arm][0], -metrics[arm][1], arm))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--unicom-checkout", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--trained-checkpoint", type=Path, required=True)
    parser.add_argument("--training-receipt", type=Path, required=True)
    parser.add_argument("--selection-receipts", type=Path, nargs=3, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--execute-official-sop-evaluation", action="store_true", required=True)
    args = parser.parse_args()
    if args.workers < 0 or args.output.exists():
        parser.error("SOP compact official evaluation invocation differs")
    return args


def main() -> None:
    args = parse_args()
    initial_source_manifest = source_manifest()
    initial_script_sha256 = sha256(Path(__file__))
    if not torch.cuda.is_available():
        raise ValueError("SOP compact official evaluation requires CUDA")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryFile(dir=args.output.parent):
        pass
    if sha256(args.source_checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("SOP compact source checkpoint differs")
    trained_sha256 = sha256(args.trained_checkpoint)
    receipt_sha256 = sha256(args.training_receipt)
    training_receipt = json.loads(args.training_receipt.read_text())
    arm, seed, updates = validate_training_receipt(training_receipt, trained_sha256)
    selection = [json.loads(path.read_text()) for path in args.selection_receipts]
    selected_arm = select_training_arm(selection)
    selection_sha256 = {
        receipt["arm"]: sha256(path)
        for receipt, path in zip(selection, args.selection_receipts, strict=True)
    }
    if arm != selected_arm or selection_sha256[arm] != receipt_sha256:
        raise ValueError("SOP compact selected arm differs")
    if (
        sha256(args.dataset_root / "Ebay_train.txt")
        != training_receipt["inputs"]["sop_train_metadata_sha256"]
    ):
        raise ValueError("SOP compact training metadata differs")
    trained = torch.load(args.trained_checkpoint, map_location="cpu", weights_only=True)
    if (
        not isinstance(trained, dict)
        or trained.get("arm") != arm
        or trained.get("seed") != seed
        or trained.get("updates") != updates
        or trained.get("schedule_sha256") != training_receipt.get("schedule_sha256")
    ):
        raise ValueError("SOP compact trained checkpoint differs")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(16)
    authenticated = load_authenticated_source_model(args.unicom_checkout, args.source_checkpoint)
    model = authenticated.encoder
    model.load_state_dict(trained["model"], strict=True)
    model = model.cuda().eval()
    head = nn.Linear(768, 128)
    head.load_state_dict(trained["head"], strict=True)
    head = head.cuda().eval()
    all_records = parse_sop_records(args.dataset_root)
    if ordered_record_sha256(all_records) != EXPECTED_SOP_RECORD_SHA256:
        raise ValueError("SOP compact official image inventory differs")
    records = tuple(record for record in all_records if record.split == "test")
    labels = tuple(record.label for record in records)
    if len(records) != 60_502 or len(set(labels)) != 11_316:
        raise ValueError("SOP compact official test inventory differs")
    loader = DataLoader(
        IndexedImages(
            tuple(record.image_path for record in records), labels, authenticated.transform
        ),
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    outputs: list[torch.Tensor] = []
    with torch.inference_mode():
        for images, _ in loader:
            source = model(images.cuda(non_blocking=True))
            features = compact_head_features(source, head)
            outputs.append(F.normalize(features, dim=1).cpu())
    encoded_seconds = time.perf_counter() - started
    values = torch.cat(outputs).contiguous()
    if values.shape != (60_502, 128):
        raise ValueError("SOP compact official feature inventory differs")
    packed = pack_int8_unit_embeddings(values)
    packed_seconds = time.perf_counter() - started - encoded_seconds
    label_tensor = torch.tensor(labels, dtype=torch.int64, device="cuda")
    float_started = time.perf_counter()
    float_score = score_symmetric(values.cuda(), label_tensor)
    float_seconds = time.perf_counter() - float_started
    packed_started = time.perf_counter()
    packed_score = score_symmetric(
        packed.codes.float().cuda(),
        label_tensor,
        inverse_norms=packed.inverse_norms.cuda(),
    )
    packed_score_seconds = time.perf_counter() - packed_started
    result = {
        "schema": "sfora-sop-compact-official-test-v1",
        "claim_eligible": False,
        "selection": "frozen train-identity holdout; official test evaluated after arm selection",
        "selection_rule": (
            "highest train-holdout packed mAP@R; then Recall@1; then lexical arm name"
        ),
        "selection_scores": {
            receipt["arm"]: {
                "packed_map_at_r": receipt["validation"]["packed"]["map_at_r"],
                "packed_recall_at_1": receipt["validation"]["packed"]["recall_at_1"],
            }
            for receipt in selection
        },
        "arm": arm,
        "seed": seed,
        "updates": updates,
        "test_images": len(records),
        "test_classes": len(set(labels)),
        "test_image_ids": [record.image_id for record in records],
        "test_labels": labels,
        "float": float_score,
        "packed": packed_score,
        "gallery_bytes_per_item": 130,
        "encoded_seconds": encoded_seconds,
        "packed_seconds": packed_seconds,
        "float_score_seconds": float_seconds,
        "packed_score_seconds": packed_score_seconds,
        "total_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_host_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "hardware": {
            "gpu": torch.cuda.get_device_name(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "inputs": {
            "source_checkpoint_sha256": CHECKPOINT_SHA256,
            "trained_checkpoint_sha256": trained_sha256,
            "training_receipt_sha256": receipt_sha256,
            "selection_receipt_sha256": selection_sha256,
            "sop_train_metadata_sha256": training_receipt["inputs"]["sop_train_metadata_sha256"],
            "sop_test_metadata_sha256": sha256(args.dataset_root / "Ebay_test.txt"),
            "ordered_test_records_sha256": ordered_record_sha256(records),
            "script_sha256": initial_script_sha256,
            "source_sha256": initial_source_manifest,
        },
        "argv": sys.argv,
    }
    if (
        sha256(Path(__file__)) != initial_script_sha256
        or source_manifest() != initial_source_manifest
    ):
        raise ValueError("SOP compact official evaluator source changed during run")
    publish_file_noreplace(
        args.output,
        lambda stream: stream.write(
            (json.dumps(result, sort_keys=True, allow_nan=False) + "\n").encode()
        ),
    )
    print(
        json.dumps(
            {
                "arm": arm,
                "float_recall_at_1": float_score["recall_at_1"],
                "packed_recall_at_1": packed_score["recall_at_1"],
                "float_map_at_r": float_score["map_at_r"],
                "packed_map_at_r": packed_score["map_at_r"],
                "receipt": str(args.output),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
