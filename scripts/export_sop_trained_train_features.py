"""Export one authenticated SOP training checkpoint's deterministic train features.

This is a train-only diagnostic source. It never reads official SOP test images.
Run after other DGX GPU jobs finish, then score the feature archive offline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import subprocess
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
from export_unicom_sop_embeddings import ordered_record_sha256, parse_sop_records
from sop_teacher_anchored_runtime import load_authenticated_source_model
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from train_sop_compact_backbone import (
    ARCHIVE_SHA256,
    CHECKPOINT_SHA256,
    EVAL_BATCH_SIZE,
    FIT_FRACTION,
    SPLIT_SEED,
    IndexedImages,
    publish_file_noreplace,
    score_validation_features,
    sha256,
)

from sfora.representation_ceiling import deterministic_class_partition
from sfora.sop_compact_training import compact_head_features

EXPECTED_SOP_RECORD_SHA256 = "ea323eb87568d3f6ab88372ca5d1bf9a7033811cc958f899c532886f526fe992"
SOURCE_RELATIVES = (
    "scripts/export_sop_trained_train_features.py",
    "scripts/export_unicom_sop_embeddings.py",
    "scripts/sop_teacher_anchored_runtime.py",
    "scripts/train_sop_compact_backbone.py",
    "src/sfora/joint_relational_compaction.py",
    "src/sfora/representation_ceiling.py",
    "src/sfora/sop_compact_training.py",
    "src/sfora/sop_evaluation.py",
)


def source_manifest() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    return {relative: sha256(root / relative) for relative in SOURCE_RELATIVES}


def no_other_gpu_compute_process() -> bool:
    """Require an idle GPU before this one-time feature export starts."""

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ValueError("SOP trained export cannot inspect GPU occupancy") from error
    pids = {int(line.strip()) for line in result.stdout.splitlines() if line.strip().isdigit()}
    return not (pids - {os.getpid()})


def verify_checkpoint_receipt(
    checkpoint: Mapping[str, object], receipt: Mapping[str, object], checkpoint_sha256: str
) -> tuple[int, int]:
    """Bind a diagnostic or final ArcFace payload to its immutable receipt."""

    diagnostic = receipt.get("schema") == "sfora-sop-compact-training-diagnostic-v1"
    final = receipt.get("schema") == "sfora-sop-compact-full-backbone-v1"
    inputs = receipt.get("inputs")
    expected_sha = (
        receipt.get("checkpoint_sha256")
        if diagnostic
        else inputs.get("checkpoint_output_sha256")
        if isinstance(inputs, Mapping)
        else None
    )
    step = receipt.get("step" if diagnostic else "updates")
    width = receipt.get("embedding_width", 128)
    head = checkpoint.get("head")
    classifier = checkpoint.get("classifier")
    if (
        not (diagnostic or final)
        or expected_sha != checkpoint_sha256
        or receipt.get("arm") != "arcface"
        or receipt.get("recipe") != "reference"
        or width not in (128, 768)
        or type(step) is not int
        or step < 1
        or checkpoint.get("arm") != receipt.get("arm")
        or checkpoint.get("recipe") != receipt.get("recipe")
        or checkpoint.get("seed") != receipt.get("seed")
        or checkpoint.get("updates") != step
        or checkpoint.get("embedding_width", 128 if width == 128 else None) != width
        or checkpoint.get("schedule_sha256") != receipt.get("schedule_sha256")
        or not isinstance(checkpoint.get("model"), Mapping)
        or not isinstance(head, Mapping)
        or not isinstance(head.get("weight"), torch.Tensor)
        or not isinstance(head.get("bias"), torch.Tensor)
        or head["weight"].shape != (width, 768)
        or head["bias"].shape != (width,)
        or not isinstance(classifier, torch.Tensor)
        or classifier.shape != (10_186, width)
    ):
        raise ValueError("SOP trained checkpoint or receipt differs")
    return int(step), int(width)


def verify_holdout_parity(actual: Mapping[str, object], expected: Mapping[str, object]) -> None:
    """Require the exported features to reproduce the run's held-out ranking."""

    for arm in ("float", "packed"):
        observed = actual.get(arm)
        reference = expected.get(arm)
        if (
            not isinstance(observed, Mapping)
            or not isinstance(reference, Mapping)
            or observed.get("per_query_r1") != reference.get("per_query_r1")
            or abs(float(observed["map_at_r"]) - float(reference["map_at_r"])) > 1e-5
        ):
            raise ValueError(f"SOP trained export does not reproduce {arm} holdout")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--unicom-checkout", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--trained-checkpoint", type=Path, required=True)
    parser.add_argument("--training-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--execute-after-gpu-idle", action="store_true", required=True)
    args = parser.parse_args()
    if (
        args.workers < 0
        or args.output.exists()
        or not torch.cuda.is_available()
        or not no_other_gpu_compute_process()
        or sha256(args.source_checkpoint) != CHECKPOINT_SHA256
    ):
        raise ValueError("SOP trained export invocation or GPU state differs")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryFile(dir=args.output.parent):
        pass
    initial_source_manifest = source_manifest()
    receipt_sha256 = sha256(args.training_receipt)
    receipt = json.loads(args.training_receipt.read_text())
    trained_sha256 = sha256(args.trained_checkpoint)
    payload = torch.load(args.trained_checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(receipt, Mapping) or not isinstance(payload, Mapping):
        raise ValueError("SOP trained export inputs differ")
    step, width = verify_checkpoint_receipt(payload, receipt, trained_sha256)
    metadata_sha = sha256(args.dataset_root / "Ebay_train.txt")
    input_record = receipt if "input_checkpoint_sha256" in receipt else receipt.get("inputs")
    if (
        not isinstance(input_record, Mapping)
        or input_record.get("input_checkpoint_sha256", input_record.get("checkpoint_sha256"))
        != CHECKPOINT_SHA256
        or input_record.get("features_archive_sha256") != ARCHIVE_SHA256
        or input_record.get("sop_train_metadata_sha256") != metadata_sha
    ):
        raise ValueError("SOP trained export source or metadata differs")

    all_records = parse_sop_records(args.dataset_root)
    if ordered_record_sha256(all_records) != EXPECTED_SOP_RECORD_SHA256:
        raise ValueError("SOP trained export ordered image inventory differs")
    records = tuple(record for record in all_records if record.split == "train")
    labels = tuple(record.label for record in records)
    if len(records) != 59_551:
        raise ValueError("SOP trained export train row count differs")
    partition = deterministic_class_partition(labels, fit_fraction=FIT_FRACTION, seed=SPLIT_SEED)
    fit = np.asarray(partition.fit_row_indexes, dtype=np.int64)
    holdout = np.asarray(partition.validation_row_indexes, dtype=np.int64)
    if (
        len(holdout) != 5_851
        or len(fit) != 53_700
        or not np.array_equal(np.sort(np.concatenate((fit, holdout))), np.arange(59_551))
        or receipt.get("validation_image_ids") != [records[index].image_id for index in holdout]
        or receipt.get("validation_labels") != [labels[index] for index in holdout]
        or receipt.get("fit_row_indexes_sha256")
        != hashlib.sha256(
            np.asarray(partition.fit_row_indexes, dtype="<i4").tobytes(order="C")
        ).hexdigest()
        or receipt.get("validation_row_indexes_sha256")
        != hashlib.sha256(holdout.astype("<i4").tobytes(order="C")).hexdigest()
    ):
        raise ValueError("SOP trained export fit/holdout split differs")

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(16)
    authenticated = load_authenticated_source_model(args.unicom_checkout, args.source_checkpoint)
    model = authenticated.encoder
    model.load_state_dict(payload["model"], strict=True)
    model = model.cuda().eval()
    head = nn.Linear(768, width)
    head.load_state_dict(payload["head"], strict=True)
    head = head.cuda().eval()
    torch.cuda.reset_peak_memory_stats()

    def encode_rows(indexes: np.ndarray) -> tuple[torch.Tensor, float]:
        loader = DataLoader(
            IndexedImages(
                tuple(records[index].image_path for index in indexes),
                tuple(labels[index] for index in indexes),
                authenticated.transform,
            ),
            batch_size=EVAL_BATCH_SIZE,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
        )
        started = time.perf_counter()
        outputs: list[torch.Tensor] = []
        with torch.inference_mode():
            for images, _ in loader:
                source = model(images.cuda(non_blocking=True))
                features = compact_head_features(source, head, output_dim=width)
                outputs.append(F.normalize(features, dim=1).float().cpu())
        encoded = torch.cat(outputs).contiguous()
        if encoded.shape != (len(indexes), width) or not bool(torch.isfinite(encoded).all()):
            raise ValueError("SOP trained export feature inventory differs")
        return encoded, time.perf_counter() - started

    holdout_values, holdout_encode_seconds = encode_rows(holdout)
    holdout_scores = score_validation_features(
        holdout_values, tuple(labels[index] for index in holdout)
    )
    expected_scores = receipt.get("validation")
    if not isinstance(expected_scores, Mapping):
        raise ValueError("SOP trained export validation receipt differs")
    verify_holdout_parity(holdout_scores, expected_scores)
    packed_scores = holdout_scores.get("packed")
    if not isinstance(packed_scores, Mapping):
        raise ValueError("SOP trained export packed validation differs")
    fit_values, fit_encode_seconds = encode_rows(fit)
    values = torch.empty((59_551, width), dtype=torch.float32)
    values[holdout] = holdout_values
    values[fit] = fit_values
    if (
        sha256(args.training_receipt) != receipt_sha256
        or sha256(args.trained_checkpoint) != trained_sha256
        or sha256(args.dataset_root / "Ebay_train.txt") != metadata_sha
        or not no_other_gpu_compute_process()
        or source_manifest() != initial_source_manifest
    ):
        raise ValueError("SOP trained export authorities changed during execution")
    metadata = {
        "schema": "sfora-sop-trained-train-features-v1",
        "claim_eligible": False,
        "dataset": "Stanford Online Products",
        "split": "official train images; includes fit and validation identities",
        "width": width,
        "step": step,
        "rows": len(records),
        "seed": receipt["seed"],
        "trained_checkpoint_sha256": trained_sha256,
        "training_receipt_sha256": receipt_sha256,
        "source_checkpoint_sha256": CHECKPOINT_SHA256,
        "sop_train_metadata_sha256": metadata_sha,
        "script_sha256": sha256(Path(__file__)),
        "source_sha256": initial_source_manifest,
        "holdout_encode_seconds": holdout_encode_seconds,
        "fit_encode_seconds": fit_encode_seconds,
        "encode_seconds": holdout_encode_seconds + fit_encode_seconds,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_host_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "hardware": {
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "python": platform.python_version(),
        },
        "holdout_packed_recall_at_1": packed_scores["recall_at_1"],
    }
    publish_file_noreplace(
        args.output,
        lambda stream: np.savez_compressed(
            stream,
            train_embeddings=values.numpy(),
            train_labels=np.asarray(labels, dtype=np.int64),
            train_image_ids=np.asarray([record.image_id for record in records], dtype=np.int64),
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True, allow_nan=False)),
        ),
    )
    print(json.dumps({"feature_archive_sha256": sha256(args.output), **metadata}, sort_keys=True))


if __name__ == "__main__":
    main()
