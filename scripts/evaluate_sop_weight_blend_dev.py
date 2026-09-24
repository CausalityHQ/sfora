#!/usr/bin/env python3
"""Screen source-to-trained B/16 weight blends on SOP train and CUB development."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import sys
import time
from collections.abc import Mapping
from io import BytesIO
from pathlib import Path
from typing import BinaryIO

import numpy as np
import torch
from evaluate_sop_cub_transfer import gpu_compute_pids
from evaluate_sop_fullwidth_cub_dev import (
    SELECTED_TRAINING_RECEIPT_SHA256,
    VerifiedCubDevelopmentImages,
    array_sha256,
    score_features,
    select_development_records,
    validate_selected_authority,
)
from evaluate_sop_fullwidth_cub_dev import (
    source_manifest as cub_source_manifest,
)
from evaluate_sop_reference_checkpoint import EXPECTED_SOP_RECORD_SHA256, VerifiedImages
from export_unicom_cub_embeddings import (
    CUB_ARCHIVE_SHA256,
    parse_cub_records,
    verify_extracted_cub_matches_archive,
)
from export_unicom_cub_embeddings import (
    ordered_record_sha256 as cub_record_sha256,
)
from export_unicom_sop_embeddings import (
    ordered_record_sha256 as sop_record_sha256,
)
from export_unicom_sop_embeddings import (
    parse_sop_records,
)
from sop_teacher_anchored_runtime import load_authenticated_source_model
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from train_sop_compact_backbone import (
    CHECKPOINT_SHA256,
    EVAL_BATCH_SIZE,
    FIT_FRACTION,
    SPLIT_SEED,
    publish_file_noreplace,
    score_validation_features,
    sha256,
)

from sfora.representation_ceiling import deterministic_class_partition
from sfora.sop_compact_training import compact_head_features

SOP_HOLDOUT_RECEIPT_SHA256 = SELECTED_TRAINING_RECEIPT_SHA256
FINAL_TRAINING_RECEIPT_SHA256 = "73408b5ceddace5f23cca4bbf071ec1a8df3e72db0cc9b20a7e7ae9ae2e1404f"
CUB_DEVELOPMENT_RECEIPT_SHA256 = "93c2e04a94f618c0e48884ae59182e327d2d7205b6c059160cf5207ac72590d9"
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
EXECUTION_ORDER = (0.0, 1.0, 0.25, 0.5, 0.75)
BATCHNORM_COUNTERS = frozenset({"feature.1.num_batches_tracked", "feature.3.num_batches_tracked"})


def blend_state(
    source: Mapping[str, torch.Tensor], trained: Mapping[str, torch.Tensor], alpha: float
) -> dict[str, torch.Tensor]:
    """Interpolate compatible model tensors, preserving exact endpoints."""

    if (
        type(alpha) is not float
        or not 0.0 <= alpha <= 1.0
        or set(source) != set(trained)
        or not source
    ):
        raise ValueError("SOP blend state differs")
    blended: dict[str, torch.Tensor] = {}
    for name, earlier in source.items():
        later = trained[name]
        if (
            type(earlier) is not torch.Tensor
            or type(later) is not torch.Tensor
            or earlier.shape != later.shape
            or earlier.dtype != later.dtype
            or earlier.device.type != "cpu"
            or later.device.type != "cpu"
            or (earlier.is_floating_point() and not bool(torch.isfinite(earlier).all()))
            or (later.is_floating_point() and not bool(torch.isfinite(later).all()))
        ):
            raise ValueError("SOP blend state differs")
        if not earlier.is_floating_point():
            if name in BATCHNORM_COUNTERS:
                if (
                    earlier.dtype != torch.int64
                    or earlier.ndim != 0
                    or int(earlier) < 0
                    or int(later) < 0
                ):
                    raise ValueError("SOP blend state differs")
                # BatchNorm ignores this counter in eval mode. Preserve each
                # endpoint; keep the pretrained counter for intermediate arms.
                blended[name] = later if alpha == 1.0 else earlier
            elif not torch.equal(earlier, later):
                raise ValueError("SOP blend state differs")
            else:
                blended[name] = earlier
        elif alpha == 0.0:
            blended[name] = earlier
        elif alpha == 1.0:
            blended[name] = later
        else:
            blended[name] = torch.lerp(earlier, later, alpha)
    return blended


@torch.inference_mode()
def encode(
    model: nn.Module, head: nn.Linear, loader: DataLoader, expected_rows: int
) -> tuple[torch.Tensor, float]:
    started = time.perf_counter()
    rows: list[torch.Tensor] = []
    for images, _labels in loader:
        source = model(images.cuda(non_blocking=True))
        rows.append(F.normalize(compact_head_features(source, head, output_dim=768), dim=1).cpu())
    values = torch.cat(rows).contiguous()
    if values.shape != (expected_rows, 768) or not bool(torch.isfinite(values).all()):
        raise ValueError("SOP blend feature geometry differs")
    return values, time.perf_counter() - started


def source_manifest() -> dict[str, str]:
    manifest = cub_source_manifest()
    root = Path(__file__).resolve().parents[1]
    for module, relative in (
        (sys.modules[__name__], "scripts/evaluate_sop_weight_blend_dev.py"),
        (
            sys.modules.get("evaluate_sop_reference_checkpoint"),
            "scripts/evaluate_sop_reference_checkpoint.py",
        ),
    ):
        if module is None or Path(module.__file__).resolve() != (root / relative).resolve():
            raise ValueError("SOP blend source imports differ")
        manifest[relative] = sha256(root / relative)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in (
        "sop-root",
        "cub-root",
        "cub-archive",
        "unicom-checkout",
        "source-checkpoint",
        "trained-checkpoint",
        "training-receipt",
        "final-training-receipt",
        "prior-cub-receipt",
        "output",
    ):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--execute-sop-weight-blend-dev", action="store_true", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    if args.output.exists() or args.output.is_symlink() or args.workers < 0:
        raise ValueError("SOP blend output authority differs")
    torch.set_num_threads(16)
    torch.backends.cuda.matmul.allow_tf32 = False
    manifest = source_manifest()
    training_bytes = args.training_receipt.read_bytes()
    final_bytes = args.final_training_receipt.read_bytes()
    prior_cub_bytes = args.prior_cub_receipt.read_bytes()
    checkpoint_bytes = args.trained_checkpoint.read_bytes()
    training_sha = hashlib.sha256(training_bytes).hexdigest()
    final_sha = hashlib.sha256(final_bytes).hexdigest()
    prior_cub_sha = hashlib.sha256(prior_cub_bytes).hexdigest()
    checkpoint_sha = hashlib.sha256(checkpoint_bytes).hexdigest()
    training = json.loads(training_bytes)
    final = json.loads(final_bytes)
    prior_cub = json.loads(prior_cub_bytes)
    validate_selected_authority(training, checkpoint_sha, training_sha)
    if (
        training_sha != SOP_HOLDOUT_RECEIPT_SHA256
        or final_sha != FINAL_TRAINING_RECEIPT_SHA256
        or final.get("schema") != "sfora-sop-compact-full-backbone-v1"
        or final.get("embedding_width") != 768
        or prior_cub_sha != CUB_DEVELOPMENT_RECEIPT_SHA256
        or prior_cub.get("schema") != "sfora-sop-fullwidth-cub-development-v1"
        or sha256(args.source_checkpoint) != CHECKPOINT_SHA256
        or sha256(args.cub_archive) != CUB_ARCHIVE_SHA256
    ):
        raise ValueError("SOP blend pinned input differs")
    trained = torch.load(BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True)
    del checkpoint_bytes
    if not isinstance(trained, Mapping) or trained.get("updates") != 48000:
        raise ValueError("SOP blend checkpoint differs")
    trained = {name: trained[name] for name in ("model", "head")}

    sop_all = parse_sop_records(args.sop_root)
    if (
        sha256(args.sop_root / "Ebay_train.txt") != final["inputs"].get("sop_train_metadata_sha256")
        or sop_record_sha256(sop_all) != EXPECTED_SOP_RECORD_SHA256
    ):
        raise ValueError("SOP blend dataset authority differs")
    sop_train = tuple(row for row in sop_all if row.split == "train")
    partition = deterministic_class_partition(
        tuple(row.label for row in sop_train), fit_fraction=FIT_FRACTION, seed=SPLIT_SEED
    )
    fit_hash = hashlib.sha256(
        np.asarray(partition.fit_row_indexes, dtype="<i4").tobytes()
    ).hexdigest()
    holdout_hash = hashlib.sha256(
        np.asarray(partition.validation_row_indexes, dtype="<i4").tobytes()
    ).hexdigest()
    sop = tuple(sop_train[index] for index in partition.validation_row_indexes)
    if (
        len(sop) != 5851
        or fit_hash != training.get("fit_row_indexes_sha256")
        or holdout_hash != training.get("validation_row_indexes_sha256")
        or [row.image_id for row in sop] != training.get("validation_image_ids")
        or [row.label for row in sop] != training.get("validation_labels")
    ):
        raise ValueError("SOP blend train holdout differs")
    sop_manifest = b"".join(hashlib.sha256(row.image_path.read_bytes()).digest() for row in sop)

    cub_all = parse_cub_records(args.cub_root)
    cub_content_sha = verify_extracted_cub_matches_archive(args.cub_archive, args.cub_root, cub_all)
    cub = select_development_records(cub_all)
    cub_manifest = b"".join(hashlib.sha256(row.image_path.read_bytes()).digest() for row in cub)
    if (
        [row.image_id for row in cub] != prior_cub.get("image_ids")
        or [row.label for row in cub] != prior_cub.get("labels")
        or hashlib.sha256(cub_manifest).hexdigest()
        != prior_cub.get("inputs", {}).get("image_manifest_sha256")
        or cub_content_sha != prior_cub.get("inputs", {}).get("cub_content_sha256")
    ):
        raise ValueError("SOP blend CUB development differs")
    preflight_seconds = time.perf_counter() - started
    if not torch.cuda.is_available() or gpu_compute_pids():
        raise ValueError("SOP blend GPU is unavailable or occupied")

    authenticated = load_authenticated_source_model(args.unicom_checkout, args.source_checkpoint)
    model = authenticated.encoder.cuda().eval()
    source_model = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }
    source_head = nn.Linear(768, 768)
    with torch.no_grad():
        nn.init.eye_(source_head.weight)
        nn.init.zeros_(source_head.bias)
    head = source_head.cuda().eval()
    source_head_state = {
        name: value.detach().cpu().clone() for name, value in source_head.state_dict().items()
    }
    sop_loader = DataLoader(
        VerifiedImages(sop, authenticated.transform, sop_manifest),
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    cub_loader = DataLoader(
        VerifiedCubDevelopmentImages(cub, authenticated.transform, cub_manifest),
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    sop_labels = torch.tensor([row.label for row in sop], dtype=torch.int64)
    cub_labels = torch.tensor([row.label for row in cub], dtype=torch.int64)
    torch.cuda.reset_peak_memory_stats()
    arms: dict[str, object] = {}
    for alpha in EXECUTION_ORDER:
        model.load_state_dict(blend_state(source_model, trained["model"], alpha), strict=True)
        head.load_state_dict(blend_state(source_head_state, trained["head"], alpha), strict=True)
        sop_values, sop_seconds = encode(model, head, sop_loader, len(sop))
        cub_values, cub_seconds = encode(model, head, cub_loader, len(cub))
        sop_scores = score_validation_features(sop_values, tuple(sop_labels.tolist()))
        cub_scores = score_features(cub_values, cub_labels)
        if alpha == 1.0:
            for scorer in ("float", "packed"):
                for metric in ("per_query_r1", "per_query_ap"):
                    if sop_scores[scorer][metric] != training["validation"][scorer][metric]:
                        raise ValueError("SOP blend trained holdout parity differs")
        if alpha == 0.0:
            for scorer in ("float", "packed"):
                for metric in ("per_query_r1", "per_query_ap"):
                    if sop_scores[scorer][metric] != final["initial_validation"][scorer][metric]:
                        raise ValueError("SOP blend source holdout parity differs")
        if alpha in (0.0, 1.0):
            prior_arm = "pretrained_fullwidth" if alpha == 0.0 else "fullwidth"
            for scorer in ("float", "packed"):
                for metric in ("per_query_r1", "per_query_ap"):
                    if cub_scores[scorer][metric] != prior_cub[prior_arm][scorer][metric]:
                        raise ValueError("SOP blend CUB endpoint parity differs")
            if array_sha256(cub_values) != prior_cub["inputs"]["feature_array_sha256"][prior_arm]:
                raise ValueError("SOP blend CUB endpoint features differ")
        arms[str(alpha)] = {
            "sop_train_holdout": sop_scores,
            "cub_development": cub_scores,
            "sop_encode_seconds": sop_seconds,
            "cub_encode_seconds": cub_seconds,
            "feature_array_sha256": {
                "sop_train_holdout": array_sha256(sop_values),
                "cub_development": array_sha256(cub_values),
            },
        }
        print(
            json.dumps(
                {
                    "alpha": alpha,
                    "sop_packed_r1": sop_scores["packed"]["recall_at_1"],
                    "cub_packed_r1": cub_scores["packed"]["recall_at_1"],
                }
            ),
            flush=True,
        )
    if gpu_compute_pids() - {os.getpid()} or source_manifest() != manifest:
        raise ValueError("SOP blend execution overlap or source change")
    result = {
        "schema": "sfora-sop-fullwidth-weight-blend-development-v1",
        "claim_eligible": False,
        "alpha_definition": "(1-alpha)*pretrained + alpha*SOP-trained, model and 768-D head",
        "alphas": list(ALPHAS),
        "execution_order": list(EXECUTION_ORDER),
        "selection_rule": (
            "development diagnostic only; no alpha selected from CUB; "
            "SOP holdout already selected checkpoint step; future independent panel needed"
        ),
        "sop_queries": len(sop),
        "cub_queries": len(cub),
        "sop_labels": sop_labels.tolist(),
        "cub_labels": cub_labels.tolist(),
        "packed_bytes_per_item": 770,
        "arms": arms,
        "preflight_seconds": preflight_seconds,
        "total_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_host_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "hardware": {
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "python": platform.python_version(),
            "unicom_revision": authenticated.revision,
            "unicom_package_file": str(authenticated.package_file),
        },
        "inputs": {
            "source_checkpoint_sha256": CHECKPOINT_SHA256,
            "trained_checkpoint_sha256": checkpoint_sha,
            "training_receipt_sha256": training_sha,
            "final_training_receipt_sha256": final_sha,
            "prior_cub_receipt_sha256": prior_cub_sha,
            "cub_archive_sha256": CUB_ARCHIVE_SHA256,
            "cub_content_sha256": cub_content_sha,
            "sop_holdout_records_sha256": sop_record_sha256(sop),
            "cub_records_sha256": cub_record_sha256(cub),
            "sop_image_manifest_sha256": hashlib.sha256(sop_manifest).hexdigest(),
            "cub_image_manifest_sha256": hashlib.sha256(cub_manifest).hexdigest(),
            "source_sha256": manifest,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def write_receipt(stream: BinaryIO) -> None:
        stream.write((json.dumps(result, sort_keys=True, allow_nan=False) + "\n").encode())

    publish_file_noreplace(args.output, write_receipt)
    print(json.dumps({"output": str(args.output), "arms": len(arms)}), flush=True)


if __name__ == "__main__":
    main()
