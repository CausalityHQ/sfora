#!/usr/bin/env python3
"""Test train-fitted PCA compression of one frozen SOP full-width checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import BinaryIO

import numpy as np
import torch
from export_unicom_sop_embeddings import load_sop_embedding_archive, parse_sop_records
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
    assert_source_imports,
    publish_file_noreplace,
    score_validation_features,
    sha256,
    source_manifest,
)

from sfora.representation_ceiling import deterministic_class_partition, fit_centered_pca
from sfora.sop_compact_training import compact_head_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in (
        "dataset-root",
        "unicom-checkout",
        "source-checkpoint",
        "features-archive",
        "fullwidth-checkpoint",
        "fullwidth-receipt",
        "compact-checkpoint",
        "compact-receipt",
        "projection-output",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--execute-sop-fullwidth-compression-probe", action="store_true", required=True
    )
    return parser.parse_args()


def validate_receipts(
    full: Mapping[str, object],
    compact: Mapping[str, object],
    *,
    fullwidth_checkpoint_sha256: str,
    compact_checkpoint_sha256: str,
    source_sha256: Mapping[str, str],
) -> None:
    """Require matched seed, schedule and train-only identity inventory."""

    common = (
        "seed",
        "step",
        "total_updates",
        "schedule_sha256",
        "fit_row_indexes_sha256",
        "validation_row_indexes_sha256",
        "validation_image_ids",
        "validation_labels",
    )
    if (
        full.get("schema") != "sfora-sop-compact-training-diagnostic-v1"
        or full.get("claim_eligible") is not False
        or full.get("arm") != "arcface"
        or full.get("recipe") != "reference"
        or full.get("step") != 8000
        or full.get("total_updates") != 53760
        or full.get("seed") != 179019
        or full.get("embedding_width") != 768
        or full.get("checkpoint_sha256") != fullwidth_checkpoint_sha256
        or full.get("source_sha256") != source_sha256
        or full.get("input_checkpoint_sha256") != CHECKPOINT_SHA256
        or full.get("features_archive_sha256") != ARCHIVE_SHA256
        or compact.get("schema") != "sfora-sop-compact-training-diagnostic-v1"
        or compact.get("claim_eligible") is not False
        or compact.get("arm") != "arcface"
        or compact.get("recipe") != "reference"
        or compact.get("embedding_width", 128) != 128
        or compact.get("checkpoint_sha256") != compact_checkpoint_sha256
        or compact.get("input_checkpoint_sha256") != CHECKPOINT_SHA256
        or compact.get("features_archive_sha256") != ARCHIVE_SHA256
        or not isinstance(compact.get("source_sha256"), Mapping)
        or any(full.get(key) != compact.get(key) for key in common)
        or not isinstance(full.get("validation"), Mapping)
        or not isinstance(compact.get("validation"), Mapping)
        or len(full.get("validation_image_ids", [])) != 5851
    ):
        raise ValueError("SOP full-width compression authority differs")


@torch.inference_mode()
def encode_split(
    model: nn.Module, head: nn.Linear, loader: DataLoader
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    head.eval()
    source_outputs: list[torch.Tensor] = []
    head_outputs: list[torch.Tensor] = []
    for images, _ in loader:
        source = model(images.cuda(non_blocking=True))
        features = compact_head_features(source, head, output_dim=head.out_features)
        source_outputs.append(F.normalize(source.float(), dim=1).cpu())
        head_outputs.append(F.normalize(features, dim=1).cpu())
    source_values = torch.cat(source_outputs).contiguous()
    head_values = torch.cat(head_outputs).contiguous()
    if (
        source_values.shape[1] != 768
        or head_values.shape[1] != head.out_features
        or not bool(torch.isfinite(source_values).all())
        or not bool(torch.isfinite(head_values).all())
    ):
        raise ValueError("SOP full-width compression features differ")
    return source_values, head_values


def tensor_sha256(values: torch.Tensor) -> str:
    if values.device.type != "cpu" or values.dtype != torch.float32 or not values.is_contiguous():
        raise ValueError("SOP full-width compression feature authority differs")
    return hashlib.sha256(values.numpy().astype("<f4", copy=False).tobytes()).hexdigest()


def main() -> None:
    args = parse_args()
    started_all = time.perf_counter()
    root = Path(__file__).resolve().parents[1]
    if Path(__file__).resolve() != root / "scripts/probe_sop_fullwidth_compression.py":
        raise ValueError("SOP full-width compression probe source differs")
    assert_source_imports()
    probe_sha256 = sha256(Path(__file__))
    if (
        args.workers < 0
        or args.output.exists()
        or args.output.is_symlink()
        or args.projection_output.exists()
        or args.projection_output.is_symlink()
        or args.output.resolve() == args.projection_output.resolve()
    ):
        raise ValueError("SOP full-width compression invocation differs")
    torch.set_num_threads(16)
    source_sha256 = source_manifest()
    if (
        sha256(args.source_checkpoint) != CHECKPOINT_SHA256
        or sha256(args.features_archive) != ARCHIVE_SHA256
    ):
        raise ValueError("SOP full-width compression source differs")
    full_digest = sha256(args.fullwidth_receipt)
    compact_digest = sha256(args.compact_receipt)
    full = json.loads(args.fullwidth_receipt.read_text())
    compact = json.loads(args.compact_receipt.read_text())
    fullwidth_checkpoint_digest = sha256(args.fullwidth_checkpoint)
    compact_checkpoint_digest = sha256(args.compact_checkpoint)
    validate_receipts(
        full,
        compact,
        fullwidth_checkpoint_sha256=fullwidth_checkpoint_digest,
        compact_checkpoint_sha256=compact_checkpoint_digest,
        source_sha256=source_sha256,
    )
    records = parse_sop_records(args.dataset_root)
    train_records = tuple(record for record in records if record.split == "train")
    archive = load_sop_embedding_archive(args.features_archive)
    if tuple(record.image_id for record in train_records) != tuple(
        int(value) for value in archive["train_image_ids"]
    ) or tuple(record.label for record in train_records) != tuple(
        int(value) for value in archive["train_labels"]
    ):
        raise ValueError("SOP full-width compression train archive differs")
    partition = deterministic_class_partition(
        tuple(record.label for record in train_records), fit_fraction=FIT_FRACTION, seed=SPLIT_SEED
    )
    fit_records = tuple(train_records[index] for index in partition.fit_row_indexes)
    validation_records = tuple(train_records[index] for index in partition.validation_row_indexes)
    validation_labels = tuple(record.label for record in validation_records)
    if (
        len(fit_records) != 53_700
        or len(validation_records) != 5851
        or [record.image_id for record in validation_records] != full["validation_image_ids"]
        or list(validation_labels) != full["validation_labels"]
        or hashlib.sha256(np.asarray(partition.fit_row_indexes, dtype="<i4").tobytes()).hexdigest()
        != full["fit_row_indexes_sha256"]
        or hashlib.sha256(
            np.asarray(partition.validation_row_indexes, dtype="<i4").tobytes()
        ).hexdigest()
        != full["validation_row_indexes_sha256"]
    ):
        raise ValueError("SOP full-width compression split differs")
    authenticated = load_authenticated_source_model(args.unicom_checkout, args.source_checkpoint)
    model = authenticated.encoder.eval()
    fullwidth_checkpoint = torch.load(
        args.fullwidth_checkpoint, map_location="cpu", weights_only=True
    )
    compact_checkpoint = torch.load(args.compact_checkpoint, map_location="cpu", weights_only=True)
    if (
        not isinstance(fullwidth_checkpoint, Mapping)
        or not isinstance(compact_checkpoint, Mapping)
        or any(
            checkpoint.get("updates") != 8000
            or checkpoint.get("embedding_width", 128) != width
            or checkpoint.get("seed") != 179019
            for checkpoint, width in ((fullwidth_checkpoint, 768), (compact_checkpoint, 128))
        )
    ):
        raise ValueError("SOP full-width compression checkpoint differs")
    model.load_state_dict(fullwidth_checkpoint["model"], strict=True)
    fullwidth_head = nn.Linear(768, 768)
    fullwidth_head.load_state_dict(fullwidth_checkpoint["head"], strict=True)
    model.load_state_dict(compact_checkpoint["model"], strict=True)
    compact_head = nn.Linear(768, 128)
    compact_head.load_state_dict(compact_checkpoint["head"], strict=True)
    if args.preflight_only:
        print(json.dumps({"preflight": "passed", "step": 8000, "fit": 53700, "holdout": 5851}))
        return
    pids = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    )
    if pids.strip() or not torch.cuda.is_available():
        raise ValueError("SOP full-width compression GPU is occupied")
    preflight_seconds = time.perf_counter() - started_all
    torch.backends.cuda.matmul.allow_tf32 = False
    model = model.cuda().eval()
    fullwidth_head = fullwidth_head.cuda().eval()
    compact_head = compact_head.cuda().eval()
    fit_loader = DataLoader(
        IndexedImages(
            tuple(record.image_path for record in fit_records),
            tuple(record.label for record in fit_records),
            authenticated.transform,
        ),
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    validation_loader = DataLoader(
        IndexedImages(
            tuple(record.image_path for record in validation_records),
            validation_labels,
            authenticated.transform,
        ),
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    torch.cuda.reset_peak_memory_stats()
    model.load_state_dict(fullwidth_checkpoint["model"], strict=True)
    full_fit_started = time.perf_counter()
    full_fit_source, full_fit_head = encode_split(model, fullwidth_head, fit_loader)
    full_fit_encode_seconds = time.perf_counter() - full_fit_started
    full_holdout_started = time.perf_counter()
    full_holdout_source, full_holdout_head = encode_split(model, fullwidth_head, validation_loader)
    full_holdout_encode_seconds = time.perf_counter() - full_holdout_started
    model.load_state_dict(compact_checkpoint["model"], strict=True)
    compact_fit_started = time.perf_counter()
    compact_fit_source, _compact_fit_head = encode_split(model, compact_head, fit_loader)
    compact_fit_encode_seconds = time.perf_counter() - compact_fit_started
    compact_holdout_started = time.perf_counter()
    compact_holdout_source, compact_holdout_head = encode_split(
        model, compact_head, validation_loader
    )
    compact_holdout_encode_seconds = time.perf_counter() - compact_holdout_started
    if any(
        values.shape != shape
        for values, shape in (
            (full_fit_source, (53_700, 768)),
            (full_fit_head, (53_700, 768)),
            (full_holdout_source, (5851, 768)),
            (full_holdout_head, (5851, 768)),
            (compact_fit_source, (53_700, 768)),
            (compact_holdout_source, (5851, 768)),
            (compact_holdout_head, (5851, 128)),
        )
    ):
        raise ValueError("SOP full-width compression feature shape differs")
    fullwidth_score = score_validation_features(full_holdout_head, validation_labels)
    compact_score = score_validation_features(compact_holdout_head, validation_labels)
    replay_checks = {}
    for name, observed, reference in (
        ("fullwidth", fullwidth_score, full["validation"]),
        ("compact", compact_score, compact["validation"]),
    ):
        actual_packed = observed["packed"]
        expected_packed = reference["packed"]
        r1_mismatches = sum(
            left != right
            for left, right in zip(
                actual_packed["per_query_r1"], expected_packed["per_query_r1"], strict=True
            )
        )
        maximum_ap_error = max(
            abs(left - right)
            for left, right in zip(
                actual_packed["per_query_ap"], expected_packed["per_query_ap"], strict=True
            )
        )
        replay_checks[name] = {
            "packed_r1_mismatches": r1_mismatches,
            "packed_maximum_ap_error": maximum_ap_error,
        }
        if r1_mismatches or maximum_ap_error > 1e-4:
            raise ValueError(f"SOP {name} holdout replay differs")
    pca_started = time.perf_counter()
    pca_inputs = {
        "fullwidth_head": (full_fit_head, full_holdout_head),
        "fullwidth_source": (full_fit_source, full_holdout_source),
        "compact_source": (compact_fit_source, compact_holdout_source),
    }
    pca_models = {}
    projected_scores = {}
    projected_feature_sha256 = {}
    for name, (fit_values, holdout_values) in pca_inputs.items():
        pca = fit_centered_pca(fit_values, dimensions=128)
        projected = pca.apply(holdout_values)
        pca_models[name] = pca
        projected_scores[name] = score_validation_features(projected, validation_labels)
        projected_feature_sha256[name] = tensor_sha256(projected)
    pca_score_seconds = time.perf_counter() - pca_started
    if (
        source_manifest() != source_sha256
        or sha256(Path(__file__)) != probe_sha256
        or sha256(args.fullwidth_checkpoint) != fullwidth_checkpoint_digest
        or sha256(args.compact_checkpoint) != compact_checkpoint_digest
    ):
        raise ValueError("SOP full-width compression source changed")
    projection = {
        **{
            f"{name}_{field}": getattr(model, field).numpy()
            for name, model in pca_models.items()
            for field in ("mean", "components")
        },
        "step": np.asarray(8000, dtype=np.int64),
        "fullwidth_checkpoint_sha256": np.asarray(fullwidth_checkpoint_digest),
        "compact_checkpoint_sha256": np.asarray(compact_checkpoint_digest),
        "fit_row_indexes_sha256": np.asarray(full["fit_row_indexes_sha256"]),
    }
    args.projection_output.parent.mkdir(parents=True, exist_ok=True)

    def write_projection(stream: BinaryIO) -> None:
        np.savez(stream, **projection)

    receipt = {
        "schema": "sfora-sop-matched-step-head-backbone-pca128-v2",
        "claim_eligible": False,
        "protocol": (
            "SOP train identity holdout, matched step 8000; PCA fit on fit identities only; "
            "no official test images; no final product checkpoint claim"
        ),
        "seed": 179019,
        "step": 8000,
        "fit_count": len(fit_records),
        "holdout_count": len(validation_records),
        "fullwidth_packed_bytes_per_item": 770,
        "projected_packed_bytes_per_item": 130,
        "fullwidth": fullwidth_score,
        "compact128": compact_score,
        "pca128": projected_scores,
        "replay_checks": replay_checks,
        "interpretation_limit": (
            "The backbones were trained with different head/proxy geometry. "
            "Source PCA probes linear compressibility, not a matched training-method ablation."
        ),
        "timing_seconds": {
            "preflight": preflight_seconds,
            "fullwidth_fit_encode": full_fit_encode_seconds,
            "fullwidth_holdout_encode": full_holdout_encode_seconds,
            "compact_fit_encode": compact_fit_encode_seconds,
            "compact_holdout_encode": compact_holdout_encode_seconds,
            "pca_fit_and_holdout_score": pca_score_seconds,
            "total": time.perf_counter() - started_all,
        },
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
            "features_archive_sha256": ARCHIVE_SHA256,
            "fullwidth_checkpoint_sha256": fullwidth_checkpoint_digest,
            "compact_checkpoint_sha256": compact_checkpoint_digest,
            "fullwidth_receipt_sha256": full_digest,
            "compact_receipt_sha256": compact_digest,
            "compact_source_sha256": compact["source_sha256"],
            "fit_features_sha256": {
                "fullwidth_source": tensor_sha256(full_fit_source),
                "fullwidth_head": tensor_sha256(full_fit_head),
                "compact_source": tensor_sha256(compact_fit_source),
            },
            "holdout_features_sha256": {
                "fullwidth_source": tensor_sha256(full_holdout_source),
                "fullwidth_head": tensor_sha256(full_holdout_head),
                "compact_source": tensor_sha256(compact_holdout_source),
                "compact_head": tensor_sha256(compact_holdout_head),
            },
            "projected_features_sha256": projected_feature_sha256,
            "projection_array_sha256": {
                name: hashlib.sha256(np.asarray(array).tobytes()).hexdigest()
                for name, array in projection.items()
            },
            "projection_sha256": "",
            "source_sha256": source_sha256,
            "probe_sha256": probe_sha256,
        },
        "argv": sys.argv,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    projection_published = False
    try:
        publish_file_noreplace(args.projection_output, write_projection)
        projection_published = True
        receipt["inputs"]["projection_sha256"] = sha256(args.projection_output)
        payload = (json.dumps(receipt, sort_keys=True, allow_nan=False) + "\n").encode()
        publish_file_noreplace(args.output, lambda stream: stream.write(payload))
    except BaseException:
        if projection_published:
            args.projection_output.unlink()
        raise
    print(
        json.dumps(
            {
                "receipt": str(args.output),
                "fullwidth_packed_r1": fullwidth_score["packed"]["recall_at_1"],
                "pca128_packed_r1": {
                    name: score["packed"]["recall_at_1"] for name, score in projected_scores.items()
                },
                "compact128_packed_r1": compact_score["packed"]["recall_at_1"],
            }
        )
    )


if __name__ == "__main__":
    main()
