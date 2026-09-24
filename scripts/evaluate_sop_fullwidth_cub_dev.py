#!/usr/bin/env python3
"""Evaluate SOP-trained full-width B/16 on CUB development identities."""

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
from evaluate_sop_cub_transfer import evaluator_source_manifest, gpu_compute_pids
from export_unicom_cub_embeddings import (
    CUB_ARCHIVE_SHA256,
    CubRecord,
    ordered_record_sha256,
    parse_cub_records,
    verify_extracted_cub_matches_archive,
)
from PIL import Image
from sop_teacher_anchored_runtime import load_authenticated_source_model
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from train_sop_compact_backbone import (
    CHECKPOINT_SHA256,
    EVAL_BATCH_SIZE,
    publish_file_noreplace,
    sha256,
)

from sfora.joint_relational_compaction import pack_int8_unit_embeddings
from sfora.representation_ceiling import CenteredPcaTransform
from sfora.sop_compact_training import compact_head_features
from sfora.sop_evaluation import score_symmetric

SELECTED_CHECKPOINT_SHA256 = "232f7cee93e39fa242d8461f8dc8cee684d7228eef01f8fe9399b9782e80a1b2"
SELECTED_TRAINING_RECEIPT_SHA256 = (
    "b0b857e02a26fe27560aa9721912db99f6e54f60ff6c43ffbead16403164010b"
)
PCA_RECEIPT_SHA256 = "458b45f31e61a1ad04c5213ca25cd53857a55dcc9b8b56e0359faa98d9294299"
PROJECTION_ARCHIVE_SHA256 = "b1ec4a628788b55242a2d8b233392f264be4a8c3a6eabe37a88b297373a09e91"
SEED = 179019


def select_development_records(
    records: tuple[CubRecord, ...], *, expected_count: int = 5864
) -> tuple[CubRecord, ...]:
    """Select only CUB classes 1–100, preserving the authenticated order."""

    if type(expected_count) is not int or expected_count < 1:
        raise ValueError("CUB development split differs")
    selected = tuple(record for record in records if record.split == "train")
    if len(selected) != expected_count or any(
        not (
            (record.split == "train" and 1 <= record.label <= 100)
            or (record.split == "test" and 101 <= record.label <= 200)
        )
        for record in records
    ):
        raise ValueError("CUB development split differs")
    return selected


def validate_pca_authority(
    receipt: Mapping[str, object],
    projection: Mapping[str, object],
    checkpoint_sha256: str,
    projection_sha256: str,
) -> None:
    """Require the fit-only projection to belong to the selected checkpoint."""

    inputs = receipt.get("inputs")
    projection_fields = receipt.get("projection")
    if (
        receipt.get("schema") != "sfora-sop-matched-fullwidth-fit-pca128-v1"
        or receipt.get("claim_eligible") is not False
        or receipt.get("step") != 48000
        or receipt.get("seed") != SEED
        or not isinstance(inputs, Mapping)
        or not isinstance(projection_fields, Mapping)
        or inputs.get("fullwidth_checkpoint_sha256") != checkpoint_sha256
        or receipt.get("projection_archive_sha256") != projection_sha256
        or projection.get("schema") != "sfora-sop-fit-only-fullwidth-pca128-projection-v1"
        or projection.get("step") != 48000
        or projection.get("fullwidth_checkpoint_sha256") != checkpoint_sha256
        or projection.get("fit_row_indexes_sha256")
        != projection_fields.get("fit_row_indexes_sha256")
        or projection_fields.get("fit_row_indexes_sha256")
        != "e1bc2b8d71de3f8ae8723d2d8548a259bceef2e19d68f8ca2d391d0d519e2199"
    ):
        raise ValueError("CUB development PCA authority differs")


def validate_selected_authority(
    training: Mapping[str, object], checkpoint_sha256: str, training_sha256: str
) -> None:
    if (
        checkpoint_sha256 != SELECTED_CHECKPOINT_SHA256
        or training_sha256 != SELECTED_TRAINING_RECEIPT_SHA256
        or training.get("schema") != "sfora-sop-compact-training-diagnostic-v1"
        or training.get("claim_eligible") is not False
        or training.get("arm") != "arcface"
        or training.get("recipe") != "reference"
        or training.get("seed") != SEED
        or training.get("embedding_width") != 768
        or training.get("step") != 48000
        or training.get("checkpoint_sha256") != checkpoint_sha256
    ):
        raise ValueError("CUB development selected authority differs")


def array_sha256(values: torch.Tensor) -> str:
    if values.dtype != torch.float32 or values.device.type != "cpu":
        raise ValueError("CUB development feature tensor differs")
    array = values.contiguous().numpy().astype("<f4", copy=False)
    return hashlib.sha256(array.tobytes()).hexdigest()


def source_manifest() -> dict[str, str]:
    manifest = evaluator_source_manifest()
    root = Path(__file__).resolve().parents[1]
    expected = root / "scripts/evaluate_sop_fullwidth_cub_dev.py"
    if Path(sys.modules[__name__].__file__).resolve() != expected.resolve():
        raise ValueError("CUB development source imports differ")
    manifest["scripts/evaluate_sop_fullwidth_cub_dev.py"] = sha256(Path(__file__))
    return manifest


class VerifiedCubDevelopmentImages(Dataset[tuple[torch.Tensor, int]]):
    """Recheck authenticated CUB development image bytes during decoding."""

    def __init__(self, records: tuple[CubRecord, ...], transform, manifest: bytes) -> None:
        if len(records) != 5864 or len(manifest) != 32 * len(records):
            raise ValueError("CUB development image inventory differs")
        self.records = records
        self.transform = transform
        self.manifest = manifest

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        record = self.records[index]
        image_bytes = record.image_path.read_bytes()
        if hashlib.sha256(image_bytes).digest() != self.manifest[index * 32 : (index + 1) * 32]:
            raise ValueError("CUB development image bytes changed")
        with Image.open(BytesIO(image_bytes)) as image:
            return self.transform(image.convert("RGB")), record.label


@torch.inference_mode()
def encode_fullwidth(
    model: nn.Module,
    head: nn.Linear,
    loader: DataLoader[tuple[torch.Tensor, int]],
) -> tuple[torch.Tensor, float]:
    model.eval()
    head.eval()
    started = time.perf_counter()
    batches: list[torch.Tensor] = []
    for images, _labels in loader:
        source = model(images.cuda(non_blocking=True))
        features = compact_head_features(source, head, output_dim=768)
        batches.append(F.normalize(features, dim=1).cpu())
    values = torch.cat(batches).contiguous()
    if values.shape != (5864, 768) or not bool(torch.isfinite(values).all()):
        raise ValueError("CUB development full-width features differ")
    return values, time.perf_counter() - started


def score_features(values: torch.Tensor, labels: torch.Tensor) -> dict[str, object]:
    """Score one float representation and its deployed signed-byte wire."""

    if (
        values.device.type != "cpu"
        or values.dtype != torch.float32
        or values.ndim != 2
        or values.shape[0] != len(labels)
        or labels.dtype != torch.int64
        or labels.device.type != "cpu"
    ):
        raise ValueError("CUB development scoring inventory differs")
    packed = pack_int8_unit_embeddings(values)
    cuda_labels = labels.cuda()
    return {
        "float": dict(score_symmetric(values.cuda(), cuda_labels)),
        "packed": dict(
            score_symmetric(
                packed.codes.float().cuda(),
                cuda_labels,
                inverse_norms=packed.inverse_norms.cuda(),
            )
        ),
    }


def load_projection(data: bytes) -> tuple[CenteredPcaTransform, dict[str, object]]:
    with np.load(BytesIO(data), allow_pickle=False) as archive:
        metadata = {
            name: archive[name].item()
            for name in (
                "schema",
                "step",
                "fullwidth_checkpoint_sha256",
                "fit_row_indexes_sha256",
            )
        }
        mean = np.ascontiguousarray(archive["mean"], dtype=np.float32)
        components = np.ascontiguousarray(archive["components"], dtype=np.float32)
    if mean.shape != (768,) or components.shape != (128, 768):
        raise ValueError("CUB development PCA geometry differs")
    if not np.isfinite(mean).all() or not np.isfinite(components).all():
        raise ValueError("CUB development PCA values differ")
    return (
        CenteredPcaTransform(mean=torch.from_numpy(mean), components=torch.from_numpy(components)),
        metadata,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in (
        "cub-root",
        "cub-archive",
        "unicom-checkout",
        "source-checkpoint",
        "selected-checkpoint",
        "selected-training-receipt",
        "pca-receipt",
        "projection-archive",
        "output",
    ):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--execute-sop-fullwidth-cub-dev", action="store_true", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    if args.output.exists() or args.output.is_symlink() or args.workers < 0:
        raise ValueError("CUB development output or worker authority differs")
    torch.set_num_threads(16)
    torch.backends.cuda.matmul.allow_tf32 = False
    source_sha256 = source_manifest()
    training_bytes = args.selected_training_receipt.read_bytes()
    pca_bytes = args.pca_receipt.read_bytes()
    projection_bytes = args.projection_archive.read_bytes()
    training_digest = hashlib.sha256(training_bytes).hexdigest()
    pca_digest = hashlib.sha256(pca_bytes).hexdigest()
    projection_digest = hashlib.sha256(projection_bytes).hexdigest()
    if (
        training_digest != SELECTED_TRAINING_RECEIPT_SHA256
        or pca_digest != PCA_RECEIPT_SHA256
        or projection_digest != PROJECTION_ARCHIVE_SHA256
    ):
        raise ValueError("CUB development pinned evidence differs")
    training = json.loads(training_bytes)
    pca_receipt = json.loads(pca_bytes)
    projection, projection_metadata = load_projection(projection_bytes)
    del projection_bytes
    checkpoint_bytes = args.selected_checkpoint.read_bytes()
    checkpoint_digest = hashlib.sha256(checkpoint_bytes).hexdigest()
    validate_selected_authority(training, checkpoint_digest, training_digest)
    validate_pca_authority(pca_receipt, projection_metadata, checkpoint_digest, projection_digest)
    pca_inputs = pca_receipt["inputs"]
    if (
        sha256(args.source_checkpoint) != CHECKPOINT_SHA256
        or pca_inputs.get("fullwidth_receipt_sha256") != training_digest
        or sha256(args.cub_archive) != CUB_ARCHIVE_SHA256
    ):
        raise ValueError("CUB development source authority differs")
    checkpoint = torch.load(BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True)
    del checkpoint_bytes
    if not isinstance(checkpoint, Mapping) or checkpoint.get("updates") != 48000:
        raise ValueError("CUB development checkpoint differs")
    records = parse_cub_records(args.cub_root)
    content_digest = verify_extracted_cub_matches_archive(args.cub_archive, args.cub_root, records)
    selected = select_development_records(records)
    manifest = b"".join(
        hashlib.sha256(record.image_path.read_bytes()).digest() for record in selected
    )
    preflight_seconds = time.perf_counter() - started
    if not torch.cuda.is_available() or gpu_compute_pids():
        raise ValueError("CUB development GPU is unavailable or occupied")
    authenticated = load_authenticated_source_model(args.unicom_checkout, args.source_checkpoint)
    model = authenticated.encoder.cuda().eval()
    head = nn.Linear(768, 768)
    with torch.no_grad():
        nn.init.eye_(head.weight)
        nn.init.zeros_(head.bias)
    head = head.cuda().eval()
    loader = DataLoader(
        VerifiedCubDevelopmentImages(selected, authenticated.transform, manifest),
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    torch.cuda.reset_peak_memory_stats()
    pretrained, pretrained_encode_seconds = encode_fullwidth(model, head, loader)
    model.load_state_dict(checkpoint["model"], strict=True)
    head.load_state_dict(checkpoint["head"], strict=True)
    del checkpoint
    fullwidth, encode_seconds = encode_fullwidth(model, head, loader)
    projected = projection.apply(fullwidth)
    labels = torch.tensor([record.label for record in selected], dtype=torch.int64)
    scored_started = time.perf_counter()
    pretrained_results = score_features(pretrained, labels)
    fullwidth_results = score_features(fullwidth, labels)
    projected_results = score_features(projected, labels)
    score_seconds = time.perf_counter() - scored_started
    if gpu_compute_pids() - {os.getpid()}:
        raise ValueError("CUB development GPU process overlap")
    if source_manifest() != source_sha256:
        raise ValueError("CUB development evaluator source changed")
    result = {
        "schema": "sfora-sop-fullwidth-cub-development-v1",
        "claim_eligible": False,
        "protocol": "CUB-200-2011 classes 1-100 development self retrieval; no CUB fitting",
        "queries": len(selected),
        "classes": 100,
        "image_ids": [record.image_id for record in selected],
        "labels": [record.label for record in selected],
        "fullwidth_packed_bytes_per_item": 770,
        "projected_packed_bytes_per_item": 130,
        "pretrained_fullwidth": pretrained_results,
        "fullwidth": fullwidth_results,
        "pca128": projected_results,
        "pretrained_encode_seconds": pretrained_encode_seconds,
        "encode_seconds": encode_seconds,
        "score_seconds": score_seconds,
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
            "selected_checkpoint_sha256": checkpoint_digest,
            "selected_training_receipt_sha256": training_digest,
            "pca_receipt_sha256": pca_digest,
            "projection_archive_sha256": projection_digest,
            "cub_archive_sha256": CUB_ARCHIVE_SHA256,
            "cub_content_sha256": content_digest,
            "ordered_records_sha256": ordered_record_sha256(selected),
            "image_manifest_sha256": hashlib.sha256(manifest).hexdigest(),
            "feature_array_sha256": {
                "pretrained_fullwidth": array_sha256(pretrained),
                "fullwidth": array_sha256(fullwidth),
                "pca128": array_sha256(projected),
            },
            "source_sha256": source_sha256,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def write_receipt(stream: BinaryIO) -> None:
        stream.write((json.dumps(result, sort_keys=True, allow_nan=False) + "\n").encode())

    publish_file_noreplace(args.output, write_receipt)
    print(json.dumps({"output": str(args.output), "queries": len(selected)}))


if __name__ == "__main__":
    main()
