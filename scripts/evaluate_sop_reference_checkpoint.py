#!/usr/bin/env python3
"""Evaluate one completed, train-selected reference SOP checkpoint on official test."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
import sys
import tempfile
import time
from collections.abc import Mapping
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from export_unicom_sop_embeddings import ordered_record_sha256
from PIL import Image
from sop_teacher_anchored_runtime import load_authenticated_source_model
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from train_sop_compact_backbone import (
    ARCHIVE_SHA256,
    CHECKPOINT_SHA256,
    EVAL_BATCH_SIZE,
    FIT_FRACTION,
    SOURCE_RELATIVES,
    SPLIT_SEED,
    UPSTREAM_RETRIEVAL_SHA256,
    UPSTREAM_SOP_B16_LAUNCH_SHA256,
    parse_sop_records,
    publish_file_noreplace,
    sha256,
)

from sfora.joint_relational_compaction import pack_int8_unit_embeddings
from sfora.representation_ceiling import deterministic_class_partition
from sfora.sop_compact_training import compact_head_features
from sfora.sop_evaluation import score_symmetric
from sfora.sop_reference_selection import (
    REFERENCE_STEPS,
    ReferenceSelection,
    select_reference_checkpoint,
)

EXPECTED_SOP_RECORD_SHA256 = "ea323eb87568d3f6ab88372ca5d1bf9a7033811cc958f899c532886f526fe992"
EXPECTED_TEST_IMAGE_MANIFEST_SHA256 = (
    "28a3ec0561cd83ee426f3d1c301c70799316af91c1e9083a5a1ffdf3414327c1"
)
EVALUATOR_RELATIVES = (
    *SOURCE_RELATIVES,
    "src/sfora/sop_reference_selection.py",
    "scripts/evaluate_sop_reference_checkpoint.py",
)


def training_source_manifest(root: Path) -> dict[str, str]:
    """Hash the immutable source snapshot named by the training receipts."""

    if not root.is_dir() or any(not (root / relative).is_file() for relative in SOURCE_RELATIVES):
        raise ValueError("reference training source snapshot differs")
    return {relative: sha256(root / relative) for relative in SOURCE_RELATIVES}


def evaluator_source_manifest() -> dict[str, str]:
    """Bind metric code hashes to the modules actually loaded in this process."""

    root = Path(__file__).resolve().parents[1]
    manifest: dict[str, str] = {}
    for relative in EVALUATOR_RELATIVES:
        path = Path(relative)
        if relative == "scripts/evaluate_sop_reference_checkpoint.py":
            loaded = sys.modules[__name__]
        elif path.parts[0] == "scripts":
            loaded = sys.modules.get(path.stem)
        else:
            loaded = sys.modules.get(f"sfora.{path.stem}")
        if (
            loaded is None
            or not getattr(loaded, "__file__", None)
            or Path(loaded.__file__).resolve() != (root / path).resolve()
        ):
            raise ValueError("reference SOP evaluator source imports differ")
        manifest[relative] = sha256(root / path)
    return manifest


def claim_official_test_once(
    final_receipt: Path,
    final_checkpoint_sha256: str,
    dataset_root: Path,
    output: Path,
) -> Path:
    """Reserve this completed training run before any official images are read."""

    if (
        not dataset_root.is_dir()
        or len(final_checkpoint_sha256) != 64
        or any(character not in "0123456789abcdef" for character in final_checkpoint_sha256)
    ):
        raise ValueError("reference SOP official test claim authority differs")
    receipt_digest = sha256(final_receipt)
    claim = dataset_root / f".sfora-sop-official-{final_checkpoint_sha256}.claim.json"
    body = (
        json.dumps(
            {
                "schema": "sfora-sop-official-test-claim-v1",
                "final_training_receipt_sha256": receipt_digest,
                "final_training_checkpoint_sha256": final_checkpoint_sha256,
                "intended_output": str(output.resolve()),
                "claimed_at_unix_seconds": time.time(),
            },
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode()
    try:
        publish_file_noreplace(claim, lambda stream: stream.write(body))
    except FileExistsError:
        raise ValueError("reference SOP official test already claimed") from None
    return claim


class VerifiedImages(Dataset):
    """Decode only the exact image bytes pinned in an ordered digest manifest."""

    def __init__(self, records, transform, manifest: bytes) -> None:
        if not records or len(manifest) != len(records) * hashlib.sha256().digest_size:
            raise ValueError("reference SOP image manifest inventory differs")
        self.records = records
        self.transform = transform
        self.manifest = manifest

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        image_bytes = record.image_path.read_bytes()
        expected = self.manifest[index * 32 : (index + 1) * 32]
        if hashlib.sha256(image_bytes).digest() != expected:
            raise ValueError("reference SOP image content differs from pinned manifest")
        with Image.open(BytesIO(image_bytes)) as image:
            return self.transform(image.convert("RGB")), record.label


def load_test_image_manifest(path: Path, expected_sha256: str, *, image_count: int) -> bytes:
    """Load one predeclared byte-level corpus pin before reading test images."""

    manifest = path.read_bytes()
    if len(manifest) != image_count * 32:
        raise ValueError("reference SOP image manifest inventory differs")
    if hashlib.sha256(manifest).hexdigest() != expected_sha256:
        raise ValueError("reference SOP image manifest digest differs")
    return manifest


def validate_selected_payload(payload: object, selection: ReferenceSelection) -> int:
    """Bind loaded model tensors to the selected receipt and output width."""

    receipt = selection.receipt
    width = receipt.get("embedding_width", 128)
    if not isinstance(payload, Mapping):
        raise ValueError("reference selected checkpoint differs")
    head = payload.get("head")
    classifier = payload.get("classifier")
    if (
        width not in (128, 768)
        or payload.get("recipe") != "reference"
        or payload.get("arm") != "arcface"
        or payload.get("seed") != receipt.get("seed")
        or payload.get("updates") != selection.step
        or payload.get("embedding_width", 128) != width
        or payload.get("schedule_sha256") != receipt.get("schedule_sha256")
        or not isinstance(payload.get("model"), Mapping)
        or not isinstance(head, Mapping)
        or not isinstance(head.get("weight"), torch.Tensor)
        or head["weight"].shape != (width, 768)
        or not isinstance(head.get("bias"), torch.Tensor)
        or head["bias"].shape != (width,)
        or not isinstance(classifier, torch.Tensor)
        or classifier.shape != (10_186, width)
    ):
        raise ValueError("reference selected checkpoint differs")
    return width


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--unicom-checkout", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--training-source-root", type=Path, required=True)
    parser.add_argument("--test-image-manifest", type=Path, required=True)
    parser.add_argument(
        "--candidate",
        type=Path,
        nargs=2,
        action="append",
        required=True,
        metavar=("CHECKPOINT", "RECEIPT"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--execute-official-sop-reference-evaluation", action="store_true", required=True
    )
    args = parser.parse_args()
    if args.workers < 0 or args.output.exists() or len(args.candidate) != len(REFERENCE_STEPS):
        parser.error("reference SOP official evaluation invocation differs")
    return args


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise ValueError("reference SOP official evaluation requires CUDA")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryFile(dir=args.output.parent):
        pass
    evaluator_sha256 = sha256(Path(__file__))
    evaluator_source = evaluator_source_manifest()
    source_manifest = training_source_manifest(args.training_source_root)
    receipt_bytes = [path.read_bytes() for _, path in args.candidate]
    receipt_digests = [hashlib.sha256(data).hexdigest() for data in receipt_bytes]
    receipts = [json.loads(data) for data in receipt_bytes]
    selection = select_reference_checkpoint(
        [
            (checkpoint, receipt)
            for (checkpoint, _), receipt in zip(args.candidate, receipts, strict=True)
        ],
        source_manifest=source_manifest,
    )
    final = next(receipt for receipt in receipts if receipt.get("updates") == REFERENCE_STEPS[-1])
    inputs = final["inputs"]
    if (
        sha256(args.source_checkpoint) != CHECKPOINT_SHA256
        or inputs["checkpoint_sha256"] != CHECKPOINT_SHA256
        or inputs["features_archive_sha256"] != ARCHIVE_SHA256
        or inputs["upstream_retrieval_sha256"] != UPSTREAM_RETRIEVAL_SHA256
        or inputs["upstream_launch_sha256"] != UPSTREAM_SOP_B16_LAUNCH_SHA256
        or sha256(args.dataset_root / "Ebay_train.txt") != inputs["sop_train_metadata_sha256"]
    ):
        raise ValueError("reference SOP training authority differs")
    final_receipt_path = next(
        path
        for (_, path), receipt in zip(args.candidate, receipts, strict=True)
        if receipt.get("updates") == REFERENCE_STEPS[-1]
    )
    if any(
        sha256(path) != digest
        for (_, path), digest in zip(args.candidate, receipt_digests, strict=True)
    ):
        raise ValueError("reference SOP training receipt changed during preflight")
    all_records = parse_sop_records(args.dataset_root)
    if ordered_record_sha256(all_records) != EXPECTED_SOP_RECORD_SHA256:
        raise ValueError("reference SOP official image inventory differs")
    train_records = tuple(record for record in all_records if record.split == "train")
    partition = deterministic_class_partition(
        tuple(record.label for record in train_records),
        fit_fraction=FIT_FRACTION,
        seed=SPLIT_SEED,
    )
    validation_records = tuple(train_records[index] for index in partition.validation_row_indexes)
    if (
        [record.image_id for record in validation_records] != final["validation_image_ids"]
        or [record.label for record in validation_records] != final["validation_labels"]
        or hashlib.sha256(
            np.asarray(partition.fit_row_indexes, dtype="<i4").tobytes(order="C")
        ).hexdigest()
        != final["fit_row_indexes_sha256"]
        or hashlib.sha256(
            np.asarray(partition.validation_row_indexes, dtype="<i4").tobytes(order="C")
        ).hexdigest()
        != final["validation_row_indexes_sha256"]
    ):
        raise ValueError("reference SOP train-only selection inventory differs")
    trained = torch.load(selection.checkpoint, map_location="cpu", weights_only=True)
    width = validate_selected_payload(trained, selection)
    selected_checkpoint_digest = (
        selection.receipt["inputs"]["checkpoint_output_sha256"]
        if selection.step == REFERENCE_STEPS[-1]
        else selection.receipt["checkpoint_sha256"]
    )
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(16)
    authenticated = load_authenticated_source_model(args.unicom_checkout, args.source_checkpoint)
    model = authenticated.encoder
    model.load_state_dict(trained["model"], strict=True)
    model = model.cuda().eval()
    head = nn.Linear(768, width)
    head.load_state_dict(trained["head"], strict=True)
    head = head.cuda().eval()
    records = tuple(record for record in all_records if record.split == "test")
    labels = tuple(record.label for record in records)
    if len(records) != 60_502 or len(set(labels)) != 11_316:
        raise ValueError("reference SOP official test inventory differs")
    test_image_manifest = load_test_image_manifest(
        args.test_image_manifest,
        EXPECTED_TEST_IMAGE_MANIFEST_SHA256,
        image_count=len(records),
    )
    loader = DataLoader(
        VerifiedImages(records, authenticated.transform, test_image_manifest),
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    claim = claim_official_test_once(
        final_receipt_path,
        inputs["checkpoint_output_sha256"],
        args.dataset_root,
        args.output,
    )
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    outputs: list[torch.Tensor] = []
    with torch.inference_mode():
        for images, _ in loader:
            source = model(images.cuda(non_blocking=True))
            features = compact_head_features(source, head, output_dim=width)
            outputs.append(F.normalize(features, dim=1).cpu())
    encode_seconds = time.perf_counter() - started
    values = torch.cat(outputs).contiguous()
    if values.shape != (60_502, width):
        raise ValueError("reference SOP official feature inventory differs")
    pack_started = time.perf_counter()
    packed = pack_int8_unit_embeddings(values)
    pack_seconds = time.perf_counter() - pack_started
    label_tensor = torch.tensor(labels, dtype=torch.int64, device="cuda")
    float_started = time.perf_counter()
    float_score = score_symmetric(values.cuda(), label_tensor)
    float_score_seconds = time.perf_counter() - float_started
    packed_started = time.perf_counter()
    packed_score = score_symmetric(
        packed.codes.float().cuda(), label_tensor, inverse_norms=packed.inverse_norms.cuda()
    )
    packed_score_seconds = time.perf_counter() - packed_started
    if (
        sha256(Path(__file__)) != evaluator_sha256
        or evaluator_source_manifest() != evaluator_source
        or training_source_manifest(args.training_source_root) != source_manifest
        or sha256(selection.checkpoint) != selected_checkpoint_digest
        or any(
            sha256(path) != digest
            for (_, path), digest in zip(args.candidate, receipt_digests, strict=True)
        )
    ):
        raise ValueError("reference SOP evaluation source changed during scoring")
    result = {
        "schema": "sfora-sop-reference-official-test-v1",
        "claim_eligible": False,
        "selection": (
            "completed train-identity holdout, packed mAP@R then Recall@1 then earlier step"
        ),
        "selection_step": selection.step,
        "selection_scores": {
            str(receipt.get("updates", receipt.get("step"))): {
                "packed_map_at_r": receipt["validation"]["packed"]["map_at_r"],
                "packed_recall_at_1": receipt["validation"]["packed"]["recall_at_1"],
            }
            for receipt in receipts
        },
        "seed": selection.receipt["seed"],
        "embedding_width": width,
        "test_images": len(records),
        "test_classes": len(set(labels)),
        "test_image_ids": [record.image_id for record in records],
        "test_labels": labels,
        "float": float_score,
        "packed": packed_score,
        "gallery_bytes_per_item": width + 2,
        "encode_seconds": encode_seconds,
        "pack_seconds": pack_seconds,
        "float_score_seconds": float_score_seconds,
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
            "selected_checkpoint_sha256": selected_checkpoint_digest,
            "training_receipt_sha256": {
                str(receipt.get("updates", receipt.get("step"))): digest
                for receipt, digest in zip(receipts, receipt_digests, strict=True)
            },
            "sop_train_metadata_sha256": inputs["sop_train_metadata_sha256"],
            "sop_test_metadata_sha256": sha256(args.dataset_root / "Ebay_test.txt"),
            "ordered_test_records_sha256": ordered_record_sha256(records),
            "test_image_manifest_sha256": EXPECTED_TEST_IMAGE_MANIFEST_SHA256,
            "training_source_sha256": source_manifest,
            "evaluator_sha256": evaluator_sha256,
            "evaluator_source_sha256": evaluator_source,
            "official_test_claim_path": str(claim),
            "official_test_claim_sha256": sha256(claim),
        },
        "argv": sys.argv,
    }
    publish_file_noreplace(
        args.output,
        lambda stream: stream.write(
            (json.dumps(result, sort_keys=True, allow_nan=False) + "\n").encode()
        ),
    )
    print(
        json.dumps(
            {
                "selection_step": selection.step,
                "float_recall_at_1": float_score["recall_at_1"],
                "packed_recall_at_1": packed_score["recall_at_1"],
                "receipt": str(args.output),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
