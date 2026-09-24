#!/usr/bin/env python3
"""Measure CUB transfer of the train-selected SOP compact checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from io import BytesIO
from pathlib import Path
from typing import BinaryIO

import numpy as np
import torch
from export_unicom_cub_embeddings import (
    CUB_ARCHIVE_SHA256,
    CubRecord,
    ordered_record_sha256,
    parse_cub_records,
    verify_extracted_cub_matches_archive,
)
from export_unicom_sop_embeddings import parse_sop_records
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
    SPLIT_SEED,
    initialize_head_and_classifier,
    publish_file_noreplace,
    sha256,
)

from sfora.joint_relational_compaction import pack_int8_unit_embeddings
from sfora.representation_ceiling import deterministic_class_partition
from sfora.sop_compact_training import compact_head_features
from sfora.sop_evaluation import score_symmetric


def validate_transfer_authority(
    official: Mapping[str, object],
    training: Mapping[str, object],
    checkpoint_sha256: str,
    training_receipt_sha256: str,
) -> None:
    """Bind the transfer model to a completed train-only checkpoint selection."""

    inputs = official.get("inputs")
    step = official.get("selection_step")
    training_inputs = training.get("inputs")
    training_digest = (
        training_inputs.get("checkpoint_output_sha256")
        if isinstance(training_inputs, Mapping)
        else training.get("checkpoint_sha256")
    )
    if (
        official.get("schema") != "sfora-sop-reference-official-test-v1"
        or official.get("claim_eligible") is not False
        or official.get("embedding_width") != 128
        or type(step) is not int
        or not isinstance(inputs, Mapping)
        or not isinstance(inputs.get("training_receipt_sha256"), Mapping)
        or inputs["training_receipt_sha256"].get(str(step)) != training_receipt_sha256
        or inputs.get("selected_checkpoint_sha256") != checkpoint_sha256
        or training.get("schema")
        not in (
            "sfora-sop-compact-training-diagnostic-v1",
            "sfora-sop-compact-full-backbone-v1",
        )
        or training.get("arm") != "arcface"
        or training.get("recipe") != "reference"
        or training.get("seed") != official.get("seed")
        or training.get("embedding_width", 128) != 128
        or training.get("step", training.get("updates")) != step
        or training_digest != checkpoint_sha256
    ):
        raise ValueError("CUB transfer authority differs")


class VerifiedCubImages(Dataset[tuple[torch.Tensor, int]]):
    """Decode the same CUB test-image bytes hashed before model inference."""

    def __init__(
        self,
        records: tuple[CubRecord, ...],
        transform: Callable[[Image.Image], torch.Tensor],
        manifest: bytes,
    ) -> None:
        if len(records) != 5924 or len(manifest) != len(records) * 32:
            raise ValueError("CUB transfer image manifest differs")
        self.records = records
        self.transform = transform
        self.manifest = manifest

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        record = self.records[index]
        image_bytes = record.image_path.read_bytes()
        if hashlib.sha256(image_bytes).digest() != self.manifest[index * 32 : (index + 1) * 32]:
            raise ValueError("CUB transfer image bytes changed")
        with Image.open(BytesIO(image_bytes)) as image:
            return self.transform(image.convert("RGB")), record.label


def gpu_compute_pids() -> set[int]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        tokens = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if any(not token.isdecimal() for token in tokens):
            raise ValueError("CUB transfer GPU inventory differs")
        return {int(token) for token in tokens}
    except (OSError, subprocess.SubprocessError) as error:
        raise ValueError("CUB transfer GPU inventory differs") from error


def evaluator_source_manifest() -> dict[str, str]:
    """Hash the source files that provided the modules used by this process."""

    root = Path(__file__).resolve().parents[1]
    relatives = (
        "scripts/evaluate_sop_cub_transfer.py",
        "scripts/export_unicom_cub_embeddings.py",
        "scripts/export_unicom_sop_embeddings.py",
        "scripts/train_sop_compact_backbone.py",
        "scripts/sop_teacher_anchored_runtime.py",
        "src/sfora/joint_relational_compaction.py",
        "src/sfora/representation_ceiling.py",
        "src/sfora/sop_compact_training.py",
        "src/sfora/sop_evaluation.py",
    )
    manifest: dict[str, str] = {}
    for relative in relatives:
        path = Path(relative)
        loaded = (
            sys.modules[__name__]
            if relative == "scripts/evaluate_sop_cub_transfer.py"
            else sys.modules.get(path.stem if path.parts[0] == "scripts" else f"sfora.{path.stem}")
        )
        loaded_file = getattr(loaded, "__file__", None)
        if (
            loaded is None
            or type(loaded_file) is not str
            or Path(loaded_file).resolve() != (root / path).resolve()
        ):
            raise ValueError("CUB transfer source imports differ")
        manifest[relative] = sha256(root / path)
    return manifest


def load_sop_train_only_archive(path: Path) -> dict[str, np.ndarray]:
    """Read and authenticate only SOP train arrays from the pinned archive."""

    train_names = ("train_embeddings", "train_labels", "train_image_ids")
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
        arrays = {name: archive[name].copy() for name in train_names}
    if (
        type(metadata) is not dict
        or metadata.get("schema") != "sfora-unicom-sop-embeddings-v1"
        or metadata.get("checkpoint_sha256") != CHECKPOINT_SHA256
        or metadata.get("embedding_dimension") != 768
        or metadata.get("split_counts") != {"train": 59551, "test": 60502}
        or arrays["train_embeddings"].shape != (59551, 768)
        or arrays["train_embeddings"].dtype != np.float32
        or not np.isfinite(arrays["train_embeddings"]).all()
        or arrays["train_labels"].shape != (59551,)
        or arrays["train_labels"].dtype != np.int64
        or arrays["train_image_ids"].shape != (59551,)
        or arrays["train_image_ids"].dtype != np.int64
        or not isinstance(metadata.get("array_sha256"), dict)
        or any(type(metadata["array_sha256"].get(name)) is not str for name in train_names)
    ):
        raise ValueError("CUB transfer SOP train archive differs")
    for name, array in arrays.items():
        if hashlib.sha256(array.tobytes(order="C")).hexdigest() != metadata["array_sha256"].get(
            name
        ):
            raise ValueError("CUB transfer SOP train array differs")
    return arrays


def reconstruct_initial_head(args: argparse.Namespace, final: Mapping[str, object]) -> nn.Linear:
    """Rebuild the fit-only PCA head used at step zero, with a receipt digest check."""

    if sha256(args.sop_features_archive) != ARCHIVE_SHA256:
        raise ValueError("CUB transfer source archive differs")
    records = parse_sop_records(args.sop_dataset_root)
    train = tuple(record for record in records if record.split == "train")
    archive = load_sop_train_only_archive(args.sop_features_archive)
    if tuple(record.image_id for record in train) != tuple(
        int(value) for value in archive["train_image_ids"]
    ) or tuple(record.label for record in train) != tuple(
        int(value) for value in archive["train_labels"]
    ):
        raise ValueError("CUB transfer source archive rows differ")
    partition = deterministic_class_partition(
        tuple(record.label for record in train), fit_fraction=FIT_FRACTION, seed=SPLIT_SEED
    )
    if hashlib.sha256(
        np.asarray(partition.fit_row_indexes, dtype="<i4").tobytes(order="C")
    ).hexdigest() != final.get("fit_row_indexes_sha256"):
        raise ValueError("CUB transfer fit partition differs")
    fit_features = torch.from_numpy(
        np.ascontiguousarray(archive["train_embeddings"][list(partition.fit_row_indexes)])
    ).float()
    fit_labels = tuple(train[index].label for index in partition.fit_row_indexes)
    head, _classifier = initialize_head_and_classifier(fit_features, fit_labels)
    digest = hashlib.sha256(
        head.weight.detach().numpy().tobytes(order="C")
        + head.bias.detach().numpy().tobytes(order="C")
    ).hexdigest()
    inputs = final.get("inputs")
    if not isinstance(inputs, Mapping) or digest != inputs.get("initial_head_sha256"):
        raise ValueError("CUB transfer initial head differs")
    return head


@torch.inference_mode()
def encode_cub(
    model: nn.Module, head: nn.Linear, loader: DataLoader[tuple[torch.Tensor, int]]
) -> tuple[torch.Tensor, float]:
    model.eval()
    head.eval()
    started = time.perf_counter()
    outputs: list[torch.Tensor] = []
    for images, _labels in loader:
        source = model(images.cuda(non_blocking=True))
        outputs.append(F.normalize(compact_head_features(source, head), dim=1).cpu())
    values = torch.cat(outputs).contiguous()
    if values.shape != (5924, 128) or not bool(torch.isfinite(values).all()):
        raise ValueError("CUB transfer features differ")
    return values, time.perf_counter() - started


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in (
        "cub-root",
        "cub-archive",
        "sop-dataset-root",
        "sop-features-archive",
        "unicom-checkout",
        "source-checkpoint",
        "selected-checkpoint",
        "selected-training-receipt",
        "final-training-receipt",
        "official-receipt",
        "features-output",
        "output",
    ):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--expected-official-receipt-sha256", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--execute-sop-cub-transfer", action="store_true", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started_all = time.perf_counter()
    if (
        args.output.exists()
        or args.output.is_symlink()
        or args.features_output.exists()
        or args.features_output.is_symlink()
        or args.output.resolve() == args.features_output.resolve()
        or args.workers < 0
    ):
        raise ValueError("CUB transfer invocation differs")
    torch.set_num_threads(16)
    source_manifest = evaluator_source_manifest()
    official_digest = sha256(args.official_receipt)
    if official_digest != args.expected_official_receipt_sha256:
        raise ValueError("CUB transfer official receipt differs")
    official = json.loads(args.official_receipt.read_text())
    selected_digest = sha256(args.selected_checkpoint)
    training_digest = sha256(args.selected_training_receipt)
    training = json.loads(args.selected_training_receipt.read_text())
    final_digest = sha256(args.final_training_receipt)
    final = json.loads(args.final_training_receipt.read_text())
    validate_transfer_authority(official, training, selected_digest, training_digest)
    inputs = official["inputs"]
    if (
        final.get("schema") != "sfora-sop-compact-full-backbone-v1"
        or final.get("arm") != "arcface"
        or final.get("recipe") != "reference"
        or final.get("seed") != official["seed"]
        or final.get("updates") != 53760
        or inputs["training_receipt_sha256"].get("53760") != final_digest
        or sha256(args.source_checkpoint) != CHECKPOINT_SHA256
        or sha256(args.cub_archive) != CUB_ARCHIVE_SHA256
    ):
        raise ValueError("CUB transfer source authority differs")
    claim_path = args.sop_dataset_root / Path(inputs["official_test_claim_path"]).name
    claim = json.loads(claim_path.read_text())
    final_inputs = final.get("inputs")
    if (
        sha256(claim_path) != inputs.get("official_test_claim_sha256")
        or claim.get("schema") != "sfora-sop-official-test-claim-v1"
        or claim.get("final_training_receipt_sha256") != final_digest
        or not isinstance(final_inputs, Mapping)
        or claim.get("final_training_checkpoint_sha256")
        != final_inputs.get("checkpoint_output_sha256")
    ):
        raise ValueError("CUB transfer official claim differs")
    trained = torch.load(args.selected_checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(trained, Mapping) or trained.get("updates") != official["selection_step"]:
        raise ValueError("CUB transfer selected checkpoint differs")
    initial_head = reconstruct_initial_head(args, final)
    records = parse_cub_records(args.cub_root)
    content_digest = verify_extracted_cub_matches_archive(args.cub_archive, args.cub_root, records)
    selected = tuple(record for record in records if record.split == "test")
    if len(selected) != 5924:
        raise ValueError("CUB transfer test split differs")
    manifest = b"".join(
        hashlib.sha256(record.image_path.read_bytes()).digest() for record in selected
    )
    if source_manifest != evaluator_source_manifest():
        raise ValueError("CUB transfer evaluator source changed")
    if args.preflight_only:
        print(json.dumps({"preflight": "passed", "selected_sop_step": official["selection_step"]}))
        return
    if gpu_compute_pids() or not torch.cuda.is_available():
        raise ValueError("CUB transfer GPU is occupied")
    preflight_seconds = time.perf_counter() - started_all
    torch.backends.cuda.matmul.allow_tf32 = False
    started = time.perf_counter()
    authenticated = load_authenticated_source_model(args.unicom_checkout, args.source_checkpoint)
    model = authenticated.encoder.cuda().eval()
    initial_head = initial_head.cuda().eval()
    loader = DataLoader(
        VerifiedCubImages(selected, authenticated.transform, manifest),
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    torch.cuda.reset_peak_memory_stats()
    baseline_values, baseline_encode_seconds = encode_cub(model, initial_head, loader)
    model.load_state_dict(trained["model"], strict=True)
    trained_head = nn.Linear(768, 128)
    trained_head.load_state_dict(trained["head"], strict=True)
    trained_head = trained_head.cuda().eval()
    trained_values, trained_encode_seconds = encode_cub(model, trained_head, loader)
    labels = torch.tensor([record.label for record in selected], dtype=torch.int64, device="cuda")
    results: dict[str, dict[str, Mapping[str, object]]] = {}
    for name, values in (("pretrained_initial_head", baseline_values), ("trained", trained_values)):
        packed = pack_int8_unit_embeddings(values)
        float_score = score_symmetric(values.cuda(), labels)
        packed_score = score_symmetric(
            packed.codes.float().cuda(), labels, inverse_norms=packed.inverse_norms.cuda()
        )
        results[name] = {"float": float_score, "packed": packed_score}
    if len(gpu_compute_pids()) > 1:
        raise ValueError("CUB transfer GPU became contended")
    if source_manifest != evaluator_source_manifest():
        raise ValueError("CUB transfer evaluator source changed")
    args.features_output.parent.mkdir(parents=True, exist_ok=True)

    def write_features(stream: BinaryIO) -> None:
        np.savez(
            stream,
            baseline=baseline_values.numpy(),
            trained=trained_values.numpy(),
            labels=np.asarray([record.label for record in selected], dtype=np.int64),
            image_ids=np.asarray([record.image_id for record in selected], dtype=np.int64),
            manifest_sha256=np.asarray(hashlib.sha256(manifest).hexdigest()),
        )

    publish_file_noreplace(args.features_output, write_features)
    receipt = {
        "schema": "sfora-sop-cub-transfer-v1",
        "claim_eligible": False,
        "protocol": "CUB classes 101-200 test self retrieval; no CUB model fitting",
        "selected_sop_step": official["selection_step"],
        "seed": official["seed"],
        "query_count": len(selected),
        "gallery_bytes_per_item": 130,
        "image_ids": [record.image_id for record in selected],
        "labels": [record.label for record in selected],
        "results": results,
        "pretrained_encode_seconds": baseline_encode_seconds,
        "trained_encode_seconds": trained_encode_seconds,
        "preflight_seconds": preflight_seconds,
        "inference_and_scoring_seconds": time.perf_counter() - started,
        "total_seconds": time.perf_counter() - started_all,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_host_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "hardware": {
            "gpu": torch.cuda.get_device_name(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "inputs": {
            "official_receipt_sha256": official_digest,
            "selected_training_receipt_sha256": training_digest,
            "final_training_receipt_sha256": final_digest,
            "selected_checkpoint_sha256": selected_digest,
            "source_checkpoint_sha256": CHECKPOINT_SHA256,
            "sop_features_archive_sha256": ARCHIVE_SHA256,
            "cub_archive_sha256": CUB_ARCHIVE_SHA256,
            "cub_content_sha256": content_digest,
            "cub_ordered_records_sha256": ordered_record_sha256(records),
            "cub_test_manifest_sha256": hashlib.sha256(manifest).hexdigest(),
            "features_sha256": sha256(args.features_output),
            "feature_array_sha256": {
                "baseline": hashlib.sha256(baseline_values.numpy().tobytes()).hexdigest(),
                "trained": hashlib.sha256(trained_values.numpy().tobytes()).hexdigest(),
                "labels": hashlib.sha256(
                    np.asarray([record.label for record in selected], dtype="<i8").tobytes()
                ).hexdigest(),
                "image_ids": hashlib.sha256(
                    np.asarray([record.image_id for record in selected], dtype="<i8").tobytes()
                ).hexdigest(),
            },
            "initial_head_sha256": final_inputs["initial_head_sha256"],
            "fit_row_indexes_sha256": final["fit_row_indexes_sha256"],
            "source_sha256": source_manifest,
        },
        "argv": sys.argv,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(receipt, sort_keys=True, allow_nan=False) + "\n").encode()

    def write_receipt(stream: BinaryIO) -> None:
        stream.write(payload)

    try:
        publish_file_noreplace(args.output, write_receipt)
    except BaseException:
        args.features_output.unlink()
        raise
    print(
        json.dumps(
            {
                "receipt": str(args.output),
                "pretrained_packed_r1": results["pretrained_initial_head"]["packed"]["recall_at_1"],
                "trained_packed_r1": results["trained"]["packed"]["recall_at_1"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
