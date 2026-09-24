#!/usr/bin/env python3
"""Measure class-disjoint Cars196 transfer of the SOP-selected compact model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from importlib.metadata import version
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO

import numpy as np
import torch
from evaluate_sop_cub_transfer import (
    evaluator_source_manifest,
    gpu_compute_pids,
    reconstruct_initial_head,
    validate_transfer_authority,
)
from PIL import Image
from sop_teacher_anchored_runtime import load_authenticated_source_model
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from train_sop_compact_backbone import (
    ARCHIVE_SHA256,
    CHECKPOINT_SHA256,
    EVAL_BATCH_SIZE,
    publish_file_noreplace,
    sha256,
)

from sfora.joint_relational_compaction import pack_int8_unit_embeddings
from sfora.sop_compact_training import compact_head_features
from sfora.sop_evaluation import score_symmetric

if TYPE_CHECKING:
    from datasets import DatasetDict

DATASET_ID = "tanganke/stanford_cars"
DATASET_REVISION = "9abf6cf7d6dfa7b95152a0d6e791ea9435b47a40"
BASELINE_FEATURE_SHA256 = "6ddfd7e2c9fd489dff51fa33697c62abf92a45247b52b336c0371f5482231ab3"
CACHE_FILE_SHA256 = {
    "stanford_cars-train-00000-of-00002.arrow": (
        "f7c2469569f646288eb04d1b802592d27c5d10794802cccf6e8ac70c054a6b38"
    ),
    "stanford_cars-train-00001-of-00002.arrow": (
        "d7f9c6bbed128e06492b91b3a94dcdad10a65f3cd3c000db8c2418dfb253463c"
    ),
    "stanford_cars-test-00000-of-00002.arrow": (
        "f93eaec0f687268607af546a7cc0d36b496d66485ea0f5a6681b1393eb91ad13"
    ),
    "stanford_cars-test-00001-of-00002.arrow": (
        "f120f8dedf8ac0fec9416b7426c80cd8574b45d147a4627cbc8605b6821f19d4"
    ),
}
SOURCE_FINGERPRINTS = {"train": "c027c63962212a03", "test": "59e3e308b987bbb7"}
CarsRow = tuple[str, int, int]


def cars_source_manifest() -> dict[str, str]:
    loaded_file = getattr(sys.modules[__name__], "__file__", None)
    if type(loaded_file) is not str or Path(loaded_file).resolve() != Path(__file__).resolve():
        raise ValueError("Cars transfer evaluator source differs")
    return {
        **evaluator_source_manifest(),
        "scripts/evaluate_sop_cars_transfer.py": sha256(Path(__file__)),
    }


def select_evaluation_rows(
    train_labels: Sequence[int], test_labels: Sequence[int], *, expected_count: int = 8131
) -> tuple[CarsRow, ...]:
    """Use classes 98-195 from both original image partitions in source order."""

    if (
        type(expected_count) is not int
        or expected_count < 1
        or any(type(label) is not int or not 0 <= label < 196 for label in train_labels)
        or any(type(label) is not int or not 0 <= label < 196 for label in test_labels)
    ):
        raise ValueError("Cars transfer split differs")
    selected = tuple(
        (split, index, label)
        for split, labels in (("train", train_labels), ("test", test_labels))
        for index, label in enumerate(labels)
        if label >= 98
    )
    if len(selected) != expected_count:
        raise ValueError("Cars transfer split differs")
    return selected


def select_development_rows(
    train_labels: Sequence[int], test_labels: Sequence[int], *, expected_count: int = 8054
) -> tuple[CarsRow, ...]:
    """Use classes 0-97 from both original image partitions in source order."""

    if (
        type(expected_count) is not int
        or expected_count < 1
        or any(type(label) is not int or not 0 <= label < 196 for label in train_labels)
        or any(type(label) is not int or not 0 <= label < 196 for label in test_labels)
    ):
        raise ValueError("Cars transfer split differs")
    selected = tuple(
        (split, index, label)
        for split, labels in (("train", train_labels), ("test", test_labels))
        for index, label in enumerate(labels)
        if label < 98
    )
    if len(selected) != expected_count:
        raise ValueError("Cars transfer split differs")
    return selected


class VerifiedCarsImages(Dataset[tuple[torch.Tensor, int]]):
    """Decode only image bytes already bound to the frozen source manifest."""

    def __init__(
        self,
        source: DatasetDict,
        rows: tuple[CarsRow, ...],
        transform: Callable[[Image.Image], torch.Tensor],
        manifest: bytes,
    ) -> None:
        if len(rows) != 8131 or len(manifest) != 32 * len(rows):
            raise ValueError("Cars transfer image inventory differs")
        self.source = source
        self.rows = rows
        self.transform = transform
        self.manifest = manifest

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        split, source_index, label = self.rows[index]
        row = self.source[split][source_index]
        payload = row["image"]
        image_bytes = payload["bytes"]
        if (
            type(image_bytes) is not bytes
            or int(row["label"]) != label
            or hashlib.sha256(image_bytes).digest() != self.manifest[index * 32 : (index + 1) * 32]
        ):
            raise ValueError("Cars transfer image bytes changed")
        with Image.open(BytesIO(image_bytes)) as image:
            return self.transform(image.convert("RGB")), label


def authenticate_cars_source(
    baseline_path: Path,
) -> tuple[DatasetDict, tuple[CarsRow, ...], bytes, dict[str, str]]:
    """Bind cached image bytes and row order to the earlier Cars196 profile."""

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    from datasets import Image as HuggingFaceImage
    from datasets import load_dataset

    if sha256(baseline_path) != BASELINE_FEATURE_SHA256:
        raise ValueError("Cars transfer baseline archive differs")
    with np.load(baseline_path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
        expected_labels = np.asarray(archive["evaluation_labels"], dtype=np.int64)
    if (
        metadata.get("schema") != "scratch-cars196-unicom-class-disjoint-v1"
        or metadata.get("dataset_id") != DATASET_ID
        or metadata.get("dataset_revision") != DATASET_REVISION
        or metadata.get("source_fingerprints") != SOURCE_FINGERPRINTS
        or metadata.get("counts") != {"fit": 8054, "evaluation": 8131}
        or expected_labels.shape != (8131,)
    ):
        raise ValueError("Cars transfer baseline authority differs")
    source = load_dataset(DATASET_ID, revision=DATASET_REVISION)
    if (
        len(source["train"]) != 8144
        or len(source["test"]) != 8041
        or {split: source[split]._fingerprint for split in ("train", "test")} != SOURCE_FINGERPRINTS
    ):
        raise ValueError("Cars transfer source inventory differs")
    cache_entries = [
        Path(entry["filename"])
        for split in ("train", "test")
        for entry in source[split].cache_files
    ]
    cache_files = {path.name: path for path in cache_entries}
    if (
        len(cache_entries) != 4
        or set(cache_files) != set(CACHE_FILE_SHA256)
        or any(
            DATASET_REVISION not in path.parts or sha256(path) != CACHE_FILE_SHA256[name]
            for name, path in cache_files.items()
        )
    ):
        raise ValueError("Cars transfer source cache differs")
    labels = {
        split: tuple(int(value) for value in source[split]["label"]) for split in ("train", "test")
    }
    rows = select_evaluation_rows(labels["train"], labels["test"])
    if not np.array_equal(np.asarray([label for _, _, label in rows]), expected_labels):
        raise ValueError("Cars transfer source labels differ")
    for split in ("train", "test"):
        source[split] = source[split].cast_column("image", HuggingFaceImage(decode=False))
    manifest = bytearray()
    for split, index, label in rows:
        row = source[split][index]
        image_bytes = row["image"]["bytes"]
        if type(image_bytes) is not bytes or int(row["label"]) != label:
            raise ValueError("Cars transfer source image differs")
        manifest.extend(hashlib.sha256(image_bytes).digest())
    cache_manifest = {name: CACHE_FILE_SHA256[name] for name in sorted(cache_files)}
    return source, rows, bytes(manifest), cache_manifest


@torch.inference_mode()
def encode_cars(
    model: nn.Module,
    head: nn.Linear,
    loader: DataLoader[tuple[torch.Tensor, int]],
) -> tuple[torch.Tensor, float]:
    model.eval()
    head.eval()
    started = time.perf_counter()
    outputs: list[torch.Tensor] = []
    for images, _labels in loader:
        source = model(images.cuda(non_blocking=True))
        outputs.append(F.normalize(compact_head_features(source, head), dim=1).cpu())
    values = torch.cat(outputs).contiguous()
    if values.shape != (8131, 128) or not bool(torch.isfinite(values).all()):
        raise ValueError("Cars transfer features differ")
    return values, time.perf_counter() - started


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in (
        "sop-dataset-root",
        "sop-features-archive",
        "cars-baseline-archive",
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
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--crossed-factorial", action="store_true")
    parser.add_argument("--execute-sop-cars-transfer", action="store_true", required=True)
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
        raise ValueError("Cars transfer invocation differs")
    torch.set_num_threads(16)
    source_manifest = cars_source_manifest()
    official_digest = sha256(args.official_receipt)
    if official_digest != args.expected_official_receipt_sha256:
        raise ValueError("Cars transfer official receipt differs")
    official = json.loads(args.official_receipt.read_text())
    selected_digest = sha256(args.selected_checkpoint)
    training_digest = sha256(args.selected_training_receipt)
    training = json.loads(args.selected_training_receipt.read_text())
    final_digest = sha256(args.final_training_receipt)
    final = json.loads(args.final_training_receipt.read_text())
    validate_transfer_authority(official, training, selected_digest, training_digest)
    inputs = official["inputs"]
    final_inputs = final.get("inputs")
    if (
        final.get("schema") != "sfora-sop-compact-full-backbone-v1"
        or final.get("arm") != "arcface"
        or final.get("recipe") != "reference"
        or final.get("seed") != official["seed"]
        or final.get("updates") != 53760
        or inputs["training_receipt_sha256"].get("53760") != final_digest
        or sha256(args.source_checkpoint) != CHECKPOINT_SHA256
        or sha256(args.sop_features_archive) != ARCHIVE_SHA256
        or not isinstance(final_inputs, Mapping)
    ):
        raise ValueError("Cars transfer source authority differs")
    claim_path = args.sop_dataset_root / Path(inputs["official_test_claim_path"]).name
    claim = json.loads(claim_path.read_text())
    if (
        sha256(claim_path) != inputs.get("official_test_claim_sha256")
        or claim.get("schema") != "sfora-sop-official-test-claim-v1"
        or claim.get("final_training_receipt_sha256") != final_digest
        or claim.get("final_training_checkpoint_sha256")
        != final_inputs.get("checkpoint_output_sha256")
    ):
        raise ValueError("Cars transfer official claim differs")
    trained = torch.load(args.selected_checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(trained, Mapping) or trained.get("updates") != official["selection_step"]:
        raise ValueError("Cars transfer selected checkpoint differs")
    initial_head = reconstruct_initial_head(args, final)
    cars, rows, manifest, cache_sha256 = authenticate_cars_source(args.cars_baseline_archive)
    if source_manifest != cars_source_manifest():
        raise ValueError("Cars transfer evaluator source changed")
    if args.preflight_only:
        print(json.dumps({"preflight": "passed", "selected_sop_step": official["selection_step"]}))
        return
    if gpu_compute_pids() or not torch.cuda.is_available():
        raise ValueError("Cars transfer GPU is occupied")
    preflight_seconds = time.perf_counter() - started_all
    torch.backends.cuda.matmul.allow_tf32 = False
    inference_started = time.perf_counter()
    authenticated = load_authenticated_source_model(args.unicom_checkout, args.source_checkpoint)
    model = authenticated.encoder.cuda().eval()
    loader = DataLoader(
        VerifiedCarsImages(cars, rows, authenticated.transform, manifest),
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    torch.cuda.reset_peak_memory_stats()
    initial_head = initial_head.cuda().eval()
    baseline_values, baseline_encode_seconds = encode_cars(model, initial_head, loader)
    trained_head = nn.Linear(768, 128)
    trained_head.load_state_dict(trained["head"], strict=True)
    trained_head = trained_head.cuda().eval()
    pretrained_trained_head_values = None
    trained_initial_head_values = None
    cross_encode_seconds: dict[str, float] = {}
    if args.crossed_factorial:
        pretrained_trained_head_values, cross_encode_seconds["pretrained_trained_head"] = (
            encode_cars(model, trained_head, loader)
        )
    model.load_state_dict(trained["model"], strict=True)
    if args.crossed_factorial:
        trained_initial_head_values, cross_encode_seconds["trained_initial_head"] = encode_cars(
            model, initial_head, loader
        )
    trained_values, trained_encode_seconds = encode_cars(model, trained_head, loader)
    labels = torch.tensor([label for _, _, label in rows], dtype=torch.int64, device="cuda")
    results: dict[str, dict[str, Mapping[str, object]]] = {}
    feature_arrays = {"baseline": baseline_values.numpy(), "trained": trained_values.numpy()}
    arms = [("pretrained_initial_head", baseline_values), ("trained", trained_values)]
    if args.crossed_factorial:
        assert pretrained_trained_head_values is not None
        assert trained_initial_head_values is not None
        arms.extend(
            (
                ("pretrained_trained_head", pretrained_trained_head_values),
                ("trained_initial_head", trained_initial_head_values),
            )
        )
        feature_arrays["pretrained_trained_head"] = pretrained_trained_head_values.numpy()
        feature_arrays["trained_initial_head"] = trained_initial_head_values.numpy()
    for name, values in arms:
        packed = pack_int8_unit_embeddings(values)
        results[name] = {
            "float": score_symmetric(values.cuda(), labels),
            "packed": score_symmetric(
                packed.codes.float().cuda(), labels, inverse_norms=packed.inverse_norms.cuda()
            ),
        }
    if len(gpu_compute_pids()) > 1:
        raise ValueError("Cars transfer GPU became contended")
    if source_manifest != cars_source_manifest():
        raise ValueError("Cars transfer evaluator source changed")
    args.features_output.parent.mkdir(parents=True, exist_ok=True)

    def write_features(stream: BinaryIO) -> None:
        np.savez(
            stream,
            **feature_arrays,
            labels=np.asarray([label for _, _, label in rows], dtype=np.int64),
            source_rows=np.asarray(
                [(0 if split == "train" else 1, index) for split, index, _ in rows],
                dtype=np.int64,
            ),
            manifest_sha256=np.asarray(hashlib.sha256(manifest).hexdigest()),
        )

    receipt = {
        "schema": (
            "sfora-sop-cars-backbone-head-factorial-v2"
            if args.crossed_factorial
            else "sfora-sop-cars-transfer-v1"
        ),
        "claim_eligible": False,
        "protocol": (
            "Cars196 classes 98-195 across original train and test images; "
            "self retrieval; no Cars fitting"
        ),
        "selected_sop_step": official["selection_step"],
        "seed": official["seed"],
        "query_count": len(rows),
        "gallery_bytes_per_item": 130,
        "source_rows": rows,
        "results": results,
        "pretrained_encode_seconds": baseline_encode_seconds,
        "trained_encode_seconds": trained_encode_seconds,
        **({"cross_encode_seconds": cross_encode_seconds} if args.crossed_factorial else {}),
        "preflight_seconds": preflight_seconds,
        "inference_and_scoring_seconds": time.perf_counter() - inference_started,
        "total_seconds": time.perf_counter() - started_all,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_host_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "hardware": {
            "gpu": torch.cuda.get_device_name(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "datasets": version("datasets"),
            "pyarrow": version("pyarrow"),
            "pillow": version("pillow"),
        },
        "inputs": {
            "dataset_id": DATASET_ID,
            "dataset_revision": DATASET_REVISION,
            "source_fingerprints": SOURCE_FINGERPRINTS,
            "cache_file_sha256": cache_sha256,
            "cars_row_authority_archive_sha256": BASELINE_FEATURE_SHA256,
            "cars_evaluation_manifest_sha256": hashlib.sha256(manifest).hexdigest(),
            "official_receipt_sha256": official_digest,
            "selected_training_receipt_sha256": training_digest,
            "final_training_receipt_sha256": final_digest,
            "selected_checkpoint_sha256": selected_digest,
            "source_checkpoint_sha256": CHECKPOINT_SHA256,
            "sop_features_archive_sha256": ARCHIVE_SHA256,
            "initial_head_sha256": final_inputs["initial_head_sha256"],
            "fit_row_indexes_sha256": final["fit_row_indexes_sha256"],
            "features_sha256": "",
            "feature_array_sha256": {
                **{
                    name: hashlib.sha256(values.tobytes()).hexdigest()
                    for name, values in feature_arrays.items()
                },
                "labels": hashlib.sha256(
                    np.asarray([label for _, _, label in rows], dtype="<i8").tobytes()
                ).hexdigest(),
                "source_rows": hashlib.sha256(
                    np.asarray(
                        [(0 if split == "train" else 1, index) for split, index, _ in rows],
                        dtype="<i8",
                    ).tobytes()
                ).hexdigest(),
            },
            "source_sha256": source_manifest,
        },
        "argv": sys.argv,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    features_published = False
    try:
        publish_file_noreplace(args.features_output, write_features)
        features_published = True
        receipt_inputs = receipt["inputs"]
        if not isinstance(receipt_inputs, dict):
            raise ValueError("Cars transfer receipt inventory differs")
        receipt_inputs["features_sha256"] = sha256(args.features_output)
        payload = (json.dumps(receipt, sort_keys=True, allow_nan=False) + "\n").encode()

        def write_receipt(stream: BinaryIO) -> None:
            stream.write(payload)

        publish_file_noreplace(args.output, write_receipt)
    except BaseException:
        if features_published:
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
