#!/usr/bin/env python3
"""Paired image-to-top-k timing of train-selected compact B/16 and OML on SOP train rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import subprocess
import time
from collections.abc import Mapping
from contextlib import ExitStack
from pathlib import Path

import benchmark_sop_image_to_topk_pair as pair_module
import export_unicom_sop_embeddings as export_module
import numpy as np
import sop_teacher_anchored_runtime as teacher_module
import torch
from benchmark_sop_image_to_topk_pair import (
    NATIVE_API_SHA256,
    NATIVE_LIBRARY_SHA256,
    OML_CHECKPOINT_SHA256,
    OML_FEATURES_SHA256,
    TEST_IMAGE_MANIFEST_SHA256,
    _decode_encode,
    _load_oml,
    _measure,
    sha256,
    verify_live_query_features,
)
from export_unicom_sop_embeddings import _parse_split
from PIL import Image
from sop_teacher_anchored_runtime import load_authenticated_source_model
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

import sfora.cutile_int8 as cutile_int8_module
import sfora.joint_relational_compaction as compaction_module
import sfora.model_profiles as profiles_module
import sfora.sop_compact_training as compact_training_module
import sfora.sop_evaluation as sop_evaluation_module
from sfora.cutile_int8 import CutilePackedInt8Gallery
from sfora.joint_relational_compaction import pack_int8_unit_embeddings
from sfora.model_profiles import load_oml_sop_compact_encoder
from sfora.sop_compact_training import compact_head_features
from sfora.sop_evaluation import score_symmetric

SOURCE_CHECKPOINT_SHA256 = "c04f324f7c3b4435667236ec6c0eca1cd62f9d64fbfc2d06f8e8e60e6497edef"
EVALUATOR_SHA256 = "af66d5e38ef722688307ed6255e4db9def415baaae18401bb5acf1afaea9e28c"
PAIR_HELPER_SHA256 = "02df351fba2db9ffb701d892900a5335816a3edf33f466a359f087ed1a9f6b05"
OML_HEAD_SHA256 = "07e6e0f38dae4d509fe1e27b10aa650acda1f08d799faa025793cf7686a78cd4"
OML_QUALITY_SHA256 = "780805d2a090a6faa048cd3b5ba39c1a692fc494641bbfa987f509c9a9f7193c"
TRAIN_SOURCE_SHA256 = {
    "scripts/export_unicom_sop_embeddings.py": (
        "8967844e48dc45bb5f0eff3692caa6079d40301e5e0905ffd17bf6f896cfec26"
    ),
    "scripts/sop_teacher_anchored_runtime.py": (
        "1f60a8d5ad4a8779be3f078f3e748120d79ab77e990c309a898e744063582cb6"
    ),
    "src/sfora/sop_compact_training.py": (
        "888bb8a71a81b8cdbd098fa97e67546edb83481e19f8bbfa98c85130133f6152"
    ),
    "src/sfora/model_profiles.py": (
        "22b926079b9880ebda3c0331f4f9655c176fd69887d5882d245dcda87adc0cfd"
    ),
    "src/sfora/joint_relational_compaction.py": (
        "4ca0de1b0579ea6165c81e9057e9afe77e6dd4141f0b4a0adb281de25300de67"
    ),
    "src/sfora/sop_evaluation.py": (
        "eda764023c8a767d2fe46daa12d87dacdc0a38774e983cf445e6a4cb26293037"
    ),
}


def validate_selected_training(
    official: Mapping[str, object], trained: Mapping[str, object], checkpoint_sha256: str
) -> None:
    """Require the loaded checkpoint to be the official run's train-selected model."""

    inputs = official.get("inputs")
    if (
        official.get("schema") != "sfora-sop-reference-official-test-v1"
        or official.get("embedding_width") != 128
        or not isinstance(inputs, Mapping)
        or len(checkpoint_sha256) != 64
        or any(character not in "0123456789abcdef" for character in checkpoint_sha256)
        or inputs.get("selected_checkpoint_sha256") != checkpoint_sha256
        or trained.get("recipe") != "reference"
        or trained.get("arm") != "arcface"
        or trained.get("embedding_width") != 128
        or trained.get("seed") != official.get("seed")
        or trained.get("updates") != official.get("selection_step")
    ):
        raise ValueError("selected training authority differs")


def validate_official_quality(
    receipt_path: Path, official: Mapping[str, object], expected_sha256: str
) -> None:
    """Bind copied quality metrics to one completed official evaluation claim."""

    inputs = official.get("inputs")
    packed = official.get("packed")
    try:
        claim_path = (
            Path(inputs["official_test_claim_path"]) if isinstance(inputs, Mapping) else None
        )
        valid = (
            len(expected_sha256) == 64
            and all(character in "0123456789abcdef" for character in expected_sha256)
            and sha256(receipt_path) == expected_sha256
            and official.get("schema") == "sfora-sop-reference-official-test-v1"
            and official.get("claim_eligible") is False
            and official.get("test_images") == 60_502
            and official.get("gallery_bytes_per_item") == 130
            and isinstance(inputs, Mapping)
            and inputs.get("evaluator_sha256") == EVALUATOR_SHA256
            and inputs.get("test_image_manifest_sha256") == TEST_IMAGE_MANIFEST_SHA256
            and claim_path is not None
            and sha256(claim_path) == inputs.get("official_test_claim_sha256")
            and isinstance(packed, Mapping)
            and all(
                isinstance(packed.get(name), (int, float)) and 0 <= packed[name] <= 1
                for name in ("recall_at_1", "map_at_r")
            )
        )
    except (OSError, KeyError, TypeError, ValueError):
        valid = False
    if not valid:
        raise ValueError("SOP official quality authority differs")


def validate_train_ids(
    metadata_ids: list[int], archive_ids: list[int], *, expected_rows: int = 59_551
) -> None:
    """Require identical official training-image order for the two paired arms."""

    if (
        len(metadata_ids) != expected_rows
        or metadata_ids != archive_ids
        or len(set(metadata_ids)) != expected_rows
    ):
        raise ValueError("SOP train row identities differ")


def validate_train_holdout_parity(
    measured: Mapping[str, object], expected: Mapping[str, object]
) -> None:
    if any(
        not isinstance(measured.get(actual), (int, float))
        or not isinstance(expected.get(reference), (int, float))
        or abs(float(measured[actual]) - float(expected[reference])) > 0.002
        for actual, reference in (
            ("map_at_r", "packed_map_at_r"),
            ("recall_at_1", "packed_recall_at_1"),
        )
    ):
        raise ValueError("SOP selected train holdout parity differs")


def validate_output_absent(path: Path) -> None:
    if not path.is_absolute() or not path.parent.is_dir():
        raise ValueError("SOP trained timing output directory differs")
    if path.exists():
        raise ValueError("SOP trained timing output already exists")


def reserve_invocation(output: Path) -> Path:
    """Keep failed and concurrent benchmark launches from silently repeating."""

    marker = output.with_name(f".{output.name}.invoked")
    try:
        marker.mkdir()
    except FileExistsError:
        raise ValueError("SOP trained timing invocation already reserved") from None
    return marker


def parse_gpu_pids(stdout: str) -> set[int]:
    """Reject any ambiguous GPU compute-process inventory."""

    pids: set[int] = set()
    for line in stdout.splitlines():
        token = line.strip()
        if not token:
            continue
        if not token.isdecimal():
            raise ValueError("SOP GPU process inventory differs")
        pids.add(int(token))
    return pids


def gpu_compute_pids() -> set[int]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ValueError("SOP GPU process inventory differs") from error
    return parse_gpu_pids(result.stdout)


class TrainImages(Dataset[torch.Tensor]):
    def __init__(self, paths: tuple[Path, ...], transform) -> None:
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert("RGB"))


class TrainedEncoder(nn.Module):
    def __init__(self, backbone: nn.Module, head: nn.Linear) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        source = self.backbone(images)
        compact = compact_head_features(source, self.head, output_dim=128)
        return F.normalize(compact, dim=1)


class IdentityProjection:
    @staticmethod
    def transform(values: torch.Tensor) -> torch.Tensor:
        return values


def encode_train_gallery(
    model: nn.Module,
    paths: tuple[Path, ...],
    transform,
    *,
    expected_width: int,
    workers: int,
) -> tuple[torch.Tensor, float]:
    loader = DataLoader(
        TrainImages(paths, transform),
        batch_size=64,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )
    outputs = []
    started = time.perf_counter()
    with torch.inference_mode():
        for images in loader:
            outputs.append(model(images.cuda(non_blocking=True)).float().cpu())
    elapsed = time.perf_counter() - started
    values = torch.cat(outputs).contiguous()
    if values.shape != (59_551, expected_width) or not bool(torch.isfinite(values).all()):
        raise ValueError("SOP trained gallery encoding differs")
    return values, elapsed


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--oml-features", required=True, type=Path)
    parser.add_argument("--oml-checkpoint", required=True, type=Path)
    parser.add_argument("--oml-quality-receipt", required=True, type=Path)
    parser.add_argument("--unicom-checkout", required=True, type=Path)
    parser.add_argument("--source-checkpoint", required=True, type=Path)
    parser.add_argument("--selected-checkpoint", required=True, type=Path)
    parser.add_argument("--selected-training-receipt", required=True, type=Path)
    parser.add_argument("--official-receipt", required=True, type=Path)
    parser.add_argument("--expected-official-receipt-sha256", required=True)
    parser.add_argument("--native-library", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--calls", type=int, default=200)
    parser.add_argument("--execute-trained-paired-replay", action="store_true", required=True)
    args = parser.parse_args()
    if args.workers < 0 or args.calls < 50 or not torch.cuda.is_available():
        raise ValueError("SOP trained timing invocation differs")
    if not args.native_library.is_absolute():
        raise ValueError("SOP trained timing native library path differs")
    validate_output_absent(args.output)
    root = Path(__file__).resolve().parents[1]
    expected_files = {
        args.oml_features: OML_FEATURES_SHA256,
        args.oml_checkpoint: OML_CHECKPOINT_SHA256,
        args.oml_quality_receipt: OML_QUALITY_SHA256,
        args.source_checkpoint: SOURCE_CHECKPOINT_SHA256,
        args.native_library: NATIVE_LIBRARY_SHA256,
        Path(cutile_int8_module.__file__): NATIVE_API_SHA256,
        root / "scripts/benchmark_sop_image_to_topk_pair.py": PAIR_HELPER_SHA256,
        **{root / name: digest for name, digest in TRAIN_SOURCE_SHA256.items()},
    }
    for path, digest in expected_files.items():
        if sha256(path) != digest:
            raise ValueError(f"SOP trained timing input differs: {path}")
    loaded_sources = {
        "scripts/benchmark_sop_image_to_topk_pair.py": pair_module,
        "scripts/export_unicom_sop_embeddings.py": export_module,
        "scripts/sop_teacher_anchored_runtime.py": teacher_module,
        "src/sfora/sop_compact_training.py": compact_training_module,
        "src/sfora/model_profiles.py": profiles_module,
        "src/sfora/joint_relational_compaction.py": compaction_module,
        "src/sfora/sop_evaluation.py": sop_evaluation_module,
        "src/sfora/cutile_int8.py": cutile_int8_module,
    }
    for relative, module in loaded_sources.items():
        loaded = getattr(module, "__file__", None)
        if not isinstance(loaded, str) or Path(loaded).resolve() != (root / relative).resolve():
            raise ValueError(f"SOP trained timing imported source differs: {relative}")
    official_digest = sha256(args.official_receipt)
    official = json.loads(args.official_receipt.read_text())
    selected_training_digest = sha256(args.selected_training_receipt)
    selected_training = json.loads(args.selected_training_receipt.read_text())
    oml_quality = json.loads(args.oml_quality_receipt.read_text())
    if (
        oml_quality.get("schema") != "sfora-oml-sop-quality-profile-verification-v1"
        or oml_quality.get("projection_sha256") != OML_HEAD_SHA256
        or oml_quality.get("external_checkpoint_sha256") != OML_CHECKPOINT_SHA256
        or oml_quality.get("feature_sha256") != OML_FEATURES_SHA256
        or oml_quality.get("served_bytes_per_row") != 130
        or oml_quality.get("test_rows") != 60_502
    ):
        raise ValueError("SOP OML quality authority differs")
    selected_digest = sha256(args.selected_checkpoint)
    trained = torch.load(args.selected_checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(official, dict) or not isinstance(trained, dict):
        raise ValueError("selected training authority differs")
    validate_official_quality(
        args.official_receipt, official, args.expected_official_receipt_sha256
    )
    validate_selected_training(official, trained, selected_digest)
    inputs = official["inputs"]
    selected_step = str(official["selection_step"])
    if (
        not isinstance(selected_training, dict)
        or selected_training_digest != inputs["training_receipt_sha256"].get(selected_step)
        or selected_training.get("seed") != official["seed"]
        or selected_training.get("step", selected_training.get("updates"))
        != official["selection_step"]
    ):
        raise ValueError("SOP selected training receipt differs")
    if inputs["source_checkpoint_sha256"] != SOURCE_CHECKPOINT_SHA256 or inputs[
        "sop_train_metadata_sha256"
    ] != sha256(args.dataset_root / "Ebay_train.txt"):
        raise ValueError("SOP trained timing metadata differs")
    records = _parse_split(args.dataset_root, "train")
    paths = tuple(record.image_path for record in records)
    with np.load(args.oml_features, allow_pickle=False) as archive:
        oml_train = torch.from_numpy(np.ascontiguousarray(archive["train_features"])).float()
        oml_ids = archive["train_ids"].astype(np.int64, copy=False).tolist()
    validate_train_ids([record.image_id for record in records], oml_ids)
    if oml_train.shape != (59_551, 384):
        raise ValueError("SOP OML train feature inventory differs")
    gpu_before_load = gpu_compute_pids()
    if gpu_before_load - {os.getpid()}:
        raise ValueError("SOP trained timing GPU is occupied")
    invocation = reserve_invocation(args.output)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(16)
    torch.cuda.reset_peak_memory_stats()
    authenticated = load_authenticated_source_model(args.unicom_checkout, args.source_checkpoint)
    backbone = authenticated.encoder
    backbone.load_state_dict(trained["model"], strict=True)
    head = nn.Linear(768, 128)
    head.load_state_dict(trained["head"], strict=True)
    model = TrainedEncoder(backbone.cuda().eval(), head.cuda().eval()).eval()
    trained_features, trained_encode_seconds = encode_train_gallery(
        model, paths, authenticated.transform, expected_width=128, workers=args.workers
    )
    pack_started = time.perf_counter()
    trained_packed = pack_int8_unit_embeddings(trained_features)
    trained_pack_seconds = time.perf_counter() - pack_started
    validation_ids = selected_training["validation_image_ids"]
    validation_labels = selected_training["validation_labels"]
    id_to_row = {record.image_id: index for index, record in enumerate(records)}
    if (
        not isinstance(validation_ids, list)
        or not isinstance(validation_labels, list)
        or len(validation_ids) != 5_851
        or len(set(validation_ids)) != len(validation_ids)
        or any(image_id not in id_to_row for image_id in validation_ids)
    ):
        raise ValueError("SOP selected train holdout inventory differs")
    validation_rows = [id_to_row[image_id] for image_id in validation_ids]
    if validation_labels != [records[index].label for index in validation_rows]:
        raise ValueError("SOP selected train holdout labels differ")
    train_holdout_score = score_symmetric(
        trained_packed.codes[validation_rows].float().cuda(),
        torch.tensor(validation_labels, dtype=torch.int64, device="cuda"),
        inverse_norms=trained_packed.inverse_norms[validation_rows].cuda(),
    )
    validate_train_holdout_parity(train_holdout_score, official["selection_scores"][selected_step])
    oml_head = load_oml_sop_compact_encoder()
    if oml_head.sha256 != OML_HEAD_SHA256:
        raise ValueError("SOP OML compact projection differs")
    oml_model, oml_transform = _load_oml(args.oml_checkpoint)
    oml_features, oml_encode_seconds = encode_train_gallery(
        oml_model, paths, oml_transform, expected_width=384, workers=args.workers
    )
    if not bool((F.cosine_similarity(oml_features, oml_train, dim=1) >= 0.999).all()):
        raise ValueError("SOP OML cached train feature parity differs")
    oml_pack_started = time.perf_counter()
    oml_packed = pack_int8_unit_embeddings(oml_head.transform(oml_features))
    oml_pack_seconds = time.perf_counter() - oml_pack_started
    arms = {
        "oml": {"name": "oml", "model": oml_model, "transform": oml_transform, "head": oml_head},
        "trained_b16": {
            "name": "trained_b16",
            "model": model,
            "transform": authenticated.transform,
            "head": IdentityProjection(),
        },
    }
    query_paths = paths[:32]
    query_feature_parity = {
        "oml": verify_live_query_features(
            _decode_encode(arms["oml"], query_paths)[0], oml_train[:32], arm="oml"
        ),
        "trained_b16": verify_live_query_features(
            _decode_encode(arms["trained_b16"], query_paths)[0],
            trained_features[:32],
            arm="trained_b16",
        ),
    }
    query_image_hashes = [hashlib.sha256(path.read_bytes()).hexdigest() for path in query_paths]
    results = []
    gallery_peak_cuda_allocated_bytes = torch.cuda.max_memory_allocated()
    timing_started = time.perf_counter()
    with ExitStack() as stack:
        galleries = {
            "oml": stack.enter_context(
                CutilePackedInt8Gallery.open(
                    args.native_library,
                    oml_packed.codes.numpy()[32:].copy(),
                    oml_packed.inverse_norms.numpy()[32:].copy(),
                )
            ),
            "trained_b16": stack.enter_context(
                CutilePackedInt8Gallery.open(
                    args.native_library,
                    trained_packed.codes.numpy()[32:].copy(),
                    trained_packed.inverse_norms.numpy()[32:].copy(),
                )
            ),
        }
        gpu_before_timing = gpu_compute_pids()
        if gpu_before_timing != {os.getpid()}:
            raise ValueError("SOP trained timing GPU process overlap")
        torch.cuda.reset_peak_memory_stats()
        for pair, order in enumerate((("oml", "trained_b16"), ("trained_b16", "oml")), 1):
            for name in order:
                for batch in (1, 32):
                    results.append(
                        {
                            "pair": pair,
                            **_measure(
                                arms[name], query_paths[:batch], galleries[name], args.calls
                            ),
                        }
                    )
        gpu_after_timing = gpu_compute_pids()
        if gpu_after_timing != {os.getpid()}:
            raise ValueError("SOP trained timing GPU process overlap")
    timing_seconds = time.perf_counter() - timing_started
    receipt = {
        "schema": "sfora-sop-trained-image-to-topk-pair-v1",
        "claim_eligible": False,
        "split": "SOP official training rows; first 32 timed queries; gallery rows 32:59551",
        "gallery_rows": 59_519,
        "gallery_bytes_per_item": 130,
        "stage_timing_note": (
            "trained B/16 projection is inside encoder_transfer_ns; OML projection is inside "
            "project_pack_ns; compare image_to_topk_ns for the paired total"
        ),
        "official_quality_receipt_sha256": official_digest,
        "official_packed_recall_at_1": official["packed"]["recall_at_1"],
        "official_packed_map_at_r": official["packed"]["map_at_r"],
        "oml_official_packed_recall_at_1": oml_quality["recall_at_1"],
        "oml_official_packed_map_at_r": oml_quality["map_at_r"],
        "selected_checkpoint_sha256": selected_digest,
        "selected_training_receipt_sha256": selected_training_digest,
        "selection_step": official["selection_step"],
        "selected_train_holdout_parity": {
            "recall_at_1": train_holdout_score["recall_at_1"],
            "map_at_r": train_holdout_score["map_at_r"],
        },
        "trained_gallery_encode_seconds": trained_encode_seconds,
        "trained_gallery_pack_seconds": trained_pack_seconds,
        "oml_gallery_encode_seconds": oml_encode_seconds,
        "oml_gallery_pack_seconds": oml_pack_seconds,
        "timing_including_gallery_create_and_warmup_seconds": timing_seconds,
        "oml_head_sha256": OML_HEAD_SHA256,
        "query_image_sha256": query_image_hashes,
        "query_feature_parity": query_feature_parity,
        "gpu_compute_pids_before_load": sorted(gpu_before_load),
        "gpu_compute_pids_before_timing": sorted(gpu_before_timing),
        "gpu_compute_pids_after_timing": sorted(gpu_after_timing),
        "hardware": {
            "gpu": torch.cuda.get_device_name(),
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
        "inputs": {str(path): digest for path, digest in expected_files.items()},
        "script_sha256": sha256(Path(__file__)),
        "invocation_marker": str(invocation),
        "gallery_peak_cuda_allocated_bytes": gallery_peak_cuda_allocated_bytes,
        "timing_peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_host_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "rows": results,
    }
    if (
        sha256(args.official_receipt) != official_digest
        or sha256(args.selected_checkpoint) != selected_digest
        or sha256(args.selected_training_receipt) != selected_training_digest
        or sha256(args.oml_quality_receipt) != OML_QUALITY_SHA256
        or any(sha256(path) != digest for path, digest in expected_files.items())
    ):
        raise ValueError("SOP trained timing input changed during replay")
    payload = (
        json.dumps(receipt, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()
    with args.output.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"receipt": str(args.output), "selection_step": official["selection_step"]}))


if __name__ == "__main__":
    main()
