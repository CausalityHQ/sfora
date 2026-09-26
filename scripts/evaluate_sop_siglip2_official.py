#!/usr/bin/env python3
"""Evaluate one frozen SigLIP2 checkpoint on authenticated SOP official TEST."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import time
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from train_sop_siglip2_compact import make_collate, paths_from_archive

from sfora.cutile_int8 import CutilePackedInt8Gallery
from sfora.joint_relational_compaction import PackedInt8Embeddings, pack_int8_unit_embeddings
from sfora.sop_compact_training import compact_head_features
from sfora.sop_evaluation import score_symmetric

ARCHIVE_SHA256 = "1ba27b2d6b9db39067aa6facd0ef8aafc303c4527f6feabed859b0512c7d921a"
TEST_IMAGE_MANIFEST_SHA256 = "28a3ec0561cd83ee426f3d1c301c70799316af91c1e9083a5a1ffdf3414327c1"
ARMS = {"arcface": "arcface", "float_rank": "float_rank", "bank": "float_rank_member_bank"}
REPLICATION_SEEDS = (179020, 179021, 179022)
BF16_SEEDS = (179023, 179024, 179025)
TRUE_FREEZE_SEEDS = (179024, 179026, 179027)
TRUE_FREEZE_TRAINER_SHA256 = "c5b8786c352ce6c8bedce9a5963ef43e2c18db61974e3c141c698227a23f1b3c"
BF16_ARMS = {
    "bank": (
        "float_rank_member_bank",
        8.0,
        "ad66b1613f0c1c8373d69a689d1556c9b250527a8045a6b85bec523f99466d23",
    ),
    "matched_float": (
        "float_rank",
        21.93,
        "328cdfbd4d35ae8037d1130c5fc889d60fe0ec25cf185c0c9ecc714a475120e4",
    ),
    "original_float": (
        "float_rank",
        8.0,
        "ad66b1613f0c1c8373d69a689d1556c9b250527a8045a6b85bec523f99466d23",
    ),
}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_decision_arm(
    decision: dict, seed: int, arm: str, receipt_sha256: str, receipt: dict
) -> str:
    schema = decision.get("schema")
    if schema == "sfora-sop-siglip2-member-bank-multiseed-v1":
        expected_arm = ARMS.get(arm)
        valid = (
            decision.get("continuation_gate_pass") is True
            and tuple(decision.get("replication_seeds", ())) == REPLICATION_SEEDS
            and seed in REPLICATION_SEEDS
            and decision.get("arms", {}).get(str(seed), {}).get(arm, {}).get("receipt_sha256")
            == receipt_sha256
        )
    elif schema == "sfora-sop-siglip2-bf16-rankmatched-float-v1":
        expected = BF16_ARMS.get(arm)
        expected_arm = expected[0] if expected else None
        valid = (
            expected is not None
            and decision.get("claim_eligible") is False
            and decision.get("bank_specific_screen_pass") is True
            and tuple(decision.get("seeds", ())) == BF16_SEEDS
            and seed in BF16_SEEDS
            and decision.get("arms", {}).get(str(seed), {}).get(f"{arm}_receipt_sha256")
            == receipt_sha256
            and receipt.get("rank_coefficient") == expected[1]
            and receipt.get("source_sha256") == expected[2]
            and receipt.get("train_vision_dtype") == "bf16"
            and receipt.get("updates") == 1_000
        )
    elif schema == "sfora-sop-siglip2-bf16-coverage-replication-v1":
        expected_arm = "float_rank_member_bank" if arm == "bank" else None
        valid = (
            expected_arm is not None
            and decision.get("claim_eligible") is False
            and decision.get("replication_gate_pass") is True
            and tuple(decision.get("seeds", ())) == BF16_SEEDS
            and seed in BF16_SEEDS
            and decision.get("arms", {}).get(str(seed), {}).get("bank", {}).get("receipt_sha256")
            == receipt_sha256
            and receipt.get("rank_coefficient") == 8.0
            and receipt.get("source_sha256")
            == "400f6ef2d992e449ff7eabf53c2982b88db0889df0586bbf94875a1b04a6e118"
            and receipt.get("train_vision_dtype") == "bf16"
            and receipt.get("updates") == 1_000
        )
    elif schema == "sfora-sop-true-freeze-paired-train-only-v1":
        expected_arm = "float_rank_member_bank" if arm in ("control", "freeze") else None
        frozen = arm == "freeze"
        valid = (
            expected_arm is not None
            and decision.get("claim_eligible") is False
            and decision.get("advance_fresh_seeds") is True
            and decision.get("quality_pass") is True
            and decision.get("cost_pass") is True
            and seed in TRUE_FREEZE_SEEDS
            and decision.get("seed") == seed
            and decision.get("arms", {}).get(arm, {}).get("receipt_sha256") == receipt_sha256
            and receipt.get("source_sha256") == TRUE_FREEZE_TRAINER_SHA256
            and receipt.get("rank_coefficient") == 8.0
            and receipt.get("train_vision_dtype") == "bf16"
            and receipt.get("updates") == 1_000
            and receipt.get("freeze_lower_stack") is frozen
            and receipt.get("frozen_encoder_blocks") == (list(range(12)) if frozen else [])
        )
    else:
        expected_arm = None
        valid = False
    if (
        not valid
        or expected_arm is None
        or receipt.get("seed") != seed
        or receipt.get("arm") != expected_arm
        or receipt.get("quality", {}).get("native_top10_exact") is not True
    ):
        raise ValueError("SOP official evaluation gate or arm authority differs")
    return expected_arm


class VerifiedRows(Dataset):  # type: ignore[misc]
    """Decode the exact image bytes in an ordered, previously pinned manifest."""

    def __init__(self, paths: tuple[Path, ...], labels: tuple[int, ...], manifest: bytes) -> None:
        if not paths or len(paths) != len(labels) or len(manifest) != 32 * len(paths):
            raise ValueError("SOP official image manifest geometry differs")
        self.paths = paths
        self.labels = labels
        self.manifest = manifest

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[Image.Image, int]:
        raw = self.paths[index].read_bytes()
        if hashlib.sha256(raw).digest() != self.manifest[32 * index : 32 * (index + 1)]:
            raise ValueError("SOP official image bytes differ from pinned manifest")
        with Image.open(BytesIO(raw)) as image:
            return image.convert("RGB"), self.labels[index]


@torch.inference_mode()  # type: ignore[untyped-decorator]
def export_verified(
    dataset: VerifiedRows,
    processor: Any,
    vision: nn.Module,
    head: nn.Linear,
    *,
    workers: int,
    batch_size: int = 64,
) -> torch.Tensor:
    if batch_size not in (1, 32, 64):
        raise ValueError("SOP official export batch size differs")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        collate_fn=make_collate(processor),
    )
    vision.eval()
    head.eval()
    output = []
    for index, (batch, _labels) in enumerate(loader, start=1):
        tensors = {key: value.cuda(non_blocking=True) for key, value in batch.items()}
        with torch.amp.autocast("cuda", dtype=torch.float16):
            pooled = vision(**tensors).pooler_output
        if pooled is None or pooled.shape != (len(batch["pixel_values"]), 1024):
            raise ValueError("SOP official SigLIP2 pooler differs")
        values = F.normalize(compact_head_features(pooled, head), dim=1).cpu()
        output.append(values)
        if index % 100 == 0:
            print(json.dumps({"export_batches": index}), flush=True)
    result = torch.cat(output).contiguous()
    if result.shape != (len(dataset), 128) or not bool(torch.isfinite(result).all()):
        raise ValueError("SOP official descriptor geometry differs")
    return result


@torch.inference_mode()  # type: ignore[untyped-decorator]
def verify_native(
    packed: PackedInt8Embeddings, labels: np.ndarray, native_library: Path
) -> dict[str, Any]:
    code = packed.codes.float().cuda()
    inverse = packed.inverse_norms.float().cuda()
    native_r1 = []
    max_score_delta = 0.0
    with CutilePackedInt8Gallery.open_packed(native_library, packed) as gallery:
        for start in range(0, len(packed.codes), 32):
            stop = min(start + 32, len(packed.codes))
            block = np.arange(start, stop, dtype=np.int64)
            query = type(packed)(
                packed.codes[start:stop].contiguous(),
                packed.inverse_norms[start:stop].contiguous(),
            )
            ordinals, native_scores = gallery.search_packed(query)
            if (
                ordinals.shape != (len(block), 10)
                or native_scores.shape != ordinals.shape
                or not np.isfinite(native_scores).all()
            ):
                raise ValueError("SOP official native top-10 geometry differs")
            oracle = (code[start:stop] @ code.T) * inverse[start:stop, None] * inverse[None, :]
            expected = torch.argsort(oracle, dim=1, descending=True, stable=True)[:, :10]
            if not np.array_equal(ordinals, expected.cpu().numpy()):
                raise ValueError("SOP official native top-10 ordinals differ")
            selected = oracle.gather(1, expected).cpu().numpy()
            max_score_delta = max(max_score_delta, float(np.max(np.abs(native_scores - selected))))
            if max_score_delta > 1e-5:
                raise ValueError("SOP official native top-10 scores differ")
            for row, ranked in zip(block, ordinals, strict=True):
                top1 = next((int(ordinal) for ordinal in ranked if int(ordinal) != int(row)), None)
                if top1 is None:
                    raise ValueError("SOP official native nonself top-1 missing")
                native_r1.append(float(labels[top1] == labels[row]))
    return {
        "native_top10_exact": True,
        "native_top10_max_score_abs_delta": max_score_delta,
        "native_per_query_r1": native_r1,
        "gallery_wire_bytes_per_row": 130,
    }


def main() -> None:
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--expected-decision-sha256", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--arm",
        choices=tuple(ARMS) + ("matched_float", "original_float", "control", "freeze"),
        required=True,
    )
    parser.add_argument("--training-receipt", type=Path, required=True)
    parser.add_argument("--training-checkpoint", type=Path, required=True)
    parser.add_argument("--training-source-root", type=Path)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--test-image-manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--native-library", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if (
        args.output_dir.exists()
        or args.output_dir.is_symlink()
        or args.workers < 0
        or not torch.cuda.is_available()
        or sha256(args.decision) != args.expected_decision_sha256
    ):
        raise ValueError("SOP official evaluation invocation differs")
    decision = json.loads(args.decision.read_text())
    receipt_sha = sha256(args.training_receipt)
    receipt = json.loads(args.training_receipt.read_text())
    expected_arm = validate_decision_arm(decision, args.seed, args.arm, receipt_sha, receipt)
    root = Path(__file__).resolve().parents[1]
    if (
        decision["schema"]
        in (
            "sfora-sop-siglip2-bf16-rankmatched-float-v1",
            "sfora-sop-siglip2-bf16-coverage-replication-v1",
            "sfora-sop-true-freeze-paired-train-only-v1",
        )
        and args.training_source_root is None
    ):
        raise ValueError("SOP BF16 training source root required")
    training_root = args.training_source_root or root
    evaluator_files = (
        "scripts/evaluate_sop_siglip2_official.py",
        "scripts/train_sop_siglip2_compact.py",
        "src/sfora/cutile_int8.py",
        "src/sfora/joint_relational_compaction.py",
        "src/sfora/sop_compact_training.py",
        "src/sfora/sop_evaluation.py",
    )
    evaluator_manifest = {relative: sha256(root / relative) for relative in evaluator_files}
    if (
        receipt.get("schema") != "sfora-sop-siglip2-compact-full-backbone-v1"
        or receipt.get("source_archive_sha256") != ARCHIVE_SHA256
        or sha256(args.source_archive) != ARCHIVE_SHA256
        or sha256(args.test_image_manifest) != TEST_IMAGE_MANIFEST_SHA256
        or sha256(args.training_checkpoint) != receipt.get("checkpoint_sha256")
        or sha256(args.native_library) != receipt.get("native_library_sha256")
        or any(
            sha256(training_root / relative) != digest
            for relative, digest in receipt["source_files_sha256"].items()
        )
        or any(
            sha256(args.model_snapshot / name) != digest
            for name, digest in receipt["model_file_sha256"].items()
        )
    ):
        raise ValueError("SOP official checkpoint or dataset source differs")
    import PIL
    import torchvision
    import transformers

    stack = receipt.get("hardware", {})
    tileiras = os.environ.get("CUTILE_TILEIRAS_PATH")
    if (
        stack.get("torch") != torch.__version__
        or stack.get("torchvision") != torchvision.__version__
        or stack.get("pillow") != PIL.__version__
        or stack.get("transformers") != transformers.__version__
        or not tileiras
        or sha256(Path(tileiras)) != receipt.get("tileiras_sha256")
    ):
        raise ValueError("SOP official runtime differs from training")
    manifest = args.test_image_manifest.read_bytes()
    with np.load(args.source_archive, allow_pickle=False) as archive:
        train_labels = np.asarray(archive["train_labels"], dtype=np.int64)
        labels = np.asarray(archive["test_labels"], dtype=np.int64)
        ids = np.asarray(archive["test_image_ids"], dtype=np.int64)
        relatives = np.asarray(archive["test_relative_paths"]).astype(str)
    if (
        train_labels.shape != (59_551,)
        or labels.shape != (60_502,)
        or ids.shape != labels.shape
        or relatives.shape != labels.shape
        or len(np.unique(labels)) != 11_316
        or set(map(int, train_labels)) & set(map(int, labels))
        or len(manifest) != 32 * len(labels)
    ):
        raise ValueError("SOP official TEST inventory differs")
    paths = paths_from_archive(args.dataset_root, relatives)
    dataset = VerifiedRows(paths, tuple(map(int, labels)), manifest)
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(
        args.model_snapshot, local_files_only=True, backend="torchvision"
    )
    if (
        type(processor).__name__ != "SiglipImageProcessor"
        or processor.size.get("height") != 256
        or processor.size.get("width") != 256
        or processor.resample != 2
    ):
        raise ValueError("SOP official image processor differs")
    full_model = AutoModel.from_pretrained(
        args.model_snapshot, local_files_only=True, use_safetensors=True, dtype=torch.float16
    )
    vision = full_model.vision_model.float().cuda().eval()
    del full_model
    head = nn.Linear(1024, 128).cuda().eval()
    checkpoint = torch.load(args.training_checkpoint, map_location="cpu", weights_only=True)
    if (
        checkpoint.get("seed") != args.seed
        or checkpoint.get("arm") != expected_arm
        or checkpoint.get("updates") != 1_000
    ):
        raise ValueError("SOP official training checkpoint identity differs")
    vision.load_state_dict(checkpoint["vision"], strict=True)
    head.load_state_dict(checkpoint["head"], strict=True)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    export_started = time.perf_counter()
    values = export_verified(dataset, processor, vision, head, workers=args.workers)
    export_seconds = time.perf_counter() - export_started
    embeddings = args.output_dir / "test_embeddings.npy"
    np.save(embeddings, values.numpy(), allow_pickle=False)
    score_started = time.perf_counter()
    label_tensor = torch.from_numpy(labels.copy()).cuda()
    float_quality = score_symmetric(values.cuda(), label_tensor)
    float_score_seconds = time.perf_counter() - score_started
    packing_started = time.perf_counter()
    packed = pack_int8_unit_embeddings(values)
    packing_seconds = time.perf_counter() - packing_started
    packed_score_started = time.perf_counter()
    packed_quality = score_symmetric(
        packed.codes.float().cuda(),
        label_tensor,
        inverse_norms=packed.inverse_norms.cuda(),
    )
    packed_score_seconds = time.perf_counter() - packed_score_started
    native_started = time.perf_counter()
    native = verify_native(packed, labels, args.native_library)
    native_verify_seconds = time.perf_counter() - native_started
    if native["native_per_query_r1"] != packed_quality["per_query_r1"]:
        raise ValueError("SOP official native recall differs from packed oracle")
    if {relative: sha256(root / relative) for relative in evaluator_files} != evaluator_manifest:
        raise ValueError("SOP official evaluator source changed during execution")
    score_seconds = time.perf_counter() - score_started
    result = {
        "schema": "sfora-sop-siglip2-official-test-v1",
        "claim_eligible": False,
        "status": "exploratory official TEST; prior Sfora work already read this protocol",
        "seed": args.seed,
        "arm": args.arm,
        "split": "SOP official TEST, symmetric query/gallery, self excluded",
        "queries": len(labels),
        "products": len(np.unique(labels)),
        "source_sha256": sha256(Path(__file__)),
        "evaluator_source_files_sha256": evaluator_manifest,
        "decision_sha256": sha256(args.decision),
        "training_receipt_sha256": receipt_sha,
        "training_source_root": str(training_root.resolve()),
        "training_checkpoint_sha256": sha256(args.training_checkpoint),
        "source_archive_sha256": ARCHIVE_SHA256,
        "test_image_manifest_sha256": TEST_IMAGE_MANIFEST_SHA256,
        "test_image_ids_sha256": hashlib.sha256(ids.tobytes()).hexdigest(),
        "test_embeddings_sha256": sha256(embeddings),
        "native_library_sha256": sha256(args.native_library),
        "gallery_wire_bytes_per_row": native["gallery_wire_bytes_per_row"],
        "float_quality": float_quality,
        "packed_quality": packed_quality,
        "native_top10_exact": native["native_top10_exact"],
        "native_top10_max_score_abs_delta": native["native_top10_max_score_abs_delta"],
        "native_per_query_r1_equal": True,
        "export_seconds": export_seconds,
        "float_score_seconds": float_score_seconds,
        "packing_seconds": packing_seconds,
        "packed_score_seconds": packed_score_seconds,
        "native_verify_seconds": native_verify_seconds,
        "score_seconds": score_seconds,
        "total_wall_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_parent_host_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "hardware": {
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "python": platform.python_version(),
        },
    }
    with (args.output_dir / "receipt.json").open("xb") as stream:
        stream.write((json.dumps(result, sort_keys=True, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    print(
        json.dumps(
            {"seed": args.seed, "arm": args.arm, "packed_r1": packed_quality["recall_at_1"]}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
