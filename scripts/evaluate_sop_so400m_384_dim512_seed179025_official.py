#!/usr/bin/env python3
"""Official SOP TEST read of the 512-D head So400m@384 SEED 179025 replicate (receipt 7b14f72e), for the
pre-registered two-seed ensemble (3428f85e). Derived from the seed-179024 evaluator: (receipt 1aaf30ed; TRAIN held 94.924, +0.41 vs
280/512-D, lower95 +0.08). Pre-registered: ONE read; new SOP best iff official packed R@1 > 93.480 (280/512-D).
Native 128-only kernel skipped; packed int8 512-D (514 B/row) scored exactly in torch (score_symmetric).

Derived from: Official SOP TEST read for one arm of the So400m vs Large substrate pair (seed 179024, 1000 updates).

Pre-registered (committed before any official number for these checkpoints exists):
  authority: TRAIN decision docs/evidence/.../sop-so400m-pair-179024-v3/RESULT.md (ADVANCE, dR@1 lower95 +0.71).
             Only the two pinned pair receipts below may be read; one read per arm, no reruns on outcome.
  protocol : identical to evaluate_sop_siglip2_official.py (fp32_autocast fp16, batch 64): pinned TEST
             manifest, symmetric 60,502 query/gallery self excluded, 128-D int8 packing, native top-10 exact.
             Substrate-aware: pooled width, eval crop and frozen count from the receipt's substrate_config,
             cross-checked against the trainer's ARMS table.
  rule     : So400m is the SOP quality Pareto candidate iff its official packed R@1 > Large pair control
             AND > the prior SOP official best 91.7419 (exploratory incumbent). mAP@R and cost reported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import sys
import time
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_sop_so400m_384_dim512_pilot  # noqa: E402,F401  (OUTPUT_WIDTH 512, 512-D head features)
import train_sop_siglip2_substrate_pair as pair  # noqa: E402

from sfora.cutile_int8 import CutilePackedInt8Gallery  # noqa: E402
from sfora.joint_relational_compaction import pack_int8_unit_embeddings  # noqa: E402
from sfora.sop_evaluation import score_symmetric  # noqa: E402

ARCHIVE_SHA256 = "1ba27b2d6b9db39067aa6facd0ef8aafc303c4527f6feabed859b0512c7d921a"
TEST_IMAGE_MANIFEST_SHA256 = "28a3ec0561cd83ee426f3d1c301c70799316af91c1e9083a5a1ffdf3414327c1"
PAIR_RECEIPTS = {
    "large": "d7f7383f9c8320e9d4ca5143880a637ed9bf4bc63ef61a10767c04e8305c8494",
    "so400": "7b14f72e49f604744e3294c3c4ed530ca8eca5404a611db4022c9cb96ca8aad2",
}
PRIOR_OFFICIAL_R1 = 0.917419


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class VerifiedRows(Dataset):  # type: ignore[misc]
    def __init__(self, paths, labels, manifest: bytes) -> None:
        if not paths or len(paths) != len(labels) or len(manifest) != 32 * len(paths):
            raise ValueError("SOP official image manifest geometry differs")
        self.paths, self.labels, self.manifest = paths, labels, manifest

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        raw = self.paths[index].read_bytes()
        if hashlib.sha256(raw).digest() != self.manifest[32 * index : 32 * (index + 1)]:
            raise ValueError("SOP official image bytes differ from pinned manifest")
        with Image.open(BytesIO(raw)) as image:
            return image.convert("RGB"), self.labels[index]


@torch.inference_mode()
def export(dataset, processor, vision, head, width: int, workers: int) -> torch.Tensor:
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=workers,
                        pin_memory=True, collate_fn=pair.make_collate(processor))
    out = []
    for index, (batch, _labels) in enumerate(loader, start=1):
        tensors = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
        with torch.amp.autocast("cuda", dtype=torch.float16):
            pooled = vision(**tensors).pooler_output
        if pooled is None or pooled.shape != (len(batch["pixel_values"]), width):
            raise ValueError("SOP official SigLIP2 pooler differs")
        out.append(F.normalize(pair.compact_head_features(pooled, head), dim=1).cpu())
        if index % 100 == 0:
            print(json.dumps({"export_batches": index}), flush=True)
    result = torch.cat(out).contiguous()
    if result.shape != (len(dataset), 512) or not bool(torch.isfinite(result).all()):
        raise ValueError("SOP official descriptor geometry differs")
    return result


@torch.inference_mode()
def verify_native(packed, labels: np.ndarray, native_library: Path) -> tuple[list[float], float]:
    code, inverse = packed.codes.float().cuda(), packed.inverse_norms.float().cuda()
    r1, delta = [], 0.0
    with CutilePackedInt8Gallery.open_packed(native_library, packed) as gallery:
        for start in range(0, len(packed.codes), 32):
            stop = min(start + 32, len(packed.codes))
            query = type(packed)(packed.codes[start:stop].contiguous(), packed.inverse_norms[start:stop].contiguous())
            ordinals, scores = gallery.search_packed(query)
            oracle = (code[start:stop] @ code.T) * inverse[start:stop, None] * inverse[None, :]
            expected = torch.argsort(oracle, dim=1, descending=True, stable=True)[:, :10]
            if ordinals.shape != (stop - start, 10) or not np.array_equal(ordinals, expected.cpu().numpy()):
                raise ValueError("SOP official native top-10 ordinals differ")
            delta = max(delta, float(np.max(np.abs(scores - oracle.gather(1, expected).cpu().numpy()))))
            if delta > 1e-5:
                raise ValueError("SOP official native top-10 scores differ")
            for row, ranked in zip(range(start, stop), ordinals, strict=True):
                top1 = next(int(o) for o in ranked if int(o) != row)
                r1.append(float(labels[top1] == labels[row]))
    return r1, delta


def main() -> None:
    started = time.perf_counter()
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--substrate", choices=("so400",), required=True)
    for name in ("training-receipt", "training-checkpoint", "training-source-root", "source-archive",
                 "test-image-manifest", "dataset-root", "model-snapshot", "native-library", "output-dir"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--workers", type=int, default=2)
    a = p.parse_args()
    arm = pair.ARMS[a.substrate]
    receipt = json.loads(a.training_receipt.read_text())
    cfg = receipt.get("substrate_config", {})
    if (
        a.output_dir.exists() or not torch.cuda.is_available()
        or sha256(a.training_receipt) != PAIR_RECEIPTS[a.substrate]
        or receipt.get("schema") != "sfora-sop-siglip2-substrate-pair-v1"
        or receipt.get("substrate") != a.substrate
        or {k: cfg.get(k) for k in ("width", "crop", "frozen_blocks", "blocks", "revision")}
        != {k: arm[k] for k in ("width", "crop", "frozen_blocks", "blocks", "revision")}
        or receipt.get("seed") != 179025 or receipt.get("updates") != 1_000
        or receipt.get("quality", {}).get("output_width") != 512
        or receipt.get("source_archive_sha256") != ARCHIVE_SHA256
        or sha256(a.source_archive) != ARCHIVE_SHA256
        or sha256(a.test_image_manifest) != TEST_IMAGE_MANIFEST_SHA256
        or sha256(a.training_checkpoint) != receipt.get("checkpoint_sha256")
        or sha256(a.native_library) != receipt.get("native_library_sha256")
        or any(sha256(a.training_source_root / r) != d for r, d in receipt["source_files_sha256"].items())
        or any(sha256(a.model_snapshot / n) != d for n, d in arm["model_hashes"].items())
        or receipt.get("model_file_sha256") != arm["model_hashes"]
        or sha256(Path(__file__).resolve().parent / "train_sop_siglip2_substrate_pair.py")
        != receipt["source_files_sha256"]["scripts/train_sop_siglip2_substrate_pair.py"]
    ):
        raise ValueError("SOP pair official authority, checkpoint or dataset differs")
    tileiras = os.environ.get("CUTILE_TILEIRAS_PATH")
    if not tileiras or sha256(Path(tileiras)) != receipt.get("tileiras_sha256"):
        raise ValueError("SOP pair official runtime differs from training")
    manifest = a.test_image_manifest.read_bytes()
    with np.load(a.source_archive, allow_pickle=False) as archive:
        train_labels = np.asarray(archive["train_labels"], dtype=np.int64)
        labels = np.asarray(archive["test_labels"], dtype=np.int64)
        relatives = np.asarray(archive["test_relative_paths"]).astype(str)
    if (labels.shape != (60_502,) or len(np.unique(labels)) != 11_316
            or set(map(int, train_labels)) & set(map(int, labels)) or len(manifest) != 32 * len(labels)):
        raise ValueError("SOP official TEST inventory differs")
    dataset = VerifiedRows(pair.paths_from_archive(a.dataset_root, relatives), tuple(map(int, labels)), manifest)
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(a.model_snapshot, local_files_only=True, backend="torchvision")
    if processor.size.get("height") != arm["crop"] or processor.size.get("width") != arm["crop"] or processor.resample != 2:
        raise ValueError("SOP official image processor differs")
    full = AutoModel.from_pretrained(a.model_snapshot, local_files_only=True, use_safetensors=True, dtype=torch.float16)
    vision = full.vision_model.float().cuda().eval()
    del full
    head = nn.Linear(arm["width"], 512).cuda().eval()
    ckpt = torch.load(a.training_checkpoint, map_location="cpu", weights_only=True)
    if ckpt.get("seed") != 179025 or ckpt.get("updates") != 1_000 or ckpt.get("freeze_lower_stack") is not True:
        raise ValueError("SOP official training checkpoint identity differs")
    vision.load_state_dict(ckpt["vision"], strict=True)
    head.load_state_dict(ckpt["head"], strict=True)
    a.output_dir.mkdir(parents=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    t = time.perf_counter()
    values = export(dataset, processor, vision, head, arm["width"], a.workers)
    export_seconds = time.perf_counter() - t
    np.save(a.output_dir / "test_embeddings.npy", values.numpy(), allow_pickle=False)
    label_tensor = torch.from_numpy(labels.copy()).cuda()
    float_quality = score_symmetric(values.cuda(), label_tensor)
    packed = pack_int8_unit_embeddings(values)
    packed_quality = score_symmetric(packed.codes.float().cuda(), label_tensor, inverse_norms=packed.inverse_norms.cuda())
    delta = None  # native kernel is 128-D only; packed 512-D scored exactly in torch
    assert packed.codes.shape[1] == 512
    result = {
        "schema": "sfora-sop-so400m-384-dim512-official-test-v1", "output_width": 512, "gallery_wire_bytes_per_row": 514, "claim_eligible": False,
        "substrate": a.substrate, "substrate_config": cfg, "seed": 179025,
        "split": "SOP official TEST, symmetric query/gallery, self excluded", "queries": len(labels),
        "source_sha256": sha256(Path(__file__)), "training_receipt_sha256": PAIR_RECEIPTS[a.substrate],
        "training_checkpoint_sha256": receipt["checkpoint_sha256"],
        "test_image_manifest_sha256": TEST_IMAGE_MANIFEST_SHA256,
        "test_embeddings_sha256": sha256(a.output_dir / "test_embeddings.npy"),
        "float_quality": float_quality, "packed_quality": packed_quality,
        "native_top10_exact": None, "native_skipped_reason": "CutilePackedInt8Gallery supports only 128-D",
        "prior_official_r1": PRIOR_OFFICIAL_R1, "export_seconds": export_seconds,
        "export_images_per_second": len(labels) / export_seconds,
        "total_wall_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_parent_host_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    }
    with (a.output_dir / "receipt.json").open("x") as stream:
        stream.write(json.dumps(result, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"substrate": a.substrate, "packed_r1": packed_quality["recall_at_1"],
                      "packed_map": packed_quality.get("map_at_r")}), flush=True)


if __name__ == "__main__":
    main()
