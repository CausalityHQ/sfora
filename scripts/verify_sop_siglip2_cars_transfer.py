#!/usr/bin/env python3
"""Fixed SOP control/freeze checkpoint transfer to Cars196's class-disjoint split."""

from __future__ import annotations

import gc
import hashlib
import json
import time
from io import BytesIO
from pathlib import Path

import torch
from datasets import Image as HFImage
from datasets import load_dataset
from PIL import Image
from verify_sop_siglip2_cub_transfer import CHECKPOINTS, MODEL, sha

from sfora.siglip2_compact_serving import Siglip2CompactEncoder
from sfora.sop_evaluation import score_symmetric

RUN = Path("/home/riomus/runs/sfora-sop-cars-transfer-public-v1")
DATASET = "tanganke/stanford_cars"
REVISION = "9abf6cf7d6dfa7b95152a0d6e791ea9435b47a40"
FINGERPRINTS = {"train": "c027c63962212a03", "test": "59e3e308b987bbb7"}
CACHE_SHA = {
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


def main() -> None:
    if not torch.cuda.is_available():
        raise ValueError("Cars transfer requires CUDA")
    source = load_dataset(DATASET, revision=REVISION)
    if (
        len(source["train"]) != 8144
        or len(source["test"]) != 8041
        or {split: source[split]._fingerprint for split in ("train", "test")} != FINGERPRINTS
    ):
        raise ValueError("Cars transfer source inventory differs")
    cache = {
        Path(row["filename"]).name: Path(row["filename"])
        for split in ("train", "test")
        for row in source[split].cache_files
    }
    if set(cache) != set(CACHE_SHA) or any(
        REVISION not in path.parts or sha(path) != CACHE_SHA[name] for name, path in cache.items()
    ):
        raise ValueError("Cars transfer source cache differs")
    rows = [
        (split, ordinal, int(label))
        for split in ("train", "test")
        for ordinal, label in enumerate(source[split]["label"])
        if int(label) >= 98
    ]
    if len(rows) != 8131 or len({label for _, _, label in rows}) != 98:
        raise ValueError("Cars transfer class-disjoint split differs")
    for split in ("train", "test"):
        source[split] = source[split].cast_column("image", HFImage(decode=False))
    digests = []
    for split, ordinal, label in rows:
        row = source[split][ordinal]
        payload = row["image"]["bytes"]
        if type(payload) is not bytes or int(row["label"]) != label:
            raise ValueError("Cars transfer image differs")
        digests.append(hashlib.sha256(payload).digest())
    manifest_sha = hashlib.sha256(b"".join(digests)).hexdigest()
    labels = torch.tensor([row[2] for row in rows], dtype=torch.int64, device="cuda")
    torch.set_num_threads(20)
    torch.backends.cuda.matmul.allow_tf32 = False
    for seed, arm, receipt_sha, checkpoint_sha in CHECKPOINTS:
        output = RUN / f"{seed}-{arm}.json"
        if output.exists():
            raise ValueError(f"Cars transfer output already exists: {output}")
        train = Path(f"/home/riomus/runs/sfora-sop-true-freeze-{arm}-{seed}-1000-v1")
        if (
            sha(train / "receipt.json") != receipt_sha
            or sha(train / "checkpoint.pt") != checkpoint_sha
        ):
            raise ValueError("SOP transfer checkpoint differs")
        training = json.loads((train / "receipt.json").read_text())
        if training.get("seed") != seed or training.get("checkpoint_sha256") != checkpoint_sha:
            raise ValueError("SOP transfer training receipt differs")
        started = time.perf_counter()
        encoder = Siglip2CompactEncoder.from_checkpoint(
            model_snapshot=MODEL,
            checkpoint=train / "checkpoint.pt",
            expected_checkpoint_sha256=checkpoint_sha,
            model_file_sha256=training["model_file_sha256"],
            precision="fp16_native",
            device=torch.device("cuda:0"),
        )
        torch.cuda.synchronize()
        load_seconds = time.perf_counter() - started
        torch.cuda.reset_peak_memory_stats()
        codes, inverse_norms = [], []
        started = time.perf_counter()
        for offset in range(0, len(rows), 32):
            images = []
            for split, ordinal, label in rows[offset : offset + 32]:
                row = source[split][ordinal]
                payload = row["image"]["bytes"]
                if (
                    int(row["label"]) != label
                    or hashlib.sha256(payload).digest() != digests[offset + len(images)]
                ):
                    raise ValueError("Cars transfer image bytes changed")
                with Image.open(BytesIO(payload)) as image:
                    images.append(image.convert("RGB"))
            packed = encoder.encode_images(images)
            codes.append(packed.codes)
            inverse_norms.append(packed.inverse_norms)
        torch.cuda.synchronize()
        export_seconds = time.perf_counter() - started
        code = torch.cat(codes).float().cuda()
        inverse = torch.cat(inverse_norms).cuda()
        started = time.perf_counter()
        quality = score_symmetric(code, labels, inverse_norms=inverse)
        torch.cuda.synchronize()
        score_seconds = time.perf_counter() - started
        result = {
            "schema": "sfora-sop-cars-public-transfer-v1",
            "claim_eligible": False,
            "source_sha256": sha(Path(__file__)),
            "serving_sha256": sha(
                Path(__import__("sfora.siglip2_compact_serving", fromlist=["x"]).__file__)
            ),
            "dataset_revision": REVISION,
            "source_fingerprints": FINGERPRINTS,
            "cache_file_sha256": CACHE_SHA,
            "cars_image_manifest_sha256": manifest_sha,
            "split": "Cars196 classes 98-195 TEST, symmetric self-excluded",
            "rows": len(rows),
            "seed": seed,
            "arm": arm,
            "training_receipt_sha256": receipt_sha,
            "checkpoint_sha256": checkpoint_sha,
            "model_load_seconds": load_seconds,
            "export_seconds": export_seconds,
            "score_seconds": score_seconds,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "gallery_wire_bytes": len(rows) * 130,
            "quality": quality,
            "hardware": {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__},
        }
        output.write_text(json.dumps(result, sort_keys=True, allow_nan=False) + "\n")
        print(
            f"{seed} {arm} R1={quality['recall_at_1']:.6f} mAP@R={quality['map_at_r']:.6f}",
            flush=True,
        )
        del encoder, code, inverse, codes, inverse_norms
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
