#!/usr/bin/env python3
"""Fixed SOP control/freeze checkpoint transfer to CUB's class-disjoint test split."""

from __future__ import annotations

import gc
import hashlib
import json
import time
from contextlib import ExitStack
from pathlib import Path

import torch
from export_unicom_cub_embeddings import (
    CUB_ARCHIVE_SHA256,
    ordered_record_sha256,
    parse_cub_records,
    verify_extracted_cub_matches_archive,
)
from PIL import Image

from sfora.siglip2_compact_serving import Siglip2CompactEncoder
from sfora.sop_evaluation import score_symmetric

RUN = Path("/home/riomus/runs/sfora-sop-cub-transfer-public-v1")
DATA = Path("/home/riomus/datasets/CUB_200_2011_official")
MODEL = Path(
    "/home/riomus/.cache/huggingface/hub/models--google--siglip2-large-patch16-256/"
    "snapshots/787800c8990e6f058423089178e718139608408c"
)
CHECKPOINTS = (
    (
        179024,
        "control",
        "1bc737d77ea19b40b74ca6ca967923696403c656c42c17930ace732f8ca0965f",
        "c8bd17acfda9b4a0996521327135ed50bd883ab12025a0fa0cd44bd58b721b47",
    ),
    (
        179024,
        "freeze",
        "07b4716b42d1291b9c195774ebd48d9df89a3b578ee54efdb94662c5e125d1c5",
        "2c838561b6c23242d74eb29329fd026cc8fba9bf965dcc4529348028dfe6d172",
    ),
    (
        179026,
        "control",
        "c69de5e542d2d919de5875b4313904efafd8d03d5a0271a0bc0199b45dbd14e0",
        "070b85d41c3e5397074c245ec27621cfba8fe862690e8720cef50a8e76a908f5",
    ),
    (
        179026,
        "freeze",
        "a3153e1d2429560ea5902bd68d3e43bee17bd0509330a581a48b40e6a824ea8a",
        "b9fd0033d6513712495882544a36998d86ee0ba34322b376757063ef65ca68a7",
    ),
    (
        179027,
        "control",
        "35b7f82ccfa44adf9bfb58c02711d017dea88a49e7b8e859b9437458a9fa8c4a",
        "fee8be498835089323e58869c9d0c184bce9885f61130c6785bb69fce9c2ecca",
    ),
    (
        179027,
        "freeze",
        "9fe8e88615ebf653cb9f0ca890b7eac472d84297fbf78370ed3b1cc5bf77b633",
        "14da31d9e9cc4c4ded146b1cf21c04d22b695e78737380a84955abb1e2520f99",
    ),
)


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    if not torch.cuda.is_available() or sha(DATA / "CUB_200_2011.tgz") != CUB_ARCHIVE_SHA256:
        raise ValueError("CUB transfer source differs")
    records = tuple(
        record
        for record in parse_cub_records(DATA / "extracted/CUB_200_2011")
        if record.split == "test"
    )
    if len(records) != 5924 or len({row.label for row in records}) != 100:
        raise ValueError("CUB test inventory differs")
    archive_manifest = verify_extracted_cub_matches_archive(
        DATA / "CUB_200_2011.tgz", DATA / "extracted/CUB_200_2011", records
    )
    paths = [row.image_path for row in records]
    labels = torch.tensor([row.label for row in records], dtype=torch.int64, device="cuda")
    torch.set_num_threads(20)
    torch.backends.cuda.matmul.allow_tf32 = False
    for seed, arm, receipt_sha, checkpoint_sha in CHECKPOINTS:
        output = RUN / f"{seed}-{arm}.json"
        if output.exists():
            raise ValueError(f"CUB transfer output already exists: {output}")
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
        for offset in range(0, len(paths), 32):
            with ExitStack() as stack:
                images = [
                    stack.enter_context(Image.open(path)) for path in paths[offset : offset + 32]
                ]
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
            "schema": "sfora-sop-cub-public-transfer-v1",
            "claim_eligible": False,
            "source_sha256": sha(Path(__file__)),
            "serving_sha256": sha(
                Path(__import__("sfora.siglip2_compact_serving", fromlist=["x"]).__file__)
            ),
            "cub_archive_sha256": CUB_ARCHIVE_SHA256,
            "cub_extracted_manifest_sha256": archive_manifest,
            "cub_test_records_sha256": ordered_record_sha256(records),
            "split": "CUB-200-2011 classes 101-200 TEST, symmetric self-excluded",
            "rows": len(paths),
            "seed": seed,
            "arm": arm,
            "training_receipt_sha256": receipt_sha,
            "checkpoint_sha256": checkpoint_sha,
            "model_load_seconds": load_seconds,
            "export_seconds": export_seconds,
            "score_seconds": score_seconds,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "gallery_wire_bytes": len(paths) * 130,
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
