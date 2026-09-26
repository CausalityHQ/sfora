#!/usr/bin/env python3
"""Locate the trained In-Shop unseen-gallery source/head/packing quality gap."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from compare_sop_siglip2_member_bank_arms import product_bootstrap
from export_inshop_siglip2_train_features import MODEL_HASHES
from export_sop_siglip2_train import MODEL_REVISION
from preflight_inshop_siglip2_unseen_gallery import PARTITION_SHA, digest_rows, sha256, split
from torch import nn
from torch.nn import functional as F
from train_sop_siglip2_compact import export_all, score_packed_full_gallery

from sfora.joint_relational_compaction import pack_int8_unit_embeddings
from sfora.unicom_inshop import parse_inshop_partition

RECEIPT_SHA = "d97a2cd402f29fe81d292880b38ae3fb1fddf9ca162c67b96d9a5b0528eb8f5c"
CHECKPOINT_SHA = "28b84beaad72ab04c0f9ac9d9e32a0a5b712d3733dd3d4402b0d135130dfbda8"
PREFLIGHT_SHA = "d5e22c6a331acbdf3b143b9836597593c2317f9488b99b72f61a4c54baf18eb8"
TRAIN_HELPER_SHA = "e2f7f8d16e2850a51aa85a3d3f4e04a80ee2a48681dcac55306fd792c8f19ba6"


@torch.inference_mode()
def float_top1_hits(values: torch.Tensor, labels: torch.Tensor) -> np.ndarray:
    if values.ndim != 2 or labels.shape != (len(values),) or not bool(torch.isfinite(values).all()):
        raise ValueError("In-Shop trained-width float geometry differs")
    hits = []
    for start in range(0, len(values), 64):
        stop = min(start + 64, len(values))
        scores = values[start:stop] @ values.T
        scores[
            torch.arange(stop - start, device=values.device),
            torch.arange(start, stop, device=values.device),
        ] = -torch.inf
        nearest = torch.argmax(scores, dim=1)
        hits.extend((labels[nearest] == labels[start:stop]).int().cpu().tolist())
    return np.asarray(hits, dtype=np.float64)


def main() -> None:
    if sys.argv[1:] == ["--self-test"]:
        values = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        labels = torch.tensor([0, 0, 1, 1])
        print(json.dumps({"hits": float_top1_hits(values, labels).astype(int).tolist()}))
        return
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt_path = args.run / "receipt.json"
    checkpoint_path = args.run / "checkpoint.pt"
    helper_file = sys.modules[export_all.__module__].__file__
    if (
        args.output.exists()
        or not torch.cuda.is_available()
        or sha256(args.dataset_root / "Eval/list_eval_partition.txt") != PARTITION_SHA
        or sha256(args.preflight) != PREFLIGHT_SHA
        or sha256(receipt_path) != RECEIPT_SHA
        or sha256(checkpoint_path) != CHECKPOINT_SHA
        or score_packed_full_gallery.__module__ != export_all.__module__
        or helper_file is None
        or sha256(Path(helper_file)) != TRAIN_HELPER_SHA
        or args.model_snapshot.resolve().name != MODEL_REVISION
        or any(
            sha256(args.model_snapshot / name) != digest for name, digest in MODEL_HASHES.items()
        )
    ):
        raise ValueError("In-Shop trained-width authority differs")
    receipt = json.loads(receipt_path.read_text())
    preflight = json.loads(args.preflight.read_text())
    train = tuple(row for row in parse_inshop_partition(args.dataset_root) if row.split == "train")
    _, held = split(tuple(row.label for row in train))
    if (
        receipt.get("arm") != "freeze"
        or receipt.get("seed") != 179023
        or receipt.get("held_rows_sha256") != digest_rows(held)
        or preflight.get("held_sha256") != digest_rows(held)
        or len(held) != 12_599
    ):
        raise ValueError("In-Shop trained-width split differs")
    torch.backends.cuda.matmul.allow_tf32 = False
    import transformers

    processor = transformers.AutoImageProcessor.from_pretrained(  # type: ignore[no-untyped-call]
        args.model_snapshot, local_files_only=True, backend="torchvision"
    )
    if (
        type(processor).__name__ != "SiglipImageProcessor"
        or processor.size["height"] != 256
        or processor.size["width"] != 256
        or processor.resample != 2
    ):
        raise ValueError("In-Shop trained-width processor differs")
    full_model = transformers.AutoModel.from_pretrained(
        args.model_snapshot, local_files_only=True, use_safetensors=True, dtype=torch.float16
    )
    vision = full_model.vision_model
    del full_model
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    vision = vision.float()
    vision.load_state_dict(checkpoint["vision"], strict=True)
    if any(
        not torch.equal(value, checkpoint["vision"][name])
        for name, value in vision.state_dict().items()
    ):
        raise ValueError("In-Shop trained-width checkpoint was rounded during load")
    vision = vision.cuda().eval()
    head = nn.Linear(1024, 128)
    head.load_state_dict(checkpoint["head"], strict=True)
    head = head.float().cuda().eval()
    source_batches: list[torch.Tensor] = []

    def capture(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
        source = output.pooler_output
        if source is None:
            raise ValueError("In-Shop trained-width source pooler missing")
        source_batches.append(F.normalize(source.float(), dim=1).cpu())

    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    hook = vision.register_forward_hook(capture)
    try:
        head_values = export_all(
            vision,
            head,
            tuple(train[i].image_path for i in held),
            tuple(range(len(held))),
            processor,
            workers=4,
            batch_size=64,
        )
    finally:
        hook.remove()
    source_values = torch.cat(source_batches).contiguous()
    if source_values.shape != (len(held), 1024) or head_values.shape != (len(held), 128):
        raise ValueError("In-Shop trained-width exported geometry differs")
    held_labels = np.asarray([train[i].label for i in held])
    encoded = {name: i for i, name in enumerate(sorted(set(held_labels)))}
    label_ids = torch.tensor(
        [encoded[name] for name in held_labels], dtype=torch.int64, device="cuda"
    )
    source_hits = float_top1_hits(source_values.cuda(), label_ids)
    head_hits = float_top1_hits(head_values.cuda(), label_ids)
    packed = pack_int8_unit_embeddings(head_values)
    packed_quality = score_packed_full_gallery(
        packed.codes.float(),
        packed.inverse_norms,
        label_ids.cpu(),
        torch.arange(len(held), dtype=torch.int64),
        device=torch.device("cuda"),
    )
    mismatches = sum(
        a != b
        for a, b in zip(
            packed_quality["per_query_r1"], receipt["quality"]["per_query_r1"], strict=True
        )
    )
    map_delta = packed_quality["map_at_r"] - receipt["quality"]["map_at_r"]
    if mismatches or abs(map_delta) > 1e-8:
        raise ValueError(
            "In-Shop trained-width packed result fails checkpoint parity: "
            f"r1_mismatches={mismatches}, "
            f"r1={packed_quality['recall_at_1']}/{receipt['quality']['recall_at_1']}, "
            f"map={packed_quality['map_at_r']}/{receipt['quality']['map_at_r']}"
        )
    source_minus_head = product_bootstrap(source_hits - head_hits, held_labels)
    head_minus_packed = product_bootstrap(
        head_hits - np.asarray(packed_quality["per_query_r1"], dtype=np.float64),
        held_labels,
    )
    quality_report = {
        "source_1024_float_r1": float(source_hits.mean()),
        "head_128_float_r1": float(head_hits.mean()),
        "head_128_packed_r1": packed_quality["recall_at_1"],
        "head_128_packed_map_at_r": packed_quality["map_at_r"],
        "source_minus_head_product_bootstrap": source_minus_head,
        "head_minus_packed_product_bootstrap": head_minus_packed,
        "source_per_query_r1": source_hits.tolist(),
        "head_float_per_query_r1": head_hits.tolist(),
    }
    report = {
        "schema": "sfora-inshop-siglip2-trained-width-train-only-v1",
        "claim_eligible": False,
        "split": "official TRAIN product-disjoint held-only symmetric gallery",
        "held_queries_and_gallery": len(held),
        "seed": 179023,
        "arm": "freeze",
        "quality": quality_report,
        "packed_receipt_parity": {
            "r1_mismatches": mismatches,
            "r1_ambiguity_fraction": mismatches / len(held),
            "map_at_r_delta": map_delta,
            "exact": mismatches == 0 and abs(map_delta) <= 1e-8,
        },
        "source_receipt_sha256": RECEIPT_SHA,
        "checkpoint_sha256": CHECKPOINT_SHA,
        "preflight_sha256": PREFLIGHT_SHA,
        "model_file_sha256": MODEL_HASHES,
        "script_sha256": sha256(Path(__file__)),
        "helper_sha256": {
            "export_all": TRAIN_HELPER_SHA,
            "score_packed_full_gallery": TRAIN_HELPER_SHA,
            "product_bootstrap": sha256(Path(product_bootstrap.__code__.co_filename)),
            "pack_int8_unit_embeddings": sha256(
                Path(pack_int8_unit_embeddings.__code__.co_filename)
            ),
        },
        "wall_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "hardware": {
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
    }
    with args.output.open("xb") as stream:
        stream.write((json.dumps(report, sort_keys=True, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    print(
        json.dumps({k: v for k, v in quality_report.items() if not k.endswith("per_query_r1")}),
        flush=True,
    )


if __name__ == "__main__":
    main()
