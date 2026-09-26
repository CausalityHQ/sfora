#!/usr/bin/env python3
"""Rescore source-bound In-Shop TRAIN checkpoints on an asymmetric held gallery."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import cast

import numpy as np
import torch
from compare_sop_siglip2_member_bank_arms import product_bootstrap
from export_inshop_siglip2_train_features import MODEL_HASHES
from export_sop_siglip2_train import MODEL_REVISION
from preflight_inshop_siglip2_unseen_gallery import PARTITION_SHA, digest_rows, sha256, split
from torch import nn
from train_sop_siglip2_compact import export_all, score_packed_full_gallery

from sfora.joint_relational_compaction import pack_int8_unit_embeddings
from sfora.unicom_inshop import parse_inshop_partition

PREFLIGHT_SHA = "f9c59db9ed6f0962963b8e203e70f98186f314226acda0f0ebd7b52c11e17034"
RUNS = (
    ("control", "c240755f35566038104a8f671c3fab86d1b12658bec55fe666d0df84e707160b"),
    ("freeze", "14049b181066b35a0bec216e577932f8af9b792cf187a7a2fdddf5e4c5eee1c9"),
    ("freeze_emb", "fb1c41d341d738676ff12c72f14b5eee66fd9d633fef9b9306c1ffd080908e89"),
)
HELPER_SHA = "e2f7f8d16e2850a51aa85a3d3f4e04a80ee2a48681dcac55306fd792c8f19ba6"


@torch.inference_mode()
def score_rectangular(
    codes: torch.Tensor,
    inverse: torch.Tensor,
    labels: torch.Tensor,
    query: torch.Tensor,
    gallery: torch.Tensor,
    *,
    device: torch.device,
) -> dict[str, object]:
    if (
        codes.ndim != 2
        or codes.shape[1] != 128
        or inverse.shape != (len(codes),)
        or labels.shape != (len(codes),)
        or query.ndim != 1
        or gallery.ndim != 1
        or not len(query)
        or not len(gallery)
        or int(query.min()) < 0
        or int(gallery.min()) < 0
        or int(query.max()) >= len(codes)
        or int(gallery.max()) >= len(codes)
        or bool(torch.isin(query, gallery).any())
    ):
        raise ValueError("In-Shop asymmetric packed geometry differs")
    code = codes.float().to(device)
    norm = inverse.float().to(device)
    label = labels.to(device)
    query = query.to(device)
    gallery = gallery.to(device)
    relevant = torch.bincount(label[gallery], minlength=int(label.max()) + 1)[label[query]]
    if int(relevant.min()) < 1:
        raise ValueError("In-Shop asymmetric positive inventory differs")
    width = int(relevant.max())
    ranks = torch.arange(1, width + 1, device=device)
    r1: list[float] = []
    ap: list[float] = []
    for block in query.split(64):  # type: ignore[no-untyped-call]
        scores = (code[block] @ code[gallery].T) * norm[block, None] * norm[gallery][None, :]
        ranked = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :width]
        matches = label[gallery][ranked] == label[block, None]
        count = relevant[len(r1) : len(r1) + len(block)]
        precision = matches.cumsum(dim=1) / ranks[None, :]
        block_ap = (precision * matches * (ranks[None, :] <= count[:, None])).sum(dim=1) / count
        r1.extend(float(value) for value in matches[:, 0].cpu().tolist())
        ap.extend(float(value) for value in block_ap.cpu().tolist())
    return {
        "recall_at_1": float(np.mean(r1)),
        "map_at_r": float(np.mean(ap)),
        "per_query_r1": r1,
        "per_query_ap": ap,
    }


def main() -> None:
    if sys.argv[1:] == ["--self-test"]:
        codes = torch.zeros((4, 128), dtype=torch.int8)
        codes[:, 0] = 127
        result = score_rectangular(
            codes,
            torch.ones(4),
            torch.tensor([0, 1, 1, 0]),
            torch.tensor([0]),
            torch.tensor([1, 3]),
            device=torch.device("cpu"),
        )
        assert result["per_query_r1"] == [0.0] and result["per_query_ap"] == [0.0]
        print("asymmetric stable-tie self-test passed")
        return
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--freeze-emb", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    script_sha = sha256(Path(__file__))
    helper_file = sys.modules[export_all.__module__].__file__
    if (
        args.output.exists()
        or not torch.cuda.is_available()
        or sha256(args.preflight) != PREFLIGHT_SHA
        or sha256(args.dataset_root / "Eval/list_eval_partition.txt") != PARTITION_SHA
        or args.model_snapshot.resolve().name != MODEL_REVISION
        or any(
            sha256(args.model_snapshot / name) != digest for name, digest in MODEL_HASHES.items()
        )
        or helper_file is None
        or sha256(Path(helper_file)) != HELPER_SHA
    ):
        raise ValueError("In-Shop asymmetric authority differs")
    train = tuple(row for row in parse_inshop_partition(args.dataset_root) if row.split == "train")
    _, held = split(tuple(row.label for row in train))
    preflight = json.loads(args.preflight.read_text())
    if len(held) != 12_599 or digest_rows(held) != preflight["held_sha256"]:
        raise ValueError("In-Shop asymmetric held split differs")
    held_labels = tuple(train[index].label for index in held)
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, label in enumerate(held_labels):
        grouped[label].append(index)
    query_rows: list[int] = []
    gallery_rows: list[int] = []
    for label in sorted(grouped):
        rows = sorted(
            grouped[label],
            key=lambda index: hashlib.sha256(
                str(train[held[index]].image_path.relative_to(args.dataset_root)).encode()
            ).digest(),
        )
        gallery_count = max(1, min(len(rows) - 1, round(len(rows) / 2)))
        gallery_rows.extend(rows[:gallery_count])
        query_rows.extend(rows[gallery_count:])
    query_rows.sort()
    gallery_rows.sort()
    if len(query_rows) != 6_354 or len(gallery_rows) != 6_245 or len(grouped) != 1_993:
        raise ValueError("In-Shop asymmetric partition differs")
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
        raise ValueError("In-Shop asymmetric processor differs")
    encoded = {name: index for index, name in enumerate(sorted(grouped))}
    label_ids = torch.tensor([encoded[name] for name in held_labels], dtype=torch.long)
    paths = tuple(train[index].image_path for index in held)
    runs = (args.control, args.freeze, args.freeze_emb)
    arms: dict[str, dict[str, object]] = {}
    for run, (arm, receipt_sha) in zip(runs, RUNS, strict=True):
        receipt_path = run / "receipt.json"
        checkpoint_path = run / "checkpoint.pt"
        if sha256(receipt_path) != receipt_sha:
            raise ValueError(f"In-Shop asymmetric {arm} source receipt differs")
        receipt = json.loads(receipt_path.read_text())
        if (
            receipt.get("seed") != 179024
            or receipt.get("preflight_sha256") != PREFLIGHT_SHA
            or receipt.get("held_rows_sha256") != preflight["held_sha256"]
            or sha256(checkpoint_path) != receipt.get("checkpoint_sha256")
        ):
            raise ValueError(f"In-Shop asymmetric {arm} checkpoint differs")
        full_model = transformers.AutoModel.from_pretrained(
            args.model_snapshot, local_files_only=True, use_safetensors=True, dtype=torch.float16
        )
        vision = full_model.vision_model.float()
        del full_model
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        vision.load_state_dict(checkpoint["vision"], strict=True)
        if any(
            not torch.equal(value, checkpoint["vision"][name])
            for name, value in vision.state_dict().items()
        ):
            raise ValueError(f"In-Shop asymmetric {arm} checkpoint rounded during load")
        head = nn.Linear(1024, 128).float()
        head.load_state_dict(checkpoint["head"], strict=True)
        vision = vision.cuda().eval()
        head = head.cuda().eval()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        values = export_all(
            vision, head, paths, tuple(range(len(held))), processor, workers=4, batch_size=64
        )
        export_seconds = time.perf_counter() - started
        packed = pack_int8_unit_embeddings(values)
        symmetric = score_packed_full_gallery(
            packed.codes.float(),
            packed.inverse_norms,
            label_ids,
            torch.arange(len(held), dtype=torch.long),
            device=torch.device("cuda"),
        )
        if (
            symmetric["per_query_r1"] != receipt["quality"]["per_query_r1"]
            or abs(symmetric["map_at_r"] - receipt["quality"]["map_at_r"]) > 1e-8
        ):
            raise ValueError(f"In-Shop asymmetric {arm} export fails exact symmetric parity")
        scored = score_rectangular(
            packed.codes,
            packed.inverse_norms,
            label_ids,
            torch.tensor(query_rows),
            torch.tensor(gallery_rows),
            device=torch.device("cuda"),
        )
        torch.cuda.synchronize()
        arms[arm] = {
            "receipt_sha256": receipt_sha,
            "checkpoint_sha256": receipt["checkpoint_sha256"],
            "symmetric_exact_parity": True,
            "asymmetric_quality": scored,
            "export_seconds": export_seconds,
            "total_wall_seconds": time.perf_counter() - started,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        }
        del vision, head, values, packed
        gc.collect()
        torch.cuda.empty_cache()
    control = arms["control"]["asymmetric_quality"]
    if not isinstance(control, dict):
        raise ValueError("In-Shop asymmetric control quality differs")
    contrasts = {}
    query_labels = np.asarray([held_labels[index] for index in query_rows])
    for arm in ("freeze", "freeze_emb"):
        trial = arms[arm]["asymmetric_quality"]
        if not isinstance(trial, dict):
            raise ValueError("In-Shop asymmetric treatment quality differs")
        contrasts[arm] = {
            "r1_product_bootstrap": product_bootstrap(
                np.asarray(trial["per_query_r1"]) - np.asarray(control["per_query_r1"]),
                query_labels,
            ),
            "map_at_r_product_bootstrap": product_bootstrap(
                np.asarray(trial["per_query_ap"]) - np.asarray(control["per_query_ap"]),
                query_labels,
            ),
        }
    symmetric_miss = (
        1.0 - json.loads((args.control / "receipt.json").read_text())["quality"]["recall_at_1"]
    )
    asymmetric_miss = 1.0 - control["recall_at_1"]
    report = {
        "schema": "sfora-inshop-asymmetric-train-proxy-v1",
        "claim_eligible": False,
        "seed": 179024,
        "query_images": len(query_rows),
        "gallery_images": len(gallery_rows),
        "held_products": len(grouped),
        "query_rows_sha256": hashlib.sha256(
            np.asarray(query_rows, dtype="<i4").tobytes()
        ).hexdigest(),
        "gallery_rows_sha256": hashlib.sha256(
            np.asarray(gallery_rows, dtype="<i4").tobytes()
        ).hexdigest(),
        "control_miss_rate_ratio_to_symmetric": asymmetric_miss / symmetric_miss,
        "miss_enrichment_at_least_2x": asymmetric_miss >= 2.0 * symmetric_miss,
        "arms": arms,
        "contrasts": contrasts,
        "script_sha256": script_sha,
        "preflight_sha256": PREFLIGHT_SHA,
        "model_file_sha256": MODEL_HASHES,
        "hardware": {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__},
    }
    if sha256(Path(__file__)) != script_sha:
        raise ValueError("In-Shop asymmetric source changed during execution")
    with args.output.open("xb") as stream:
        stream.write((json.dumps(report, sort_keys=True, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    print(
        json.dumps(
            {
                "miss_ratio": report["control_miss_rate_ratio_to_symmetric"],
                "r1": {
                    name: cast(dict[str, object], arm["asymmetric_quality"])["recall_at_1"]
                    for name, arm in arms.items()
                },
            }
        )
    )


if __name__ == "__main__":
    main()
