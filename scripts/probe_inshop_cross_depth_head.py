#!/usr/bin/env python3
"""Fit-only 22-to-24-block head transfer on cached In-Shop TRAIN sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch
from compare_sop_siglip2_member_bank_arms import product_bootstrap
from export_inshop_siglip2_train_features import MODEL_HASHES
from preflight_inshop_siglip2_unseen_gallery import PARTITION_SHA, digest_rows, sha256, split
from torch.nn import functional as F
from train_sop_siglip2_compact import score_packed_full_gallery

from sfora.joint_relational_compaction import pack_int8_unit_embeddings
from sfora.representation_ceiling import fit_centered_pca
from sfora.unicom_inshop import parse_inshop_partition


def procrustes(fit_22: torch.Tensor, fit_24: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Least-squares orthogonal row map and fit-only translation."""

    if fit_22.shape != fit_24.shape or fit_22.ndim != 2 or fit_22.dtype != torch.float32:
        raise ValueError("cross-depth fit geometry differs")
    left, right = fit_22.double(), fit_24.double()
    mean_left, mean_right = left.mean(dim=0), right.mean(dim=0)
    u, _, vh = torch.linalg.svd((left - mean_left).T @ (right - mean_right))
    rotation = (u @ vh).float().contiguous()
    translation = (mean_right - mean_left @ (u @ vh)).float().contiguous()
    if not bool(torch.isfinite(rotation).all()) or not bool(torch.isfinite(translation).all()):
        raise ValueError("cross-depth map nonfinite")
    return rotation, translation


def main() -> None:
    if sys.argv[1:] == ["--self-test"]:
        torch.manual_seed(7)
        x = torch.randn(32, 8)
        q, _ = torch.linalg.qr(torch.randn(8, 8))
        y = x @ q + 0.2
        rotation, bias = procrustes(x, y)
        torch.testing.assert_close(x @ rotation + bias, y, rtol=0, atol=2e-6)
        print("cross-depth Procrustes self-test passed")
        return
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("dataset-root", "preflight", "full-cache", "short-cache", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() or not torch.cuda.is_available():
        raise ValueError("cross-depth output or CUDA authority differs")
    total_started = time.perf_counter()
    torch.set_num_threads(16)
    if sha256(args.dataset_root / "Eval/list_eval_partition.txt") != PARTITION_SHA:
        raise ValueError("cross-depth partition differs")
    preflight = json.loads(args.preflight.read_text())
    train = tuple(row for row in parse_inshop_partition(args.dataset_root) if row.split == "train")
    fit, held = split(tuple(row.label for row in train))
    row_sha = hashlib.sha256(
        "\n".join(
            f"{row.label}\0{row.image_path.relative_to(args.dataset_root)}" for row in train
        ).encode()
    ).hexdigest()
    if (
        len(train) != 25_882
        or len(fit) != 13_283
        or len(held) != 12_599
        or preflight["fit_sha256"] != digest_rows(fit)
        or preflight["held_sha256"] != digest_rows(held)
        or preflight["partition_sha256"] != PARTITION_SHA
    ):
        raise ValueError("cross-depth split differs")
    caches = {}
    for name, path, tail in (("full24", args.full_cache, 0), ("short22", args.short_cache, 2)):
        receipt = json.loads((path / "receipt.json").read_text())
        array = np.load(path / "train_features.npy", mmap_mode="r")
        if (
            receipt["schema"] != "sfora-inshop-siglip2-train-feature-export-v1"
            or receipt["partition_sha256"] != PARTITION_SHA
            or receipt["model_file_sha256"] != MODEL_HASHES
            or receipt["ordered_rows_sha256"] != row_sha
            or receipt.get("tail_blocks_dropped", 0) != tail
            or sha256(path / "train_features.npy") != receipt["features_sha256"]
            or array.shape != (len(train), 1024)
            or array.dtype != np.float32
        ):
            raise ValueError(f"cross-depth {name} cache differs")
        caches[name] = (array, receipt, sha256(path / "receipt.json"))
    if caches["full24"][1]["model_file_sha256"] != caches["short22"][1]["model_file_sha256"]:
        raise ValueError("cross-depth pretrained model differs")
    started = time.perf_counter()
    x = F.normalize(torch.from_numpy(np.asarray(caches["short22"][0][list(fit)]).copy()), dim=1)
    y = F.normalize(torch.from_numpy(np.asarray(caches["full24"][0][list(fit)]).copy()), dim=1)
    pca22 = fit_centered_pca(x, dimensions=128)
    pca24 = fit_centered_pca(y, dimensions=128)
    rotation, translation = procrustes(x, y)
    weight = (pca24.components @ rotation.T).contiguous()
    bias = (pca24.components @ (translation - pca24.mean)).contiguous()
    fit_seconds = time.perf_counter() - started
    held22 = F.normalize(
        torch.from_numpy(np.asarray(caches["short22"][0][list(held)]).copy()), dim=1
    )
    held24 = F.normalize(
        torch.from_numpy(np.asarray(caches["full24"][0][list(held)]).copy()), dim=1
    )
    features = {
        "short22_own_pca": pca22.apply(held22),
        "short22_transferred": F.normalize(held22 @ weight.T + bias, dim=1),
        "full24_own_pca": pca24.apply(held24),
    }
    names = tuple(train[row].label for row in held)
    encoded = {name: index for index, name in enumerate(sorted(set(names)))}
    labels = torch.tensor([encoded[name] for name in names], dtype=torch.long)
    quality = {}
    for name, values in features.items():
        if not bool(torch.isfinite(values).all()):
            raise ValueError("cross-depth projected feature nonfinite")
        wire = pack_int8_unit_embeddings(values)
        quality[name] = score_packed_full_gallery(
            wire.codes.float(),
            wire.inverse_norms,
            labels,
            torch.arange(len(held), dtype=torch.long),
            device=torch.device("cuda"),
        )
    own, transfer = quality["short22_own_pca"], quality["short22_transferred"]
    ap_delta = np.asarray(transfer["per_query_ap"]) - np.asarray(own["per_query_ap"])
    bootstrap = product_bootstrap(ap_delta, np.asarray(names))
    r1_delta = transfer["recall_at_1"] - own["recall_at_1"]
    advance = bootstrap["point"] >= 0.005 and bootstrap["lower_95"] > 0 and r1_delta >= 0
    args.output_dir.mkdir(parents=True, exist_ok=False)
    head_path = args.output_dir / "transferred_head.npz"
    np.savez(head_path, weight=weight.numpy(), bias=bias.numpy())
    report = {
        "schema": "sfora-inshop-cross-depth-head-falsifier-v1",
        "claim_eligible": False,
        "source_sha256": sha256(Path(__file__)),
        "source_files_sha256": {
            Path(function.__code__.co_filename).name: sha256(Path(function.__code__.co_filename))
            for function in (
                product_bootstrap,
                score_packed_full_gallery,
                pack_int8_unit_embeddings,
                fit_centered_pca,
                parse_inshop_partition,
                split,
            )
        },
        "partition_sha256": PARTITION_SHA,
        "preflight_sha256": sha256(args.preflight),
        "fit_rows_sha256": digest_rows(fit),
        "held_rows_sha256": digest_rows(held),
        "caches": {
            name: {"receipt_sha256": receipt_sha, "features_sha256": receipt["features_sha256"]}
            for name, (_array, receipt, receipt_sha) in caches.items()
        },
        "head_sha256": sha256(head_path),
        "fit_seconds": fit_seconds,
        "total_wall_seconds": time.perf_counter() - total_started,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_parent_host_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "map_at_r_delta_product_bootstrap": bootstrap,
        "packed_r1_delta": r1_delta,
        "advance_training": advance,
        "quality": quality,
        "hardware": {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__},
    }
    with (args.output_dir / "receipt.json").open("xb") as stream:
        stream.write((json.dumps(report, sort_keys=True, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    print(
        json.dumps(
            {
                "advance_training": advance,
                "map_at_r_delta_product_bootstrap": bootstrap,
                "packed_r1_delta": r1_delta,
                "quality": {
                    name: {"r1": x["recall_at_1"], "mapr": x["map_at_r"]}
                    for name, x in quality.items()
                },
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
