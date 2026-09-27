#!/usr/bin/env python3
"""Fit-only dual-depth PCA head screen on cached In-Shop TRAIN features."""

from __future__ import annotations

import argparse
import json
import os
import resource
import time
from pathlib import Path

import numpy as np
import torch
from compare_sop_siglip2_member_bank_arms import product_bootstrap
from preflight_inshop_siglip2_unseen_gallery import PARTITION_SHA, digest_rows, sha256, split
from torch.nn import functional as F
from train_sop_siglip2_compact import score_packed_full_gallery

from sfora.joint_relational_compaction import pack_int8_unit_embeddings
from sfora.representation_ceiling import fit_centered_pca
from sfora.unicom_inshop import parse_inshop_partition

BASELINE_SHA = "943faa33e74a1bbf769e6b39d86482ac024a85f574ff6b829dc7328a034c84d7"
CACHE_SHA = {
    "short22": (
        "60b277021c858124310b00937e947805fb70739ccaebe8d9142b3bbf6aea2704",
        "187f8e8637bba29afa85f954cb29a6b90b7eeb9012e858e5b7319cdb3ad3d7e4",
    ),
    "full24": (
        "da3a2cdfb3d4fc90bfb562c15c9eb0e3b1747b6547883501b596f00ef8ec6e1b",
        "f232584491bf4ed1cf75daa7fcc03f218e7db5df797f4d57dd2bf034ac110885",
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("dataset-root", "short-cache", "full-cache", "baseline", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.output.exists()
        or not torch.cuda.is_available()
        or sha256(args.dataset_root / "Eval/list_eval_partition.txt") != PARTITION_SHA
        or sha256(args.baseline) != BASELINE_SHA
    ):
        raise ValueError("dual-depth In-Shop authority differs")
    baseline = json.loads(args.baseline.read_text())
    train = tuple(row for row in parse_inshop_partition(args.dataset_root) if row.split == "train")
    fit, held = split(tuple(row.label for row in train))
    if (
        len(train) != 25_882
        or len(fit) != 13_283
        or len(held) != 12_599
        or digest_rows(fit) != baseline["fit_rows_sha256"]
        or digest_rows(held) != baseline["held_rows_sha256"]
    ):
        raise ValueError("dual-depth In-Shop split differs")
    arrays = {}
    for name, path in (("short22", args.short_cache), ("full24", args.full_cache)):
        expected_receipt, expected_features = CACHE_SHA[name]
        if (
            sha256(path / "receipt.json") != expected_receipt
            or sha256(path / "train_features.npy") != expected_features
        ):
            raise ValueError(f"dual-depth {name} cache differs")
        array = np.load(path / "train_features.npy", mmap_mode="r")
        if array.shape != (25_882, 1024) or array.dtype != np.float32:
            raise ValueError(f"dual-depth {name} features differ")
        arrays[name] = array
    torch.set_num_threads(16)
    torch.backends.cuda.matmul.allow_tf32 = False
    started = time.perf_counter()

    def joined(rows: tuple[int, ...]) -> torch.Tensor:
        return torch.cat(
            [
                F.normalize(torch.from_numpy(np.asarray(arrays[name][list(rows)]).copy()), dim=1)
                for name in ("short22", "full24")
            ],
            dim=1,
        ).contiguous()

    transform = fit_centered_pca(joined(fit), dimensions=128)
    fit_seconds = time.perf_counter() - started
    values = transform.apply(joined(held))
    wire = pack_int8_unit_embeddings(values)
    names = tuple(train[row].label for row in held)
    classes = {name: index for index, name in enumerate(sorted(set(names)))}
    quality = score_packed_full_gallery(
        wire.codes.float(),
        wire.inverse_norms,
        torch.tensor([classes[name] for name in names]),
        torch.arange(len(held)),
        device=torch.device("cuda"),
    )
    old = baseline["quality"]["short22_own_pca"]
    delta = quality["map_at_r"] - old["map_at_r"]
    interval = product_bootstrap(
        np.asarray(quality["per_query_ap"]) - np.asarray(old["per_query_ap"]),
        np.asarray(names),
    )
    report = {
        "schema": "sfora-inshop-dual-depth-head-preflight-v1",
        "claim_eligible": False,
        "source_sha256": sha256(Path(__file__)),
        "baseline_sha256": BASELINE_SHA,
        "cache_sha256": CACHE_SHA,
        "fit_rows_sha256": digest_rows(fit),
        "held_rows_sha256": digest_rows(held),
        "fit_seconds": fit_seconds,
        "total_wall_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_parent_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "quality": quality,
        "baseline": {"r1": old["recall_at_1"], "map_at_r": old["map_at_r"]},
        "map_at_r_delta": delta,
        "map_at_r_delta_product_bootstrap": interval,
        "advance_training": (
            delta >= 0.005
            and interval["lower_95"] > 0
            and quality["recall_at_1"] >= old["recall_at_1"]
        ),
        "hardware": {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__},
    }
    if sha256(Path(__file__)) != report["source_sha256"]:
        raise ValueError("dual-depth probe source changed during execution")
    with args.output.open("xb") as stream:
        stream.write((json.dumps(report, sort_keys=True, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    print(
        json.dumps(
            {
                "r1": quality["recall_at_1"],
                "mapr": quality["map_at_r"],
                "advance": report["advance_training"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
