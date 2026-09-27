#!/usr/bin/env python3
"""Check the direct public processor against pinned In-Shop TRAIN roles."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from preflight_inshop_siglip2_unseen_gallery import split
from score_inshop_crop_view_pair import (
    GALLERY_SHA,
    NATIVE_SHA,
    PARTITION_SHA,
    QUERY_SHA,
    roles,
    sha256,
)

from sfora.siglip2_compact_serving import Siglip2CompactEncoder, Siglip2CompactIndex
from sfora.unicom_inshop import parse_inshop_partition

CHECKPOINT_SHA = "ad58838e2492cd664a447308317b11a53f362a4c5a36a99f28978f32a3b71089"
RECEIPT_SHA = "76f1293ac158f64f1e98a9412033ca37667a2baaadb592bf6051d92b61c9c8cc"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in (
        "dataset-root",
        "model-snapshot",
        "training-receipt",
        "checkpoint",
        "native-library",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.output.exists()
        or not torch.cuda.is_available()
        or sha256(args.dataset_root / "Eval/list_eval_partition.txt") != PARTITION_SHA
        or sha256(args.training_receipt) != RECEIPT_SHA
        or sha256(args.checkpoint) != CHECKPOINT_SHA
        or sha256(args.native_library) != NATIVE_SHA
    ):
        raise ValueError("In-Shop direct parity authority differs")
    training = json.loads(args.training_receipt.read_text())
    if training.get("seed") != 179026 or training.get("arm") != "freeze_emb":
        raise ValueError("In-Shop direct parity checkpoint differs")
    train = tuple(row for row in parse_inshop_partition(args.dataset_root) if row.split == "train")
    _, held = split(tuple(row.label for row in train))
    paths = tuple(train[row].image_path for row in held)
    labels = tuple(train[row].label for row in held)
    query, gallery = roles(labels, paths, args.dataset_root)
    if (len(query), len(gallery)) != (6_354, 6_245):
        raise ValueError("In-Shop direct parity role geometry differs")
    for rows, digest in ((query, QUERY_SHA), (gallery, GALLERY_SHA)):
        if hashlib.sha256(np.asarray(rows, dtype="<i4").tobytes()).hexdigest() != digest:
            raise ValueError("In-Shop direct parity role digest differs")
    torch.set_num_threads(20)
    torch.backends.cuda.matmul.allow_tf32 = False
    encoder = Siglip2CompactEncoder.from_checkpoint(
        model_snapshot=args.model_snapshot,
        checkpoint=args.checkpoint,
        expected_checkpoint_sha256=CHECKPOINT_SHA,
        model_file_sha256=training["model_file_sha256"],
        precision="fp16_native",
        device=torch.device("cuda:0"),
    )
    if not encoder._use_direct_preprocess:
        raise ValueError("In-Shop direct processor was not selected")
    rows = np.linspace(0, len(query) - 1, 640, dtype=np.int64)
    if len(set(rows.tolist())) != 640:
        raise ValueError("In-Shop direct query rows repeat")
    sampled = [paths[query[row]] for row in rows]
    with Siglip2CompactIndex.from_image_paths(
        encoder=encoder,
        native_library=args.native_library,
        image_paths=[paths[row] for row in gallery],
        expected_native_library_sha256=NATIVE_SHA,
    ) as index:
        assert index.gallery is not None
        for position, path in enumerate(sampled):
            with Image.open(path) as image:
                encoder._use_direct_preprocess = False
                base = encoder.encode_images([image])
                base_result = index.gallery.search_packed(base)
                encoder._use_direct_preprocess = True
                direct = encoder.encode_images([image])
                direct_result = index.gallery.search_packed(direct)
            if (
                not torch.equal(base.codes, direct.codes)
                or not torch.equal(base.inverse_norms, direct.inverse_norms)
                or not np.array_equal(base_result[0], direct_result[0])
                or not np.array_equal(base_result[1], direct_result[1])
            ):
                raise ValueError(f"In-Shop direct packed/top-10 parity differs at {position}")
            if (position + 1) % 160 == 0:
                print(json.dumps({"exact": position + 1}), flush=True)
        report = {
            "schema": "sfora-inshop-direct-public-parity-v1",
            "source_sha256": sha256(Path(__file__)),
            "serving_source_sha256": sha256(
                Path(__import__("sfora.siglip2_compact_serving", fromlist=["x"]).__file__)
            ),
            "training_receipt_sha256": RECEIPT_SHA,
            "checkpoint_sha256": CHECKPOINT_SHA,
            "native_library_sha256": NATIVE_SHA,
            "query_roles_sha256": QUERY_SHA,
            "gallery_roles_sha256": GALLERY_SHA,
            "query_rows": [int(query[row]) for row in rows],
            "query_image_sha256": [sha256(path) for path in sampled],
            "exact_packed_and_top10": True,
            "gallery_rows": len(gallery),
            "gallery_wire_bytes": 130 * len(gallery),
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_parent_host_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        }
    with args.output.open("xb") as stream:
        stream.write((json.dumps(report, sort_keys=True, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"exact": len(sampled), "gallery_rows": len(gallery)}), flush=True)


if __name__ == "__main__":
    main()
