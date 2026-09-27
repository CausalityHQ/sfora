#!/usr/bin/env python3
"""Short paired In-Shop TRAIN public latency screen before p99 certification."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from certify_inshop_siglip2_direct_parity import CHECKPOINT_SHA, RECEIPT_SHA
from certify_sop_siglip2_serving_threads_p99 import (
    decode,
    digest_result,
    sha256,
    summary,
    timed_call,
)
from preflight_inshop_siglip2_unseen_gallery import split
from score_inshop_crop_view_pair import GALLERY_SHA, NATIVE_SHA, PARTITION_SHA, QUERY_SHA, roles

from sfora.siglip2_compact_serving import Siglip2CompactEncoder, Siglip2CompactIndex
from sfora.unicom_inshop import parse_inshop_partition

PARITY_SHA = "09a426dd90911afe271e61eb6b026dd5fbbee98ed02ef48f110beffb54f9055d"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in (
        "dataset-root",
        "model-snapshot",
        "training-receipt",
        "checkpoint",
        "native-library",
        "parity-receipt",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.output.exists()
        or not torch.cuda.is_available()
        or torch.get_num_threads() != 20
        or sha256(args.dataset_root / "Eval/list_eval_partition.txt") != PARTITION_SHA
        or sha256(args.training_receipt) != RECEIPT_SHA
        or sha256(args.checkpoint) != CHECKPOINT_SHA
        or sha256(args.native_library) != NATIVE_SHA
        or sha256(args.parity_receipt) != PARITY_SHA
    ):
        raise ValueError("In-Shop direct pilot authority differs")
    parity = json.loads(args.parity_receipt.read_text())
    if parity.get("exact_packed_and_top10") is not True:
        raise ValueError("In-Shop direct pilot parity differs")
    training = json.loads(args.training_receipt.read_text())
    train = tuple(row for row in parse_inshop_partition(args.dataset_root) if row.split == "train")
    _, held = split(tuple(row.label for row in train))
    paths = tuple(train[row].image_path for row in held)
    labels = tuple(train[row].label for row in held)
    query, gallery = roles(labels, paths, args.dataset_root)
    for rows, digest in ((query, QUERY_SHA), (gallery, GALLERY_SHA)):
        if hashlib.sha256(np.asarray(rows, dtype="<i4").tobytes()).hexdigest() != digest:
            raise ValueError("In-Shop direct pilot role digest differs")
    selected = np.linspace(0, 639, 64, dtype=np.int64)
    query_paths = [paths[parity["query_rows"][int(row)]] for row in selected]
    if [sha256(path) for path in query_paths] != [
        parity["query_image_sha256"][int(row)] for row in selected
    ]:
        raise ValueError("In-Shop direct pilot image bytes differ")
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
        raise ValueError("In-Shop direct pilot processor was not selected")
    raw = {label: [] for label in ("baseline", "direct")}
    with Siglip2CompactIndex.from_image_paths(
        encoder=encoder,
        native_library=args.native_library,
        image_paths=[paths[row] for row in gallery],
        expected_native_library_sha256=NATIVE_SHA,
    ) as index:
        assert index.gallery is not None
        expected = []
        for path in query_paths:
            image = decode([path])
            encoder._use_direct_preprocess = False
            baseline = encoder.encode_images(image)
            baseline_result = index.gallery.search_packed(baseline)
            encoder._use_direct_preprocess = True
            direct = encoder.encode_images(image)
            direct_result = index.gallery.search_packed(direct)
            if (
                not torch.equal(baseline.codes, direct.codes)
                or not torch.equal(baseline.inverse_norms, direct.inverse_norms)
                or not np.array_equal(baseline_result[0], direct_result[0])
                or not np.array_equal(baseline_result[1], direct_result[1])
            ):
                raise ValueError("In-Shop direct pilot packed/top-10 parity differs")
            expected.append(digest_result(baseline_result))
        timing_started = time.perf_counter()
        for i, path in enumerate(query_paths):
            order = (
                ("baseline", "direct", "direct", "baseline")
                if i % 2 == 0
                else ("direct", "baseline", "baseline", "direct")
            )
            for label in order:
                encoder._use_direct_preprocess = label == "direct"
                raw[label].append(timed_call(index, [path], expected[i]))
        timing_wall_s = time.perf_counter() - timing_started
    statistics = {label: summary([raw[label]], 1) for label in raw}
    passed = (
        timing_wall_s <= 60
        and statistics["direct"]["p50_ms"] <= 0.95 * statistics["baseline"]["p50_ms"]
        and statistics["direct"]["p95_ms"] <= 0.98 * statistics["baseline"]["p95_ms"]
    )
    report = {
        "schema": "sfora-inshop-direct-public-pilot-v1",
        "source_sha256": sha256(Path(__file__)),
        "serving_source_sha256": sha256(
            Path(__import__("sfora.siglip2_compact_serving", fromlist=["x"]).__file__)
        ),
        "parity_receipt_sha256": PARITY_SHA,
        "training_receipt_sha256": RECEIPT_SHA,
        "checkpoint_sha256": CHECKPOINT_SHA,
        "native_library_sha256": NATIVE_SHA,
        "selected_parity_ordinals": selected.tolist(),
        "timing_wall_seconds": timing_wall_s,
        "packed_and_top10_exact": True,
        "pilot_rule": "timing <= 60 s, p50 direct <= .95 baseline, p95 direct <= .98 baseline",
        "advance_p99": passed,
        "statistics": statistics,
        "raw_ns": raw,
    }
    with args.output.open("xb") as stream:
        stream.write((json.dumps(report, sort_keys=True, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    print(
        json.dumps(
            {"advance_p99": passed, "statistics": statistics, "timing_wall_seconds": timing_wall_s}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
