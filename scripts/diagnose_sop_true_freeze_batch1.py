#!/usr/bin/env python3
"""Screen public batch-1 SOP queries against the pinned batch-32 official gallery."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import resource
import time
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from compare_sop_siglip2_member_bank_arms import product_bootstrap
from PIL import Image
from train_sop_siglip2_compact import paths_from_archive
from verify_sop_true_freeze_official_loader import SOURCE_HASHES, digest

from sfora import (
    cutile_int8,
    joint_relational_compaction,
    siglip2_compact_serving,
    sop_compact_training,
)
from sfora.joint_relational_compaction import PackedInt8Embeddings, pack_int8_unit_embeddings
from sfora.siglip2_compact_serving import Siglip2CompactIndex


def main() -> None:
    total_started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    root = Path("/home/riomus/runs")
    output = root / (
        "sfora-sop-true-freeze-official-batch1-full-179024-v1.json"
        if args.full
        else "sfora-sop-true-freeze-official-batch1-179024-v1.json"
    )
    if output.exists() or not torch.cuda.is_available():
        raise ValueError("SOP batch-1 diagnostic invocation differs")
    imported = {
        name: digest(Path(inspect.getfile(module)))
        for name, module in (
            ("siglip2_compact_serving", siglip2_compact_serving),
            ("joint_relational_compaction", joint_relational_compaction),
            ("cutile_int8", cutile_int8),
            ("sop_compact_training", sop_compact_training),
        )
    }
    if imported != SOURCE_HASHES:
        raise ValueError("SOP batch-1 imported source differs")
    archive = Path("/home/riomus/sfora-relational-sop-e1/unicom-l14-sop-v1.npz")
    manifest_path = root / "sfora-sop-reference-b8f85611-179019/sop-test-image-sha256.bin"
    training = root / "sfora-sop-true-freeze-freeze-179024-1000-v1"
    official = root / "sfora-sop-true-freeze-public-official-179024-freeze-v1"
    training_receipt = training / "receipt.json"
    official_receipt = official / "receipt.json"
    if (
        digest(archive) != "1ba27b2d6b9db39067aa6facd0ef8aafc303c4527f6feabed859b0512c7d921a"
        or digest(manifest_path)
        != "28a3ec0561cd83ee426f3d1c301c70799316af91c1e9083a5a1ffdf3414327c1"
        or digest(training_receipt)
        != "07b4716b42d1291b9c195774ebd48d9df89a3b578ee54efdb94662c5e125d1c5"
        or digest(official_receipt)
        != "7f5edf6b8ecb526e8ba29944119035245725055787cb088bd2635cc734b0eb6c"
    ):
        raise ValueError("SOP batch-1 pinned authority differs")
    with np.load(archive, allow_pickle=False) as source:
        labels = np.asarray(source["test_labels"], dtype=np.int64)
        relatives = np.asarray(source["test_relative_paths"]).astype(str)
    if labels.shape != (60_502,) or relatives.shape != labels.shape:
        raise ValueError("SOP batch-1 inventory differs")
    sample = (
        np.arange(len(labels), dtype=np.int64)
        if args.full
        else np.sort(np.random.default_rng(20260926).choice(len(labels), 2_048, replace=False))
    )
    paths = paths_from_archive(
        Path("/home/riomus/datasets/Stanford_Online_Products"), relatives[sample]
    )
    manifest = manifest_path.read_bytes()
    official_quality = json.loads(official_receipt.read_text())["packed_quality"]
    reference = np.asarray(official_quality["per_query_r1"], dtype=np.float64)
    reference_ap = np.asarray(official_quality["per_query_ap"], dtype=np.float64)
    if reference.shape != labels.shape or reference_ap.shape != labels.shape:
        raise ValueError("SOP batch-1 reference vector differs")
    embeddings = official / "test_embeddings.npy"
    gallery = pack_int8_unit_embeddings(
        torch.from_numpy(np.asarray(np.load(embeddings, mmap_mode="r", allow_pickle=False)).copy())
    )
    library = Path(
        "/home/riomus/sfora-rc5-pointer-b7c57022/libsfora_cutile_int8_score_sha39602d0e.so"
    )
    torch.backends.cuda.matmul.allow_tf32 = False
    codes = []
    norms = []
    native_ordinals = []
    native_scores = []
    hits = []
    code_matches = 0
    started = time.perf_counter()
    with Siglip2CompactIndex.from_artifacts(
        model_snapshot=Path(
            "/home/riomus/.cache/huggingface/hub/models--google--siglip2-large-patch16-256/"
            "snapshots/787800c8990e6f058423089178e718139608408c"
        ),
        training_receipt=training_receipt,
        training_checkpoint=training / "checkpoint.pt",
        train_embeddings=training / "train_embeddings.npy",
        native_library=library,
        expected_receipt_sha256="07b4716b42d1291b9c195774ebd48d9df89a3b578ee54efdb94662c5e125d1c5",
        official_gallery_receipt=official_receipt,
        official_gallery_embeddings=embeddings,
        expected_official_gallery_receipt_sha256="7f5edf6b8ecb526e8ba29944119035245725055787cb088bd2635cc734b0eb6c",
    ) as index:
        assert index.encoder is not None and index.gallery is not None
        for position, (row, path) in enumerate(zip(sample, paths, strict=True), start=1):
            raw = path.read_bytes()
            if hashlib.sha256(raw).digest() != manifest[32 * row : 32 * (row + 1)]:
                raise ValueError("SOP batch-1 image bytes differ")
            with Image.open(BytesIO(raw)) as image:
                query = index.encoder.encode_images([image.convert("RGB")])
            code_matches += int(
                torch.equal(query.codes[0], gallery.codes[row])
                and torch.equal(query.inverse_norms[0], gallery.inverse_norms[row])
            )
            ordinals, scores = index.gallery.search_packed(query)
            if (
                ordinals.shape != (1, 10)
                or scores.shape != (1, 10)
                or not np.isfinite(scores).all()
            ):
                raise ValueError("SOP batch-1 native top-10 differs")
            top1 = next((int(value) for value in ordinals[0] if int(value) != row), None)
            if top1 is None:
                raise ValueError("SOP batch-1 nonself top-1 missing")
            hits.append(float(labels[top1] == labels[row]))
            codes.append(query.codes[0])
            norms.append(query.inverse_norms[0])
            native_ordinals.append(ordinals[0])
            native_scores.append(scores[0])
            if position % 256 == 0:
                print(json.dumps({"query_rows": position}), flush=True)
    query_wall = time.perf_counter() - started
    packed = PackedInt8Embeddings(torch.stack(codes), torch.stack(norms))
    g_codes = gallery.codes.float().cuda()
    g_norms = gallery.inverse_norms.float().cuda()
    g_labels = torch.from_numpy(labels.copy()).cuda()
    _classes, inverse_labels, counts = np.unique(labels, return_inverse=True, return_counts=True)
    relevant = counts[inverse_labels] - 1
    rank_width = max(int(relevant.max()), 1_000)
    ranks = torch.arange(1, rank_width + 1, device="cuda")
    oracle_hits = []
    oracle_ap = []
    max_score_delta = 0.0
    for start in range(0, len(sample), 32):
        stop = start + 32
        scores = packed.codes[start:stop].float().cuda() @ g_codes.T
        scores *= packed.inverse_norms[start:stop].float().cuda()[:, None]
        scores *= g_norms[None, :]
        expected = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :10]
        expected_scores = scores.gather(1, expected).cpu().numpy()
        if not np.array_equal(np.asarray(native_ordinals[start:stop]), expected.cpu().numpy()):
            raise ValueError("SOP batch-1 native top-10 ordinals differ")
        max_score_delta = max(
            max_score_delta,
            float(np.max(np.abs(np.asarray(native_scores[start:stop]) - expected_scores))),
        )
        if args.full:
            rows = torch.from_numpy(sample[start:stop].copy()).cuda()
            scores[torch.arange(len(rows), device="cuda"), rows] = -torch.inf
            ranked = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :rank_width]
            matches = g_labels[ranked].eq(g_labels[rows, None])
            precision = torch.cumsum(matches, dim=1) / ranks[None, :]
            relevant_rows = torch.from_numpy(relevant[sample[start:stop]].copy()).cuda()
            valid = ranks[None, :] <= relevant_rows[:, None]
            ap = (precision * matches * valid).sum(dim=1) / relevant_rows
            oracle_hits.extend(float(x) for x in matches[:, 0].cpu().tolist())
            oracle_ap.extend(float(x) for x in ap.cpu().tolist())
    if not np.isfinite(max_score_delta) or max_score_delta > 1e-5:
        raise ValueError("SOP batch-1 native top-10 scores differ")
    interval = product_bootstrap(np.asarray(hits) - reference[sample], labels[sample])
    passed = interval["point"] >= -0.005 and interval["lower_95"] > -0.005
    result = {
        "schema": (
            "sfora-sop-true-freeze-official-batch1-full-v1"
            if args.full
            else "sfora-sop-true-freeze-official-batch1-screen-v1"
        ),
        "status": "exploratory previously observed TEST sample",
        "seed": 179024,
        "sample_rows": len(sample),
        "sample_ordinals_sha256": hashlib.sha256(sample.astype("<i8").tobytes()).hexdigest(),
        "sample_products": len(np.unique(labels[sample])),
        "batch1_r1": float(np.mean(hits)),
        "batch32_reference_r1": float(reference[sample].mean()),
        "paired_product_bootstrap": interval,
        "packed_code_norm_matches": code_matches,
        "native_top10_exact": True,
        "native_max_score_abs_delta": max_score_delta,
        "query_wall_seconds": query_wall,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_parent_host_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "source_sha256": digest(Path(__file__)),
        "imported_source_sha256": imported,
        "bootstrap_helper_sha256": digest(Path(inspect.getfile(product_bootstrap))),
        "training_receipt_sha256": digest(training_receipt),
        "official_receipt_sha256": digest(official_receipt),
        "gallery_embeddings_sha256": digest(embeddings),
        "screen_pass": bool(passed),
    }
    if args.full:
        result.pop("screen_pass")
        if oracle_hits != hits:
            raise ValueError("SOP batch-1 native R@1 differs from packed oracle")
        map_interval = product_bootstrap(np.asarray(oracle_ap) - reference_ap, labels)
        passed = passed and map_interval["point"] >= -0.005
        result.update(
            {
                "batch1_map_at_r": float(np.mean(oracle_ap)),
                "batch32_reference_map_at_r": float(reference_ap.mean()),
                "paired_map_product_bootstrap": map_interval,
                "per_query_r1": oracle_hits,
                "per_query_ap": oracle_ap,
                "native_per_query_r1_exact": True,
                "quality_pass": bool(passed),
            }
        )
    result["total_wall_seconds"] = time.perf_counter() - total_started
    with output.open("xb") as stream:
        stream.write((json.dumps(result, sort_keys=True, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"screen_pass": passed, "batch1_r1": result["batch1_r1"], "delta": interval}))


if __name__ == "__main__":
    main()
