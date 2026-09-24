#!/usr/bin/env python3
"""Audit single-image UNICOM encoding against the frozen SOP TRAIN quality export."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import platform
import resource
import time
from pathlib import Path, PurePosixPath

import numpy as np
import torch
from benchmark_sop_siglip2_vs_unicom import ARCHIVE_SHA256, NATIVE_SHA256, encode_live
from probe_l14_parallel_fold import load_authenticated_l14
from score_sop_pretrained_substrate import fit_project_pack, sha256
from torch.nn import functional as F
from verify_sop_siglip2_live_batch1 import first_nonself

from sfora.cutile_int8 import CutilePackedInt8Gallery
from sfora.joint_relational_compaction import PackedInt8Embeddings, pack_int8_unit_embeddings
from sfora.representation_ceiling import deterministic_class_partition

TILEIRAS_SHA256 = "df2e9ef3804cab682f605a5c9e50045a24404ba22c3be0903454e1a60fcd78ae"
CHECKPOINT_SHA256 = "3916ab5aed3b522fc90345be8b4457fe5dad60801ad2af5a6871c0c096e8d7ea"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in (
        "unicom-checkout",
        "l14-checkpoint",
        "unicom-l14-archive",
        "score-dir",
        "dataset-root",
        "native-library",
        "output",
    ):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    tileiras = os.environ.get("CUTILE_TILEIRAS_PATH")
    if (
        args.output.exists()
        or args.output.is_symlink()
        or sha256(args.unicom_l14_archive) != ARCHIVE_SHA256
        or sha256(args.native_library) != NATIVE_SHA256
        or sha256(args.l14_checkpoint) != CHECKPOINT_SHA256
        or not tileiras
        or sha256(Path(tileiras)) != TILEIRAS_SHA256
    ):
        raise ValueError("UNICOM live batch-one source authority differs")
    score = json.loads((args.score_dir / "receipt.json").read_text())
    arm_score = json.loads((args.score_dir / "unicom_l14_336.json").read_text())
    if (
        score.get("schema") != "sfora-sop-pretrained-substrate-screen-v1"
        or score.get("source_archive_sha256") != ARCHIVE_SHA256
    ):
        raise ValueError("UNICOM live batch-one quality authority differs")
    with np.load(args.unicom_l14_archive, allow_pickle=False) as archive:
        labels = np.asarray(archive["train_labels"], dtype=np.int64)
        ids = np.asarray(archive["train_image_ids"], dtype=np.int64)
        relatives = np.asarray(archive["train_relative_paths"]).astype(str)
        source = np.asarray(archive["train_embeddings"], dtype=np.float32)
    if labels.shape != (59_551,) or source.shape != (59_551, 768):
        raise ValueError("UNICOM live batch-one TRAIN inventory differs")
    partition = deterministic_class_partition(
        tuple(map(int, labels)), fit_fraction=0.9, seed=179019
    )
    fit_rows = np.asarray(partition.fit_row_indexes, dtype=np.int64)
    held_rows = np.asarray(partition.validation_row_indexes, dtype=np.int64)
    if len(fit_rows) != 53_700 or len(held_rows) != 5_851:
        raise ValueError("UNICOM live batch-one partition differs")
    fitted = fit_project_pack(source, fit_rows)
    pca_hash = hashlib.sha256(
        fitted.pca.mean.numpy().tobytes() + fitted.pca.components.numpy().tobytes()
    ).hexdigest()
    if pca_hash != arm_score["pca_sha256"]:
        raise ValueError("UNICOM live batch-one PCA differs")
    expected_r1 = np.asarray(
        arm_score["score"]["packed_pca128"]["full_train_gallery"]["per_query_r1"],
        dtype=np.bool_,
    )
    if expected_r1.shape != (5_851,):
        raise ValueError("UNICOM live batch-one quality geometry differs")
    paths = []
    for row in held_rows:
        relative = PurePosixPath(str(relatives[row]))
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ValueError("UNICOM live batch-one image path differs")
        path = args.dataset_root.joinpath(*relative.parts)
        if not path.is_file() or path.is_symlink():
            raise ValueError("UNICOM live batch-one image missing")
        paths.append(path)
    torch.set_num_threads(16)
    torch.backends.cuda.matmul.allow_tf32 = False
    model, transform = load_authenticated_l14(args.unicom_checkout, args.l14_checkpoint)
    model = model.cuda().eval()
    arm = {
        "name": "unicom_l14_336",
        "model": model,
        "transform": transform,
        "pca": fitted.pca,
        "width": 768,
    }
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    observed_r1 = []
    top1_ordinals = []
    code_changed = []
    norm_changed = []
    top10_changed = []
    feature_cosines = []
    max_feature_abs = 0.0
    max_score_abs = 0.0
    with CutilePackedInt8Gallery.open_packed(args.native_library, fitted.packed) as gallery:
        for index, (row, path) in enumerate(zip(held_rows, paths, strict=True), start=1):
            feature, _host, _transfer, _encode = encode_live(arm, [path])
            cached_feature = torch.from_numpy(source[row : row + 1].copy())
            cosine = float(F.cosine_similarity(feature, cached_feature, dim=1)[0])
            if not np.isfinite(cosine) or cosine < 0.999:
                raise ValueError(f"UNICOM live batch-one feature drift at query {index}: {cosine}")
            feature_cosines.append(cosine)
            max_feature_abs = max(max_feature_abs, float((feature - cached_feature).abs().max()))
            query = pack_int8_unit_embeddings(fitted.pca.apply(F.normalize(feature, dim=1)))
            cached_query = PackedInt8Embeddings(
                fitted.packed.codes[row : row + 1].contiguous(),
                fitted.packed.inverse_norms[row : row + 1].contiguous(),
            )
            code_changed.append(bool((query.codes != cached_query.codes).any()))
            norm_changed.append(bool((query.inverse_norms != cached_query.inverse_norms).any()))
            ordinals, scores = gallery.search_packed(query)
            cached_ordinals, cached_scores = gallery.search_packed(cached_query)
            cached_top1 = first_nonself(cached_ordinals, int(row))
            if bool(labels[cached_top1] == labels[row]) != bool(expected_r1[index - 1]):
                raise ValueError(f"UNICOM live batch-one cached quality differs at query {index}")
            top10_changed.append(not np.array_equal(ordinals, cached_ordinals))
            max_score_abs = max(max_score_abs, float(np.max(np.abs(scores - cached_scores))))
            top1 = first_nonself(ordinals, int(row))
            top1_ordinals.append(top1)
            observed_r1.append(bool(labels[top1] == labels[row]))
            if index % 500 == 0:
                print(
                    json.dumps(
                        {"queries": index, "elapsed_seconds": time.perf_counter() - started}
                    ),
                    flush=True,
                )
    observed: np.ndarray = np.asarray(observed_r1, dtype=np.bool_)
    receipt = {
        "schema": "sfora-sop-unicom-live-batch1-quality-v1",
        "claim_eligible": False,
        "split": "SOP official TRAIN product-disjoint holdout, full TRAIN gallery",
        "fit_images": len(fit_rows),
        "holdout_queries": len(held_rows),
        "gallery_images": len(labels),
        "seed": 179019,
        "source_sha256": sha256(Path(__file__)),
        "encode_live_source_sha256": sha256(Path(inspect.getfile(encode_live))),
        "fit_project_pack_source_sha256": sha256(Path(inspect.getfile(fit_project_pack))),
        "archive_sha256": ARCHIVE_SHA256,
        "quality_receipt_sha256": sha256(args.score_dir / "receipt.json"),
        "l14_checkpoint_sha256": CHECKPOINT_SHA256,
        "native_library_sha256": NATIVE_SHA256,
        "tileiras_sha256": TILEIRAS_SHA256,
        "pca_sha256": pca_hash,
        "query_image_ids_sha256": hashlib.sha256(ids[held_rows].tobytes()).hexdigest(),
        "cached_export_r1": float(expected_r1.mean()),
        "live_batch1_r1": float(observed.mean()),
        "live_minus_cached_r1_percentage_points": float(
            (observed.mean() - expected_r1.mean()) * 100
        ),
        "per_query_live_r1": observed.astype(np.uint8).tolist(),
        "per_query_cached_r1": expected_r1.astype(np.uint8).tolist(),
        "top1_ordinals_sha256": hashlib.sha256(
            np.asarray(top1_ordinals, dtype=np.int64).tobytes()
        ).hexdigest(),
        "code_changed_queries": int(sum(code_changed)),
        "inverse_norm_changed_queries": int(sum(norm_changed)),
        "top10_changed_queries": int(sum(top10_changed)),
        "r1_changed_queries": int(np.count_nonzero(observed != expected_r1)),
        "minimum_feature_cosine": min(feature_cosines),
        "maximum_feature_absolute_delta": max_feature_abs,
        "maximum_top10_score_absolute_delta": max_score_abs,
        "wall_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_parent_host_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "hardware": {
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "python": platform.python_version(),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write((json.dumps(receipt, sort_keys=True, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    print(
        json.dumps(
            {
                "live_batch1_r1": receipt["live_batch1_r1"],
                "cached_export_r1": receipt["cached_export_r1"],
                "r1_changed_queries": receipt["r1_changed_queries"],
                "top10_changed_queries": receipt["top10_changed_queries"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
