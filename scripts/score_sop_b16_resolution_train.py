#!/usr/bin/env python3
"""Fit-only PCA and paired SOP TRAIN quality screen for B/16 resolution."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
import time
from pathlib import Path

import numpy as np
import torch
from analyze_sop_reference_progress import product_bootstrap
from torch.nn import functional as F

from sfora.joint_relational_compaction import pack_int8_unit_embeddings
from sfora.representation_ceiling import deterministic_class_partition, fit_centered_pca
from sfora.sop_evaluation import score_gallery_r1, score_symmetric

SOURCE_ARCHIVE_SHA256 = "16b4554d3868363905f1e1cd385783a8033513835723a7b89f4b762d893d757f"
SEED = 179019
BOOTSTRAP_DRAWS = 5000
ARMS = ("b16_224", "b16_336_detail", "b16_336_upsampled")


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def compare(
    candidate: dict[str, object], baseline: dict[str, object], labels: np.ndarray
) -> dict[str, object]:
    result = {}
    for key, row_key in (("recall_at_1", "per_query_r1"), ("map_at_r", "per_query_ap")):
        if row_key not in candidate or row_key not in baseline:
            continue
        delta = np.asarray(candidate[row_key], dtype=np.float64) - np.asarray(
            baseline[row_key], dtype=np.float64
        )
        if (
            len(delta) != len(labels)
            or abs(float(delta.mean()) - (candidate[key] - baseline[key])) > 1e-6
        ):
            raise ValueError("SOP resolution paired metric differs")
        result[key] = product_bootstrap(delta, labels, seed=SEED, replicates=BOOTSTRAP_DRAWS)
    return result


def score_arm(
    features: np.ndarray,
    labels: torch.Tensor,
    fit_rows: np.ndarray,
    held_rows: np.ndarray,
) -> dict[str, object]:
    if (
        features.shape != (59_551, 768)
        or features.dtype != np.float32
        or not np.isfinite(features).all()
    ):
        raise ValueError("SOP B/16 resolution feature geometry differs")
    device = torch.device("cuda")
    source = torch.from_numpy(np.asarray(features).copy())
    fitted = F.normalize(source[fit_rows].contiguous(), dim=1)
    projection_started = time.perf_counter()
    pca = fit_centered_pca(fitted, dimensions=128)
    pca_seconds = time.perf_counter() - projection_started
    all_normalized = F.normalize(source, dim=1)
    projected = pca.apply(all_normalized)
    packed = pack_int8_unit_embeddings(projected)
    if packed.codes.shape != (59_551, 128) or packed.inverse_norms.shape != (59_551,):
        raise ValueError("SOP B/16 resolution packed geometry differs")
    full_labels = labels.to(device)
    held_index = torch.from_numpy(held_rows).to(device)
    held_labels = full_labels[held_index]
    source_gpu = source.to(device)
    codes_gpu = packed.codes.float().to(device)
    inverse_gpu = packed.inverse_norms.to(device)
    started = time.perf_counter()
    float_holdout = score_symmetric(source_gpu[held_index], held_labels)
    float_full = score_gallery_r1(source_gpu, full_labels, held_index)
    packed_holdout = score_symmetric(
        codes_gpu[held_index], held_labels, inverse_norms=inverse_gpu[held_index]
    )
    packed_full = score_gallery_r1(codes_gpu, full_labels, held_index, inverse_norms=inverse_gpu)
    score_seconds = time.perf_counter() - started
    return {
        "float": {"holdout_only": float_holdout, "full_train_gallery": float_full},
        "packed_pca128": {"holdout_only": packed_holdout, "full_train_gallery": packed_full},
        "pca_fit_seconds": pca_seconds,
        "score_seconds": score_seconds,
        "gallery_bytes_per_row": {"float": 768 * 4, "packed_pca128": 130},
        "pca_fit_sha256": hashlib.sha256(
            pca.mean.numpy().tobytes() + pca.components.numpy().tobytes()
        ).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("source-archive", "extract-receipt", "feature-dir", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.output.exists()
        or args.output.is_symlink()
        or not torch.cuda.is_available()
        or sha256(args.source_archive) != SOURCE_ARCHIVE_SHA256
    ):
        raise ValueError("SOP B/16 resolution quality invocation differs")
    started = time.perf_counter()
    torch.set_num_threads(16)
    torch.backends.cuda.matmul.allow_tf32 = False
    extraction = json.loads(args.extract_receipt.read_text())
    if (
        extraction.get("schema") != "sfora-sop-b16-resolution-train-extract-v1"
        or extraction.get("source_archive_sha256") != SOURCE_ARCHIVE_SHA256
    ):
        raise ValueError("SOP B/16 resolution extraction receipt differs")
    with np.load(args.source_archive, allow_pickle=False) as archive:
        labels = np.asarray(archive["train_labels"], dtype=np.int64)
        base_features = np.asarray(archive["train_embeddings"], dtype=np.float32)
    if labels.shape != (59_551,) or base_features.shape != (59_551, 768):
        raise ValueError("SOP B/16 TRAIN archive differs")
    partition = deterministic_class_partition(tuple(map(int, labels)), fit_fraction=0.9, seed=SEED)
    fit_rows = np.asarray(partition.fit_row_indexes, dtype=np.int64)
    held_rows = np.asarray(partition.validation_row_indexes, dtype=np.int64)
    if (len(fit_rows), len(held_rows), len(np.unique(labels[held_rows]))) != (53_700, 5_851, 1_132):
        raise ValueError("SOP B/16 resolution partition differs")
    train_labels = torch.from_numpy(labels)
    results = {}
    feature_sha = {"b16_224": SOURCE_ARCHIVE_SHA256}
    for arm in ARMS:
        if arm == "b16_224":
            features = base_features
        else:
            path = args.feature_dir / f"{arm}.npy"
            expected = extraction["features"][arm]["sha256"]
            if sha256(path) != expected:
                raise ValueError(f"SOP B/16 {arm} feature digest differs")
            feature_sha[arm] = expected
            features = np.load(path, mmap_mode="r", allow_pickle=False)
        arm_started = time.perf_counter()
        results[arm] = score_arm(features, train_labels, fit_rows, held_rows)
        print(
            json.dumps(
                {
                    "arm": arm,
                    "elapsed_seconds": time.perf_counter() - arm_started,
                    "packed_full_gallery_r1": results[arm]["packed_pca128"]["full_train_gallery"][
                        "recall_at_1"
                    ],
                }
            ),
            flush=True,
        )
        torch.cuda.empty_cache()
    held_labels = labels[held_rows]
    paired = {}
    for challenger, control in (
        ("b16_336_detail", "b16_224"),
        ("b16_336_detail", "b16_336_upsampled"),
    ):
        comparison = {}
        for representation in ("float", "packed_pca128"):
            comparison[representation] = {
                gallery: compare(
                    results[challenger][representation][gallery],
                    results[control][representation][gallery],
                    held_labels,
                )
                for gallery in ("holdout_only", "full_train_gallery")
            }
        paired[f"{challenger}_minus_{control}"] = comparison
    receipt = {
        "schema": "sfora-sop-b16-resolution-train-quality-v1",
        "claim_eligible": False,
        "split": "SOP TRAIN product-disjoint fit/holdout; no official TEST rows read",
        "fit_images": len(fit_rows),
        "heldout_images": len(held_rows),
        "heldout_products": len(np.unique(held_labels)),
        "full_train_gallery_images": len(labels),
        "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
        "extract_receipt_sha256": sha256(args.extract_receipt),
        "feature_sha256": feature_sha,
        "script_sha256": sha256(Path(__file__)),
        "evaluator_sha256": sha256(Path(score_gallery_r1.__code__.co_filename)),
        "seed": SEED,
        "results": results,
        "paired_product_bootstrap": paired,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_host_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "hardware": {
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "python": platform.python_version(),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(receipt, stream, sort_keys=True, indent=2)
        stream.write("\n")
    print(
        json.dumps({"result": "completed", "elapsed_seconds": receipt["elapsed_seconds"]}),
        flush=True,
    )


if __name__ == "__main__":
    main()
