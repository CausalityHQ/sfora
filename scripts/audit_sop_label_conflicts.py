#!/usr/bin/env python3
"""Census nearest cross-label neighbors in pretrained SOP fit features only."""

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

from sfora.representation_ceiling import deterministic_class_partition

ARCHIVE_SHA256 = "16b4554d3868363905f1e1cd385783a8033513835723a7b89f4b762d893d757f"
SPLIT_SEED = 179019
THRESHOLDS = (0.90, 0.95, 0.98)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def nearest_cross_label_cosines(
    features: np.ndarray, labels: np.ndarray, *, block_rows: int = 128
) -> tuple[np.ndarray, np.ndarray]:
    """Return the maximum normalized cosine and first matching row for each query."""

    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels)
    if (
        features.ndim != 2
        or labels.shape != (len(features),)
        or len(features) < 2
        or features.shape[1] < 1
        or not np.issubdtype(labels.dtype, np.integer)
        or len(np.unique(labels)) < 2
        or not np.isfinite(features).all()
        or np.any(np.linalg.norm(features, axis=1) <= 0)
        or type(block_rows) is not int
        or block_rows < 1
    ):
        raise ValueError("SOP conflict feature inventory differs")
    gallery = torch.nn.functional.normalize(torch.from_numpy(np.ascontiguousarray(features)), dim=1)
    label_tensor = torch.from_numpy(np.ascontiguousarray(labels.astype(np.int64)))
    transposed = gallery.T.contiguous()
    scores = np.empty(len(features), dtype=np.float32)
    neighbors = np.empty(len(features), dtype=np.int64)
    for start in range(0, len(features), block_rows):
        stop = min(start + block_rows, len(features))
        block_scores = gallery[start:stop] @ transposed
        same_product = label_tensor[start:stop, None] == label_tensor[None, :]
        best, indexes = block_scores.masked_fill(same_product, -float("inf")).max(dim=1)
        scores[start:stop] = best.numpy()
        neighbors[start:stop] = indexes.numpy()
    return scores, neighbors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--features-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--execute-sop-fit-conflict-census", action="store_true", required=True)
    args = parser.parse_args()
    if args.output.exists() or args.threads < 1 or sha256(args.features_archive) != ARCHIVE_SHA256:
        raise ValueError("SOP conflict census input differs")
    started = time.perf_counter()
    with np.load(args.features_archive, allow_pickle=False) as archive:
        features = np.ascontiguousarray(archive["train_embeddings"], dtype=np.float32)
        labels = np.asarray(archive["train_labels"], dtype=np.int64)
        image_ids = np.asarray(archive["train_image_ids"], dtype=np.int64)
        relative_paths = np.asarray(archive["train_relative_paths"])
    if (
        features.shape != (59_551, 768)
        or labels.shape != (59_551,)
        or image_ids.shape != (59_551,)
        or relative_paths.shape != (59_551,)
        or len(set(map(int, image_ids))) != 59_551
    ):
        raise ValueError("SOP conflict census train inventory differs")
    partition = deterministic_class_partition(
        tuple(map(int, labels)), fit_fraction=0.9, seed=SPLIT_SEED
    )
    fit = np.asarray(partition.fit_row_indexes, dtype=np.int64)
    if len(fit) != 53_700 or len(np.unique(labels[fit])) != 10_186:
        raise ValueError("SOP conflict census fit inventory differs")
    torch.set_num_threads(args.threads)
    scores, neighbors = nearest_cross_label_cosines(features[fit], labels[fit])
    if not np.all(np.isfinite(scores)) or not np.all(labels[fit] != labels[fit[neighbors]]):
        raise ValueError("SOP conflict census ranking differs")
    ordered = np.argsort(-scores, kind="stable")[:20]
    result = {
        "schema": "sfora-sop-pretrained-fit-cross-label-census-v1",
        "claim_eligible": False,
        "dataset": "Stanford Online Products",
        "split": (
            "official train fit identities only; 53700 queries against 53700 fit gallery images"
        ),
        "source": "authenticated pretrained UNICOM ViT-B/16@224, normalized full-width 768-D",
        "seed": SPLIT_SEED,
        "query_count": len(fit),
        "product_count": len(np.unique(labels[fit])),
        "metric": "maximum cosine to a different labeled product, after row L2 normalization",
        "thresholds_chosen_after_2048_query_screen": True,
        "threshold_counts": {
            str(threshold): int(np.count_nonzero(scores >= threshold)) for threshold in THRESHOLDS
        },
        "score_quantiles": {
            str(quantile): float(np.quantile(scores, quantile))
            for quantile in (0.5, 0.9, 0.99, 0.999, 1.0)
        },
        "per_fit_query_cross_label_cosine": scores.astype(float).tolist(),
        "per_fit_query_cross_label_neighbor_index": neighbors.tolist(),
        "top_pairs": [
            {
                "cosine": float(scores[index]),
                "query_image_id": int(image_ids[fit[index]]),
                "neighbor_image_id": int(image_ids[fit[neighbors[index]]]),
                "query_product": int(labels[fit[index]]),
                "neighbor_product": int(labels[fit[neighbors[index]]]),
                "query_relative_path": str(relative_paths[fit[index]]),
                "neighbor_relative_path": str(relative_paths[fit[neighbors[index]]]),
            }
            for index in ordered
        ],
        "inputs": {
            "features_archive_sha256": ARCHIVE_SHA256,
            "fit_row_indexes_sha256": hashlib.sha256(fit.astype("<i4").tobytes()).hexdigest(),
            "script_sha256": sha256(Path(__file__)),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "hardware": {
            "machine": platform.machine(),
            "torch": torch.__version__,
            "threads": args.threads,
            "peak_host_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, sort_keys=True, separators=(",", ":"), allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "threshold_counts": result["threshold_counts"],
                "score_quantiles": result["score_quantiles"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
