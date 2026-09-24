#!/usr/bin/env python3
"""Census gallery-size failures in authenticated SOP TRAIN full-width features."""

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
from evaluate_sop_reference_checkpoint import EXPECTED_SOP_RECORD_SHA256
from export_unicom_sop_embeddings import ordered_record_sha256, parse_sop_records
from train_sop_compact_backbone import FIT_FRACTION, SPLIT_SEED, sha256

from sfora.representation_ceiling import deterministic_class_partition

ARCHIVE_SHA256 = "e5f83f81e3c7bae29b5cbf1ecc08fc0d5c244d2c52e49302307c90b19dbda744"
THRESHOLDS = (0.90, 0.95, 0.98)


@torch.inference_mode()
def nearest_full_and_holdout(
    features: torch.Tensor,
    query_rows: np.ndarray,
    fit_rows: np.ndarray,
    *,
    block_rows: int = 128,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """First-ordinal cosine nearest neighbors, excluding the query itself."""

    if (
        features.ndim != 2
        or len(features) < 2
        or features.shape[1] < 1
        or query_rows.ndim != 1
        or fit_rows.ndim != 1
        or len(query_rows) < 1
        or len(fit_rows) < 1
        or len(np.unique(query_rows)) != len(query_rows)
        or np.intersect1d(query_rows, fit_rows).size
        or not np.array_equal(
            np.sort(np.concatenate((query_rows, fit_rows))), np.arange(len(features))
        )
        or block_rows < 1
    ):
        raise ValueError("SOP full-gallery census partition differs")
    device = features.device
    source = torch.nn.functional.normalize(features.float(), dim=1)
    source_t = source.T.contiguous()
    fit_index = torch.as_tensor(fit_rows, device=device, dtype=torch.long)
    full_index = np.empty(len(query_rows), dtype=np.int64)
    full_score = np.empty(len(query_rows), dtype=np.float32)
    holdout_index = np.empty(len(query_rows), dtype=np.int64)
    holdout_score = np.empty(len(query_rows), dtype=np.float32)
    for start in range(0, len(query_rows), block_rows):
        stop = min(start + block_rows, len(query_rows))
        rows = torch.as_tensor(query_rows[start:stop], device=device, dtype=torch.long)
        scores = source[rows] @ source_t
        scores[torch.arange(len(rows), device=device), rows] = -float("inf")
        values, indexes = scores.max(dim=1)
        full_index[start:stop] = indexes.cpu().numpy()
        full_score[start:stop] = values.cpu().numpy()
        scores[:, fit_index] = -float("inf")
        values, indexes = scores.max(dim=1)
        holdout_index[start:stop] = indexes.cpu().numpy()
        holdout_score[start:stop] = values.cpu().numpy()
    return full_index, full_score, holdout_index, holdout_score


def image_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--sop-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--execute-sop-train-fullgallery-census", action="store_true", required=True
    )
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink() or sha256(args.archive) != ARCHIVE_SHA256:
        raise ValueError("SOP full-gallery census input differs")
    started = time.perf_counter()
    torch.backends.cuda.matmul.allow_tf32 = False
    records = parse_sop_records(args.sop_root)
    if ordered_record_sha256(records) != EXPECTED_SOP_RECORD_SHA256:
        raise ValueError("SOP ordered metadata differs")
    train = tuple(row for row in records if row.split == "train")
    with np.load(args.archive, allow_pickle=False) as archive:
        features = np.ascontiguousarray(archive["train_embeddings"], dtype=np.float32)
        labels = np.asarray(archive["train_labels"], dtype=np.int64)
        image_ids = np.asarray(archive["train_image_ids"], dtype=np.int64)
        metadata = json.loads(str(archive["metadata_json"].item()))
    if (
        features.shape != (59_551, 768)
        or labels.shape != (59_551,)
        or len(train) != len(labels)
        or not np.array_equal(image_ids, [row.image_id for row in train])
        or not np.array_equal(labels, [row.label for row in train])
        or metadata.get("step") != 48_000
        or metadata.get("width") != 768
        or metadata.get("trained_checkpoint_sha256")
        != "232f7cee93e39fa242d8461f8dc8cee684d7228eef01f8fe9399b9782e80a1b2"
        or not np.isfinite(features).all()
    ):
        raise ValueError("SOP full-gallery trained feature authority differs")
    partition = deterministic_class_partition(
        tuple(map(int, labels)), fit_fraction=FIT_FRACTION, seed=SPLIT_SEED
    )
    fit = np.asarray(partition.fit_row_indexes, dtype=np.int64)
    query = np.asarray(partition.validation_row_indexes, dtype=np.int64)
    if len(fit) != 53_700 or len(query) != 5_851:
        raise ValueError("SOP full-gallery split differs")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(16)
    vectors = torch.from_numpy(features).to(device)
    full_idx, full_sim, hold_idx, hold_sim = nearest_full_and_holdout(vectors, query, fit)
    if not np.isfinite(full_sim).all() or not np.isfinite(hold_sim).all():
        raise ValueError("SOP full-gallery similarity differs")
    full_correct = labels[full_idx] == labels[query]
    hold_correct = labels[hold_idx] == labels[query]
    wrong = np.flatnonzero(~full_correct)
    new_failures = np.flatnonzero(hold_correct & ~full_correct)
    hash_cache: dict[int, str] = {}

    def hash_row(row: int) -> str:
        index = int(row)
        if index not in hash_cache:
            hash_cache[index] = image_sha256(train[index].image_path)
        return hash_cache[index]

    exact_duplicate = np.asarray(
        [hash_row(query[row]) == hash_row(full_idx[row]) for row in wrong], dtype=bool
    )
    result = {
        "schema": "sfora-sop-trained-fullgallery-error-census-v1",
        "claim_eligible": False,
        "dataset": "Stanford Online Products official train split only",
        "query_count": len(query),
        "full_gallery_count": len(labels) - 1,
        "holdout_gallery_count": len(query) - 1,
        "fit_identities": len(np.unique(labels[fit])),
        "holdout_identities": len(np.unique(labels[query])),
        "holdout_only_recall_at_1": float(hold_correct.mean()),
        "full_train_gallery_recall_at_1": float(full_correct.mean()),
        "full_gallery_wrong_queries": len(wrong),
        "additional_failures_from_fit_gallery": len(new_failures),
        "wrong_top1_cosine_threshold_counts": {
            str(value): int(np.count_nonzero(full_sim[wrong] >= value)) for value in THRESHOLDS
        },
        "wrong_top1_exact_image_byte_duplicates": int(exact_duplicate.sum()),
        "additional_failure_cosine_threshold_counts": {
            str(value): int(np.count_nonzero(full_sim[new_failures] >= value))
            for value in THRESHOLDS
        },
        "wrong_top1_cosine_quantiles": {
            str(value): float(np.quantile(full_sim[wrong], value))
            for value in (0.0, 0.5, 0.9, 0.99, 1.0)
        },
        "query_row_indexes": query.tolist(),
        "full_top1_row_indexes": full_idx.tolist(),
        "full_top1_cosines": full_sim.astype(float).tolist(),
        "holdout_top1_row_indexes": hold_idx.tolist(),
        "holdout_top1_cosines": hold_sim.astype(float).tolist(),
        "inputs": {
            "archive_sha256": ARCHIVE_SHA256,
            "train_metadata_sha256": sha256(args.sop_root / "Ebay_train.txt"),
            "script_sha256": sha256(Path(__file__)),
        },
        "hardware": {
            "device": device,
            "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
            "torch": torch.__version__,
            "python": platform.python_version(),
            "peak_cuda_allocated_bytes": (
                torch.cuda.max_memory_allocated() if device == "cuda" else 0
            ),
            "peak_host_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, sort_keys=True, allow_nan=False, separators=(",", ":"))
        stream.write("\n")
    print(
        json.dumps(
            {
                "holdout_recall": result["holdout_only_recall_at_1"],
                "full_recall": result["full_train_gallery_recall_at_1"],
                "new_failures": len(new_failures),
                "wrong_cosine_ge_095": result["wrong_top1_cosine_threshold_counts"]["0.95"],
                "wrong_exact_duplicates": int(exact_duplicate.sum()),
                "output": str(args.output),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
