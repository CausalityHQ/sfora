#!/usr/bin/env python3
"""Separate SOP gallery size from seen-class distractor effects on TRAIN rows."""

from __future__ import annotations

import argparse
import json
import platform
import resource
import time
from pathlib import Path

import numpy as np
import torch
from evaluate_sop_cub_transfer import gpu_compute_pids
from evaluate_sop_reference_checkpoint import EXPECTED_SOP_RECORD_SHA256
from export_unicom_sop_embeddings import ordered_record_sha256, parse_sop_records
from scipy.special import gammaln
from train_sop_compact_backbone import FIT_FRACTION, SPLIT_SEED, sha256

from sfora.representation_ceiling import deterministic_class_partition

PRETRAINED_ARCHIVE_SHA256 = "16b4554d3868363905f1e1cd385783a8033513835723a7b89f4b762d893d757f"
TRAINED_ARCHIVE_SHA256 = "e5f83f81e3c7bae29b5cbf1ecc08fc0d5c244d2c52e49302307c90b19dbda744"
NEGATIVE_COUNTS = (1_000, 3_000, 5_000)


def expected_recall_random_negatives(
    pool_sizes: np.ndarray, outranking_counts: np.ndarray, sample_count: int
) -> np.ndarray:
    """Exact P(no outranking negative in a uniform without-replacement sample)."""

    pool = np.asarray(pool_sizes, dtype=np.int64)
    bad = np.asarray(outranking_counts, dtype=np.int64)
    if (
        pool.ndim != 1
        or bad.shape != pool.shape
        or len(pool) < 1
        or sample_count < 1
        or np.any(pool < sample_count)
        or np.any(bad < 0)
        or np.any(bad > pool)
    ):
        raise ValueError("SOP equal-size negative pool differs")
    good = pool - bad
    answer = np.zeros(len(pool), dtype=np.float64)
    valid = good >= sample_count
    p = pool[valid].astype(np.float64)
    g = good[valid].astype(np.float64)
    n = float(sample_count)
    answer[valid] = np.exp(
        gammaln(g + 1) - gammaln(g - n + 1) - gammaln(p + 1) + gammaln(p - n + 1)
    )
    return answer


@torch.inference_mode()
def count_outranking_negatives(
    features: torch.Tensor,
    labels: np.ndarray,
    query_rows: np.ndarray,
    fit_rows: np.ndarray,
    *,
    block_rows: int = 128,
) -> dict[str, np.ndarray]:
    """Count seen and unseen negatives that beat each query's best positive."""

    labels = np.asarray(labels, dtype=np.int64)
    query_rows = np.asarray(query_rows, dtype=np.int64)
    fit_rows = np.asarray(fit_rows, dtype=np.int64)
    if (
        features.ndim != 2
        or labels.shape != (len(features),)
        or query_rows.ndim != 1
        or fit_rows.ndim != 1
        or len(query_rows) < 1
        or len(fit_rows) < 1
        or block_rows < 1
        or not np.array_equal(
            np.sort(np.concatenate((query_rows, fit_rows))), np.arange(len(features))
        )
        or np.intersect1d(labels[query_rows], labels[fit_rows]).size
    ):
        raise ValueError("SOP seen/unseen partition differs")
    device = features.device
    vectors = torch.nn.functional.normalize(features.float(), dim=1)
    gallery_t = vectors.T.contiguous()
    all_labels = torch.as_tensor(labels, device=device)
    query_index = torch.as_tensor(query_rows, device=device)
    fit_index = torch.as_tensor(fit_rows, device=device)
    seen_bad = np.empty(len(query_rows), dtype=np.int64)
    unseen_bad = np.empty(len(query_rows), dtype=np.int64)
    seen_pool = np.full(len(query_rows), len(fit_rows), dtype=np.int64)
    unseen_pool = np.empty(len(query_rows), dtype=np.int64)
    best_positive = np.empty(len(query_rows), dtype=np.float32)
    for start in range(0, len(query_rows), block_rows):
        stop = min(start + block_rows, len(query_rows))
        rows = query_index[start:stop]
        row_labels = all_labels[rows]
        scores = vectors[rows] @ gallery_t
        scores[torch.arange(len(rows), device=device), rows] = -float("inf")
        unseen_scores = scores[:, query_index]
        positive_mask = row_labels[:, None] == all_labels[query_index][None, :]
        positive = unseen_scores.masked_fill(~positive_mask, -float("inf"))
        best = positive.max(dim=1).values
        if not bool(torch.isfinite(best).all()):
            raise ValueError("SOP query lacks a positive gallery image")
        positive_ordinal = torch.where(
            positive == best[:, None], query_index[None, :], len(features)
        ).min(dim=1).values
        unseen_wrong = ~positive_mask
        unseen_outranks = (unseen_scores > best[:, None]) | (
            (unseen_scores == best[:, None])
            & (query_index[None, :] < positive_ordinal[:, None])
        )
        seen_scores = scores[:, fit_index]
        seen_outranks = (seen_scores > best[:, None]) | (
            (seen_scores == best[:, None])
            & (fit_index[None, :] < positive_ordinal[:, None])
        )
        unseen_bad[start:stop] = (unseen_wrong & unseen_outranks).sum(dim=1).cpu().numpy()
        seen_bad[start:stop] = seen_outranks.sum(dim=1).cpu().numpy()
        unseen_pool[start:stop] = unseen_wrong.sum(dim=1).cpu().numpy()
        best_positive[start:stop] = best.cpu().numpy()
    return {
        "seen_pool_size": seen_pool,
        "unseen_pool_size": unseen_pool,
        "seen_outranking": seen_bad,
        "unseen_outranking": unseen_bad,
        "best_positive_cosine": best_positive,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--pretrained-archive", type=Path, required=True)
    parser.add_argument("--trained-archive", type=Path, required=True)
    parser.add_argument("--sop-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute-sop-seen-gallery-effect", action="store_true", required=True)
    args = parser.parse_args()
    if (
        args.output.exists()
        or args.output.is_symlink()
        or sha256(args.pretrained_archive) != PRETRAINED_ARCHIVE_SHA256
        or sha256(args.trained_archive) != TRAINED_ARCHIVE_SHA256
        or not torch.cuda.is_available()
        or gpu_compute_pids()
    ):
        raise ValueError("SOP seen-gallery invocation differs")
    started = time.perf_counter()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(16)
    records = parse_sop_records(args.sop_root)
    if ordered_record_sha256(records) != EXPECTED_SOP_RECORD_SHA256:
        raise ValueError("SOP ordered train inventory differs")
    train = tuple(row for row in records if row.split == "train")
    labels = np.asarray([row.label for row in train], dtype=np.int64)
    partition = deterministic_class_partition(
        tuple(map(int, labels)), fit_fraction=FIT_FRACTION, seed=SPLIT_SEED
    )
    fit = np.asarray(partition.fit_row_indexes, dtype=np.int64)
    query = np.asarray(partition.validation_row_indexes, dtype=np.int64)
    if len(fit) != 53_700 or len(query) != 5_851:
        raise ValueError("SOP class-disjoint split differs")
    arms = {}
    for name, path in (("pretrained", args.pretrained_archive), ("trained", args.trained_archive)):
        with np.load(path, allow_pickle=False) as archive:
            values = np.ascontiguousarray(archive["train_embeddings"], dtype=np.float32)
            archive_labels = np.asarray(archive["train_labels"], dtype=np.int64)
            image_ids = np.asarray(archive["train_image_ids"], dtype=np.int64)
            metadata = json.loads(str(archive["metadata_json"].item()))
        if (
            values.shape != (59_551, 768)
            or not np.isfinite(values).all()
            or not np.array_equal(archive_labels, labels)
            or not np.array_equal(image_ids, [row.image_id for row in train])
            or (
                name == "trained"
                and (
                    metadata.get("step") != 48_000
                    or metadata.get("trained_checkpoint_sha256")
                    != "232f7cee93e39fa242d8461f8dc8cee684d7228eef01f8fe9399b9782e80a1b2"
                )
            )
            or (name == "pretrained" and metadata.get("model_identifier") != "UNICOM-ViT-B/16")
        ):
            raise ValueError(f"SOP {name} feature authority differs")
        raw = count_outranking_negatives(torch.from_numpy(values).cuda(), labels, query, fit)
        holdout_r1 = float(np.mean(raw["unseen_outranking"] == 0))
        full_r1 = float(
            np.mean((raw["unseen_outranking"] == 0) & (raw["seen_outranking"] == 0))
        )
        if name == "trained" and (
            abs(holdout_r1 - 0.9603486583490002) > 1e-9
            or abs(full_r1 - 0.8858314817979832) > 1e-9
        ):
            raise ValueError("SOP trained full-gallery baseline parity differs")
        size_ladder = {}
        for negatives in NEGATIVE_COUNTS:
            unseen = expected_recall_random_negatives(
                raw["unseen_pool_size"], raw["unseen_outranking"], negatives
            )
            seen = expected_recall_random_negatives(
                raw["seen_pool_size"], raw["seen_outranking"], negatives
            )
            size_ladder[str(negatives)] = {
                "unseen_expected_recall_at_1": float(unseen.mean()),
                "seen_expected_recall_at_1": float(seen.mean()),
                "seen_minus_unseen": float((seen - unseen).mean()),
            }
        arms[name] = {
            "holdout_only_recall_at_1": holdout_r1,
            "all_train_recall_at_1": full_r1,
            "size_ladder": size_ladder,
            "per_query": {key: value.astype(float).tolist() for key, value in raw.items()},
        }
        del values, raw
        torch.cuda.empty_cache()
    interaction = {
        str(negatives): (
            arms["trained"]["size_ladder"][str(negatives)]["seen_minus_unseen"]
            - arms["pretrained"]["size_ladder"][str(negatives)]["seen_minus_unseen"]
        )
        for negatives in NEGATIVE_COUNTS
    }
    result = {
        "schema": "sfora-sop-seen-gallery-effect-v1",
        "claim_eligible": False,
        "dataset": "Stanford Online Products official TRAIN split only",
        "split": (
            "seed-179019 class-disjoint 53700 fit / 5851 holdout; "
            "queries are all holdout rows"
        ),
        "random_gallery_estimand": (
            "exact hypergeometric expectation under a uniform sample of N wrong negative images "
            "without replacement, plus every other positive in the query's holdout product"
        ),
        "negative_counts": NEGATIVE_COUNTS,
        "arms": arms,
        "trained_minus_pretrained_seen_effect": interaction,
        "inputs": {
            "pretrained_archive_sha256": PRETRAINED_ARCHIVE_SHA256,
            "trained_archive_sha256": TRAINED_ARCHIVE_SHA256,
            "train_metadata_sha256": sha256(args.sop_root / "Ebay_train.txt"),
            "script_sha256": sha256(Path(__file__)),
        },
        "hardware": {
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "python": platform.python_version(),
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_host_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, sort_keys=True, allow_nan=False, separators=(",", ":"))
        stream.write("\n")
    print(json.dumps({"size_ladder": {name: arms[name]["size_ladder"] for name in arms}}))


if __name__ == "__main__":
    main()
