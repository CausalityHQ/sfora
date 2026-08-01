"""Measure five-seed all-miss rescues and correct-class ranks on saved packs."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np


def _l2(values: np.ndarray) -> np.ndarray:
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=256)
    args = parser.parse_args()

    paths = sorted(glob.glob(args.test))
    if len(paths) != 5:
        raise SystemExit(f"expected exactly five packs, found {len(paths)}: {paths}")
    blocks: list[np.ndarray] = []
    labels: np.ndarray | None = None
    ids: np.ndarray | None = None
    for path in paths:
        with np.load(path, allow_pickle=False) as payload:
            blocks.append(_l2(np.asarray(payload["embeddings"], dtype=np.float32)))
            current_labels = np.asarray(payload["labels"], dtype=np.int64)
            current_ids = np.asarray(payload["example_ids"])
        if labels is None:
            labels, ids = current_labels, current_ids
        elif not np.array_equal(labels, current_labels) or not np.array_equal(ids, current_ids):
            raise SystemExit(f"row mismatch in {path}")
    assert labels is not None

    rescued_worst_ranks: list[int] = []
    all_miss_count = 0
    concat_hits = 0
    individual_hits = np.zeros(len(blocks), dtype=np.int64)
    n = labels.shape[0]
    for start in range(0, n, args.chunk_size):
        stop = min(start + args.chunk_size, n)
        query_labels = labels[start:stop]
        seed_scores = [block[start:stop] @ block.T for block in blocks]
        rows = np.arange(stop - start)
        columns = np.arange(start, stop)
        for scores in seed_scores:
            scores[rows, columns] = -np.inf
        hits = np.stack(
            [labels[np.argmax(scores, axis=1)] == query_labels for scores in seed_scores], axis=1
        )
        individual_hits += hits.sum(axis=0)
        mean_scores = np.mean(seed_scores, axis=0)
        concat_hit = labels[np.argmax(mean_scores, axis=1)] == query_labels
        concat_hits += int(concat_hit.sum())
        all_miss = ~hits.any(axis=1)
        all_miss_count += int(all_miss.sum())
        rescue_rows = np.flatnonzero(all_miss & concat_hit)
        for row in rescue_rows:
            positive = labels == query_labels[row]
            positive[start + row] = False
            ranks = []
            for scores in seed_scores:
                best_positive = float(np.max(scores[row, positive]))
                ranks.append(1 + int(np.sum(scores[row] > best_positive)))
            rescued_worst_ranks.append(max(ranks))

    rescue_count = len(rescued_worst_ranks)
    ranks_array = np.asarray(rescued_worst_ranks, dtype=np.int64)
    result = {
        "paths": paths,
        "queries": int(n),
        "individual_r1": (individual_hits / n).tolist(),
        "concat_r1": concat_hits / n,
        "all_miss_queries": all_miss_count,
        "all_miss_rescues": rescue_count,
        "all_miss_rescue_fraction": rescue_count / n,
        "rescued_worst_rank_median": float(np.median(ranks_array)) if rescue_count else None,
        "rescued_worst_rank_p90": float(np.percentile(ranks_array, 90)) if rescue_count else None,
        "rescues_all_seeds_top10_fraction": float(np.mean(ranks_array <= 10)) if rescue_count else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
