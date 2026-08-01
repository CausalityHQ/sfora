"""Frozen analysis for the preregistered pre-normalisation magnitude diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


def _load_embeddings(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return (
            np.asarray(payload["embeddings"], dtype=np.float32),
            np.asarray(payload["labels"], dtype=np.int64),
            np.asarray(payload["example_ids"]),
        )


def _icc_one_way(values: np.ndarray, labels: np.ndarray) -> float:
    groups = [values[labels == label] for label in np.unique(labels)]
    groups = [group for group in groups if group.size >= 2]
    if len(groups) < 2:
        return float("nan")
    counts = np.asarray([group.size for group in groups], dtype=np.float64)
    total = float(counts.sum())
    grand_mean = float(np.concatenate(groups).mean())
    ss_between = sum(group.size * (float(group.mean()) - grand_mean) ** 2 for group in groups)
    ss_within = sum(float(np.sum((group - group.mean()) ** 2)) for group in groups)
    ms_between = ss_between / (len(groups) - 1)
    ms_within = ss_within / (int(total) - len(groups))
    effective_n = (total - float(np.sum(counts**2)) / total) / (len(groups) - 1)
    return float((ms_between - ms_within) / (ms_between + (effective_n - 1.0) * ms_within))


def _identity_decomposition(
    norms: np.ndarray,
    correctness: np.ndarray,
    margins: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float | int]:
    residual_norms: list[np.ndarray] = []
    residual_correctness: list[np.ndarray] = []
    residual_margins: list[np.ndarray] = []
    mean_norms: list[float] = []
    mean_correctness: list[float] = []
    mean_margins: list[float] = []
    for label in np.unique(labels):
        selected = labels == label
        if int(selected.sum()) < 2:
            continue
        group_norms = norms[selected]
        group_correctness = correctness[selected]
        group_margins = margins[selected]
        residual_norms.append(group_norms - group_norms.mean())
        residual_correctness.append(group_correctness - group_correctness.mean())
        residual_margins.append(group_margins - group_margins.mean())
        mean_norms.append(float(group_norms.mean()))
        mean_correctness.append(float(group_correctness.mean()))
        mean_margins.append(float(group_margins.mean()))

    pooled_norms = np.concatenate(residual_norms)
    pooled_correctness = np.concatenate(residual_correctness)
    pooled_margins = np.concatenate(residual_margins)
    return {
        "query_identities_with_replicates": len(mean_norms),
        "within_identity_rows": int(pooled_norms.size),
        "within_identity_correctness_pearson": float(
            np.corrcoef(pooled_norms, pooled_correctness)[0, 1]
        ),
        "within_identity_margin_spearman": float(
            spearmanr(pooled_norms, pooled_margins).statistic
        ),
        "between_identity_correctness_pearson": float(
            np.corrcoef(mean_norms, mean_correctness)[0, 1]
        ),
        "between_identity_margin_spearman": float(
            spearmanr(mean_norms, mean_margins).statistic
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", type=Path, required=True)
    parser.add_argument("--gallery", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--norms", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=256)
    args = parser.parse_args()

    query, query_labels, query_ids = _load_embeddings(args.query)
    gallery, gallery_labels, gallery_ids = _load_embeddings(args.gallery)
    train, train_labels, train_ids = _load_embeddings(args.train)
    del train
    with np.load(args.norms, allow_pickle=False) as payload:
        query_norms = np.asarray(payload["query_norms"], dtype=np.float64)
        gallery_norms = np.asarray(payload["gallery_norms"], dtype=np.float64)
        train_norms = np.asarray(payload["train_norms"], dtype=np.float64)
        if not np.array_equal(query_ids, payload["query_example_ids"]):
            raise SystemExit("query norm rows do not match query embeddings")
        if not np.array_equal(gallery_ids, payload["gallery_example_ids"]):
            raise SystemExit("gallery norm rows do not match gallery embeddings")
        if not np.array_equal(train_ids, payload["train_example_ids"]):
            raise SystemExit("train norm rows do not match train embeddings")

    correctness: list[np.ndarray] = []
    margins: list[np.ndarray] = []
    for start in range(0, query.shape[0], args.chunk_size):
        stop = min(start + args.chunk_size, query.shape[0])
        scores = query[start:stop] @ gallery.T
        batch_labels = query_labels[start:stop]
        same = batch_labels[:, None] == gallery_labels[None, :]
        best_positive = np.max(np.where(same, scores, -np.inf), axis=1)
        best_negative = np.max(np.where(~same, scores, -np.inf), axis=1)
        top_labels = gallery_labels[np.argmax(scores, axis=1)]
        correctness.append(top_labels == batch_labels)
        margins.append(best_positive - best_negative)
    correct = np.concatenate(correctness).astype(np.float64)
    margin = np.concatenate(margins).astype(np.float64)

    result = {
        "queries": int(query.shape[0]),
        "gallery": int(gallery.shape[0]),
        "train": int(train_labels.shape[0]),
        "recall_at_1": float(correct.mean()),
        "query_norm_mean": float(query_norms.mean()),
        "query_norm_std": float(query_norms.std(ddof=1)),
        "gallery_norm_mean": float(gallery_norms.mean()),
        "correctness_pearson": float(np.corrcoef(query_norms, correct)[0, 1]),
        "margin_spearman": float(spearmanr(query_norms, margin).statistic),
        "train_identity_icc": _icc_one_way(train_norms, train_labels),
    }
    result.update(_identity_decomposition(query_norms, correct, margin, query_labels))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
