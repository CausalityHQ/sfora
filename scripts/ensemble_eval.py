"""Ensemble retrieval evaluation over several saved test-embedding sets.

Each input .npz must contain `embeddings` (N, D) and `labels` (N,) in the SAME
sample order (the test loader is deterministic, so independently-trained seeds
align row-for-row). We L2-normalise each model's embeddings, concatenate them
per sample (feature-concatenation ensemble), L2-normalise the concatenation, and
compute cosine Recall@1 with the project's own retrieval scorer.

Usage:
    uv run python scripts/ensemble_eval.py a.npz b.npz c.npz
"""

from __future__ import annotations

import sys

import numpy as np

from sfora.image_benchmark import image_self_retrieval_score


def _l2(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1.0e-12)


def main(paths: list[str]) -> None:
    if not paths:
        raise SystemExit("provide one or more .npz embedding files")
    embeddings_list: list[np.ndarray] = []
    labels_reference: np.ndarray | None = None
    per_model_recall: list[float] = []
    for path in paths:
        data = np.load(path)
        embeddings = _l2(np.asarray(data["embeddings"], dtype=np.float64))
        labels = np.asarray(data["labels"], dtype=np.int64)
        if labels_reference is None:
            labels_reference = labels
        elif not np.array_equal(labels_reference, labels):
            raise SystemExit(f"label order mismatch in {path}; cannot ensemble")
        embeddings_list.append(embeddings)
        single = image_self_retrieval_score(embeddings, labels, random_state=0)
        per_model_recall.append(single.recall_at_1)
        print(f"{path}: R@1={single.recall_at_1:.4f}")

    assert labels_reference is not None
    concatenated = _l2(np.concatenate(embeddings_list, axis=1))
    ensemble = image_self_retrieval_score(concatenated, labels_reference, random_state=0)
    mean_single = float(np.mean(per_model_recall))
    best_single = max(per_model_recall)
    print(
        f"\n=== ENSEMBLE of {len(paths)} models: R@1={ensemble.recall_at_1:.4f} "
        f"(mean single {mean_single:.4f}, best single {best_single:.4f}) ==="
    )


if __name__ == "__main__":
    main(sys.argv[1:])
