"""Ensemble retrieval evaluation over several saved test-embedding sets.

Each input .npz must contain `embeddings` (N, D) and `labels` (N,) in the SAME
sample order (the test loader is deterministic, so independently-trained seeds
align row-for-row). We L2-normalise each model's embeddings, concatenate them
per sample (feature-concatenation ensemble), L2-normalise the concatenation, and
compute cosine Recall@1 with the project's own retrieval scorer.

Optionally, `--compress-dim D` PCA-compresses the concatenated ensemble embedding
down to D dimensions before retrieval — a cheap way to keep most of the ensemble
gain while shrinking the N*512-dim concatenation back to a deployable size.

Usage:
    uv run python scripts/ensemble_eval.py a.npz b.npz c.npz
    uv run python scripts/ensemble_eval.py --compress-dim 512 reports/emb/*.npz
    uv run python scripts/ensemble_eval.py --compress-sweep reports/emb/*.npz
"""

from __future__ import annotations

import argparse

import numpy as np

from sfora.image_benchmark import image_self_retrieval_score


def _l2(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1.0e-12)


def _pca_compress(features: np.ndarray, dim: int) -> np.ndarray:
    """Project mean-centred features onto their top-`dim` principal directions."""
    centred = features - features.mean(axis=0, keepdims=True)
    # Economy SVD: right singular vectors are the principal axes.
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    components = vt[:dim]
    return centred @ components.T


def _recall(features: np.ndarray, labels: np.ndarray) -> float:
    return image_self_retrieval_score(_l2(features), labels, random_state=0).recall_at_1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="one or more .npz embedding files")
    parser.add_argument(
        "--compress-dim",
        type=int,
        default=None,
        help="PCA-compress the concatenated ensemble to this many dimensions before retrieval",
    )
    parser.add_argument(
        "--compress-sweep",
        action="store_true",
        help="report R@1 for the full concat and a sweep of PCA-compressed dimensions",
    )
    args = parser.parse_args()

    embeddings_list: list[np.ndarray] = []
    labels_reference: np.ndarray | None = None
    per_model_recall: list[float] = []
    for path in args.paths:
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
    concatenated = np.concatenate(embeddings_list, axis=1)
    full_dim = concatenated.shape[1]
    ensemble_recall = _recall(concatenated, labels_reference)
    mean_single = float(np.mean(per_model_recall))
    best_single = max(per_model_recall)
    print(
        f"\n=== ENSEMBLE of {len(args.paths)} models: R@1={ensemble_recall:.4f} "
        f"(dim {full_dim}, mean single {mean_single:.4f}, best single {best_single:.4f}) ==="
    )

    if args.compress_sweep:
        print("\n--- PCA compression of the concatenated ensemble ---")
        dims = [d for d in (256, 512, 1024, 1536, 2048) if d < full_dim]
        for dim in dims:
            r = _recall(_pca_compress(concatenated, dim), labels_reference)
            kept = r / ensemble_recall
            print(f"  {full_dim:>5d} -> {dim:<5d}  R@1={r:.4f}  (retains {kept:.1%})")
    elif args.compress_dim is not None and args.compress_dim < full_dim:
        r = _recall(_pca_compress(concatenated, args.compress_dim), labels_reference)
        print(
            f"\n=== COMPRESSED {full_dim} -> {args.compress_dim}: R@1={r:.4f} "
            f"(retains {r / ensemble_recall:.1%} of the ensemble) ==="
        )


if __name__ == "__main__":
    main()
