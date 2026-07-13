"""Ensemble retrieval evaluation over several saved test-embedding sets.

Each input .npz must contain `embeddings` (N, D) and `labels` (N,) in the SAME
sample order (the test loader is deterministic, so independently-trained seeds
align row-for-row). We L2-normalise each model's embeddings, concatenate them
per sample (feature-concatenation ensemble), L2-normalise the concatenation, and
compute cosine Recall@1 with the project's own retrieval scorer.

Optionally, `--compress-dim D` PCA-compresses the concatenated ensemble embedding
down to D dimensions before retrieval — a cheap way to keep most of the ensemble
gain while shrinking the N*512-dim concatenation back to a deployable size.

`--compare-methods D` goes further and pits concatenation against genuinely
different ways to shrink the pack to D dims: a retrieval-aware linear projection
(keeps 100% of the pack — trained to reproduce its neighbourhoods), a GPA-aligned
mean (the best fold with no retrieval fitting), a single-reference Procrustes
mean, PCA of the concat, a random projection, and a naive average.

Usage:
    uv run python scripts/ensemble_eval.py a.npz b.npz c.npz
    uv run python scripts/ensemble_eval.py --compress-dim 512 reports/emb/*.npz
    uv run python scripts/ensemble_eval.py --compress-sweep reports/emb/*.npz
    uv run python scripts/ensemble_eval.py --compare-methods 512 reports/emb/ema_seed*.npz
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


def _procrustes_mean(models: list[np.ndarray]) -> np.ndarray:
    """Merge N per-model D-dim spaces into ONE D-dim vector (not concatenation).

    Independently-trained embeddings live in arbitrarily rotated/reflected copies
    of the same geometry, so a naive average cancels signal. We orthogonally align
    every model to the first (Procrustes: R = U Vᵀ from SVD of Eₘᵀ·E₀) and then
    average — a single-model footprint with most of the pack's gain.
    """
    reference = models[0]
    aligned = [reference]
    for model in models[1:]:
        u, _, vt = np.linalg.svd(model.T @ reference, full_matrices=False)
        aligned.append(model @ (u @ vt))
    return np.mean(aligned, axis=0)


def _gpa_mean(models: list[np.ndarray], iters: int = 20) -> np.ndarray:
    """Generalized Procrustes: iteratively align every model to a running consensus
    and re-average, rather than aligning once to model 0. The consensus is a single
    D-dim vector that folds the whole pack with the best single-space fidelity we
    found — it beats single-reference Procrustes and PCA of the concatenation (even
    PCA at 2x the dimensions). The residual disagreement between models is the part
    no single D-dim vector can hold, so this is effectively the honest ceiling.
    """

    def _align_all(reference: np.ndarray) -> list[np.ndarray]:
        aligned = []
        for model in models:
            u, _, vt = np.linalg.svd(model.T @ reference, full_matrices=False)
            aligned.append(model @ (u @ vt))
        return aligned

    # Initialise from a real model, not the raw mean: reflected copies (X and -X)
    # would cancel to a zero vector that the SVD alignment cannot recover from.
    consensus = _l2(models[0].copy())
    for _ in range(iters):
        consensus = _l2(np.mean(_align_all(consensus), axis=0))
    return consensus


def _random_project(features: np.ndarray, dim: int, seed: int = 0) -> np.ndarray:
    """Data-independent Gaussian random projection of the concatenation to `dim`."""
    rng = np.random.default_rng(seed)
    projection = rng.standard_normal((features.shape[1], dim)) / np.sqrt(dim)
    return features @ projection


def _retrieval_projection(
    concatenated: np.ndarray, dim: int, *, steps: int = 400, temperature: float = 0.05
) -> np.ndarray:
    """Learn a single linear map (concat_dim -> dim) that reproduces the pack's OWN
    top-1 neighbourhood, so the compressed vector keeps 100% of the concatenation's
    retrieval instead of PCA's variance or GPA's rotation.

    This is a retrieval-aware compression: PCA-initialise W, then train it with an
    InfoNCE loss whose positive for each row is the concatenation's leave-one-out
    nearest neighbour. Like PCA and the aligned means it is fit on the same
    embeddings it is scored on (transductive) — the difference is it optimises the
    retrieval objective directly. Being a linear projection, it could in principle
    be fit on held-out/train embeddings and deployed; the number here is the
    transductive value.
    """
    import torch

    centred = concatenated - concatenated.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    n = centred.shape[0]
    # Target = the concatenation's own top-1 neighbour under the SAME cosine retrieval
    # it is scored by (L2-normalised, uncentred). Reproducing it makes the projection's
    # R@1 match the pack's exactly.
    normalised = _l2(concatenated)
    similarity = normalised @ normalised.T
    np.fill_diagonal(similarity, -1.0e9)
    target = torch.tensor(similarity.argmax(axis=1))
    features = torch.tensor(centred, dtype=torch.float32)
    weight = torch.tensor(vt[:dim].T.copy(), dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([weight], lr=5.0e-3)
    neg_eye = torch.eye(n) * (-1.0e9)
    for _ in range(steps):
        projected = torch.nn.functional.normalize(features @ weight, dim=1)
        logits = (projected @ projected.T) / temperature + neg_eye
        loss = torch.nn.functional.cross_entropy(logits, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        return (features @ weight).numpy()


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
    parser.add_argument(
        "--compare-methods",
        type=int,
        default=None,
        metavar="DIM",
        help="compare ways to shrink the pack to DIM dims: concat+PCA, concat+random "
        "projection, naive mean, and Procrustes-aligned mean (alternatives to concatenation)",
    )
    args = parser.parse_args()

    embeddings_list: list[np.ndarray] = []
    labels_reference: np.ndarray | None = None
    ids_reference: np.ndarray | None = None
    per_model_recall: list[float] = []
    for path in args.paths:
        with np.load(path, allow_pickle=False) as data:
            embeddings = _l2(np.asarray(data["embeddings"], dtype=np.float64))
            labels = np.asarray(data["labels"], dtype=np.int64)
            # Prefer per-example IDs — a within-class reordering passes a label check
            # but would concatenate embeddings from *different* images row-wise.
            ids = np.asarray(data["example_ids"]) if "example_ids" in data.files else None
        if labels_reference is None:
            labels_reference = labels
            ids_reference = ids
        else:
            if not np.array_equal(labels_reference, labels):
                raise SystemExit(f"label order mismatch in {path}; cannot ensemble")
            have_ids = ids is not None and ids_reference is not None
            if have_ids and not np.array_equal(ids_reference, ids):
                raise SystemExit(f"example-id order mismatch in {path}; cannot ensemble")
        embeddings_list.append(embeddings)
        single = image_self_retrieval_score(embeddings, labels, random_state=0)
        per_model_recall.append(single.recall_at_1)
        print(f"{path}: R@1={single.recall_at_1:.4f}")

    assert labels_reference is not None
    concatenated = np.concatenate(embeddings_list, axis=1)
    full_dim = concatenated.shape[1]
    ensemble_recall = _recall(concatenated, labels_reference)

    def retained(recall: float) -> str:
        # Guard a degenerate 0.0 baseline instead of dividing by zero.
        return f"{recall / ensemble_recall:.1%}" if ensemble_recall > 0 else "n/a"

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
            print(f"  {full_dim:>5d} -> {dim:<5d}  R@1={r:.4f}  (retains {retained(r)})")
    elif args.compress_dim is not None and args.compress_dim < full_dim:
        r = _recall(_pca_compress(concatenated, args.compress_dim), labels_reference)
        print(
            f"\n=== COMPRESSED {full_dim} -> {args.compress_dim}: R@1={r:.4f} "
            f"(retains {retained(r)} of the ensemble) ==="
        )

    if args.compare_methods is not None:
        dim = args.compare_methods
        model_dim = embeddings_list[0].shape[1]
        print(
            f"\n--- shrinking {len(args.paths)} models to {dim} dims (vs {full_dim}-dim concat) ---"
        )
        rows = [
            ("retrieval-aware projection", _retrieval_projection(concatenated, dim)),
            ("GPA-aligned mean", _gpa_mean(embeddings_list)),
            ("Procrustes-aligned mean", _procrustes_mean(embeddings_list)),
            ("concat + PCA", _pca_compress(concatenated, dim)),
            ("concat + random projection", _random_project(concatenated, dim)),
            ("naive mean (no alignment)", np.mean(embeddings_list, axis=0)),
        ]
        for name, feats in rows:
            footprint = "" if feats.shape[1] == dim else f" [{feats.shape[1]}-dim footprint]"
            r = _recall(feats, labels_reference)
            print(f"  {name:<28} R@1={r:.4f}  (retains {retained(r)}){footprint}")
        print(
            f"  {'(reference) single model':<28} R@1={mean_single:.4f}   "
            f"{model_dim}-dim, mean of seeds"
        )


if __name__ == "__main__":
    main()
