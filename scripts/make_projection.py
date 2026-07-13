"""Project a saved test-embedding set to 2D (t-SNE) and 3D (PCA) for the website.

Given an .npz of `embeddings` (N, D) and `labels` (N,), pick the most-populous
classes, project their samples to 2D and 3D, and write a compact JSON the site's
comparison visualisation reads. Run it on a HERD model and on a baseline
(Proxy Anchor / frozen) to show, side by side, how much tighter HERD's clusters
are — the difference in *how* the embedding space is organised.

Usage:
    uv run python scripts/make_projection.py reports/emb/ema_seed0.npz \
        --classes 8 --per-class 45 --out site/src/data/proj_herd.json --label HERD
"""

from __future__ import annotations

import argparse
import json

import numpy as np


def _l2(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def _silhouette(x: np.ndarray, labels: np.ndarray) -> float:
    """Mean cosine silhouette — a single number for 'how separated are the classes'."""
    try:
        from sklearn.metrics import silhouette_score

        return float(silhouette_score(x, labels, metric="cosine"))
    except Exception:
        return float("nan")


def _recall_at_1(x: np.ndarray, labels: np.ndarray) -> float:
    """Leave-one-out cosine Recall@1 over the FULL embedding set — the honest
    'how good is retrieval' number, not tied to the sampled projection subset."""
    xn = _l2(x)
    sim = xn @ xn.T
    np.fill_diagonal(sim, -1e9)
    nn = labels[sim.argmax(1)]
    return float((nn == labels).mean())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("npz")
    ap.add_argument("--classes", type=int, default=8)
    ap.add_argument("--per-class", type=int, default=45)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="model")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data = np.load(args.npz)
    emb = _l2(np.asarray(data["embeddings"], dtype=np.float64))
    lab = np.asarray(data["labels"], dtype=np.int64)

    # Full-test-set Recall@1 (all classes) — the grounded quality number.
    recall1 = round(_recall_at_1(emb, lab), 4)
    total_classes = int(len(np.unique(lab)))

    # Pick the most-populous classes, then a fixed sample per class (deterministic).
    uniq, counts = np.unique(lab, return_counts=True)
    top = uniq[np.argsort(-counts)[: args.classes]]
    rng = np.random.default_rng(args.seed)
    idx: list[int] = []
    for c in top:
        c_idx = np.where(lab == c)[0]
        rng.shuffle(c_idx)
        idx.extend(int(i) for i in c_idx[: args.per_class])
    idx_arr = np.array(idx)
    x = emb[idx_arr]
    y = lab[idx_arr]
    remap = {int(c): i for i, c in enumerate(top)}
    colors = np.array([remap[int(v)] for v in y])

    # 3D via PCA (fast, deterministic).
    xc = x - x.mean(0, keepdims=True)
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    p3 = xc @ vt[:3].T
    p3 = p3 / np.abs(p3).max()

    # 2D via t-SNE (falls back to PCA if scikit-learn is unavailable).
    try:
        from sklearn.manifold import TSNE

        p2 = TSNE(
            n_components=2,
            metric="cosine",
            init="pca",
            perplexity=min(30, max(5, len(x) // 12)),
            random_state=args.seed,
        ).fit_transform(x)
        method2d = "t-SNE"
    except Exception:
        p2 = xc @ vt[:2].T
        method2d = "PCA"
    p2 = (p2 - p2.mean(0)) / (np.abs(p2 - p2.mean(0)).max() + 1e-9)

    out = {
        "label": args.label,
        "classes": int(len(top)),
        "totalClasses": total_classes,
        "recall1": recall1,
        "method2d": method2d,
        "silhouette": round(_silhouette(x, y), 3),
        "points2d": [
            {"x": round(float(a), 4), "y": round(float(b), 4), "c": int(c)}
            for (a, b), c in zip(p2, colors, strict=True)
        ],
        "points3d": [
            {
                "x": round(float(a), 4),
                "y": round(float(b), 4),
                "z": round(float(cc), 4),
                "c": int(c),
            }
            for (a, b, cc), c in zip(p3, colors, strict=True)
        ],
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh)
    print(
        f"{args.label}: {len(x)} points, {len(top)} classes, 2d={method2d}, "
        f"silhouette={out['silhouette']}, full-set R@1={recall1} "
        f"({total_classes} classes) -> {args.out}"
    )


if __name__ == "__main__":
    main()
