"""Push the *train-clean* 512-dim projection as close to the full pack (100%) as
honestly possible — every projection here is fit on the disjoint TRAIN classes and
only *applied* to TEST. Nothing about the test split informs the fold.

Baseline to beat: plain inductive GPA = 98.0% (rotate each model to a train-fit
consensus, then AVERAGE). Averaging collapses 1536 -> 512 by summation, cancelling the
complementary cross-model directions that the concat keeps. The idea here: align first,
then keep the top-512 directions with a TRAIN-fit PCA instead of averaging them away.

    uv run python scripts/explore_trainclean_projection.py \
        --train 'reports/emb/herd_tt_seed*.train.npz' --test 'reports/emb/herd_tt_seed*.test.npz'
"""

from __future__ import annotations

import argparse
import glob

import numpy as np

from sfora.image_benchmark import image_self_retrieval_score


def _l2(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def _load_blocks(pattern: str) -> tuple[list[np.ndarray], np.ndarray]:
    blocks: list[np.ndarray] = []
    labels: np.ndarray | None = None
    for path in sorted(glob.glob(pattern)):
        with np.load(path, allow_pickle=False) as data:
            blocks.append(_l2(np.asarray(data["embeddings"], dtype=np.float64)))
            lab = np.asarray(data["labels"], dtype=np.int64)
        if labels is None:
            labels = lab
        elif not np.array_equal(labels, lab):
            raise SystemExit(f"label mismatch in {path}")
    if not blocks:
        raise SystemExit(f"no files match {pattern}")
    assert labels is not None
    return blocks, labels


def gpa_rotations(blocks: list[np.ndarray], *, iters: int = 20) -> list[np.ndarray]:
    """Generalized Procrustes: per-model rotations to a running consensus (train-fit)."""
    aligned = [b.copy() for b in blocks]
    rotations = [np.eye(b.shape[1]) for b in blocks]
    for _ in range(iters):
        consensus = _l2(np.mean(aligned, axis=0))
        for i, block in enumerate(blocks):
            u, _, vt = np.linalg.svd(block.T @ consensus, full_matrices=False)
            correction = np.eye(vt.shape[0])
            correction[-1, -1] = np.sign(np.linalg.det(u @ vt))
            rotations[i] = u @ correction @ vt
            aligned[i] = block @ rotations[i]
    return rotations


def _r1(emb: np.ndarray, labels: np.ndarray) -> float:
    return image_self_retrieval_score(_l2(emb), labels).recall_at_1


def _aligned_concat(blocks: list[np.ndarray], rotations: list[np.ndarray]) -> np.ndarray:
    return np.concatenate([_l2(b) @ r for b, r in zip(blocks, rotations, strict=True)], axis=1)


def _pca_fit(train: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(train - mean, full_matrices=False)
    return mean, vt[:k].T  # (mean, components DxK)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", default="reports/emb/herd_tt_seed*.train.npz")
    ap.add_argument("--test", default="reports/emb/herd_tt_seed*.test.npz")
    args = ap.parse_args()

    train_blocks, _ = _load_blocks(args.train)
    test_blocks, y_test = _load_blocks(args.test)
    n, dim = len(train_blocks), train_blocks[0].shape[1]

    concat_test = _l2(np.concatenate(test_blocks, axis=1))
    concat_r1 = _r1(concat_test, y_test)
    single_r1 = _r1(test_blocks[0], y_test)

    rotations = gpa_rotations(train_blocks)
    tr_aligned = _aligned_concat(train_blocks, rotations)
    te_aligned = _aligned_concat(test_blocks, rotations)

    results: dict[str, float] = {}

    # 1. plain inductive GPA: rotate to train consensus, then AVERAGE (the 98.0% baseline)
    results["inductive GPA (average)"] = _r1(
        np.mean([_l2(b) @ r for b, r in zip(test_blocks, rotations, strict=True)], axis=0), y_test
    )

    # 2. train-fit PCA on the RAW concat (no alignment)
    raw_tr = _l2(np.concatenate(train_blocks, axis=1))
    mean, comp = _pca_fit(raw_tr, dim)
    results["train-PCA on raw concat"] = _r1((concat_test - mean) @ comp, y_test)

    # 3. GPA-align THEN train-fit PCA to 512 (keep top directions, don't average)
    mean_a, comp_a = _pca_fit(tr_aligned, dim)
    results["GPA-align + train-PCA(512)"] = _r1((te_aligned - mean_a) @ comp_a, y_test)

    # 4. GPA-align + train-fit PCA-whiten to 512 (unit-variance directions)
    mean_w = tr_aligned.mean(axis=0, keepdims=True)
    u_tr, s_tr, vt_tr = np.linalg.svd(tr_aligned - mean_w, full_matrices=False)
    whiten = vt_tr[:dim].T / (s_tr[:dim] / np.sqrt(len(tr_aligned)))
    results["GPA-align + train-PCA-whiten(512)"] = _r1((te_aligned - mean_w) @ whiten, y_test)

    single_pct = 100 * single_r1 / concat_r1
    print(f"{n} models x {dim}-dim  (CUB test, train-clean folds — nothing fit on test)")
    print(f"  single model (512d)             R@1={single_r1:.4f}  ({single_pct:.2f}%)")
    print(f"  full concat ({n * dim}d)            R@1={concat_r1:.4f}   <- 100% target\n")
    for name, r1 in sorted(results.items(), key=lambda kv: -kv[1]):
        print(f"  {name:36s} R@1={r1:.4f}   retained={100 * r1 / concat_r1:.2f}%")


if __name__ == "__main__":
    main()
