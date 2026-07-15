"""Inductive Generalized Procrustes Analysis fold: fit the per-model rotations on the
disjoint TRAIN split, freeze them, and apply to TEST — an honest, train/test-clean
compression of the ensemble pack to a single model's 512-dim footprint.

GPA was the best *transductive* fold (99.4%), but that computed the alignment on the
test embeddings. The Procrustes rotation that aligns model i's coordinate frame to the
shared consensus is a property of the *model*, not of any class — so it should transfer
from train classes to disjoint test classes almost losslessly. This tests exactly that:
nothing about the test split informs the projection.

    uv run python scripts/train_fit_gpa.py \
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
    """Return per-model L2-normalised embedding blocks + shared labels."""
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
    """Generalized Procrustes: iteratively rotate each block to a running consensus.

    Returns the frozen per-model rotation matrices (D x D), det = +1 (proper rotations).
    """
    aligned = [b.copy() for b in blocks]
    rotations = [np.eye(b.shape[1]) for b in blocks]
    for _ in range(iters):
        consensus = _l2(np.mean(aligned, axis=0))
        for i, block in enumerate(blocks):
            u, _, vt = np.linalg.svd(block.T @ consensus, full_matrices=False)
            correction = np.eye(vt.shape[0])
            correction[-1, -1] = np.sign(np.linalg.det(u @ vt))  # keep a proper rotation
            rotation = u @ correction @ vt
            rotations[i] = rotation
            aligned[i] = block @ rotation
    return rotations


def _apply(blocks: list[np.ndarray], rotations: list[np.ndarray]) -> np.ndarray:
    aligned = [_l2(b) @ r for b, r in zip(blocks, rotations, strict=True)]
    return _l2(np.mean(aligned, axis=0))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", default="reports/emb/herd_tt_seed*.train.npz")
    ap.add_argument("--test", default="reports/emb/herd_tt_seed*.test.npz")
    args = ap.parse_args()

    train_blocks, _ = _load_blocks(args.train)
    test_blocks, y_test = _load_blocks(args.test)
    n = len(train_blocks)
    dim = train_blocks[0].shape[1]

    concat_test = _l2(np.concatenate(test_blocks, axis=1))
    concat_r1 = image_self_retrieval_score(concat_test, y_test).recall_at_1
    single_r1 = image_self_retrieval_score(_l2(test_blocks[0]), y_test).recall_at_1

    # Fit rotations on TRAIN, freeze, apply to TEST.
    rotations = gpa_rotations(train_blocks)
    gpa_test = _apply(test_blocks, rotations)
    gpa_r1 = image_self_retrieval_score(gpa_test, y_test).recall_at_1

    print(f"{n} models, {dim}-dim each")
    print(f"single model (512d)              R@1={single_r1:.4f}")
    print(f"full concat ({n * dim}d)             R@1={concat_r1:.4f}   <- 100% target")
    retained = 100 * gpa_r1 / concat_r1
    print(f"INDUCTIVE GPA (512d, TRAIN-fit)   R@1={gpa_r1:.4f}   retained={retained:.2f}%")
    print("(rotations fit on the disjoint train split only — nothing fit on test.)")


if __name__ == "__main__":
    main()
