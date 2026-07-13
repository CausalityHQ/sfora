"""Honest dimensionality reduction of the ensemble pack: fit the fold on the TRAIN
split, freeze it, evaluate on the disjoint-class TEST split.

Loads matching train/test embeddings for several HERD seeds (saved with
`--save-train-embeddings` / `--save-test-embeddings`), builds the feature-concatenation
pack on each split, then compares folds that are fit **only on train**:

- PCA fit on the train concat (unsupervised geometry) -> applied to test
- a supervised metric-learning head trained on the train concat (using train labels)
  -> applied to test

Baselines: the full test concat (100% reference), and the single-model test R@1.
Nothing is fit on the test split. Reports R@1/R@2/R@4/R@8/MAP@R.

    uv run python scripts/train_fit_fold.py --dim 512 \
        --train 'reports/emb/herd_tt_seed*.train.npz' --test 'reports/emb/herd_tt_seed*.test.npz'
"""

from __future__ import annotations

import argparse
import glob

import numpy as np

from sfora.compose import Head, Pca, evaluate
from sfora.image_benchmark import image_self_retrieval_score


def _l2(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def _load_pack(pattern: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit(f"no files match {pattern}")
    blocks: list[np.ndarray] = []
    labels: np.ndarray | None = None
    ids: np.ndarray | None = None
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            blocks.append(_l2(np.asarray(data["embeddings"], dtype=np.float64)))
            lab = np.asarray(data["labels"], dtype=np.int64)
            eid = np.asarray(data["example_ids"]) if "example_ids" in data.files else None
        if labels is None:
            labels, ids = lab, eid
        else:
            if not np.array_equal(labels, lab):
                raise SystemExit(f"label mismatch in {path}")
            # All-or-none example-id agreement (a labels-only match can hide a
            # within-class row permutation that mixes different images).
            if (eid is None) != (ids is None):
                raise SystemExit(f"{path} disagrees on example_ids presence")
            if eid is not None and not np.array_equal(ids, eid):
                raise SystemExit(f"example-id mismatch in {path}")
    concat = _l2(np.concatenate(blocks, axis=1))
    return concat, labels, paths  # type: ignore[return-value]


def _metrics(features: np.ndarray, labels: np.ndarray) -> str:
    m = image_self_retrieval_score(_l2(features), labels, random_state=0)
    return (
        f"R@1={m.recall_at_1:.4f} R@2={m.recall_at_2:.4f} R@4={m.recall_at_4:.4f} "
        f"R@8={m.recall_at_8:.4f} MAP@R={m.map_at_r:.4f}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", default="reports/emb/herd_tt_seed*.train.npz")
    ap.add_argument("--test", default="reports/emb/herd_tt_seed*.test.npz")
    ap.add_argument("--dim", type=int, default=512)
    args = ap.parse_args()

    train_concat, y_train, train_paths = _load_pack(args.train)
    test_concat, y_test, test_paths = _load_pack(args.test)
    # Pair train/test blocks by their per-seed stem (e.g. herd_tt_seed3.train.npz
    # <-> herd_tt_seed3.test.npz) so mismatched or misordered runs can't be joined.
    train_stems = [p.rsplit("/", 1)[-1].replace(".train.npz", "") for p in train_paths]
    test_stems = [p.rsplit("/", 1)[-1].replace(".test.npz", "") for p in test_paths]
    if train_stems != test_stems:
        raise SystemExit(
            "train/test packs do not correspond by seed:\n"
            f"  train {train_stems}\n  test  {test_stems}"
        )
    full_dim = train_concat.shape[1]
    n_models = len(train_paths)
    print(f"{n_models} models | train {train_concat.shape} | test {test_concat.shape}\n")

    # Reference: the full test concat (the 100% target) and single-model test R@1.
    single = image_self_retrieval_score(
        _l2(test_concat[:, : full_dim // n_models]), y_test
    ).recall_at_1
    print(
        f"single model (512d, test)     {_metrics(test_concat[:, : full_dim // n_models], y_test)}"
    )
    print(f"full concat ({full_dim}d, test)   {_metrics(test_concat, y_test)}   <- 100% target")

    # Folds fit ONLY on the train split, then applied frozen to test.
    print(f"\n--- {args.dim}-dim folds fit on TRAIN, evaluated on TEST ---")
    r_pca = evaluate(
        Pca(dim=args.dim),
        train=(train_concat, y_train),
        test=(test_concat, y_test),
        name="pca-on-train",
    )
    print(
        f"PCA (train-fit)               R@1={r_pca.recall_at_1:.4f} R@2={r_pca.recall_at_2:.4f} "
        f"R@4={r_pca.recall_at_4:.4f} R@8={r_pca.recall_at_8:.4f} MAP@R={r_pca.map_at_r:.4f}"
    )

    for objective in ("proxy_anchor", "group_supcon_xbm_radius"):
        head = Head(
            objective=objective,
            steps=200,
            params={
                "output_dimensions": args.dim,
                "learning_rate": 0.001,
                "normalize_embeddings": True,
            },
        )
        rep = evaluate(
            head, train=(train_concat, y_train), test=(test_concat, y_test), name=objective
        )
        print(
            f"head:{objective:<24} R@1={rep.recall_at_1:.4f} R@2={rep.recall_at_2:.4f} "
            f"R@4={rep.recall_at_4:.4f} R@8={rep.recall_at_8:.4f} MAP@R={rep.map_at_r:.4f}"
        )
    del single


if __name__ == "__main__":
    main()
