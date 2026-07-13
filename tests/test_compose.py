"""Tests for the composable projection-brick API (sfora.compose)."""

from __future__ import annotations

import numpy as np

from sfora.compose import (
    Head,
    Identity,
    Join,
    L2Normalize,
    Pca,
    Pipeline,
    Projection,
    RetrievalReport,
    compare,
    evaluate,
    grid,
)


def _two_class_data(
    per_class: int = 8, dim: int = 6, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    centre = np.zeros(dim)
    centre[0] = 3.0
    a = rng.normal(centre, 0.25, size=(per_class, dim))
    b = rng.normal(-centre, 0.25, size=(per_class, dim))
    embeddings = np.concatenate([a, b], axis=0)
    labels = np.array([0] * per_class + [1] * per_class)
    return embeddings, labels


def test_leaf_bricks_shapes_and_values() -> None:
    x, y = _two_class_data()
    assert np.allclose(Identity().fit(x, y).transform(x), x)

    normed = L2Normalize().fit(x, y).transform(x)
    assert np.allclose(np.linalg.norm(normed, axis=1), 1.0)

    reduced = Pca(dim=3).fit(x, y).transform(x)
    assert reduced.shape == (x.shape[0], 3)
    # PCA of a clean two-cluster set keeps the clusters separable on axis 0.
    assert reduced[y == 0, 0].mean() * reduced[y == 1, 0].mean() < 0


def test_pca_dim_capped_to_rank() -> None:
    x, y = _two_class_data(per_class=4, dim=5)
    reduced = Pca(dim=999).fit(x, y).transform(x)
    assert reduced.shape[1] <= x.shape[1]


def test_pipeline_composes_in_order() -> None:
    x, y = _two_class_data()
    pipe = Pipeline([Pca(dim=4), L2Normalize()])
    out = pipe.fit(x, y).transform(x)
    assert out.shape == (x.shape[0], 4)
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0)


def test_join_concat_stacks_branch_dims() -> None:
    x, y = _two_class_data()
    joined = Join("concat", [Pca(dim=3), Pca(dim=2), Identity()]).fit(x, y).transform(x)
    assert joined.shape == (x.shape[0], 3 + 2 + x.shape[1])
    assert np.allclose(np.linalg.norm(joined, axis=1), 1.0)


def test_join_mean_and_aligned_mean_keep_dim() -> None:
    x, y = _two_class_data()
    for join in (Join("mean", [Pca(dim=4), Pca(dim=4)]), Join("aligned_mean", [Pca(4), Pca(4)])):
        joined = join.fit(x, y).transform(x)
        assert joined.shape == (x.shape[0], 4)
        assert np.allclose(np.linalg.norm(joined, axis=1), 1.0)


def test_evaluate_returns_full_metric_report() -> None:
    train = _two_class_data(seed=1)
    test = _two_class_data(seed=2)
    report = evaluate(Pca(dim=4), train=train, test=test, name="pca4")
    assert isinstance(report, RetrievalReport)
    assert report.name == "pca4"
    assert report.output_dim == 4
    # Well-separated clusters -> near-perfect retrieval.
    assert report.recall_at_1 > 0.9
    assert 0.0 <= report.map_at_r <= 1.0


def test_compare_ranks_best_first_and_grid_expands() -> None:
    train = _two_class_data(seed=1)
    test = _two_class_data(seed=2)
    candidates: dict[str, Projection] = {
        "identity": Identity(),
        "l2": L2Normalize(),
        **grid("pca", Pca, {"dim": [2, 4]}),
    }
    assert "pca[dim=2]" in candidates and "pca[dim=4]" in candidates
    ranking = compare(candidates, train=train, test=test, rank_by="recall_at_1")
    assert len(ranking) == len(candidates)
    scores = [report.recall_at_1 for report in ranking]
    assert scores == sorted(scores, reverse=True)


def test_bricks_satisfy_projection_protocol() -> None:
    for brick in (
        Identity(),
        L2Normalize(),
        Pca(2),
        Pipeline([Identity()]),
        Join("concat", [Identity()]),
    ):
        assert isinstance(brick, Projection)


def test_head_brick_trains_and_transforms() -> None:
    # A trainable head on frozen embeddings — small + fast.
    train = _two_class_data(per_class=8, seed=1)
    head = Head(objective="triplet", steps=10, seed=0)
    projected = head.fit(*train).transform(train[0])
    assert projected.shape[0] == train[0].shape[0]
    assert np.isfinite(projected).all()
