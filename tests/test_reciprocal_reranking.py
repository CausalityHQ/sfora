from __future__ import annotations

import numpy as np
import pytest

from sfora.reciprocal_reranking import (
    cosine_topk,
    gallery_reciprocal_sets,
    rerank_queries,
)


def _unit(rows: list[list[float]]) -> np.ndarray:
    values = np.asarray(rows, dtype=np.float32)
    return values / np.linalg.norm(values, axis=1, keepdims=True)


def test_cosine_topk_is_exact_blockwise_and_breaks_ties_by_index() -> None:
    gallery = _unit([[1, 0], [1, 0], [0, 1], [-1, 0]])
    queries = _unit([[1, 0], [0, 1]])

    scores, indices = cosine_topk(queries, gallery, k=3, block_size=1)

    np.testing.assert_array_equal(indices, [[0, 1, 2], [2, 0, 1]])
    np.testing.assert_allclose(scores, [[1, 1, 0], [1, 0, 0]], atol=1e-7)


def test_gallery_reciprocal_sets_exclude_self_and_require_mutual_topk() -> None:
    gallery = _unit([[1, 0], [0.98, 0.2], [0, 1], [-1, 0]])

    reciprocal = gallery_reciprocal_sets(gallery, k=1, block_size=2)

    assert reciprocal == ((1,), (0,), (), ())


def test_reranker_uses_query_specific_reciprocity_and_jaccard() -> None:
    gallery = _unit([[1, 0], [0.98, 0.2], [0.8, 0.6], [0, 1]])
    query = _unit([[0.9, 0.435]])

    result = rerank_queries(
        query,
        gallery,
        k=2,
        candidate_depth=4,
        blend=1.0,
        block_size=2,
    )

    assert result.raw_indices.shape == (1, 4)
    assert result.reranked_indices.shape == (1, 4)
    assert result.structural_scores.shape == (1, 4)
    assert result.reranked_indices[0, 0] == 0
    gallery_zero_column = int(np.flatnonzero(result.raw_indices[0] == 0)[0])
    assert result.structural_scores[0, gallery_zero_column] == pytest.approx(1.0)
    assert np.all(result.structural_scores >= 0.0)
    assert np.all(result.structural_scores <= 1.0)


def test_zero_blend_is_identical_to_raw_cosine_ranking() -> None:
    gallery = _unit([[1, 0], [0.7, 0.7], [0, 1], [-1, 0]])
    queries = _unit([[0.8, 0.6], [0.1, 0.9]])

    result = rerank_queries(
        queries,
        gallery,
        k=2,
        candidate_depth=4,
        blend=0.0,
        block_size=1,
    )

    np.testing.assert_array_equal(result.reranked_indices, result.raw_indices)
    np.testing.assert_allclose(result.reranked_scores, result.raw_scores)


@pytest.mark.parametrize(
    ("queries", "gallery", "match"),
    [
        (np.ones((2, 2), dtype=np.int64), np.ones((2, 2), dtype=np.float32), "floating"),
        (np.asarray([[np.nan, 0]], dtype=np.float32), np.ones((2, 2), dtype=np.float32), "finite"),
        (np.ones((2,), dtype=np.float32), np.ones((2, 2), dtype=np.float32), "rank-2"),
        (np.ones((2, 3), dtype=np.float32), np.ones((2, 2), dtype=np.float32), "dimensions"),
    ],
)
def test_cosine_topk_rejects_invalid_inputs(
    queries: np.ndarray, gallery: np.ndarray, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        cosine_topk(queries, gallery, k=1, block_size=1)


@pytest.mark.parametrize(
    ("k", "candidate_depth", "blend", "match"),
    [
        (0, 2, 0.5, "k"),
        (2, 1, 0.5, "candidate_depth"),
        (1, 3, 0.5, "candidate_depth"),
        (1, 2, -0.1, "blend"),
        (1, 2, 1.1, "blend"),
    ],
)
def test_reranker_rejects_invalid_parameters(
    k: int, candidate_depth: int, blend: float, match: str
) -> None:
    values = _unit([[1, 0], [0, 1]])
    with pytest.raises(ValueError, match=match):
        rerank_queries(
            values[:1],
            values,
            k=k,
            candidate_depth=candidate_depth,
            blend=blend,
            block_size=1,
        )
