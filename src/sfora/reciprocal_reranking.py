"""Deterministic reciprocal-neighborhood re-ranking for cosine retrieval."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RerankResult:
    """Top candidate windows before and after structural re-ranking."""

    raw_scores: np.ndarray
    raw_indices: np.ndarray
    structural_scores: np.ndarray
    reranked_scores: np.ndarray
    reranked_indices: np.ndarray


def _validate_embeddings(name: str, value: np.ndarray) -> None:
    if not isinstance(value, np.ndarray) or value.ndim != 2:
        raise ValueError(f"{name} must be a rank-2 NumPy array")
    if not np.issubdtype(value.dtype, np.floating):
        raise ValueError(f"{name} must have a floating dtype")
    if value.shape[0] == 0 or value.shape[1] == 0:
        raise ValueError(f"{name} must be nonempty")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")


def _row_topk(scores: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Return exact top-k with deterministic index tie-breaking."""

    partition = np.argpartition(scores, scores.size - k)[-k:]
    threshold = np.min(scores[partition])
    better = np.flatnonzero(scores > threshold)
    tied = np.flatnonzero(scores == threshold)
    needed = k - better.size
    chosen = np.concatenate((better, tied[:needed]))
    order = np.lexsort((chosen, -scores[chosen]))
    indices = chosen[order]
    return scores[indices], indices


def _cosine_topk(
    queries: np.ndarray,
    gallery: np.ndarray,
    *,
    k: int,
    block_size: int,
    exclude_aligned_self: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if type(k) is not int or k <= 0:
        raise ValueError("k must be a positive integer")
    if type(block_size) is not int or block_size <= 0:
        raise ValueError("block_size must be a positive integer")
    available = gallery.shape[0] - (1 if exclude_aligned_self else 0)
    if k > available:
        raise ValueError("k exceeds the available gallery items")
    if exclude_aligned_self and queries.shape != gallery.shape:
        raise ValueError("aligned self-exclusion requires equal query/gallery shapes")

    all_scores = np.empty((queries.shape[0], k), dtype=np.float32)
    all_indices = np.empty((queries.shape[0], k), dtype=np.int64)
    gallery_t = np.asarray(gallery.T, dtype=np.float32)
    for start in range(0, queries.shape[0], block_size):
        stop = min(start + block_size, queries.shape[0])
        block = np.asarray(queries[start:stop], dtype=np.float32) @ gallery_t
        if exclude_aligned_self:
            rows = np.arange(stop - start)
            block[rows, start + rows] = -np.inf
        for local_row, row in enumerate(block):
            scores, indices = _row_topk(row, k)
            all_scores[start + local_row] = scores
            all_indices[start + local_row] = indices
    return all_scores, all_indices


def cosine_topk(
    queries: np.ndarray,
    gallery: np.ndarray,
    k: int,
    block_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute exact cosine top-k in bounded memory.

    Inputs are expected to be L2-normalized by the caller. Ties are resolved by
    the lower gallery index, independent of block size.
    """

    _validate_embeddings("queries", queries)
    _validate_embeddings("gallery", gallery)
    if queries.shape[1] != gallery.shape[1]:
        raise ValueError("query and gallery embedding dimensions differ")
    return _cosine_topk(
        queries,
        gallery,
        k=k,
        block_size=block_size,
        exclude_aligned_self=False,
    )


def _gallery_reciprocal_data(
    gallery: np.ndarray, *, k: int, block_size: int
) -> tuple[tuple[tuple[int, ...], ...], np.ndarray]:
    scores, indices = _cosine_topk(
        gallery,
        gallery,
        k=k,
        block_size=block_size,
        exclude_aligned_self=True,
    )
    thresholds = scores[:, -1]
    reciprocal: list[tuple[int, ...]] = []
    for row_scores, row_indices in zip(scores, indices, strict=True):
        members = sorted(
            int(index)
            for score, index in zip(row_scores, row_indices, strict=True)
            if score >= thresholds[index]
        )
        reciprocal.append(tuple(members))
    return tuple(reciprocal), thresholds


def gallery_reciprocal_sets(
    gallery: np.ndarray, k: int, block_size: int = 256
) -> tuple[tuple[int, ...], ...]:
    """Return each gallery item's mutual top-k neighbors."""

    _validate_embeddings("gallery", gallery)
    reciprocal, _ = _gallery_reciprocal_data(
        gallery, k=k, block_size=block_size
    )
    return reciprocal


def _jaccard(left: tuple[int, ...], right: tuple[int, ...]) -> float:
    left_set = set(left)
    right_set = set(right)
    union = left_set | right_set
    if not union:
        return 0.0
    return len(left_set & right_set) / len(union)


def rerank_queries(
    queries: np.ndarray,
    gallery: np.ndarray,
    *,
    k: int,
    candidate_depth: int,
    blend: float,
    block_size: int = 256,
) -> RerankResult:
    """Blend cosine with reciprocal-neighborhood agreement in a top window."""

    _validate_embeddings("queries", queries)
    _validate_embeddings("gallery", gallery)
    if queries.shape[1] != gallery.shape[1]:
        raise ValueError("query and gallery embedding dimensions differ")
    if type(k) is not int or k <= 0:
        raise ValueError("k must be a positive integer")
    if type(candidate_depth) is not int or candidate_depth < k:
        raise ValueError("candidate_depth must be an integer at least k")
    if candidate_depth > gallery.shape[0]:
        raise ValueError("candidate_depth exceeds gallery size")
    if type(blend) is not float or not 0.0 <= blend <= 1.0:
        raise ValueError("blend must be a float in [0, 1]")

    gallery_sets, gallery_thresholds = _gallery_reciprocal_data(
        gallery, k=k, block_size=block_size
    )
    raw_scores, raw_indices = cosine_topk(
        queries, gallery, candidate_depth, block_size
    )
    query_k_scores = raw_scores[:, :k]
    query_k_indices = raw_indices[:, :k]
    structural = np.empty_like(raw_scores, dtype=np.float32)
    reranked_scores = np.empty_like(raw_scores, dtype=np.float32)
    reranked_indices = np.empty_like(raw_indices, dtype=np.int64)

    for row_index in range(queries.shape[0]):
        query_set = tuple(
            sorted(
                int(index)
                for score, index in zip(
                    query_k_scores[row_index], query_k_indices[row_index], strict=True
                )
                if score >= gallery_thresholds[index]
            )
        )
        for column, gallery_index in enumerate(raw_indices[row_index]):
            structural[row_index, column] = _jaccard(
                query_set, gallery_sets[int(gallery_index)]
            )
        blended = (
            (1.0 - blend) * raw_scores[row_index]
            + blend * structural[row_index]
        )
        order = np.lexsort((raw_indices[row_index], -blended))
        reranked_scores[row_index] = blended[order]
        reranked_indices[row_index] = raw_indices[row_index, order]

    return RerankResult(
        raw_scores=raw_scores,
        raw_indices=raw_indices,
        structural_scores=structural,
        reranked_scores=reranked_scores,
        reranked_indices=reranked_indices,
    )
