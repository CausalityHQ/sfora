"""Deterministic gallery-density correction for cosine retrieval."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sfora.reciprocal_reranking import cosine_topk


@dataclass(frozen=True)
class HubnessSummary:
    """Incoming raw top-1 counts and their population skewness."""

    counts: np.ndarray
    maximum_count: int
    skewness: float


def _validate_embeddings(name: str, value: np.ndarray) -> None:
    if not isinstance(value, np.ndarray) or value.ndim != 2:
        raise ValueError(f"{name} must be a rank-2 NumPy array")
    if not np.issubdtype(value.dtype, np.floating):
        raise ValueError(f"{name} must have a floating dtype")
    if value.shape[0] == 0 or value.shape[1] == 0:
        raise ValueError(f"{name} must be nonempty")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")


def _validate_block_size(block_size: int) -> None:
    if type(block_size) is not int or block_size <= 0:
        raise ValueError("block_size must be a positive integer")


def gallery_local_density(
    gallery: np.ndarray, k: int | str, block_size: int = 256
) -> np.ndarray:
    """Return mean non-self gallery cosine among top-k or all neighbors."""

    _validate_embeddings("gallery", gallery)
    _validate_block_size(block_size)
    count = gallery.shape[0]
    if count < 2:
        raise ValueError("no non-self gallery items are available")
    if k == "all":
        summed = np.sum(gallery, axis=0, dtype=np.float32)
        totals = np.asarray(gallery, dtype=np.float32) @ summed
        return np.asarray((totals - 1.0) / (count - 1), dtype=np.float32)
    if type(k) is not int or k <= 0:
        if isinstance(k, str):
            raise ValueError("k must be an integer or 'all'")
        raise ValueError("k must be a positive integer")
    if k > count - 1:
        raise ValueError("k exceeds the available non-self gallery items")

    result = np.empty(count, dtype=np.float32)
    gallery32 = np.asarray(gallery, dtype=np.float32)
    gallery_t = np.asarray(gallery32.T, dtype=np.float32)
    for start in range(0, count, block_size):
        stop = min(start + block_size, count)
        scores = gallery32[start:stop] @ gallery_t
        rows = np.arange(stop - start)
        scores[rows, start + rows] = -np.inf
        selected = np.partition(scores, scores.shape[1] - k, axis=1)[:, -k:]
        result[start:stop] = np.mean(selected, axis=1, dtype=np.float32)
    return result


def corrected_cosine_top1(
    queries: np.ndarray,
    gallery: np.ndarray,
    density: np.ndarray,
    block_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact top-1 under ``2*cosine - gallery_density``."""

    _validate_embeddings("queries", queries)
    _validate_embeddings("gallery", gallery)
    _validate_block_size(block_size)
    if queries.shape[1] != gallery.shape[1]:
        raise ValueError("query and gallery embedding dimensions differ")
    if (
        not isinstance(density, np.ndarray)
        or density.shape != (gallery.shape[0],)
        or not np.issubdtype(density.dtype, np.floating)
        or not np.isfinite(density).all()
    ):
        raise ValueError("density must be a finite floating gallery-length vector")

    indices = np.empty(queries.shape[0], dtype=np.int64)
    top_scores = np.empty(queries.shape[0], dtype=np.float32)
    gallery_t = np.asarray(gallery.T, dtype=np.float32)
    density32 = np.asarray(density, dtype=np.float32)
    for start in range(0, queries.shape[0], block_size):
        stop = min(start + block_size, queries.shape[0])
        scores = 2.0 * (np.asarray(queries[start:stop], dtype=np.float32) @ gallery_t)
        scores -= density32
        block_indices = np.argmax(scores, axis=1)
        indices[start:stop] = block_indices
        top_scores[start:stop] = scores[np.arange(stop - start), block_indices]
    return indices, top_scores


def top1_hubness(
    queries: np.ndarray, gallery: np.ndarray, block_size: int = 256
) -> HubnessSummary:
    """Summarize incoming raw cosine top-1 assignments."""

    _validate_embeddings("queries", queries)
    _validate_embeddings("gallery", gallery)
    if queries.shape[1] != gallery.shape[1]:
        raise ValueError("query and gallery embedding dimensions differ")
    _, indices = cosine_topk(queries, gallery, 1, block_size)
    counts = np.bincount(indices[:, 0], minlength=gallery.shape[0]).astype(np.int64)
    centered = counts.astype(np.float64) - float(np.mean(counts))
    standard_deviation = float(np.sqrt(np.mean(centered * centered)))
    skewness = (
        0.0
        if standard_deviation == 0.0
        else float(np.mean(centered**3) / standard_deviation**3)
    )
    return HubnessSummary(
        counts=counts,
        maximum_count=int(np.max(counts)),
        skewness=skewness,
    )

