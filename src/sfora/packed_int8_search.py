"""Bounded-memory exact CPU search over packed int8 embeddings."""

from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray

from sfora.joint_relational_compaction import PackedInt8Embeddings

_QUERY_BLOCK_ROWS = 64


def _ordered_topk_indexes(
    scores: NDArray[np.float32], ordinals: NDArray[np.int64], k: int
) -> NDArray[np.intp]:
    """Select exact top-k without sorting every score in a large block."""

    if k == len(scores):
        return np.lexsort((ordinals, -scores))
    cutoff = np.partition(scores, len(scores) - k)[len(scores) - k]
    better = np.flatnonzero(scores > cutoff)
    equal = np.flatnonzero(scores == cutoff)
    needed = k - len(better)
    if len(equal) > needed:
        equal = equal[np.argpartition(ordinals[equal], needed - 1)[:needed]]
    chosen = np.concatenate((better, equal))
    return chosen[np.lexsort((ordinals[chosen], -scores[chosen]))]


class CpuPackedInt8Gallery:
    """Owned CPU gallery with deterministic blockwise exact cosine top-k."""

    def __init__(self, embeddings: PackedInt8Embeddings, *, block_rows: int) -> None:
        if (
            type(embeddings) is not PackedInt8Embeddings
            or type(block_rows) is not int
            or block_rows < 1
        ):
            raise ValueError("CPU packed gallery authority differs")
        self._codes = embeddings.codes.clone().contiguous()
        self._inverse_norms = embeddings.inverse_norms.clone().contiguous()
        self._block_rows = block_rows

    @classmethod
    def open_packed(
        cls,
        embeddings: PackedInt8Embeddings,
        *,
        block_rows: int = 65_536,
    ) -> CpuPackedInt8Gallery:
        """Own a validated gallery without expanding its persistent storage."""

        return cls(embeddings, block_rows=block_rows)

    def search_packed(
        self,
        queries: PackedInt8Embeddings,
        *,
        k: int = 10,
    ) -> tuple[NDArray[np.int64], NDArray[np.float32]]:
        """Return exact cosine top-k, breaking equal scores by lower ordinal."""

        if (
            type(queries) is not PackedInt8Embeddings
            or queries.codes.shape[1] != self._codes.shape[1]
            or type(k) is not int
            or k < 1
            or k > self._codes.shape[0]
            or not bool(torch.isfinite(queries.inverse_norms).all())
            or bool((queries.inverse_norms <= 0).any())
        ):
            raise ValueError("CPU packed query authority differs")
        query_count = queries.codes.shape[0]
        best_ordinals = np.empty((query_count, 0), dtype=np.int64)
        best_scores = np.empty((query_count, 0), dtype=np.float32)
        with torch.no_grad():
            query_codes = queries.codes.float()
            query_norms = queries.inverse_norms.float().unsqueeze(1)
            for start in range(0, self._codes.shape[0], self._block_rows):
                stop = min(start + self._block_rows, self._codes.shape[0])
                gallery_codes = self._codes[start:stop].float().T
                gallery_norms = self._inverse_norms[start:stop].float().unsqueeze(0)
                keep = min(k, best_ordinals.shape[1] + stop - start)
                next_ordinals = np.empty((query_count, keep), dtype=np.int64)
                next_scores = np.empty((query_count, keep), dtype=np.float32)
                for query_start in range(0, query_count, _QUERY_BLOCK_ROWS):
                    query_stop = min(query_start + _QUERY_BLOCK_ROWS, query_count)
                    integer_dots = query_codes[query_start:query_stop] @ gallery_codes
                    block_scores = (
                        integer_dots * query_norms[query_start:query_stop] * gallery_norms
                    ).numpy()
                    block_ordinals = np.broadcast_to(
                        np.arange(start, stop, dtype=np.int64), block_scores.shape
                    )
                    candidate_ordinals = np.concatenate(
                        (best_ordinals[query_start:query_stop], block_ordinals), axis=1
                    )
                    candidate_scores = np.concatenate(
                        (best_scores[query_start:query_stop], block_scores), axis=1
                    )
                    selected = np.stack(
                        [
                            _ordered_topk_indexes(
                                candidate_scores[row], candidate_ordinals[row], keep
                            )
                            for row in range(query_stop - query_start)
                        ]
                    )
                    next_ordinals[query_start:query_stop] = np.take_along_axis(
                        candidate_ordinals, selected, axis=1
                    )
                    next_scores[query_start:query_stop] = np.take_along_axis(
                        candidate_scores, selected, axis=1
                    )
                best_ordinals, best_scores = next_ordinals, next_scores
        return best_ordinals, best_scores
