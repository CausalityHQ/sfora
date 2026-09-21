"""Bounded-memory exact CPU search over packed int8 embeddings."""

from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray

from sfora.joint_relational_compaction import PackedInt8Embeddings


class CpuPackedInt8Gallery:
    """Owned CPU gallery with deterministic blockwise exact cosine top-k."""

    def __init__(self, embeddings: PackedInt8Embeddings, *, block_rows: int) -> None:
        if (
            type(embeddings) is not PackedInt8Embeddings
            or type(block_rows) is not int
            or block_rows < 1
            or embeddings.codes.shape[0] < 10
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
                integer_dots = query_codes @ self._codes[start:stop].float().T
                block_scores = (
                    integer_dots
                    * query_norms
                    * self._inverse_norms[start:stop].float().unsqueeze(0)
                ).numpy()
                block_ordinals = np.broadcast_to(
                    np.arange(start, stop, dtype=np.int64), block_scores.shape
                )
                candidate_ordinals = np.concatenate((best_ordinals, block_ordinals), axis=1)
                candidate_scores = np.concatenate((best_scores, block_scores), axis=1)
                keep = min(k, candidate_scores.shape[1])
                selected = np.stack(
                    [
                        np.lexsort((candidate_ordinals[row], -candidate_scores[row]))[:keep]
                        for row in range(query_count)
                    ]
                )
                best_ordinals = np.take_along_axis(candidate_ordinals, selected, axis=1)
                best_scores = np.take_along_axis(candidate_scores, selected, axis=1)
        return np.ascontiguousarray(best_ordinals), np.ascontiguousarray(best_scores)
