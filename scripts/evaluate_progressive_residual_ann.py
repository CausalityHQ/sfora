#!/usr/bin/env python3
"""Evaluate progressive residual retrieval against authenticated ANN truth."""

from __future__ import annotations

import numpy as np


def candidate_containment_hits(
    candidate_ordinals: np.ndarray,
    truth_ordinals: np.ndarray,
    *,
    candidate_width: int,
    truth_width: int,
) -> np.ndarray:
    """Count truth IDs present anywhere in each bounded candidate prefix."""

    if (
        type(candidate_ordinals) is not np.ndarray
        or candidate_ordinals.dtype != np.int64
        or candidate_ordinals.ndim != 2
        or candidate_ordinals.shape[0] < 1
        or candidate_ordinals.shape[1] < 1
        or not candidate_ordinals.flags.c_contiguous
        or type(truth_ordinals) is not np.ndarray
        or truth_ordinals.dtype != np.int64
        or truth_ordinals.ndim != 2
        or truth_ordinals.shape[0] != candidate_ordinals.shape[0]
        or truth_ordinals.shape[1] < 1
        or not truth_ordinals.flags.c_contiguous
        or type(candidate_width) is not int
        or not 1 <= candidate_width <= candidate_ordinals.shape[1]
        or type(truth_width) is not int
        or not 1 <= truth_width <= truth_ordinals.shape[1]
        or bool((candidate_ordinals < 0).any())
        or bool((truth_ordinals < 0).any())
        or any(
            len(np.unique(row)) != len(row)
            for row in candidate_ordinals
        )
        or any(len(np.unique(row)) != len(row) for row in truth_ordinals)
    ):
        raise ValueError("candidate containment authority differs")
    hits = [
        int(
            np.isin(
                truth_ordinals[index, :truth_width],
                candidate_ordinals[index, :candidate_width],
                assume_unique=True,
            ).sum()
        )
        for index in range(candidate_ordinals.shape[0])
    ]
    return np.asarray(hits, dtype=np.int64)
