import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from diagnose_sop_seen_gallery_effect import (
    count_outranking_negatives,
    expected_recall_random_negatives,
)


def test_hypergeometric_expected_recall_at_equal_negative_count() -> None:
    expected = expected_recall_random_negatives(
        np.asarray([4, 4, 4]), np.asarray([0, 1, 3]), 2
    )
    assert np.allclose(expected, [1.0, 0.5, 0.0])


def test_outranking_counts_exclude_self_and_keep_positives_fixed() -> None:
    features = torch.tensor(
        [[0.9, 0.43589], [0.0, 1.0], [1.0, 0.0], [0.8, 0.6], [0.0, -1.0]]
    )
    labels = np.asarray([10, 11, 20, 20, 21])
    queries = np.asarray([2, 3], dtype=np.int64)
    fit = np.asarray([0, 1, 4], dtype=np.int64)
    result = count_outranking_negatives(features, labels, queries, fit, block_rows=1)
    assert result["seen_pool_size"].tolist() == [3, 3]
    assert result["unseen_pool_size"].tolist() == [0, 0]
    assert result["seen_outranking"].tolist() == [1, 1]
    assert result["unseen_outranking"].tolist() == [0, 0]
