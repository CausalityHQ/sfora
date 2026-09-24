"""The published UNICOM SOP scorer can rank differently from full cosine."""

import math

import pytest
import torch

from sfora.sop_evaluation import score_symmetric


def test_reference_prefix_euclidean_changes_rank_after_full_vector_normalization():
    values = torch.tensor(
        [
            [0.2, 0.0, math.sqrt(0.96), 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.1, 0.1, 0.0, math.sqrt(0.98)],
            [0.1, 0.2, 0.0, math.sqrt(0.95)],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 0, 1, 1], dtype=torch.int64)

    full_cosine = score_symmetric(values, labels)
    reference = score_symmetric(values, labels, prefix_euclidean_dimensions=2)

    assert full_cosine["per_query_r1"][0] == 1.0
    assert reference["per_query_r1"][0] == 0.0


def test_reference_prefix_rejects_packed_scores_and_invalid_width():
    values = torch.eye(4, dtype=torch.float32)
    labels = torch.tensor([0, 0, 1, 1], dtype=torch.int64)
    inverse_norms = torch.ones(4, dtype=torch.float16)

    with pytest.raises(ValueError, match="prefix"):
        score_symmetric(values, labels, prefix_euclidean_dimensions=4)
    with pytest.raises(ValueError, match="prefix"):
        score_symmetric(
            values,
            labels,
            inverse_norms=inverse_norms,
            prefix_euclidean_dimensions=2,
        )
