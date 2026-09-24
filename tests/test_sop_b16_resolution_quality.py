"""Check exact leave-one-out and ordinal ties in the resolution quality screen."""

import torch

from sfora.sop_evaluation import score_gallery_r1


def test_full_gallery_float_and_packed_use_first_ordinal_after_self_exclusion():
    features = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    labels = torch.tensor([1, 1, 2, 2, 2], dtype=torch.int64)
    queries = torch.tensor([0, 2, 3], dtype=torch.int64)
    expected = [1, 0, 4]
    for inverse_norms in (None, torch.ones(5, dtype=torch.float16)):
        result = score_gallery_r1(features, labels, queries, inverse_norms=inverse_norms)
        assert result["nearest_ordinals"] == expected
        assert result["per_query_r1"] == [1, 0, 1]
        assert result["recall_at_1"] == 2 / 3
