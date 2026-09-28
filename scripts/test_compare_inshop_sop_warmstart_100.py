"""Packed held-role scorer preserves gallery ordinal ties."""

import numpy as np
import torch
from compare_inshop_sop_warmstart_100 import packed_quality
from probe_inshop_sop_product_prior import score


def test_packed_role_tie_uses_first_gallery_ordinal():
    values = np.zeros((4, 128), dtype=np.float32)
    values[:, 0] = 1.0
    result = packed_quality(
        values, ("a", "a", "b", "b"), [0, 2], [1, 3], device=torch.device("cpu")
    )
    assert result["per_query_r1"] == [1, 0]
    assert result["recall_at_1"] == 0.5
    assert result["map_at_r"] == 0.5


def test_float_role_tie_uses_first_gallery_ordinal():
    values = torch.ones((4, 3))
    result = score(values, ("a", "a", "b", "b"), [0, 2], [1, 3], device=torch.device("cpu"))
    assert result["per_query_r1"] == [1, 0]
    assert result["map_at_r"] == 0.5
