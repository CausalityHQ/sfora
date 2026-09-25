"""Small asymmetric query/gallery oracle for the frozen In-Shop evaluator."""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "evaluate_inshop_siglip2_official", SCRIPTS / "evaluate_inshop_siglip2_official.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_asymmetric_score_respects_gallery_relevance_and_ordinal_ties() -> None:
    query = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    gallery = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    labels = ("A", "B", "A", "B")
    float_result = MODULE.score_asymmetric(query, gallery, ("A", "B"), labels)
    packed_result = MODULE.score_asymmetric(
        query,
        gallery,
        ("A", "B"),
        labels,
        query_inverse=torch.ones(2),
        gallery_inverse=torch.ones(4),
    )
    assert float_result["per_query_r1"] == [1.0, 0.0]
    assert float_result["recall_at_1"] == 0.5
    assert float_result["map_at_r"] == pytest.approx(0.375)
    assert packed_result == float_result
