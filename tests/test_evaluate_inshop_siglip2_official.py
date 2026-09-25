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


def test_packed_inverse_norms_can_change_top_one() -> None:
    query = torch.tensor([[1.0, 0.0]])
    gallery = torch.tensor([[2.0, 0.0], [1.0, 0.0]])
    assert MODULE.score_asymmetric(query, gallery, ("B",), ("A", "B"))["recall_at_1"] == 0
    packed = MODULE.score_asymmetric(
        query,
        gallery,
        ("B",),
        ("A", "B"),
        query_inverse=torch.tensor([1.0]),
        gallery_inverse=torch.tensor([0.25, 1.0]),
    )
    assert packed["recall_at_1"] == 1


def test_loaded_helper_sources_must_match_receipt() -> None:
    assert {path.name for path in MODULE.loaded_helper_paths()} == {
        "train_inshop_siglip2_compact.py",
        "train_sop_siglip2_compact.py",
        "joint_relational_compaction.py",
        "unicom_inshop.py",
    }
    sources = {
        str(Path(module_path).resolve()): MODULE.sha256(Path(module_path))
        for module_path in MODULE.loaded_helper_paths()
    }
    MODULE.verify_loaded_helper_sources(sources)
    path = next(iter(sources))
    sources[path] = "0" * 64
    with pytest.raises(ValueError, match="loaded helper"):
        MODULE.verify_loaded_helper_sources(sources)
