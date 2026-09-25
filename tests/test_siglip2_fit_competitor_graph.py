"""Frozen fit-only competitor graph ranking boundaries."""

import importlib.util
from pathlib import Path

import torch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/build_sop_siglip2_fit_competitor_graph.py"
SPEC = importlib.util.spec_from_file_location("build_sop_siglip2_fit_competitor_graph", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
ten_negative_neighbors = MODULE.ten_negative_neighbors


def test_ten_negative_neighbors_excludes_product_and_uses_ordinal_ties() -> None:
    labels = torch.tensor([1, 1] + [2] * 12)
    scores = torch.tensor([[1.0, 0.9] + [0.5] * 12])
    neighbors = ten_negative_neighbors(scores, labels, torch.tensor([0]))
    assert neighbors.tolist() == [list(range(2, 12))]
