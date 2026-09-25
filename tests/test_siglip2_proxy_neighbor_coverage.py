"""Ordinal tie behavior for the proxy-neighbor preflight."""

import importlib.util
from pathlib import Path

import torch

SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts/probe_sop_siglip2_proxy_neighbor_coverage.py"
)
SPEC = importlib.util.spec_from_file_location("probe_sop_siglip2_proxy_neighbor_coverage", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
proxy_top_four = MODULE.proxy_top_four


def test_proxy_top_four_excludes_self_and_breaks_score_ties_by_ordinal() -> None:
    classifier = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]
    )
    graph = proxy_top_four(classifier)
    assert graph.shape == (6, 4)
    assert graph[0].tolist() == [1, 2, 3, 5]
