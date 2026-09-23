"""The SOP screen must exclude each query's identical gallery row."""

from __future__ import annotations

import torch

from sfora.sop_evaluation import score_symmetric


def test_score_symmetric_excludes_self_before_recall() -> None:
    values = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [0.9, 0.1], [0.1, 0.9]],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 0, 1, 1], dtype=torch.int64)
    result = score_symmetric(values, labels, block_rows=2)
    assert result["recall_at_1"] == 0.0
    assert result["map_at_r"] == 0.0


def test_score_symmetric_finds_correct_other_product() -> None:
    values = torch.tensor(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 0, 1, 1], dtype=torch.int64)
    result = score_symmetric(values, labels, block_rows=2)
    assert result["recall_at_1"] == 1.0
    assert result["map_at_r"] == 1.0


def test_packed_score_uses_lower_gallery_ordinal_for_ties() -> None:
    codes = torch.tensor(
        [[100.0, 0.0], [100.0, 0.0], [100.0, 0.0], [0.0, 100.0]],
        dtype=torch.float32,
    )
    inverse_norms = torch.full((4,), 0.01, dtype=torch.float16)
    labels = torch.tensor([0, 1, 0, 1], dtype=torch.int64)
    result = score_symmetric(codes, labels, block_rows=2, inverse_norms=inverse_norms)
    assert result["recall_at_1"] == 0.25
    assert result["map_at_r"] == 0.25


def test_map_at_r_counts_only_first_r_for_multiple_positives() -> None:
    values = torch.tensor(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.8, 0.2], [-1.0, 0.0]],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 0, 0, 1, 1], dtype=torch.int64)
    result = score_symmetric(values, labels, block_rows=2)
    assert result["max_relevant"] == 2
    assert result["per_query_r1"][0] == 1.0
    assert result["per_query_ap"][0] == 0.5
