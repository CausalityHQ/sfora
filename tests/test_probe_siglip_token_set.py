from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

_SCRIPT = Path(__file__).parents[1] / "scripts" / "probe_siglip_token_set.py"
_SPEC = importlib.util.spec_from_file_location("probe_siglip_token_set", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_select_attention_tokens_uses_lowest_index_for_ties() -> None:
    tokens = torch.tensor(
        [
            [
                [10.0, 0.0],
                [20.0, 0.0],
                [30.0, 0.0],
                [40.0, 0.0],
            ]
        ]
    )
    attention = torch.tensor([[0.4, 0.4, 0.1, 0.1]])

    selected, weights, indices = _MODULE.select_attention_tokens(tokens, attention, top_k=3)

    assert indices.tolist() == [[0, 1, 2]]
    assert selected[:, :, 0].tolist() == [[10.0, 20.0, 30.0]]
    torch.testing.assert_close(weights.sum(dim=1), torch.ones(1))


def test_score_token_sets_reports_pooled_set_and_fixed_hybrid_recall() -> None:
    # Two classes, two images each. Global descriptors swap the nearest neighbour,
    # while one localized token is class-specific and repairs every query.
    labels = torch.tensor([0, 0, 1, 1])
    global_embeddings = torch.nn.functional.normalize(
        torch.tensor(
            [
                [1.0, 0.0],
                [-1.0, 0.0],
                [0.9, 0.1],
                [-0.9, 0.1],
            ]
        ),
        dim=-1,
    )
    token_sets = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0]],
            [[-1.0, 0.0], [0.0, -1.0]],
            [[-1.0, 0.0], [0.0, -1.0]],
        ]
    )
    weights = torch.full((4, 2), 0.5)

    result = _MODULE.score_token_sets(
        global_embeddings,
        token_sets,
        weights,
        labels,
        set_weight=0.75,
        query_block=2,
    )

    assert result == {"pooled_recall_at_1": 0.0, "set_recall_at_1": 1.0, "hybrid_recall_at_1": 1.0}


def test_train_holdout_rejects_test_or_wrong_class_partition() -> None:
    with pytest.raises(ValueError, match="train split"):
        _MODULE.validate_train_holdout(
            split="test",
            labels=torch.tensor([82, 82, 83, 83]),
        )
    with pytest.raises(ValueError, match="82 through 97"):
        _MODULE.validate_train_holdout(
            split="train",
            labels=torch.tensor([81, 81, 82, 82]),
        )
