from __future__ import annotations

import pytest
import torch

from sfora.substrate_screen import (
    SubstrateRetrievalError,
    score_frozen_substrate,
    score_frozen_substrate_evidence,
    validate_substrate_holdout,
)


def test_frozen_substrate_scoring_is_exact_leave_one_out() -> None:
    embeddings = torch.nn.functional.normalize(
        torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]]), dim=-1
    )
    labels = torch.tensor([82, 82, 83, 83])
    result = score_frozen_substrate(embeddings, labels, query_block=3)
    assert result.correct == 4
    assert result.queries == 4
    assert result.recall_at_1 == 1.0


def test_frozen_substrate_scoring_rejects_nonfinite_rows() -> None:
    with pytest.raises(ValueError, match="finite"):
        score_frozen_substrate(
            torch.tensor([[1.0, 0.0], [float("nan"), 1.0]]),
            torch.tensor([82, 82]),
            query_block=2,
        )


def test_frozen_substrate_scoring_rejects_nonunit_input() -> None:
    with pytest.raises(ValueError, match="incoming descriptors must have unit norm"):
        score_frozen_substrate(
            torch.tensor([[2.0, 0.0], [0.0, 1.0]]),
            torch.tensor([82, 82]),
            query_block=2,
        )


def test_holdout_rejects_any_nonregistered_surface() -> None:
    labels = torch.repeat_interleave(torch.arange(82, 98), 2)
    validate_substrate_holdout(split="train", labels=labels)
    with pytest.raises(ValueError, match="train split"):
        validate_substrate_holdout(split="test", labels=labels)
    with pytest.raises(ValueError, match="exactly Cars train classes"):
        validate_substrate_holdout(split="train", labels=labels[:-2])


def test_error_evidence_preserves_exact_lowest_index_neighbours() -> None:
    embeddings = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [-1.0, 0.0]]
    )
    labels = torch.tensor([82, 83, 84, 85])
    evidence = score_frozen_substrate_evidence(embeddings, labels, query_block=3)
    assert evidence.metrics.correct == 0
    assert evidence.metrics.queries == 4
    assert evidence.errors == (
        SubstrateRetrievalError(0, 1, 82, 83),
        SubstrateRetrievalError(1, 2, 83, 84),
        SubstrateRetrievalError(2, 1, 84, 83),
        SubstrateRetrievalError(3, 1, 85, 83),
    )
