"""Tie and product exclusion checks for the fit-only member-bank gate."""

import pytest
import torch
from probe_sop_siglip2_member_bank import contains_impostor_in_topk


def test_negative_topk_excludes_all_own_product_rows_and_breaks_ties() -> None:
    scores = torch.tensor([[0.99, 0.98, 0.7, 0.7, 0.7, 0.1]])
    labels = torch.tensor([0, 0, 1, 2, 3, 4])
    queries = torch.tensor([0])
    assert contains_impostor_in_topk(scores, queries, labels, torch.tensor([2]), k=2).tolist() == [
        True
    ]
    assert contains_impostor_in_topk(scores, queries, labels, torch.tensor([3]), k=2).tolist() == [
        True
    ]
    assert contains_impostor_in_topk(scores, queries, labels, torch.tensor([4]), k=2).tolist() == [
        False
    ]


def test_negative_topk_rejects_same_product_impostor() -> None:
    with pytest.raises(ValueError, match="impostor"):
        contains_impostor_in_topk(
            torch.tensor([[0.9, 0.8, 0.1]]),
            torch.tensor([0]),
            torch.tensor([0, 0, 1]),
            torch.tensor([1]),
            k=1,
        )
