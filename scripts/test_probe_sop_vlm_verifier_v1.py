import math

import pytest
import torch
from probe_sop_vlm_verifier_v1 import K, choose, fuse_perm, gated_rerank, verdict, yes_no_ids


def test_gated_rerank_touches_only_gated_rows_and_first_k():
    ranked = torch.arange(20).view(2, 10)
    cos = torch.tensor([[0.9, 0.89, 0.88] + [0.5] * 7] * 2)
    v = torch.tensor([[-5.0, 0.0, 5.0], [math.nan] * 3])
    gate = torch.tensor([True, False])
    out = gated_rerank(ranked, cos, v, gate, 2.0)
    assert out[0, :K].tolist() == [2, 1, 0]
    assert torch.equal(out[0, K:], ranked[0, K:])
    assert torch.equal(out[1], ranked[1])
    with pytest.raises(ValueError):
        gated_rerank(ranked, cos, v, torch.tensor([True, True]), 1.0)


def test_alpha_inf_is_verifier_only_and_small_alpha_keeps_cosine():
    cos = torch.tensor([[0.9, 0.5, 0.1]])
    v = torch.tensor([[0.0, 1.0, 2.0]])
    assert fuse_perm(cos, v, math.inf).tolist() == [[2, 1, 0]]
    assert fuse_perm(cos, v, 0.5).tolist() == [[0, 1, 2]]
    # ties in v keep the base order
    assert fuse_perm(cos, torch.tensor([[1.0, 1.0, 0.0]]), math.inf).tolist() == [[0, 1, 2]]


class FakeTokenizer:
    vocab = {"yes": [5], "Yes": [6], " yes": [7], " Yes": [8, 9], "no": [1], "No": [2], " no": [3], " No": [4]}

    def encode(self, text, add_special_tokens=False):
        return self.vocab[text]


def test_yes_no_extraction_takes_max_over_single_token_variants():
    ids = yes_no_ids(FakeTokenizer())
    assert ids == {"yes": {"yes": 5, "Yes": 6, " yes": 7}, "no": {"no": 1, "No": 2, " no": 3, " No": 4}}
    logits = torch.zeros(10)
    logits[6], logits[3], logits[8] = 4.0, 1.5, 99.0  # 8 belongs to a multi-token variant: ignored
    assert verdict(logits, ids) == pytest.approx(2.5)


def test_missing_token_raises():
    class Bad(FakeTokenizer):
        vocab = {**FakeTokenizer.vocab, "no": [1, 2], "No": [1, 2], " no": [1, 2], " No": [1, 2]}

    with pytest.raises(ValueError):
        yes_no_ids(Bad())


def test_grid_ties_prefer_smaller_alpha_then_smaller_tau():
    grid = {(0.05, 2.0): 90.0, (0.05, 0.5): 90.0, (0.02, math.inf): 90.0, (0.10, 0.5): 90.0, (0.02, 1.0): 80.0}
    assert choose(grid) == (0.05, 0.5)
    grid[(0.02, 0.5)] = 90.0
    assert choose(grid) == (0.02, 0.5)
    grid[(0.10, math.inf)] = 91.0
    assert choose(grid) == (0.10, math.inf)
