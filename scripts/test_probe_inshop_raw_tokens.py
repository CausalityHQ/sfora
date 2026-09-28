import torch
from probe_inshop_spatial_parts import token_score


def test_token_score_is_symmetric_and_independent_of_positions():
    tokens = torch.eye(1024)[:256]
    permuted = tokens.flip(0)
    assert token_score(tokens, permuted) == token_score(permuted, tokens) == 1.0
    assert token_score(tokens, torch.eye(1024)[256:512]) == 0.0
