"""Token readout contracts for the cached SOP architecture falsifier."""

import pytest
import torch
from torch import nn

from sfora.token_slot_readout import MeanTokenReadout, SlotTokenReadout


def test_zero_query_slot_initialization_matches_fitted_mean_readout():
    torch.manual_seed(17)
    mean = MeanTokenReadout(width=8, output_width=6)
    slots = SlotTokenReadout(width=8, output_width=6, slots=4)
    slots.initialize_from_mean(mean, seed=19, query_std=0.0)
    tokens = torch.randn(3, 7, 8)
    torch.testing.assert_close(slots(tokens), mean(tokens), rtol=0, atol=2e-7)
    assert slots.parameter_count == 4 * 8 * 6 + 6 + 4 * 8


def test_slot_readout_is_token_permutation_invariant_and_trainable():
    torch.manual_seed(23)
    mean = MeanTokenReadout(width=8, output_width=6)
    slots = SlotTokenReadout(width=8, output_width=6, slots=3)
    slots.initialize_from_mean(mean, seed=29)
    tokens = torch.randn(3, 9, 8, requires_grad=True)
    output = slots(tokens)
    torch.testing.assert_close(output, slots(tokens[:, [8, 3, 5, 1, 0, 7, 2, 4, 6]]))
    output.square().mean().backward()
    assert tokens.grad is not None and float(tokens.grad.abs().sum()) > 0
    assert slots.queries.grad is not None and float(slots.queries.grad.abs().sum()) > 0


def test_readouts_reject_wrong_token_geometry():
    mean = MeanTokenReadout(width=8, output_width=6)
    slots = SlotTokenReadout(width=8, output_width=6, slots=2)
    for readout in (mean, slots):
        with pytest.raises(ValueError, match="geometry"):
            readout(torch.randn(2, 7, 9))
    with pytest.raises(ValueError, match="geometry"):
        slots.initialize_from_mean(nn.Linear(9, 6), seed=19)
