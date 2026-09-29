"""Mergeable rank-8 updates for the frozen lower SigLIP2 Large attention stack."""

import math

import torch
from torch import nn
from torch.nn.utils import parametrize


class LowRankWeight(nn.Module):
    def __init__(self, weight, rank):
        super().__init__()
        assert weight.ndim == 2 and weight.dtype == torch.float32
        assert min(weight.shape) >= rank and weight.device.type == "cpu"
        self.A = nn.Parameter(torch.empty(rank, weight.shape[1]))
        self.B = nn.Parameter(torch.zeros(weight.shape[0], rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, weight):
        assert weight.dtype == self.A.dtype == self.B.dtype == torch.float32
        with torch.autocast(weight.device.type, enabled=False):
            return weight + self.B @ self.A


def install(vision, *, rank=8, seed=179032, width=1024):
    """Install exactly 24 zero-output adapters; original lower weights stay frozen."""
    assert rank == 8 and len(vision.encoder.layers) == 24
    sites = []
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        for layer in vision.encoder.layers[:12]:
            for name in ("q_proj", "v_proj"):
                module = getattr(layer.self_attn, name)
                original = module.weight
                assert isinstance(module, nn.Linear)
                assert original.shape == (width, width) and not original.requires_grad
                assert original.dtype == torch.float32 and original.device.type == "cpu"
                parametrize.register_parametrization(module, "weight", LowRankWeight(original, rank))
                assert not module.parametrizations.weight.original.requires_grad
                sites.append((module, "weight"))
    assert len(sites) == 24
    return sites


@torch.no_grad()
def merge(sites):
    for module, name in sites:
        expected = getattr(module, name).detach().clone()
        parametrize.remove_parametrizations(module, name, leave_parametrized=True)
        assert torch.equal(getattr(module, name), expected)
