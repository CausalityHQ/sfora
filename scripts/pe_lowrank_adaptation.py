#!/usr/bin/env python3
"""Fixed rank32 native PE adaptation mechanics for a bounded research arm."""

import math

import torch
from torch import nn
from torch.nn.utils import parametrize

from pe_core_training import named_training_parameters


class LowRankWeight(nn.Module):
    def __init__(self, weight):
        super().__init__()
        assert weight.dtype == torch.float32 and weight.ndim == 2
        assert min(weight.shape) >= 32 and weight.device.type == "cpu"
        self.A = nn.Parameter(torch.empty(32, weight.shape[1]))
        self.B = nn.Parameter(torch.zeros(weight.shape[0], 32))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, weight):
        assert weight.dtype == self.A.dtype == self.B.dtype == torch.float32
        # alpha=rank=32. Materialize before the native Linear's autocast.
        with torch.autocast(weight.device.type, enabled=False):
            return weight + self.B @ self.A


def install(vision, native_inventory):
    """Call after native freeze_prefix; return the exact 24 registered sites."""
    before = dict(named_training_parameters(vision, "pe"))
    assert {n for n, p in before.items() if p.requires_grad} == set(
        native_inventory["trainable"]
    )
    shapes = {
        "attn.in_proj_weight": (2304, 768),
        "attn.out_proj.weight": (768, 768),
        "mlp.c_fc.weight": (3072, 768),
        "mlp.c_proj.weight": (768, 3072),
    }
    names = sorted(
        f"transformer.resblocks.{i}.{suffix}" for i in range(6, 12) for suffix in shapes
    )
    assert all(n in native_inventory["trainable"] for n in names)
    sites = {}
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(179034)
        for name in names:
            parent, key = name.rsplit(".", 1)
            module = vision.get_submodule(parent)
            original = getattr(module, key)
            suffix = name.split(".", 3)[3]
            assert tuple(original.shape) == shapes[suffix]
            original.requires_grad_(False)
            parametrize.register_parametrization(module, key, LowRankWeight(original))
            sites[name] = (module, key)
    assert len(sites) == 24
    assert all(
        not getattr(m.parametrizations, k).original.requires_grad
        for m, k in sites.values()
    )
    return sites


def parameter_groups(vision, sites):
    factors = [
        p
        for m, k in sites.values()
        for p in getattr(m.parametrizations, k)[0].parameters()
    ]
    assert len(factors) == 48 and sum(p.numel() for p in factors) == 2359296
    factor_ids = {id(p) for p in factors}
    dense = [
        p
        for _, p in named_training_parameters(vision, "pe")
        if p.requires_grad and id(p) not in factor_ids
    ]
    all_ids = [id(p) for p in dense + factors]
    assert len(set(all_ids)) == len(all_ids)
    assert set(all_ids) == {
        id(p) for _, p in named_training_parameters(vision, "pe") if p.requires_grad
    }
    originals = {id(getattr(m.parametrizations, k).original) for m, k in sites.values()}
    assert not originals.intersection(all_ids)
    return [{"params": dense, "lr": 1e-5}, {"params": factors, "lr": 1e-4}]


@torch.no_grad()
def merge(sites):
    for module, key in sites.values():
        expected = getattr(module, key).detach().clone()
        parametrize.remove_parametrizations(module, key, leave_parametrized=True)
        assert torch.equal(getattr(module, key), expected)
