#!/usr/bin/env python3
"""One CPU check for zero-B gradients, autocast and materialized merge."""

import io

import torch
from torch import nn
from torch.nn.utils import parametrize

from pe_lowrank_adaptation import LowRankWeight


def main():
    torch.manual_seed(179034)
    for bare in (False, True):
        module = nn.Module() if bare else nn.Linear(96, 64, bias=False)
        name = "in_proj_weight" if bare else "weight"
        if bare:
            module.register_parameter(name, nn.Parameter(torch.randn(64, 96)))
        original = getattr(module, name).detach().clone()
        keys = set(module.state_dict())
        original_parameter = getattr(module, name)
        original_parameter.requires_grad_(False)
        parametrization = LowRankWeight(original)
        parametrize.register_parametrization(module, name, parametrization)
        assert torch.equal(getattr(module, name), original)
        x = torch.randn(4, 96)
        output = x @ getattr(module, name).T
        output.square().sum().backward()
        assert parametrization.A.grad is not None and not parametrization.A.grad.any()
        assert parametrization.B.grad is not None and parametrization.B.grad.norm() > 0
        assert original_parameter.grad is None
        with torch.no_grad():
            parametrization.B.copy_(torch.randn_like(parametrization.B) * 0.001)
        module.zero_grad(set_to_none=True)
        (x @ getattr(module, name).T).square().sum().backward()
        assert parametrization.A.grad.norm() > 0 and parametrization.B.grad.norm() > 0
        expected = getattr(module, name).detach().clone()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            assert torch.equal(getattr(module, name), expected)
            before = x @ getattr(module, name).T
        parametrize.remove_parametrizations(module, name, leave_parametrized=True)
        assert set(module.state_dict()) == keys
        assert torch.equal(getattr(module, name), expected)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            assert torch.equal(x @ getattr(module, name).T, before)
        stream = io.BytesIO()
        torch.save(module.state_dict(), stream)
        stream.seek(0)
        module.load_state_dict(torch.load(stream, weights_only=True), strict=True)
        assert torch.equal(getattr(module, name), expected)
    print(
        "PASS bare/Linear zero-B gradients, FP32 materialization under autocast, nonzero merge/strict save-reload"
    )


if __name__ == "__main__":
    main()
