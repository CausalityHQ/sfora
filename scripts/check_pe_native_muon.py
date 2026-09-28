#!/usr/bin/env python3
"""Small CPU multi-optimizer/scaler check; no image training or CUDA."""

import torch
from torch import nn

from pe_native_muon import step


def trial(nonfinite):
    torch.manual_seed(179035)
    matrix, bias = nn.Parameter(torch.randn(32, 64)), nn.Parameter(torch.randn(32))
    opts = (
        torch.optim.Muon(
            [matrix], lr=1e-5, weight_decay=0.05, adjust_lr_fn="match_rms_adamw"
        ),
        torch.optim.AdamW([bias], lr=1e-4, weight_decay=0.05),
    )
    scaler = torch.amp.GradScaler("cpu", init_scale=128)
    before = [p.detach().clone() for p in (matrix, bias)]
    scaler.scale((matrix @ torch.linspace(-1, 1, 64) + bias).square().mean()).backward()
    if nonfinite:
        bias.grad[0] = float("inf")
        try:
            step(opts, scaler, (matrix, bias))
        except RuntimeError:
            assert all(
                torch.equal(p, old)
                for p, old in zip((matrix, bias), before, strict=True)
            )
            assert all(not opt.state for opt in opts)
        else:
            raise AssertionError("nonfinite gradient was accepted")
    else:
        step(opts, scaler, (matrix, bias))
        assert all(
            not torch.equal(p, old)
            for p, old in zip((matrix, bias), before, strict=True)
        )
        assert scaler.get_scale() == 128 and all(opt.state for opt in opts)
        for opt in opts:
            opt.zero_grad(set_to_none=True)
        scaler.scale(
            (matrix @ torch.linspace(-1, 1, 64) + bias).square().mean()
        ).backward()
        step(opts, scaler, (matrix, bias))
        assert scaler.get_scale() == 128


if __name__ == "__main__":
    assert not torch.cuda.is_available()
    trial(False)
    trial(True)
    print(
        "PASS two optimizer updates share scaler/global clip; nonfinite Adam gradient prevents either update"
    )
