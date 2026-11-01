#!/usr/bin/env python3
"""TRAIN-only pilot: SOP So400m@384/512 recipe with FP8 quantisation-aware training (fake-quant, straight-through).

Identical to train_sop_so400m_384_dim512_pilot.py except every vision nn.Linear outside nn.MultiheadAttention (the set
`_swap_fp8` replaces at serving) becomes a fake-quant Linear: per-row e4m3 activations and per-output-channel e4m3
weights (dequantised, fp32 math, straight-through gradients). State-dict keys are unchanged, so checkpoints load into
So400mCompactEncoder / CUTLASS serving as-is.
Why (measured): every FP8 SOP serving path loses 0.07–0.22 held R@1 vs fp32 (served 95.197, fused 95.129, CUTLASS
95.044 vs 95.266), so the faster paths fail the gate 95.166. QAT trains the weights for the served numerics.
Pre-registered: train seeds 179024 and 179025 (1000 updates each). Gate = the 2-seed ensemble served on CUTLASS
(scripts/check_sop_fused_fp8_held_v2.py, arm cutlass): held R@1 >= 95.166 => ONE official read on CUTLASS; new served
SOP path iff official >= 94.029 (served-FP8 acceptance rule). Else KILL. In-trainer held numbers are reported only.
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_sop_so400m_384_dim512_pilot as base  # noqa: E402

FMAX = torch.finfo(torch.float8_e4m3fn).max


def _fq(x: torch.Tensor, dim: int) -> torch.Tensor:
    xf = x.float()
    s = xf.abs().amax(dim=dim, keepdim=True).clamp(min=1e-12) / FMAX
    return xf + ((xf / s).to(torch.float8_e4m3fn).float() * s - xf).detach()


class FakeQuantLinear(nn.Linear):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(_fq(x, -1), _fq(self.weight, 1), None if self.bias is None else self.bias.float())
        return y.to(x.dtype)


def _swap(module: nn.Module) -> int:
    n = 0
    for child in module.children():
        if type(child) is nn.Linear and child.in_features % 16 == 0 and child.out_features % 16 == 0:
            child.__class__ = FakeQuantLinear
            n += 1
        elif not isinstance(child, nn.MultiheadAttention):
            n += _swap(child)
    return n


import transformers  # noqa: E402

_orig = transformers.AutoModel.from_pretrained


def _from_pretrained(*a, **k):
    model = _orig(*a, **k)
    n = _swap(model.vision_model)
    print(f"FP8-QAT: {n} vision linears fake-quantised", flush=True)
    return model


transformers.AutoModel.from_pretrained = _from_pretrained

if __name__ == "__main__":
    assert sys.argv[sys.argv.index("--substrate") + 1] == "so400"
    base.pair.main()
