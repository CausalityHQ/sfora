"""A fused pair must equal its explicit parallel residual approximation."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.nn import functional as F

SCRIPT = Path(__file__).parents[1] / "scripts" / "probe_l14_parallel_fold.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("probe_l14_parallel_fold", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ToyAttention(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.num_heads = heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, rows, dim = x.shape
        qkv = self.qkv(x).reshape(batch, rows, 3, self.num_heads, dim // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        attended = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        return self.proj(attended.transpose(1, 2).reshape(batch, rows, dim))


class ToyBlock(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = ToyAttention(dim, heads)
        self.mlp = nn.Sequential()
        self.mlp.fc1 = nn.Linear(dim, dim * 2)
        self.mlp.act = nn.ReLU6()
        self.mlp.fc2 = nn.Linear(dim * 2, dim)
        self.drop_path = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x + self.attn(self.norm1(x))
        return y + self.mlp.fc2(self.mlp.act(self.mlp.fc1(self.norm2(y))))


def test_fused_pair_matches_parallel_branch_oracle() -> None:
    torch.manual_seed(179019)
    first, second = ToyBlock(8, 2).eval(), ToyBlock(8, 2).eval()
    with torch.no_grad():
        for block in (first, second):
            for norm in (block.norm1, block.norm2):
                norm.weight.copy_(torch.randn_like(norm.weight))
                norm.bias.copy_(torch.randn_like(norm.bias))
    x = torch.randn(2, 5, 8)
    fused = MODULE.ParallelFoldBlock(first, second).eval()
    expected = x.clone()
    for block in (first, second):
        expected = expected + block.attn(block.norm1(x))
        expected = expected + block.mlp.fc2(block.mlp.act(block.mlp.fc1(block.norm2(x))))
    assert torch.allclose(fused(x), expected, rtol=1e-5, atol=1e-5)
    assert not torch.allclose(fused(x), second(first(x)), rtol=1e-4, atol=1e-4)


def test_gpu_occupancy_guard_ignores_self_but_rejects_foreign(monkeypatch) -> None:
    monkeypatch.setattr(MODULE.os, "getpid", lambda: 101)
    monkeypatch.setattr(MODULE, "gpu_compute_pids", lambda: {101})
    MODULE.assert_no_foreign_gpu_processes()
    monkeypatch.setattr(MODULE, "gpu_compute_pids", lambda: {101, 202})
    with pytest.raises(ValueError, match="GPU process overlap"):
        MODULE.assert_no_foreign_gpu_processes()


def test_blocks_only_wrapper_executes_original_sequential_blocks() -> None:
    torch.manual_seed(7)
    first, second = ToyBlock(8, 2).eval(), ToyBlock(8, 2).eval()
    x = torch.randn(1, 5, 8)
    wrapped = MODULE.BlocksOnly(nn.ModuleList([first, second])).eval()
    assert torch.allclose(wrapped(x), second(first(x)))
