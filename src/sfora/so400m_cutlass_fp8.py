"""CUTLASS (sm_120/sm_121 native FP8 MMA) GEMMs with fused epilogues for the fused FP8 So400m encoder.

Same layer as `so400m_fused_fp8.FusedLayer(mx_mlp=True)`, with three GEMMs moved to CUTLASS kernels whose epilogues
absorb the work that otherwise round-trips through memory (`native/cutlass_fp8/cutlass_fp8.cu`):
  o-proj : rowwise FP8 GEMM + scales + bias -> fp16, + residual h (C operand)    (no separate d tensor)
  fc1    : rowwise FP8 GEMM + scales + bias -> fp16 -> tanh-GELU -> fp16 -> MXFP8 codes + 1x32 e8m0 scales
  fc2    : MXFP8 block-scaled GEMM + per-column residual r + bias -> fp16, + residual h (C operand)
LayerNorm + per-row quantisation stay in the Triton `add_ln_quant` (a full-row reduction no GEMM tile sees);
q/k/v runs as a CUTLASS pingpong GEMM without extra fusion (default; `qkv_cutlass=False` keeps cuBLAS), flash attention +
`attn_quant` unchanged.
Build the library with `native/cutlass_fp8/build.sh CUTLASS_DIR` (or point SFORA_CUTLASS_FP8_LIB at it).
"""
from __future__ import annotations

import ctypes
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from sfora.so400m_fused_fp8 import E4M3, FusedLayer, FusedSharedTrunkEnsemble, _mm, add_ln_quant, attn_quant

_P, _I = ctypes.c_void_p, ctypes.c_int
_SIG = {"cf8_linear": 6, "cf8_linear_res": 7, "cf8_fc1_gelu_mx": 8, "cf8_mx_res": 8}  # pointer args before M, N, K
_LIB: dict[str, ctypes.CDLL] = {}


def _fn(name: str):
    if not _LIB:
        default = Path(__file__).resolve().parents[2] / "native" / "cutlass_fp8" / "libcutlass_fp8.so"
        lib = ctypes.CDLL(os.environ.get("SFORA_CUTLASS_FP8_LIB", str(default)))
        for n, k in _SIG.items():
            getattr(lib, n).argtypes, getattr(lib, n).restype = [_P] * k + [_I] * 3 + [_P], _I
        _LIB["lib"] = lib
    return getattr(_LIB["lib"], name)


def _run(name: str, *args) -> None:
    ptrs = [a.data_ptr() if isinstance(a, torch.Tensor) else a for a in args]
    rc = _fn(name)(*ptrs, torch.cuda.current_stream().cuda_stream)
    if rc:
        raise RuntimeError(f"{name} failed (code {rc})")


def linear(q, s, w, sw, bias) -> torch.Tensor:
    """fp8 rows (M, K) + row scales (M, 1) x e4m3 weight (N, K) + col scales (N,) + fp16 bias -> fp16 (M, N)."""
    M, K = q.shape
    out = torch.empty(M, w.shape[0], device=q.device, dtype=torch.float16)
    _run("cf8_linear", q, w, s, sw, bias, out, M, w.shape[0], K)
    return out


def linear_res(q, s, w, sw, bias, h) -> torch.Tensor:
    """h + linear(...) with the add in the epilogue (fp16 GEMM output, fp16 add, as the served module)."""
    M, K = q.shape
    out = torch.empty_like(h)
    _run("cf8_linear_res", q, w, s, sw, bias, h, out, M, w.shape[0], K)
    return out


def fc1_gelu_mx(q, s, w, sw, bias, one) -> tuple[torch.Tensor, torch.Tensor]:
    """-> MXFP8 GELU activations (M, NP) + flat 128x4-swizzled e8m0 scales (w/sw/bias padded to NP rows)."""
    M, K = q.shape
    NP = w.shape[0]
    out = torch.empty(M, NP, device=q.device, dtype=E4M3)
    e = torch.empty(-(-M // 128) * 128 * (NP // 32), device=q.device, dtype=torch.uint8)
    _run("cf8_fc1_gelu_mx", q, w, s, sw, bias, out, e, one, M, NP, K)
    return out, e


def mx_res(q, e, w, ew, r, bias, h) -> torch.Tensor:
    """h + fp16(MXFP8(q, e) @ MXFP8(w, ew).T * r + bias), add in the epilogue."""
    M, K = q.shape
    out = torch.empty_like(h)
    _run("cf8_mx_res", q, e, w, ew, r, bias, h, out, M, w.shape[0], K)
    return out


class CutlassLayer:
    def __init__(self, layer, qkv_cutlass: bool = True) -> None:
        f = self.f = FusedLayer(layer, mx_mlp=True)  # reuses the served weights / scales / MX fc2 split
        dev = f.fc1.w_t.device
        self.qkv_cutlass = qkv_cutlass
        self.wqkv, self.sqkv = f.qkv.w_t.t().contiguous(), f.qkv.w_scale.flatten().contiguous()
        self.wo, self.so = f.o.w_t.t().contiguous(), f.o.w_scale.flatten().contiguous()
        n, kp = f.fc1.out_features, f.kp
        self.w1 = torch.zeros(kp, f.w1.shape[1], device=dev, dtype=E4M3)
        self.w1[:n] = f.w1
        self.s1 = torch.zeros(kp, device=dev, dtype=torch.float32)
        self.s1[:n] = f.w1_scale
        self.b1 = torch.zeros(kp, device=dev, dtype=torch.float16)
        self.b1[:n] = f.fc1.bias
        self.one = torch.ones(1, device=dev, dtype=torch.float32)
        self.w2, self.e2 = f.mx.w_t.t().contiguous(), f.mx.e

    def __call__(self, h: torch.Tensor, B: int, T: int) -> torch.Tensor:
        f = self.f
        _, q, s = add_ln_quant(h, None, f.ln1)
        qkv = linear(q, s, self.wqkv, self.sqkv, f.qkv.bias) if self.qkv_cutlass else _mm(q, s, f.qkv)
        qkv = qkv.view(B, T, 3, f.heads, f.hd)
        o = F.scaled_dot_product_attention(*(qkv[:, :, i].transpose(1, 2) for i in range(3)), scale=f.scale)
        q, s = attn_quant(o)
        h = linear_res(q, s, self.wo, self.so, f.o.bias, h)
        _, q, s = add_ln_quant(h, None, f.ln2)
        qm, e = fc1_gelu_mx(q, s, self.w1, self.s1, self.b1, self.one)
        return mx_res(qm, e, self.w2, self.e2, f.mx.r, f.mx.bias, h)


def run_cutlass(layers: list[CutlassLayer], h: torch.Tensor) -> torch.Tensor:
    B, T, D = h.shape
    x = h.reshape(B * T, D).contiguous()
    for layer in layers:
        x = layer(x, B, T)
    return x.view(B, T, D)


class CutlassSharedTrunkEnsemble(FusedSharedTrunkEnsemble):
    """FusedSharedTrunkEnsemble(mx_mlp=True) with the CUTLASS fused-epilogue layers."""

    def __init__(self, members, *, split: int = 13, compile_heads: bool = True, qkv_cutlass: bool = True) -> None:
        super().__init__(members, split=split, compile_heads=compile_heads, mx_mlp=True)
        layers = self.first.vision.encoder.layers
        self.trunk_layers = [CutlassLayer(l, qkv_cutlass) for l in layers[:split]]
        self.upper_layers = [[CutlassLayer(l, qkv_cutlass) for l in m.vision.encoder.layers[split:]]
                             for m in self.members]

    @torch.no_grad()
    def encode_pixels(self, px: torch.Tensor) -> torch.Tensor:
        f = self.first
        h = run_cutlass(self.trunk_layers, f.vision.embeddings(px.half(), interpolate_pos_encoding=f.interpolate))
        outs = []
        for m, layers, pool in zip(self.members, self.upper_layers, self.pool):
            pooled = pool(m.vision.post_layernorm(run_cutlass(layers, h)))
            outs.append(F.normalize(m.head(F.normalize(pooled.float(), dim=1)), dim=1))
        return torch.cat(outs, 1) / math.sqrt(len(outs))


__all__ = ["CutlassLayer", "CutlassSharedTrunkEnsemble", "fc1_gelu_mx", "linear", "linear_res", "mx_res", "run_cutlass"]
