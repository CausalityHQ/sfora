"""Fused FP8 encoder layers for the shared-trunk So400m ensemble (inference only, CUDA sm_89+).

Same arithmetic as the served `_FP8Linear` path (dynamic per-row e4m3 activation scales, per-output-channel weight
scales, fp16 out); only the work between the FP8 GEMMs is fused into three Triton kernels per layer:
  add_ln_quant : residual add (deferred from the previous layer's fc2) + LayerNorm + per-row amax + FP8 cast
  attn_quant   : SDPA output read in place (any strides, no transpose copy) + per-row amax + FP8 cast
  gelu_quant   : tanh-GELU + per-row amax + FP8 cast
q/k/v run as ONE GEMM (concatenated, identical FP8 weights/scales). Intermediates are rounded to fp16 before
quantisation exactly where the served fp16 module rounds them.
"""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from sfora.so400m_shared_trunk_ensemble import SharedTrunkEnsemble

E4M3 = torch.float8_e4m3fn
FMAX = torch.finfo(E4M3).max


@triton.jit
def _add_ln_quant(h_ptr, d_ptr, ds_ptr, db_ptr, hout_ptr, w_ptr, b_ptr, q_ptr, s_ptr, N, eps, fmax,
                  HAS_D: tl.constexpr, D_AFFINE: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    m = cols < N
    x = tl.load(h_ptr + row * N + cols, mask=m, other=0.0)
    if HAS_D:
        d = tl.load(d_ptr + row * N + cols, mask=m, other=0.0)
        if D_AFFINE:  # deferred fc2 epilogue (MX path): column scale + bias, rounded to fp16 like a GEMM output
            ds = tl.load(ds_ptr + cols, mask=m, other=0.0)
            db = tl.load(db_ptr + cols, mask=m, other=0.0).to(tl.float32)
            d = (d.to(tl.float32) * ds + db).to(tl.float16)
        x = x + d  # fp16 add, as the served residual add
        tl.store(hout_ptr + row * N + cols, x, mask=m)
    x = x.to(tl.float32)
    mean = tl.sum(x, 0) / N
    xc = tl.where(m, x - mean, 0.0)
    rstd = 1.0 / tl.sqrt(tl.sum(xc * xc, 0) / N + eps)
    w = tl.load(w_ptr + cols, mask=m, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + cols, mask=m, other=0.0).to(tl.float32)
    y = (xc * rstd * w + b).to(tl.float16).to(tl.float32)
    scale = tl.maximum(tl.max(tl.where(m, tl.abs(y), 0.0), 0), 1e-12) / fmax
    tl.store(q_ptr + row * N + cols, (y / scale).to(tl.float8e4nv), mask=m)
    tl.store(s_ptr + row, scale)


@triton.jit
def _gelu_quant(x_ptr, q_ptr, s_ptr, N, fmax, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    m = cols < N
    x = tl.load(x_ptr + row * N + cols, mask=m, other=0.0).to(tl.float32)
    u = 0.7978845608028654 * (x + 0.044715 * x * x * x)
    t = 1.0 - 2.0 / (1.0 + tl.exp(2.0 * u))  # tanh(u)
    y = (0.5 * x * (1.0 + t)).to(tl.float16).to(tl.float32)
    scale = tl.maximum(tl.max(tl.where(m, tl.abs(y), 0.0), 0), 1e-12) / fmax
    tl.store(q_ptr + row * N + cols, (y / scale).to(tl.float8e4nv), mask=m)
    tl.store(s_ptr + row, scale)


@triton.jit
def _attn_quant(o_ptr, q_ptr, s_ptr, T, sb, sh, st, N, fmax, HD: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    bi, ti = (row // T).to(tl.int64), (row % T).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    m = cols < N
    off = bi * sb + ti * st + (cols // HD) * sh + cols % HD
    y = tl.load(o_ptr + off, mask=m, other=0.0).to(tl.float32)
    scale = tl.maximum(tl.max(tl.where(m, tl.abs(y), 0.0), 0), 1e-12) / fmax
    tl.store(q_ptr + row.to(tl.int64) * N + cols, (y / scale).to(tl.float8e4nv), mask=m)
    tl.store(s_ptr + row, scale)


_FC1_CONFIGS = [triton.Config({"BM": bm, "BN": bn, "BK": bk}, num_stages=s, num_warps=w) for bm, bn, bk, s, w in
                [(128, 128, 64, 4, 4), (128, 128, 128, 3, 8), (128, 128, 128, 4, 8), (128, 256, 64, 3, 8),
                 (256, 128, 64, 3, 8), (128, 128, 64, 5, 8), (64, 128, 128, 4, 4)]]


@triton.autotune(_FC1_CONFIGS, key=["M", "N", "K"])
@triton.jit
def _fc1_gelu_mx(a_ptr, w_ptr, sa_ptr, sw_ptr, bias_ptr, q_ptr, e_ptr, M, N, NP, K, fmax,
                 BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """fp8 rowwise GEMM + bias -> fp16 -> tanh-GELU -> fp16 -> MXFP8 (1x32 e8m0, cuBLAS-swizzled scales).
    Output columns [N, NP) are zero padding (masked weights)."""
    pid = tl.program_id(0)
    npm, npn = tl.cdiv(M, BM), tl.cdiv(NP, BN)
    gid = pid // (8 * npn)
    fm = gid * 8
    gs = min(npm - fm, 8)
    pm = fm + (pid % (8 * npn)) % gs
    pn = (pid % (8 * npn)) // gs
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    ap = a_ptr + rm[:, None].to(tl.int64) * K + rk[None, :]
    wp = w_ptr + rn[None, :] * K + rk[:, None]
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(0, K, BK):
        a = tl.load(ap, mask=(rm[:, None] < M) & (rk[None, :] + k < K), other=0.0)
        w = tl.load(wp, mask=(rn[None, :] < N) & (rk[:, None] + k < K), other=0.0)
        acc = tl.dot(a, w, acc)
        ap += BK
        wp += BK
    nm = rn < N
    acc = acc * tl.load(sa_ptr + rm, mask=rm < M, other=0.0)[:, None] * tl.load(sw_ptr + rn, mask=nm, other=0.0)[None, :]
    x = (acc + tl.load(bias_ptr + rn, mask=nm, other=0.0).to(tl.float32)[None, :]).to(tl.float16).to(tl.float32)
    u = 0.7978845608028654 * (x + 0.044715 * x * x * x)
    y = (0.5 * x * (2.0 - 2.0 / (1.0 + tl.exp(2.0 * u)))).to(tl.float16).to(tl.float32)
    y3 = tl.reshape(y, (BM, BN // 32, 32))
    amax = tl.max(tl.abs(y3), 2)
    e = tl.where(amax > 0, tl.ceil(tl.log2(amax / fmax)), 0.0)  # zero (padding) blocks: scale 1, codes 0
    e = tl.minimum(tl.maximum(e, -126.0), 127.0)
    q = tl.reshape(tl.clamp(y3 / tl.exp2(e)[:, :, None], -fmax, fmax), (BM, BN))
    mm = rm[:, None] < M
    tl.store(q_ptr + rm[:, None].to(tl.int64) * NP + rn[None, :], q.to(tl.float8e4nv), mask=mm & (rn[None, :] < NP))
    cb = pn * (BN // 32) + tl.arange(0, BN // 32)
    r = rm[:, None]
    off = ((r // 128) * (NP // 128) + cb[None, :] // 4).to(tl.int64) * 512 + (r % 32) * 16 + ((r % 128) // 32) * 4 \
        + cb[None, :] % 4
    tl.store(e_ptr + off, (e + 127.0).to(tl.uint8), mask=mm)


def swizzle_scales(s: torch.Tensor) -> torch.Tensor:
    """(rows, K/32) e8m0 bytes -> flat cuBLAS 128x4 tile layout (rows padded to 128, cols to 4)."""
    R, C = s.shape
    Rp, Cp = -(-R // 128) * 128, -(-C // 4) * 4
    out = torch.zeros(Rp * Cp, dtype=s.dtype, device=s.device)
    r, c = torch.meshgrid(torch.arange(R, device=s.device), torch.arange(C, device=s.device), indexing="ij")
    out[((r // 128) * (Cp // 4) + c // 4) * 512 + (r % 32) * 16 + ((r % 128) // 32) * 4 + c % 4] = s
    return out


def _new_q(rows: int, n: int, dev: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty(rows, n, device=dev, dtype=E4M3), torch.empty(rows, 1, device=dev, dtype=torch.float32)


def add_ln_quant(h: torch.Tensor, d: torch.Tensor | None, ln: torch.nn.LayerNorm, d_scale=None, d_bias=None):
    """-> (h + d materialised in fp16, fp8 LN rows, per-row scales). h, d: (M, N) contiguous fp16.
    With d_scale/d_bias (fp32 / fp16 (N,)), d is first replaced by fp16(d * d_scale + d_bias)."""
    M, N = h.shape
    q, s = _new_q(M, N, h.device)
    hout = torch.empty_like(h) if d is not None else h
    _add_ln_quant[(M,)](h, d if d is not None else h, d_scale if d_scale is not None else ln.weight,
                        d_bias if d_bias is not None else ln.bias, hout, ln.weight, ln.bias, q, s, N, ln.eps, FMAX,
                        HAS_D=d is not None, D_AFFINE=d_scale is not None, BLOCK=triton.next_power_of_2(N),
                        num_warps=4)
    return hout, q, s


def fc1_gelu_mx(q: torch.Tensor, s: torch.Tensor, w: torch.Tensor, w_scale: torch.Tensor, bias: torch.Tensor, NP: int):
    """fp8 rows (M, K) x e4m3 weight (N, K) -> MXFP8 GELU activations (M, NP) + swizzled e8m0 scales."""
    M, K = q.shape
    N = w.shape[0]
    out = torch.empty(M, NP, device=q.device, dtype=E4M3)
    e = torch.zeros(-(-M // 128) * 128 * (NP // 32), device=q.device, dtype=torch.uint8)
    _fc1_gelu_mx[lambda c: (triton.cdiv(M, c["BM"]) * triton.cdiv(NP, c["BN"]),)](
        q, w, s, w_scale, bias, out, e, M, N, NP, K, FMAX)
    return out, e.view(torch.float8_e8m0fnu)


def gelu_quant(x: torch.Tensor):
    M, N = x.shape
    q, s = _new_q(M, N, x.device)
    _gelu_quant[(M,)](x, q, s, N, FMAX, BLOCK=triton.next_power_of_2(N), num_warps=8)
    return q, s


def attn_quant(o: torch.Tensor):
    """o: SDPA output (B, H, T, HD), any strides with stride(-1) == 1 -> fp8 (B*T, H*HD) rows + scales."""
    B, H, T, HD = o.shape
    assert o.stride(-1) == 1
    q, s = _new_q(B * T, H * HD, o.device)
    _attn_quant[(B * T,)](o, q, s, T, o.stride(0), o.stride(1), o.stride(2), H * HD, FMAX, HD=HD,
                          BLOCK=triton.next_power_of_2(H * HD), num_warps=4)
    return q, s


def _mm(q: torch.Tensor, s: torch.Tensor, lin: Any) -> torch.Tensor:
    return torch._scaled_mm(q, lin.w_t, scale_a=s, scale_b=lin.w_scale, bias=lin.bias, out_dtype=torch.float16)


class _QKV:
    """q/k/v `_FP8Linear`s concatenated along outputs: same e4m3 weights, scales and bias."""

    def __init__(self, *lins: Any) -> None:
        self.w_t = torch.cat([l.w_t.t() for l in lins]).t()
        self.w_scale = torch.cat([l.w_scale for l in lins], 1).contiguous()
        self.bias = torch.cat([l.bias for l in lins]).contiguous()


class _MXFC2:
    """fc2 as MXFP8: the served e4m3 weight codes unchanged, per-row scale s split into a power-of-two block scale
    (e8m0, every 1x32 block) and a residual r = s / 2^floor(log2 s) in [1, 2) applied with the bias downstream."""

    def __init__(self, lin: Any, kp: int) -> None:
        w = lin.w_t.t()  # (N, K) e4m3
        n, k = w.shape
        wp = torch.zeros(n, kp, dtype=E4M3, device=w.device)
        wp[:, :k] = w
        self.w_t = wp.t()
        s = lin.w_scale.flatten().float()
        p = torch.floor(torch.log2(s))
        self.r = (s / torch.exp2(p)).contiguous()
        e = (p + 127).to(torch.uint8)[:, None].expand(n, kp // 32).contiguous()
        self.e = swizzle_scales(e).view(torch.float8_e8m0fnu)
        self.bias = lin.bias

    def __call__(self, q: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        bw, sw = F.ScalingType.BlockWise1x32, F.SwizzleType.SWIZZLE_32_4_4
        return F.scaled_mm(q, self.w_t, e, bw, self.e, bw, swizzle_a=sw, swizzle_b=sw, output_dtype=torch.float16)


class FusedLayer:
    def __init__(self, layer: Any, mx_mlp: bool = False) -> None:
        layer = getattr(layer, "_orig_mod", layer)
        a, mlp = layer.self_attn, layer.mlp
        if getattr(a, "dropout", 0.0) or not hasattr(mlp.fc1, "w_t") or not hasattr(a.q_proj, "w_t"):
            raise ValueError("fused FP8 layer needs an FP8-swapped, eval SigLIP layer")
        self.ln1, self.ln2, self.o, self.fc1, self.fc2 = layer.layer_norm1, layer.layer_norm2, a.out_proj, mlp.fc1, mlp.fc2
        self.qkv = _QKV(a.q_proj, a.k_proj, a.v_proj)
        self.heads, self.hd, self.scale = a.num_heads, a.head_dim, a.scale
        self.mx = None
        if mx_mlp:
            self.kp = -(-self.fc1.out_features // 128) * 128
            self.w1 = self.fc1.w_t.t()
            self.w1_scale = self.fc1.w_scale.flatten().contiguous()
            self.mx = _MXFC2(self.fc2, self.kp)

    def __call__(self, h: torch.Tensor, d: tuple | None, B: int, T: int):
        """Residual stream = h + affine(d) (d deferred from the previous fc2). Returns (h', d')."""
        dd, ds, db = d or (None, None, None)
        h, q, s = add_ln_quant(h, dd, self.ln1, ds, db)
        qkv = _mm(q, s, self.qkv).view(B, T, 3, self.heads, self.hd)
        o = F.scaled_dot_product_attention(*(qkv[:, :, i].transpose(1, 2) for i in range(3)), scale=self.scale)
        q, s = attn_quant(o)
        h, q, s = add_ln_quant(h, _mm(q, s, self.o), self.ln2)
        if self.mx is not None:
            q, e = fc1_gelu_mx(q, s, self.w1, self.w1_scale, self.fc1.bias, self.kp)
            return h, (self.mx(q, e), self.mx.r, self.mx.bias)
        q, s = gelu_quant(_mm(q, s, self.fc1))
        return h, (_mm(q, s, self.fc2), None, None)


def run_fused(layers: list[FusedLayer], h: torch.Tensor) -> torch.Tensor:
    B, T, D = h.shape
    x, d = h.reshape(B * T, D).contiguous(), None
    for layer in layers:
        x, d = layer(x, d, B, T)
    dd, r, b = d
    if r is not None:
        dd = (dd.float() * r + b.float()).half()
    return (x + dd).view(B, T, D)


class FusedSharedTrunkEnsemble(SharedTrunkEnsemble):
    """SharedTrunkEnsemble with fused FP8 encoder layers; embeddings and pooling heads as in the served path."""

    def __init__(self, members, *, split: int = 13, compile_heads: bool = True, mx_mlp: bool = False) -> None:
        super().__init__(members, split=split, compile=False)
        cfg = self.first.vision.config
        if not self.first.fp16_native or cfg.hidden_act != "gelu_pytorch_tanh":
            raise ValueError("fused FP8 ensemble needs fp8 members with tanh GELU")
        self.trunk_layers = [FusedLayer(l, mx_mlp) for l in self.first.vision.encoder.layers[:split]]
        self.upper_layers = [[FusedLayer(l, mx_mlp) for l in m.vision.encoder.layers[split:]] for m in self.members]
        self.pool = [torch.compile(m.vision.head, mode="max-autotune-no-cudagraphs") if compile_heads else m.vision.head
                     for m in self.members]

    @torch.no_grad()
    def encode_pixels(self, px: torch.Tensor) -> torch.Tensor:
        f = self.first
        h = run_fused(self.trunk_layers, f.vision.embeddings(px.half(), interpolate_pos_encoding=f.interpolate))
        outs = []
        for m, layers, pool in zip(self.members, self.upper_layers, self.pool):
            pooled = pool(m.vision.post_layernorm(run_fused(layers, h)))
            outs.append(F.normalize(m.head(F.normalize(pooled.float(), dim=1)), dim=1))
        return torch.cat(outs, 1) / math.sqrt(len(outs))


__all__ = ["FusedSharedTrunkEnsemble", "FusedLayer", "add_ln_quant", "attn_quant", "gelu_quant", "fc1_gelu_mx", "run_fused", "swizzle_scales"]
