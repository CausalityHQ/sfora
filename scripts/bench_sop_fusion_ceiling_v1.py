#!/usr/bin/env python3
"""Megakernel / fusion ceiling for the served FP8 shared-trunk SOP ensemble (GPU-only, no quality claim).

Measures, on the real shapes (So400m/14@384: 729 tokens, width 1152, 16 heads x 72, MLP 4304; 13 trunk + 2x14 upper
layer passes = 41):
  served_b64       : img/s of the served path (segment compile, FP8 dynamic per-row)
  graphs_b64       : same with CUDA graphs (mode='max-autotune')            -> launch-overhead share
  floor_b64        : img/s if ONLY the FP8 GEMMs (q,k,v,o,fc1,fc2 on pre-quantised inputs) + flash attention ran
                     (perfect fusion of every LN / GELU / quant / residual into them)  -> megakernel ceiling
  gemm_tflops      : achieved FP8 GEMM throughput in the floor
  b1 p50 latency   : served vs CUDA graphs (the regime where megakernels win)
Pre-registered: build a fused encoder only if floor_b64 / served_b64 >= 1.4 (b64) or graphs cut b1 p50 >= 30 %.
"""
import json, statistics, sys, time
from pathlib import Path
import torch
import torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sfora.so400m_compact_serving import So400mCompactEncoder
from sfora.so400m_shared_trunk_ensemble import SharedTrunkEnsemble

snap, ca, cb = map(Path, sys.argv[1:4]); dev = torch.device("cuda"); E8 = torch.float8_e4m3fn


def rate(fn, x, n=8):
    for _ in range(3): fn(x)
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(n): fn(x)
    torch.cuda.synchronize(); return (time.perf_counter() - t) / n


def p50(fn, x, n=60):
    for _ in range(5): fn(x)
    ts = []
    for _ in range(n):
        torch.cuda.synchronize(); t = time.perf_counter(); fn(x); torch.cuda.synchronize(); ts.append(time.perf_counter() - t)
    return 1e3 * statistics.median(ts)


enc = [So400mCompactEncoder.from_checkpoint(snap, c, resolution=384, output_dim=512, device=dev, fp16_native=True, fp8=True) for c in (ca, cb)]
out = {}
x64, x1 = torch.randn(64, 3, 384, 384, device=dev), torch.randn(1, 3, 384, 384, device=dev)
ens = SharedTrunkEnsemble(enc, compile=True)
with torch.no_grad():
    out["served_b64_img_s"] = 64 / rate(ens.encode_pixels, x64)
    out["served_b1_p50_ms"] = p50(ens.encode_pixels, x1)
    torch._dynamo.reset()
    ens._trunk_fn = torch.compile(ens._trunk, mode="max-autotune"); ens._upper_fn = torch.compile(ens._upper, mode="max-autotune")
    g = lambda x: ens.encode_pixels(x).clone()
    try:
        out["graphs_b64_img_s"] = 64 / rate(g, x64); out["graphs_b1_p50_ms"] = p50(g, x1)
    except Exception as e:
        out["graphs_error"] = repr(e)[:300]

# floor: only GEMMs + attention, per layer pass, b64
M, D, H, HD, FF, L = 64 * 729, 1152, 16, 72, 4304, 13 + 2 * 14
def q8(t): return t.to(E8)
a_d, a_f = q8(torch.randn(M, D, device=dev)), q8(torch.randn(M, FF, device=dev))
w = {n: q8(torch.randn(o, i, device=dev)).t() for n, (i, o) in {"q": (D, D), "k": (D, D), "v": (D, D), "o": (D, D), "fc1": (D, FF), "fc2": (FF, D)}.items()}
sa_d, sa_f = torch.ones(M, 1, device=dev), torch.ones(M, 1, device=dev)
sb = {n: torch.ones(1, t.shape[1], device=dev) for n, t in w.items()}
qkv = torch.randn(64, H, 729, HD, device=dev, dtype=torch.float16)
def layer(_):
    for n in ("q", "k", "v", "o", "fc1"): torch._scaled_mm(a_d, w[n], scale_a=sa_d, scale_b=sb[n], out_dtype=torch.float16)
    torch._scaled_mm(a_f, w["fc2"], scale_a=sa_f, scale_b=sb["fc2"], out_dtype=torch.float16)
def attn(_): F.scaled_dot_product_attention(qkv, qkv, qkv)
with torch.no_grad():
    t_gemm, t_attn = rate(layer, None, 20), rate(attn, None, 20)
flops_gemm = 2 * M * (4 * D * D + 2 * D * FF)
out.update({"floor_b64_img_s": 64 / (L * (t_gemm + t_attn)), "gemm_tflops": flops_gemm / t_gemm / 1e12,
            "attn_ms_per_layer": 1e3 * t_attn, "gemm_ms_per_layer": 1e3 * t_gemm})
out["fusion_ceiling"] = out["floor_b64_img_s"] / out["served_b64_img_s"]
if "graphs_b1_p50_ms" in out: out["b1_graphs_cut"] = 1 - out["graphs_b1_p50_ms"] / out["served_b1_p50_ms"]
out["decision"] = "BUILD" if out["fusion_ceiling"] >= 1.4 or out.get("b1_graphs_cut", 0) >= 0.30 else "CLOSE"
print("RESULT", json.dumps(out))
