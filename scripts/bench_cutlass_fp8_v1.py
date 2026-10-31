#!/usr/bin/env python3
"""CUTLASS fused-epilogue FP8 encoder vs current fused (mx) path, single In-Shop So400m@384 member, GPU-only.
Keep iff new/current b64 img/s >= 1.20 and max cos drift <= 0.01 on 512 In-Shop TRAIN images. Else KILL.
Usage: bench_cutlass_fp8_v1.py SNAPSHOT CKPT INSHOP_ROOT [--check-only]"""
import json, statistics, sys, time
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
from PIL import Image
from sfora.so400m_compact_serving import So400mCompactEncoder, _FP8Linear
from sfora.so400m_fused_fp8 import FMAX, FusedSharedTrunkEnsemble, _MXFC2, _mm, add_ln_quant
from sfora.so400m_fused_fp8 import fc1_gelu_mx as tri_fc1
from sfora.so400m_cutlass_fp8 import CutlassSharedTrunkEnsemble, fc1_gelu_mx, linear, linear_res, mx_res
from sfora.unicom_inshop import parse_inshop_partition

snap, ckpt, root = map(Path, sys.argv[1:4]); dev = torch.device("cuda")


def unswizzle(e, R, C):
    """flat cuBLAS/CUTLASS 128x4 e8m0 tile layout -> (R, C) exponents (float, unbiased)."""
    r, c = torch.meshgrid(torch.arange(R, device=e.device), torch.arange(C, device=e.device), indexing="ij")
    Cp = -(-C // 4) * 4
    return e.view(torch.uint8)[((r // 128) * (Cp // 4) + c // 4) * 512 + (r % 32) * 16 + ((r % 128) // 32) * 4 + c % 4].float() - 127


def dequant_mx(q, e):
    M, N = q.shape
    return (q.float().view(M, N // 32, 32) * torch.exp2(unswizzle(e, M, N // 32))[:, :, None]).view(M, N)


# ---- kernel checks vs the current (cuBLAS / Triton) arithmetic, odd M exercises the tile residue
chk = {}
with torch.no_grad():
    for M in (4096, 4100):
        ln = torch.nn.LayerNorm(1152, eps=1e-6).to(dev).half(); ln.weight.data.normal_(1, .2); ln.bias.data.normal_(0, .2)
        h = torch.randn(M, 1152, device=dev, dtype=torch.float16) * 3
        lin = [_FP8Linear(torch.nn.Linear(i, o).to(dev).half()) for i, o in ((1152, 1152), (1152, 4304), (4304, 1152))]
        o, fc1, fc2 = lin
        _, q, s = add_ln_quant(h, None, ln)
        ref = _mm(q, s, o)
        y = linear(q, s, o.w_t.t().contiguous(), o.w_scale.flatten().contiguous(), o.bias)
        chk[f"M{M}_linear_max_abs"] = float((y.float() - ref.float()).abs().max())
        y = linear_res(q, s, o.w_t.t().contiguous(), o.w_scale.flatten().contiguous(), o.bias, h)
        chk[f"M{M}_linear_res_max_abs"] = float((y.float() - (h + ref).float()).abs().max())
        kp = 4352
        w1 = torch.zeros(kp, 1152, device=dev, dtype=torch.float8_e4m3fn); w1[:4304] = fc1.w_t.t()
        s1 = torch.zeros(kp, device=dev); s1[:4304] = fc1.w_scale.flatten()
        b1 = torch.zeros(kp, device=dev, dtype=torch.float16); b1[:4304] = fc1.bias
        qc, ec = fc1_gelu_mx(q, s, w1, s1, b1, torch.ones(1, device=dev))
        qt, et = tri_fc1(q, s, fc1.w_t.t(), fc1.w_scale.flatten().contiguous(), fc1.bias, kp)
        g = F.gelu(_mm(q, s, fc1).float(), approximate="tanh")
        rel = lambda a, b: float((a - b).norm() / b.norm())
        chk[f"M{M}_fc1_cutlass_rel_err"] = rel(dequant_mx(qc, ec)[:, :4304], g)
        chk[f"M{M}_fc1_triton_rel_err"] = rel(dequant_mx(qt, et)[:, :4304], g)
        chk[f"M{M}_fc1_sf_equal_frac"] = float((unswizzle(ec, M, kp // 32) == unswizzle(et, M, kp // 32)).float().mean())
        chk[f"M{M}_fc1_pad_zero"] = bool((qc[:, 4304:].float() == 0).all())
        mx2 = _MXFC2(fc2, kp)
        y = mx_res(qc, ec, mx2.w_t.t().contiguous(), mx2.e, mx2.r, mx2.bias, h)
        ref = h + (mx2(qc, ec.view(torch.float8_e8m0fnu)).float() * mx2.r + mx2.bias.float()).half()
        chk[f"M{M}_mx_res_max_abs"] = float((y.float() - ref.float()).abs().max())
        chk[f"M{M}_mx_res_ref_absmax"] = float(ref.float().abs().max())
print("KERNEL_CHECK", json.dumps(chk), flush=True)
if "--check-only" in sys.argv:
    sys.exit(0)


def rate(fn, x, n=8):
    for _ in range(3): fn(x)
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(n): fn(x)
    torch.cuda.synchronize(); return n * x.shape[0] / (time.perf_counter() - t)


def p50(fn, x, n=60):
    for _ in range(5): fn(x)
    ts = []
    for _ in range(n):
        torch.cuda.synchronize(); t = time.perf_counter(); fn(x); torch.cuda.synchronize(); ts.append(time.perf_counter() - t)
    return 1e3 * statistics.median(ts)


out_dim = int(torch.load(ckpt, map_location="cpu", weights_only=True)["head"]["weight"].shape[0])
enc = So400mCompactEncoder.from_checkpoint(snap, ckpt, resolution=384, output_dim=out_dim, device=dev, fp16_native=True, fp8=True)
train = [r for r in parse_inshop_partition(root) if r.split == "train"]
pick = np.random.default_rng(0).permutation(len(train))[:512]
test = [enc.processor(images=[Image.open(train[i].image_path).convert("RGB") for i in pick[k:k + 64]],
                      return_tensors="pt")["pixel_values"].to(dev) for k in range(0, 512, 64)]
x64, x1 = torch.randn(64, 3, 384, 384, device=dev), torch.randn(1, 3, 384, 384, device=dev)
arms = {"current_mx": FusedSharedTrunkEnsemble([enc], mx_mlp=True), "cutlass": CutlassSharedTrunkEnsemble([enc]),
        "cutlass_cublas_qkv": CutlassSharedTrunkEnsemble([enc], qkv_cutlass=False)}
res, rows = {"kernel_check": chk, "out_dim": out_dim}, {}
from torch.profiler import ProfilerActivity, profile
with torch.inference_mode():
    for name, ens in arms.items():
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        rate(ens.encode_pixels, x64, 2)
        res[f"{name}_peak_gib"] = torch.cuda.max_memory_allocated() / 2 ** 30
        res[f"{name}_b1_p50_ms"] = p50(ens.encode_pixels, x1)
        rows[name] = torch.cat([ens.encode_pixels(p).float() for p in test])
    for rep in range(3):  # interleaved repeats: no order effect between arms
        for name, ens in arms.items():
            res.setdefault(f"{name}_b64_img_s", []).append(rate(ens.encode_pixels, x64))
    for name in ("current_mx", "cutlass"):
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            arms[name].encode_pixels(x64); torch.cuda.synchronize()
        ka = prof.key_averages(); tot = sum(e.device_time_total for e in ka)
        res[f"{name}_profile_top"] = [[e.key[:60], round(e.device_time_total / tot, 3), round(e.device_time_total / 27e3, 3)]
                                      for e in sorted(ka, key=lambda e: -e.device_time_total)[:10]]  # share, ms/layer
base = statistics.median(res["current_mx_b64_img_s"])
for name in arms:
    res[f"{name}_b64_img_s_median"] = statistics.median(res[f"{name}_b64_img_s"])
    res[f"{name}_speedup"] = res[f"{name}_b64_img_s_median"] / base
    drift = 1 - (rows[name] * rows["current_mx"]).sum(1)
    res[f"{name}_max_cos_drift"], res[f"{name}_mean_cos_drift"] = float(drift.abs().max()), float(drift.mean())
best = max(("cutlass", "cutlass_cublas_qkv"), key=lambda n: res[f"{n}_speedup"] if res[f"{n}_max_cos_drift"] <= 0.01 else 0)
res["best_arm"] = best
res["decision"] = "KEEP" if res[f"{best}_speedup"] >= 1.20 and res[f"{best}_max_cos_drift"] <= 0.01 else "KILL"
print("RESULT", json.dumps(res))
