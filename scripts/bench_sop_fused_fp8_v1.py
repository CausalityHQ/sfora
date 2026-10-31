#!/usr/bin/env python3
"""Fused FP8 encoder vs served FP8 shared-trunk SOP ensemble (GPU-only; random pixels for speed, 512 TRAIN-FIT
images for drift). Pre-registered keep: fused/served b64 img/s >= 1.25 and max cos drift <= 0.02. Else KILL.
Usage: bench_sop_fused_fp8_v1.py SNAPSHOT CKPT_A CKPT_B ARCHIVE SOP_ROOT"""
import json, statistics, sys, time
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_sop_siglip2_substrate_pair as pair
from sfora.representation_ceiling import deterministic_class_partition
from sfora.so400m_compact_serving import So400mCompactEncoder
from sfora.so400m_shared_trunk_ensemble import SharedTrunkEnsemble
from sfora.so400m_compact_serving import _FP8Linear
from sfora.so400m_fused_fp8 import FMAX, FusedSharedTrunkEnsemble, _MXFC2, _mm, add_ln_quant, fc1_gelu_mx, gelu_quant

snap, ca, cb, arch, root = map(Path, sys.argv[1:6]); dev = torch.device("cuda")


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


# kernel self-check against the served torch arithmetic (dequantised fp8 rows)
with torch.no_grad():
    ln = torch.nn.LayerNorm(1152, eps=1e-6).to(dev).half(); ln.weight.data.normal_(1, .2); ln.bias.data.normal_(0, .2)
    h, d, z = (torch.randn(4096, n, device=dev, dtype=torch.float16) * 3 for n in (1152, 1152, 4304))
    def ref_q(y):
        s = y.abs().amax(1, keepdim=True).float().clamp(min=1e-12) / FMAX
        return (y.float() / s).to(torch.float8_e4m3fn).float() * s
    hs, q, s = add_ln_quant(h, d, ln)
    assert torch.equal(hs, h + d)
    e_ln = float(((q.float() * s) - ref_q(ln(h + d))).abs().max())
    q, s = gelu_quant(z)
    e_gelu = float(((q.float() * s) - ref_q(F.gelu(z, approximate="tanh"))).abs().max())
    # MX MLP: fc1+GELU -> MXFP8 vs served fc1 -> GELU (fp32 reference), and MX fc2 (exact served weight codes)
    fc1, fc2 = (_FP8Linear(torch.nn.Linear(i, o).to(dev).half()) for i, o in ((1152, 4304), (4304, 1152)))
    q, s = add_ln_quant(h, None, ln)[1:]
    g = F.gelu(_mm(q, s, fc1).float(), approximate="tanh")
    qm, em = fc1_gelu_mx(q, s, fc1.w_t.t(), fc1.w_scale.flatten().contiguous(), fc1.bias, 4352)
    mx2 = _MXFC2(fc2, 4352)
    y_mx = (mx2(qm, em).float() * mx2.r + fc2.bias.float())
    gs = g.half().abs().amax(1, keepdim=True).float().clamp(min=1e-12) / FMAX
    y_ref = _mm((g.half().float() / gs).to(torch.float8_e4m3fn), gs, fc2).float()
    y_true = g @ (fc2.w_t.float() * fc2.w_scale) + fc2.bias.float()
    relerr = lambda y: float((y - y_true).norm() / y_true.norm())
    print("KERNEL_CHECK", json.dumps({"ln_max_abs": e_ln, "gelu_max_abs": e_gelu, "mlp_mx_rel_err": relerr(y_mx),
                                      "mlp_rowwise_rel_err": relerr(y_ref), "mx_pad_zero": bool((qm[:, 4304:].float() == 0).all())}))

with np.load(arch, allow_pickle=False) as z: rel = np.asarray(z["train_relative_paths"]).astype(str); lab = np.asarray(z["train_labels"], dtype=np.int64)
held = set(deterministic_class_partition(tuple(map(int, lab)), fit_fraction=0.9, seed=179019).validation_row_indexes)
fit = [i for i in np.random.default_rng(179019).permutation(len(lab)) if i not in held][:512]
paths = pair.paths_from_archive(root, rel)
enc = [So400mCompactEncoder.from_checkpoint(snap, c, resolution=384, output_dim=512, device=dev, fp16_native=True, fp8=True) for c in (ca, cb)]
test = [enc[0].processor(images=[Image.open(paths[i]).convert("RGB") for i in fit[k:k + 64]], return_tensors="pt")["pixel_values"].to(dev) for k in range(0, 512, 64)]
x64, x1 = torch.randn(64, 3, 384, 384, device=dev), torch.randn(1, 3, 384, 384, device=dev)

out = {}
from torch.profiler import ProfilerActivity, profile
arms = {"fused": FusedSharedTrunkEnsemble(enc), "fused_mx": FusedSharedTrunkEnsemble(enc, mx_mlp=True)}  # before served wrap
rows = {}
with torch.inference_mode():
    for name, ens in arms.items():
        torch.cuda.reset_peak_memory_stats(); out[f"{name}_b64_img_s"] = rate(ens.encode_pixels, x64)
        out[f"{name}_peak_gib"] = torch.cuda.max_memory_allocated() / 2 ** 30
        out[f"{name}_b1_p50_ms"] = p50(ens.encode_pixels, x1)
        rows[name] = torch.cat([ens.encode_pixels(p).float() for p in test])
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            ens.encode_pixels(x64); torch.cuda.synchronize()
        ka = prof.key_averages(); tot = sum(e.device_time_total for e in ka)
        out[f"{name}_profile_top"] = [[e.key[:50], round(e.device_time_total / tot, 3)] for e in sorted(ka, key=lambda e: -e.device_time_total)[:8]]
served = SharedTrunkEnsemble(enc, compile=True)
with torch.inference_mode():
    torch.cuda.reset_peak_memory_stats(); out["served_b64_img_s"] = rate(served.encode_pixels, x64)
    out["served_peak_gib"] = torch.cuda.max_memory_allocated() / 2 ** 30
    out["served_b1_p50_ms"] = p50(served.encode_pixels, x1)
    rows_s = torch.cat([served.encode_pixels(p).float() for p in test])
    for name, ens in arms.items(): out[f"{name}_b64_img_s_rerun"] = rate(ens.encode_pixels, x64)  # order-effect check
for name in arms:
    drift = 1 - (rows[name] * rows_s).sum(1)
    out[f"{name}_speedup"] = out[f"{name}_b64_img_s"] / out["served_b64_img_s"]
    out[f"{name}_max_cos_drift"], out[f"{name}_mean_cos_drift"] = float(drift.abs().max()), float(drift.mean())
best = max(arms, key=lambda n: out[f"{n}_speedup"] if out[f"{n}_max_cos_drift"] <= 0.02 else 0)
out["best_arm"] = best
out["decision"] = "KEEP" if out[f"{best}_speedup"] >= 1.25 and out[f"{best}_max_cos_drift"] <= 0.02 else "KILL"
print("RESULT", json.dumps(out))
