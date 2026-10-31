#!/usr/bin/env python3
"""Efficiency falsifier: static per-layer FP8 activation scales vs dynamic per-row (shared-trunk SOP 2-seed ensemble).

Profile (FP8 compiled, b64): cutlass FP8 GEMM 41 %, fused amax/div/cast quantisation kernels ~31 %, flash attn 15 %.
Static scale s_l = margin * max|x| over calibration batches of TRAIN FIT images (never held); forward quantises with a
fixed scale (no amax reduction). Pre-registered: proceed to the held quality gate only if GPU-only img/s improves
>= 10 % over dynamic; report max cos drift vs dynamic on 512 FIT images. Else KILL.
"""
import json, sys, time
from pathlib import Path
import numpy as np, torch
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_sop_siglip2_substrate_pair as pair
from sfora.representation_ceiling import deterministic_class_partition
from sfora.so400m_compact_serving import So400mCompactEncoder, _FP8Linear
from sfora.so400m_shared_trunk_ensemble import SharedTrunkEnsemble

snap, ca, cb, arch, root = map(Path, sys.argv[1:6]); margin = float(sys.argv[6]) if len(sys.argv) > 6 else 1.0
dev = torch.device("cuda")
with np.load(arch, allow_pickle=False) as z: rel = np.asarray(z["train_relative_paths"]).astype(str); lab = np.asarray(z["train_labels"], dtype=np.int64)
held = set(deterministic_class_partition(tuple(map(int, lab)), fit_fraction=0.9, seed=179019).validation_row_indexes)
fit = [i for i in np.random.default_rng(179019).permutation(len(lab)) if i not in held]
paths = pair.paths_from_archive(root, rel)
enc = [So400mCompactEncoder.from_checkpoint(snap, c, resolution=384, output_dim=512, device=dev, fp16_native=True, fp8=True) for c in (ca, cb)]
def batches(ids): return [enc[0].processor(images=[Image.open(paths[i]).convert("RGB") for i in ids[k:k + 64]], return_tensors="pt")["pixel_values"].to(dev) for k in range(0, len(ids), 64)]
calib, test = batches(fit[:512]), batches(fit[512:1024])
lin = [m for e in enc for m in e.vision.modules() if isinstance(m, _FP8Linear)]
amax = {id(m): torch.zeros((), device=dev) for m in lin}
def track(m, a):
    amax[id(m)].copy_(torch.maximum(amax[id(m)], a[0].abs().amax().float()))  # returns None: keep inputs
hooks = [m.register_forward_pre_hook(track) for m in lin]
ens = SharedTrunkEnsemble(enc)
with torch.inference_mode():
    for px in calib: ens.encode_pixels(px)
    ref = torch.cat([ens.encode_pixels(px).float() for px in test])
for h in hooks: h.remove()

def static_forward(self, x):
    x2 = x.reshape(-1, x.shape[-1]); s = self.static_scale.expand(x2.shape[0], 1)
    q = (x2.float() * self.static_inv).clamp(-self.fmax, self.fmax).to(torch.float8_e4m3fn)
    y = torch._scaled_mm(q, self.w_t, scale_a=s, scale_b=self.w_scale, bias=self.bias, out_dtype=torch.float16)
    return y.reshape(*x.shape[:-1], self.out_features)

def timed(fn, px):
    for _ in range(3): fn(px)
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(6): fn(px)
    torch.cuda.synchronize(); return 6 * px.shape[0] / (time.perf_counter() - t)

out = {"margin": margin}
for m in enc:
    ls = m.vision.encoder.layers
    for i in range(len(ls)): ls[i] = torch.compile(ls[i], mode="max-autotune-no-cudagraphs")
    m.vision.head = torch.compile(m.vision.head, mode="max-autotune-no-cudagraphs")
with torch.inference_mode():
    out["dynamic_img_s"] = timed(ens.encode_pixels, calib[0])
for m in lin:
    s = (margin * amax[id(m)] / m.fmax).clamp(min=1e-12).reshape(1, 1)
    m.register_buffer("static_scale", s); m.register_buffer("static_inv", 1 / s)
    m.forward = static_forward.__get__(m)
torch._dynamo.reset()
for m in enc:  # recompile with the static forward
    for i in range(len(m.vision.encoder.layers)): m.vision.encoder.layers[i] = torch.compile(m.vision.encoder.layers[i]._orig_mod, mode="max-autotune-no-cudagraphs")
    m.vision.head = torch.compile(m.vision.head._orig_mod, mode="max-autotune-no-cudagraphs")
with torch.inference_mode():
    out["static_img_s"] = timed(ens.encode_pixels, calib[0])
    st = torch.cat([ens.encode_pixels(px).float() for px in test])
out["speedup"] = out["static_img_s"] / out["dynamic_img_s"]
out["max_cos_drift_vs_dynamic"] = float((1 - (st * ref).sum(1)).abs().max()); out["mean_cos_drift"] = float((1 - (st * ref).sum(1)).abs().mean())
out["decision"] = "GATE" if out["speedup"] >= 1.10 else "KILL"
print("RESULT", json.dumps(out))
