#!/usr/bin/env python3
"""Efficiency falsifier on the served FP8 shared-trunk SOP ensemble (GPU-only, b64, random pixels for speed;
512 TRAIN-FIT images for drift). Arms: per-layer compile (served), segment compile (trunk / each member upper stack +
post-LN + pooling head in ONE graph, under no_grad so residual adds fuse across layer boundaries), and each with cuDNN
SDPA. Pre-registered keep: >= 5 % GPU-only img/s over per-layer and max cos drift vs per-layer <= 0.02. Else KILL."""
import json, sys, time
from pathlib import Path
import numpy as np, torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_sop_siglip2_substrate_pair as pair
from sfora.representation_ceiling import deterministic_class_partition
from sfora.so400m_compact_serving import So400mCompactEncoder
from sfora.so400m_shared_trunk_ensemble import SharedTrunkEnsemble, _hidden

snap, ca, cb, arch, root = map(Path, sys.argv[1:6]); dev = torch.device("cuda")
with np.load(arch, allow_pickle=False) as z: rel = np.asarray(z["train_relative_paths"]).astype(str); lab = np.asarray(z["train_labels"], dtype=np.int64)
held = set(deterministic_class_partition(tuple(map(int, lab)), fit_fraction=0.9, seed=179019).validation_row_indexes)
fit = [i for i in np.random.default_rng(179019).permutation(len(lab)) if i not in held][:512]
paths = pair.paths_from_archive(root, rel)
enc = [So400mCompactEncoder.from_checkpoint(snap, c, resolution=384, output_dim=512, device=dev, fp16_native=True, fp8=True) for c in (ca, cb)]
test = [enc[0].processor(images=[Image.open(paths[i]).convert("RGB") for i in fit[k:k + 64]], return_tensors="pt")["pixel_values"].to(dev) for k in range(0, 512, 64)]
px = torch.randn(64, 3, 384, 384, device=dev)
M = "max-autotune-no-cudagraphs"


def timed(fn):
    for _ in range(3): fn(px)
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(8): fn(px)
    torch.cuda.synchronize(); return 8 * 64 / (time.perf_counter() - t)


def run_trunk(px):
    f = enc[0]; h = f.vision.embeddings(px.half(), interpolate_pos_encoding=f.interpolate)
    for l in f.vision.encoder.layers[:13]: h = _hidden(l(h, None))
    return h


def make_upper(m):
    def up(h):
        for l in m.vision.encoder.layers[13:]: h = _hidden(l(h, None))
        pooled = m.vision.head(m.vision.post_layernorm(h))
        return torch.nn.functional.normalize(m.head(torch.nn.functional.normalize(pooled.float(), dim=1)), dim=1)
    return up


seg_trunk = torch.compile(run_trunk, mode=M); seg_up = [torch.compile(make_upper(m), mode=M) for m in enc]


def segment(px):
    with torch.no_grad():
        h = seg_trunk(px); return torch.cat([u(h) for u in seg_up], 1) / 2 ** 0.5


out = {}
with torch.no_grad():
    out["segment_img_s"] = timed(segment); seg_rows = torch.cat([segment(p).float() for p in test])
    with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
        torch._dynamo.reset(); seg_trunk = torch.compile(run_trunk, mode=M); seg_up = [torch.compile(make_upper(m), mode=M) for m in enc]
        try: out["segment_cudnn_img_s"] = timed(segment)
        except Exception as e: out["segment_cudnn_error"] = repr(e)[:200]
torch._dynamo.reset()
ens = SharedTrunkEnsemble(enc, compile=True)
with torch.inference_mode():
    out["per_layer_img_s"] = timed(ens.encode_pixels); ref = torch.cat([ens.encode_pixels(p).float() for p in test])
out["segment_speedup"] = out["segment_img_s"] / out["per_layer_img_s"]
if "segment_cudnn_img_s" in out: out["segment_cudnn_speedup"] = out["segment_cudnn_img_s"] / out["per_layer_img_s"]
out["segment_max_cos_drift"] = float((1 - (seg_rows * ref).sum(1)).abs().max())
best = max(out.get("segment_speedup", 0), out.get("segment_cudnn_speedup", 0))
out["decision"] = "KEEP" if best >= 1.05 and out["segment_max_cos_drift"] <= 0.02 else "KILL"
print("RESULT", json.dumps(out))
