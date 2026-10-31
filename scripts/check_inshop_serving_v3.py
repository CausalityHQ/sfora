#!/usr/bin/env python3
"""v3 (2026-10-31): v2 + encoder `cutlass` (src/sfora/so400m_cutlass_fp8.py, lib sha256 bec8c18d...). Pre-registered:
KEEP cutlass iff net >= -3/14,218 vs the official read AND e2e img/s >= 1.05x fused_mx measured in the same session.

v2 (2026-10-31): v1 + --encoder {served,fused_plain,fused_mx} (src/sfora/so400m_fused_fp8.py, single member).
Pre-registered for the fused arms: KEEP iff v1 PASS rule (net >= -3/14,218 vs the official read) AND e2e img/s >= 1.05x
the served FP8 arm on the same model; order: served, fused_mx, then fused_plain only if fused_mx is not kept.

Loop P gate for the SHIPPED In-Shop model: full serving pipeline on the official In-Shop images.

The library path is So400mCompactEncoder.from_checkpoint(fp16_native=True). The vision tower is
torch.compile'd (max-autotune-no-cudagraphs) and fed by 8 preprocessing workers: the 384 px keep from
so400m-p-variants-v1. Query and gallery images are encoded, packed int8, and searched with
So400mCompactIndex (top-1).

This is a parity/efficiency check of the already-read model, not a selection. Pre-registered:
  PASS iff the R@1 net change vs that model's single official read (official_packed.npz, the
  reference fp32-autocast path) is >= -3/14,218, AND end-to-end img/s > 46.4 (the bitwise 384 path).
Also reported: per-query flips, search seconds, and single-image p50/p99 latency (fp16+compile, b1).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_so400m_serving_variants_v1 import Pixels  # noqa: E402
from sfora.joint_relational_compaction import pack_int8_unit_embeddings  # noqa: E402
from sfora.so400m_compact_serving import So400mCompactEncoder, So400mCompactIndex  # noqa: E402
from sfora.unicom_inshop import parse_inshop_partition  # noqa: E402
from torch.nn import functional as F  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--model-snapshot", type=Path, required=True)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--resolution", type=int, choices=(280, 384), default=384)
    p.add_argument("--encoder", choices=("served", "fused_plain", "fused_mx", "cutlass"), default="served")
    p.add_argument("--fp8", action="store_true", help="dynamic FP8 vision linears (kept on SOP TRAIN-held 95.1 img/s, net -2); same PASS rule")
    a = p.parse_args()
    if a.output.exists() or not torch.cuda.is_available():
        raise SystemExit("output exists or CUDA unavailable")
    ref = np.load(a.run_dir / "official_packed.npz")
    records = parse_inshop_partition(a.dataset_root)
    query = [r for r in records if r.split == "query"]
    gallery = [r for r in records if r.split == "gallery"]
    enc = So400mCompactEncoder.from_checkpoint(a.model_snapshot, a.run_dir / "checkpoint.pt", resolution=a.resolution,
                                               output_dim=512, device=torch.device("cuda"), fp16_native=True,
                                               fp8=a.fp8 and a.encoder != "served")  # fused arms need _FP8Linear
    if a.fp8 and a.encoder == "served":
        from bench_so400m_fp8_linear_v1 import swap
        swap(enc.vision)
    vision = torch.compile(enc.vision, mode="max-autotune-no-cudagraphs")
    fused = None
    if a.encoder != "served":
        if not a.fp8:
            raise SystemExit("fused encoders need --fp8")
        from sfora.so400m_fused_fp8 import FusedSharedTrunkEnsemble
        if a.encoder == "cutlass":
            from sfora.so400m_cutlass_fp8 import CutlassSharedTrunkEnsemble
            fused = CutlassSharedTrunkEnsemble([enc])
        else:
            fused = FusedSharedTrunkEnsemble([enc], mx_mlp=a.encoder == "fused_mx")

    @torch.no_grad()
    def forward(pixels: torch.Tensor) -> torch.Tensor:
        if fused is not None:
            return fused.encode_pixels(pixels.to("cuda", non_blocking=True)).float().cpu()
        pooled = vision(pixel_values=pixels.to("cuda", non_blocking=True).half(),
                        interpolate_pos_encoding=enc.interpolate).pooler_output
        with torch.autocast(device_type="cuda", enabled=False):
            return F.normalize(enc.head(F.normalize(pooled.float(), dim=1)), dim=1).cpu()

    def encode(rows):
        loader = DataLoader(Pixels([r.image_path for r in rows], enc.processor), batch_size=None, num_workers=8,
                            prefetch_factor=4, pin_memory=True)
        return torch.cat([forward(batch) for batch in loader])

    forward(Pixels([query[0].image_path] * 64, enc.processor)[0])  # compile/autotune outside timing
    torch.cuda.synchronize()
    started = time.perf_counter()
    qv, gv = encode(query), encode(gallery)
    encode_seconds = time.perf_counter() - started
    qp, gp = pack_int8_unit_embeddings(qv), pack_int8_unit_embeddings(gv)
    index = So400mCompactIndex(gp, device=torch.device("cuda"))
    started = time.perf_counter()
    top1, _ = index.search(qp, 1)
    search_seconds = time.perf_counter() - started
    ql, gl = ref["query_labels"], ref["gallery_labels"]
    observed = (gl[top1[:, 0]] == ql).astype(np.float64)
    # reference per-query top-1 from the official read's saved codes (same exact scoring)
    rq = torch.from_numpy(ref["query_codes"]).float().cuda()
    rg = torch.from_numpy(ref["gallery_codes"]).float().cuda()
    rqn, rgn = torch.from_numpy(ref["query_inv"]).cuda(), torch.from_numpy(ref["gallery_inv"]).cuda()
    expected = np.concatenate([
        (gl[torch.argsort((rq[i:i + 64] @ rg.T) * rqn[i:i + 64, None] * rgn[None], dim=1, descending=True,
                          stable=True)[:, 0].cpu().numpy()] == ql[i:i + 64]) for i in range(0, len(rq), 64)
    ]).astype(np.float64)
    images = [Image.open(r.image_path).convert("RGB") for r in query[:220]]
    lat = []
    for i, image in enumerate(images):
        torch.cuda.synchronize()
        t = time.perf_counter()
        forward(enc.processor(images=[image], return_tensors="pt")["pixel_values"])
        if i >= 20:
            lat.append((time.perf_counter() - t) * 1e3)
    img_s = (len(query) + len(gallery)) / encode_seconds
    net = int(observed.sum() - expected.sum())
    result = {"schema": "sfora-inshop-serving-v3", "encoder": a.encoder, "run_dir": str(a.run_dir), "queries": len(query),
              "gallery": len(gallery), "official_read_r1": float(expected.mean()),
              "serving_r1": float(observed.mean()), "net_correct_vs_read": net,
              "fixed": int(((observed == 1) & (expected == 0)).sum()),
              "broken": int(((observed == 0) & (expected == 1)).sum()),
              "encode_images_per_second": img_s, "encode_seconds": encode_seconds,
              "search_seconds_all_queries": search_seconds,
              "latency_b1_ms": {"p50": float(np.percentile(lat, 50)), "p99": float(np.percentile(lat, 99))},
              "gallery_index_bytes": len(index.to_bytes()), "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
              "resolution": a.resolution, "fp8": a.fp8,
              "decision": "PASS" if net >= -3 and img_s > 46.4 else "FAIL", "torch": torch.__version__}
    a.output.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
