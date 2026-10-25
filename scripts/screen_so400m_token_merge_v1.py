#!/usr/bin/env python3
"""Training-free token-merging screen for the So400m@384 512-D encoder (efficiency lever, In-Shop TRAIN-held).

Pre-registered (committed before any number exists):
  model : In-Shop So400m@384 512-D pilot checkpoint (trained on fit products only, seed 179024), fp16 native.
  rows  : In-Shop TRAIN, screen partition (13,121 held queries vs all 25,882 rows, self excluded), packed int8.
  arms  : base (no merging) vs merge (bipartite soft matching, ToMe-style). After every encoder layer
          i >= START (4) of 27, merge R=24 token pairs by cosine similarity of the layer output.
          Merged tokens are averaged. The attention-pooling head sees the remaining tokens.
          729 -> 201 tokens at the last layer; about 0.69x linear FLOPs.
  rule  : CONTINUE iff merge GPU-only img/s (batch 64, synthetic, both compiled) >= 1.30x base
          AND paired held R@1 (merge - base, product bootstrap 5000) lower95 > -0.15 pp.
          CONTINUE => fine-tune with merging on (a later, separately pre-registered step); else KILL.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_so400m_serving_variants_v1 import Pixels  # noqa: E402
from screen_inshop_pilot_pair_v1 import paired  # noqa: E402
from sfora.joint_relational_compaction import pack_int8_unit_embeddings  # noqa: E402
from sfora.so400m_compact_serving import So400mCompactEncoder, So400mCompactIndex  # noqa: E402
from train_inshop_so400m_pilot import inshop_split  # noqa: E402

START, R = 4, 24


def merge(x: torch.Tensor, r: int) -> torch.Tensor:
    """ToMe bipartite soft matching: merge r tokens of set A (even) into their most similar B (odd) token."""
    m = F.normalize(x.float(), dim=-1)
    a, b = m[:, ::2], m[:, 1::2]
    node_max, node_idx = (a @ b.transpose(1, 2)).max(dim=-1)
    order = node_max.argsort(dim=-1, descending=True)[..., None]
    unm_idx, src_idx = order[:, r:], order[:, :r]
    dst_idx = node_idx[..., None].gather(1, src_idx)
    xa, xb = x[:, ::2], x[:, 1::2]
    c = x.shape[-1]
    unm = xa.gather(1, unm_idx.expand(-1, -1, c))
    src = xa.gather(1, src_idx.expand(-1, -1, c))
    dst = xb.scatter_reduce(1, dst_idx.expand(-1, -1, c), src, reduce="mean")
    return torch.cat([unm, dst], dim=1)


def install_merging(vision) -> None:
    for i, layer in enumerate(vision.encoder.layers):
        if i < START:
            continue
        forward = layer.forward

        def merged(hidden_states, *args, _forward=forward, **kwargs):
            out = _forward(hidden_states, *args, **kwargs)
            hs = out[0] if isinstance(out, tuple) else out
            hs = merge(hs, R)
            return (hs,) + tuple(out[1:]) if isinstance(out, tuple) else hs

        layer.forward = merged


@torch.inference_mode()
def gpu_img_s(vision, iters=10):
    x = torch.randn(64, 3, 384, 384, device="cuda", dtype=torch.float16)
    for _ in range(3):
        vision(pixel_values=x)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        vision(pixel_values=x)
    torch.cuda.synchronize()
    return 64 * iters / (time.perf_counter() - t)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--model-snapshot", type=Path, required=True)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    paths, labels, _, held, _, _ = inshop_split(a.dataset_root)
    result = {"schema": "sfora-so400m-token-merge-screen-v1", "start": START, "r": R, "arms": {}}
    per_query = {}
    for name in ("base", "merge"):
        enc = So400mCompactEncoder.from_checkpoint(a.model_snapshot, a.checkpoint, resolution=384, output_dim=512,
                                                   device=torch.device("cuda"), fp16_native=True)
        if name == "merge":
            install_merging(enc.vision)
        vision = torch.compile(enc.vision, mode="max-autotune-no-cudagraphs", dynamic=False)
        speed = gpu_img_s(vision)

        @torch.inference_mode()
        def forward(pixels):
            pooled = vision(pixel_values=pixels.to("cuda", non_blocking=True).half()).pooler_output
            with torch.autocast(device_type="cuda", enabled=False):
                return F.normalize(enc.head(F.normalize(pooled.float(), dim=1)), dim=1).cpu()

        loader = DataLoader(Pixels(list(paths), enc.processor), batch_size=None, num_workers=8, prefetch_factor=4,
                            pin_memory=True)
        t = time.perf_counter()
        values = torch.cat([forward(b) for b in loader])
        e2e = len(paths) / (time.perf_counter() - t)
        packed = pack_int8_unit_embeddings(values)
        index = So400mCompactIndex(packed, device=torch.device("cuda"))
        q = type(packed)(packed.codes[held].contiguous(), packed.inverse_norms[held].contiguous())
        top1, _ = index.search(q, 1, exclude=held)
        per_query[name] = (labels[top1[:, 0]] == labels[held]).astype(np.float64)
        result["arms"][name] = {"gpu_img_s": speed, "e2e_img_s": e2e, "held_r1": float(per_query[name].mean())}
        print(json.dumps({name: result["arms"][name]}), flush=True)
        del enc, vision
        torch.compiler.reset()
        torch.cuda.empty_cache()
    result["r1_merge_minus_base"] = paired(per_query["merge"], per_query["base"], labels[held])
    result["speedup_gpu"] = result["arms"]["merge"]["gpu_img_s"] / result["arms"]["base"]["gpu_img_s"]
    result["decision"] = ("CONTINUE" if result["speedup_gpu"] >= 1.30 and result["r1_merge_minus_base"]["lower95"] > -0.15
                          else "KILL")
    a.output.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
