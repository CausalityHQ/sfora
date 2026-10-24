#!/usr/bin/env python3
"""Zero-quality GPU microbench: So400m@384 vision tower, fp16 vs dynamic-FP8 Linear layers (torch._scaled_mm).

Pre-registered: FP8 lane continues to a real held-quality gate iff the FP8 forward img/s at batch 64 is
>= 1.20x the fp16 forward. Both arms are compiled (max-autotune-no-cudagraphs) and fed synthetic pixels,
so this measures GPU forward only. Also reports the Linear share of fp16 forward time and the pooled
cosine drift FP8 vs fp16 on random pixels.
FP8 scheme: e4m3 weights with per-output-channel scales, e4m3 activations with per-row dynamic scales,
fp16 output, bias added in fp16.
"""
import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

E4M3 = torch.float8_e4m3fn
FMAX = torch.finfo(E4M3).max


class FP8Linear(nn.Module):
    def __init__(self, linear: nn.Linear) -> None:
        super().__init__()
        w = linear.weight.detach().float()
        scale = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / FMAX
        self.register_buffer("w_t", (w / scale).to(E4M3).t())  # (in, out), column-major as _scaled_mm wants
        self.register_buffer("w_scale", scale.t().contiguous())  # (1, out)
        self.bias = None if linear.bias is None else nn.Parameter(linear.bias.detach().half(), requires_grad=False)
        self.out_features = linear.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        x_scale = x2.abs().amax(dim=1, keepdim=True).float().clamp(min=1e-12) / FMAX
        xq = (x2.float() / x_scale).to(E4M3)
        y = torch._scaled_mm(xq, self.w_t, scale_a=x_scale, scale_b=self.w_scale, bias=self.bias,
                             out_dtype=torch.float16)
        return y.reshape(*shape[:-1], self.out_features)


def swap(module: nn.Module) -> int:
    count = 0
    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and child.in_features % 16 == 0 and child.out_features % 16 == 0:
            setattr(module, name, FP8Linear(child))
            count += 1
        elif not isinstance(child, nn.MultiheadAttention):  # MHA reads out_proj.weight directly; keep fp16
            count += swap(child)
    return count


@torch.inference_mode()
def bench(vision, pixels, iters=20):
    for _ in range(3):
        vision(pixel_values=pixels)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        out = vision(pixel_values=pixels).pooler_output
    torch.cuda.synchronize()
    return pixels.shape[0] * iters / (time.perf_counter() - t), out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-snapshot", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    from transformers import AutoModel

    torch.manual_seed(0)
    vision = AutoModel.from_pretrained(a.model_snapshot, local_files_only=True, dtype=torch.float16).vision_model
    vision = vision.cuda().eval()
    pixels = torch.randn(64, 3, 384, 384, device="cuda", dtype=torch.float16)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof, torch.inference_mode():
        vision(pixel_values=pixels)
    gemm = sum(e.device_time_total for e in prof.key_averages() if "gemm" in e.key.lower() or "matmul" in e.key.lower())
    total = sum(e.device_time_total for e in prof.key_averages())
    fp16_img_s, ref = bench(torch.compile(vision, mode="max-autotune-no-cudagraphs"), pixels)
    torch.compiler.reset()
    swapped = swap(vision)
    fp8_img_s, out = bench(torch.compile(vision, mode="max-autotune-no-cudagraphs"), pixels)
    cos = F.cosine_similarity(out.float(), ref.float(), dim=1)
    result = {"schema": "sfora-so400m-fp8-linear-bench-v1", "batch": 64, "resolution": 384,
              "fp16_compiled_img_s": fp16_img_s, "fp8_compiled_img_s": fp8_img_s, "speedup": fp8_img_s / fp16_img_s,
              "linears_swapped": swapped, "gemm_share_of_fp16_gpu_time": gemm / max(total, 1),
              "pooled_cos_min": float(cos.min()), "pooled_cos_mean": float(cos.mean()),
              "decision": "CONTINUE" if fp8_img_s >= 1.2 * fp16_img_s else "KILL",
              "device": torch.cuda.get_device_name(), "torch": torch.__version__}
    a.output.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
