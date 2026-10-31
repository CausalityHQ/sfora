#!/usr/bin/env python3
"""Time each CUTLASS fused GEMM at the b64 encoder shape (M = 64*729) for one or more built libraries.
Usage: PYTHONPATH=src tune_gemms.py LIB.so [LIB.so ...]  -> ms per call: qkv, o+res, fc1+GELU+MX, MX fc2+res."""
import json, os, sys
import torch

M, dev, e4 = 64 * 729, torch.device("cuda"), torch.float8_e4m3fn


def bench(fn, n=20):
    for _ in range(3): fn()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(n): fn()
    b.record(); torch.cuda.synchronize(); return a.elapsed_time(b) / n


r = lambda *s: (torch.randn(*s, device=dev) * 0.5).to(e4)
sw = {n: torch.rand(n, device=dev) / 100 for n in (1152, 3456, 4352)}
bb = {n: torch.randn(n, device=dev, dtype=torch.float16) for n in (1152, 3456, 4352)}
q, s1, h = r(M, 1152), torch.rand(M, 1, device=dev) / 100, torch.randn(M, 1152, device=dev, dtype=torch.float16)
w3, w1, wo, w2 = r(3456, 1152), r(4352, 1152), r(1152, 1152), r(1152, 4352)
qm, e = r(M, 4352), torch.full((-(-M // 128) * 128 * 136,), 120, device=dev, dtype=torch.uint8)
ew, one = torch.full((1152 * 136,), 120, device=dev, dtype=torch.uint8), torch.ones(1, device=dev)
import sfora.so400m_cutlass_fp8 as c
for lib in sys.argv[1:]:
    os.environ["SFORA_CUTLASS_FP8_LIB"] = lib
    c._LIB.clear()
    t = {"qkv": bench(lambda: c.linear(q, s1, w3, sw[3456], bb[3456])),
         "o_res": bench(lambda: c.linear_res(q, s1, wo, sw[1152], bb[1152], h)),
         "fc1_gelu_mx": bench(lambda: c.fc1_gelu_mx(q, s1, w1, sw[4352], bb[4352], one)),
         "fc2_mx_res": bench(lambda: c.mx_res(qm, e, w2, ew, sw[1152], bb[1152], h))}
    print(os.path.basename(lib), json.dumps({k: round(v, 3) for k, v in t.items()}), flush=True)
