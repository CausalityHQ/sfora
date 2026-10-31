# Fused FP8 encoder kernels, SOP shared-trunk ensemble: KILL (1.17× < 1.25×)

Base b56cee66. GB10 sm_121, torch 2.12.1+cu130, Triton 3.7.1. Bench `scripts/bench_sop_fused_fp8_v1.py`
(log `/home/riomus/runs/sfora-fused-fp8-v1/run3.log`), code `src/sfora/so400m_fused_fp8.py`.
2-member ensemble (13 trunk + 2×14 upper layers), b64 random pixels for speed, 512 TRAIN-FIT SOP images (held split
179019 excluded) for drift. Pre-registered keep: ≥ 1.25× GPU-only img/s and max cos drift ≤ 0.02.

| arm | b64 img/s (rerun) | speedup | max / mean cos drift vs served | b1 p50 ms | peak GiB |
|---|---|---|---|---|---|
| served (per-layer Inductor) | 67.88 | 1.00 | — | 19.27 | 3.39 |
| fused (Triton elementwise, cuBLAS GEMMs) | 77.87 (77.72) | 1.147 | 0.0121 / 0.0018 | 15.72 | 3.94 |
| fused_mx (+ Triton fc1·GELU→MXFP8, MXFP8 fc2) | **79.48** (78.72) | **1.171** | 0.0110 / 0.0023 | 15.76 | 3.72 |

Decision: **KILL** (drift passes, speed misses). Run 2 reproduced within 0.3 %. Batch-1 latency improves 18 %.

## What was fused
- `fused`: per layer 3 Triton kernels + 4 cuBLAS GEMMs. `add_ln_quant` = residual add (deferred from the previous
  fc2/o-proj, so no standalone add) + LayerNorm + per-row amax + e4m3 cast; `attn_quant` reads the SDPA output in place
  (no transpose copy) + amax + cast; `gelu_quant` = tanh-GELU + amax + cast. q/k/v run as ONE 1152→3456 GEMM with the
  identical e4m3 weights/scales/bias. fp16 rounding points copied from the served module.
- `fused_mx`: additionally fc1 GEMM + bias + GELU + MXFP8 (1×32, e8m0, cuBLAS-swizzled) quantisation in one Triton
  kernel; fc2 runs as cuBLAS MXFP8 on the served e4m3 weight codes unchanged (per-row scale split into a power-of-two
  block scale and a [1,2) residual applied with the bias inside the next `add_ln_quant`). Kernel check: MLP rel. error
  vs fp32 0.02659 (MX) vs 0.02658 (served rowwise).

## Why it stops at ~1.17× (profile, fused_mx, per layer pass ≈ 19.6 ms)
fc1·GELU Triton 25.5 %, cuBLAS qkv+o 18.3 %, flash attention 18.1 %, add_ln_quant 16.1 %, MXFP8 fc2 12.6 %, attn_quant 3.3 %.
- The elementwise kernels are at the memory roof: add_ln_quant moves ~375 MB in 1.57 ms (~240 GB/s), gelu_quant
  ~600 MB in 2.46 ms (~244 GB/s) on GB10's ~273 GB/s LPDDR5x. The 1.79× "GEMM+attention only" ceiling assumes this
  traffic is free; on this part it cannot be, unless it moves into GEMM epilogues.
- GEMM-epilogue fusion needs a custom GEMM, and Triton 3.7.1 cannot issue FP8 MMA on sm_121: `tl.dot` on e4m3 lowers to
  `cvt e4m3x2→f16x2` + `mma.m16n8k16.f16` (≈ 102 TFLOPS vs cuBLAS FP8 131–172), `tl.dot_scaled` lowers to bf16
  emulation (≈ 82 TFLOPS). The fc1·GELU fusion therefore only nets ~0.5 ms/layer (5.1 ms vs 3.2 + 2.5).
- Attention: flash on strided q/k/v 3.48 ms vs contiguous 2.93 ms (the copy costs more), cuDNN 3.9–4.0 ms, efficient 9.0 ms.

## Tried, not helping
Triton FP8 GEMM autotune (8 configs, best 128×128×64 s4 w4) ~102 TF; `tl.dot_scaled` e4m3 (emulated, slower);
cuDNN SDPA; contiguous q/k/v; Triton o-proj with residual epilogue (estimated +0.26 ms GEMM vs −0.4 ms traffic, not built).
Remaining lever (not attempted, out of brief scope): CUTLASS sm_120 FP8 GEMM with custom epilogues (GELU/quant, residual)
as a compiled extension — the only route to native-FP8 fused epilogues on this part.

Assumptions: drift is fused-vs-served on the same process/weights; served arm measured after the fused arms (rerun of
fused arms after served confirms no order effect); GPU-only timing excludes preprocessing.

## Root held quality gate (scripts/check_sop_fused_fp8_held_v1.py, pre-registered order mx → plain): FAIL
| arm | held R@1 | mAP@R | e2e img/s (8 workers, full TRAIN, incl. warmup) | peak GB |
|---|---|---|---|---|
| mx | 95.129 | 84.36 | 76.3 | 2.68 |
| plain | 95.061 | 84.26 | 76.2 | 2.73 |
Gate ≥ 95.166 (fp32 ensemble 95.266 − 0.10); served per-layer FP8: 95.197 / 84.34 @ 63.7 img/s. Not served.
Note for future gates: all FP8 variants measured (per-layer 95.197, segment 95.078, mx 95.129, plain 95.061) fall in a
±0.07 band, so a −0.10 point threshold vs fp32 is inside FP8 rounding noise. Future serving numerics changes should be
pre-registered with a paired product-bootstrap non-inferiority test against the served path, not a point threshold.
