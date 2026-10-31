# Shared-trunk serving of the SOP 2-seed So400m@384/512 ensemble — KEEP (exact)
Code 2ad1a1e1 (`src/sfora/so400m_shared_trunk_ensemble.py`, `scripts/check_sop_shared_trunk_ensemble_v1.py`).
Premise verified: embeddings + blocks 0–12 bitwise equal in seeds 179024/179025 (211/211 tensors).
1,024 SOP TRAIN images, batch 64, best of 3, GB10:
| precision | separate (2 towers) img/s | shared trunk img/s | speedup | max cos drift |
|---|---|---|---|---|
| fp16 native | 28.6 | 38.3 | 1.34× | 4.8e-7 |
| fp8 linears (no compile) | 14.8 | 19.8 | 1.34× | 4.8e-7 |
Caveats: preprocessing is single-threaded in this harness (absolute img/s understated vs worker-fed serving; the
separate arm also preprocesses twice). Uncompiled FP8 is slower than fp16 here — FP8 needs compile to pay off.
Output equal to float rounding ⇒ official 94.129 carries over unchanged.

## End-to-end (8 decode/preprocess workers, 2,048 images, batch 64; scripts/bench_sop_shared_trunk_e2e_v1.py)
| arm | img/s | warmup s | peak GB | max cos drift vs fp16 eager |
|---|---|---|---|---|
| fp16 eager | 39.8 | 53 | 3.15 | — |
| fp16 compiled (per-layer) | 42.8 | 69 | 3.05 | 9e-5 |
| **fp8 compiled (per-layer)** | **63.7** | 45 | 3.09 | 0.041 |
GPU-bound: GPU-only single tower 62.3 img/s vs shared ensemble 41.0 (= 62/1.52, as predicted).
Compiling the inference_mode method crashed (storage weakref); compiling layer modules in place works.

## FP8 held quality gate (pre-registered, scripts/check_sop_shared_trunk_fp8_held_v1.py): PASS
Held R@1 95.197 (gate ≥ 95.166; fp32 incumbent 95.266, −0.07), mAP@R 84.34 (−0.13). FP8-compiled shared-trunk
ensemble is the serving Pareto point: 63.7 img/s e2e for the 2-seed SOP ensemble (vs 39.8 fp16, ~30 for two separate
fp16 towers). Official TEST stays 94.129 for fp32 arithmetic; FP8 official not read (serving drift, not selection).

## Official SOP TEST read of the served FP8 configuration: ACCEPT
scripts/evaluate_sop_shared_trunk_fp8_official_v1.py: R@1 94.042 (gate ≥ 94.029; fp32 ensemble 94.129, −0.09),
mAP@R 82.95, 60,502 images, concat/√2 packed int8 1,026 B/row. Above the fp32 single model (93.967).
Served SOP frontier: 94.042 at 63.7 img/s e2e on one GB10.

## Batch-size sweep (GPU-only, FP8 compiled shared trunk): not a lever
b64 67.8 img/s (2.16 GB), b128 67.9 (3.02 GB), b256 66.3 (5.09 GB). Compute-saturated; e2e 63.7 = 94 % of GPU-only.

## GPU profile of the served FP8 ensemble (b64, compiled)
cutlass FP8 GEMM 41 %; fused dynamic-quant kernels (amax/div/cast fused with GELU/LN) ~31 %; flash attention (fp16) 15 %;
residual add 10 %; rest < 3 %.

## Static per-layer FP8 activation scales (scripts/bench_sop_fp8_static_scale_v1.py): KILL
Calibrated on 512 TRAIN-FIT images (margin 1.0). GPU-only 68.1 → 72.6 img/s = 1.065× (bar 1.10×); cos drift vs dynamic
max 0.0145 / mean 0.0033. The amax reduction is fused with neighbouring elementwise work, so removing it saves little.

## Segment compile (trunk + per-member upper graphs): FAIL on quality gate → opt-in only
GPU-only 1.057× (67.6 → 71.4), e2e 63.7 → 66.7 img/s, cos drift vs per-layer FP8 0.006; but held R@1 95.078 < gate
95.166 (per-layer FP8: 95.197). FP8 rounding differences flip ~7 held queries. Served default stays per-layer.

## Fusion / megakernel ceiling (scripts/bench_sop_fusion_ceiling_v1.py): BUILD
Served b64 70.9 img/s GPU-only; GEMM+attention-only floor 126.8 img/s → fusion ceiling 1.79×. FP8 GEMM 149.5 TFLOPS
achieved; per layer pass GEMM 9.5 ms + attention 2.8 ms (b64). b1 p50 18.7 ms (≈53 img/s: batch 1 is not launch-bound).
CUDA-graph arm errored (trunk output overwritten between replays; needs a clone between segments).
Any fused kernel changes FP8 rounding, so it must pass the same held gate (≥ 95.166) before serving.
