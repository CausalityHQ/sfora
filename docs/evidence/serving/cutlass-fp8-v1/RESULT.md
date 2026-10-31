# CUTLASS fused-epilogue FP8 GEMMs, single In-Shop So400m@384 member: KEEP (1.27×, max drift 0.0049)

Code commit 9fc43b84 (base 20be5c91). GB10 sm_121, torch 2.12.1+cu130, Triton 3.7.1, nvcc 13.3 (`cuda-13.3-complete` +
CCCL headers from `cuda133-root`), CUTLASS 4.8.1 @ e13f5cd4 (cloned into the run dir). Code: `src/sfora/so400m_cutlass_fp8.py`,
`native/cutlass_fp8/{cutlass_fp8.cu,build.sh,tune_gemms.py}`. Bench: `scripts/bench_cutlass_fp8_v1.py`; run from a
`git archive` of 9fc43b84 in `/home/riomus/runs/sfora-cutlass-fp8-v1/repo-9fc43b84`, log `.../run3.log`
(run2.log = same code before the qkv default flip, reproduces within 0.5 %). Checkpoint
`sfora-inshop-so400m-full-c384-179024-2000-v1/checkpoint.pt` (single member, out_dim 512).
Pre-registered keep (brief): new/current GPU-only b64 img/s ≥ 1.20 and max cos drift ≤ 0.01 vs current fused (mx).

| arm | b64 img/s (3 interleaved repeats) | median | speedup | max / mean cos drift vs current | b1 p50 ms | peak GiB |
|---|---|---|---|---|---|---|
| current fused mx (`FusedSharedTrunkEnsemble([enc], mx_mlp=True)`) | 120.86 / 120.18 / 120.39 | 120.39 | 1.000 | (self 2.4e-7 / 9e-9) | 10.28 | 3.71 |
| **cutlass** (`CutlassSharedTrunkEnsemble([enc])`) | 153.32 / 152.84 / 152.05 | **152.84** | **1.270** | **0.00492 / 0.00154** | **9.29** | **3.56** |
| cutlass, cuBLAS qkv (`qkv_cutlass=False`) | 148.20 / 147.99 / 148.79 | 148.20 | 1.231 | 0.00492 / 0.00154 | 9.44 | 3.56 |

Decision: **KEEP** (1.270 ≥ 1.20, drift 0.0049 ≤ 0.01). Batch-1 p50 −9.6 %, peak memory −0.15 GiB.
Drift: 512 In-Shop TRAIN images (`parse_inshop_partition`, split == "train", `default_rng(0)` permutation, first 512),
drift = 1 − cos(new, current) on the unit embeddings. Speed: random pixels, GPU-only (`encode_pixels`, sync-timed, 8
batches per repeat after 3 warmup), arms interleaved per repeat.

## What was fused (per layer, b64 = 46,656 rows)
CUTLASS 3.x sm120 TMA warp-specialized FP8 GEMMs (native e4m3 / MXFP8 MMA) with epilogue visitor trees, called via ctypes:
- **fc1 + GELU + MXFP8**: rowwise e4m3 GEMM, epilogue = acc·s_a[m]·s_w[n] + bias → fp16 → tanh-GELU (exact-exp formula,
  as Triton) → fp16 → 1×32 amax → e8m0 scale (`Sm120BlockScaleFactorRowStore`, 128×4-swizzled = cuBLAS layout) → e4m3
  codes. Replaces the Triton fc1·GELU·MX kernel (fp16-emulated MMA): 5.20 → 3.44 ms.
- **fc2 (MXFP8) + residual**: block-scaled GEMM on the served e4m3 codes, epilogue = fp16(acc·r[n] + bias) + h (C operand).
  2.56 ms cuBLAS → 2.63 ms, but the deferred affine + residual add no longer go through `add_ln_quant`.
- **o-proj + residual**: rowwise GEMM, epilogue = fp16(acc·s_a·s_w + bias) + h. No standalone d tensor.
- **qkv**: same rowwise kernel without residual (bitwise equal to cuBLAS output), 128×128×64 pingpong tile beats
  torch's rowwise kernel (qkv+o: 3.53 → 2.15 + 1.19 ms).
- Net effect: `add_ln_quant` (both calls) 3.07 → 1.29 ms (reads h only, writes fp8 rows; no d read, no h write).
  Profile per layer: fc1 3.44, flash 3.67, fc2 2.63, qkv 2.15, add_ln_quant 1.29, o 1.19, attn_quant 0.65 ms.

Kernel checks (M = 4096 and 4100 for the tile residue, random weights): linear and linear+res max |Δ| = 0 vs cuBLAS
(+ fp16 add); fc1 dequantised rel. error vs fp32 GELU 0.02657 = Triton's 0.02657 exactly, e8m0 scales identical on every
non-padding block (0.9926 = 135/136; the padding block encodes zero differently), padding codes 0; MX fc2+res vs cuBLAS
MXFP8 + affine + add max |Δ| 0.0078 at |y| ≈ 16 (1 fp16 ulp, fma order).

## Tuning / what did not help (`tune_gemms.py`, ms per GEMM at M = 46,656)
| config (all four GEMMs) | qkv | o+res | fc1+GELU+MX | fc2 MX+res |
|---|---|---|---|---|
| cooperative 128×128×128 (CUTLASS auto) | 2.63 | 1.34 | 4.00 | 2.76 |
| pingpong 128×128×128 | 2.37–2.45 | 1.25–1.28 | 3.08–3.23 | 2.65–2.74 |
| pingpong 128×128×64 (dense only) | **2.11** | 1.30 | 3.20 | — |
| pingpong 128×64×128 | 2.49 | 1.24 | 3.66 | — |
| pingpong 64×256×128 | 2.39 | 1.25 | 3.42 | — |
| pingpong 128×256×64 (dense, spills) | 18.0 | 6.8 | 21.4 | — |

(Rows marked dense / with "—" kept fc2 at pingpong 128×128×128; its time there was 2.57–2.70 ms.)
- First build used CUTLASS's auto (cooperative) schedule: 1.166× (run1.log), a KILL. Pingpong is the lever: the second
  consumer warpgroup's mainloop hides the heavy GELU + amax + quant epilogue (fc1 −0.8 ms).
- K-tile 64 does not compile for the MX block-scaled GEMM (TMA static assert); 64×128 pingpong does not compile.
- Not attempted: LayerNorm in an epilogue (needs a full 1152-wide row reduction per tile; the Triton kernel already runs
  at ~250 GB/s), dropping `attn_quant` (flash output is fp16), writing q/k/v in a flash-friendly layout from the qkv
  epilogue (non-affine output layout), stream-K/split tiles, CuTe DSL.

## Assumptions / caveats
- Drift is new vs current fused (mx) in one process on the same weights; current's own run-to-run drift is 2.4e-7.
  The numerics differ from current only by fp32 op order in the epilogues (fma vs mul+add) — no rounding point moved.
- Not a serving decision: the In-Shop official/held parity gate (current serving path) was not run here; run it before
  swapping the served encoder. The 2-member SOP ensemble was not measured (same layers; speed should transfer).
- The library needs a CUTLASS checkout at build time (`build.sh CUTLASS_DIR`) and is loaded from
  `native/cutlass_fp8/libcutlass_fp8.so` or `SFORA_CUTLASS_FP8_LIB`; workspace is a process-global grow-only buffer
  (single device, single stream assumed).
