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
