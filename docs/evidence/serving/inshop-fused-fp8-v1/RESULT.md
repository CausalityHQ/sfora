# In-Shop serving: fused FP8 encoder on the 97.039 incumbent — KEEP (new In-Shop serving path)
Pre-registered: scripts/check_inshop_serving_v2.py (keep iff net >= −3/14,218 vs the official read AND e2e >= 1.05×
served FP8). Model sfora-inshop-so400m-full-c384-179024-2000-v1 (official 97.039). Full official In-Shop pipeline:
14,218 queries + 12,612 gallery, 8 decode workers, packed int8, So400mCompactIndex top-1.
| encoder | serving R@1 | net vs read | e2e img/s | b1 p50 / p99 ms | decision |
|---|---|---|---|---|---|
| served FP8 (per-layer compiled vision) | 96.955 | −12 | 102.4 | 19.5 / 23.2 | FAIL (parity) |
| fused_mx (src/sfora/so400m_fused_fp8.py, single member) | 97.039 | 0 | 112.5 | 17.4 / 21.5 | KEEP |
fused_plain not run (pre-registered skip once fused_mx kept).
Finding: the per-layer FP8 path that passed on the 1000-update model (net +7) fails parity on the 2000-update
incumbent (net −12). The fused MXFP8-MLP path reproduces the official read exactly while being 9.9 % faster.
In-Shop served frontier: 97.039 @ 112.5 img/s e2e on one GB10.
