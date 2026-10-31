# In-Shop serving: CUTLASS fused-epilogue FP8 encoder — KEEP (new In-Shop serving path)
Pre-registered: scripts/check_inshop_serving_v3.py (keep iff net >= −3/14,218 vs the official read AND e2e >= 1.05×
fused_mx in the same session). Model sfora-inshop-so400m-full-c384-179024-2000-v1 (official 97.039); library
libcutlass_fp8.so sha256 bec8c18d26bb94e8… (default build of native/cutlass_fp8 at dabc7dac, CUTLASS 4.8.1, sm_121a).
| encoder | serving R@1 | net vs read | e2e img/s | b1 p50 / p99 ms | peak GB |
|---|---|---|---|---|---|
| fused_mx (previous serving) | 97.039 | 0 | 110.9 | 18.0 / 20.9 | 2.09 |
| **cutlass** | 97.018 | −3 (gate boundary) | **143.2** | 16.1 / 19.4 | 2.10 |
e2e 1.29× (GPU-only 1.27×). In-Shop served frontier: 97.018 serving / 97.039 official @ 143.2 img/s on one GB10.
Note: net −3 sits exactly on the pre-registered boundary; any further numerics change must re-run this gate.
