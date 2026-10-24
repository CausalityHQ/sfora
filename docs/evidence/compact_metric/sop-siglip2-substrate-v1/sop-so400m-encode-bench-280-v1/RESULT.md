# So400m@280 encode-variant benchmark (Loop P iteration 1) — GB10, synthetic 2048 rows

Harness: scripts/bench_sop_so400m_encode_variants_v1.py (cd73781f, contextless specialist 98aec147); checkpoint
res280 128-D (9e4a05db). GPU-resident pixels → unit 128-D (decode/packing excluded). Drift vs reference on synthetic
inputs only (not SOP ranking quality). Result sha f1e252e1….

| config (attention, weights, batch, compile) | GPU img/s | x ref | peak GB | packed-code identical | top-10 agreement |
|---|---|---|---|---|---|
| sdpa, fp32+autocast, 64, none (REFERENCE = training export) | 87.4 | 1.00 | 2.8 | 1.000 | 1.000 |
| sdpa, fp16 native, 64, none | 115.4 | 1.32 | 1.6 | 0.331 | 0.966 |
| sdpa, fp16 native, 64, max-autotune-no-cudagraphs | 127.1 | 1.45 | 1.6 | 0.416 | 0.978 |
| sdpa, fp32+autocast, 64, max-autotune | 116.7 | 1.34 | 2.6 | 0.506 | 0.982 |
| sdpa, bf16 native, any | 114–129 | ~1.4 | — | 0.000 | 0.78–0.85 (reject) |
| eager, fp32+autocast, 64, none | 53.0 | 0.61 | 3.7 | 0.474 | 0.979 |

Findings: sdpa (installed default) is required; batch >64 adds ≤3%; bf16 drifts heavily; fp16-native and compile
are the levers (1.3–1.45x) but change codes for ~60% of rows — admissible only if held per-query R@1 is
statistically unchanged on REAL images (next Loop P iteration). sdpa fp32 b128 was not bit-identical to b64.
