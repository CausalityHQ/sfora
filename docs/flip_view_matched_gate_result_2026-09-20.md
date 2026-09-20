# Matched horizontal-flip view gate

The corrected, same-process horizontal-flip information gate is negative. It
compared freshly extracted identity embeddings with freshly extracted
identity-plus-horizontal-flip averages using the same UNICOM checkpoint,
transform, device, runtime, dataset rows, and process. This removes the
cross-runtime floating-point replay mismatch that invalidated the earlier
attempt without changing the frozen scientific thresholds.

| Decision dataset | Identity R@1 | Averaged-view R@1 | Delta | Mean identity/flip cosine |
|---|---:|---:|---:|---:|
| Food-101, 50 authorized official-test classes, 12,500 rows | 0.939120 | 0.940240 | +0.001120 | 0.986665 |
| Oxford-IIIT Pet, 19 authorized trainval classes, 1,889 rows | 0.958708 | 0.961355 | +0.002647 | 0.988384 |

The row-weighted pooled R@1 delta is `+0.001320`, below the frozen `+0.003`
minimum. The class-clustered bootstrap 95% lower bound is `-0.000139`, not
strictly positive. Both datasets also fail the frozen information condition
that mean identity-to-flip cosine be below `0.98`. The method therefore fails
all three decision conditions and is closed after its two authorized decision
datasets. No crop, scale, weighting, per-dataset selection, or additional view
variant is authorized.

This is a claim-ineligible fit-only mechanism gate, not an official benchmark
or a serving measurement. It used one bounded DGX process and completed in
`1407.726` seconds. It does not authorize library integration, 1M serving
benchmarks, CUDA/cuTile work, or a SOTA claim.

Authorities:

- sealed implementation commit: `ab4210be868548254fb8a683e9906a7de1b3c76e`
- scientific source base: `e154dbcd96b7a68267ddec20ff01201bc61f418e`
- runner SHA-256: `e29cf25488dff1ddd2113cec8ccc1b22d01737c96a36d6f967f516f0ebdd7ef2`
- preregistration SHA-256: `3f9531716fd0b8c371a5a95710cd6fcd6460a1f5e01b482dbc9d870e73b67034`
- full receipt SHA-256: `5663fc06a670a5e52d52d7bdb4dbb9e84ced9bc95b8c5a0d7deec42eff6c1535`
- UNICOM checkpoint SHA-256: `3916ab5aed3b522fc90345be8b4457fe5dad60801ad2af5a6871c0c096e8d7ea`

An independent replay recomputed the row-weighted pooled delta and the frozen
boolean decision from the canonical receipt and matched exactly. The original
DGX process exited zero, no duplicate ran, the GPU returned idle, and no paid
external compute was used.
