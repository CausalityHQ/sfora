# Specific FP16 cast-elision diagnostic

After the default compiler packed-parity KILL, read-only inspection found
`triton_per_fused_add_native_layer_norm_view_5` performs three residual additions
in FP32, directly followed by layer normalization, without intermediate FP16
casts. The eager FP16 additions materialize FP16 results. Installed Torch
2.12.1 `_inductor/config.py:2722` documents this difference and native
`emulate_precision_casts`; `lowering.py` inserts the casts and Triton codegen
also disables floating-point fusion when that option is enabled. This is a
demonstrated numerical-path difference, not proof it alone caused all packed
changes. No generic option sweep or compiler retry is justified by speculation.

Frozen cheapest test: one synthetic GPU unit, no images/weights/quality reads,
60-second external cap,8G host memory. Seed179059001, four32×257×1024FP16 tensors,
ones/zero layer-norm affine parameters, epsilon1e-6. Compare eager versus default
compiler versus the single native cast-preservation option on the three-add/LN
function. Require finite outputs, nonzero default mismatch, and at least5×
reduction in BOTH RMS and maximum absolute error. Otherwise KILL this cast
repair before a full encoder run. No latency benchmark or public code change.

A positive result supports only the synthetic numerical mechanism. Actual
packed/top10 parity remains mandatory in a separately frozen <=120-second
model smoke before timing; failure there stops without an option search.

## Original terminal: KILL this cast repair

Original unit `sfora-compiler-cast-diagnostic-v1`, invocation
`2caf2e1657744e4285e92dee88f4fb90`, exited0/inactive/MainPID0. This is a
synthetic mechanism diagnostic, **no dataset or split** was evaluated.

| Eager-reference error over8,421,376 FP16 output entries | Default compiler | Preserve casts |
|---|---:|---:|
| Mismatched entries |3,687,581|346|
| Maximum absolute error |0.00390625|0.001953125|
| RMS error |0.000443330384|0.000004085804|
| Compilation plus one call,s |1.579581|0.230675|

RMS error decreased108.505×, supporting the cast-elision mechanism. Maximum
error decreased only2×, missing the frozen5× floor. **KILL_CAST_REPAIR**;
do not relax the floor, try more options, run the full encoder again, or add
a production compiler option. Remaining differences are not explained by this
diagnostic, and no particular reduction operator has been isolated as their
cause. This result does not prove all compiled packed outputs would differ.

Full image-to-top-k p50/p95/p99/QPS, R@1/mAP@R and training cost are all
unmeasured. Compilation-plus-call numbers are setup cost, not serving latency
comparisons; the second arm can benefit from process/compiler warmup. No
statistical uncertainty or actual-model exactness is established. The public
eager path remains qualified, the overall quality-and-speed target remains
unmet, and the DGX lock is free.

[Receipt](evidence/compact_metric/sop-siglip2-substrate-v1/compiler-cast-diagnostic-v1/receipt.json),
[independent metadata/decision verification](evidence/compact_metric/sop-siglip2-substrate-v1/compiler-cast-diagnostic-v1/verification.json),
[original generated kernel](evidence/compact_metric/sop-siglip2-substrate-v1/compiler-cast-diagnostic-v1/original-fused-kernel.py),
[journal](evidence/compact_metric/sop-siglip2-substrate-v1/compiler-cast-diagnostic-v1/journal.txt).
Receipt SHA256
`e2bc75ec5b19e12a3602aa9c71b6f83c0c1be89cd23ea21e2f91103a51022b65`.
Actual synthetic tensors were not retained or rerun; verification independently
replayed the recorded decision and source/count checks.
