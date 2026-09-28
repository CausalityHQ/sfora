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
