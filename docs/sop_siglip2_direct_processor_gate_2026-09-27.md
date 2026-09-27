# SOP TRAIN direct Torchvision processor gate

The source-bound public query profile measured 6.412 ms median CPU processor
time in a 16.952 ms whole batch-1 call. The previous PIL-resize lookup failed
pixel exactness on 9,970/10,000 SOP TRAIN images. This candidate instead uses
the pinned processor's actual Torchvision v2 tensor path: PIL-to-tensor,
antialiased bilinear 256×256 resize, then fused float32 normalization by
127.5 per channel. It changes no model, training, packed wire or scorer.

Before production editing, compare its `pixel_values` bitwise to the
processor-pinned public path for the same 10,000 unique-byte SOP official
TRAIN images. At 20 PyTorch intra-op threads, time 1,000 distinct images with
two calls per arm per image, alternating ABBA/BAAB. Include the public RGB
conversion and exclude JPEG read/decode in both arms. Require zero pixel
mismatches, candidate p50 ≤0.85× and p95 ≤0.90× processor control, and
combined process peak RSS <2 GB. Failure stops this lane before public code.

A pass permits the minimal source change plus focused pixel regression tests.
The subsequent public gate must preserve packed bytes, inverse norms and
native top-10 score/ordinal bytes for the fixed SOP TRAIN inventory and
In-Shop TRAIN roles, and improve paired batch-1 image-to-top-10 p50 ≥5%
without p99 regression at 20 threads on DGX Spark GB10. Measure batch-32
nonregression, gallery build and resource use as well. This gate is about
serving cost only; retrieval quality and training cost are unchanged by an
exact pixel transformation.

Frozen source SHA-256: `ae6db5e2bbc49353315801cd4cddebabc8f7f7bc6a88df0e001c686206fd0d95`.

## Terminal processor screen

The sole DGX Spark GB10 unit exited 0, invocation
`bdc92634171743e780bbfe900b676bfe`. Its [raw receipt](evidence/compact_metric/sop-siglip2-substrate-v1/sop-processor-direct-v1/receipt.json)
has SHA-256 `8186e99d85ec1aeff526a046fe5a8fa3e38af516979215c4200fd83f9bfd35f7`;
the [journal](evidence/compact_metric/sop-siglip2-substrate-v1/sop-processor-direct-v1/journal.log)
has SHA-256 `b8d705a6317c788bc6d9ea52739f3f0f0381976e371d634ca4db33630bb022c7`.
All **10,000/10,000** unique-byte SOP TRAIN images produced bitwise equal
`pixel_values`. The 2,000-call/arm paired screen measured control versus
direct p50 **4.342712 vs 2.686296 ms** (−38.1%) and p95 **5.816719 vs
4.084796 ms** (−29.8%). Combined peak RSS was **886,628,352 bytes**.
All frozen screen gates pass. This is processor-only; public
image-to-top-k speed, native top-10 parity and In-Shop transfer remain to be
verified before production promotion.
