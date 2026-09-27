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

## Paired public gate

The fixed follow-up runs the same seed-179024 SOP checkpoint, full 59,551-row
TRAIN gallery, native scorer and 640 distinct TRAIN query images at 20 CPU
threads on DGX Spark GB10. The source-bound script
[`certify_sop_siglip2_direct_public.py`](../scripts/certify_sop_siglip2_direct_public.py)
has SHA-256 `19a3f4423f883df607729bb0f951362eec939af2197aa3a7c9e77cd3446a9817`.
It requires exact packed codes, inverse norms, top-10 ordinals and scores for
all 640 batch-1 calls and 20 batch-32 calls. Twenty alternating ABBA/BAAB
blocks make 10,000 full decode-to-top-10 calls per arm at batch 1. Promotion
requires direct/control p50 ≤0.95, p99 point ratio ≤1.00, and paired-block
bootstrap p99 ratio upper 95% ≤1.05. A 200-call/arm batch-32 diagnostic must
keep p95 ≤1.10× control. The receipt also records throughput, peak CUDA and
parent RSS; the serving source is runtime and preprocessor
config guarded, and batch 32 keeps the existing processor path.

For In-Shop transfer parity, use the seed-179026 true-freeze TRAIN checkpoint
and fixed 6,354-query/6,245-gallery product-disjoint held roles. Take 640
evenly spaced query-role ordinals; compare baseline and direct packed codes,
inverse norms, and native top-10 ordinal/score bytes against the same built
gallery. Any mismatch rejects production promotion. The frozen
[`certify_inshop_siglip2_direct_parity.py`](../scripts/certify_inshop_siglip2_direct_parity.py)
source SHA-256 is `0af157967734e2aca4b65d4667e4bae32a853063ecd74d40e58aa6f937686ba5`.

## Terminal SOP public result

The sole DGX Spark GB10 unit `sfora-sop-direct-public-v1` (invocation
`0beb4fb1347d4914969bbf4eec91e814`) exited 0. Its
[raw receipt](evidence/compact_metric/sop-siglip2-substrate-v1/sop-direct-public-v1/receipt.json)
has SHA-256 `379ef1d8be37001b485679d7ccd1194339cae83b93af0ae401b27ea68f64f495`;
the [journal](evidence/compact_metric/sop-siglip2-substrate-v1/sop-direct-public-v1/journal.log)
has SHA-256 `9d31c5be05c4d269da5b49f5afa9d52ed171503b7798dd4e02c8bed064467cf5`.
The receipt binds the staged serving source SHA-256
`639e7bcbaef4ee2901220272a851a1c2c9d70c95f727568b99f5870a171ff1c4`.
An independent local replay verified source digests, all 20 × 500 raw
call samples per arm, quantiles and paired bootstrap output.

| Full SOP TRAIN image-to-top-10, seed 179024, 59,551-row gallery | Baseline processor | Direct processor |
| --- | ---: | ---: |
| Batch-1 p50 / p95 / p99, ms, 10,000 calls/arm | 16.394 / 19.224 / 20.916 | **13.404 / 16.009 / 17.879** |
| Batch-1 mean, ms / throughput, images/s | 16.766 / 59.64 | **13.804 / 72.44** |
| Batch-32 p50 / p95 / p99, ms, 200 calls/arm | 285.367 / 342.302 / 474.174 | 289.999 / **340.604** / 475.542 |

All 640 batch-1 and 20 batch-32 packed codes, inverse norms and native
top-10 ordinal/score arrays were exact. The paired batch-1 p99 ratio was
**0.8525**, with block-bootstrap 95% interval **[0.8351, 0.9503]** and
paired-superblock sign p **0.00098**. The frozen public gate passed; the
batch-32 p95 ratio was **0.9950**, within its nonregression floor. Peak
PyTorch CUDA allocation was **882,524,672 bytes** and peak parent host RSS
**3,875,676,160 bytes**, including model and gallery load. This is a
same-checkpoint local speed improvement, not a matched external SOTA claim.
In-Shop parity and clean-package checks remain release gates.

## Terminal In-Shop transfer parity

The sole DGX Spark GB10 unit `sfora-inshop-direct-parity-v1` (invocation
`fbb12aaa0ec34eea9ffc15aed5d2f9aa`) exited 0. Its
[raw receipt](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-direct-public-parity-v1/receipt.json)
has SHA-256 `09a426dd90911afe271e61eb6b026dd5fbbee98ed02ef48f110beffb54f9055d`;
the [journal](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-direct-public-parity-v1/journal.log)
has SHA-256 `1bc1bcd05bb57c5d60d77a0dfc792dd08c3dd0cbbe48338d2cdfa575e25456b5`.
All **640/640** selected official In-Shop TRAIN held query images produced
identical packed codes, inverse norms, native top-10 ordinals and score bytes
between baseline and direct paths against the same **6,245-row** gallery.
This is exact output parity, not a new In-Shop quality or latency measurement.
The gallery uses **811,850 wire bytes**; peak PyTorch CUDA allocation was
**1,266,973,184 bytes**, and peak parent host RSS was **4,060,184,576 bytes**
including model and gallery construction.

The measured serving source SHA-256 was
`639e7bcbaef4ee2901220272a851a1c2c9d70c95f727568b99f5870a171ff1c4`.
The production edit adds only a mypy suppression comment on that measured
return statement (production SHA-256
`ef454200ca17b80917fe7aa52cf232703779d21af7e8ffe0d25f8e811724bc62`);
an AST comparison confirms identical executable code. The focused serving
tests pass **18**, with **one** CUDA graph skip on the local CPU host.
The local wheel `sfora-0.3.0rc4-py3-none-any.whl` built successfully with
SHA-256 `98be87f65e030902d48f0544e45baab67243ed40674dd7cf50228c6d2a373ea0`;
the serving module inside has the production source SHA-256 above. The full
local pytest suite passed **5,549 tests**, with **13 skips**. Ruff formatting
and lint passed for the changed source, tests and gate scripts. The local mypy
run still reports 12 errors in these two touched files, all on lines present
before this edit; the two errors introduced by the initial direct path were
removed with type-only comments. Thus this is a measured production serving
increment for the pinned runtime, while repo-wide type-check cleanliness
remains a separate release gap.
