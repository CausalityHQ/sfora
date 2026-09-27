# SOP TRAIN exact image-processor gate, 27 September 2026

The current public batch-1 SigLIP2 image-to-top-10 path spends 6.412 ms
median in the Hugging Face image processor (source-bound stage profile). Test
one replacement: keep its RGB conversion and PIL bilinear 256×256 resize,
but map each resulting uint8 channel value through a 256-value float32 lookup
table obtained from the installed processor itself. This changes no weights,
training, gallery, scorer, descriptor or search semantics.

Use the pinned SigLIP2 Large/256 processor and 10,000 unique-byte SOP official
TRAIN images from the already qualified public p99 inventory. Compare exact
`pixel_values` with the installed processor on every image. Time 1,000 of
those distinct images through both processors, including RGB conversion and
resize but excluding JPEG read/decode, in alternating ABBA/BAAB blocks with
single-threaded CPU execution. Record p50/p95 and the complete per-image
parity count. The screen is exploratory and TRAIN-only.

Advance to a production change only if all 10,000 tensors are bitwise equal,
candidate p50 is ≤0.85× and p95 ≤0.90× control, and peak process RSS remains
below 2 GB. Otherwise stop this lane. A pass
authorizes one minimal serving change and then the public 10,000-image paired
top-10 p50/p95/p99 gate under the same DGX Spark hardware, checkpoint, 59,551
TRAIN gallery, 20 PyTorch threads, FP16, graph opt-in and exact native scorer.
The public gate must preserve packed bytes, inverse norms, top-10 ordinals and
scores for every image, improve batch-1 p50 by at least 5%, not regress p99,
and retain <3 GB post-construction PyTorch allocated peak. Do not infer any
training or retrieval-quality gain from processor timing alone.

## Terminal screen and decision

The first two service invocations stopped before opening the image archive:
the script's configuration guard incorrectly assumed `SizeDict` was a plain
dictionary and then called `.items()` on it. The guard was corrected to read
its `height` and `width` attributes; a separate one-line check verified all
five processor fields before the terminal invocation. No comparison data were
read in either failed service.

The terminal DGX Spark GB10 service exited 0 (invocation
`119cc37f7d0b4514a15a4ed9396c2119`). The [raw receipt](evidence/compact_metric/sop-siglip2-substrate-v1/processor-lut-v1/receipt.json)
has SHA-256 `b65a8adb52e3b620c36636e01cd66ea691a718ae436c863429e9f7c53ab06621`;
the [journal](evidence/compact_metric/sop-siglip2-substrate-v1/processor-lut-v1/service-journal.log)
has SHA-256 `56e84e2b060c19785ceefcc1da33bd50ae1fa4622700710fe4cca2d55acc7dab`.
The source SHA-256 is
`92419f3e774d0ac3ab3da9f9f6b838c87e868bc0361372f37ad4e15f52b3dbb7`.

| SOP official TRAIN screen | Installed processor | Lookup candidate | Frozen gate |
| --- | ---: | ---: | --- |
| Exact pixel tensors, 10,000 unique-byte images | reference | **30/10,000** | Fail: 9,970 mismatches |
| Processor-only p50, 2,000 calls/arm on 1,000 distinct images | **1.216254 ms** | **0.910399 ms** | Timing pass, unusable without parity |
| Processor-only p95 | **1.684032 ms** | **1.095036 ms** | Timing pass, unusable without parity |
| Process peak RSS | — | **887,029,760 bytes** combined process | Pass <2 GB, no isolated arm attribution |

The installed `backend="torchvision"` processor uses Torchvision tensor
resize with antialiasing. The candidate used PIL bilinear resize before its
lookup, which changes pixel values on nearly every real image. The LUT itself
matched the installed processor on the unchanged 256×256 synthetic ramp.
This identifies the resize path as the cause of this candidate's failed
equivalence. Stop the lookup lane. Do not promote it to the public API or run
the downstream top-10 timing gate. The existing processor, model and public
CUDA-graph path remain the production choice; no retrieval-quality or training
metric was remeasured here.
