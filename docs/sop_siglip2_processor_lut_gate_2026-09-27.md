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
