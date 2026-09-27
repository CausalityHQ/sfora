# SOP native vision CUDA graph feasibility, 27 September 2026

Test PyTorch's native CUDA graph replay around the existing seed-179024
true-freeze SigLIP2 Large/256 FP16 vision forward. The model, image processor,
input pixels, weights, head and native packed scorer remain unchanged. This is
an isolated vision-stage screen, not an end-to-end speed or quality claim.

Use the authenticated first 32 SOP official **TRAIN** images from the pinned
archive, with the same processor and checkpoint as the public stage profile.
Capture one static graph for batch 1 and one for batch 32. For each input,
copy the already-processed GPU pixels into the static graph input, replay,
and require the pooled FP16 tensor to be bitwise equal to an eager forward.
Measure 20 alternating ABBA/BAAB blocks of 10 eager and 10 graph calls per
position after warmup, including the static-input copy in graph timings.
Synchronize CUDA after each call. Both arms see the same preprocessed pixels;
the processor and JPEG decoder are outside this feasibility timing.

Advance only if every output is bitwise equal, batch-1 graph p50 is at most
**90%** of eager p50, batch-32 graph p95 is at most **105%** of eager p95, and
graph peak allocated CUDA memory is below **3 GB**. A pass permits a separate
exact packed top-10 and varied-image public p50/p95/p99 screen with a
single-threaded serving wrapper and explicit capture lifecycle. Failure stops
before production code. Neither case changes training throughput or retrieval
quality.

## Terminal isolated-stage result

The sole DGX Spark GB10 service `sfora-sop-cuda-graph-vision-v1.service`
completed successfully (invocation `d1d7559c80c642219cdb94f0c579d53a`).
Its [receipt](evidence/compact_metric/sop-siglip2-substrate-v1/sop-cuda-graph-vision-v1/receipt.json)
has SHA-256 `408699dda7c7a2aeba400ddd0895c6df2c7bc7621394650b67af8bde8eb7efc8`;
the [journal](evidence/compact_metric/sop-siglip2-substrate-v1/sop-cuda-graph-vision-v1/service.journal.log)
has SHA-256 `e759d51b2bb5b7f18f4537505803ab56ad2d8e279b2167aa2b8bd3dac093526e`.
The source script SHA-256 is
`3e5486e3a43c796e7fdcc34446e6b8d4c2073be629aa346247c197b17d7d8abd`.
All 32 batch-1 and all 32 batch-32 pooled outputs were bitwise equal to eager.
Each arm had 400 synchronized calls per batch size.

| Isolated FP16 vision, SOP TRAIN pixels | Eager p50/p95 | Graph p50/p95 | Frozen gate |
| --- | ---: | ---: | --- |
| Batch 1 | 8.043/8.564 ms | **7.161/7.707 ms** | p50 ratio **0.8903**, pass ≤0.90 |
| Batch 32 | 214.839/217.444 ms | 214.862/216.856 ms | p95 ratio **0.9973**, pass ≤1.05 |

Peak allocated CUDA was **995,443,712 bytes**, below 3 GB. The three frozen
gates pass. This authorizes the opt-in single-batch public-path implementation
and varied-image top-10/tail screen. Batch-32 graph capture adds no material
median speed here; retain eager batch-32 in the candidate. The measured
batch-1 vision saving is **0.882 ms**, so a full-call speed gain remains
uncertain until JPEG decode, preprocessing, packing and native search are
included. No training or quality metric was measured in this probe.

## Public-path candidate gate

The opt-in `cuda_graph_batch1=True` candidate captures only the FP16 vision
forward at encoder construction and uses it only for one-image calls. The
existing index lock and an encoder lock protect the static graph buffer through
packing. Batch sizes 2–32 use the existing eager path. Keep the seed-179024
checkpoint, 59,551-image SOP TRAIN gallery, native exact scorer, FP16 arithmetic
and 20 intra-op threads fixed.

Before any production promotion, compare opt-in and default indexes in one
DGX process on the authenticated first 32 TRAIN images. Require bitwise equal
packed codes, inverse norms, top-10 ordinals and scores for every single-image
call and the full 32-image batch. Interleave 20 ABBA/BAAB blocks of 10
whole-call timings per position, including JPEG read/decode, preprocessing,
host-to-GPU transfer, vision, packing and native search: 400 calls per arm at
batch 1. Require graph batch-1 p50 ≤0.95× eager and p95 no higher; batch-32
path must be bitwise equal and p95 no more than 1.05× eager over at least 100
calls per arm. Peak allocated CUDA must stay below 3 GB. A failure removes the
opt-in path. A pass only authorizes the varied-image 10,000-call batch-1 p99
gate and concurrency review; it is not a quality or SOTA claim.

## Public-path result and final promotion gate

The frozen public-path screen completed on the DGX Spark GB10, invocation
`7bb2f32c7dea40f6b52b9bdfad4f725f`, with receipt SHA-256
`c90a477dcf1f357bb577bbeb916502bfac5a7867a5f8f527865ec4a06a45ffc2`
at `docs/evidence/compact_metric/sop-siglip2-substrate-v1/cuda-graph-public-v1/receipt.json`.
On the first 32 SOP TRAIN images and the same 59,551-image TRAIN gallery,
batch-1 full-call p50 was 16.373 ms eager versus 15.072 ms graph; p95 was
19.072 versus 17.739 ms (400 calls/arm). Batch-32 p95 was 290.932 versus
286.654 ms (100 calls/arm). Peak allocated CUDA was 1,582,993,408 bytes.
All frozen gates and bitwise packed/top-10 checks passed. These are
exploratory TRAIN serving measurements, not official TEST quality or a SOTA
claim.

Before promoting the opt-in path, run 10,000 distinct SOP TRAIN paths through
both indexes, including JPEG read/decode in each timed public call. Alternate
arm order per image, check exact top-10 ordinals and scores for every image,
record every synchronized wall time, and require graph p50 ≤0.95× eager and
graph p99 ≤ eager p99. Retain the same source/checkpoint/gallery/precision,
20 threads and <3 GB peak CUDA ceiling. Also run concurrent calls to one graph
index from four workers on 100 of those images, checking exact top-10 against
the serial outputs. Its lifecycle lock deliberately serializes calls; the
concurrency check verifies the index's call serialisation, not overlapping
graph replay or a throughput claim. Failure at
either gate keeps the default eager path and withdraws this opt-in candidate.

The frozen final gate passed on the same DGX Spark GB10, invocation
`a043c35d365a4a4cb169a7e6bca614c1`. The [raw receipt](evidence/compact_metric/sop-siglip2-substrate-v1/cuda-graph-public-v1/p99-receipt.json)
has SHA-256 `7237d9eaeeaa0acf7f1aa1bce63ce7b6102c9d2f720f7a263027ff5d9ce946df`.
The [terminal journal](evidence/compact_metric/sop-siglip2-substrate-v1/cuda-graph-public-v1/p99-service-journal.log)
has SHA-256 `bb8b8d66fe0c7c784e1b35932a91eddb750be64a4eaba6bfd0ac83332f4adbf2`.
For 10,000 distinct SOP TRAIN paths (9,746 unique image byte hashes), eager
versus graph full-call p50 was
**16.368 vs 15.405 ms** and p99 **20.113 vs 19.085 ms**. Every paired top-10
ordinal and score row matched exactly. Four concurrent workers returned the
same top-10 for 100 selected images as the serial calls. Peak allocated CUDA
was **1,504,481,280 bytes**. Both p50 and p99 speed gates and the concurrency
and resource gates passed for that source. The default stays eager and batch
2–32 stays eager. This is one GB10/runtime/checkpoint/gallery configuration,
not a general GPU speed guarantee or a new retrieval-quality measurement. The
CUDA figure is post-construction peak *PyTorch allocated* memory of the
two-index benchmark process, not total startup or reserved GPU memory.

## Review correction and v2 qualification

Independent Opus/Astra review found that capture inherited an ambient BF16
autocast context even when `precision="fp16_native"`. A new GPU regression
reproduced the bug (BF16 captured pooler) before the fix and passed after
capture explicitly disabled CUDA autocast. The capture now selects its own
CUDA device and rejects input shapes that could broadcast into the static
buffer. The v1 receipts remain valid measurements of default-context capture,
but they do not qualify that old source for arbitrary application contexts.
Also, the first 10,000 TRAIN paths contained only 9,746 unique image byte
hashes. The corrected v2 gate skips duplicate bytes in archive order until it
has 10,000 unique images. It carries forward the same bitwise packed/top-10,
p50, p95, p99, batch-32, concurrency and <3 GB post-construction PyTorch
allocated gates; a source-bound v2 public screen must pass before v2 p99.
Graph index construction is supported during quiescent startup, before
concurrent CUDA requests; captured vision weights must remain on their device
for the index lifetime. No hot replacement is qualified. The v2 result will
decide whether to expose the opt-in path as qualified on this configuration.
