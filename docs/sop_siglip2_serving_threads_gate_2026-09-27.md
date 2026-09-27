# SOP serving intra-op thread gate, 27 September 2026

The source-bound public stage profile found 6.412 ms p50 in CPU image
preprocessing for one fixed SOP TRAIN query. A one-process exploratory ABBA
screen of PyTorch intra-op threads 20 versus 1 on the existing seed-179024
true-freeze index measured batch-1 image-to-top-10 p50 **15.809 vs 10.005
ms**, p95 **18.305 vs 10.738 ms**. Its first query's packed codes, inverse
norm, top-10 ordinals and scores were bitwise equal. This is an operational
runtime candidate, not a new learning method or quality result. Global thread
count affects the host process, so Sfora must not silently change it on index
load or per call.

Freeze the next **TRAIN-only** screen before its batch-32 read. Use the same
checkpoint, model snapshot, native library, 59,551-row gallery and 32 pinned
SOP TRAIN images as the previous public latency gate. In one DGX Spark GB10
service, load one index and interleave 20 ABBA/BAAB blocks of 10 calls per arm
per position, yielding 400 measured calls per arm at batch 1 and batch 32.
Set `torch.set_num_threads(20)` or `torch.set_num_threads(1)` **outside** each
timed block, warm five calls after each switch, and restore the original
setting at exit. Each call includes file read/decode, the unmodified
`Siglip2CompactIndex.search_images`, and synchronized native top-10. Require
bitwise equal 32-image packed codes/norms and top-10 ordinals/scores across
thread settings, stable repeated outputs, complete source/checkpoint/image
hashes, 400 calls per arm per batch, candidate p95 at most **0.95×** baseline
for both batch sizes, and nonregressing batch-32 images/s. Any failure closes
the threads=1 production lane without a p99 run.

A passing screen authorizes a distinct 10,000-call paired p99 certification
at batch 1 and batch 32 with varied TRAIN images and block-bootstrap
uncertainty, plus a documented process configuration for single-purpose
serving. Do not promote `torch.set_num_threads(1)` as a library side effect.
Official TEST quality was already observed and is not used for this decision;
this runtime setting can improve serving speed only, not retrieval quality or
training cost. No SOTA claim follows from a one-query screen.

## Terminal screen result

The sole DGX Spark GB10 service exited 0. Its [raw receipt](evidence/compact_metric/sop-siglip2-substrate-v1/sop-serving-threads-179024/receipt.json)
has SHA-256 `1d42c2928db1bb2e106f8cda61b79de39459d421713e8307997616d8419fd8b5`;
the [service journal](evidence/compact_metric/sop-siglip2-substrate-v1/sop-serving-threads-179024/service-journal.txt)
has SHA-256 `534598a107db1c6a6f0726f3031d09a5502412aea9898422d4f32c3a05dc8614`.
All 32 packed codes/inverse norms and native top-10 ordinals/scores matched
bitwise across settings. Every repeated top-10 result was stable, with 400
measured calls per arm at each batch size.

| SOP TRAIN public image-to-top-10 | 20 threads | 1 thread | Change |
| --- | ---: | ---: | ---: |
| Batch 1 p50 | 16.203 ms | 10.353 ms | −36.1% |
| Batch 1 p95 | 18.907 ms | 11.219 ms | −40.7% |
| Batch 32 p50 | 284.217 ms | 275.335 ms | −3.1% |
| Batch 32 p95 | 309.605 ms | 279.029 ms | −9.9% |
| Batch 32 effective throughput | 113.04 images/s | 116.37 images/s | +2.9% |

Both p95 ratios clear the frozen 0.95 gate and batch-32 throughput did not
regress. The screen advances to varied-image p99 certification. It has one
checkpoint, one fixed 32-image set and no loaded-service concurrency, so it
does not establish a production p99 guarantee or a quality gain.

## Frozen varied-image p99 certification

Keep the screen's seed-179024 checkpoint, FP16 public index, TRAIN gallery,
native library and serving source. Draw 640 distinct SOP TRAIN rows at evenly
spaced ordinals from the authenticated 59,551-row archive, divided into 20
ordered blocks of 32. Each block uses its own 32 images: batch 1 cycles through
them in each 250-call segment, while batch 32 submits all 32 per call. Thus
both arms see identical images in every paired block. Include disk read/decode,
unmodified `search_images`, CUDA synchronization and exact native top-10 in
each timed call. Set thread count outside timing; warm five calls after every
switch. Alternate ABBA/BAAB arm order, 250 measured calls per segment, 20
blocks, yielding **10,000 calls per arm per batch size**. Run batch 1 first;
only an exactness or statistical failure may stop before batch 32.

Require exact packed codes/norms and top-10 ordinals/scores for all 640
images in batch 32 and for all 640 individually, plus stable per-image
repeated outputs. Preserve every raw latency, source/checkpoint/image hash,
peak CUDA memory, host RSS and hardware identity. Analyze each batch with the
existing paired-superblock p99 bootstrap helper (5,000 resamples, seed
179019): its 95% upper p99 ratio must be <1, p50 and mean ratios ≤1, and
one-sided sign p<0.05. Also require candidate p95 ≤ baseline p95 and batch-32
images/s ≥ baseline. A pass supports an opt-in **single-purpose serving
process** recommendation for `torch.set_num_threads(1)`; it does not authorize
a silent Sfora library side effect, a concurrency or scale claim, or a new
retrieval-quality claim. A failure closes this runtime lane and redirects work
to the next measured bottleneck.
