# SOP full TRAIN gallery public serving scale gate

Freeze the current production `Siglip2CompactEncoder.from_checkpoint` and
`Siglip2CompactIndex.from_image_paths` path before measuring a full external
gallery. Use the seed-179024 true-freeze checkpoint SHA-256
`2c838561b6c23242d74eb29329fd026cc8fba9bf965dcc4529348028dfe6d172`
and training receipt SHA-256
`07b4716b42d1291b9c195774ebd48d9df89a3b578ee54efdb94662c5e125d1c5`.
Pin the same 130-byte native scorer SHA-256
`39602d0e4e8b0d5ec441be460ad7f18e288241bef19fb6e6c5df14f4033ac73c`.

Build a new gallery from all **59,551 Stanford Online Products TRAIN** image
files in archive order. Use the first 1,000 distinct TRAIN image byte hashes as
public image-to-top-10 inputs; they may overlap the gallery because this gate
tests serving, not retrieval quality. Compare 32 public top-10 rows with an
independent stable packed-matrix oracle, requiring identical ordinals and
maximum score error ≤`1e-5`. On the idle DGX Spark GB10, require gallery build
≤`600 s`, 1,000 synchronized batch-1 calls p99 ≤`30 ms`, post-load peak
PyTorch allocated CUDA <`3 GB`, and peak parent RSS <`6 GB`. Record batch-32
throughput, p50/p95/p99, model load, build time, and gallery wire bytes.

This is a TRAIN-only public serving qualification. No SOP TEST quality or
state-of-the-art claim can be inferred from it. A failed threshold keeps this
full-gallery configuration unqualified and directs a code-level bottleneck fix.

The first invocation (`385f78d86fcd4756b584de551d1f802c`) exited before
model load: its script assumed the first 1,000 archive paths had distinct
image bytes. They do not. The [failed journal](evidence/compact_metric/sop-siglip2-substrate-v1/sop-public-scale-train-v2/failed-v1-journal.log)
is retained. Version 2 selects the first 1,000 distinct byte hashes while
computing the unchanged full-gallery digest; all thresholds and artifacts
above remain frozen. No quality or latency was observed in version 1.

## Version 2 result and source-level repair

The version 2 unit (`e3a6c79b805448c7bf338b4662b0baea`) wrote its
[receipt](evidence/compact_metric/sop-siglip2-substrate-v1/sop-public-scale-train-v2/receipt.json),
SHA-256 `2b6e59d1c6e6d009e496b2b0261b185a139f17c54414bd861b7b5ee5f0857053`,
then exited 1 on the frozen RSS floor. All 59,551 TRAIN images built in
**535.276 s**. The 7,741,630-byte gallery produced exact packed top-10
ordinals and scores for 32 TRAIN queries (maximum score error 0.0). Across
1,000 distinct TRAIN query byte hashes, synchronized image-to-top-10
p50/p95/p99 was **15.786/18.521/19.978 ms**, and batch-32 throughput was
**112.51 images/s**. Post-load peak allocated CUDA was **0.937 GB**.
Peak parent RSS **6.194 GB** exceeded the frozen **6.0 GB** limit by 0.194 GB;
the [terminal journal](evidence/compact_metric/sop-siglip2-substrate-v1/sop-public-scale-train-v2/journal.log)
records the nonzero exit. Thus full-gallery serving was not qualified.

The loader alone peaked at 3.857 GB RSS and the first 10,000 encodes stayed
near 4.006 GB in a separate [profile](evidence/compact_metric/sop-siglip2-substrate-v1/sop-public-scale-train-v3/profile-encode-journal.log).
The largest later 32-image batch contains **162,171,603 source pixels**.
Splitting its model inference into smaller batches changed 31 of 32 packed
rows, so that remedy was discarded. The production encoder now preprocesses
images individually only when their combined source pixels exceed 64 million,
then concatenates the exact processed pixels for the unchanged 32-image model
forward. On that worst batch, standalone per-image versus batched processor
pixels were bitwise equal. A [source-level paired check](evidence/compact_metric/sop-siglip2-substrate-v1/sop-public-scale-train-v3/large-batch-journal.log)
found packed codes and inverse norms bitwise equal; peak RSS through the bounded
path was **4.575 GB** versus **6.130 GB** after the original full-batch
preprocessing in the same process. Version 3 will rerun the unchanged full
TRAIN scale thresholds on this repaired public code before qualification.
