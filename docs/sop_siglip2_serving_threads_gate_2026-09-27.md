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
