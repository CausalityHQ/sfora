# Frozen CPU-only direct batch32 preprocessing screen

The trained FP16 batch32 source receipt records host preprocessing48.686ms,
host-to-device0.524ms, and whole267.254ms p50. These are fixed-query historical
measurements, not current public or independent latency qualification. Even
eliminating that transfer stage would save only0.196% of whole latency; it
does not justify a transfer/cast intervention against a5% full-call floor.
The receipt is
`sop-true-freeze-serving-179024/sfora-sop-true-freeze-serving-179024-freeze-fp16-v1.json`.

One selected serving intervention: for batch32 concatenate the existing
qualified `_direct_preprocess` results in input order instead of calling
the Hugging Face batch processor. Reuse the same antialiased Torchvision
resize and float32 normalization; no new arithmetic, pooler, packing, scorer,
weights, loss, dataset, process-wide thread change or public API flag.
The public code currently selects direct preprocessing for batch1 only.

Before any production or GPU change, compare exact pixels on1024 unique-byte
SOP official TRAIN images, first qualifying archive rows in order, grouped
into32 consecutive batches of32. Pinned archive/preprocessor/runtime and
current direct helper hashes are recorded. CPU-only DGX Spark,20 Torch threads,
no CUDA/model load/forward, shared GPU lock prevents heavy-job overlap.
Each batch performs equality first; any mismatch stops before timing.
For each valid batch alternate ABBA/BAAB, two timed calls per arm per batch
(64 per arm), including RGB conversion/resize/normalization/concatenation and
excluding image read/decode equally. Record every timing and image digest.

Require zero mismatches, direct p50≤0.70×control, p95≤0.80×control,
peak combined RSS<2GB, entire process≤120seconds with an external130second
timeout. The large processor floor targets meaningful full-call savings,
but cannot prove them. No budget extension, concurrency/normalization variant
or thread-count sweep after failure. Synthetic32-image exactness self-check
must pass first. A positive screen licenses only separately frozen public
packed/top10 parity and a≤60second matched full-call batch1/32 pilot, with
≥5% batch32 p50/p95 gain and batch1 nonregression; p99 remains unqualified.
Any failure closes this fixed batching candidate before GPU work. Production
remains unchanged pending all gates. No new quality or SOTA claim follows.
