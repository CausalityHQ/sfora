# Exact normalization lookup: combined CPU gate STOP

The 256-entry float32 lookup matches the original uint8 normalization exactly.
It improves the measured batch-1 stage but regresses batch32. The
[frozen gate](sop_normalization_lookup_gate_2026-09-28.md) requires both sizes
to pass, so close this configuration before GPU testing or API integration.
No lookup implementation, layout, thread-count or threshold tuning follows.

| Official SOP TRAIN CPU preprocessing | Parent20 median / p95, ms | Lookup20 median / p95, ms | Median savings, ms | p95 ratio | Gate |
|---|---:|---:|---:|---:|---|
| Batch1 | 2.142 / 3.454 | 0.765 / 0.964 | +1.378 | 0.279 | PASS |
| Batch32 | 49.508 / 73.192 | 56.833 / 90.878 | -7.326 | 1.242 | STOP |

These verified measurements use32 authenticated, evenly spaced TRAIN images,
20 calls per arm and size, interleaved ABBA/BAAB. Both arms retain20 torch
threads and identical CPU resizing, grouping and RGB conversion. Timing
includes preprocessing and result allocation; file read/decode, encoder,
packing and search are excluded. The lookup uses existing NumPy `take` into
an output tensor allocated with the original strides. It replaces only the
post-resize uint8 conversion/normalization, with no spawned process or IPC.
No end-to-end speed, p99, throughput or quality claim follows from this screen.

Exhaustive fixtures checked all256 byte values against both original direct
and processor normalization, contiguous/channels-last layouts at B1/B32,
and RGB/L/P/RGBA image preprocessing. Unsupported float input was rejected.
Every measured pixel tensor matched baseline values and strides. An initial
setup call to a cached processor method passed lists rather than its required
tuples; it was corrected before timing. Ruff and fixtures then passed, and
`d064a723` froze the implementation and thresholds before the sole timing job.

Independent local standard-library replay of all80 raw durations reproduced
medians, interpolated p95, means and the combined STOP. Source, helper,
baseline-script and serving hashes matched the frozen local files.

DGX CPU user unit `sfora-normalization-lookup-cpu-v1`, invocation
`a01eea1ae97f49cda474bfe1738f1a54`, exited0 in5.87s, maximum RSS1,166,360KiB
according to `/usr/bin/time`; CUDA was hidden and limits were120s/8GiB.
The transient service receipt printed an implausibly small768KiB memory
peak; do not use it as a process-memory measurement. Original time output
and service journal are retained. No GPU/training job ran; DGX idle afterward.

[Receipt](evidence/compact_metric/sop-siglip2-substrate-v1/sop-normalization-lookup-v1/receipt.json),
[cost](evidence/compact_metric/sop-siglip2-substrate-v1/sop-normalization-lookup-v1/time.txt)
and [independent replay](evidence/compact_metric/sop-siglip2-substrate-v1/sop-normalization-lookup-v1/verification.json).
Receipt SHA256:
`8e9ac7a92204bc0cb5591df5002e85747018c1d2952d7ba5cdfff3ac843fc363`.

This result shows that exact CPU normalization substitution is possible and
that this particular implementation fails the required combined screen. It
does not prove normalization is universally unhelpful or identify which
internal allocation/lookup cost causes the B32 regression. No further
profiling or post-hoc batch-specific production selection was performed.

Official SOP/In-Shop quality and published frontier qualification remain
unchanged. The whole production goal is active and unmet. Next work must
select a distinct supported mechanism, keeping exact packed/native output
parity before matched full-pipeline timing; product p99 needs10,000 varied
paired calls with uncertainty. No new operator decision is required.
