# SOP preprocessing worker: CPU stop, 28 September 2026

Close the one-worker, one-thread, standard-library spawn/IPC configuration.
It preserves pixel values and public tensor strides, but is slower than the
unchanged parent with 20 threads at both batch sizes. No GPU pilot, serving API
integration, worker-count or serialization rescue follows.

The [frozen gate](sop_preprocessing_worker_gate_2026-09-28.md) required worker
p95 at most 0.8 times control and median savings of at least 0.81 ms at batch 1
and 14.21 ms at batch 32. These thresholds stayed unchanged.

| SOP official TRAIN input screen | Parent median / p95, ms | Worker median / p95, ms | Median savings, ms | Worker/control p95 | Gate |
|---|---:|---:|---:|---:|---|
| Batch 1 | 1.759 / 2.375 | 2.702 / 5.261 | -0.943 | 2.215 | STOP |
| Batch 32 | 45.116 / 77.018 | 89.332 / 92.064 | -44.216 | 1.195 | STOP |

These are verified stage measurements: 20 calls per arm and size, interleaved
ABBA/BAAB, on 32 fixed authenticated TRAIN images. Decode precedes timing;
RGB conversion, preprocessing and full worker input/output serialization and
round trip are included. No encoder, packing, search or image-to-top-k timing
is included. The small screen provides no p99 or production throughput claim.
It identifies this combined worker path as unhelpful; it does not isolate the
individual contributions of IPC, serialization and computation.

Every measured output matched baseline float32 pixels and strides exactly.
Real spawned-process fixtures also passed RGB, grayscale, palette and RGBA
inputs at batch 1/32, rejecting forced contiguous conversion. Parent thread
count remained 20; the child used one. Source and image hashes passed.
An independent local standard-library replay recomputed all reported
quantiles, means and frozen decisions from the 80 raw durations.

DGX user unit `sfora-preprocessing-worker-cpu-v2`, invocation
`ad5bb69fe30a47de91f1d4f6f5eccde2`, exited 0. Whole process cost was 9.35 s,
maximum RSS 1,177,056 KiB; systemd reported 1.3G aggregate peak and no swap.
Cold worker initialization took 2.141 s, outside warmed stage timing. CUDA
was hidden, with 180 s / 8 GiB limits. No training or GPU job ran; the DGX was
idle after completion.

The original v1 invocation exited 1 before paired timing because an invented
ordinary-contiguity guard rejected the valid public channels-last control
(batch-1 stride `(3,1,768,3)`). Preserve its log and cost. The correction was
frozen in `631fb37e` before v2 timing and enforces baseline strides without
changing the public transform. The v1 failure produced no method result.

Raw [receipt](evidence/compact_metric/sop-siglip2-substrate-v1/sop-preprocessing-worker-v1/worker-cpu-v2.json),
[cost](evidence/compact_metric/sop-siglip2-substrate-v1/sop-preprocessing-worker-v1/worker-cpu-v2-time.txt)
and [replay](evidence/compact_metric/sop-siglip2-substrate-v1/sop-preprocessing-worker-v1/verification.json)
are retained. Receipt SHA256:
`6be3b3cb21016ab5f602543c40272760ccba10e931c24cb6d89260c120889cbe`.

No quality was measured. Previous SOP/In-Shop official results and published
reference qualification remain unchanged. The full joint production goal is
active and unmet. Next selection needs a distinct supported mechanism; this
screen does not authorize another worker variant. Any later serving candidate
must pass exact output parity before matched full-pipeline latency testing;
a product p99 claim still requires 10,000 interleaved paired calls and an
uncertainty interval. No operator decision or credential is needed to close
this negative arm.
