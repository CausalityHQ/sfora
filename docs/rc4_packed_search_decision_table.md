# SFORA 0.3.0rc4 packed search decision

**Decision:** retain the batch-specific merge width in production. Batch 32
uses 512 entries per merge group; batch 1 retains the RC3 width of 2048. The
public Python API, native ABI, exact score arithmetic, stable ordinal tie rule,
and 130-byte served gallery item stay intact. The candidate code is commit
`8dd89c8b02c3b88c0a5b05ea6ae1e4e44b9d785e`; the RC4 package source is
`086eb6c29ddc759c336aa9863367a3d001b11d42`.

## Matched one-million-row GB10 result

Each row is a 50-call Python FFI replay after five warmups on the same frozen
gallery and query files. Pair 1 ran RC3 then RC4; pair 2 reversed that order.
Latency is p50 / p95 / p99 in milliseconds. Throughput is queries per second
from the 50-sample mean. Peak RSS is the process high-water mark and is shared
by both batch rows from each process.

| Pair | Batch | Library | p50 / p95 / p99 (ms) | Throughput (q/s) | Peak RSS (MB) |
| ---: | ---: | --- | ---: | ---: | ---: |
| 1 | 1 | RC3 | 1.051 / 1.236 / 1.468 | 929.2 | 781.3 |
| 1 | 1 | RC4 | 1.059 / 1.438 / 1.500 | 912.0 | 779.9 |
| 1 | 32 | RC3 | 4.526 / 4.944 / 5.157 | 6,983.1 | 781.3 |
| 1 | 32 | RC4 | 2.802 / 3.245 / 3.307 | 11,249.5 | 779.9 |
| 2 | 1 | RC3 | 1.067 / 1.333 / 1.401 | 916.3 | 781.3 |
| 2 | 1 | RC4 | 1.056 / 1.290 / 1.436 | 924.0 | 780.0 |
| 2 | 32 | RC3 | 4.562 / 5.070 / 5.227 | 6,924.6 | 781.3 |
| 2 | 32 | RC4 | 2.809 / 3.230 / 3.307 | 11,137.5 | 780.0 |

RC4 batch-32 p99 improved by **35.9% and 36.7%** relative to its paired RC3
run. Batch-1 p99 was **2.2% and 2.5% higher**, inside the preregistered 5%
guardrail. The frozen RC3 release receipt separately reports 1.550 / 5.274 ms
p99 for batches 1 / 32 and 1.552 GB peak RSS; it is not substituted for these
matched pair measurements. The earlier flat 512-width pilot was rejected at
the public FFI boundary because its single batch-1 p99 regressed 6.4%; its
raw output is retained.

| Check | RC4 evidence |
| --- | --- |
| Exactness | Score bits and ordered top-10 ordinals match the frozen fused path for batches 1 and 32 at 1,000,000 and 1,000,003 rows; paired FFI replays match it too. Signed extremes and stable ties are covered by Rust tests and the clean-wheel smoke. |
| Rust replay p50 / p95 / p99 | Batch 1: 1.038 / 1.272 / 1.365 ms; batch 32: 2.807 / 3.257 / 3.308 ms. |
| Tracked CUDA pool peak | Batch 1: 133.0 MB; batch 32: 165.0 MB. Maximum pool utilization came from separate memory-enabled Nsight traces. |
| Process RSS | 779.9–780.0 MB in matched RC4 FFI runs, below the 2 GiB gate. |
| Storage | 128 signed code bytes and one 2-byte inverse norm per served item: 130 bytes. |
| Package | Clean Git-export `0.3.0rc4` wheel installed into a new DGX virtual environment; both native batches returned finite scores, valid shapes, and the 0-before-128 tie. The native library is an explicit optional backend and is not bundled in the pure-Python wheel. |
| Pet quality | Untouched class-disjoint learned int8-128 receipt: mAP@R 0.862759, Recall@1 0.969595. |
| In-Shop quality | Untouched official query/gallery rank-finished receipt: mAP@R 0.800020, Recall@1 0.954283. |

The candidate library SHA-256 is
`39602d0e4e8b0d5ec441be460ad7f18e288241bef19fb6e6c5df14f4033ac73c`;
the pinned RC3 library SHA-256 is
`a9e583881201760088f3d99c180367e587ca68219c79dfe1760a3467f06b8dea`.
The clean wheel SHA-256 is
`38843a0d3b5c583f0cdfe25aeb971fa34dece10b356afb030c026fe479f5fc8e`.
The source distribution SHA-256 is
`c08e5f1626d4de64082938888f48c46ae9dd8bc3db7489a449e9fb019913f988`.
The authenticated fixture manifest SHA-256 is
`2d34832dfe22378b767f9cbedc0911e04ff2e06add2d1fdea66e6d0d55497799`.

Machine-readable values and raw samples are in
`docs/evidence/packed_topk_rc4_candidate_v1.json` and
`docs/evidence/packed_topk_rc4_candidate_raw_v1.tar.gz` (archive SHA-256
`d872fa567f57d32b9757c7a824e7a641a58857a6c580ea637ade35cfa30596cb`).
The archive includes the executed FFI replay script, Rust source, Nsight
reports and SQLite exports; the 260 MB fixture is reproducible from the frozen
generator and authenticated manifest described in
`docs/packed_topk_rc4_stage_result_2026-09-23.md`. The candidate receipt is
recomputed by `uv run python scripts/collect_packed_topk_rc4_candidate.py`.
The unchanged Pet/In-Shop hashes and metrics are in
`docs/evidence/packed_topk_rc4_quality_revalidation_v1.json`; clean-wheel
output is in `docs/evidence/packed_topk_rc4_clean_wheel_smoke_v1.json`.

## Claim limit

This is a GB10 systems result on one deterministic synthetic packed gallery,
not a descriptor-quality improvement, a cross-device latency guarantee, or a
scientific SOTA claim. Fifty post-warmup samples support the reported empirical
percentiles but do not establish a confidence interval. First-call JIT and
gallery admission are excluded from steady-state latency. The GPU memory value
is tracked CUDA pool use, not device-wide peak allocation. Pet and In-Shop
quality results are inherited frozen evidence, not rerun model training; their
original claim-eligibility limits remain in force.
