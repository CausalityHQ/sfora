# Packed int8 million-row serving profile

The exact 130-byte Sfora int8 wire was profiled on one million 128-dimensional
rows on the NVIDIA GB10. The current public scorer materializes both packed
operands as float32 for every request. A matched resident control performs the
same float32 dot product and norm scaling after a one-time persistent expansion.
Both arms returned bit-identical top-10 indices for batch 1 and batch 32.

| Query batch | Current p50 / p99 | Resident p50 / p99 | Current / resident throughput | Materialization share |
|---:|---:|---:|---:|---:|
| 1 | 18.794 / 20.049 ms | 2.390 / 2.597 ms | 53.3 / 417.4 q/s | 87.24% |
| 32 | 24.095 / 25.635 ms | 8.080 / 8.665 ms | 1,323.7 / 3,902.7 q/s | 66.08% |

The packed gallery occupies exactly 130,000,000 bytes. The resident float32
control occupies 516,000,000 GPU bytes, so it demonstrates the latency ceiling
but is not a storage-equivalent solution. Process peak RSS was 1,745,264,640
bytes and peak allocated CUDA memory was 1.190 GB in the batch-32 arm.

Both measured materialization shares exceed the frozen 30% threshold for
custom-kernel work. This authorizes a fail-fast Rust/cuTile signed-int8 score
kernel whose first gate is exact score/top-k equivalence and whose performance
target is the resident control without its four-times gallery expansion. It
does not yet establish a custom-kernel speedup, a SOTA serving result, or a
library default.

Authorities:

- source commit: `0cbbd0ace901f0852893b4f59437b73288326c06`
- profiler SHA-256: `964c37c3bf689e7a9c7198d218f16930f1430d37bc13ea41964c94868c7e546d`
- full receipt SHA-256: `598149e753d4c7f0d5fc9091f325ad725c9116ef2779051efdecaaba4e42de26`
- Torch/CUDA: `2.12.1+cu130` / CUDA `13.0`
- device: NVIDIA GB10

The subsequent pinned `cutile-rs` `v0.1.1` canary succeeded using the system
CUDA 13.0 host headers and the installed user-space CUDA 13.4 `tileiras`.
Upstream commit: `c299d449cd0cc58c77705cf559caa02b3790e4bd`.
