# Packed int8 cuTile score-plane result

The preregistered exact signed-int8 cuTile score kernel advances. This is a
score-plane result on a deterministic synthetic one-million-row gallery, not a
retrieval-quality or end-to-end serving claim.

## Authority

- Source commit: `87203558b79a4f49b63668431477fb1c1a75a1f8`
- Device: NVIDIA GB10
- Rust: `rustc 1.98.1 (48a229cea 2026-09-01)`
- CUDA Tile IR: 13.4.92
- Receipt: `docs/evidence/packed_int8_cutile_score_summary.json`
- Receipt SHA-256: `c7b1403de61a3fe457fa92eeab08f814fbfcc5cd60a1d4f1fe241c6eb4374fb4`
- Protocol: 1,000,000 rows, 128 signed-int8 dimensions, 5 warmups, 50
  steady-state samples per arm and batch. First compile/JIT launches are
  recorded separately and excluded from percentiles.

## Verified result

| Batch | Packed p50 | Packed p99 | Throughput | Resident-f32 p50 | Resident-f32 p99 | p99 speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.619 ms | 1.002 ms | 1,616.7 query/s | 2.150 ms | 2.201 ms | 2.20x |
| 32 | 1.195 ms | 1.477 ms | 26,779.7 query/s | 7.460 ms | 8.293 ms | 5.62x |

Every packed score matched the scalar Rust authority bit-for-bit on the
registered exactness fixtures, and deterministic top-10 ordinals matched for
both batches. The packed gallery uses exactly 130,000,000 persistent bytes
(130 bytes/item), versus 512,000,000 bytes for the matched resident-f32
control. Dense score-plane temporary storage is 4,000,000 bytes at batch 1 and
128,000,000 bytes at batch 32.

The sole corrected execution exited zero in 3.66 seconds, used 922,232 KiB peak
host RSS, and reported no swaps. Nearest-rank p50/p99, raw-sample cardinality,
byte arithmetic, throughput, exactness flags, and the advance decision were
independently recomputed from the canonical receipt.

## Decision and limits

Both batches satisfy the frozen gate: exact score bits, exact top-10, and
packed score-plane p99 below the in-process resident-f32 control. The next
slice is therefore a separately reviewed device-top-k and library-FFI design.
It must avoid returning the 4/128 MB dense score plane to the host, preserve
ascending-ordinal ties, and prove an end-to-end improvement of at least 20%
before becoming a public serving path.

No SOTA, model-quality, ANN, training, or end-to-end latency claim follows from
this kernel result alone.
