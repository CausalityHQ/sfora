# Packed int8 cuTile device top-k result

The persistent cuTile scorer passed its frozen one-million-row integration gate
on an NVIDIA GB10 at source `e67a5d1dccce6ec005bd87f13779c1b4c110b393`.
The canonical receipt is
`docs/evidence/packed_int8_cutile_topk_summary.json` (SHA-256
`88e6874c93ca4f33e8856f823e82319d93773413dc9ca19630ac00674e73e42d`).

| Batch | Native p50 / p99 | Throughput | Prior materializing p99 | Resident-f32 p99 |
|---:|---:|---:|---:|---:|
| 1 | 1.021 / 1.550 ms | 954.6 q/s | 20.049 ms | 2.597 ms |
| 32 | 4.594 / 5.274 ms | 6,870.4 q/s | 25.635 ms | 8.665 ms |

Native p99 is 12.93x and 4.86x faster than the matched materializing path for
batches 1 and 32, respectively. It is also 1.67x and 1.64x faster than the
matched resident-float32 Torch control. These comparisons reuse the frozen
matched baseline receipt whose SHA-256 is bound by the new receipt.

Exact score bits and exact deterministic top-10 ordinals matched at the
1,000,003-row boundary for both batches. The persistent representation is 130
bytes/item (130,000,000 bytes at one million rows); maximum physical candidate
storage is 32,002,048 device bytes; process peak RSS was 1,552,142,336 bytes,
below the 2 GiB gate. The first JIT launches (0.896 s and 6.215 s) are recorded
separately and excluded from steady-state latency.

This is performance evidence only. It does not change descriptor quality and
does not by itself establish a matched SOTA retrieval-quality claim.
