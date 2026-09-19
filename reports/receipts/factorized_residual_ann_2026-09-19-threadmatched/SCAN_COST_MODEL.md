# What actually limits the candidate scan

Measured on the DGX host against the real 2.4 GB `codes.u8` role of the
BigANN100M artifact, with the real access pattern (640 posting lists, ~1620 rows
each, 24-byte codes). Sources: `tlb_probe.c`, `scan_variants.c`.

## The scan is bytes-bound, not compute-bound

Varying only the bytes consumed per row, everything else held fixed:

| Bytes/row | ms per query-equivalent | rows/s/core | Effective GB/s |
| ---: | ---: | ---: | ---: |
| 24 | 17.193 | 60,303,688 | 1.45 |
| 16 | 11.376 | 91,140,098 | 1.46 |
|  8 |  5.355 | 193,627,984 | 1.55 |

Time tracks bytes almost exactly (24/16 = 1.50 against a 1.51 time ratio; 24/8 =
3.00 against 3.21), and effective bandwidth is flat at about 1.5 GB/s per core.
So to a good approximation:

    candidate_ms  ~  rows_scanned * bytes_per_row / (1.5 GB/s * effective_cores)

## Three things this rules out

- **Arithmetic is not the bottleneck.** The same inner loop over a
  cache-resident buffer runs at 2.28 G rows/s, so the real scan spends about
  **1.9%** of its time on arithmetic. A branch-free unroll of the 6-bit
  extraction measured 2.63x faster in isolation with bit-identical scores, and
  is therefore not worth landing: it optimises 1.9% of the cost.
- **TLB pressure is not the bottleneck.** `MADV_HUGEPAGE` plus `MADV_WILLNEED`
  over the 2.4 GB mapping changed throughput by 0.5% (57.45 -> 57.72 M rows/s).
- **The scattered access pattern is not the bottleneck.** Scanning 1,036,800
  contiguous rows from the same file gives 60.6 M rows/s against 57.5 M for 640
  scattered runs, a 5% difference.

## Consequence for method choice

Going faster means reading fewer bytes per candidate, or scanning fewer
candidates. It does not mean writing a faster kernel, and on this evidence it
does not yet justify a GPU kernel either.

This also reframes the Faiss fast-scan screen in `fastscan-screen-t20.json`.
That screen measured 302.5 M rows/s/core at 16 bytes/row, or 4.84 GB/s, against
1.46 GB/s here at the same 16 bytes/row. The gap is bandwidth efficiency, not
code width, and the screen's index holds 160 MB of codes against this artifact's
2.40 GB, a 15x working-set difference. The screen therefore cannot be read as a
kernel comparison; a like-for-like fast-scan control at 100M rows is required
before any conclusion about the two scan kernels.
