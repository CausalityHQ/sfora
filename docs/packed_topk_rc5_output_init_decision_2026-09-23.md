# RC5 output-initialization pilot: rejected

**Decision:** reject the uninitialized-output candidate and keep the RC3
production scorer, `MERGE_WIDTH=2048`, public API, and package version
`0.3.0rc3`. The candidate preserved exact first-call score bits and ordered
top-10 ordinals for batches 1 and 32 at 1,000,000 and 1,000,003 rows. It
removed six output-fill kernel launches per warmed batch-32 search and kept
the tracked CUDA pool peak at 164.264 MB. Its public-call batch-32 p99 failed
the predeclared 10% paired-gain gate in both independent replay pairs.

The candidate source SHA-256 was
`0063e164654a78535f8d553c40bf7d34488c655754c931117b7a3288086993e5`;
its native library SHA-256 was
`b7440d394724249647e681a966345f4fb6ce5f1902871137d9e19eefdc524318`.
The exact candidate source, library, reference files, paired raw samples, and
matched Nsight reports are archived in
`docs/evidence/packed_topk_rc5_output_init_failed_raw_v1.tar.gz`, SHA-256
`b2f30944430219c49c40cb30fb954ea6db8708b43520fa7ea840c554a43878e9`.
All eight consumed fixture files were checked against manifest SHA-256
`2d34832dfe22378b767f9cbedc0911e04ff2e06add2d1fdea66e6d0d55497799`.
Unlike the older RC4 public-call archive, this raw archive also retains and
hashes the four consumed exact-reference JSON files. The public Python API
SHA-256 remained
`7e585fa716ad79b0ad9f998ac6eb63a4f9c7a89409eefee2d9367885686c3818`.

Each row below is a fresh 50-call Python replay after five warmups on the
same authenticated one-million-row gallery. Pair 1 ran RC3 first; pair 2
ran the candidate first. Throughput is queries per second from the sample
mean, not the median. p99 is the maximum of 50 samples.

| Pair | Batch | Library | p50 / p95 / p99 (ms) | Throughput (q/s) | Peak RSS (MB) |
| ---: | ---: | --- | ---: | ---: | ---: |
| 1 | 1 | RC3 | 1.053 / 1.175 / 1.443 | 936.3 | 718.4 |
| 1 | 1 | candidate | 1.024 / 1.125 / 1.344 | 964.7 | 719.4 |
| 1 | 32 | RC3 | 4.561 / 5.201 / 5.313 | 6,951.8 | 718.4 |
| 1 | 32 | candidate | 4.308 / 4.625 / **20.171** | 6,875.0 | 719.4 |
| 2 | 1 | RC3 | 1.044 / 1.223 / 1.282 | 945.4 | 718.2 |
| 2 | 1 | candidate | 1.016 / 1.248 / **1.355** | 960.7 | 718.8 |
| 2 | 32 | RC3 | 4.599 / 5.162 / 5.309 | 6,895.5 | 718.2 |
| 2 | 32 | candidate | 4.339 / 4.975 / **20.074** | 6,801.8 | 718.8 |

Both candidate batch-32 tails were at sample 12, and pair 2 also breached
the 5% batch-1 regression guardrail by 5.72%. The 2 GiB RSS gate passed.
No samples were removed as outliers. The non-tail batch-32 p50 improved by
about 0.25 ms, smaller than the 0.795 ms initialization estimate from the
earlier diagnostic split. The matched Nsight trace confirms three candidate
GPU kernels versus nine RC3 kernels per warmed batch-32 call, with the same
164.264 MB tracked CUDA pool maximum. Kernel time at representative sample 0
fell from 4.765 to 4.255 ms; that is a diagnostic ~0.51 ms GPU reduction,
not a qualifying public-call p99 gain. The pool figure is tracked utilization,
not a device-wide allocation peak.

## Tail attribution and next measured boundary

A separate matched RC3 Nsight/GC run captured an actual 20.441 ms batch-32
public call at sample 12. A generation-2 Python GC overlapped it for
15.673 ms, while its CUDA kernels totaled 4.417 ms and pool utilization
stayed constant at 164.264 MB. This proves that a shared host GC pause can
produce a sample-12 tail. The candidate's failed paired calls lack simultaneous
GC telemetry, so their exact cause remains unproved. The independent RC5
tail probe and claim limits are recorded in
`docs/packed_topk_rc5_tail_probe_2026-09-23.md`.

The output-initialization source change was reverted after the failed gate.
The next bounded causal pilot targets the Python/`ctypes` boundary, which
creates per-call argument objects and was unchanged in both the rejected
512-width and output-initialization candidates. It will compare the shipped
`ndpointer` dispatch with raw address dispatch after the same existing input
validation, measure GC allocation counts, and require exact ABI/API outputs.
This is a different measured boundary from the CUDA merge and fill kernels.
No RC5 performance release is qualified by this result.
