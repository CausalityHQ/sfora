# Preregistration: thread-matched Sfora vs Faiss (BigANN100M dev 0-999)

Written 2026-09-19 before any arm was executed.

## Question

Is Sfora's measured latency advantage over the Faiss control real, or an
artifact of unmatched scan parallelism? The existing control sets 20 OpenMP
threads but never sets `parallel_mode`, and submits one query per call, so
Faiss's default query-parallel mode leaves its scan effectively single-threaded
while Sfora parallelizes across posting lists with thread_count=20.

## Arms (same host, same index/codes/routing, batch-1, dev queries 0-999)

- A  Sfora native + direct I/O, thread_count=20   ALREADY MEASURED p99 14.411351 ms, RSS 2,975,289,344 B
- B  Sfora native + direct I/O, thread_count=1    NEW
- C  Faiss exact-coarse, parallel_mode=0, 20 OMP  ALREADY MEASURED p99 56.760716 ms, RSS 3,920,982,016 B
- D  Faiss exact-coarse, parallel_mode=1, 20 OMP  NEW

Thread-matched pairs: B vs C (1 effective scan thread), A vs D (20 threads).
All arms record ru_utime+ru_stime over the measured loop so total CPU work is
comparable, not just wall clock.

## Predictions (point estimate, then interval)

1. D p99 = 8 ms (interval 4-20 ms). D BEATS A's 14.411 ms.
2. B p99 = 180 ms (interval 100-300 ms). B is 2-5x WORSE than C's 56.760 ms.
3. CPU per query: Sfora ~150 thread-ms vs Faiss ~50 thread-ms, i.e. Sfora does
   roughly 3x more total work per query.
4. Recall is unchanged by parallel_mode: both Faiss arms stay at 0.988120.

Net prediction: Sfora LOSES both thread-matched comparisons. Stated plainly so
it can be wrong.

## Decision rule, fixed in advance

- If D < A and B > C: the latency advantage is an artifact. The current scalar
  6-bit scan has no performance story at matched threads. Do not tune it; the
  only paths worth pursuing are reducing bytes scanned per candidate (4-bit
  fast-scan-style layout, RaBitQ-style estimators) or moving the scan to the
  GPU. Report the negative plainly and reset the claim.
- If D > A at matched threads: the advantage is real and attributable to the
  method. Strengthen it and claim it, with the matched receipt as evidence.
- If results straddle (e.g. D beats A but B beats C): report as mixed, attribute
  to per-thread efficiency vs parallel scaling, and measure single-core
  rows-scanned-per-second directly before any claim.

Recall must stay at 0.988120 in every arm; any arm that moves recall is invalid
and is rerun, not interpreted.

---

# Preregistration 2: OpenMP schedule for the candidate scan

Written after arms B/C2/D, before changing any code.

## Measured baseline (dev 0-999, same host, recall 0.988120 in every arm)

| Arm | Threads | mean ms | p99 ms | CPU ms/query |
| --- | ---: | ---: | ---: | ---: |
| Sfora  | 1  | 27.750 | 42.024 | ~27.8 |
| Faiss  | 1  | 36.553 | 57.188 | 40.05 |
| Sfora  | 20 | 11.476 | 14.411 | - |
| Faiss  | 20 |  8.815 | 10.968 | 67.40 |

Sfora stage split: candidate 24.507 ms (1 thread) -> 7.600 ms (20 threads),
i.e. 3.22x scaling, 16% parallel efficiency. Exact stage 3.237 -> 2.963 ms,
i.e. 1.09x, essentially serial.

## Hypothesis

`src/sfora/_native/factorized_residual_ann.c:625` scans probed lists with
`#pragma omp parallel for schedule(static)`. The probe heap is ordered by
centroid distance, so static contiguous chunking gives thread 0 the 32 nearest
lists. IVF lists near a query are systematically the densest, so the static
partition is close to worst case and wall time tracks the unluckiest thread.
The loop body is independent, per-thread heaps are merged afterwards, and the
merged top-k is partition-invariant, so scheduling cannot change results.

## Prediction

Switching to `schedule(dynamic, 1)`:

1. Candidate stage at 20 threads falls from 7.600 ms to 2.0-4.0 ms.
2. End-to-end mean falls from 11.476 ms to 5.5-7.5 ms, BEATING Faiss's 8.815 ms.
3. Single-thread timing is unchanged within noise (one chunk either way).
4. The ordered-output SHA-256 stays exactly
   ec16438f507a7331b24f60f6291115fa13346e3ec332bb0d51f3258589a8b444 and recall
   stays 0.988120. Any change here invalidates the experiment.

## Decision rule

- Prediction 4 fails -> revert immediately; scheduling is not result-neutral.
- Candidate stage does not improve -> the bottleneck is not imbalance; measure
  per-thread row counts directly before trying anything else.
- Prediction 1 holds -> keep it, then attack the ~3 ms serial exact stage
  (per-query io_uring setup) as the next target.

---

# Preregistration 3: reuse the direct-I/O buffer mapping

## Measurement that motivates it

Microbenchmark on the same host, 200 iterations each:

- 8 MiB anonymous mmap + MADV_DONTFORK + first-touch of 2048 pages + munmap:
  **3.243 ms per iteration**
- io_uring setup + 3 ring mmaps + munmaps + close: **0.023 ms per iteration**

So the per-query fixed cost is the buffer mapping and its page faults, not the
ring. An earlier reviewer attributed the exact stage to io_uring setup; that
attribution is wrong by two orders of magnitude and is not what I am fixing.

For BigANN the buffer is candidate_count(1024) x buffer_stride(8192) = 8 MiB,
mapped and unmapped on every single query.

## Change

Cache the buffer mapping in a process-wide static, reuse it whenever the
required size matches, and add `sfora_direct_release_buffer` so close paths can
release it. Ownership transfers to the quarantine record on the poisoned path,
which must clear the cache. The ledger already reserves these bytes for the one
admitted direct context, so residency does not change the memory budget.

## Prediction

1. Exact stage falls from 2.548 ms to 0.2-0.8 ms at thread_count=20.
2. End-to-end mean falls from 8.625 ms to 6.2-7.2 ms, p99 from 9.849 to
   7.0-8.5 ms, against Faiss 8.815 mean / 10.968 p99.
3. Recall stays 0.988120 and the ordered-output digest stays byte-identical.
4. No mapping or descriptor leak across repeated load/search/close, as the
   design requires; the existing native tests must stay green.

## Decision rule

- Prediction 3 fails, or any quarantine/fork/close test regresses -> revert.
- Exact stage does not fall below ~1.5 ms -> the cost is not the mapping;
  revert and measure where the exact stage actually spends its time.

---

# Preregistration 4: parallelize the coarse centroid search

## Evidence

Candidate-stage scaling with dynamic scheduling already applied:

| threads | candidate ms | scaling |
| ---: | ---: | ---: |
| 1  | 24.507 | 1.00x |
| 4  | 10.042 | 2.44x |
| 10 |  6.930 | 3.54x |
| 20 |  6.045 | 4.05x |

An Amdahl model with a 4.5 ms serial term and a 20 ms parallel term predicts
9.5 / 6.5 / 5.5 ms at 4 / 10 / 20 threads against measured 10.04 / 6.93 / 6.05.
`factorized_residual_ann.c:640` scans all 65,536 centroids x 128 dimensions
serially before the parallel posting scan: 8.4M FMAs and 33.5 MB streamed per
query, which is more bytes than the posting scan itself.

## Change

Parallelize the coarse search with per-thread top-nprobe heaps merged
afterwards, the same pattern the posting scan already uses. Raise the native
admission term from probe_count*8 to thread_count*probe_count*8 to cover the
per-thread coarse heaps.

## Prediction

1. Candidate stage at 20 threads falls from 6.045 ms to 2.0-3.0 ms.
2. End-to-end mean falls from 8.235 ms to 4.3-5.4 ms, p99 from 9.459 to
   5.2-6.8 ms, against Faiss 8.815 mean / 10.968 p99.
3. Recall stays 0.988120 and the ordered-output digest stays byte-identical;
   top-nprobe selection is partition-invariant exactly as the posting scan is.
4. The ledger regression test still passes with the raised native term.

## Decision rule

- Prediction 3 fails -> revert; the merge is not order-invariant.
- Candidate stage does not fall below 4 ms -> the serial term is not the coarse
  search; revert and profile the stage directly instead of modelling it.
