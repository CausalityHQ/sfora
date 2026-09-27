# SOP public query stage profile, 27 September 2026

Profile the selected seed-179024 true-freeze production `Siglip2CompactIndex`
against its 59,551-row official **TRAIN** gallery on DGX Spark GB10. Use the
same first pinned SOP TRAIN image as the completed batch-1 p99 gate. Keep the
authenticated checkpoint, FP16 mode, processor, 130-byte wire, native exact
scorer and runtime. Warm five calls, then time 100 complete calls with separate
CUDA synchronizations after transfer, vision, head/packing and native search.
Require the staged output to match the public API's top-10 ordinals and scores
on every call. Record per-stage p50/p95, hardware, source/image/checkpoint
hashes and peak allocated CUDA memory.

This is a stage attribution diagnostic. Synchronizations change its latency,
so its sum and 100-call tail are **not** a substitute for the existing 10,000
call public p99 result. If one non-vision stage consumes at least 25% of the
staged whole-call median, inspect one exact-output production optimization of
that stage. Otherwise target vision execution. Any candidate must pass exact
packed top-10 parity and a separate paired public p50/p95/p99 latency gate
before production promotion. No retrieval quality or SOTA conclusion follows
from this profile.

## Terminal result

The first service stopped before a receipt because the profiler paired seven
timestamps with six stage labels under `zip(..., strict=True)`. The corrected
one-line slice was locally checked and pushed before the single rerun; the
[failed journal](evidence/compact_metric/sop-siglip2-substrate-v1/sop-public-stage-profile-179024/failed-v1-journal.log)
has SHA-256 `4ede5b2fbe4af5ca0bfe9605c24efeb8b06194142e4691ecd1a5b32d797d9665`.
The rerun exited successfully with 100 output-stable staged calls, each
matching the public API's exact top-10 ordinal and score bytes. Its
[source-bound receipt](evidence/compact_metric/sop-siglip2-substrate-v1/sop-public-stage-profile-179024/receipt.json)
has SHA-256 `62780c024d52ab65cf5b9f7a0fe8c82c0175a6ed0dcba45bc83a51bd07dc1987`,
profiler SHA-256 `b0b9678e8edb7753a5394fe18c7e764f375110ddccbafd4178531f0a569260fb`,
and historical serving source SHA-256 `17deaca3b7a07c30664d52bcd3ef2abae56eb3a81458f4371530f00ca68d8079`.
The [successful journal](evidence/compact_metric/sop-siglip2-substrate-v1/sop-public-stage-profile-179024/v2-journal.log)
has SHA-256 `995cfe377c3dc017f3423a4ca4611cc589e4dfdfee78552e8c802e34e4a4d79c`.

| Synchronized stage, one fixed SOP TRAIN image | p50 | p95 |
| --- | ---: | ---: |
| JPEG read/decode | 0.547 ms | 0.570 ms |
| Image processor | **6.412 ms** | **8.246 ms** |
| Host-to-GPU transfer | 0.693 ms | 1.086 ms |
| SigLIP2 vision | **8.425 ms** | **9.249 ms** |
| 128-D head and packing | 0.321 ms | 0.403 ms |
| Exact native search | 0.329 ms | 0.376 ms |
| Whole staged call | 16.952 ms | 19.303 ms |

Peak allocated CUDA memory was **675,695,104 bytes**. The processor is
**37.8%** of whole-call p50 and clears the frozen 25% inspection trigger.
An independent 100-call wrapper timing inside the unmodified public API
observed 6.543 ms processor p50, confirming that the staged processor cost
was not just the extra stage synchronization. This wrapper timing is
exploratory and has no durable per-call receipt.

Two standard input forms were screened without changing production code.
Passing a CPU tensor to the same Transformers processor produced exact input
pixels, packed codes, ordinals and scores for the fixed query, but a short
interleaved public-call screen measured **16.167 vs 15.763 ms** batch-1 p50
and **18.949 vs 18.064 ms** p95, so it was rejected. Passing a GPU tensor
reduced isolated processor p50 from 0.431 to 0.166 ms, but its different
resize arithmetic changed **46 of 128 packed coordinates** and returned
scores on that first query. These are exploratory one-image checks, not a
paired p99 or quality panel. Neither path advances to production. Keep the
current serving code and seek an exact-output preprocessing or encoder
optimization before another public tail gate.
