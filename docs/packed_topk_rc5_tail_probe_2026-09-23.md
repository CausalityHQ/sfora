# RC5 public-call tail localization probe

RC3 `0.3.0rc3` remains the release. The RC4 512-entry merge candidate is a
diagnostic arm only. Its batch-32 public-call p99 failed in four paired runs,
with the largest call at timed sample 12 in each candidate arm. The cause is
unknown; the RC4 archives contain no per-call phase timing.

The first RC5 probe reuses the frozen 1,000,000-row GB10 fixture, the shipped
RC4 Python API bytes, the pinned RC3 native library, and the rejected candidate
native library. `scripts/profile_packed_topk_rc5_tail.py` authenticates all
eight fixture files, the API, and each library, and checks the first call's
score bits and ordered top-10 against the preserved reference. In separate
processes it replays batch 1 then batch 32, with five warmups and 50 timed
calls per shape. Run pair 1 RC3 then candidate; run pair 2 candidate then RC3.
Archive every output, including failures, with command lines and hashes. No
new release gate is inferred from these instrumented timings.

Each call records total time, time inside the `ctypes` native function call,
Python residual, GC callback intervals, allocated-block counts, minor/major
faults, and optionally `tracemalloc` current/peak bytes. This traces the
Python-to-native boundary without changing production code. The initial pair
uses allocation tracing; if tracing suppresses the tail, repeat once without
it. A candidate spike of at least 10 ms inside the native interval assigns the
stall to the native/driver side only if at least 80% of the excess over that
arm's median is inside the native interval. A GC overlap or allocation jump is
reported as correlation, not automatically cause. If the spike is inside the
native boundary, use one bounded Nsight CUDA API/memory trace to distinguish
pool growth, synchronization, and kernel time against RC3. If the spike is in
the Python residual, trace the public wrapper's validation, allocation, and
result conversion before proposing a fix.

The causal pilot changes one suspected mechanism at a time, keeps the same
fixture and call order, and retains all paired samples. Candidate promotion
requires a separate predeclared release gate and independent replications;
this localization probe alone cannot qualify the rejected merge change.

The first instrumented runs and two ordinary paired replays did not reproduce
the candidate tail. The second ordinary pair instead showed an 18.337 ms RC3
batch-32 call at index 12. A lightly instrumented replay also observed a
generation-2 Python GC lasting about 13.6 ms inside a batch-1 call; that
replay's batch-32 calls did not spike.
This makes GC a testable hypothesis, not a proven explanation of the RC4
batch-32 calls. The next bounded control uses the original replay loop with
only a GC callback (`scripts/profile_packed_topk_rc5_gc.py`), first with GC
enabled, then with GC disabled, each in fresh processes and matched RC3 and
candidate arms. If a >=10 ms call overlaps a generation-2 GC interval, that
directly attributes that call's stall; absence of a stall alone cannot prove
GC causality. Retain all runs even if they contradict the hypothesis.

The minimal callback runs found a fast generation-0 collection at batch-32
sample 12 in both libraries. For the required CUDA boundary check, the next
Nsight pair uses `scripts/profile_packed_topk_rc5_timeline.py`, which adds
preallocated start/end arrays and a wall/monotonic clock offset to correlate
each public call with CUDA API, memory-pool, synchronization, and kernel
records. This is diagnostic tracing only; profiler-inflated timings are not
release measurements. Capture one matched RC3/candidate pair and inspect
sample 12 plus the slowest call in each arm.

## Observed result and decision

All outputs, including the contradictory runs and four original Nsight reports,
are in `docs/evidence/packed_topk_rc5_tail_probe_raw_v1.tar.gz`, SHA-256
`8abad1bde341c876d33dac162e443d101f6988dbce77627cd6d8c0674d194dc3`.
The profiler source hashes are `521d07df7c494e28568790c1d64f9ad72bb81be55fd6ab1e2b4c927a4d67b721`
(per-call Python/native trace),
`809b93b9f30e037957b86b61895b6420534d969906ad79e74d7c6357bd35df65`
(minimal GC control), and
`8730601a97a1293579d459c22f89fc3ec9f20781ba199f7411ecb3ebc2b691f7`
(Nsight call timeline). The same RC3/candidate native library hashes, Python
API hash, and fixture manifest hash as the RC4 decision were verified before
each replay. All first calls matched frozen exact score bits and ordinals.

| Replay, batch 32 | RC3 p99 | Candidate p99 | Relevant event |
| --- | ---: | ---: | --- |
| Per-call trace, allocation tracking, pair 1 | 5.642 ms | 3.906 ms | No 10 ms spike |
| Per-call trace, allocation tracking, pair 2 | 5.598 ms | 3.918 ms | No 10 ms spike |
| Per-call trace, no allocation tracking, pair 1 | 5.334 ms | 3.453 ms | No batch-32 spike; candidate batch-1 generation-2 GC lasted 13.6 ms |
| Per-call trace, no allocation tracking, pair 2 | 5.211 ms | 3.465 ms | No 10 ms spike |
| Original uninstrumented loop, pair 1 | 5.174 ms | 3.437 ms | No 10 ms spike |
| Original uninstrumented loop, pair 2 | **18.337 ms** | 3.270 ms | RC3 sample 12 spiked |
| Minimal GC callback, enabled | 5.179 ms | 3.407 ms | Fast generation-0 GC at sample 12 in both arms |
| Minimal GC callback, disabled | 5.231 ms | 3.434 ms | No GC, no 10 ms spike |
| Nsight per-call timeline | 5.265 ms | 3.426 ms | Fast generation-0 GC at sample 12 in both arms |

The matched Nsight timeline maps public-call start/end timestamps to CUDA API,
kernel, and memory-pool records. At sample 12, RC3 took 4.714 ms with eight
`cuMemAllocAsync` calls, 13 `cuStreamSynchronize` calls, nine kernels, and a
164.264 MB maximum tracked CUDA pool utilization. The candidate took 2.985 ms
with ten allocations, 16 synchronizations, 12 kernels, and 165.014 MB maximum
pool utilization. These pool maxima matched ordinary calls in each arm; neither
arm had a sample-12 CUDA allocation or synchronization surge. The profiler
raises process RSS to about 1.29 GB, so its timings are diagnostic only.

The archived 16–19 ms candidate stalls remain real failed RC4 gate samples.
Their cause is **not proven**: no archived call has both a stall and GC/CUDA
telemetry. The new RC3 stall at the same index and the measured 13.6 ms
generation-2 pause make Python GC a plausible shared host cause, but the
GC-disabled control had no corresponding enabled stall and cannot prove that
hypothesis. There is no justified candidate-specific fix from this probe.
The next RC5 implementation targets the separately measured 0.795 ms batch-32
buffer-initialization stage in the retained 2048-width scorer. Its benchmark
gate must be declared before the source change, and the rejected 512-width
candidate is not promoted using these favorable new samples.
