# Isolated preprocessing worker: bounded CPU falsifier

The global threads=1 serving certification passed batch1 but failed batch32
tail criteria. Test a distinct operational mechanism: one persistent spawned
CPU worker runs only the existing pinned public preprocessing path (direct
batch1 transform, HF processor at batch32) with one intra-op
thread; the caller remains at20 threads. No encoder, CUDA, training, packed
scorer or library default changes. This keeps inference/packing launch behavior
in the parent and explicitly includes serialized input/output and IPC overhead.
Use stdlib ProcessPoolExecutor with spawn, one worker, no new dependencies or
shared-memory buffer framework. Reject it if this simple version is too slow.

First CPU-only fixtures cover RGB, grayscale, palette and RGBA images with
different source sizes; both paths perform the same RGB conversion. Require
bitwise equal float32 1/32×3×256×256 pixel tensors, identical public-path strides
and finite values.
Then use32 fixed evenly spaced SOP TRAIN archive rows, authenticate archive,
processor/config, serving source, actual input files and package versions.
Decode before timing; stage times include RGB conversion, processor and full
worker round trip, not file read/decode, encoder or search. Test batch1 and32,
five ABBA/BAAB blocks, two measured calls per position:20 calls per arm/size.
Warm both arms first; keep every raw duration and verify each output outside
the timed interval. Separate cold worker initialization cost. One DGX CPU user
unit, CUDA hidden,180s/8GiB, process-group kill, no duplicate/retry/cap rescue.

CPU advance only if exactness passes, worker p95<=0.8×parent at both sizes,
and median stage savings>=0.81ms at batch1 and>=14.21ms at batch32. These
savings floors correspond to5% of previously measured20-thread full-call
medians16.203/284.217ms; they are an engineering plausibility screen, not a
new full-call measurement or proof that savings add directly to latency.
Any failure closes this specific worker/IPC configuration before GPU or API
implementation. No tuning worker counts, serializers, thresholds or thread
settings after outcomes. A pass permits one separately reviewed/frozen matched
public image-to-top-k parity and latency pilot. A product p99 claim still
requires10,000 interleaved paired calls, varied inputs and confidence intervals.

Worker startup, lifecycle, concurrent-call safety and bounded input memory would
need qualification before integration. This spike creates no worker service
API. Preserve the protected Rust change and every closed quality arm. The
production SOP/In-Shop quality and matched speed goal remains active and unmet.

## Control-layout correction before any timing read

The initial CPU unit5a4f05a5219a4c04ac731fcfd86e1dae exited1 before paired
timing: the control itself failed an invented contiguity guard. The unchanged
public batch1 direct transform is channels-last with stride(3,1,768,3), not
ordinary contiguous NCHW. Its values and layout are valid. Preserve the failed
log/cost; no method timing or decision was produced. Correct the engineering
contract to identical baseline strides, with a runnable real-spawn fixture
that rejects forced contiguous conversion. Permit one corrected CPU execution
with new output/log/unit names, unchanged resource caps and performance floors.
This repairs a control-side guard before outcomes; it does not reopen a negative
worker performance gate. Do not convert public pixels or change encoder layout.
