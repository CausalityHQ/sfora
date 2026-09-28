# Frozen same-handle thread gate

Rebuild an isolated preserved runtime-bound candidate plus committed FFI guard,
with all source hashes checked and the crate's shared Cargo output invalidated
before compilation. CPU-only self-check and offline locked full test compilation:
120s/8GiB external cap. No dependency installation or source tuning.

One DGX GPU controller,90s/8GiB host cap and shared Sfora GPU lock. Execute the
four compiled Rust test targets first; require all15 tests to pass. Then one
fresh Python process creates one gallery on the main thread and searches it
from two distinct OS threads in sequence. Keep the first thread alive until the
second finishes, avoiding reused thread identifiers. Each thread performs first
and repeat searches at batch1, then first and repeat at batch32: eight calls.
Use the existing59551-row synthetic fixture and independent scalar-bit oracle.
Require every ordinal and score bit exact, all outputs retained, distinct thread
IDs, and no JIT misses on either repeat call. Exceptions must propagate.

Record native `CUTILE_JIT_TIMING=1` entries separately by thread/batch/phase.
Installed thread-local cache predicts new compilation on the second thread;
this gate measures that behavior, not global reuse or a speed improvement.
Six misses per thread are expected: two fill plus score/merge for each batch.
Unexpected counts invalidate that cache prediction; they do not authorize a
tuning loop. Any exactness/test/authority failure stops this configuration.
No retry on a failed GPU result, no training, images, official quality or product
latency claim. Promotion remains a separate decision; the original dirty top-k
file stays untouched and unstaged. The joint quality-and-speed goal is unmet.

## Terminal result

The sole GPU invocationc1c5fd7c7d644010b146ddf112281370 passed all15 compiled
Rust tests and all eight thread searches exactly. OS native IDs1017162/1017175
were distinct. Misses by frozen phase were4,0,2,0 for EACH thread, confirming
six unique keys compiled independently on each thread and no repeat misses.
Every retained ordinal and score bit also matches the previous independent
59551-row scalar-oracle archive; the fixture helper is byte-identical.

| Individual call, ms | Thread1 | Thread2 |
|---|---:|---:|
| Batch1 first |695.679|676.251|
| Batch1 repeat |0.259152|0.250847|
| Batch32 first |639.143|661.405|
| Batch32 repeat |0.349232|0.344767|

These eight individual calls are diagnostics, not quantiles, QPS or a matched
speed estimate. A new caller thread recompiles even with the same gallery
handle. Deployment must account for per-thread warmup; global cache reuse and
concurrent search behavior were not established. No threading implementation
or scheduling policy changed.

CPU invocation7724bffc14af4366b700689c7b987a2a passed in16.16s GNU wall,
934548KiB RSS. GPU controller20.58s/479696KiB RSS, exit0, including the complete
Rust suite and thread subprocess. CUDA peak memory was not measured. Source
manifest checks preserve the exact candidate top-k hash and committed FFI guard;
each executable was copied and hash-checked before execution, with no GPU build.
The [raw receipt and stdlib replay](evidence/compact_metric/sop-siglip2-substrate-v1/cutile-threads-v1/receipt.json)
verify source identity,15 tests, all eight scalar-bit outputs and native counts.
All jobs are terminal. This closes the requested sequential thread coverage;
it does not promote the protected dirty file or change dataset quality.

The next work must return to the unresolved representation/quality gap and
qualified full-pipeline comparison. Additional native microbenchmarks are not
justified by this passing gate. The prior sub-center proposal remains rejected;
a later converged-gradient read would require the missing exact TRAIN fit
embeddings. The checkpoint writer saves classifier/model weights but not the
member bank, so its checkpoint cannot substitute for that tensor.
