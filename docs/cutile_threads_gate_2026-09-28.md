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
