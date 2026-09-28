# Frozen matched native comparison

The completed dual review6cf20281a3c24920 conditionally permits this single
diagnostic. No training, image inference, official quality read or promotion.
Original committed top-k versus the preserved runtime-bound snapshot; all other
crate files, locked dependencies, toolchain, Python API and checker identical.

Use `scripts/probe_cutile_runtime_bounds.py`, synthetic59551-row/128-dimension
gallery, top10, distinct queries, batches1/32, late tied winners across merge
groups and the final partial block, negative/zero dots and a separate all-tie
gallery. Independent integer-dot/float32 scalar oracle checks ordinals AND
score bits for the first call, five warmups, all50 timed calls and all-tie call.
Checks occur outside timed intervals; retain all main outputs. CPU self-check
must pass including full geometry and rejection of deliberately corrupt bits.

One offline locked release library build per arm, sequential, CPU-only:
240s total/8GiB host cap. Stop on failure; no dependency installation.
Copy each built library before building the other, hash both and source manifest.
One GPU controller, shared Sfora GPU lock,600s/8GiB host cap,75s per child.
No overlapping GPU job. Fresh processes in fixed order: b1 original,candidate,
candidate,original, then b32 original,candidate,candidate,original. Eight total;
five warmups and50 samples each,100 samples per arm/batch. Same caller thread,
assembler13.3.36 and `CUTILE_JIT_TIMING=1`. No hardware-cold claim.

Mandatory gates: every output exact; identical fixture hashes between arms;
no JIT misses during PHASE_HOT; candidate pooled p50<=1.05x original and
p95<=1.10x original for BOTH batches. Failure stops this configuration without
retuning, repeating or changing the frozen tolerance. Report constructor, first
synchronous search, native miss counts/stages and hot latency separately.
Startup improvement is descriptive, not required and not presumed. Pooled
quantiles are diagnostic; process replicates are not independent product calls.
No p99, QPS or full image-to-top-k speed claim follows from100 samples.

The first CPU self-check passed0.67s/194944KiB RSS; the expanded full-geometry
check passed0.96s/432576KiB RSS, exit0, CUDA hidden. Original invocations
33beac05d02e494780c32417d229192a and90e3a11e7af644b9aa55ea7a33992431.
The candidate remains untouched and unstaged; shared FFI size validation,
sequential cross-thread cache coverage and full assurance remain release gates.
The joint SOP/In-Shop quality and full-pipeline speed goal remains unmet.

## Build-authority correction before GPU launch

The first CPU controller exited0 in1.82s, but its two libraries had identical
hashes: the candidate build said `Finished` without recompiling. Both snapshots'
source mtimes were identical and Cargo's shared target reused the crate output
across roots; its dependency file changed to the candidate path without a new
compile. This invalidates the candidate binary authority, not the frozen method.
Retain this failed check. Invalidate only this crate's release build output and
rebuild the candidate offline, with a238s cap so total CPU caps remain240s.
Require an actual candidate compilation and different final library hashes
before GPU launch. No fixture, arithmetic, performance threshold or process
order changes; no GPU retry is authorized by this correction.
