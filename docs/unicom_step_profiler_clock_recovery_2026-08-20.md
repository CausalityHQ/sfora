# UniCOM step-profiler clock recovery

## Observed failure and preserved evidence

The seed-2 random and imprinted 16-epoch training arms executed from source
`387d6979848c22f409e68b86b81df4d4e99ae03f` and both exited zero.  Their
checkpoints, initialization receipts, training logs, elapsed-time records, peak-memory
records, and held-out histories are immutable inputs to any recovery.  The first fresh
A-B-B-A profile, `random-a`, started at `2026-08-20T21:56:10Z`, exited `2` at
`2026-08-20T22:03:19Z`, and published no JSON profile.  Its preserved log ends with
`profiling failed: CUDA step exceeds wall step`.  The original queue then exited `1`
without starting another profile or another seed.  This failed attempt is never
relabelled as successful and is not a scientific measurement.

The failure occurred after all registered warm-up, timing, and profiler steps.  The
producer measured `step_wall_seconds` with the CPU `time.perf_counter()` clock and
measured the eight contiguous CUDA components with CUDA events.  It then required the
sum of the CUDA-event intervals to be no greater than the independent CPU-clock span
within relative tolerance `1e-9`.  That ordering is not a valid structural invariant:
the clocks have independent resolution and calibration, so a microscopic CUDA-over-
CPU difference can occur even though the events are correctly ordered and the final
CUDA synchronization completed before the CPU stop time.  The registered sample
schema, positivity checks, finite checks, and exact component order remain valid.  A
cross-clock sanity check is still useful, but it must allow physical clock uncertainty
and the whole-step CUDA span must be measured independently rather than defined as the
component sum it is supposed to check.

## Prospective repair

Before any further GPU process, make and independently review one source commit that:

1. replaces the `1e-9` cross-clock relative tolerance with exact registered constant
   `0.01`: at least 100 times a conservative 100-parts-per-million oscillator offset,
   but below the five-percent structural-error falsifier;
2. measures `cuda_step_seconds` independently as CUDA event `0` to event `8`, retains
   the eight component spans, and requires their sum to agree with that independent
   span within absolute tolerance `1e-5` seconds;
3. validates and names each measured row immediately, before starting the next row;
4. retains the exact timing/profile schemas, step counts, A-B-B-A order, CUDA event
   boundaries, CPU wall measurement, synchronization, component-sum check, bootstrap,
   kernel threshold `0.1`, evaluator formulas, training recipe, seeds, checkpoints,
   quality gates, and all Pareto gates;
5. adds RED-then-GREEN regressions accepting 1, 10, 50, and 100 ppm cross-clock skew,
   rejecting 5%, 100%, and unit-scale structural discrepancies, detecting an
   independent whole-span/component-sum mismatch, reporting the sample index and both
   durations, and aborting on the first invalid measured row; and
6. passes the complete profiler test file, Ruff, `py_compile`, and `git diff --check`,
   followed by one independent adversarial source review with no Critical or Important
   finding.

No observed metric selects a new tolerance or changes a decision boundary.  The
one-percent tolerance is fixed from a conservative physical clock-error bound and a
separate five-percent structural-error falsifier; the failed sample's unavailable
numerical difference cannot tune it.  Raw CPU wall time remains the profiled-compute
input; CUDA component times and the objective-only profiler remain separately reported
diagnostics.

## Recovery execution

The completed seed-2 training arms may be reused because they finished and were
published before the profiler process began; the profiler is read-only with respect to
their checkpoints.  Authenticate their original trainer SHA, receipts, logs,
checkpoints, and histories byte-for-byte under the reviewed recovery source.  Do not
rerun either training arm.

The failed `random-a` files and original queue exit remain untouched.  Use a new queue
stem and a new absent profile directory keyed by the reviewed recovery source.  From a
fresh clean detached checkout, with an idle GPU, run an entirely fresh four-profile
sequence in exact order `random-a`, `imprinted-b`, `imprinted-c`, `random-d`; never mix
an old or failed profile into the new A-B-B-A set.  Each profile must use the original
seed-2 epoch-4 checkpoint for its arm and must independently publish and strict-reload
its result.  Only after all four pass may the unchanged measurement builder and pair-v2
evaluator run once.

If seed 2 validates, continue seeds 3 through 6 serially under the same reviewed
recovery source and unchanged training/profile/evaluation recipes.  Preserve every
valid outcome.  Stop without automatic retry on any further structural, training,
profile, evaluator, resource, or provenance failure.  Build the six-seed summary only
after exact pair reports for seeds 1 through 6 exist and independently validate.
