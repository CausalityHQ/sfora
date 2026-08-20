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
schema, positivity checks, finite checks, exact component order, and same-clock
component-sum recomputation already validate the timing evidence without this
cross-clock ordering assertion.

## Prospective repair

Before any further GPU process, make and independently review one source commit that:

1. removes only the `cuda_step_seconds <= step_wall_seconds * (1 + 1e-9)` rejection;
2. retains the exact timing/profile schemas, step counts, A-B-B-A order, CUDA event
   boundaries, CPU wall measurement, synchronization, component-sum check, bootstrap,
   kernel threshold `0.1`, evaluator formulas, training recipe, seeds, checkpoints,
   quality gates, and all Pareto gates;
3. adds a RED-then-GREEN regression with a finite positive sample whose exact
   same-clock CUDA component sum exceeds the independent CPU wall span by one
   microsecond, while retaining the existing inconsistent-component rejection; and
4. passes the complete profiler test file, Ruff, `py_compile`, and `git diff --check`,
   followed by one independent adversarial source review with no Critical or Important
   finding.

No observed metric selects a new tolerance or changes a decision boundary.  The
repair deletes an invalid type of comparison rather than widening it until the failed
sample passes.  Raw CPU wall time remains the profiled-compute input; CUDA component
times and the objective-only profiler remain separately reported diagnostics.

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
