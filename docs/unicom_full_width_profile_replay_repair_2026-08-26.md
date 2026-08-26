# UniCOM full-width terminal-checkpoint profile repair

This amendment was frozen after both seed-0 training arms and their paired
quality evaluation completed, but before any A-B-B-A profile or seed-0
decision existed. The first position-1 `sampled_512` profile attempt produced
no output: restoring the registered epoch-16 checkpoint also restored a
completed 2,576-step `OneCycleLR`, and the first synthetic warmup step tried to
advance it to step 2,577. PyTorch rejected that impossible schedule step.

## Frozen repair

The epoch-16 checkpoints, A-B-B-A order, replay batches, warmup/measurement
counts, profiler counts, and every scientific/resource threshold remain
unchanged. Synthetic replay now treats an exhausted scheduler as immutable:
when `last_epoch == total_steps`, the replay step leaves the scheduler and its
terminal learning rates unchanged. A nonterminal scheduler continues through
the real `scheduler.step()` path. The optimizer, model, classifier, EMA,
scaler, masks, RNG streams, forward pass, loss, backward pass, and optimizer
step are otherwise unchanged.

The exhausted branch does not execute the invalid scheduler call, so terminal
replay wall samples exclude only that common post-update CPU bookkeeping. No
scheduler GPU work is removed; CUDA-event timestamps remain descriptive rather
than gating. Omitting a common positive cost is conservative when testing
whether the wider candidate is slower than the control because it moves a
candidate/control slowdown ratio away from one.

This rule avoids inventing unregistered learning-rate steps beyond the
completed training schedule and applies identically to both arms and both
reload positions. Regression tests use a real `OneCycleLR` to require both the
terminal no-op and the nonterminal advance behavior.

The A-B-B-A comparator additionally requires all four profile artifacts to bind
one identical profiler digest and one identical objective-module digest. A
mixture of pre-repair and post-repair profile code is therefore structurally
invalid even if every profile is individually well formed.

After the repaired A-B-B-A sequence completed, the first seed-0 decision
launch stopped before publication because the decision producer incorrectly
required the already-completed training receipts to claim the newer profiling
source commit and refrozen config digest. Those immutable receipts correctly
claim the source and config under which training actually ran. The repaired
decision contract therefore carries a separate `training_receipt_authority`
binding: the original detached config commit, its source parent, and its exact
config SHA-256. The decision cross-binds both training receipts and the profile
comparison to that historical training authority. At decision launch, the
producer independently authenticates the historical config commit's sole edge,
source parent, exact config bytes, and embedded source binding. It also
authenticates the executing checkout as the exact detached, clean, config-only
handoff for the newer profiler/decision implementation. This replaces the
incorrect requirement that historical training receipts carry the current
run-config digest; the current config digest is instead recorded in the
decision input binding. All scientific fields shared with the training config
remain exactly equal. No receipt, profile, quality value, timing value, or
threshold is changed, and the failed decision launch published no output or
temporary sibling.

## Retry and interpretation

The failed position-1 launch is recorded but does not consume the registered
position attempt: it returned no timing sample, published no artifact, and
left neither the registered output nor its temporary sibling. This exception
is limited to the observed prepublication scheduler-exhaustion failure; the
refrozen run configuration binds it mechanically and does not create a general
profile retry allowance. After this repair is committed and independently
reviewed, its source, tests, and this amendment are bound in `source.files` by
a new config-only handoff. The four profiles then run once from the beginning
in the original
`sampled_512/full_768/full_768/sampled_512` order. No result from the failed
process enters the comparison, and no quality or resource gate changes.

The prepublication seed-0 decision failure is likewise non-consuming only for
this observed historical-authority mismatch. The next config-only handoff must
bind the original training config commit/source/digest literally, and the
decision runs once from the already validated paired-quality and A-B-B-A
artifacts. This does not create a general decision retry allowance.
