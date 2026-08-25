# UniCOM full-width prelaunch repair

This amendment is prospective. No full-width training process, checkpoint,
profile, paired result, or decision existed when it was written. It repairs the
historical handoff `c3e147237fc1ff40e2a01911582b15cd36d72598` after an
independent prelaunch review found three executable-contract gaps.

## Frozen repairs

1. Seed 0 has one production decision command and one registered immutable
   decision path. The producer strict-loads and validates the paired evaluator
   result, A-B-B-A comparison, pair inventory, and both training receipts;
   cross-binds every checkpoint path/hash/byte count; recomputes the registered
   prediction and operational predicates; then publishes with strict reload and
   no-clobber semantics. No ad-hoc post-result calculation may authorize the
   confirmation panel.
2. `ratios.step_wall` is the sole operational step-time authority. The CUDA
   event ratio remains descriptive evidence and cannot select or reject the
   candidate. Peak allocated/reserved memory remain authoritative in the two
   training receipts.
3. A registered training arm requires its output directory to be absent and
   its parent to be a real directory before training begins. A crashed partial
   directory therefore permanently consumes that arm's one attempt instead of
   being silently reused or overwritten.
4. The decision producer requires receipt runtime values to equal the frozen
   Python, Torch, and CUDA versions, and requires exactly 3,200 optimization
   identities for every registered arm including seed 0.

The run configuration is refrozen only after these source changes receive an
independent no-Critical/Important review. The new handoff is a direct child of
the reviewed repair source and changes only the run configuration.

## Interpretation frozen before outcome

The candidate keeps the official ArcFace calibration (`scale=32`,
`margin=0.25`) while changing the loss width from 512 to 768. A negative result
therefore closes full-width training at that official calibration; it does not
close every separately preregistered full-width scale. No scale is changed in
this experiment.

The CPU preflight must also prove that changing only the frozen protocol width
preserves serialized checkpoint byte count on a real checkpoint fixture. This
protects the exact-byte resource gate before two costly 16-epoch runs.
