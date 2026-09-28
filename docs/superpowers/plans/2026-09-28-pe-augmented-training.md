# Paired PE actual-image TRAIN gate plan

Spec: docs/inshop_pe_augmented_training_gate_2026-09-28.md. Inline execution
under existing autonomous authorization; preserve protected Rust and prior runs.

1. One minimal runnable CPU check first: identical augmentation across unrelated
   RNG state, RNG restoration, changed-file rejection, singleton rank policy and
   changed preflight rejection. Observe missing implementation fail.
2. Add train_inshop_pe_pair.py reusing native source/freeze/gradient/initializer/
   bank/scaler/packed-score/role/bootstrap helpers. CPU preflight freezes full-fit
   initializers, 100-of1000 schedule, native pixel+RGB hashes and executable/source
   authority. One paired runtime produces actual checkpoint, held vectors and raw
   integrity/cost/quality receipts. Keep old immutable root; clone a new small root.
3. Run fixture/preflight/startup CPU checks and Ruff. One combined Opus/Astra
   executable/design critique, reconcile findings and rerun affected CPU checks.
   Freeze commit/push BEFORE one600s8GiBshared-lock DGX pair after idle check.
4. Observe same service, collect original terminal/cost. Independently replay
   actual checkpoints and held packed scores with class/query uncertainty; report
   gate decision and next action, publish master and notify operator once.

Review focus: singleton-containing bank loss, CPU versus CUDA RNG seeding,
matched augmented RGB before different native processors, immutable input/source
and external manifest hashes, saved checkpoints' frozen native/foreign PE state.
No official read, loss variant, cap/threshold rescue or duplicate job.
