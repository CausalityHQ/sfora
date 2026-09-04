# UniCOM Finish Causal Panel Design

## Purpose

Determine whether the observed rank-finish gain is caused by additional
training, identity-balanced sampling, or the SmoothAP objective. The existing
seed-0/1/2 receipts remain immutable robustness evidence for the combined
recipe. This panel is a new, claim-ineligible experiment from the same
authenticated epoch-4 parent.

## Phase-one arms

Use finish seed 3 and execute epochs 5--8 from the exact parent checkpoint.

- `A_CLASSIFICATION_PADDED`: original eight-mask ArcFace objective and original
  `PaddedEpochSampler`.
- `B_CLASSIFICATION_PK`: the same ArcFace objective with the registered
  32-identities-by-4-images schedule.
- `C_SMOOTH_AP_PK`: the existing deployment-prefix SmoothAP objective with the
  byte-identical PK schedule used by arm B.

All arms restore the parent's model, classifier, AdamW moments, 16-epoch
OneCycleLR state, scaler, mask generator, and EMA. They run exactly 161
attempted updates per epoch and 644 attempted updates total. No arm constructs
a new four-epoch scheduler. Compilation, fused AdamW, and BF16 remain disabled.
Evaluation must save and restore Python, NumPy, CPU Torch, and CUDA RNG state so
evaluation cannot change later training batches.

Arm A uses `PaddedEpochSampler(seed=3).set_epoch(epoch)` for zero-based epochs
4--7. Arms B and C use `identity_balanced_batches` with seed 3 and registered
epochs 5--8. B and C schedule digests must be identical before execution.
ArcFace arms keep the classifier and masks active. SmoothAP leaves the restored
classifier in the optimizer with `grad is None`; it must never install a zero
gradient that would advance classifier optimizer state.

## Evidence

Persist a versioned config before any outcome. Each result binds the config,
source, parent checkpoint and receipt, partition, arm, finish seed, schedules,
initial/final model and classifier state hashes, attempted/successful/skipped
updates, epoch metrics, raw model, EMA model, elapsed time, and peak allocated
and reserved CUDA memory.

At parent epoch 4 and arm epoch 8, persist an immutable evaluation evidence
bundle containing ordered query and gallery paths and labels, full 768-D FP32
descriptors, deployment geometry identity, per-query AP@R and Recall@1/10/20/30
indicators, relevant counts, and deterministic ranked prefixes. Recompute every
aggregate from the rows. Existing aggregate-only rank-finish receipts are not
rewritten.

Report the source-image schedule census: unique images exposed, omitted images,
identity visit range, repeated slots, and same-source directed positive pairs.
The observed seed-2 diagnostic is recorded but cannot select an arm.

## Phase-one decision

All three arms finish absent an operational stop. The primary contrasts are
`C-A` and `C-B`. Continue toward a method claim only if:

- C improves mAP@R over both A and B by at least `0.003`;
- each contrast is positive on paired per-query mean AP@R;
- C's Recall@1 and Recall@10 decline by no more than `0.001` from A, B, and the
  epoch-4 parent;
- no run, evidence row, update counter, or authority is missing.

Report a 10,000-resample identity-clustered paired bootstrap interval using
statistics seed `20260904`. Also report win/tie/loss and discordant recall
counts. These intervals are conditional on one shared parent and one holdout;
they are not independent end-to-end replication.

If C fails either contrast, close the SmoothAP causal claim and do not launch a
new-objective campaign. If C passes, add the preregistered second phase:
soft-triplet ranking and full-768 SmoothAP geometry, followed by independent
parents and another dataset.

## Resource and leakage boundary

Run one DGX process at a time with a two-hour wall cap. Stop below 8 GiB host
memory available, above 256 MiB swap growth, after twelve consecutive five
second samples with memory PSI full avg10 at least 0.79, on OOM/nonfinite
values, or after ten minutes without epoch progress.

The development holdout is burned and may only decide whether to continue the
research program. It must not tune temperature, coefficients, finish length,
arm ordering, or release-model selection. The historical standard test is not
an untouched holdout and cannot select training choices. All outputs remain
`claim_eligible=false` until independent-parent and independent-dataset
replication.

## Performance follow-up

Only after the causal gate passes, optimize runtime in this order: remove unused
EMA work when the endpoint is fixed to raw weights; precompute schedules and
label masks; implement positive-slot-only SmoothAP comparisons; then chunk
distance work. Every claimed exact optimization must match loss, full gradients,
optimizer/scaler/scheduler/EMA state, RNG state, and model tensors after one,
ten, and 644 replayed steps. A divergent implementation is a numerical variant
and requires new quality evidence.
