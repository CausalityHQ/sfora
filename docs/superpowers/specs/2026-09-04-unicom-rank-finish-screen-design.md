# UniCOM Rank-Finish Screen Design

## Status and purpose

This is a claim-ineligible, seed-0 falsifier for a late rank-sensitive training
phase.  It does not reopen the rejected FEPF initializer family.  The input is
the authenticated imprinted epoch-4 control checkpoint from the completed FEPF
campaign.  The only question is whether directly optimizing the deployment
ranking geometry can improve mAP@R without materially reducing recall.

The motivating evidence is unusually specific.  The control holdout has
mAP@R 0.8975116742477199, R@1 0.986198243412798, and R@10
0.9974905897114178.  Only 11 of 797 queries have an incorrect top-1 result;
87.63564145920169% of the total AP@R deficit belongs to queries whose top-1 is
already correct.  FEPF mean initialization reduced mAP@R by
0.0037290107925470606 while leaving R@1 and R@10 unchanged.  The open failure
mode is therefore deep positive ordering, not class discovery or initialization.

## Frozen method

Resume the exact imprinted epoch-4 model and optimizer state.  Train the
backbone for four more epochs using deterministic identity-balanced batches of
32 identities and four images per identity.  The classifier is retained only
to restore the optimizer state and receives no gradient during the finish.

For every image, normalize the complete 768-D descriptor and retain the first
512 coordinates, exactly matching the deployed evaluation geometry.  For each
anchor and each of its three positive batch peers, approximate its rank among
all non-self batch examples with `sigmoid((d_positive - d_candidate) / 0.01)`.
Approximate its positive rank with the same expression restricted to positive
peers.  The per-anchor differentiable AP is the mean positive-rank/rank ratio;
the loss is one minus mean AP across anchors.  Accumulation order is fixed and
all distances, losses, and gradients must be finite.

The identity-balanced schedule is deterministic from the registered training
seed and epoch.  It exposes exactly the same number of complete batches as the
original epoch schedule, samples identities without replacement within each
batch, and cycles a per-identity deterministically shuffled image order only
after exhausting its physical images. Repeated indices for sparse identities
therefore receive independent augmentations instead of excluding the identity.

## Evidence boundary and stops

The screen consumes only the registered optimization split during training.
The identity-disjoint holdout is read after epochs 6 and 8.  No standard-test
descriptor, label, or result is an input.  The previously executed query
expansion diagnostic is not an input and is scientifically closed because it
traded recall for mAP.

The epoch-4 control metrics are immutable comparison values.  Stop after epoch
6 if delta mAP@R is at most -0.003.  At epoch 8:

- reject if delta mAP@R is below +0.003;
- record an exploratory improvement if delta mAP@R is at least +0.003 and the
  R@1 delta is at least -0.001;
- promote to a separately preregistered multi-seed confirmation only if delta
  mAP@R is at least +0.010, R@1 delta is at least -0.001, and R@10 delta is at
  least -0.001.

Every terminal result is canonical JSON with exact source, checkpoint, run
receipt, partition, method, schedule, metric, elapsed-time, peak-allocation,
and stop-reason bindings.  `claim_eligible` is always false.  The screen never
overwrites or extends the FEPF campaign artifact tree.

## Resource envelope

Use one DGX GPU process, deterministic execution, batch size 128, and the
current non-compiled/non-fused runtime required by the resumed optimizer state.
The hard wall limit is two hours.  Stop on CUDA OOM, nonfinite computation,
memory PSI full avg10 at least 0.79 immediately or at least 0.50 for three
consecutive samples, or a ten-minute progress gap.  Expected runtime is
60--90 minutes.
