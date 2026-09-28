# Saved-state attribution implementation

1. Reuse qualified loading, source/receipt authority, unaugmented image decoding,
   atomic feature export, packed scorer and bootstrap. One diagnostic script
   authenticates original/final heads/vision and bothsources during CPU preflight;
   emits four1280-D state matrices (raw1024+bothheads128), no gradients oroptimizer.
2. Add one toy arithmetic/parity check and run CPU preflight/startup/code checks;
   reconcile focused review of this fixed inference-only diagnostic beforeGPU.
   Freezecommit/push one600s8Gshared-lock launch, inspectidle first.
3. Observe samejob; verify original process exit, unchangedparameters, final/head
   golden-vector/score parity and decomposition, archive actualcost/uncertainty.
   Publishone honest layer decision; retain source recipe STOP and wholegoal.
