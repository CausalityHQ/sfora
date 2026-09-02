# ASG-CV Forced-Gradient Distillation Design

## Goal

Test whether the train-only Qwen forced-verdict gradient can be amortized by the
existing rank-16 patch-gradient predictor and transfer to disjoint Cars196
training classes without sampled rollouts at student-training time.

## Frozen protocol

- Use only `predictor_train` and `e0_validation` from the authenticated
  `sfora-cars-train-p32-manifest-v1`; official test access remains false.
- Build deterministic image-disjoint balanced schedules of 128
  predictor-training pairs and 32 validation pairs.  Schedule seeds bind source
  commit, input authority digest,
  and role (`train` or `validation`).
- Qwen branch scores always use the fixed observable order `SAME`, `DIFFERENT`.
  The captured teacher target is the SAME-first gradient multiplied by the
  registered relation sign.  This is exact because the binary collapsed-GRPO
  coefficient is symmetric under exchanging the two branches.
- Persist each pair as canonical receipt plus non-pickle float32 NPY patch-token
  and relation-correct-gradient arrays.  Shape is exactly `[2,256,4096]`
  (four authenticated 64-patch Qwen boundaries per image at the registered
  Qwen text hidden width) and
  every receipt binds both array digests.
- Train `AsgcvPatchGradientPredictor(rank=16)` for exactly 20 epochs in ordinal
  order with batch size one, AdamW learning rate `1e-3`, weight decay `1e-4`,
  and the existing normalized dense-plus-SRHT loss with a 256-dimensional
  source-bound SRHT.  No validation-driven hyperparameter choice is allowed.
- Evaluate once on all 32 class-disjoint validation pairs.  The primary gate is
  median per-pair dense cosine at least 0.50; secondary gates are positive
  cosine on at least 75% of pairs and finite nonzero predictions on every pair.
  A finite zero prediction is recorded as cosine zero and failed liveness rather
  than aborting the result.  The result is claim-ineligible and cannot authorize
  an official test read.

## Boundaries

`src/sfora/asgcv_forced_distill.py` owns schedules, capture receipts, training
authority, and recomputed metrics.  `scripts/run_asgcv_forced_distill.py` owns
local authenticated I/O, Qwen capture, resumability, predictor fitting, and
canonical output.  It has no network or official-test flag.  Unit tests use a
small fake adapter and synthetic arrays; the scientific DGX run is separately
authenticated after code verification and commit.

## Decision

A pass means the semantic field is compressible enough to integrate as a cheap
training-time control variate.  A fail rejects this rank-16 student and stops
retrieval integration; it does not negate the positive teacher diagnostic.
