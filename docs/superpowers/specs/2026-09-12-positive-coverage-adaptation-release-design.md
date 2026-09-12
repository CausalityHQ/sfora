# Positive-Coverage Adaptation Release Design

## Goal

Release the successful positive-coverage retrieval objective as a dataset-agnostic
Sfora training primitive. The library accepts embeddings and integer labels; dataset
loading, split construction, metric evaluation, and scientific receipts remain outside
the core.

## Frozen behavior

- Normalize training dose by class exposure, not a fixed update count. The default
  reference is 2,000 updates, 64 classes per update, and 9,054 eligible classes.
- A class is eligible when it has at least `rows_per_class` rows. Ineligible rows remain
  valid negative-bank rows but are never sampled as anchors.
- Each update samples distinct eligible classes and distinct rows within each selected
  class using one PCG64 stream seeded by the caller.
- Hard negatives are the exact top-k different-label rows. Boundary ties are resolved
  by lower bank-row ordinal. Returned row indexes are in ascending row-ordinal order;
  the loss is permutation-invariant over negatives. Ambient automatic mixed precision
  cannot change this float32 membership authority.
- The positive-coverage loss remains the existing all-positive average of independent
  hard-negative competitions, with the pooled-positive objective retained as a matched
  control.
- All APIs reject bool-as-int, nonfinite embeddings, invalid labels, insufficient
  eligible classes, impossible widths, and non-unit input codes.

## Interfaces

`exposure_normalized_update_count(...) -> int` computes

`ceil(reference_updates * reference_classes_per_update * eligible_class_count /
      (reference_class_count * classes_per_update))`.

`class_balanced_anchor_schedule(...) -> ClassBalancedAnchorSchedule` returns an
immutable `[updates, classes_per_update * rows_per_class]` int64 matrix, the eligible
class count, and a SHA-256 authority digest.

`stable_different_class_topk(...) -> torch.Tensor` returns exact deterministic negative
row membership for every anchor without CPU transfer or a full sort.

## Evidence and claim boundary

The implementation is generic library code. Current SOP and In-Shop results are
cross-dataset scientific evidence, not defaults silently fitted by dataset name. CUB is
an explicit negative/inconclusive replication for the coverage-over-pooled increment.
Release evidence must retain seeds, ordered input hashes, schedule hashes, model-state
hashes, and both control and treatment outcomes. CUB and In-Shop checkpoints embed both
the trained 768-to-128 base head and the 128-to-128 adaptation, so reported embeddings
can be reconstructed without retraining. Checkpoints are atomically published without
replacement before the canonical receipt; concurrent writers cannot delete or replace
another writer's files. Official test data is evaluated only after the recipe and seed
panel are frozen.

## Current evidence

Across seeds 0 through 4, SOP coverage averages 0.579613 packed mAP@R and 0.825964
R@1 versus 0.575642/0.823685 for the pooled control; every clustered bootstrap lower
bound is positive. In-Shop averages 0.591990/0.834069 versus 0.577528/0.827078, again
with five positive lower bounds. Exposure-normalized CUB is 0.644469/0.891458 versus
0.644160/0.891290 and has a slightly negative lower bound, so it is retained as a
non-pass rather than generalized away. These are compact 128-dimensional packed-code
adaptation results, not yet an official-test or architecture-independent SOTA claim.
