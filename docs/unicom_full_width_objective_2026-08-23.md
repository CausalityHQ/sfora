# UniCOM full-width ArcFace objective

## Status

This is a prospective training comparison after the registered classifier-
imprinting result and the independent spherical-proxy screen.  The promoted
baseline is the imprinted-raw UniCOM ViT-L/14-336 recipe.  On the five
prospective official In-Shop seeds it improved query-weighted mAP@R by
`0.026484` with paired 95% interval `[0.024766, 0.028202]`, improved Recall@1
by `0.014039`, and reached the matched random baseline's epoch-16 quality by
epoch 8 on every seed.  Its mean Recall@1 is 95.15%, below the published 96.7%
UniCOM L/14-336 anchor.

The spherical-proxy screen is permanently `CLOSE_DIRECTION`; none of its
thresholds or candidate settings change here.  An uncommitted prefix-alignment
draft was rejected before implementation or experiment because prefix-512 is
the registered legacy view, its proposed headroom value was already visible,
and fixing one mask row would affect only one contiguous class shard.  This
document replaces that draft rather than amending it after a result.

## Evidence and prior art

The registered primary readout L2-normalizes all 768 coordinates and ranks by
exact Euclidean distance.  The legacy secondary view normalizes all 768,
retains prefix `[0,512)`, and does not renormalize.  In the already-published
five-seed imprinted epoch-16 panel, full-768 primary mAP@R exceeds legacy
prefix mAP@R on every seed by `(0.002762, 0.002223, 0.003529, 0.002586,
0.002846)`, mean `0.002789`.  These are retrospective official-split facts,
not a new selection gate.

Training still samples eight independent 512-of-768 coordinate sets on every
optimizer step.  Each set supplies logits for one contiguous class shard, and
the shard logits participate in one global ArcFace softmax.  The random masks
make low-dimensional subspaces useful, as intended by UniCOM, but no training
step uses the registered full-768 primary geometry.

The candidate is deliberately the simplest established control, not a new
primitive.  The repository's 2026-08-12 defect audit already identified direct
full-768 training as the dominant control; Matryoshka-style or stochastic
multi-width objectives are deferred until this control is measured.  The
scientific question is whether UniCOM's feature sampling improves or harms the
quality-efficiency Pareto point under our registered full-width deployment.

## Candidate and control

Both arms use class-mean imprinting and the same pretrained checkpoint,
partition, identity holdout, FP32 model, augmentation, padded batch order,
optimizer, OneCycle schedule, ArcFace margin and scale, 16 epochs, checkpoint
epochs `(4,8,12,16)`, BatchNorm treatment, and full-width evaluation view.

- `sampled_512` control: `objective=official-eight-mask`,
  `selected_features=512`, `evaluation_features=768`.
- `full_768` candidate: `objective=official-eight-mask`,
  `selected_features=768`, `evaluation_features=768`.

With `selected_features=768`, every argsort mask contains every coordinate.
Different coordinate orders are algebraically irrelevant because embeddings
and their corresponding class weights use the same permutation.  All eight
class shards therefore compute full-width normalized cosines while retaining
the official implementation's floating-point reduction order and exact eight
random draws per step.  The two arms consequently consume the same number and
shape of mask-generator draws; tests bind the post-draw generator state.  No
model parameter, checkpoint shape, descriptor, inference operation, or
deployment byte changes.

The candidate adds 256 coordinates to every classifier dot product.  No
analytical FLOP estimate is used as evidence because the prior 0.006% estimate
was per shard and did not bound the single-device eight-shard implementation.
Exact A-B-B-A step profiling, wall time, peak allocated and reserved GPU
memory, and checkpoint bytes are measured.

The existing trainer couples `selected_features` to both the loss and its
in-training holdout readout.  That behavior is forbidden here: it would score
the control on legacy prefix-512 and the candidate on primary full-768, a
retrospective difference of 0.002789 that is 93% of the promotion floor.  The
implementation adds an independent `evaluation_features` binding and fixes it
to 768 in both arms.  Tests must prove that changing only loss width leaves
evaluation coordinates and per-query ordering byte-identical.  Legacy
prefix-512 is recomputed only by the separate paired evaluator and is never
read from either trainer history.

## Stage 1: paired selection screen

Run two fresh seed-0 jobs, control then candidate, in the same reviewed source,
runtime, and idle GPU state.  The candidate run is forbidden unless the
control completes, all four checkpoints strict-load, and its epoch-16 hardened
full-768 holdout metrics are finite.  Both jobs publish disjoint initialization
receipts, histories, and checkpoints.  GPU arithmetic is statistically rather
than bitwise reproducible, so no claim depends on byte-identical floating-point
results.

A separate paired evaluator reloads all four checkpoint epochs from both arms,
authenticates their source,
configuration, checkpoint, history, partition, label order, and query/gallery
row order, and emits per-query AP@R and top-1 evidence under both registered
views.  The primary view is full-768 unit-normalized Euclidean retrieval.  The
legacy prefix view is diagnostic and cannot select the candidate.

`full_768` advances only when all of the following hold at epoch 16:

1. primary mAP@R delta is at least `0.003`;
2. primary top-1 loses at most one of the exact paired holdout queries;
3. candidate reaches the control epoch-16 primary mAP@R no later than control
   on the frozen `(4,8,12,16)` grid;
4. elapsed training time, peak allocated GPU memory, and checkpoint bytes are
   each no more than 2% above control.

The paired 10,000-replicate PCG64 query bootstrap interval is reported but is
non-gating at seed 0 because the 797-query selection holdout is not powered to
put the lower bound of a 0.003 effect above zero.  A failure closes full-width
training without trying mixed-width losses, cadence schedules, learning-rate
changes, or a smaller effect threshold on the same holdout.

## Stage 2: five-seed confirmation

Only a promoted seed-0 candidate advances.  Seeds 2 through 6 each receive a
fresh `sampled_512` control and a fresh `full_768` candidate under the exact
same reviewed source, runtime, initial checkpoint, data bytes, partition, and
seed.  Retained historical checkpoints are context only and cannot enter a
confirmation delta.  This avoids silently pairing new candidates with controls
from an older trainer build or a different nondeterministic GPU trajectory.
Arm order is prospectively balanced to reduce thermal/order bias:
`sampled_512` first for seeds `(2,4,6)` and `full_768` first for `(3,5)`.
Each pair completes and the GPU returns idle before the next seed starts; no
training jobs overlap.

Confirmation requires:

- mean primary mAP@R delta at least `0.003`;
- paired Student-t 95% lower bound for the five primary mAP@R deltas above
  zero;
- at least four of five primary mAP@R deltas positive;
- aggregate top-1 losses across all five paired holdouts no greater than five
  queries and no individual seed losing more than two queries;
- candidate time to the matched control epoch-16 mAP@R no later on at least
  four of five seeds; and
- mean training time, mean peak memory, checkpoint bytes, measured inference
  latency, and deployment storage all within 2% of control, except that a
  statistically supported fixed-epoch quality gain may trade up to 2% training
  time while inference and deployment storage remain unchanged.

These gates provide five paired seeds and a practical-effect floor; no separate
sign-test claim is made.  The paired query bootstrap is reported per seed and
as a descriptively pooled query analysis, but the seed-level paired-t interval
is the inferential gate because queries within one training seed are not
independent training replicates.  Seed 1 remains sensitivity-only because its
retained baseline used an earlier trainer source.  Its historical value may be
reported as context but cannot promote or reject the candidate.

The official query/gallery split remains untouched until confirmation passes.
A later one-shot official readout requires its own prospective configuration.
No global state-of-the-art claim follows from the holdout comparison alone.

## Result and safety contract

Every evaluator artifact uses strict duplicate-key-rejecting finite JSON,
binds the exact source commit and file hashes, records the literal command and
environment, and publishes with same-directory no-clobber atomic semantics.
The producer strict-reloads and recursively validates persisted bytes.  A
structural failure publishes no scientific result; a finite failed candidate
is durable and cannot be rerun under changed gates.

The reviewed run configuration freezes both training and evaluation widths,
arm order, seeds, all paths and hashes, exact epoch set, thresholds, and one
attempt per arm.  The trainer records both widths independently in every
checkpoint and history row.  The paired evaluator rejects mixed commits,
missing epochs, unexpected trainer metrics, a non-768 primary view, duplicate
or reordered query/gallery IDs, and any recomputed scalar or decision that
differs from persisted evidence.

The full-width candidate needs no custom kernel.  Kernel work remains
ineligible unless an A-B-B-A profiler attributes at least 10% of step time to a
fusible non-backbone operation and an exact-output prototype improves measured
time-to-quality.  Hyperbolic or other non-Euclidean geometry remains outside
scope because no Euclidean-inexpressible residual has been demonstrated.
