# PCA-Anchored Relational Transfer Design

## Purpose

Test whether relational compression contains a transferable correction that can
be retained without sacrificing PCA's class-disjoint behavior. This is a
generic class-labelled retrieval experiment, not a CUB repair. The deployed
item remains 128 signed int4 coordinates plus one little-endian f16 inverse
norm: exactly 66 bytes/item.

The authenticated CUB receipt at
`docs/evidence/relational_linear_compaction/cub/cub-relational-int4-evaluation-v1.json`
shows relational float mAP@R about 0.012 below PCA float, with about 0.006 more
loss from int4 and less than 0.0005 spread across three seeds. This establishes
stable representation-family drift on CUB, not a causal explanation. The
corresponding 128-dimensional SOP premise is not authority until its controls
and model bytes are retained under the same training recipe.

## Comparable SOP authority

Before cross-dataset interpolation, retain a SOP receipt containing PCA float
and int4, rotated-PCA int4, relational float and int4, and seed-17 model bytes.
CUB and SOP relational maps use the same optimizer recipe and exactly 1,000
updates. The retained CUB seed-17 model is loaded from bytes, not retrained.

## Fixed-path exploratory screen

Development uses fixed 70/30 class-disjoint splits of each training partition.
Test outcomes remain unopened during selection. For each dataset:

1. Reproduce train-only PCA `W0` and the comparable relational map `W1`.
2. Express each endpoint as an affine map `W x + b`. For centered PCA,
   `b0=-W0 mean`; an uncentered relational endpoint has `b1=0`. Form
   row-normalized training outputs `A=normalize(X W1.T+b1)` and
   `B=normalize(X W0.T+b0)`. For `A.T B=L S R.T`, use deterministic full SVD
   `Q=L R.T`, fold `Q.T` into both `W1` and `b1`, and reject nonfinite or
   rank-deficient authority. Multiply the aligned `W1` and `b1` by the same
   scalar needed to match the PCA pre-normalization training-output RMS. At
   coefficient `a`, use exactly `W(a)=(1-a)W0+aW1` and
   `b(a)=(1-a)b0+ab1`, followed by row normalization.
3. Parameterize the path by mean training-output angular displacement from PCA,
   not raw map coefficient. Evaluate fixed fractions `(0,.05,.10,.20,.40)` of
   the aligned endpoint displacement. Evaluate displacement in float64 on the
   fixed grid `k/256`, choose the first adjacent pair that brackets the target,
   then refine only that bracket for 48 bisection iterations. Recheck
   `d(lower) <= target <= d(upper)` at every iteration and fail closed if the
   target is unreachable or the bracket invariant breaks. Choose the lower
   coefficient on an exact tie and emit coefficient plus achieved displacement.
   This defines one branch even when the global displacement curve is not
   monotone.
4. Apply the same seed-17 output rotation and canonical int4 packing to every
   path point. Emit float and exact-int4 retrieval plus unaligned and aligned
   relational endpoint controls. The aligned endpoint is a new arm, not the
   historical unrotated CUB endpoint.

Selection uses query-weighted mAP@R with self-exclusion and a fixed gallery.
Uncertainty is paired class-cluster resampling of fixed per-query outcomes:
10,000 replicates, seed 17, and quantile `0.05/8` as a one-sided 95% familywise
lower bound for four nonzero points across two datasets. A class without a
relevant gallery item is invalid authority; score ties resolve by source
ordinal.

The family count is eight because the inferential claims are the four nonzero
MAP deltas on each of two datasets. R@1 and float-path bounds are preregistered
non-inferiority guardrails, not additional positive claims; their one-sided
lower bounds can only veto a candidate.

One identical nonzero displacement fraction survives only if it:

- improves SOP int4 mAP@R over rotated PCA128 int4 by at least 0.003 with lower
  bound above zero; and
- has CUB int4 point difference at least -0.001, lower bound at least -0.005,
  and R@1 lower bound at least -0.005.

The same gates are evaluated separately for float outputs. Float-pass/int4-fail
indicates a codec-coordinate problem; float-fail/int4-pass indicates only a
quantization-coordinate effect; both-pass supports this fixed path; both-fail
rejects only this path. Every point recomputes
`delta_int4 = delta_float + ((candidate_int4-candidate_float) -
(pca_int4-pca_float))`.

All points, endpoints, coefficients, displacements, and per-query outcomes are
emitted. Burned CUB/SOP test results may falsify whether the inner-split choice
transfers, but cannot confirm or select a publishable method. The canonical
claim-ineligible receipt binds input hashes, ordered IDs and splits, complete
recipes, serialized models/alignment/rotation, executed source/environment,
replay tolerances, raw outcomes, and gate calculations. Failure is retained.

## Separate rank-8 hypothesis

The fixed path is not a mathematical ceiling for a rank-8 residual: its
direction may have rank 128 and neither feasible set contains the other. Before
rank-8 work, report the singular spectrum and train-output rank-8 approximation
error of `W1_aligned-W0`. Stopping after a negative screen is a resource choice,
not rejection of all constrained corrections.

The separate trainable hypothesis is `W=W0+U V.T` with rank 8 and an angular
trust-region penalty. A frozen inner class split chooses from a fully specified
three-value penalty ladder using worst-group relational loss subject to PCA
angular-displacement and source-neighborhood bounds. PCA and preprocessing fit
only inner-training rows during selection. Quantization uses a straight-through
canonical int4 operator; validation always uses the exact codec. The final map
remains one 128-by-input affine projection.

## Method-specific confirmation

Confirmation must be unopened for this method and source model. Cars196 is
preferred only after an authenticated UNICOM exporter, exposure ledger, split
identity, duplicate audit, and source/teacher provenance exist; it is not
described as globally unseen because its test split has historical uses here.
CUB and SOP cannot choose confirmation hyperparameters or gates.

Confirmation requires the same 66-byte representation, mAP@R gain at least
0.003 with familywise lower bound above zero, R@1 lower bound at least -0.005,
and the resident performance evidence below. The initial generality claim is
limited to class-labelled retrieval with legitimate train-only groups.

## Alternatives and semantics

If the fixed path fails, the next representation lane is nuisance-covariance
normalization from legitimate paired views. Retrieval-aware discrete error
shaping is considered only when float transfer succeeds but int4 fails.

Class-name semantics are excluded from the primary method. Test names leak
evaluation structure. A later training-only study must compare semantic names
against ordinary IDs and a shuffled name-to-class control without changing the
confirmation protocol.

## Performance boundary

Persistent storage is 66 bytes/item. The CPU int8 execution layout is 130
bytes/item excluding the packed source (196 if both coexist); resident float32
is 516 bytes/item. Integer accumulation must reproduce reference scores and
stable ranking exactly. A benchmark receipt binds 10,000 raw timings,
construction and peak memory, hardware, threads, source hashes, and backend.
Integer performance evidence requires the integer backend and fails closed;
fallback timings are separately labelled. The observed benefit is specifically
against per-call decode, while resident float BLAS is the faster, wider control.
A direct-nibble SIMD kernel is future optimization, not a quality dependency.
