# Cars Twin Reachability Design

Date: 2026-09-01
Status: **PREREGISTERED REPRESENTATION FALSIFIER**

## Question

The frozen SigLIP-so400m Cars burned-band manifest contains `103` strict
retrieval errors, of which `63` are Dodge Caliber Wagon 2012 versus 2007. This
design asks one narrow question before another expensive training run:

> Does a supplied fixed descriptor plane contain either a nearest-centroid cue
> or a cue recoverable by one fixed regularized linear discriminant for labels
> `82` and `83`?

A negative result falsifies these two preregistered readouts, not every possible
nonlinear readout or the descriptor plane itself. It does **not** prove that the
images are mislabeled or visually indeterminate. Raw-image reachability belongs
to the separately sealed Qwen evidence-panel gate.

## Inputs and authority

The pure operator consumes:

- a nonempty concrete plane name;
- one rank-two finite `float32` or `float64` descriptor matrix;
- one rank-one concrete integer label vector of equal length;
- every row in the already-authenticated burned diagnostic band whose label is
  `82` or `83`, in ascending registered example order; outcome-selected error
  subsets are forbidden;
- at least `20` rows per label, with exact per-class counts and unique
  registered example IDs recorded by the caller.

The caller owns the plane, source revision/tree digest, dataset revision and
manifest digest, model name/revision, frozen-model or checkpoint producer
identity, ordered Caliber example digest, label-vector digest, and exact
descriptor digest. The operator never opens images, checkpoints, datasets, or
network resources. The canonical artifact binds its numeric evidence to those
caller-authenticated identities, requires an independently supplied expected
plane during validation, and remains `claim_eligible=false`.

## Deterministic leave-one-out score

For every selected row `i`, let `x_i` be its finite descriptor, converted to
`float64` and L2-normalized. Compute the mean of every other normalized row in
its own class and the mean of all normalized rows in the other class. Normalize
both means. The authoritative score is

```text
s_i = dot(x_i, centroid_83_without_i_if_needed)
    - dot(x_i, centroid_82_without_i_if_needed)
```

where only the query's own class excludes `i`. Arrays enter the pinned NumPy
`float64` reductions in ascending source-row order. Label `83` is the positive class. AUC is the exact
Mann-Whitney statistic with half credit for tied scores; no library classifier,
random split, or learned hyperparameter is authoritative.

This score is deliberately capacity-limited. It answers whether the existing
geometry exposes a stable global year cue without fitting a new nonlinear
readout on the burned band.

## Fixed leave-one-out shrinkage LDA

Nearest-centroid geometry can hide a real linear cue behind high-variance
nuisance directions. Therefore every row also receives a leave-one-out
regularized LDA score. For each held-out row, compute the two class means and
pooled within-class covariance `S` from the remaining normalized rows. Freeze

```text
lambda = max(0.1 * trace(S) / dimension, 1e-8)
w = inverse(S + lambda * I) * (mean_83 - mean_82)
b = 0.5 * dot(mean_83 + mean_82, w)
lda_i = dot(x_i, w) - b
```

The implementation evaluates the mathematically identical dual Woodbury solve
so dimensions larger than the sample count remain bounded. Label `83` fixes the
positive orientation before evaluation; the gate is intentionally one-sided.
Flipping a below-chance result after observing its AUC would be evaluation
leakage. Leave-one-out discriminants are negatively biased under the null at
small sample sizes, so below-chance AUC is non-evidential and has no campaign
branch. The fixed LDA AUC is computed with the same Mann-Whitney authority.

## Conditional high-evidence mode

Some viewpoints may expose a model-year cue while others cannot. Fit one- and
two-component Gaussian models to `abs(s_i)` only. The two-component fit uses a
fixed deterministic one-dimensional EM implementation:

- lower/upper empirical quartiles initialize the two means;
- weights start at `0.5`;
- both variances start at the population variance, floored at `1e-8`;
- exactly `128` EM iterations execute with arrays in ascending source-row order;
- component order is canonicalized by `(mean, variance, weight)` after every
  iteration;
- log likelihood and BIC are recomputed in `float64`.

The high-evidence component is the component with the larger mean absolute
margin. Membership uses posterior `> 0.5`, ties assigned to the lower
component. Record:

- `bic_improvement = BIC_one - BIC_two`;
- high-evidence fraction;
- AUC of the original signed scores within high-evidence members, or `0.5` if
  both labels are not represented.

The Gaussian BIC and retained fraction are selection heuristics, not independent
evidence for genuine bimodality: `abs(score)` is not generally Gaussian. The
held-out signed-score AUC is the evidential condition in this branch.

## Frozen decision rule

The plane is `cue_present=true` when either the fixed LDA AUC is at least
`0.80`, or the centroid readout satisfies either:

1. full leave-one-out AUC is at least `0.80`; or
2. BIC improvement is at least `10.0`, the high-evidence fraction is at least
   `0.25`, and high-evidence AUC is at least `0.80`.

No threshold is tuned from the observed Cars result. Report all statistics even
when the gate fails. Nonfinite inputs/results, zero-norm rows or centroids,
cardinality/type drift, fewer than `20` rows in either class, or inconsistent
labels fail closed. The frozen `0.80` gate is registered only for this
minimum-count regime and is not a general small-sample threshold.

## Campaign interpretation

Run the same operator on:

1. frozen SigLIP-so400m pooled descriptors;
2. trained seed-17 raw pooled descriptors; and
3. trained seed-17 projected descriptors.

Interpretation is ordered:

- frozen LDA pass but centroid fail: a linear cue exists but is hidden from the
  current nearest-neighbour geometry;
- frozen centroid pass: the cue is already nearest-centroid exposed and the
  current retrieval objective or aggregation fails to exploit it;
- frozen fail plus trained pass: ordinary metric learning creates the cue but
  does not make it consistently nearest-neighbour useful;
- all three fail: these descriptor planes fail both registered readouts; this
  rejects another unmotivated loss/head sweep but not all possible nonlinear
  representation work;
- no descriptor outcome adjudicates label correctness or raw-pixel
  reachability.

Only a descriptor pass permits the cheap frozen-feature PRISM realization. The
raw-image Qwen panel must separately show predictive channel information before
any backbone-scale PRISM training.

## Verification

Synthetic tests cover global separation, a high-variance nuisance case where
centroid geometry fails but shrinkage LDA succeeds, realistically jittered
conditional separation, score ties, row permutations, scale invariance, zero
norms, nonfinite values, concrete-type drift, insufficient class counts,
deterministic byte equality, and mutations of every recorded statistic, plane
identity, and gate. Deterministic bytes are scoped to the authenticated software/platform
because NumPy linear algebra owns the solve. GPU or custom kernels are
considered only after profiling the later panel objective.
