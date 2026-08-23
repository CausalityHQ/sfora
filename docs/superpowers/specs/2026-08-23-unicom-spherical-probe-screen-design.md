# UniCOM spherical-probe causal screen

## Goal and claim boundary

Test whether a conservatively optimized, fixed-norm ArcFace classifier head changes
the initial learning signal enough to justify a fine-tuning experiment against the
current class-mean-imprinted head. The screen uses only official In-Shop `train`
records and frozen released-UniCOM features.

The classifier is discarded at retrieval time. Consequently, frozen-head loss or
accuracy cannot establish an open-set retrieval improvement and cannot be charged
toward the published 96.7% Recall@1 anchor. `PROMOTE` authorizes only the prospective
2x2 fine-tuning screen below. Only that screen and a later full-budget reproduction
can support a retrieval-quality claim.

The retained official result is the starting evidence: class-mean imprinting beats a
matched random head by +0.026484 mAP@R and +0.014039 Recall@1 over five seeds, while
the fair legacy-512 mean is 95.04% Recall@1, 1.66 points below the published anchor.

## Records and deterministic split

Apply `identity_holdout(fraction=0.2, seed=0)` and use only its 3,200 optimization
identities. Preserve its class vocabulary and `[3200, 768]` head shape.

Group records by class in label-map order and sort each group by exact path string.
Use NumPy PCG64 stream 23,000 to draw one validation index for every class with at
least two images. If the class has multiple acquisition-series tokens, exclude every
other row from the selected validation series from fitting; this makes that validation
row series-disjoint while keeping at least one other series in fitting. If the class
has only one series, exclude only the selected validation row. A singleton stays in
fitting, has no validation row, and retains its classifier row. The frozen partition
counts are 20,650 optimization images, 14,330 fitting images, 3,188 validation rows,
12 singleton classes, and 3,132 additional same-series exclusions.

Parse the In-Shop filename's acquisition-series token (the token before the first
underscore). For every validation row publish whether the same class and series is
represented in fitting. The frozen split has 2,162 represented and 1,026 unrepresented
validation rows. Any count drift is structural. Metrics and decisions use both strata.

Encode every fitting and validation image exactly once with the authenticated
released model's deterministic evaluation transform. Cache only finite contiguous
FP32 features and contiguous int64 labels. Persist no image or embedding. Restore
Python, NumPy, Torch CPU, and all Torch CUDA RNG states on success or failure.

## Heads and fitting

`class_mean` normalizes every fitting feature using the same divide-by-vector-norm
operations as the current trainer, accumulates means in FP64, normalizes each mean,
casts once to FP32, and multiplies by `0.01 * sqrt(768)`.

Fit three preregistered heads, fit seeds 0, 1, and 2. Every `spherical_probe_seed_N`
starts byte-identically from `class_mean` and runs exactly 512 AdamW steps with
learning rate `1e-4`, betas `(0.9, 0.999)`, epsilon `1e-8`, and zero weight decay.
Batches contain 128 cached fitting rows. Each step uses eight independent
512-of-768 masks and the existing sharded masked ArcFace objective (margin 0.25,
scale 32). Batch stream 23,001 and mask stream 23,002 are namespaced by fit seed
through `experiment_stream_seed`. After each update, normalize every head row and
rescale it to `0.01 * sqrt(768)`.

Projection is a structural norm match for the later 2x2 and keeps the effective
directional step scale stable; ArcFace itself is invariant to positive row scaling.
Row norms are validated structurally, not counted as scientific evidence. Publish
each fitted head's rowwise cosine-to-class-mean minimum, fifth percentile, median,
and mean. Require minimum at least 0.80 and mean at least 0.95 as a prospective
warm-start tripwire.

Measure fitting start/end loss on a separately drawn batch and mask set from stream
23,004, never on an optimization step's batch or masks. This is an implementation
diagnostic, not a representative full-fitting-set estimate.

## Validation and causal gradient diagnostic

For each head, evaluate the disjoint validation rows with the same 64 mask sets from
namespaced stream 23,003. For every mask set:

- compute mean ArcFace cross-entropy with margin 0.25;
- compute top-1 accuracy from separate logits with margin 0.0;
- retain the paired class-mean-minus-probe mean-loss delta.

Accumulate in FP64/Python integers. Report overall results and separate acquisition-
series-represented/unrepresented results. For each probe report the mean paired loss
delta, its two-sided paired 95% Student-t lower bound (df=63), and the number of the
64 deltas strictly above zero.

For every fit seed, draw one separate 128-row fitting batch and one eight-mask set
from stream 23,005. Clone the cached features with gradients enabled, compute the
summed registered loss for each head, and obtain `dL/dz`; never update the backbone.
Report both gradient Frobenius norms, relative L2 difference
`||g_probe-g_mean||/||g_mean||`, and the minimum, fifth percentile, median, and mean
of finite per-sample gradient cosine. Zero per-sample gradients must match in both
arms; an unmatched zero is structural failure. The gate requires relative difference
at least 0.05 and median cosine at most 0.995 for every fit seed.

## Decision

`PROMOTE` requires, for each of fit seeds 0, 1, and 2:

- the independent diagnostic fitting loss strictly decreases;
- cosine-to-class-mean minimum is at least 0.80 and mean at least 0.95;
- overall validation paired loss delta is positive, its paired 95% lower bound is
  above zero, and at least 48 of 64 mask deltas are positive;
- margin-free validation accuracy is noninferior to `class_mean`;
- mean validation loss delta is positive in both acquisition-series strata;
- gradient relative L2 difference is at least 0.05 and median per-sample gradient
  cosine is at most 0.995.

All three seeds must pass symmetrically. Otherwise the valid decision is
`CLOSE_DIRECTION`. Structural/schema/hash/nonfinite failures publish no scientific
result. The former 1% point estimate and row-norm predicate are removed: the former
had no noise calibration and the latter was a tautological implementation check.

## Implementation and publication

Pure split, fitting, evaluation, uncertainty, gradient, and decision logic lives in
`src/sfora/unicom_probe.py`. The CLI is
`scripts/screen_unicom_spherical_probe.py`. Reuse
`sharded_mask_arcface_logits`, `sharded_mask_arcface_loss`,
`padded_epoch_indices`, `sample_shard_masks`, and `experiment_stream_seed`; do not
duplicate the registered objective or samplers.

The CLI authenticates the exact UniCOM revision/checkpoint, source bytes, partition,
and output path before image/model work. It emits an exact ordered schema containing
all counts, streams, head hashes, paired distributions, strata, gradient diagnostics,
structural norm checks, elapsed time, and one decision. Publication uses a same-
directory exclusive temporary, fsync, hard-link/no-replace publication, strict reload,
and revalidation. Progress may expose only phase/count/time/memory, never partial
scientific values.

## Continuation

On `PROMOTE`, preregister and run
`{class_mean, spherical_probe_seed_0} x {classifier_lr=1e-4, 3e-5}` under the existing
16-epoch train/holdout protocol. Promote beyond that only if the best probe cell beats
the best class-mean cell by at least +0.003 holdout mAP@R with Recall@1 delta at least
-0.00125, followed by paired confirmation and a full-budget reproduction.

On `CLOSE_DIRECTION`, close optimized proxy direction as the next quality mechanism
and return to profiling/evidence synthesis for a different training-time candidate.
The official query/gallery artifact is never reopened for candidate selection.
