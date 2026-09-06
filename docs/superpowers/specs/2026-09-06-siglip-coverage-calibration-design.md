# SigLIP Coverage Calibration Design

## Purpose

Test whether the failed cross-version retrieval result is caused by missing
deployment-neighborhood coverage in the descriptor alignment set rather than by
an intrinsic incompatibility between the fast 18-block student and the frozen
27-block teacher space. The deliverable is an authenticated, image-disjoint,
few-shot calibration experiment. It is not yet a publication claim.

The existing fast student is already useful in its own space: development
self-retrieval is `99.6355%` Recall@1 and `0.967225` mAP@R. Its complete encoder
latency is at most `0.676688` of the teacher and the previously measured complete
fast path is at most `0.700374` of teacher latency. The remaining release blocker
is cross-version compatibility with a frozen teacher gallery:

- identity student-to-teacher: `69.3803%` Recall@1, `0.663144` mAP@R;
- identity teacher-to-student: `79.8299%` Recall@1, `0.708658` mAP@R;
- best registered fitting-class affine: `70.9599%` / `0.672157` forward and
  `78.9793%` / `0.705102` reverse;
- paired fitting cosine `0.965190`, sealed development cosine `0.565075`.

Errors are concentrated in development classes 5, 19, and 32 even though those
classes remain well separated in the student space. This is consistent with
good student clusters being placed at the wrong teacher coordinates.

## Measured exploratory premise

The exact sealed descriptor bank contains 3,963 paired 512-dimensional student
and teacher descriptors from optimization classes 0 through 48. An independent
read-only calculation, reproduced locally with the repository's retrieval
authority, found that an unregularized affine map becomes accurate when its fit
contains a small image-disjoint support set from each held-out development
class. For deterministic seeds 17, 29, and 43 with eight support images per
development class, the remaining images score:

- mapped student query against teacher gallery: Recall@1 `0.99327` to
  `0.99865`, mAP@R `0.98575` to `0.99252`;
- teacher query against mapped student gallery: Recall@1 `0.99865` for all
  three seeds, mAP@R `0.97212` to `0.98007`;
- native student query against an independently mapped teacher gallery:
  Recall@1 `0.99058` to `0.99327`, mAP@R `0.98722` to `0.99094`;
- mapped teacher query against native student gallery: Recall@1 `0.99865`,
  mAP@R `0.98860` to `0.98938`.

Every value clears the compatibility targets of `0.97` Recall@1 and `0.95`
mAP@R. These measurements are exploratory because the development labels have
already informed method selection. They establish a strong, cheap mechanism
and freeze the method before the historical 49-through-81 diagnostic band is
reopened; they are not final evidence.

The frozen SHA-ranked eight-shot split gives the same conclusion without a
random seed: mapped-student forward compatibility is `0.997308` Recall@1 and
`0.989596` mAP@R; the reverse forward cell is `0.998654` and `0.975170`;
native-student queries against the offline-mapped teacher gallery are `0.993271`
and `0.988406`; and the reverse offline cell is `0.998654` and `0.988576`.

The affine objective itself is label-free: it regresses paired descriptors of
the same image. Class labels are used only to define a reproducible eight-shot
support protocol and to score retrieval. A deployment may instead re-encode a
representative subset of an existing gallery without using its labels.

## Scientific hypothesis

The student and teacher representations are related by a single affine map over
the deployment domain, but fitting-class-only regression leaves low-variance,
class-local directions underconstrained. Eight paired support images per unseen
identity provide sufficient local coverage to identify those directions.

This predicts that a fixed ordinary least-squares affine will pass both
directions on untouched images from external classes 49 through 81 without
changing the student model. If it fails despite verified support coverage, the
mapping is condition-specific and the next justified experiment is
compatibility-aware student training; larger post-hoc maps are not justified.

## Frozen protocol

### Inputs and isolation

Use the exact checkpoint, control binding, optimization manifest, preprocessing,
student tokenwise-tail artifact, and image namespace already authenticated by
the compatibility-capacity diagnostic. Bind every input URI/path, byte length,
and SHA-256 in the result.

Development uses only the already-burned classes 0 through 48. Classes 49
through 81 were previously accessed by the attention-readout campaign, so they
are not a pristine publication holdout. They remain inaccessible to this new
method until the implementation, tests, solver contract, support rule, result
schema, and deployment wrapper are committed and independently reviewed. Their
one frozen result is confirmatory, claim-ineligible evidence. A publication
claim requires a new dataset or a genuinely untouched class split afterward.

For each external class, rank image IDs by
`SHA256("sfora-coverage-support-v1\0" || decimal_class || "\0" || utf8(id))`
and select the first eight as support. All remaining images are evaluation.
Require at least ten images per class. Support pixels and descriptors may be
read before map sealing. The authenticated manifest's IDs and labels may be
read once to commit the support/evaluation partition before descriptor phases;
labels are used only for that partition and later scoring. Evaluation pixels
and descriptors may not be read until both maps are sealed and reloaded. No
checkpoint, hyperparameter, regularization, support count, or map family may be
selected from external results.

### Descriptor extraction

For every support image, run the frozen fast student and frozen teacher once
under the existing preprocessing and BF16 boundaries. Produce normalized FP32
512-dimensional descriptors with exact ID and source-image bindings. Fitting
uses all optimization-class pairs plus the eight external support pairs per
class. Evaluation support images are excluded from query and gallery roles.

### Affine maps

Fit two independent augmented affine maps in CPU FP64:

`[S, 1] W_st = T`

`[T, 1] W_ts = S`

Use `torch.linalg.lstsq(..., rcond=1e-12, driver="gelsd")` with the input rows in
ascending authenticated ID order. Require augmented rank 513, finite singular
values, and finite solutions. There is no identity regularization, nonlinear
adapter, class loss, or hyperparameter sweep. Application is FP64 matrix
multiplication followed by L2 normalization and a contiguous FP32 result.

Record the PyTorch version, BLAS configuration, CPU identity, and thread count.
The sealed map artifact is the numerical authority: a later environment must
authenticate and load it, not claim that refitting is bitwise identical across
BLAS implementations.

Seal both weights, biases, solver evidence, training row identities, and input
digests in one safetensors artifact before external evaluation is decoded.

### Cells

On the support-disjoint evaluation images from classes 49 through 81, compute
exactly:

1. `identity`: native student and teacher spaces without calibration;
2. `forward`: mapped student queries against the teacher gallery and teacher
   queries against the mapped student gallery;
3. `offline-gallery`: native student queries against the mapped teacher gallery
   and mapped teacher queries against the native student gallery;
4. native student self-retrieval and teacher self-retrieval as immutable
   controls.

Use exact same-ID exclusion, stable lowest-ordinal ties, the existing validated
FP32 cosine-score authority,
Recall@1, and mAP@R from the existing validated retrieval authority. Recompute
micro and class-macro aggregates from per-query evidence. The frozen decision
uses the micro metrics; class-macro metrics remain diagnostic.

## Decision and stop rules

Decision precedence is:

1. `invalid`: any digest, schema, support/evaluation isolation, solver rank,
   finiteness, artifact reload, identity, metric recomputation, or process safety
   check fails;
2. `student-quality-rejected`: native student self-retrieval misses either
   `0.97` Recall@1 or `0.95` mAP@R;
3. `coverage-calibration-qualified`: both directions of `forward` and both
   directions of `offline-gallery` each achieve Recall@1 at least `0.97` and
   mAP@R at least `0.95`;
4. `offline-gallery-qualified`: both `offline-gallery` directions pass all four
   thresholds but at least one `forward` direction does not;
5. `forward-only-qualified`: both `forward` directions pass but at least one
   `offline-gallery` direction does not;
6. `coverage-calibration-rejected`: neither deployment route passes.

Stop after this one eight-shot confirmatory evaluation. Do not adapt the support
count, solver, map, preprocessing, or scoring on its results. A rejection
authorizes the already-defined compatibility-aware training design, not another
post-hoc map sweep. A pass authorizes replication and product hardening; it does
not by itself establish SOTA.

## Performance contract

The preferred deployment is `offline-gallery`: transform each stored teacher
descriptor once during gallery migration, then serve the unchanged native
student query encoder. It adds zero query-encoder operations. This does not by
itself prove identical end-to-end index/search latency, so measure the complete
serving path separately. Record offline migration throughput and artifact size,
but do not mix them with online latency.

The `forward` map is retained as a compatibility reference. It adds one
512-by-512 affine per query and requires a fresh paired latency measurement if
selected for deployment. Do not claim that it folds exactly into the existing
projection because the registered map acts on an already normalized student
descriptor and includes a bias.

## Resource and publication boundaries

Run one original process on DGX with the existing pressure monitor: 90-minute
wall cap, 96-GiB CUDA reserved-memory cap, 110-GiB process-group RSS cap,
immediate memory-PSI full avg10 stop at `79.0` percent, sustained stop at
`50.0` percent for three samples, and stop on swap growth. Linux exposes PSI
averages directly in percentage units from `0.00` through `100.00`; the first
hardened attempt incorrectly treated them as fractions and stopped at `0.79`
percent before producing science. Every subsequent execution receipt binds the
thresholds and the complete sampled RSS, CUDA, PSI, and swap timeline, with
recomputed peaks and terminal values. Preserve each revision's first terminal
result and never restart it automatically.

This experiment tests low-shot, label-free paired calibration for backward-
compatible retrieval. It is distinct from retraining the student and from using
class-name semantics. A positive confirmatory result defines a credible method
to take to a fresh public benchmark with multiple seeds and comparisons to
standard backward-compatible training; it is not itself publishable evidence.
Class-name, hierarchical, or hyperbolic losses remain optional later ablations;
the present evidence does not require them.
