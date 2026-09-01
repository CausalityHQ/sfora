# SigLIP Shrunk-Fisher-Quotient Diagnostic Design

Date: 2026-09-01

## Purpose

The current SigLIP Proxy Anchor control reaches 97.98% leave-one-out Recall@1 on
its optimization classes but only 94.54% on class-disjoint clean validation.
This diagnostic tests one narrow explanation before more GPU work: the learned
head over-amplifies noisy seen-class directions, while class-generic
within-class nuisance suppression may transfer.

The diagnostic is prospective, train-only, claim-ineligible, and CPU-capable.
It may authorize a later clean evaluation, but it never reads clean-validation,
burned-diagnostic, or official-test feature rows itself.

## Inputs and authority

The command consumes the existing authenticated cached-feature manifest and its
expected SHA-256. Its optimization-only loader authenticates the complete
manifest but opens only the `optimization-train` feature file; clean and burned
feature files are not process capabilities. It then passes only that band into
the SFQ library boundary. The library requires:

- one finite CPU `float32` feature matrix;
- one CPU `int64` label vector with at least four classes and two examples per
  class;
- a validated `FeatureSplitAuthority` whose role is exactly
  `optimization-train` and whose `official_test_access` is false; and
- a requested output dimension no larger than the feature rank.

No result-emitting API or CLI path accepts evaluation features. Before writing, the CLI validates the
result against separate registered cache-manifest and control-manifest digests,
the ordered-example, feature-matrix, and ordered-label digests, and authenticates
the cache source commit against an explicit registered CLI identity. The result records those identities, fold membership,
and every derived projection digest; the canonical result file's own digest is
the external byte authority.

## Deterministic fold construction

The diagnostic constructs four class-disjoint folds solely from optimization
features. It unit-normalizes rows, computes normalized class means, sorts every
class pair by decreasing mean cosine with label-order ties, greedily forms
disjoint nearest-mean pairs, and treats an odd remaining class as a singleton.
Groups are assigned in that fixed order to the currently smallest fold by
example count, then fold ordinal. This keeps likely near-twin classes together
without using an evaluation outcome.

Every optimization class appears in exactly one validation fold. For fold
`f`, all data-dependent metric state is fit only on classes outside `f`, and
Recall@1 is computed only among rows whose classes are in `f`.

## SFQ metric

Let normalized fit rows be `x_i in R^d`, labels be `y_i`, class means be
`mu_c`, and the global mean be `mu`. Define within-class residuals
`r_i = x_i - mu_{y_i}`. Fit the Ledoit-Wolf covariance with the residual mean
assumed zero:

`C_w = n/(n-C) ((1-rho) S_w + rho tr(S_w)/d I)`.

The implementation uses the pinned scikit-learn estimator, corrects for the
`C` class-mean constraints, and records `rho`.
It eigendecomposes `C_w = Q diag(a) Q^T`, rejects nonpositive eigenvalues, and
forms the symmetric whitening operator

`Phi = Q diag(a^(-1/2)) Q^T`.

For each fit class, form a row of the whitened mean matrix

`G_c = sqrt(n_c) (mu_c - mu) Phi`.

Let `G = U diag(s) V^T`, sample spikes `lambda_k = s_k^2 / C`, and aspect ratio
`gamma = d/C`. Retain only spikes above the 99% finite-sample upper edge using
the Johnstone centering/scaling and the fixed Tracy-Widom-1 quantile 2.023449.
For each retained spike compute

`theta_k = ((lambda_k - 1 - gamma) +
            sqrt((lambda_k - 1 - gamma)^2 - 4 gamma)) / 2`,

`alignment_k = (1 - gamma/theta_k^2) / (1 + gamma/theta_k)`,

`gain_k = alignment_k theta_k`.

Negative radicands beyond a small floating-point tolerance, nonpositive
`theta_k`, nonfinite values, or zero retained rank fail closed. The full metric
factor is

`A = (I + V_r^T diag(gain) V_r) Phi`.

SFQ must deploy as the existing 512-dimensional bias-free linear head. To make
the reduction explicit rather than silently returning a 1152-dimensional
metric, transform the fit rows with `A`, compute an uncentered SVD, retain its
top `m` right singular vectors `P_m`, and return

`W_sfq = P_m A`, with shape `m x d`.

The executable exposes no output-dimension knob: it uses exactly 512 dimensions
for the scientific cache (and deterministically caps at the smaller physical
matrix dimension only for undersized synthetic fixtures).

Singular/eigenvector signs are canonicalized by making the largest-absolute
coordinate positive, with lowest-coordinate ties. A whitening-only comparator
uses the same final uncentered reduction with `A = Phi`. A raw uncentered
spectral comparator applies the same row normalization and deployable output
dimension, but no whitening or Fisher reweighting. Full-dimensional raw cosine
is recorded as context and is not a dimension-matched pass gate.

## Evidence and decision

For each fold, the canonical result records fit/validation labels and counts,
Ledoit-Wolf shrinkage, within-covariance extrema, BBP threshold, observed sample
spikes, retained reliable rank, projection digests, and exact hit/query counts
for raw cosine, raw spectral, whitening-only, and SFQ.

Aggregate Recall@1 is recomputed from summed integer hits. The diagnostic passes
only when:

- all four folds complete with finite deterministic evidence;
- SFQ aggregate Recall@1 exceeds whitening-only by at least 2,000 ppm
  (0.2 percentage points); and
- SFQ aggregate Recall@1 is not below the dimension-matched raw spectral arm.

A failure rejects this preregistered deployable SFQ pipeline and finite-sample
spike rule on the optimization-only folds; it does not reject all Fisher
metrics. A pass authorizes one separately frozen clean evaluation and permits
SFQ as an initializer for the existing subclass-head screen. It does not itself
establish SOTA or permit official-test access.

## Files and verification

- `src/sfora/siglip_sfq.py`: authority, folds, SFQ construction, scoring, and
  canonical result validation.
- `scripts/diagnose_siglip_sfq.py`: strict local-only CLI that authenticates the
  existing cache and emits one new canonical result.
- `tests/test_siglip_sfq.py`: formula, fold isolation, determinism, mutation,
  dimensionality, and failure tests.
- `tests/test_diagnose_siglip_sfq.py`: CLI refusal, cache binding, and canonical
  integration tests.

Implementation follows focused RED/GREEN tests, scoped Ruff/mypy/format checks,
the dependency-complete Python suite, and an independent read-only review.
