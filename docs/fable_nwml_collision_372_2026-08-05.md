# Fable NWML collision audit (candidate 372)

Date: 2026-08-05.

## Execution and proposal

This was a fresh repository-blind, outcome-only Fable pass. The prompt supplied
the zero-shot similarity-learning problem, benchmark constraints, and audited
external frontier, but no candidate mechanisms or repository failure catalogue.
Fable had web search and fetch but no filesystem or shell access.

Fable returned one proposal, **Nuisance-Whitened Metric Learning (NWML)**. For
same-image controlled augmentation pairs `(z_i, z_i')`, it estimates

```text
Sigma_N = (1 / (2p)) sum_i (z_i - z_i') (z_i - z_i')^T,
```

shrinks and exponentially averages that covariance, applies
`A = Sigma_N^(-alpha)` to descriptors and proxies, normalizes, and otherwise
uses Proxy Anchor unchanged. The proposed deployed descriptor folds `A` into
the embedding head. Fable proposed differentiating through `A`, sampling second
views for one quarter of images, and claimed about 1.25x training cost with no
test-time overhead.

Its frozen forecasts were about **0.717 CUB** (+2.0 points), at least +1.5 Cars
points with colour jitter, and +0.8 In-Shop points. Fable explicitly said these
forecasts do **not** reach the audited comparable-capacity frontier of 0.766 CUB,
0.949 Cars, and 0.939 In-Shop.

## Gate 1: provenance fails

Fable explicitly said the supplied repository evidence did not motivate this
mechanism; it instead appealed to external observations about spectral
correlation and dimensional collapse.

The nearest repository measurement is IPSR's corrected official-pixel Gate 0:
17,093 preferences, response-graph density 0.361375, 56.09% of closest-quartile
pairs rejected, and 29.15% of farthest-quartile pairs accepted. This establishes
that controlled augmentation response has deterministic structure not reducible
to embedding distance. It does **not** show that global response covariance is
nuisance rather than identity-bearing, that whitening it repairs retrieval
errors, or that the intervention funds the roughly 2.5--5 point gap to the
audited frontier.

The premise is therefore an external analogy, not an action identified by a
measurement in this repository. It fails search-protocol Gate 1.

## Gate 2: the operation is classical

### Relevant Component Analysis

Bar-Hillel et al., *Learning a Mahalanobis Metric from Equivalence Constraints*
(JMLR 2005), introduce Relevant Component Analysis (RCA). Given automatically
obtainable positive equivalence constraints grouped into chunklets, RCA computes
the within-chunklet covariance and applies its inverse square-root. Equivalently,
it uses the inverse covariance as a Mahalanobis metric, suppressing directions
with high variability inside equivalence sets.

A same-image original/augmentation pair is a two-point chunklet. NWML's exact
whitening case is therefore RCA with controlled transformations as the source
of equivalence constraints, followed by an established proxy loss. EMA,
shrinkage, minibatch estimation, and differentiation through the estimator are
implementation and optimization choices, not a new similarity or supervision
mechanism.

Primary source: https://www.jmlr.org/papers/v6/bar-hillel05a.html

### Transformation-invariant Mahalanobis metrics

Fetaya and Ullman, *Learning Local Invariant Mahalanobis Distances* (ICML 2015),
explicitly learn a Mahalanobis metric that is insensitive to known
transformations, using vectors of the form `T(x) - x`. This occupies the more
specific claim that controlled image transformations should define the
directions suppressed by a learned metric.

Primary source: https://proceedings.mlr.press/v37/fetaya15.html

### NAP and WCCN

Nuisance Attribute Projection (NAP) and Within-Class Covariance Normalization
(WCCN) already estimate within-identity or intersession variability and suppress
it for identity matching. Candidate 23's IDNR audit had already killed the same
controlled-augmentation-to-global-nuisance-geometry action: paired
interventions can be a cleaner estimator, but do not define a new method class.
Weighted and nonlinear NAP also cover covariance weighting and learned feature
spaces.

Primary WCCN source:
https://www.isca-archive.org/interspeech_2006/hatch06_interspeech.html

NAP/WCCN comparison:
https://www.sri.com/publication/speech-natural-language-pubs/nap-and-wccn-comparison-of-approaches-using-mllr-svm-speaker-verification-system/

This also repeats candidate 176's nuisance-tangent quotient and candidate 312's
augmentation-response tangent/Fisher/Mahalanobis metric. Gate 2 is closed by
both exact and adjacent prior art.

## Mathematical correction

Fable claimed exact invariance to an arbitrary invertible reparameterization of
the embedding. That statement only holds for exact full whitening
(`alpha = 1/2`) with a full-rank exact covariance: two whitened coordinate
systems then differ by an orthogonal transform. It does not hold for a general
fractional power `Sigma_N^(-alpha)`. Shrinkage toward the identity and an EMA in
a fixed coordinate system also break the congruence property.

Consequently, the practical stabilized algorithm does not have the claimed
exact invariance. The special case that does have it is precisely classical RCA
whitening. Differentiating through the covariance changes optimization but does
not change that operation or create a new supervision relation.

The proposed preliminary diagnostic was also not execution-ready. It asked for
the fraction of held-out between-class scatter lying in directions with large
within-class but small augmentation variance, predicting more than 25%, without
fixing a generalized eigensystem, rank or thresholds, shrinkage, or an
attribution formula. It would require new augmented forward passes rather than
being a free CPU diagnostic. None of these details need repair because Gate 2
already kills the candidate.

## Verdict

**DEAD at Gates 1 and 2. No diagnostic, implementation, preregistration, or GPU
run.** NWML is not motivated by a repository measurement, is an application of
RCA/WCCN/NAP and transformation-invariant Mahalanobis learning, and does not
reach the project's benchmark objective even under its author's own forecast.

