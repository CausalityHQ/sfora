# IDNR Gate-2 prior-art audit

**Decision: DEAD at Gate 2 on 2026-07-31. No implementation or GPU run.**

## Claim audited

Interventional Differencing Nuisance Residualization (IDNR) proposed estimating
a global nuisance subspace from paired embedding displacements between a
training image and controlled augmentations of that same image, then projecting
embeddings and proxies onto the orthogonal complement for Proxy Anchor training
and retrieval. The motivating measurements were that augmentation response is
stable and heterogeneous (ARCG), but using response agreement as relevance
supervision does not transfer (IPSR).

## Decisive collision: nuisance attribute projection

Solomonoff, Campbell, and Quillen, *Nuisance Attribute Projection* (2007;
building on their Odyssey 2004 and ICASSP 2005/2006 work), define nuisance
attribute projection (NAP) for identity matching. NAP estimates directions of
within-identity nuisance variation by an eigenvalue problem, removes those
directions with an orthogonal projection, and performs identity discrimination
in the retained space. The paper explicitly presents the method as applicable
beyond its speaker-verification implementation to high-dimensional observation
vectors.

This occupies IDNR at the mechanism level. Controlled image interventions give
IDNR a cleaner, paired estimator of the nuisance covariance than NAP's observed
cross-session variation, and applying the projection inside Proxy Anchor is a
new benchmark/implementation combination. Neither changes the method class:
estimate nuisance variation, construct its subspace, and quotient it out before
identity comparison.

Weighted NAP (Campbell, Odyssey 2010) further weakens any novelty claim based on
the exact covariance weighting: it extends the same projection mechanism with
variable metrics and instance-weighted training. Nonlinear/kernel NAP work also
shows that moving the projection into a learned feature space is established.

## Other adjacent prior art

- Simard, LeCun, and Denker, *Efficient Pattern Recognition Using a New
  Transformation Distance* (NeurIPS 1992), removes sensitivity to known image
  transformations locally through tangent distance. It is local rather than
  IDNR's fixed global quotient, but already establishes transformation tangents
  as nuisance geometry for recognition.
- TangentProp and later manifold-tangent classifiers regularize output
  sensitivity along transformation directions during learning.
- Concept-erasure methods such as INLP and LEACE estimate and remove linearly
  encoded attribute subspaces. They generally require attribute labels and are
  less direct than NAP, but occupy the broader projection/erasure operation.

## Mechanism-level verdict

IDNR's useful scientific hypothesis remains testable—controlled intervention
differences may estimate nuisance more cleanly than naturally paired sessions—but
that is an estimator and benchmark hypothesis for an established method, not a
novel similarity-learning method. The search protocol therefore kills it before
diagnostic design, implementation, or GPU use.

Primary sources:

- A. Solomonoff, W. M. Campbell, and C. Quillen, *Nuisance Attribute
  Projection*, 2007.
- W. M. Campbell, *Weighted Nuisance Attribute Projection*, Odyssey 2010.
- P. Simard, Y. LeCun, and J. Denker, *Efficient Pattern Recognition Using a
  New Transformation Distance*, NeurIPS 1992.

