# MCNS — matched-control negative supervision

**Gate 1 recorded 2026-07-31 before prior-art audit, implementation, or GPU
use.**

## Repository provenance

Two independent measurements motivate MCNS:

1. At the exact epoch-10 In-Shop operating checkpoint, leave-one-out top-1 has
   1,600 errors. **21.19% of those errors belong to an ordered
   `(true identity, retrieved identity)` pair that occurs at least twice**, and
   one identity pair accounts for nine errors. A material error subset is
   systematic confusion between whole identities, not isolated sample noise.
2. Controlled-augmentation response varies strongly within In-Shop identities
   (the RSPG agreement gate retains only 8.63% of pairs), but IPSR established
   that response agreement is nuisance structure rather than retrieval
   relevance. It can therefore serve as a matching covariate without being used
   as a target.

Ordinary hard-negative mining selects cross-class pairs by embedding proximity.
That mixes identity difference with pose/crop sensitivity. The repeated
class-pair errors suggest controlling that nuisance may expose the fine identity
difference that must generalize.

## Cross-disciplinary mechanism

MCNS borrows matched case-control design from causal inference. Within a
systematically confused pair of training identities, it matches images across
the identities by a *separate, label-free nuisance covariate*: their frozen
response vectors to controlled augmentations. A negative margin is then applied
to matched cross-identity pairs. Matching holds nuisance response roughly
constant, so the supervised contrast isolates identity rather than rewarding an
easy pose mismatch.

At a frozen warm-up checkpoint:

1. find confused identity pairs from aggregate cross-class retrieval errors;
2. compute each image's controlled-augmentation response signature;
3. use one-to-one minimum-cost matching between the identities in response
   space, without test data or attributes;
4. retain ordinary Proxy Anchor unchanged and add a bounded negative margin on
   matched controls only; refresh once if preregistered.

The claimed distinction from hard-negative mining is that eligibility is based
on *nuisance balance in an independent intervention-response space*, not the
distance being optimized. The distinction from ARCG/IPSR is that response is
neither a positive graph nor a relevance order; it is a control variable for a
label-certain negative comparison.

## Gate-2 attack required

Before implementation, search primary sources for pose/viewpoint/illumination-
matched negative mining in face recognition and re-identification;
nuisance-aware and attribute-conditioned contrastive sampling; causal or
propensity-score matching for metric-learning negatives; cross-class optimal-
transport matching; and augmentation-response descriptors used to match
negative instances.

MCNS is dead if prior work already matches different-identity images on a
nuisance covariate before applying metric supervision. It is also dead if using
a learned response signature instead of explicit pose is merely an estimator
substitution for that established mechanism.

