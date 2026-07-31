# Post-ARCG candidate batch

**Gate 1 recorded 2026-07-31 before any new prior-art search, implementation,
or GPU work.** This batch reopens the loop because ARCG produced a new measured
constraint, not because an idle GPU needs an arm.

## Measurement that changes the search

ARCG established that controlled-intervention responses contain genuine
within-class structure on In-Shop. Its full-run graph retained 36.40% of
same-class edges, rejected 53.37% of the closest-quartile pairs, and accepted
28.00% of the farthest-quartile pairs. Thus response compatibility is neither
the class label nor ordinary embedding proximity.

ARCG failed because those compatible pairs were already beyond Proxy Anchor's
positive margin. Replacing the own-class proxy term with their detached pair
loss removed the only unsatisfied attractive force: loss fell from 2.3593 to
0.0017 and R@1 collapsed from 0.8463 to 0.7005 in one epoch. RSPG independently
failed through the same positive-to-unknown interface.

The next candidate must therefore satisfy all three constraints:

1. use the measured response/distance disagreement rather than inventing a new
   descriptor;
2. define a target that is provably unsatisfied at construction; and
3. retain the useful own-class Proxy Anchor attraction.

## Ranked shortlist

### 1. IPSR — interventional principal-stratum ranking

**External-science source:** principal stratification in causal inference groups
units by their potential response under controlled interventions; paired-
comparison models in psychometrics turn preferences into ordinal rather than
absolute targets.

At the fixed epoch-10 operating checkpoint, use ARCG's unchanged response
signatures to define latent principal strata. For anchor `a`, construct a
training comparison only when a response-compatible same-class image `p` is
currently farther than a response-incompatible same-class image `u`. The new
supervision is the ordinal statement `sim(a,p) > sim(a,u)`. Keep ordinary Proxy
Anchor unchanged and add a zero-margin Bradley–Terry ranking likelihood over
these registered inversions. No incompatible pair becomes a class negative and
no absolute pair margin is introduced.

This target is unsatisfied by construction, directly using ARCG's measured
53.37% close rejection / 28.00% far acceptance. It differs from ARCG by editing
within-class ordering rather than deleting class-positive supervision. The
novelty claim that Gate 2 must attack is: **controlled-intervention response
agreement between distinct labelled images defines an ordinal preference over
same-class neighbours, while the identity-level attractive objective remains
intact.** Ordinary hard-positive mining, metric learning to rank, causal
representation learning, and augmentation-aware weighting are all dangerous
neighbours.

### 2. RNT — response-norm transport

**External-science source:** reaction norms in quantitative genetics describe
how a phenotype changes across environments; optimal transport matches entire
response distributions rather than individual measurements.

Treat each image's five intervention displacements as a reaction norm. Within
each identity, transport random training views only between images whose
reaction-norm ranks agree, thereby adding cross-image/cross-view positives that
do not exist in standard augmentation. This expands supervision instead of
removing centre-image positives.

It ranks below IPSR because a faithful implementation needs additional image
forwards during every training step (roughly 2x rather than 1x), and
augmentation-aware multi-view contrastive methods may already occupy the
mechanism.

### 3. HFC — homeostatic factor competition

**External-science source:** homeostatic control reallocates effort toward the
largest normalized error while preserving a system-level set point.

Keep Proxy Anchor's total per-class positive force fixed, but allocate that
force across samples in proportion to response-stratum undercoverage. A class
whose minibatch spans several ARCG strata must contribute positive gradient
from each stratum rather than letting its easiest mode dominate.

It ranks last because it is probably continuous loss reweighting, an occupied
class that Gate 2 should kill cheaply, and because its provenance is weaker:
ARCG measured pair inversions, not per-stratum gradient starvation.

## Gate-1 decision

Advance **IPSR** to an adversarial Gate-2 audit. It is the only candidate that
turns the exact observed inversions into a guaranteed unsatisfied target at
roughly unchanged training cost. Do not implement it unless the audit can state
a mechanism-level distinction from hard-positive mining, metric learning to
rank, augmentation-aware pair weighting, and causal principal-stratum methods.
