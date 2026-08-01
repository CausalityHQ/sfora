# Post-TIRD candidate batch

Date: 2026-08-01. This shortlist was written before any implementation or GPU
work on the candidates below.

## Measurements that constrain the batch

TIRD transferred the closed two-class 2x2 interaction measured in five CUB
HERD packs. That component was reproducible across seeds (Pearson 0.5710) but
accounted for only 4.75% of cross-class similarity variance. The In-Shop screen
failed decisively: raw R@1 0.8301 versus paired Proxy Anchor 0.9024, and
selection-corrected paired delta -7.405 points. Its curve lagged, oscillated,
and converged to a much lower ceiling.

The operative defect is testable from the loss definition. Cosine matching is
invariant to the interaction matrix's norm, so it promoted a small residual to
a unit-scale global target. The next candidate must preserve the measured
effect size or change the supervision instruction; merely lowering the same
loss weight after seeing the result is not a candidate.

## Ranked shortlist

### 54. Effect-size-calibrated tetrad distillation (ECTD)

Instead of cosine-aligning flattened interaction matrices, match the closed
tetrads in their original scale and normalize by the teacher's *total* Gram
energy, not the residual energy. This preserves the 4.75% ANOVA share: a weak
component remains weak while its sign and magnitude remain supervised. The
base Proxy Anchor loss is unchanged.

Why first: it is the only proposal that directly removes TIRD's measured
failure mechanism without changing the discovered supervision object. It is
also cheap to kill analytically and at Gate 2.

Gate-2 threat: similarity-preserving KD, RKD, centered-kernel alignment, and
variance-normalized feature distillation may already occupy scale-preserving
Gram residual matching. If the closed labelled two-class interaction is not a
substantive distinction, ECTD is dead. Even if novel, the likely outcome is
near-inert rather than a benchmark win because the target contains little
energy.

### 55. Cross-fitted tetrad eligibility (CFTE)

Estimate tetrad signs independently under two deterministic augmentation views.
Only sign-agreeing, high-magnitude tetrads create an ordinal cross-class quartet
constraint; all other relations remain unknown and Proxy Anchor is retained.
This changes a continuous teacher match into a new eligibility decision for
which cross-class comparisons exist.

Why second: TIRD showed that dense matching is destructive, while the underlying
interaction is reproducible. Cross-fitting imports the replication principle
from measurement science and prevents a single noisy view from manufacturing
supervision.

Gate-2 threat: quadruplet losses, uncertainty-filtered pseudo-labels,
multi-view-consistency mining, and multi-teacher agreement KD may jointly
occupy the mechanism. A new mask over an established ordinal quartet loss is
not enough.

### 56. Augmentation-complement positive completion (ACPC)

Retain every original Proxy Anchor positive and add a separate attractive term
only for same-class pairs with strongly *different* controlled-augmentation
response signatures. ARCG selected agreeing responses and erased the base
attraction; ACPC instead treats response disagreement as missing within-class
coverage and never turns an existing positive unknown.

Why third: ARCG measured a selective response graph independent of distance,
but its replacement interface failed. Complement selection directly targets
the unobserved appearance-factor span.

Gate-2 threat: hard-positive mining, diversity-aware positive sampling,
augmentation-aware metric learning, and Metrix-style synthetic support are
close. If response disagreement only chooses a harder positive, the descriptor
is cosmetic and ACPC is dead.

## Decision

Run Gate 2 on ECTD first. Do not implement or queue it unless the audit can
defend both (1) the closed labelled two-class tetrad as the transferred object
and (2) total-energy calibration as more than ordinary loss weighting. If that
second distinction fails, record ECTD as dead rather than tuning TIRD.
