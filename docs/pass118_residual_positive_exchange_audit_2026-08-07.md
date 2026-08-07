# Pass 118 — Residual positive exchange (RPEX) audit

Date: 2026-08-07.

## Gate 1: measured motivation

The corrected In-Shop PEBH CPU gate found that exchanging a same-identity
donor's full learned evidence increased held-out positive similarity by a mean
`+0.004747`, but increased nearest-foreign similarity by `+0.008855`; one seed
also reduced leave-one-out R@1. RPEX proposed exchanging only the donor
residual orthogonal to the receiver's own-class direction, leaving the
receiver's identity component unchanged. This is a measurement-derived
hypothesis: the positive-transfer benefit survived, but exchanged identity
content created a larger hub effect.

## Gate 2: prior-art reduction

RPEX is not defensibly novel. Its training object is still a cross-image
representation: a same-label peer supplies a feature to a receiver, while a
self-only descriptor is retained for deployment. Wang et al., *Joint Learning
of Single-Image and Cross-Image Representations for Person Re-Identification*
(CVPR 2016), explicitly trains a shared single-image branch and a
pair-conditioned cross-image branch. X-ReID (Shen et al., 2023) uses
cross-attention between different images of the same identity to transfer
identity-level information and train a self-image representation. Cross-GAN
(2019) likewise learns cross-image representations from paired observations.
Residualizing the transferred feature and projecting away a proxy direction
changes the feature algebra, not the supervision object. Feature-exchange and
feature-transfer blocks occupy the same family.

## Disposition

**DEAD at Gate 2.** The PEBH measurement remains useful: full positive exchange
has a measurable hub failure. It does not authorize a residualized exchange
variant. No implementation, CPU probe, preregistration, or GPU run occurred.

