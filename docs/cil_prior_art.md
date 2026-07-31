# CIL Gate-1/2 record: counterfactual identity-label invalidation

**Decision: DEAD at Gate 2 on 2026-07-31. No implementation or GPU run.**

## Repository provenance

ARCG measured stable, heterogeneous response to controlled crops, while IPSR
showed that response agreement between different images is not retrieval
relevance. A narrower interpretation is that a severe crop sometimes removes
the evidence needed to support the source identity. Counterfactual
Identity-label Invalidation (CIL) would compare a crop with its centre view at a
frozen operating checkpoint and change a crop whose own-class evidence
collapses from positive to unknown, while retaining ordinary Proxy Anchor for
label-preserving crops.

This changes whether an augmented observation is eligible for its inherited
label; it is not per-image crop-strength selection (RAAD) or cross-image
positive gating (RSPG/ARCG).

## Decisive collision

Miyai et al., *Rethinking Rotation in Self-Supervised Contrastive Learning:
Adaptive Positive or Negative Data Augmentation* (WACV 2023), introduce PNDA.
PNDA decides per image whether a transformed observation preserves semantics
and treats the transformed pair as positive when it does and negative when it
does not. This is the same supervision-eligibility operation. CIL substitutes a
supervised proxy-response test for PNDA's unsupervised rotation test and uses
unknown rather than negative for invalid crops; those are detector and target
choices, not a new method mechanism.

Related work on semantic-drift filtering, learned/view-specific augmentation,
and label-destroying augmentations makes the surrounding territory still more
occupied. Candidate 24 therefore fails prior art before preregistration.

Primary source:

- A. Miyai, Q. Yu, D. Ikami, G. Irie, and K. Aizawa, *Rethinking Rotation in
  Self-Supervised Contrastive Learning: Adaptive Positive or Negative Data
  Augmentation*, WACV 2023.

