# OBPA Gate-1/2 record: occupancy-bag Proxy Anchor

**Decision: DEAD at Gate 2 on 2026-07-31. No implementation or GPU run.**

## Repository provenance

ARCG established that controlled crops produce stable but heterogeneous model
responses; IPSR established that this response is nuisance, not a useful order
between identities. CIL then collided with per-image positive/negative
augmentation assignment. Inspired by ecological repeated-survey occupancy
models, OBPA instead placed several augmented views of one source into a bag:
the source identity is present in at least one survey, without claiming that
every occluded crop retains identifiable evidence. A noisy-OR or smooth-max bag
term would replace per-view positive Proxy Anchor supervision while leaving
negative-class proxy terms intact.

## Decisive collision

Martinez-Cortes, Gonzalez-Diaz, and Diaz-de-Maria, *Training Deep Retrieval
Models with Noisy Datasets: Bag Exponential Loss* (Pattern Recognition 112,
2021), already formulate deep image retrieval under the multiple-instance
assumption that a bag of matching pairs contains at least one true positive.
Their loss dynamically estimates/weights which members are relevant during
training, precisely avoiding the claim that every bag member is valid matching
evidence. They explicitly describe the formulation as general beyond noisy
datasets.

OBPA changes the source of bag instances from noisy retrieved images to repeated
augmentations and attaches the bag objective to a class proxy. That does not
change the latent-positive MIL mechanism. General MIL, multi-instance metric
learning, and multi-view contrastive learning further occupy the abstraction.
Candidate 25 therefore fails novelty before preregistration.

Primary source:

- T. Martinez-Cortes, I. Gonzalez-Diaz, and F. Diaz-de-Maria, *Training Deep
  Retrieval Models with Noisy Datasets: Bag Exponential Loss*, Pattern
  Recognition 112:107811, 2021.

