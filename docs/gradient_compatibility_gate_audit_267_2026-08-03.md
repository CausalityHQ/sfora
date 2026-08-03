# Candidate 267: gradient-compatibility positive gate

**Verdict: DEAD at Gate 2. No implementation and no GPU.**

## Gate 0/1: measured provenance

RSPG retained **0.6449** of within-class pairs on CUB but only **0.0863** at its
epoch-10 In-Shop operating point. Subject to the already-recorded contaminated
decision-path caveat, that split says a target-excluded rival-class signature is
nearly vacuous on CUB and highly selective on In-Shop. It motivates looking for an
intrinsic, training-derived compatibility signal that does not merely ask which rival
classes an image resembles.

Candidate 267 borrowed coherent-gradient language from optimization and influence
analysis. For two different images with the same identity, it would compute a cheap
per-example gradient signature and retain the pair as positive only when the two
updates agree; a disagreeing labelled pair would become unknown. The hoped-for
mechanism was that compatible updates identify a shared within-class factor even when
the current embeddings are visually distant.

This provenance motivates the question, not an effect: this repository has not yet
measured that gradient agreement predicts retrieval correctness.

## Gate 2: adversarial prior-art result

The mechanism is already substantially occupied. Zeng et al.,
[Additional Positive Enables Better Representation Learning for Medical Images
(2023)](https://arxiv.org/abs/2306.00112), extend TracIn to self-supervised learning,
compute batch-wise per-sample gradients, use pairwise gradient influence as
cross-instance similarity, and select another image as an additional positive. That is
the central operator proposed here: gradient compatibility decides which different
image receives positive supervision.

The broader ingredients are also established:

- Chatterjee and Zielinski,
  [Making Coherence Out of Nothing At All
  (2020)](https://arxiv.org/abs/2008.01217), define per-example gradient coherence as
  whether a step on one example benefits others.
- Jackson and Schulman,
  [Semi-Supervised Learning by Label Gradient Alignment
  (2019)](https://arxiv.org/abs/1902.02336), explicitly map examples into model-gradient
  space and use distance there as a semantic metric.
- Liu et al.,
  [Debugging and Explaining Metric Learning Approaches: An Influence Function Based
  Perspective](https://openreview.net/pdf?id=ocg4JWjYZ96), apply influence analysis to
  deep metric learning itself.

The proposed hard **positive-to-unknown** decision and restriction to labelled
same-class pairs are implementation choices relative to Zeng et al.'s top-positive
selection. They do not create a defensible new source of supervision: in both cases,
cross-example gradient influence determines which other image is pulled as a positive.
Moreover, a Proxy Anchor per-example gradient is strongly determined by its proxy and
negative-proxy responses, so it risks re-encoding the cross-class signature whose CUB
degeneracy motivated leaving that family.

## Decision

Candidate 267 stops at Gate 2. No diagnostic, preregistration, code, or GPU run follows.
The useful residue is narrower: an eventual candidate needs an intrinsic within-class
signal whose **source**, not merely its thresholding policy, differs from embedding
proximity, contextual/rival signatures, augmentation response, and gradient influence.
