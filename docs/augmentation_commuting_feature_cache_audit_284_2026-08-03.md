# Candidate 284: augmentation-commuting feature cache

**Verdict: DEAD at Gate 2; no implementation or GPU.**

## Proposed mechanism and provenance

The audited operation count establishes that the image backbone, not Proxy Anchor's
sample--proxy matrix, dominates training cost. The proposal was to freeze an early
convolutional stem, cache a canonical spatial feature field once per image, and apply
random crops/flips (plus limited channel transforms) to that field. Later epochs would
train only the remaining network, avoiding repeated stem forwards while retaining more
augmentation diversity than a single cached global embedding.

## Gate 2 prior-art audit

The mechanism is directly occupied.

- Yang et al., *Rethinking the Potential of Layer Freezing for Efficient DNN Training*
  (2025), explicitly identify that frozen layers still incur forward computation,
  cache frozen-layer feature maps as a new dataset, handle augmentation-sensitive
  channels, and progressively compress the cache:
  <https://arxiv.org/abs/2508.15033>.
- Bär et al., *Frozen Feature Augmentation for Few-Shot Image Classification* (CVPR
  2024), cache frozen visual features and apply a systematic family of feature-space
  augmentations:
  <https://openaccess.thecvf.com/content/CVPR2024/html/Bar_Frozen_Feature_Augmentation_for_Few-Shot_Image_Classification_CVPR_2024_paper.html>.
- *Learning from Offline Foundation Features with Tensor Augmentations* likewise
  trains from cached features and applies tensor augmentations because enumerating
  image augmentations in the cache is infeasible:
  <https://openreview.net/forum?id=VVd3iOKPMJ>.

Restricting these operators to DML or to geometric transforms of a convolutional field
does not create a new learning mechanism. It also inherits a fidelity risk: crop/resize
does not exactly commute with a strided, padded, nonlinear stem, so quality preservation
would require the same sensitivity machinery already studied by Yang et al. Candidate
284 therefore fails novelty before preregistration or implementation.

