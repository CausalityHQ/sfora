# Pixel-observable candidate audit: 164--169

Date: 2026-08-01. Claude was used as an adversarial co-designer after the
equivalence-relation limitation in candidates 159--163. No implementation or
GPU run followed.

## 164. Partial-information-decomposition synergy

Making the class label decodable from combinations of regions but not individual
regions is higher-order feature interaction plus local/global aggregation. It
does not add a measured target beyond the class label, overlaps latent visual
concept learning, and its deployment path is local-to-global distillation.
**DEAD AT GATE 2.**

## 165. Minimum-description-length class grammar

Coding one image using a same-class image or learned grammar supervises shared
reconstructability. Rate or code length is a monotone wrapper around a
reconstruction likelihood; AdvRF and class-conditional sparse/dictionary models
occupy the mechanism. **DEAD AT GATE 2.**

## 166. Causal negative controls

Hand-designed identity-preserving or identity-destroying transformations supply
augmentation invariance/equivariance and adversarial mining. Calling an
augmentation a negative control does not identify a new causal quantity.
**DEAD AT GATE 2.**

## 167. Algorithmic teaching residual

Selecting a minimal class exemplar set and predicting the remaining examples is
learned exemplar/prototype selection plus reconstruction. Training identities'
selected teaching sets are also unavailable for unseen test identities.
**DEAD AT GATE 2.**

## 168. Metamorphic retrieval relations

Logical relations among outputs under composed transformations are transformation
consistency and equivariance. Candidate 152 and ARCG already close the operator.
**DEAD AT GATE 2.**

## 169. Same-class gradient-attribution alignment

The proposed observable aligns input-gradient saliency maps for same-class
images. Li et al., *Unsupervised Deep Metric Learning with Transformed Attention
Consistency and Contrastive Clustering Loss* (2020), already impose pairwise
attention consistency across matched transformed images in metric learning.
MAMC (Sun et al., ECCV 2018) likewise regulates corresponding attention regions
across images in a metric framework. Changing the attention estimator to an
input gradient is not a new supervision relation. Moreover, optimizing a loss
of input gradients requires second-order differentiation, contradicting the
claimed roughly 1x cost. **DEAD AT GATE 2.**

Primary sources:

- <https://arxiv.org/abs/2008.04378>
- <https://arxiv.org/abs/1806.05372>
