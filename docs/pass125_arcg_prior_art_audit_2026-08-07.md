# Pass 125 — Augmentation-Response Compatibility Gate (ARCG)

**Status: LIVE-NARROW at Gate 2; no implementation or GPU authorization.**

## Proposed mechanism

For each image, record its response vector to a fixed panel of controlled
augmentations (crop displacement, colour, blur, and scale).  For two different
same-class images, promote the pair from positive-to-unknown only when their
response vectors agree in a preregistered ordering/metric.  The gate changes
which labelled pairs provide positive supervision; it is not an auxiliary
single-image prediction task and not a soft pair weight.

## Adversarial prior-art checks

* **AugSelf (Lee et al., NeurIPS 2021)** predicts the difference between
  augmentation parameters for two views of the *same image*.  It preserves
  augmentation information as a single-image auxiliary signal; it does not
  compare two different same-class images or decide whether their pair is a
  positive.  ARCG survives this comparison only at the narrow mechanism level.

* **NNCLR (Dwibedi et al., ICCV 2021), pNNCLR, and related support-set
  methods** choose cross-instance positives by embedding proximity (or a
  stochastic perturbation of a nearest neighbour).  They do not gate based on
  augmentation-response signatures.  Replacing the signature with embedding
  proximity would be prior art and is explicitly excluded.

* **Equivariance/augmentation-aware representation learning** generally adds
  an invariance/equivariance loss or predicts augmentation parameters for one
  image.  A cross-instance eligibility test is not present in the primary
  AugSelf/NNCLR mechanisms checked here.  This is not a claim that every
  face/re-identification paper has been exhausted; an additional targeted
  search is mandatory before any implementation.

The targeted search found two important adjacent warnings.  Pose-Aware Metric
Learning (Deng et al., *Pattern Recognition* 2018) synthesizes pose/illumination
variation and learns pose-specific metrics, while PAST (Zhang et al., 2019)
uses ranking-based selection of reliable re-identification pairs under
progressive augmentation.  Neither paper gates labelled same-class pairs by
agreement of per-image augmentation-response signatures, but both mean that a
future ARCG claim must include pose/illumination and reliability-selection
controls rather than comparing only to AugSelf and NNCLR.

## Decision

ARCG is kept **LIVE-NARROW**, not called novel.  Its Gate-1 provenance still
needs an operating-point diagnostic: RSPG showed that cross-class signatures
are nearly vacuous on CUB (64.49% retained) but selective on In-Shop (8.63%),
so an augmentation-response signature must demonstrate nontrivial within-class
variation on the dataset to be screened.  No GPU work is justified before that
diagnostic and a deeper face/re-id prior-art pass.  SRC remains ahead of ARCG.

## Primary sources

- Lee et al., *Improving Transferability of Representations via
  Augmentation-Aware Self-Supervision*, NeurIPS 2021:
  https://openreview.net/forum?id=U34rQjnImpM
- Dwibedi et al., *With a Little Help from My Friends: Nearest-Neighbor
  Contrastive Learning of Visual Representations*, ICCV 2021:
  https://openaccess.thecvf.com/content/ICCV2021/html/Dwibedi_With_a_Little_Help_From_My_Friends_Nearest-Neighbor_Contrastive_Learning_of_Visual_Representations_ICCV_2021_paper.html
- Deng et al., *From One to Many: Pose-Aware Metric Learning for Single-Sample
  Face Recognition*, Pattern Recognition 2018:
  https://doi.org/10.1016/j.patcog.2017.10.020
- Zhang et al., *Self-training with Progressive Augmentation for Unsupervised
  Cross-domain Person Re-identification*, arXiv:1907.13315:
  https://arxiv.org/abs/1907.13315
