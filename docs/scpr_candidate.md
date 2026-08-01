# Candidate 48: same-class patch recombination (SCPR)

Status: **DEAD AT GATE 2; no implementation and no GPU.**

## Gate 1: repository provenance

The repository's local-feature audit recovered roughly 6.7 CUB R@1 points with
MaxSim relative to global matching, showing that useful same-class evidence can
remain locally aligned when pose and framing displace it globally. SCPR would
cut authentic patches from two training images of the same class and compose a
new positive image, expanding combinations of observed parts without an external
generator or text model.

## Gate 2: prior art

The operation is occupied. CutMix establishes patch replacement as image-level
augmentation, and PartMix applies the class-conditional version directly to
retrieval: it synthesizes positive samples by mixing part descriptors within the
same identity and trains them through contrastive learning.

- Yun et al., *CutMix: Regularization Strategy to Train Strong Classifiers with
  Localizable Features*, ICCV 2019: <https://arxiv.org/abs/1905.04899>
- Kim et al., *PartMix: Regularization Strategy To Learn Part Discovery for
  Visible-Infrared Person Re-Identification*, CVPR 2023:
  <https://openaccess.thecvf.com/content/CVPR2023/html/Kim_PartMix_Regularization_Strategy_To_Learn_Part_Discovery_for_Visible-Infrared_Person_CVPR_2023_paper.html>

Mixing pixels instead of descriptors, omitting modality labels, or using a
different patch sampler changes where and how the established within-identity
part recombination is implemented. It does not create a new supervision
mechanism. Candidate 48 is **DEAD at Gate 2**.

