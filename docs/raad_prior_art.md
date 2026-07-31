# RAAD Gate-2 prior-art audit

**Verdict: DEAD at Gate 2 on 2026-07-31, before implementation or GPU work.**

RAAD proposed to measure each image's response to spatial interventions, then
assign a milder crop distribution to sensitive images and retain the reference
distribution for robust images. The dose-titration analogy does not create a
new learning mechanism.

The direct collision is **InstaAug**. Miao et al. learn an input-specific
augmentation module that maps each image to tailored transformation parameters,
explicitly because valid crop invariances are input-dependent. They demonstrate
instance-specific cropping and allow the policy to be learned jointly or for a
pretrained model
([ICML 2023](https://proceedings.mlr.press/v202/miao23a.html)). RAAD replaces
their learned invariance module with a fixed checkpoint's embedding-response
score, but the operator is the same: infer a per-instance safe transformation
distribution and train on samples drawn from it.

Two adjacent papers remove any defensible narrower claim. **AdaAug** learns
class- and potentially instance-adaptive augmentation policies
([Cheung and Yeung, ICLR 2022](https://openreview.net/forum?id=rWXfFogxRJN)).
**iMAS** measures instance hardness and proportionally weakens strong
augmentations for hard samples so they are not pushed out of distribution
([Zhao et al., CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Zhao_Instance-Specific_and_Model-Adaptive_Supervision_for_Semi-Supervised_Semantic_Segmentation_CVPR_2023_paper.html)).
Soft Augmentation separately establishes that crop severity can make a hard
label unreliable and softens targets as severity increases
([Liu et al., CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Liu_Soft_Augmentation_for_Image_Classification_CVPR_2023_paper.html)).

RAAD's response score is different machinery, not a different supervision
operator. “Use model-measured per-instance sensitivity to reduce that instance's
augmentation strength” is already occupied. No diagnostic, implementation, or
benchmark run is warranted.
