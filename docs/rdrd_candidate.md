# Candidate 51: residualized dark-relation distillation (RDRD)

Status: **DEAD AT GATE 2; no implementation and no GPU.**

## Gate 1: repository provenance

Class-pair means explain 52.57--58.90% of cross-class similarity variance in
five final CUB HERD packs. After subtracting those means, image-level residuals
remain stable across seeds (mean Pearson 0.6980, Spearman 0.6756). RDRD would
apply a two-way fixed-effect projection to teacher and student cross-class
similarity matrices and distil only the residual. Proxy Anchor already supplies
class-level separation; the residual supplies which particular image has an
atypical relation to another class.

## Gate 2: prior art

The operation is occupied by differential relational distillation:

- Xie et al., *Pairwise Difference Relational Distillation for Object
  Re-identification*, Pattern Recognition 2024, transfers differences between
  pairwise similarities to preserve teacher ranking:
  <https://doi.org/10.1016/j.patcog.2024.110455>
- Xie et al., *D3still: Decoupled Differential Distillation for Asymmetric Image
  Retrieval*, CVPR 2024, explicitly decouples and transfers pairwise similarity
  differential matrices:
  <https://openaccess.thecvf.com/content/CVPR2024/html/Xie_D3still_Decoupled_Differential_Distillation_for_Asymmetric_Image_Retrieval_CVPR_2024_paper.html>

Subtracting a class-pair fixed effect is a particular linear contrast of the
pairwise similarity matrix. It selects a scientifically useful differential,
but the supervision still asks the student to reproduce teacher pairwise
similarity differences. Candidate 51 is **DEAD at Gate 2**.

