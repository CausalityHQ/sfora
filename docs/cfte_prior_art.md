# CFTE Gate-2 prior-art audit

Date: 2026-08-01. Outcome: **DEAD AT GATE 2; no implementation or GPU.**

Cross-fitted tetrad eligibility (CFTE) proposed computing the closed tetrad
under two deterministic augmentation views, retaining only sign-agreeing
high-magnitude relations, and imposing an ordinal quartet constraint.

The operator is already decomposed by primary prior art:

- *Pairwise Ranking Distillation for Deep Face Recognition* defines a relational
  function over arbitrary input n-tuples and minimizes ranking inversions between
  teacher and student relational-function values. A tetrad is one choice of that
  function, not a new ranking-distillation mechanism.
  <https://ceur-ws.org/Vol-2744/paper30.pdf>
- Zhang et al., *Cross-View Consistency Regularisation for Knowledge
  Distillation*, MM 2024, combine within/cross-view consistency with
  confidence-based soft-label mining to select reliable teacher signals.
  <https://openreview.net/forum?id=i4LEfrFPB4>
- Xie et al., *D3still*, CVPR 2024, transfer strict retrieval-ranking
  differentials between similarity relations.
  <https://openaccess.thecvf.com/content/CVPR2024/html/Xie_D3still_Decoupled_Differential_Distillation_for_Asymmetric_Image_Retrieval_CVPR_2024_paper.html>

CFTE composes an arbitrary-n-tuple ranking-distillation operator with an
established cross-view reliability filter. The closed tetrad changes the
relational descriptor; sign agreement changes the mask. Neither changes what
the selected ordinal constraint instructs. This is precisely the descriptor
plus mask pattern that Gate 2 is intended to reject.
