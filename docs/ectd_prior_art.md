# ECTD Gate-2 prior-art audit

Date: 2026-08-01. Outcome: **DEAD AT GATE 2; no implementation or GPU.**

## Proposed distinction

Effect-size-calibrated tetrad distillation (ECTD) would replace TIRD's cosine
alignment with squared-error matching of class-centred interaction Gram entries,
normalized by the teacher's total Gram energy. The intent was to retain the
measured 4.75% interaction variance share rather than promote the residual to a
unit-scale target.

## Primary-source neighbours

- Tung and Mori, *Similarity-Preserving Knowledge Distillation*, ICCV 2019,
  derive teacher and student pairwise-similarity matrices and minimize their
  matrix discrepancy. <https://openaccess.thecvf.com/content_ICCV_2019/html/Tung_Similarity-Preserving_Knowledge_Distillation_ICCV_2019_paper.html>
- Qian, Li, and Hu, *Improved Knowledge Distillation via Full Kernel Matrix
  Transfer*, SDM 2022, explicitly transfer the teacher's full pairwise kernel
  matrix and bound the matrix-difference objective through a partial matrix.
  <https://doi.org/10.1137/1.9781611977172.69>
- Bhattarai et al., *Knowledge Distillation through Geometry-Aware
  Representational Alignment*, 2025, explicitly use the Frobenius norm of
  teacher/student feature Gram matrices as a distillation loss.
  <https://arxiv.org/abs/2509.25253>
- Bao et al., *Difficulty-aware and Relational Decoupled Knowledge
  Distillation*, Neurocomputing 2026, use a centred-similarity loss with Gram
  matrix matching for non-target relational structure.
  <https://doi.org/10.1016/j.neucom.2026.133850>

## Verdict

TIRD's closed labelled two-class contrast may still be an unreported mask of the
Gram matrix, but ECTD's proposed post-mortem correction is not a new supervision
operator. It is squared/Frobenius Gram matching plus a scalar normalization.
Normalizing by total teacher energy rather than residual energy changes the
effective loss weight; it does not change which relations are supervised or
what target they receive. That cannot carry a new method claim, and screening it
would amount to tuning a failed regularizer after observing its result.

ECTD is therefore dead before implementation. This gate prevents converting a
clean TIRD failure into an unregistered loss-weight sweep.
