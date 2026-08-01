# Candidate 42: negative-control differential distillation (NCDD)

**Gate-2 death recorded 2026-07-31; no implementation or GPU run.**

## Gate 1: PASS

The In-Shop same-versus-cross acquisition cosine gap grows from `0.0251` after
one optimization step to `0.1804` at epoch 10, a **7.18× amplification**. Proxy
Anchor training increases same-group similarity by 0.1252 while decreasing
cross-group similarity by 0.0300.

Inspired by econometric difference-in-differences, NCDD would snapshot the early
model and constrain only the *increase* in

`sim(anchor, same-session positive) - sim(anchor, cross-session positive)`.

It would not force session invariance or reproduce the full early embedding; it
would allow the baseline gap but prevent training from amplifying it.

## Gate 2: FAIL

The one-sided nuisance mask does not create a new relational operator:

- [Pairwise Difference Relational Distillation (Xie et al., Pattern Recognition
  2024)](https://doi.org/10.1016/j.patcog.2024.110455) explicitly distils
  differences between pairwise similarities for object re-identification.
- [Relational Knowledge Distillation (Park et al., CVPR
  2019)](https://openaccess.thecvf.com/content_CVPR_2019/papers/Park_Relational_Knowledge_Distillation_CVPR_2019_paper.pdf)
  transfers distance and angle relations and demonstrates metric-learning
  students that can outperform teachers.
- [Similarity-Preserving Knowledge Distillation (Tung and Mori,
  2019)](https://arxiv.org/abs/1907.09682) aligns teacher and student pairwise
  similarity structure.
- Cross-camera triplet losses already select positives and negatives from other
  cameras to target training-induced camera shortcuts.

NCDD selects two entries of a teacher relation matrix, subtracts them, and applies
a one-sided hinge. That is a nuisance-specific PDRD mask. The early snapshot and
difference-in-differences interpretation alter teacher choice and regularization
direction, not what knowledge is transferred.

**Verdict: DEAD at Gate 2.** The amplification measurement is new and causal;
the executable remedy is established relational distillation.
