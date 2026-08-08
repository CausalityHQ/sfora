# Pass 159 — norm-ranked cotangent transplant (2026-08-08)

## Gate 1: provenance

Pass 158 found that identity-centered embedding norm predicts within-identity
correctness and margin (Pearson `+0.1417`, Spearman `+0.2097`), while using
norm directly in similarity or a margin is already occupied by quality-aware
metric learning.  This pass asked whether the signal could instead route
updates between same-identity examples: use a high-norm donor's first-order
Proxy-Anchor angular gradient, parallel-transport it to a low-norm receiver's
tangent space, and backpropagate the transported update.  Rival signatures and
augmentation response would be abstention checks only.

## Gate 2: prior art

The operator is not defensibly unoccupied.  If the transported update is
collinear with the receiver gradient it is pair/sample weighting (Multi-
Similarity loss with General Pair Weighting, Wang et al., CVPR 2019; DML-ALA,
Zheng et al., CVPR 2020).  If it is non-collinear it is direct gradient-field
manipulation (PCGrad, Yu et al., NeurIPS 2020; CAGrad, Liu et al., NeurIPS
2021).  A forward-loss rewrite using donor features is embedding-space
expansion/variation transfer (Ko et al., CVPR 2020; Meta Variance Transfer,
Park et al., AISTATS 2020).  Letting magnitude affect the objective returns to
MagFace (Meng et al., CVPR 2021) or AdaFace (Kim et al., CVPR 2022).  Using
response to choose augmentation strength is input-conditioned augmentation
(InstaAug, Miao et al., AISTATS 2023; AdaAug), while using it to admit or weight
pairs is mining/weighting.

## Verdict

**DEAD at Gate 2; no GPU run.**  Norm-ranked donor selection changes the
controller statistic, not the underlying operator.  The proposed mechanism
therefore collapses into occupied families under every formulation.

A CPU falsifier would have compared the first-order nearest-positive-minus-
nearest-foreign margin change against random-donor, norm-permuted, and
cosine-matched controls, but it cannot rescue the prior-art collision.

## Repaired Gate-2 re-audit (2026-08-08)

**Revised status: LIVE-NARROW at Gate 2; Gate 1 unresolved.** The original judgement
treated every non-collinear gradient manipulation as mechanism-equivalent to
task-level PCGrad/CAGrad. Those methods combine distinct objective gradients; they do
not transport one example's cotangent into another example's sphere tangent space.
Scalar pair weighting and forward variation transfer likewise differ in both data
flow and decision point. The candidate is reopened only to a prospectively frozen
training-only causal diagnostic with random-donor, norm-permuted, cosine-matched,
receiver-own-gradient, and generic-surgery controls. No GPU is authorized by this
correction. Full re-audit:
`docs/repaired_gate2_reaudit_pass159_pass181_2026-08-08.md`.
