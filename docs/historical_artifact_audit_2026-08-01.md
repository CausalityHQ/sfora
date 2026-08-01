# Historical artifact audit: frozen foundation-model projections

Date: 2026-08-01

This audit was performed as a Gate-1 evidence-mining pass over JSON artifacts on
the DGX. It is not a benchmark claim.

## Apparent lead

Two legacy frozen-backbone experiments contain unusually high standard-split
Recall@1:

| dataset | frozen backbone | frozen | Group SupCon + XBM + radius | delta |
| --- | --- | ---: | ---: | ---: |
| CUB | DINOv2-small | 0.85466 | 0.85770 | +0.304 pt |
| Cars196 | SigLIP-base-patch16-224 | 0.96458 | 0.96987 | +0.529 pt |

The source artifacts are `image_retrieval_cub.radius_fix.json` and
`image_retrieval_cars.radius_fix.json`. Both declare the standard class-disjoint
train/test sizes (5,864/5,924 and 8,054/8,131), seed 0, 80 projection steps, and
the same composite objective.

## Why this is not Gate-1 provenance for a new method

The increment has one seed and combines three established operations without an
ablation: supervised contrastive grouping, cross-batch memory, and a class-radius
penalty. The radius term is density/compactness regularization, already closed in
the method catalogue. A positive composite delta cannot identify a new mechanism.

The absolute scores use large externally pretrained foundation encoders rather
than the corrected BN-Inception/ResNet-50 reference recipes. Neither artifact
contains a pretraining-data contamination audit for CUB or Cars. Consequently
the absolute values cannot be used as like-for-like evidence that the projection
objective outperforms PFML, HIST, or Proxy Anchor.

Claude independently returned `DEAD`; its defensible point was the same operator
decomposition. Its unsupported numerical claim about expected seed variance is
discarded.

## Decision

Do not queue replications or ablations. Even a statistically confirmed increment
would establish a foundation-feature projection recipe, not a novel
similarity-learning operator. The artifacts remain useful only as evidence that
external representation capacity dominates the older end-to-end benchmark scale.
