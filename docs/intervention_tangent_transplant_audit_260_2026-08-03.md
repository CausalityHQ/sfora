# Intervention-tangent transplantation audit (candidate 260)

Date: 2026-08-03. Gate 1 and adversarial Gate 2 completed before implementation or
GPU.

## Gate 1: measurement provenance

ARCG measured a stable, non-distance augmentation-response relation on In-Shop:
response agreement retained about 36.3% of same-class pairs, rejected about 53.1% of
the closest quartile, and accepted about 28.0% of the farthest quartile. Its hard
positive-to-unknown gate then self-erased because selected pairs were already beyond
the pair margin. The response differences remain a real measured object even though
using them to remove positive supervision failed.

Candidate 260 proposed treating an observed controlled-transform displacement
`d_j,t = z(augment_t(x_j)) - z(x_j)` as a transferable reaction coordinate. Apply it
to another same-class embedding, `z(x_i) + d_j,t`, and label that virtual point as class
`y_i` under the unchanged Proxy Anchor loss. This would preserve the base attraction
while expanding which within-class states receive supervision, at roughly one extra
view rather than a generative model.

## Gate 2: prior art

The mechanism is occupied, not merely adjacent:

- Schwartz et al., *Delta-Encoder* (2018), explicitly learn transferable intra-class
  deformation deltas from pairs and apply them to other examples/classes.
- Hariharan and Girshick, *Low-shot Visual Recognition by Shrinking and Hallucinating
  Features* / feature-space transfer, and subsequent Feature Space Transfer methods,
  transfer pose/depth or intra-class feature transformations to synthesize examples.
- Park et al., *Meta Variance Transfer* (ICML 2020), learn factors of variation from
  other classes and apply them to scarce examples.
- Ko and Gu, *Embedding Expansion* (CVPR 2020), is benchmark-matched deep metric
  learning that constructs synthetic embedding points and trains metric losses on the
  expanded set.
- ISDA (NeurIPS 2019) estimates class feature variation directions/covariance and trains
  on implicit semantic perturbations.

Using a controlled augmentation to observe the delta instead of learning or sampling it
narrows the source of the transformation but leaves the operator unchanged: transfer a
feature-space variation vector to synthesize a labelled embedding. The ARCG panel does
not create a new supervision type once routed through this established operator.

## Verdict

**DEAD at Gate 2; no implementation or GPU.** The candidate is a transparent,
data-only version of transferable feature deformation / embedding-space augmentation.
It may be cheaper than learned hallucination, but that is an implementation choice, not
a novel similarity-learning mechanism. The search returns to the corrected SOP
artifacts rather than spending a run on a renamed Delta-Encoder/Embedding Expansion.

Primary sources:

- Schwartz et al., arXiv:1806.04734, https://arxiv.org/abs/1806.04734.
- Park et al., ICML 2020, https://proceedings.mlr.press/v119/park20b.html.
- Ko and Gu, CVPR 2020, https://openaccess.thecvf.com/content_CVPR_2020/html/Ko_Embedding_Expansion_Augmentation_in_Embedding_Space_for_Deep_Metric_Learning_CVPR_2020_paper.html.
- Wang et al., NeurIPS 2019, https://proceedings.neurips.cc/paper/2019/hash/15f99f2165aa8c86c9dface16fefd281-Abstract.html.
