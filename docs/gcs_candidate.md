# Candidate 37: gradient-coalition supervision (GCS)

**Gate-2 death recorded 2026-07-31; no implementation or GPU run.**

## Gate 1: PASS

At the exact In-Shop epoch-10 operating point, **17.94%** of 153,115 same-class
pairs have opposing full-dataset Proxy Anchor embedding gradients. Gradient
agreement correlates only `0.2119` with embedding similarity. Cooperative-game
language therefore captures a real fact: a labelled class is not a uniformly
helpful optimization coalition.

## Gate 2 and operator audit: FAIL

The proposed action was to let compatible examples share supervision and withhold
transfer across conflicting examples. Proxy Anchor exposes no pairwise positive
relation to gate, so every implementable realization collapses into established
machinery:

1. selecting samples or batches to match a desired aggregate gradient is
   [GRAD-MATCH (Killamsetty et al., ICML
   2021)](https://proceedings.mlr.press/v139/killamsetty21a.html), including
   per-class last-layer gradient approximations;
2. weighting DML examples by whether a one-step update improves held-out classes
   is [DML-ALA (Zheng et al., CVPR
   2020)](https://openaccess.thecvf.com/content_CVPR_2020/html/Zheng_Deep_Metric_Learning_via_Adaptive_Learnable_Assessment_CVPR_2020_paper.html);
3. removing opposing gradient components is [PCGrad (Yu et al., NeurIPS
   2020)](https://proceedings.neurips.cc/paper_files/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html);
4. adding attraction only between compatible image pairs returns to the
   positive-to-unknown auxiliary interface already tested by RSPG and ARCG, and
   to general positive-pair mining.

The gradient statistic is new to this repository, but substituting it for the
score inside selection, weighting, or gradient surgery is not a new mechanism.
Nor is there a direct pair gate inside Proxy Anchor that preserves its objective.

**Verdict: DEAD at Gate 2.** The measurement may explain why a class label does
not imply uniform optimization effects, but it does not earn a novelty claim or
GPU screen.
