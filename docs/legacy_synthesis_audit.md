# Legacy Proxy Synthesis audit

**Audit recorded 2026-07-31. No new GPU work.** The repository retained source
comments describing group-mean and confusion-guided Proxy Synthesis variants as
“novel,” while their existing DGX results and provenance were absent from the
current verdict. This document retracts that unsupported wording.

## Existing paired CUB results

These are final-epoch ResNet-50 values from an older matched harness, not the
current corrected BN-Inception benchmark. Each delta is paired to the same seed's
plain Proxy Anchor run.

| arm | seed deltas (R@1 points) | mean | paired sd |
|---|---:|---:|---:|
| vanilla Proxy Synthesis | +1.165, +0.186, +0.219 | **+0.523** | 0.556 |
| group-mean source mixing | +1.232, +0.203, -0.169 | **+0.422** | 0.726 |
| confusion-guided source pairs | +0.793, -0.101, +0.101 | **+0.264** | 0.469 |

The apparent vanilla and group means are driven by seed 0. The artifacts have no
test curves, so best-over-training and selection-corrected values cannot be
computed. They were not screened on In-Shop, and the variants do not establish an
effect under the current protocol.

## Provenance and novelty

The base operation is [Proxy Synthesis (Gu et al., AAAI
2021)](https://arxiv.org/abs/2103.15454): mix real embeddings and proxies into
virtual classes. Replacing individual samples by group means is set aggregation
inside that established synthesis operation. Choosing nearby proxy pairs is hard
class/pair mining inside the same operation; it does not create a new source of
supervision. Related embedding interpolation is also established by
[Embedding Expansion (Ko and Gu, CVPR
2020)](https://openaccess.thecvf.com/content_CVPR_2020/papers/Ko_Embedding_Expansion_Augmentation_in_Embedding_Space_for_Deep_Metric_Learning_CVPR_2020_paper.pdf)
and [Metrix (Venkataramanan et al., ICLR
2022)](https://openreview.net/pdf?id=ZKy2X3dgPA).
[Memory-Based Virtual Classes (Ko, Gu, and Kim, ICCV
2021)](https://openaccess.thecvf.com/content/ICCV2021/html/Ko_Learning_With_Memory-Based_Virtual_Classes_for_Deep_Metric_Learning_ICCV_2021_paper.html)
is an additional direct neighbour: it stores embeddings and class weights as
virtual classes specifically to reduce over-focus on seen classes and improve
unseen-class generalization.

**Verdict:** the stale novelty labels are withdrawn. The old numbers are
inconclusive evidence about established Proxy Synthesis variants and do not earn
new GPU time under the search protocol. The executable CLI retained the stale
“novel” labels after this audit was written; they are now replaced with accurate
“experimental variant” wording without changing behavior.
