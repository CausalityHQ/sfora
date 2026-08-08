# Pass 196 — Nearest-Positive Exclusion Supervision (NPES)

## Provenance

The corrected partner-exclusion audit found that removing each query's closest
same-class partner reversed the fragmented-versus-connected gap from `+5.875`
to `−3.910` R@1 points.  This shows that nearest-positive coupling can create a
spurious apparent advantage and motivates making the closest same-class pair
positive-to-unknown during training while retaining the other same-class
positives.

## Gate 2 decision: DEAD

NPES changes positive eligibility, but that mechanism is already occupied by
hard-positive/negative mining and sampling methods, including Hardness-Aware
Deep Metric Learning (CVPR 2019), Sampling Matters in Deep Embedding Learning
(ICCV 2017), and nearest-positive/class-collapsing analyses.  It is also the
same positive-to-unknown gate family already audited for RSPG.  The fixed
nearest-partner selector would additionally be dataset/protocol-specific.
Changing the selector from rival signatures to nearest-neighbour distance is
not a defensible new supervised object.  No implementation or GPU run is
authorized.

Primary sources:

- https://openaccess.thecvf.com/content_CVPR_2019/html/Zheng_Hardness-Aware_Deep_Metric_Learning_CVPR_2019_paper.html
- https://openaccess.thecvf.com/content_ICCV_2017/html/Wu_Sampling_Matters_in_ICCV_2017_paper.html
- https://arxiv.org/abs/2006.05162
