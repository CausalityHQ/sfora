# Signature-Matched Product Geometry SOTA Gate

**Status:** rejected at adversarial design review; do not implement.

## Decision

Non-Euclidean geometry remains an idea for reaching state of the art, not a
requirement. This exact product-geometry program is closed because its proposed
falsifiers and serving claim cannot support that objective. No new geometry
code, training arm, or GPU run is authorized by this document.

The active queue retains the already implemented Lorentz L0/L1 experiment as a
cheap frozen-embedding diagnostic. That experiment may establish a
quality/storage result or close a function family. It cannot reopen curved
training automatically and cannot establish an absolute SOTA claim.

## Why the draft was rejected

### 1. The proposed triangle statistic was invalid on this artifact

The draft used

```text
xi(a;b,c) = d(a,m)^2 + d(b,c)^2 / 4
            - (d(a,b)^2 + d(a,c)^2) / 2
```

with `m` the midpoint of `b,c`. On the queued Euclidean point cloud, using
`m=(b+c)/2`, Apollonius' identity makes `xi` exactly zero for every triple.
The proposed requirement that its interval exclude zero was therefore
unpassable by construction. Gu et al.'s discrete graph-midpoint estimator
cannot be transplanted to a point cloud without defining and validating a
graph metric, midpoint-row rule, and normalization. Creating that new
instrument is not justified while the existing Lorentz falsifier is pending.

### 2. The claimed one-GEMM product score was algebraically false

For one Lorentz factor, maximizing a Lorentz inner product is equivalent to
minimizing its geodesic distance because `-cosh(d)` is monotone. This does not
extend to a sum of squared geodesic distances. For multiple factors,

```text
sum_f alpha_f arccosh(-<q_f,g_f>_L)^2
```

is not affine in concatenated inner products; the spherical `arccos^2` case
has the same problem. Exact product-geodesic retrieval requires factorwise
matrix products plus elementwise transcendental transforms and loses ordinary
single-index FAISS compatibility. Replacing geodesic distance by an affine
surrogate would define a different, already occupied method family.

### 3. The available artifact cannot answer the intended frontier question

The queued released UNICOM ViT-B/16 export targets the 74.6 zero-shot point.
The published 95.5 In-Shop point requires a 128-epoch, four-GPU supervised
fine-tune and has no released checkpoint. Residual-error composition and
no-training rescoring on the zero-shot model cannot authorize a claim about
the residual errors of the unavailable 95.5 model. Waiting for a local 95.5
reproduction would make the allegedly free falsifier depend on the expensive
experiment it was intended to precede.

### 4. The program could not produce the requested absolute result

The draft's minimum successful outcome was 95.5 plus 0.5 Recall@1 point, or
96.0. That only ties the published UNICOM ViT-L/14 point and remains below the
audited 96.7 ViT-L/14@336 reference. Its inference path was slower rather than
Pareto-better, and its 512-D descriptor supplied no registered compression
advantage. It was a lane-closure experiment, not an absolute SOTA program.

### 5. The research territory is occupied

Mixed positive/zero/negative-curvature image retrieval is directly occupied by
Xu et al., *Mixed-Curvature Metric Learning for Image Retrieval*, IEEE TMM
2026. Product manifolds, kappa-stereographic networks, Hyp-ViT, HIER, MERU,
AMCAD, and hyperbolic prototype classifiers occupy the surrounding components.
Yue et al. show that hyperbolic DML gains can be explained by implicit
hard-negative weighting. Repository Pass 70 had already closed generic
hyperbolic/product manifolds as prior art or an already controlled
regularization family. The TMM paper's exact In-Shop row remains worth
recovering as a baseline fact, but it cannot rescue novelty for this design.

## SOTA program after closure

The research target remains unchanged:

1. reproduce and validate the strongest feasible modern anchor before
   attributing improvements;
2. treat 95.5 as the matched ViT-B/16 frontier and 96.7 as the audited absolute
   In-Shop descriptor reference until a stronger comparable source is
   verified;
3. use the active UNICOM audit to measure real evaluator and distributed-mask
   defects, then prefer full-width, synchronized-mask, corrected-normalization,
   and EMA controls over a speculative geometry head;
4. require any quality candidate to improve the matched final checkpoint by at
   least 0.5 Recall@1 point with a one-sided paired confidence bound above zero;
5. call a result absolute SOTA only if it exceeds the strongest verified
   descriptor-only point in its declared lane; otherwise require a genuine
   Pareto result with the authoritative quality-equivalence and cost gates;
6. replicate a surviving direction on at least two of CUB, Cars196, and SOP.

The already queued order remains correct: finish the three-seed
PA/MCPS/compactness comparison, export and validate UNICOM, run Lorentz and CTM
as frozen diagnostics, and run the DADA compatibility smoke. The next learning
candidate is selected from those measured results, not from geometric
intuition.

## What would be required to revisit geometry

Geometry may be reconsidered only after all of the following independently
exist:

- a locally reproduced supervised modern anchor, rather than the 74.6
  zero-shot export;
- an error analysis showing a geometry-specific repair ceiling large enough to
  cross the absolute or Pareto frontier;
- a scorer with an exact maintained-index reduction and measured cost;
- a result that uniquely beats temperature, norm-weighting, hard-negative,
  spatial-only, and power-law controls;
- a novelty distinction from MCML and the other published families.

That would be a new design with new prospective gates. It does not inherit the
invalid midpoint statistic, product score, or thresholds from the rejected
draft.

## Primary sources

- UNICOM, ICLR 2023: <https://openreview.net/forum?id=3YFDsSRSxB->
- Mixed-Curvature Metric Learning for Image Retrieval, IEEE TMM 2026:
  <https://doi.org/10.1109/TMM.2026.3651105>
- Learning Mixed-Curvature Representations in Product Spaces, ICLR 2019:
  <https://openreview.net/forum?id=HJxeWnCcF7>
- Understanding Hyperbolic Metric Learning Through Hard Negative Sampling,
  WACV 2024:
  <https://openaccess.thecvf.com/content/WACV2024/html/Yue_Understanding_Hyperbolic_Metric_Learning_Through_Hard_Negative_Sampling_WACV_2024_paper.html>
- Hyperbolic Vision Transformers, CVPR 2022:
  <https://openaccess.thecvf.com/content/CVPR2022/html/Ermolov_Hyperbolic_Vision_Transformers_Combining_Improvements_in_Metric_Learning_CVPR_2022_paper.html>
- HIER, CVPR 2023:
  <https://openaccess.thecvf.com/content/CVPR2023/html/Kim_HIER_Metric_Learning_Beyond_Class_Labels_via_Hierarchical_Regularization_CVPR_2023_paper.html>
