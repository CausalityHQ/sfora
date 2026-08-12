# Signature-Matched Product Geometry SOTA Gate

**Status:** prospective zero-training falsifier. Curved training is forbidden
unless every gate in this document passes.

## Objective and SOTA boundary

The objective is not to demonstrate that non-Euclidean geometry can compress a
descriptor. It is to decide cheaply whether curvature has a credible,
method-attributable path beyond the supervised UNICOM ViT-B/16 In-Shop point
of 95.5 Recall@1 at 512 stored dimensions.

The published absolute In-Shop reference is 96.7 for UNICOM ViT-L/14 at 336
pixels. That is a larger backbone, larger input, and eight-GPU lane. Therefore:

- a result above a locally reproduced 95.5 with the same ViT-B/16 backbone,
  pretraining, input, 512-D descriptor, evaluator, and training budget is a
  matched frontier improvement;
- an absolute descriptor-only SOTA claim additionally requires exceeding the
  strongest verified contemporary comparable point, including 96.7 when
  backbone/compute are not constrained;
- a quality/efficiency Pareto claim may instead match the strongest quality
  within a preregistered equivalence band while materially lowering training,
  inference, or storage cost;
- larger models, extra data, reranking, ensembles, and test-selected
  checkpoints are separate lanes and cannot be credited to the method.

## Evidence and collision boundary

Plain Poincare, Lorentz, and fixed product-manifold heads are not novel. The
closest direct collision is Xu et al., *Mixed-Curvature Metric Learning for
Image Retrieval*, IEEE TMM 2026, which combines positive-, zero-, and
negative-curvature spaces and evaluates image retrieval. Earlier occupied
families include mixed-curvature product spaces, kappa-stereographic networks,
Hyp-ViT, HIER, MERU, AMCAD, and hyperbolic prototype classifiers.

Published In-Shop curved-geometry systems audited in this repository remain
well below the modern frontier: Hyp-ViT/Hyp-DINO are about 92.6--92.7 and HIER
about 92.4 in different backbone/dimension lanes. Yue et al. further show that
hyperbolic metric-learning improvements can be explained by implicit
hard-negative weighting. The repository's Lorentz scorer also reduces exactly
to query-conditioned norm-weighted cosine. Consequently, temperature,
hard-negative, spatial-only, and power-law controls are load-bearing.

## Alternatives and decision

Three approaches are considered:

1. **Run a mixed-curvature training head immediately.** This has the highest
   cost, collides with prior art, and lacks evidence that geometry explains
   modern UNICOM errors. Rejected before the falsifiers.
2. **Run zero-training geometry falsifiers on the immutable UNICOM export.**
   This is selected. It costs no GPU, can rule out the lane, and measures the
   exact assumptions needed by a curved head.
3. **Ignore geometry and repair the modern Euclidean anchor.** This remains the
   default SOTA path: reproduce UNICOM, measure its fixed-prefix evaluator and
   distributed-mask defects, compare full-width/synchronized controls, and
   optimize only a measured bottleneck. It proceeds regardless of the geometry
   result.

## Steelman candidate, conditional only

If all zero-training gates pass, the strongest surviving candidate is a
signature-matched kappa-stereographic product head inside an otherwise matched
UNICOM ViT-B/16 fine-tune.

Split the 512-dimensional descriptor into factors with dimensions `d_f` and
`sum_f d_f = 512`. For factor `f`, let `v_f = W_f h`, learn

```text
kappa_f = 4 tanh(theta_f),
z_f = exp_0^{kappa_f}(v_f),
```

and use the kappa-stereographic distance `d_{kappa_f}`. The product distance is

```text
D(z, p_c)^2 = sum_f alpha_f d_{kappa_f}(z_f, p_{c,f})^2,
alpha_f = softmax(a)_f.
```

The classifier logit is `-s * (D(z,p_c)^2 + m * 1[c=y])`, with the original
class set, PartialFC sharding, optimizer, sampling, and schedule retained.
Each `kappa_f` is initialized from a train-only triangle-comparison signature;
`kappa_f -> 0` must recover the Euclidean implementation numerically.

This candidate is not claimed novel by its ingredients. Its only potentially
defensible contribution would be a measured-signature initialization and an
exact matched transfer to the modern UNICOM frontier, if it uniquely beats all
controls.

## Frozen no-training falsifiers

All falsifiers consume the same authenticated, immutable 768-D UNICOM train,
query, and gallery export already queued. No model weights or test labels are
used to fit parameters.

### F0: curvature signature

Reuse the registered ten fixed 2,000-row train subsamples. Measure relative
delta-hyperbolicity with its column-permutation and spectrum-matched Gaussian
nulls. On the same rows, measure the triangle-comparison statistic

```text
xi(a;b,c) = d(a,m)^2 + d(b,c)^2 / 4
            - (d(a,b)^2 + d(a,c)^2) / 2,
```

where `m` is the `b,c` midpoint. Negative `xi` indicates hyperbolic tendency;
positive `xi` indicates spherical tendency. Persist replicate values and
train-identity clustered intervals.

### F1: residual-error ceiling

Using only the frozen official-512 retrieval result, partition incorrect
queries into coarse-category-crossing and within-category errors from the
official In-Shop paths. This is a generous upper bound, not an estimate of
achievable gain: hierarchy-shaped geometry cannot plausibly repair more errors
than exist in its relevant stratum.

### F2: product-family no-training scorer

Fit PCA on train embeddings only. Evaluate registered block factorizations
`F in {1,2,4,8}` and a fixed curvature/scale grid. Rank with the summed
factorwise score. The paired bootstrap must recompute the maximum over all
interior settings inside each replicate.

Every candidate is compared against, at identical stored width:

- PCA Euclidean and PCA cosine endpoints;
- the existing spatial-only Lorentz control;
- fixed power-law norm scorers with powers one and three;
- a Euclidean product scorer with independently fitted factor temperatures;
- a hard-negative-weight-equivalent calibration control when labels are used
  in a later training smoke.

The scorer and all controls must use maintained matrix multiplication plus
elementwise transforms. A custom retrieval kernel is forbidden at this stage.

## Frozen go/no-go rules

Curved training is authorized only if all conditions hold:

1. `F0`: mean relative delta-hyperbolicity is at most 0.30, is separated from
   both nulls, and the clustered interval for the triangle signature excludes
   zero. A low delta alone is insufficient.
2. `F1`: at least 33% of the reproduced baseline's residual errors cross the
   registered coarse-category boundary. This supplies at least a 1.5-point
   gross ceiling for a required 0.5-point net effect.
3. `F2`: the best interior product geometry improves Recall@1 by at least 0.30
   point over the best endpoint, has a positive 95% clustered-bootstrap lower
   bound, and beats every spatial, power-law, and temperature-matched control.
4. The direct MCML paper is recovered and checked. A matching method/result
   removes novelty; a strong modern result becomes a mandatory baseline.

Failure of any rule closes curved training. Thresholds, partitions,
factorizations, and curvature grids are not changed after viewing outcomes.

## Conditional smoke and frontier experiment

Only after F0--F2 pass, run one six-epoch paired smoke from the same UNICOM
initialization:

- official Euclidean head;
- temperature/hard-negative-matched Euclidean product twin;
- signature-matched curved product head.

The curved arm proceeds only if its label-disjoint validation Recall@1 exceeds
the twin by at least 0.30 point, fitted curvatures remain bounded away from
zero, all values remain finite, and step time is no more than 1.10x the twin.

A matched frontier claim then requires a local faithful 95.5 reproduction and
six paired final-checkpoint seeds. Mean curved gain over every control must be
at least 0.50 Recall@1 point with a one-sided 95% paired lower bound above zero.
R@10 and mAP@R may not regress by more than 0.10 point. Direction must replicate
on at least two of CUB, Cars196, and SOP.

An absolute SOTA claim requires the final system to exceed the strongest
verified descriptor-only point in its declared compute/backbone lane. A Pareto
claim instead requires a preregistered quality-equivalence interval plus at
least 20% end-to-end training or inference improvement, or a materially
smaller descriptor at matched quality.

## Systems path

Product distances are sums of factor scores. Hyperbolic factors use the
Lorentz sign-flip MIPS representation, spherical factors use inner products,
and flat factors use the standard augmented-MIPS reduction. Concatenating the
weighted factor representations yields one ordinary GEMM/FAISS inner-product
search. Geometry-specific operations stay in a compiled elementwise projection
head; a native kernel is considered only after profiling shows an unfused
operation consumes at least 10% of runtime.

## Failure handling and interpretation

Nonfinite values, unstable curvature, a requirement for FP64 model execution,
failed baseline reproduction, or a result explained by a matched Euclidean
control closes the geometry lane. A positive frozen-embedding result is only a
mechanism premise. A positive smoke is only a training premise. Neither is
reported as SOTA.

The expected outcome is closure at F2. That is useful: it prevents an expensive
curved training campaign and concentrates the SOTA budget on the reproducible
modern baseline, measured defects, and maintained-kernel performance work.

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
