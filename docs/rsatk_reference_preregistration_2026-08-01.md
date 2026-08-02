# Recall@k-surrogate reference reproduction preregistration (2026-08-01)

## Status and purpose

This document was committed before the first GPU execution of the native port.
The experiment is a **strong-baseline reproduction**, not a novel-method claim.
It tests whether this repository can faithfully reproduce the published
Recall@k Surrogate (RS@k) result before judging new mechanisms against an older
Proxy Anchor/HIST baseline.

Primary implementation source: Patel, Tolias, and Matas, *Recall@k Surrogate
Loss with Large Batches and Similarity Mixup* (CVPR 2022), official repository
revision `ed052029d258555df2f94dd82d6f7df60ef7cc6f`.

## Locked first experiment

- Dataset: Cars196 official train/test class partition.
- Method: exact RS@k surrogate **without optional Similarity Mixup**, matching
  the paper's separately reported no-SiMix row.
- Backbone/output: ImageNet ResNet-50, learnable GeM pooling, LayerNorm on the
  pooled 2048-dimensional feature, 512-dimensional linear head, L2-normalized
  retrieval embedding.
- Sampling: balanced batch 392, four images per class.
- Loss: k in `{1, 2, 4, 8, 16}`, rank temperature 0.01, membership temperature
  1.0.
- Optimisation: Adam, learning rate `1e-4`, weight decay `4e-4`, 170 epochs,
  learning-rate multiplier 0.3 at epochs 80 and 140; frozen BatchNorm running
  statistics.
- Augmentation/evaluation: random resized 224 crop plus horizontal flip for
  training; resize 256 and center crop 224 for evaluation.

The native implementation uses direct full-batch autograd if the exact batch
fits. This is mathematically the same objective and gradient as the source
repository's two-pass replay, whose purpose is activation-memory reduction. If
batch 392 does not fit, replay will be implemented; lowering the batch size is
not an acceptable faithful-reproduction workaround.

## Predictions and falsification

The official no-SiMix Cars196 row reports raw R@1 = **0.807** (with R@2 0.883,
R@4 0.928, and R@8 0.957). Before observing this port:

1. A one-step exact-batch smoke run should complete with finite loss and finite
   gradients. OOM or non-finite gradients fails implementation readiness and
   triggers replay/debugging, not a changed recipe.
2. The completed native run is predicted to achieve raw best-over-training
   R@1 in **[0.787, 0.827]**. Raw best R@1 below **0.780** falsifies faithful
   reproduction and bars use of the result as a stronger baseline until the
   discrepancy is explained.
3. No novelty or superiority claim follows from matching the published number.
   Any subsequent candidate must be compared against this reproduced baseline
   under matched evaluation.

## Reporting rule

Report the complete R@k curve and both raw best-over-training and
selection-corrected R@1 whenever the available evaluation trajectory supports
the repository's selection-bias estimator. Do not silently substitute the
final checkpoint for either quantity. Record deviations from the pinned source
recipe and do not use single-run variance language.

## First full run invalidated before completion (2026-08-02)

The first full Cars run was stopped at epoch 52 and is excluded from every
result. It had reached raw best R@1 **0.7664** at epoch 51, but a direct audit
against pinned authors' `src/losses.py` found that the native port omitted
their `min(soft_retrieved_count, k)` operation before normalisation. With four
examples per class, several positive membership values can sum above k; without
the cap surrogate recall can exceed one and the loss rewards an impossible
negative-loss regime. The existing two-example-per-class unit test could not
exercise this because every query had only one positive.

The correction was made before a deciding artifact existed. A new
four-example-per-class test now proves the cap: perfectly separated classes at
k=1 otherwise produce soft count 1.5, while the pinned source and corrected
port both return recall 1 and loss 0. The numerical prediction and falsification
threshold above remain unchanged; the corrected rerun starts from scratch at
seed 0. The partial 0.7664 trajectory is implementation-debug evidence only and
may not be pooled, selected, or quoted as RS@k performance.
