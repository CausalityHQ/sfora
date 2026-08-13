# Cached foundation adapter result (2026-08-13)

## Current outcome

The frozen SigLIP2 representation is not competitive by itself, but train-only
ridge and CosFace maps recover most of the official quality gap. Neither is yet
a Pareto winner: both exceed the comparator's official mAP@R, while the stronger
CosFace head remains 1.78 Recall@1 points below it. A validation-selected
ridge/CosFace fusion is the strongest candidate at 89.84 R@1 / 66.13 mAP@R,
but still misses the comparator by 0.84 R@1 point. There is no SOTA claim.

## Official results

| representation | R@1 | mAP@R |
|---|---:|---:|
| raw SigLIP2 cosine | 72.9779 | 44.1428 |
| train-only SigLIP2-to-ProxyAnchor ridge stitch | 88.3598 | 62.9783 |
| train-only nested CosFace, 256-D | 88.5990 | 64.9540 |
| train-only nested CosFace, 512-D | 88.9014 | 65.3007 |
| validation-selected ridge + one CosFace score fusion, 1024-D | 89.8368 | 66.1339 |
| validation-selected ridge + four-CosFace ensemble, 2560-D | 89.7806 | 66.7463 |
| identity-disjoint BN-Inception ProxyAnchor | 90.6808 | 62.9518 |
| released UNICOM ViT-B/16 zero-shot | 75.4959 | 47.6457 |

The ridge map uses 20,638 optimization rows from the original 80% training-
identity split. Its fixed regularization is 0.1. Query and gallery pixels were
not reread: the selected map was applied to the existing cached official
SigLIP2 embeddings.

## Corrected unseen-identity instrument

An initial four-fold instrument incorrectly repartitioned all training
identities even though the comparator had already trained on 80% of them. That
inflated the comparator ceiling and invalidated the first reported calibration,
adapter deltas, and ridge-kill conclusion. The issue was found during the
adversarial review and fixed in `b52c08e`: optimization is now exactly the
comparator's original 80% identity split, while every evaluation fold and the
permanent distractor pool come only from its original unseen 20%.

| raw cached representation | mean R@1 | mean mAP@R |
|---|---:|---:|
| SigLIP2 | 90.6679 | 65.4291 |
| BN-Inception ProxyAnchor | 97.6142 | 80.3759 |
| ridge-stitched SigLIP2 | 97.2018 | 81.3905 |

The corrected instrument reproduces the official ordering of the raw arms. On
unseen training identities, the ridge stitch matches the comparator within
0.41 R@1 point and exceeds it by 1.01 mAP@R point. The corresponding official
result shows that this recovery mostly transfers, but not enough at R@1.

## Superseded residual screen

The earlier four-seed residual-versus-linear numbers are not used. Besides the
identity contamination above, the residual received ten additional epochs
after the linear checkpoint. A continued-linear control is now implemented in
`350006b`; any future residual comparison must use that matched control and the
corrected unseen-identity folds.

## Corrected four-seed screens

The fixed-20-epoch linear CosFace head gains 7.00 R@1 points and 18.73 mAP@R
points over raw SigLIP2 on average. All four optimizer/sampler seeds are
positive. The 1.31M-parameter nonlinear ridge correction gains 1.23 mAP@R
points over ridge but only 0.10 R@1 point; one of four seeds is negative on
R@1, so it is closed without official evaluation.

## Efficiency

Cached-feature fitting takes seconds per arm on the GB10, and the ridge map is a
single 768-by-512 matrix plus two means. It adds about 1.57 MB in FP32 and one
small GEMM after the unchanged SigLIP2 encoder. A custom kernel is not justified:
the bottleneck remains encoder inference and quality, not adapter training.

The one-head fusion was selected from weights `{0, .25, .5, .75, 1}` using
only the corrected unseen-identity folds. It selected .25 CosFace / .75 ridge,
raising corrected-fold R@1 from 97.20 ridge and 97.67 CosFace to 97.91. The
official result was evaluated once at that weight. The four-seed ensemble
selected .5 on the same folds but slightly reduced official R@1 and expands the
descriptor to 2,560 dimensions, so it is closed.

## Reproduction

Core implementation commits are `60a79cf`, `a3f0f18`, `d940f61`, `515f345`,
`05255f9`, `921c4b8`, `38ced09`, `b52c08e`, `350006b`, and `58ec7f8`.

```bash
python scripts/evaluate_foundation_hardened_folds.py --cache CACHE
python scripts/evaluate_foundation_ridge_stitch.py \
  --source-cache SIGLIP_TRAIN --target-cache PROXYANCHOR_TRAIN \
  --regularization 0.1
python scripts/evaluate_foundation_stitch_official.py \
  --source-train SIGLIP_TRAIN --target-train PROXYANCHOR_TRAIN \
  --source-query SIGLIP_QUERY --source-gallery SIGLIP_GALLERY \
  --regularization 0.1
python scripts/evaluate_foundation_score_fusion.py --mode screen \
  --source-train SIGLIP_TRAIN --target-train PROXYANCHOR_TRAIN \
  --checkpoint LINEAR_SEED0 --checkpoint LINEAR_SEED1 \
  --checkpoint LINEAR_SEED2 --checkpoint LINEAR_SEED3 --output FUSION_SCREEN
python scripts/evaluate_foundation_score_fusion.py --mode official \
  --source-train SIGLIP_TRAIN --target-train PROXYANCHOR_TRAIN \
  --source-query SIGLIP_QUERY --source-gallery SIGLIP_GALLERY \
  --checkpoint LINEAR_SEED0 --weight 0.25 --output FUSION_OFFICIAL
```

## Next experiment

Cached heads are now closed: neither selected fusion closes the R@1 gap, and
the ensemble worsens storage and inference. Move to image-level training with a
compact or partially unfrozen encoder, using the successful ridge/CosFace
geometry as teacher supervision and ProxyAnchor as the matched baseline.
UNICOM remains an alternate retrieval-native anchor, but its 75.50% zero-shot
R@1 does not by itself justify expensive full-image training.
