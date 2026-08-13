# Cached foundation adapter result (2026-08-13)

## Outcome

Close the pooled SigLIP2 cached-feature adapter lane. The frozen encoder lost the
official In-Shop comparison and the subsequent hardened experiments show that a
small head cannot recover enough of the missing retrieval geometry. There is no
SOTA claim.

## Official starting point

| model | R@1 | mAP@R | batch-32 p50 | parameters |
|---|---:|---:|---:|---:|
| SigLIP2 base, normalized cosine | 72.9779 | 44.1428 | 190.793 ms | 375.23M |
| identity-disjoint BN-Inception ProxyAnchor | 90.6808 | 62.9518 | 56.988 ms | 11.82M |

The direct frozen foundation encoder is worse on quality and cost.

## Hardened training-identity instrument

The original identity-disjoint validation split saturated near 97% and had
predicted the official result poorly. The replacement uses four evaluation
identity folds and a fifth permanent distractor-identity pool. Fitting excludes
both the evaluated identities and the distractor identities. mAP@R is primary.

| raw cached representation | mean R@1 | mean mAP@R |
|---|---:|---:|
| SigLIP2 | 81.0571 | 49.1657 |
| BN-Inception ProxyAnchor | 98.7069 | 88.8865 |

The 39.72-point mAP@R separation reproduces the known official ordering and
clears the predeclared 8-point instrument-power requirement.

## Fixed-epoch paired adapter result

All arms use the same cached features, P-by-K batches, nested CosFace loss, and
fixed epoch counts. The candidate is a <=5M-parameter residual MLP initialized
as the exact fitted linear map; its nonlinear branch starts at zero. Results are
means across four folds for each of four paired seeds.

| width | mean R@1 gain vs linear | mean mAP@R gain | all four seeds positive |
|---:|---:|---:|:---:|
| 128 | +0.2584 point | +0.6476 point | yes |
| 256 | +0.1846 point | +0.6395 point | yes |
| 512 | +0.3509 point | +0.6548 point | yes |

The residual effect is consistent but fails the registered >=0.50-point mean
R@1 requirement. It is not promoted.

## Recoverability controls

| SigLIP2 transform | mean R@1 | mean mAP@R |
|---|---:|---:|
| raw 768-D | 81.0571 | 49.1657 |
| first 512 coordinates | 81.2566 | 49.3547 |
| train-fold PCA 512-D | 81.5364 | 50.5009 |
| ridge stitch into comparator 512-D geometry | 93.1740 | 67.8937 |
| comparator target | 98.7069 | 88.8865 |

The ridge map is fitted only on each fold's optimization identities. Its best
fixed-grid regularization is 0.1. Although it recovers substantial R@1, it
remains 20.99 mAP@R points below the comparator, failing the <=8-point
recoverability requirement. This is evidence that the pooled SigLIP2 vector is
the wrong anchor for this task, not that a larger adapter is needed.

## Reproduction

The implementation is in commits `60a79cf`, `a3f0f18`, `d940f61`, `515f345`,
`05255f9`, `921c4b8`, and `38ced09`. The executable entry points are:

```bash
python scripts/evaluate_foundation_hardened_folds.py --cache CACHE
python scripts/run_foundation_adapter_screen.py --train-cache CACHE \
  --model linear --seed SEED --epochs 20 --hardened-fold FOLD \
  --output RESULT.json --checkpoint MODEL.pt
python scripts/run_foundation_adapter_screen.py --train-cache CACHE \
  --model mlp --seed SEED --epochs 10 --learning-rate 0.0001 \
  --hardened-fold FOLD --initialize-from-linear MODEL.pt \
  --output RESULT.json --checkpoint WARM.pt
python scripts/evaluate_foundation_ridge_stitch.py \
  --source-cache SIGLIP_CACHE --target-cache COMPARATOR_CACHE \
  --regularization 0.1
```

Cached-feature training took approximately 6--11 seconds per arm on the GB10.
No custom kernel is justified: adapter compute is already negligible and the
quality bottleneck is the source representation.

## Next experiment

Export and screen retrieval-native frozen anchors (UNICOM B/16 first, then
MLCD-B and DINOv3 if available) with the same hardened instrument. Only an
anchor that materially narrows the raw/comparator mAP@R gap enters adapter
training. Hyperbolic geometry and custom kernels remain closed until evidence
shows a geometry or time-to-quality bottleneck rather than a representation
bottleneck.
