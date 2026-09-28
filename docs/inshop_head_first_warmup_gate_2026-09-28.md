# Head-first gradient-pressure gate

One mechanism: allowing the PCA head/classifier to adapt **before** encoder
updates could reduce early encoder pressure from a poorly adapted downstream
classifier. This is a training-phase change, not an auxiliary source loss or
cached lower-layer training. It leaves the deployed architecture and scorer
unchanged and makes no novelty claim.

First run one <=120-second CPU screen on the authenticated pretrained TRAIN
cache, fit-only 13,283 rows/2,004 products, seed179024 and original schedule.
Warm only the original 128-D head/classifier for **100** cached batches using
the unchanged native ArcFace margin0.3/scale64, AdamW1e-4/decay0.05 and global
clip1. Source vectors are detached. These cached vectors have no stochastic
image augmentation, so this is a mechanism falsifier rather than a faithful
full-training comparison. Warm-up does not run the image bank objective;
the subsequent image phase would retain original ArcFace+8xSmoothAP.

Before and after warm-up, rebuild the same fit member bank in each head's
geometry. On the SAME first17 cached batches, measure the complete image-phase
objective's gradient at each raw source vector, with the original singleton
rank skips. This raw-pooler gradient is a proxy for encoder pressure, not
an encoder-gradient or clipping claim. Freeze advance rules:

1. Median paired warm/initial source-gradient norm ratio <=0.75.
2. Probe mean native ArcFace loss falls at least20%, with no nonfinite loss,
   gradient, parameter or zero/undefined source-gradient statistic.
3. Normalized compact outputs on those same rows retain >=80% of initial
   centered variance and participation effective rank.
4. Partition/cache/fit/PCA/schedule authority and 100 cached-step/all17 probe
   inventories match, total CPU wall<=120s.

Any failure kills this fixed100-step head-first configuration without step
count/rate/margin search. Passing only permits a frozen <=2min GPU gradient/
cost smoke; it does not automatically authorize100/1000 image updates. A
later matched-seed TRAIN gate must account for6400 extra cached exposures,
warm-up wall and memory, full training cost, selection uncertainty and a new
independent holdout. No official labels or scores are read by this screen.

## Terminal CPU decision: KILL

The single original process exited0. Frozen protocol/code was pushed at
`cf1aa595` before the receipt was read. Receipt
`docs/evidence/compact_metric/sop-siglip2-substrate-v1/inshop-head-first-cache-v1.json`
SHA-256 `1fcdb6d3a7474b8ee48aa2d623e45c42e8c3a1520ded3c94480c75f297a568ed`.

| Official In-Shop TRAIN fit only | Measured cached result | Decision |
|---|---:|---|
| Same17 batches/1,088 nominal raw-source probes, full objective | Median warm/initial gradient norm0.9880844; only1.19156% reduction | KILL: needs<=0.75 |
| Same probes' native ArcFace | Mean loss ratio0.8590362;14.09638% reduction | KILL: needs<=0.8 |
| Compact centered variance | Warm/initial0.9911187 | Pass collapse guard |
| Compact participation rank | Warm/initial1.0877785 | Pass collapse guard |
|100 cached head-only steps/6,400 exposures plus diagnostics | CPU main wall7.112750s | Pass<=120s; no encoder training cost inferred |
| R@1/mAP@R, encoder VRAM, image-to-top-k p50/p95/p99/QPS | Not measured | No quality/performance claim |

Independent receipt replay verified100 losses,17 native-loss entries and
1,088 source norms per arm, exact paired ratios, all criteria and decision.
The runnable check passed: warm-up changed downstream weights while leaving
the source without gradients. Ruff passes. The original fit/PCA/cache/
partition/schedule hashes match authority. No GPU job, official read,
production edit or package rebuild was performed. Increasing warm-up steps
or rate after this read is not permitted for this fixed configuration.
The result rejects the predicted large pressure relief; it does not establish
that all head-first training algorithms are ineffective. Goal remains active.
