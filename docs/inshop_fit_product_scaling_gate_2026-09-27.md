# In-Shop fit-product scaling screen, 27 September 2026

This is a TRAIN-only causal screen for the effect of the number of seen
products on unseen-product packed retrieval. The original product-disjoint
split has 13,283 fit images from 2,004 products and 12,599 held images from
1,993 products. The treatment keeps a SHA-selected nested half of fit products,
with all held rows unchanged. No official TEST row selects the treatment.

Use the existing seed-179024 `freeze_emb` 1,000-update full-fit receipt
(`fb1c41d341d738676ff12c72f14b5eee66fd9d633fef9b9306c1ffd080908e89`)
as the comparator: packed held-only symmetric R@1 98.5475%, mAP@R 0.839099,
training wall 747.058 s and peak allocated CUDA 12.939 GB on DGX Spark GB10.
The half-fit arm uses the same authenticated SigLIP2 Large/256, cached
pretrained source features, ArcFace plus SmoothAP bank recipe, 1,000 updates ×
64 images, BF16, optimizer and 128-D int8 packed scorer. The initial PCA head,
classifier, bank and batch schedule are recomputed from the selected half, so
the arms share a seed and protocol but **not identical batches**. This screen
does not estimate a paired training-noise effect. It varies both product count
and image diversity; at a fixed update budget, the remaining images receive
more repeat exposure. A surviving result supports a broader-data recipe,
not a pure class-count mechanism.

Before reading treatment quality, freeze this decision:

1. A 17-update half-fit smoke must pass all authority, split, schedule, finite
   loss and checkpoint gates. It has no quality read.
2. The 1,000-update half-fit run must finish with finite loss and a verified
   receipt/checkpoint. Compare both arms on exactly the same 12,599 held rows
   with the same packed scorer. If full-fit R@1 exceeds half-fit R@1 by at
   least **0.4 percentage points**, and a product-cluster bootstrap 95% lower
   bound on the difference is above zero, the seen-product scaling hypothesis
   survives. Otherwise stop before an all-TRAIN retrain. mAP@R, training wall,
   throughput and peak VRAM are recorded, not used to tune this threshold.
3. A surviving screen only authorizes a new full-TRAIN recipe paired with a
   matched fit-only baseline and an explicitly exploratory official TEST read.
   It does not establish a novel learning method, a serving speedup, or a
   published SOTA result. The all-TRAIN model must independently pass quality
   and measured image-to-top-k latency gates before production promotion.

The source is [the trainer](../scripts/train_inshop_siglip2_unseen_gallery.py)
at SHA-256 `aea6718f4b03e0bbc326f4f9d6547ea20f9da778f336fdb3cae69fbf1f756e28`.

## Smoke checkpoint

The sole DGX Spark smoke unit (`sfora-inshop-fit-scaling-half-179024-smoke-v1`)
ended with exit 0. It selected 6,764 fit images, kept all 12,599 held images,
completed 17 finite updates (15 rank-active), and wrote a checkpoint whose
hash matches its [receipt](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-fit-product-scaling-half-179024/smoke.json)
SHA-256 `fbb3accaa735e49b51ff503808b84b6d0ab484d33b8bd604c01161ab11e2b56a`.
The [journal](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-fit-product-scaling-half-179024/smoke.journal.log)
SHA-256 is `4a105567b3dd0bf0de21d479de46da3908003311d0e5e01b22e17d643d41c8f4`.
This has no retrieval-quality read. The subsequent single 1,000-update run
completed as `sfora-inshop-fit-scaling-half-179024-full-v1` (invocation
`6ada418d04994abaa739d6360bd96ba5`).

## Full-run decision

The sole full run completed 1,000 finite updates, held-image export and packed
scoring with service exit 0. The [source-bound decision](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-fit-product-scaling-half-179024/decision.json)
compares the same 12,599 In-Shop official **TRAIN** held-only symmetric queries
and gallery, with 1,993 held products:

| Seed 179024 arm | Fit products / images | Packed R@1 | mAP@R | Training wall, 64,000 images | Images/s | Peak allocated CUDA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Archived full fit | 2,004 / 13,283 | 98.5475% | 0.839099 | 747.058 s | 85.67 | 12.939 GB |
| Nested half fit | 1,002 / 6,764 | 98.1824% | 0.819225 | 743.215 s | 86.11 | 11.851 GB |

Full minus half is **+0.3651 percentage points** packed R@1; the paired
product-cluster bootstrap 95% interval on these held queries is
**[+0.1535, +0.5740] pp**. The interval clears zero, but the point misses
the frozen **+0.4 pp** advance floor. `advance_full_train=false`: stop before
an all-TRAIN retrain or another official TEST read. More fit data helped this
one held panel, but the result does not establish enough gain to close the
official-quality gap, and neither arm changes public serving speed.

The half-fit [receipt](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-fit-product-scaling-half-179024/full.json)
SHA-256 is `f07f8ccbf4163e6c711946196c74feed988d8b2805d37f1a90143b9f9e7bd631`;
its [terminal journal](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-fit-product-scaling-half-179024/full.journal.log)
SHA-256 is `cbf80d78aef42bb44cf0a17c6b22857860437ab001f705ac7414da2bf4483385`.
The [decision](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-fit-product-scaling-half-179024/decision.json)
SHA-256 is `02eefd3e72a159ece90ac5075d3d27c6b5b75703129e945736469442b0eb99e1`.
