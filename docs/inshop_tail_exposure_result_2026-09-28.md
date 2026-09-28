# Tail exposure diagnostic: identify the sampling layer

This read-only check replayed the native seed-179026 TRAIN-fit schedule exactly
against its original receipt hash. It read only partition metadata and prior
fit-tail margins: no images, feature exports, training or new quality scoring.
The corrected calculation took **0.356 seconds**; process wall including
imports was 2.756 seconds. Two bookkeeping failures preceded the result and
are preserved with their fixes; the frozen criterion was unchanged.

The prior trained-fit diagnostic identified **2,506 tail anchors** among
13,271 eligible fit images: best negative exceeds worst positive by >0.03.
Counts below include only batches where native SmoothAP ranking was active.

| Native schedule window | Rank-active updates | Tail normalized exposure within product | Other normalized exposure | Tail mean raw appearances | Other mean raw appearances |
| --- | ---: | ---: | ---: | ---: | ---: |
| First 100 updates; visited products only | 89 | 1.0245× | 0.9949× | 0.2438 | 0.6812 |
| All 1,000 updates | 923 | 0.9950× | 1.0012× | 2.1800 | 4.9799 |

**Reject systematic within-product tail underexposure:** its full-budget ratio
is 0.995, above the frozen ≤0.8 advance threshold. This does not rule out every
hardness-aware sampler. It removes that particular explanation for changing
the order of images within products. At 1,000 updates, 0.918% of tail anchors
had no rank-active appearance.

## One next hypothesis

The substantial difference is absolute exposure across products. In the actual
sampler, coverage-first initializes its queue once. After exhaustion, remaining
batches choose identities uniformly, rather than starting another coverage
pass. Within-product normalized exposures near one, alongside raw tail
exposure 2.18 versus 4.98, are consistent with product-level image weighting
contributing to the difference. This is an association with a retrospective
trained-fit tail, not proof of what caused poor quality.

Test **repeated coverage passes** at the same total image budget, preserving
the initial coverage pass and native losses, architecture and scorer. Known
sampling technique; no novelty or quality claim. First use a bounded CPU
schedule preflight to require a material increase in tail exposure without
invalid labels, missing rows or broken identity batches. Then, only if it
passes, freeze an internal TRAIN-fit-only cached-feature paired quality smoke
whose step budget reaches the queue-exhaustion boundary. A 17-step smoke from
initialization would not exercise the changed branch. A quality failure stops
this fixed arm; an exposure increase alone cannot justify a DGX training run.

Production defaults remain unchanged. No new retrieval quality, image-to-top-k
latency, QPS, training throughput or GPU VRAM was measured by this diagnostic.
The prior worst-positive loss and category-pooling proxy remain closed.

Raw freeze, receipt, replay source and failure records are in
[the evidence archive](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-tail-exposure-diagnostic-v1/).
