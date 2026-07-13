# Group Learning Image Retrieval Report

## Abstract

This report evaluates **Group SupCon + XBM + Radius** for image retrieval with frozen vision backbones. The method trains a lightweight projection head while keeping the base encoder fixed. The research question is whether adding group-centroid contrast, cross-batch memory, and radius control improves retrieval quality over the best same-backbone non-proposed method.

The primary metrics are **MAP@R** and **Recall@1** on held-out image retrieval queries. Objective loss is not used to compare methods because each objective optimizes a different loss surface.

## Research Question

Can group-aware supervised contrastive training improve frozen image embeddings for retrieval without changing the base image model?

The proposed method is evaluated against frozen encoders, triplet losses, batch-hard triplet, supervised contrastive learning, proxy methods, angular-margin methods, hybrid group objectives, XBM variants, and radius variants.

## Proposed Method

**Group SupCon + XBM + Radius** combines four ingredients:

| Component | Purpose |
| --- | --- |
| Point SupCon | Keeps dense same-class example supervision. |
| Group centroids | Adds same-class representative units so groups, not only individual examples, shape the space. |
| XBM memory | Adds recent embeddings to expose harder negatives than one mini-batch provides. |
| Radius control | Keeps class neighborhoods compact enough for retrieval reuse. |

Raw Group SupCon is treated as an ablation. The headline claim is the full method.

## Experimental Protocol

Each image dataset is split into train and held-out test examples. Frozen backbones produce image embeddings. Projection heads are trained only on train features, then evaluated on held-out retrieval queries.

| Dataset | Images | Train | Held-out Test | Retrieval Queries |
| --- | ---: | ---: | ---: | ---: |
| CUB | 11,788 | 5,864 | 5,924 | 5,924 |
| Cars196 | 16,185 | 8,054 | 8,131 | 8,131 |
| Stanford Online Products | 50,654 | 25,361 | 25,293 | 25,293 |

Backbones:

- `facebook/dinov2-small`
- `openai/clip-vit-base-patch32`
- `google/siglip-base-patch16-224`

## Primary Results

The proposed method wins against the best same-backbone non-proposed method on all three image datasets.

| Dataset | Backbone | Best Prior | Prior MAP@R | Proposed MAP@R | Result Gain | Recall@1 Delta |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| CUB | `openai/clip-vit-base-patch32` | Supervised Contrastive | 0.1893 | 0.2121 | +12.1% | +0.0405 |
| Cars196 | `facebook/dinov2-small` | Supervised Contrastive | 0.1702 | 0.2208 | +29.7% | +0.0786 |
| Stanford Online Products | `google/siglip-base-patch16-224` | Supervised Contrastive | 0.3764 | 0.4082 | +8.5% | +0.0408 |

Result gain is computed as:

```text
(ours MAP@R - previous MAP@R) / previous MAP@R
```

## Ablation Reading

The ablation separates the core group idea from the complete method.

| Method Family | Interpretation |
| --- | --- |
| Raw SupCon | Strong external point-level supervised contrastive baseline. |
| Group SupCon | Tests whether same-class group representatives help by themselves. |
| Group SupCon + XBM + Radius | Main method: group representatives plus memory-backed negatives and compactness pressure. |

The results support the full method rather than raw grouping alone. Group representatives are most useful when they are combined with dense point supervision, harder negatives, and radius control.

## Secondary Text Transfer Context

IMDb transfer results are retained only as secondary evidence for downstream reuse of embedding spaces. They are not the primary claim. The image retrieval benchmark is the research target for this report.

## Artifacts

| Artifact | Path |
| --- | --- |
| CUB image retrieval | `reports/generated/image_retrieval_cub.json` |
| Cars196 image retrieval | `reports/generated/image_retrieval_cars.json` |
| Stanford Online Products image retrieval | `reports/generated/image_retrieval_sop.json` |
| Interactive Astro report | `reports/site/index.html` |

## Interpretation

The evidence supports **Group SupCon + XBM + Radius** as the strongest tested projection-head objective across the current image retrieval benchmarks. The effect is largest on Cars196, where the proposed method improves MAP@R from 0.1702 to 0.2208 against the best same-backbone prior, a +29.7% result gain.

The next research step is replication across seeds and tuning budgets for the winning image settings. The public claim should remain tied to held-out MAP@R and Recall@1, not to training loss.
