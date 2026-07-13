---
library_name: sentence-transformers
license: apache-2.0
tags:
- metric-learning
- image-retrieval
- supervised-contrastive-learning
- representation-learning
---

# sfora

This repository contains reproducible code and reports for **Group SupCon + XBM + Radius**, a group-aware metric-learning objective for improving frozen image embeddings with a lightweight projection head.

## Summary

The primary report evaluates image retrieval on CUB, Cars196, and Stanford Online Products. The proposed method beats the best same-backbone non-proposed method on all three datasets.

| Dataset | Backbone | Best Prior | Prior MAP@R | Proposed MAP@R | Result Gain |
| --- | --- | --- | ---: | ---: | ---: |
| CUB | `openai/clip-vit-base-patch32` | Supervised Contrastive | 0.1893 | 0.2121 | +12.1% |
| Cars196 | `facebook/dinov2-small` | Supervised Contrastive | 0.1702 | 0.2208 | +29.7% |
| Stanford Online Products | `google/siglip-base-patch16-224` | Supervised Contrastive | 0.3764 | 0.4082 | +8.5% |

Result gain is computed from actual MAP@R values:

```text
(ours MAP@R - previous MAP@R) / previous MAP@R
```

## Method

**Group SupCon + XBM + Radius** trains a projection head on frozen image features. It combines:

- point-level supervised contrastive learning,
- same-class group-centroid contrast,
- cross-batch memory for harder negatives,
- radius control for compact class neighborhoods.

Raw Group SupCon is an ablation. The headline method is the full recipe.

## Evaluation

The image benchmark uses held-out retrieval:

- **Recall@1** measures nearest-neighbor correctness.
- **MAP@R** measures ranked retrieval quality over relevant items.
- Training loss is not used as a cross-method quality metric.

## Artifacts

- Interactive report: `reports/site/index.html`
- Markdown report: `reports/REPORT.md`
- CUB results: `reports/generated/image_retrieval_cub.json`
- Cars196 results: `reports/generated/image_retrieval_cars.json`
- Stanford Online Products results: `reports/generated/image_retrieval_sop.json`

IMDb transfer experiments are included only as secondary downstream context; the primary research claim is image retrieval quality.
