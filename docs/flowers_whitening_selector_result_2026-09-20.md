# Flowers102 whitening-family selector result

## Outcome

The preregistered four-family selector retained PCA on the fit-only Flowers102
classes. On the disjoint evaluation classes, PCA was within 0.001795 mAP@R of
the numerically best compact arm and kept the highest Recall@1. This is a safe
negative replication: whitening did not earn promotion on this domain, and the
selector avoided an unnecessary quality tradeoff.

This diagnostic is prospective for the whitening-family rule, but it uses a
custom class split of an already available official-test feature archive. It is
therefore `claim_eligible=false` and is not a publication benchmark claim.

## Frozen protocol

- Dataset: Flowers102 official-test frozen UNICOM ViT-L/14@336 features.
- Feature rows/dimensions/classes: 6,149 / 768 / 102.
- Feature SHA-256: `49101a147db72565fa1ebb984a0deb2057d4995383f5ae6e7c3ed5d24e30127b`.
- Split: classes ordered by
  `SHA256("flowers-whitening-selector-v1:" + decimal_label)`; first 51 fit,
  remaining 51 evaluation.
- Fit-only selector: learned replaces PCA only at +0.003 mAP@R with no Recall@1
  loss. Whitening-only or whitening-initialized learning replaces that
  incumbent only at +0.003 mAP@R with at most 0.003 Recall@1 loss.
- Library source: `512f66c82d0bf4119e17fba111e196f52700ece9`.
- Preregistration SHA-256:
  `fe66698acd7daee0073e5b8880fa87d44124b0cc4f86e9ae9347bfa8af996d8b`.

## Fit-only evidence

| int8-128 family | pooled mAP@R | pooled Recall@1 |
|---|---:|---:|
| **PCA (selected)** | **0.967347** | **0.997801** |
| learned | 0.966298 | 0.997801 |
| within-class whitening | 0.965493 | 0.997487 |
| whitening-initialized learned | 0.965052 | 0.997172 |

Neither learned nor whitening cleared its frozen gate, so the sealed decision
was PCA before evaluation-class metrics were computed.

## Disjoint evaluation evidence

| representation | mAP@R | Recall@1 |
|---|---:|---:|
| **selected PCA int8-128** | 0.941168 | **0.997640** |
| learned int8-128 | **0.942962** | 0.997303 |
| within-class whitening int8-128 | 0.942678 | 0.997303 |
| whitening-initialized learned int8-128 | 0.941847 | 0.996628 |
| float768 teacher | 0.952863 | **0.997640** |

The selected PCA arm trails the external best compact arm by 0.001795 mAP@R
and leads it by 0.000337 Recall@1. The observed mAP difference is smaller than
the preregistered minimum useful gain, so the selector's conservative decision
is supported.

## Interpretation

Whitening is not a universal replacement for PCA or supervised learning. Its
value is domain-dependent, while fit-only selection can preserve the best
known safe family without changing the serving representation. Across the
measured panel, the four-family rule now has one prospective method-specific
replication that correctly declines whitening. The default library selector is
unchanged; whitening remains an explicit experimental primitive.

No custom CUDA kernel was used or justified: fitting completed in 35.60 seconds
on the existing GB10, and serving remains the same affine, normalization, and
int8 path.

## Evidence

- Canonical result:
  `docs/evidence/compact_metric/flowers-whitening-selector-v1/result.json`
- Result SHA-256:
  `874418b82f7dcc345794c06f04b37d5bd053c19c41ce3dfffe966f03b501b25d`
- Executed preregistration and script:
  `scripts/_scratch_flowers_whitening_preregistration.json` and
  `scripts/_scratch_flowers_whitening_selector_probe.py`.
