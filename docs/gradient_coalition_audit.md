# In-Shop same-class gradient-coalition audit

**CPU-only measurement recorded 2026-07-31.** This uses the exact epoch-10
Proxy Anchor checkpoint and all 25,882 normalized In-Shop training embeddings.

## Question

Appearance-based positive graphs repeatedly failed. A cooperative-game/influence
alternative is to ask whether two same-class examples propose compatible updates:
if their loss gradients agree, training on one is locally helpful to the other;
if they oppose, treating them as an undifferentiated coalition may be harmful.

For each image, the audit computes the gradient of the full-dataset Proxy Anchor
objective with respect to its normalized embedding. It uses the learned proxies,
`alpha=32`, `margin=0.1`, the exact positive and negative log-sum-exp denominators,
and projects the gradient onto the hypersphere tangent plane. Pairwise cosine is
then measured for all 153,115 within-identity pairs.

| statistic | value |
|---|---:|
| mean gradient cosine | 0.4162 |
| standard deviation | 0.4224 |
| pairs with opposing gradients | **17.94%** |
| pairs below cosine 0.5 | **46.08%** |
| correlation with embedding cosine | **0.2119** |
| 1st / 50th / 99th percentile | -0.6735 / 0.5319 / 0.9888 |

This is genuine heterogeneity not reducible to current embedding distance.

## Necessary estimator correction

A first pass used a per-image proxy-softmax gradient. It found zero negative
same-class pairs and mean cosine 0.891. That estimator was wrong for the proposed
mechanism because it discarded Proxy Anchor's proxy-anchored exponential
aggregation. The exact objective reverses the conclusion. This is another case
where a convenient surrogate answers a different question from the operating
loss.

## Operator consequence

The measurement passes provenance but does not by itself define a novel,
implementable supervision operator. Proxy Anchor has image-to-proxy relations,
not image-to-image positive edges. Using gradient compatibility must therefore
become sample/batch selection, example weighting, gradient projection, or an
auxiliary pair loss. Those routes are assessed in `docs/gcs_candidate.md`.

## Acquisition groups do not explain the conflicts

Crossing this audit with the filename acquisition groups rejects a tempting
unification of two findings:

| relation | pairs | mean gradient cosine | opposing gradients | cosine below 0.5 |
|---|---:|---:|---:|---:|
| same acquisition group | 41,312 | 0.4746 | **21.02%** | 38.66% |
| different acquisition group | 111,803 | 0.3947 | **16.80%** | 48.82% |

Same-group pairs are visually much closer, yet their odds of outright gradient
conflict are 1.32 times the cross-group odds. Cross-group pairs have more moderate
disagreement, not more sign reversals. The session shortcut therefore does not
cause the gradient-conflict population and cannot justify a session-targeted
gradient-surgery arm.
