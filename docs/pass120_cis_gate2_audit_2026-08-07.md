# Pass 120 Gate-2 audit — Coalition Interference Supervision (CIS)

## Decision

**LIVE-NARROW; no GPU is authorized yet.** The exact proposed training object is
not mechanism-equivalent to the checked prior art, but the novelty claim is
narrow and requires explicit controls.

## Mechanism under review

CIS samples a bundle of images from distinct labelled classes, sums their
deployed embeddings, and applies a multi-label proxy objective whose positives
are exactly the bundle's class proxies. The bundle is a training-only object;
the deployed model still emits one 512-D descriptor for one image. The causal
claim is that a cross-class coalition exposes shared off-class interference
that independent one-image proxy terms do not constrain.

## Prior-art checks

* **Deep Sets** (Zaheer et al., NeurIPS 2017) proves the sum-then-transform form
  for permutation-invariant set functions and applies it to set tasks. It
  occupies the set encoder architecture, not a metric-learning objective in
  which a sum of *distinct-class image embeddings* is assigned the union of
  their proxy labels.
* **Set Transformer** (Lee et al., ICML 2019) models interactions in sets with
  attention. It occupies set-valued architectures, not CIS's summed coalition
  supervision or its interference hypothesis.
* **HyP² Loss** (Xu et al., 2022) and multi-label supervised contrastive
  learning (Zhang & Wu, AAAI 2024) use multi-label relations or overlapping
  labels in the loss. Their positive relation is between ordinary individual
  samples; neither forms a synthetic coalition by summing embeddings of
  *different labelled classes* and trains the sum against the union of class
  proxies.
* **Proxy Anchor** (Kim et al., CVPR 2020) already permits data-to-data
  interactions through its batch gradients, but does not create a multi-class
  sum target. This is an adjacent control, not an exact prior-art match.

Thus the mechanism-level distinction is: **CIS changes what is supervised (a
union-labelled cross-class coalition), rather than changing pair weighting,
proxy count, or a permutation-invariant set encoder.** The distinction could
still collapse empirically to a regularizer or to ordinary multi-label loss;
that is why the frozen controls must include class-dropout, a single-image
multi-label target, proxy orthogonality, and ordinary Proxy Anchor.

## Sources

Zaheer et al., *Deep Sets*, NeurIPS 2017; Lee et al., *Set Transformer*, ICML
2019; Xu et al., *HyP² Loss*, 2022; Zhang & Wu, *Multi-Label Supervised
Contrastive Learning*, AAAI 2024; Kim et al., *Proxy Anchor Loss*, CVPR 2020.

No benchmark result is claimed by this audit. Gate 3 preregistration and an
operator/gradient test are required before any queue submission.
