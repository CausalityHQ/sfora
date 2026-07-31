# EGPU — exposure-gated proxy updates

**Gate 1 recorded 2026-07-31 before prior-art audit, diagnostic, implementation,
or GPU use.**

## Repository provenance

The corrected In-Shop Proxy Anchor recipe uses uniform image batching
(`batch_size=180`, `samples_per_class=0`) over 25,882 images and 3,997
identities. Most identities have four or five images. A four-image identity is
therefore exposed positively in only about
`1-(1-4/25882)^180 = 2.74%` of batches: its proxy receives roughly **35.5
negative-only batches per positive-exposure batch**. For five images the ratio
is about 28.3; at the 6.48-image mean it is about 21.7.

The exact epoch-10 audit shows the expected directional defect:

- 99.975% of proxies choose their own empirical class centroid;
- only 70.303% of empirical centroids choose their own proxy;
- only 65.308% of images score their own proxy highest;
- mean own proxy-centroid cosine is only 0.0989.

Proxy Anchor's negative term updates every proxy on every batch, even when no
positive sample for that proxy is present. Thus rare positive observations must
continually correct a proxy moved by thousands of unrelated negative samples.
Class-balanced batching would reduce the problem but sacrifices identity
diversity and is an established sampler, not the proposed operation.

## Mechanism

EGPU separates a proxy's two computational roles:

1. **landmark role:** every normalized proxy remains in the forward similarity
   matrix, and every embedding receives the exact original gradient from all
   nonmatching proxies;
2. **state-update role:** a proxy parameter row may move on a step only if its
   own class is represented in that batch.

Operationally, the scalar Proxy Anchor loss is unchanged. A gradient hook masks
the proxy-parameter gradient rows whose labels are absent from the batch; model
embedding gradients are not masked. Present proxies receive the complete
positive and negative gradient for that step. This costs one Boolean mask and
requires no memory bank, extra forward, new loss weight, or test-time change.

The cross-disciplinary analogy is event-triggered state estimation under
missing observations: a landmark can be used to update an observation without
allowing unobserved landmark state to drift. The novelty claim is the asymmetric
backward pass—full negative supervision for samples, exposure-gated motion for
proxies—not proxy freezing or negative subsampling.

## Gate-2 attack required

Search primary sources and code for:

- stop-gradient/detached proxies in proxy-based DML;
- updating class proxies only when their class appears in a batch;
- asymmetric sample/proxy gradient routing in Proxy Anchor, Proxy-NCA, face
  classifiers, sampled softmax, and memory-bank methods;
- partial-FC and large-class classification update rules;
- momentum, frozen, and data-derived proxies.

EGPU is dead if an existing method retains all proxy logits for sample gradients
while masking parameter gradients of absent-class proxies. It is not killed by
methods that omit absent proxies from the forward loss, use class-balanced
batches, or replace trainable proxies with EMA centroids; those change the
sample supervision or proxy estimator.

