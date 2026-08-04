# DGSL-RCF 2026 horizon audit

Date: 2026-08-04. Primary source: Fei et al., *Dynamic Graph Structure
Learning via Resistance Curvature Flow*, arXiv:2601.08149v1,
[paper](https://arxiv.org/abs/2601.08149).

## Reported result

The paper inserts a geometric-flow layer into BN-Inception and reports the
following highest values in a table labelled `Recall` (not explicitly
`Recall@1`): 82.97 on CUB, 89.28 on Cars196, and 79.94 on SOP. The corresponding
plain-triplet CUB value is 52.98, while a static kNN version of the same layer is
already 81.87. The paper says its split and base training protocol follow
*Deep Metric Learning with Spherical Embedding* and grid-searches separate
`k`, flow rate, and iteration values for every dataset/loss combination.

These numbers initially look capable of raising the comparable CUB horizon, but
the primary source does not establish the required single-image descriptor
protocol.

## Why it does not raise the present horizon

The geometric-flow layer is not a per-image map. For image `j`, its next-layer
activation is an affinity-weighted sum over every other image `l` in the current
minibatch:

```
X[j] = (sigma[j] + lambda * sum_{l != j} sigma[l] * w[j,l])
       / (1 + lambda * sum_{l != j} w[j,l]).
```

The weights are built from a minibatch kNN graph and evolved by resistance
curvature flow. Therefore the representation of one image depends on which
other images accompany it. The paper says the layer is inserted between a
convolution and activation, but does not specify removal at inference, a
gallery-independent descriptor export, query/gallery batching, or any other
single-image deployment rule. If the layer remains active during evaluation,
the result is transductive and batch-composition dependent; if it is removed,
the paper does not report that ablation or resulting score. Either way, the
published table cannot presently be treated as single-model, single-view cosine
descriptor evidence comparable to this project's target.

Additional evidential limits reinforce, but do not replace, that protocol
failure:

- the table says only `Recall`, never which `K`;
- no seed count, variance, confidence interval, or error bar is reported for the
  DML experiments;
- no embedding dimension or cosine/Euclidean retrieval definition is given in
  the accessible source;
- the cited project repository, <https://github.com/cqfei/RCF>, exists but is
  empty as of this audit, despite the paper saying the code is open-sourced;
- the static kNN layer alone produces the extraordinary CUB jump from 52.98 to
  81.87, yet there is no per-image or non-transductive control explaining it.

## Search consequence

DGSL-RCF does **not** replace the audited 76.6 ResNet/cosine or 87.8
higher-capacity CUB targets in the repository-blind Fable brief. It is,
however, direct prior art for minibatch kNN graph construction, resistance-
curvature evolution, and cross-instance graph aggregation inside DML networks.
Any future graph-curvature proposal must distinguish itself from both its static
and dynamic geometric-flow operators and must prove a gallery-independent
descriptor at evaluation.
